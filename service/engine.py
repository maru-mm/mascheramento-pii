# -*- coding: utf-8 -*-
"""
Motore di mascheratura: modello mmBERT + rete regex/checksum -> testo con
placeholder reversibili ([FULLNAME_1], [IBAN_1], ...) + dizionario.

Nessuna dipendenza da Flask, e torch/transformers si importano solo in
`load_pipeline()`: chunking, fusione e ripristino sono testabili senza modello
passando una `nlp` finta (qualunque callable con l'interfaccia della pipeline HF
"token-classification" con aggregation_strategy="simple").
"""

import bisect
import re

from detectors import SOFT_REGEX_LABELS, complete_time, detect_regex

MAX_WORDS = 120      # parole per chunk (~180 subword, ben sotto il MAX_LEN del training)
OVERLAP = 20         # parole di sovrapposizione tra chunk consecutivi

# I 22 tag del modello (docs/TASSONOMIA_TAG.md) + URL e IPADDR, che sono solo-regex.
TAGS = [
    ("FULLNAME", "Nome di persona (anche ruoli legali: giudice, avvocato, parti, teste)"),
    ("AGE", "Età"),
    ("GENDER", "Sesso / genere"),
    ("DATE", "Data di calendario"),
    ("TIME", "Ora"),
    ("STREET", "Via / piazza / corso"),
    ("BUILDINGNUM", "Numero civico"),
    ("ZIPCODE", "CAP"),
    ("CITY", "Città"),
    ("PROVINCE", "Sigla della provincia"),
    ("EMAIL", "Email, PEC inclusa"),
    ("TELEPHONENUM", "Numero di telefono"),
    ("CF", "Codice fiscale (checksum verificato)"),
    ("PIVA", "Partita IVA (checksum verificato)"),
    ("ID_DOC", "Numero di documento d'identità (carta, passaporto, patente)"),
    ("IBAN", "IBAN / numero di conto (checksum verificato)"),
    ("CREDITCARDNUMBER", "Numero di carta di credito (Luhn verificato)"),
    ("AMOUNT", "Importo in denaro"),
    ("TARGA", "Targa di veicolo"),
    ("ORG", "Ragione sociale privata: società, studio legale, banca"),
    ("DOCID", "Codice di un atto: ruolo generale, protocollo, repertorio, sentenza"),
    ("CATASTO", "Dati catastali: foglio, particella, subalterno"),
    ("URL", "Indirizzo web (solo rete regex)"),
    ("IPADDR", "Indirizzo IP, anche subnet CIDR o range (solo rete regex)"),
]
TAG_NAMES = [t for t, _ in TAGS]


def load_pipeline(model_dir, batch_size=8):
    """Pipeline HF token-classification sul modello locale (CPU, o GPU se c'e')."""
    import torch
    from transformers import pipeline

    return pipeline(
        "token-classification",
        model=model_dir,
        tokenizer=model_dir,
        aggregation_strategy="simple",
        device=0 if torch.cuda.is_available() else -1,
        batch_size=batch_size,
    )


# --------------------------------------------------------------------------- #
# Chunking word-safe + inferenza del modello
# --------------------------------------------------------------------------- #
def chunk_text(text, max_words=MAX_WORDS, overlap=OVERLAP):
    """Ritorna [(sottostringa, offset_char_globale), ...] senza tagliare parole."""
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    chunks, i = [], 0
    step = max(1, max_words - overlap)
    while i < len(words):
        block = words[i:i + max_words]
        start, end = block[0].start(), block[-1].end()
        chunks.append((text[start:end], start))      # slice esatto -> offset diretti
        if i + max_words >= len(words):
            break
        i += step
    return chunks


def detect_model(texts, nlp):
    """Entita' del modello per ogni testo, su offset globali.

    Tutti i chunk di tutti i testi passano in UNA chiamata alla pipeline, cosi'
    una richiesta batch sfrutta il batching del modello. Ritorna
    [(entita', n_chunk), ...] nello stesso ordine di `texts`."""
    per_text = [chunk_text(t) for t in texts]
    flat = [c for chunks in per_text for c, _ in chunks]
    results = nlp(flat) if flat else []
    if len(flat) == 1 and (not results or isinstance(results[0], dict)):
        results = [results]                           # singolo input -> normalizza
    out, k = [], 0
    for chunks in per_text:
        ents = []
        for _, off in chunks:
            for e in results[k]:
                ents.append({
                    "label": e["entity_group"],
                    "start": int(e["start"]) + off,
                    "end": int(e["end"]) + off,
                    "score": float(e["score"]),
                    "validated": False,
                    "source": "modello",
                })
            k += 1
        out.append((ents, len(chunks)))
    return out


# --------------------------------------------------------------------------- #
# Fusione modello + regex
# --------------------------------------------------------------------------- #
def _is_word(ch):
    """Carattere interno a una parola (lettere accentate e cifre incluse)."""
    return ch.isalnum() or ch == "_"


def _union_model_overlaps(cands):
    """Unisce le span del MODELLO con la stessa label che si sovrappongono.

    Dentro un chunk la pipeline (aggregation "simple") non produce sovrapposizioni:
    nascono solo nella zona di overlap fra due chunk. Li' un'entita' tagliata dal
    bordo del primo chunk ("Mario") convive con quella intera vista dal secondo
    ("Mario Rossi"); senza unione il greedy di _merge puo' tenere la parziale (score
    piu' alto) e scartare l'intera, lasciando "Rossi" in chiaro."""
    model = sorted((e for e in cands if e["source"] == "modello"),
                   key=lambda e: (e["label"], e["start"]))
    united = []
    for e in model:
        prev = united[-1] if united else None
        if prev and prev["label"] == e["label"] and e["start"] < prev["end"]:
            prev["end"] = max(prev["end"], e["end"])
            prev["score"] = max(prev["score"], e["score"])
        else:
            united.append(dict(e))
    return united + [e for e in cands if e["source"] != "modello"]


def merge(cands, text):
    """Greedy senza overlap. Priorita': checksum-valido > regex (non soft) > score
    > lunghezza.
    La rete regex copre campi a forma molto specifica: per quegli span e' piu' affidabile
    del modello (evita la frammentazione di CF/IBAN/carta in piu' pezzi). Fanno eccezione
    i tag in SOFT_REGEX_LABELS, dove la forma non ha un checksum a confermarla."""
    order = sorted(
        _union_model_overlaps(cands),
        key=lambda e: (1 if e["validated"] else 0,
                       1 if (e["source"] == "regex"
                             and e["label"] not in SOFT_REGEX_LABELS) else 0,
                       e["score"], e["end"] - e["start"]),
        reverse=True,
    )
    kept = []
    for e in order:
        # kept resta ordinata per start e senza sovrapposizioni: un candidato puo'
        # accavallarsi solo con i due vicini, che la ricerca binaria trova subito.
        i = bisect.bisect_right(kept, e["start"], key=lambda k: k["start"])
        if (i and kept[i - 1]["end"] > e["start"]) or \
           (i < len(kept) and kept[i]["start"] < e["end"]):
            continue
        kept.insert(i, e)
    # niente spazi inglobati nei placeholder (il modello a volte include lo spazio iniziale)
    for e in kept:
        while e["start"] < e["end"] and text[e["start"]].isspace():
            e["start"] += 1
        while e["end"] > e["start"] and text[e["end"] - 1].isspace():
            e["end"] -= 1
    kept = [e for e in kept if e["end"] > e["start"]]

    # Allineamento ai confini di parola. Il modello etichetta i sotto-token e a volte ne
    # copre solo una parte ("No" di "Novara"): sostituendo la span cosi' com'e' resterebbe
    # "[CITY_1]vara", cioe' un valore ancora ricostruibile. Se una span taglia una parola
    # a meta', la si estende fino a coprirla: per un anonimizzatore l'errore per eccesso
    # e' l'unico accettabile.
    for e in kept:
        while (e["start"] > 0
               and _is_word(text[e["start"] - 1]) and _is_word(text[e["start"]])):
            e["start"] -= 1
        while (e["end"] < len(text)
               and _is_word(text[e["end"]]) and _is_word(text[e["end"] - 1])):
            e["end"] += 1

    # L'estensione puo' rendere due span sovrapposte o adiacenti: si fondono, altrimenti
    # una stessa parola verrebbe sostituita due volte ("[CATASTO_1][CATASTO_2]").
    kept.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    merged = []
    for e in kept:
        if merged and e["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
            continue
        if merged and e["start"] == merged[-1]["end"] and e["label"] == merged[-1]["label"]:
            merged[-1]["end"] = e["end"]
            continue
        merged.append(e)
    return merged


# --------------------------------------------------------------------------- #
# Anonimizzazione
# --------------------------------------------------------------------------- #
def _norm(s):
    return re.sub(r"\s+", " ", s.strip()).casefold()


def _build(text, model_ents, n_chunks, excluded, include_mapping):
    # il modello taglia i minuti delle ore: la regex dell'ora completa SOLO le sue span TIME
    complete_time(model_ents, text)
    cands = model_ents + detect_regex(text)
    if excluded:
        cands = [e for e in cands if e["label"] not in excluded]
    kept = merge(cands, text)

    # ID reversibili: stesso (label, valore-normalizzato) -> stesso placeholder.
    counters, seen, mapping = {}, {}, {}
    entities, anon, by_label, pos = [], [], {}, 0
    for e in kept:
        val = text[e["start"]:e["end"]]
        key = (e["label"], _norm(val))
        ph = seen.get(key)
        if ph is None:
            counters[e["label"]] = counters.get(e["label"], 0) + 1
            ph = seen[key] = f"[{e['label']}_{counters[e['label']]}]"
            if include_mapping:
                mapping[ph] = val
        entities.append({
            "label": e["label"],
            "placeholder": ph,
            "start": e["start"],
            "end": e["end"],
            "source": e["source"],
            "validated": e["validated"],
            "score": round(e["score"], 4),
        })
        anon.append(text[pos:e["start"]])
        anon.append(ph)
        by_label[e["label"]] = by_label.get(e["label"], 0) + 1
        pos = e["end"]
    anon.append(text[pos:])

    out = {
        "anonymized_text": "".join(anon),
        "entities": entities,
        "stats": {
            "n_chars": len(text),
            "n_chunks": n_chunks,
            "n_entities": len(kept),
            "n_unique": len(seen),
            "by_label": dict(sorted(by_label.items(), key=lambda x: -x[1])),
        },
    }
    # Senza dizionario l'anonimizzazione e' DEFINITIVA: la chiave manca del tutto.
    # La numerazione resta (due [FULLNAME_1] = stesso soggetto), utile a un LLM.
    if include_mapping:
        out["mapping"] = mapping
    return out


def anonymize_many(texts, nlp, excluded=(), include_mapping=True):
    """Anonimizza una lista di testi con una sola passata del modello."""
    excluded = set(excluded or ())
    return [_build(t, ents, n, excluded, include_mapping)
            for t, (ents, n) in zip(texts, detect_model(texts, nlp))]


def anonymize(text, nlp, excluded=(), include_mapping=True):
    return anonymize_many([text], nlp, excluded, include_mapping)[0]


# --------------------------------------------------------------------------- #
# Ripristino
# --------------------------------------------------------------------------- #
_BRACKETED = re.compile(r"\[\s*(.*?)\s*\]", re.S)


def restore(text, mapping):
    """Rimette i valori veri al posto dei placeholder nella risposta di un LLM.

    Una sola passata sul testo ORIGINALE (non N replace in cascata: un valore che
    contiene per caso il nome di un'altra chiave verrebbe ri-sostituito). Tollera
    quello che gli LLM fanno ai placeholder:
      - forma nuda senza parentesi ("FULLNAME_1"), ma solo a confine di parola
        Unicode, cosi' "CF_1e'" o "ilCF_1" restano intatti;
      - parentesi TUTTO-O-NIENTE: un indice inventato ([CF_12] con CF_1 in mappa)
        resta visibile invece di diventare un valore SBAGLIATO;
      - grassetto markdown simmetrico ("**FULLNAME_1**"), assorbito a coppie."""
    by_inner = {(ph[1:-1] if ph.startswith("[") and ph.endswith("]") else ph): val
                for ph, val in mapping.items()}
    if not by_inner:
        return text
    alts = "|".join(rf"\[\s*{re.escape(k)}\s*\]|(?<!\w){re.escape(k)}(?!\w)"
                    for k in by_inner)
    rx = re.compile(rf"(\*{{0,2}})(?:{alts})\1")

    def sub(m):
        stars = len(m.group(1))
        core = m.group(0)[stars:len(m.group(0)) - stars]
        br = _BRACKETED.fullmatch(core)
        val = by_inner.get(br.group(1) if br else core)
        return m.group(0) if val is None else val

    return rx.sub(sub, text)

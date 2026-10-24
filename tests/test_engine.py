# -*- coding: utf-8 -*-
"""`engine`: anonimizzazione end-to-end con un modello FINTO, e ripristino.

La pipeline HF e' sostituita da un callable con la stessa interfaccia
(lista di chunk -> lista di liste di entita'), quindi i test girano sulla sola
libreria standard. Tutti i valori sono SINTETICI.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

import engine  # noqa: E402


def fake_nlp(*rules):
    """Modello finto: ogni (pezzo, label, score) e' etichettato ovunque compaia."""
    def nlp(chunks):
        out = []
        for c in chunks:
            res = []
            for piece, label, score in rules:
                i = c.find(piece)
                while i >= 0:
                    res.append({"entity_group": label, "start": i, "end": i + len(piece),
                                "score": score, "word": piece})
                    i = c.find(piece, i + len(piece))
            out.append(res)
        return out
    return nlp


NO_MODEL = fake_nlp()


class Anonymize(unittest.TestCase):
    def test_placeholder_e_dizionario(self):
        t = "Mario Rossi paga Mario Rossi. C.F. RSSMRA85H12F205Y."
        out = engine.anonymize(t, fake_nlp(("Mario Rossi", "FULLNAME", 0.99)))
        self.assertEqual(out["anonymized_text"], "[FULLNAME_1] paga [FULLNAME_1]. C.F. [CF_1].")
        self.assertEqual(out["mapping"], {"[FULLNAME_1]": "Mario Rossi",
                                          "[CF_1]": "RSSMRA85H12F205Y"})
        self.assertEqual(out["stats"]["n_entities"], 3)
        self.assertEqual(out["stats"]["n_unique"], 2)
        cf = [e for e in out["entities"] if e["label"] == "CF"][0]
        self.assertTrue(cf["validated"])
        self.assertEqual(t[cf["start"]:cf["end"]], "RSSMRA85H12F205Y")

    def test_senza_dizionario_niente_chiave_mapping(self):
        out = engine.anonymize("IBAN IT60X0542811101000000123456", NO_MODEL,
                               include_mapping=False)
        self.assertNotIn("mapping", out)
        self.assertEqual(out["anonymized_text"], "IBAN [IBAN_1]")

    def test_tag_esclusi_restano_in_chiaro(self):
        t = "Mario Rossi paga € 1.200,00"
        out = engine.anonymize(t, fake_nlp(("Mario Rossi", "FULLNAME", 0.99)),
                               excluded={"AMOUNT"})
        self.assertEqual(out["anonymized_text"], "[FULLNAME_1] paga € 1.200,00")

    def test_testo_vuoto(self):
        out = engine.anonymize("", NO_MODEL)
        self.assertEqual(out["anonymized_text"], "")
        self.assertEqual(out["entities"], [])

    def test_batch_ordine_e_numerazione_per_testo(self):
        nlp = fake_nlp(("Mario Rossi", "FULLNAME", 0.99), ("Anna Bianchi", "FULLNAME", 0.99))
        res = engine.anonymize_many(["Anna Bianchi e Mario Rossi", "Mario Rossi", "niente"], nlp)
        self.assertEqual([r["anonymized_text"] for r in res],
                         ["[FULLNAME_1] e [FULLNAME_2]", "[FULLNAME_1]", "niente"])
        self.assertEqual(res[1]["mapping"], {"[FULLNAME_1]": "Mario Rossi"})

    def test_ora_completata_dalla_regex(self):
        """Il modello taglia i minuti, l'output no."""
        t = "Ingresso ore 18:28, uscita ore 19:30."
        out = engine.anonymize(t, fake_nlp(("18", "TIME", 0.9), ("19:", "TIME", 0.9)))
        self.assertEqual(out["anonymized_text"], "Ingresso ore [TIME_1], uscita ore [TIME_2].")
        self.assertEqual(out["mapping"], {"[TIME_1]": "18:28", "[TIME_2]": "19:30"})

    def test_entita_a_cavallo_fra_due_chunk(self):
        """Il primo chunk finisce su "Mario" e lo etichetta con score ALTO; il secondo
        vede "Mario Rossi" intero con score piu' basso. Prima dell'unione delle span
        sovrapposte vinceva la parziale e "Rossi" restava in chiaro."""
        words = ["parola"] * (engine.MAX_WORDS - 1) + ["Mario", "Rossi"] + ["fine"] * 5
        t = " ".join(words)

        def nlp(chunks):
            out = []
            for c in chunks:
                if "Mario Rossi" in c:
                    i = c.index("Mario Rossi")
                    out.append([{"entity_group": "FULLNAME", "start": i, "end": i + 11,
                                 "score": 0.90, "word": "Mario Rossi"}])
                elif c.endswith("Mario"):
                    i = len(c) - 5
                    out.append([{"entity_group": "FULLNAME", "start": i, "end": i + 5,
                                 "score": 0.99, "word": "Mario"}])
                else:
                    out.append([])
            return out

        self.assertGreater(len(engine.chunk_text(t)), 1)
        out = engine.anonymize(t, nlp)
        self.assertNotIn("Rossi", out["anonymized_text"])
        self.assertEqual(out["mapping"], {"[FULLNAME_1]": "Mario Rossi"})


class Restore(unittest.TestCase):
    def r(self, text, mapping):
        return engine.restore(text, mapping)

    def test_parentesi_e_forma_nuda(self):
        m = {"[FULLNAME_1]": "Mario Rossi"}
        self.assertEqual(self.r("Il [FULLNAME_1] ha firmato.", m), "Il Mario Rossi ha firmato.")
        self.assertEqual(self.r("Il [ FULLNAME_1 ] ha firmato.", m), "Il Mario Rossi ha firmato.")
        self.assertEqual(self.r("Il FULLNAME_1 ha firmato.", m), "Il Mario Rossi ha firmato.")
        self.assertEqual(self.r("col1\tFULLNAME_1\tcol3", m), "col1\tMario Rossi\tcol3")
        self.assertEqual(self.r("riga\nFULLNAME_1\nriga", m), "riga\nMario Rossi\nriga")

    def test_indice_inventato_resta_visibile(self):
        m = {"[CF_1]": "RSSMRA78S03L750K"}
        self.assertEqual(self.r("vedi [CF_12] qui", m), "vedi [CF_12] qui")
        self.assertEqual(self.r("vedi CF_12 qui", m), "vedi CF_12 qui")

    def test_confine_di_parola_unicode(self):
        m = {"[CF_1]": "RSSMRA78S03L750K"}
        self.assertEqual(self.r("ilCF_1 resta intatto", m), "ilCF_1 resta intatto")
        self.assertEqual(self.r("il codice CF_1è già", m), "il codice CF_1è già")
        self.assertEqual(self.r("il codice CF_1 è già", m), "il codice RSSMRA78S03L750K è già")

    def test_grassetto_markdown(self):
        m = {"[CF_1]": "AAAA111", "[CF_2]": "BBBB222"}
        self.assertEqual(self.r("**CF_1****CF_2**", m), "AAAA111BBBB222")
        self.assertEqual(self.r("Vedi CF_1**parola importante** qui", m),
                         "Vedi AAAA111**parola importante** qui")

    def test_nessuna_sostituzione_in_cascata(self):
        m = {"[ORG_1]": "Studio Legale CF_2 & Partners", "[CF_2]": "RSSMRA80A01H501U"}
        self.assertEqual(self.r("Rivolgersi a ORG_1 per informazioni.", m),
                         "Rivolgersi a Studio Legale CF_2 & Partners per informazioni.")

    def test_andata_e_ritorno(self):
        t = "Mario Rossi, IBAN IT60X0542811101000000123456, tel. 333 1234567."
        out = engine.anonymize(t, fake_nlp(("Mario Rossi", "FULLNAME", 0.99)))
        self.assertEqual(self.r(out["anonymized_text"], out["mapping"]), t)


if __name__ == "__main__":
    unittest.main()

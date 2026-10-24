# CLAUDE.md

API REST stateless di mascheratura PII per Cloud Run. Panoramica, API e deploy in
[README.md](README.md); tag in [docs/TASSONOMIA_TAG.md](docs/TASSONOMIA_TAG.md).

## Codice

- `service/detectors.py` — rete regex + checksum (CF/PIVA/IBAN/Luhn…). Solo libreria
  standard: non importare torch/transformers/flask qui (lo verifica `test_detector_formats`).
- `service/engine.py` — chunking (120 parole, overlap 20), inferenza, `merge` (priorita':
  checksum valido > regex non-soft > score > lunghezza; unione delle span del modello
  sovrapposte fra chunk; estensione ai confini di parola), placeholder, `restore`.
  torch/transformers si importano solo in `load_pipeline`.
- `service/main.py` — Flask, factory `create_app(nlp=None)`; gunicorn la invoca con
  `'main:create_app()'`. Configurazione solo da env, nessuno stato fra richieste.

## Regole

- Mai scrivere nei log il testo delle richieste o i valori del dizionario.
- Nel dubbio si maschera un carattere in piu': per un anonimizzatore l'errore per
  eccesso e' l'unico accettabile.
- Niente dipendenze AGPL (es. PyMuPDF): il servizio e' pensato per essere integrato.
- Test: `python -m unittest discover -v tests` (modello finto, nessun download).

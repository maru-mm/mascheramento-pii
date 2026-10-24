# rizzo-pii — API di mascheratura PII per Google Cloud Run

Servizio REST **stateless** che maschera i dati personali in testi italiani (atti,
contratti, sentenze, email) sostituendoli con placeholder reversibili
(`[FULLNAME_1]`, `[CF_1]`, `[IBAN_1]`, …) e restituisce il dizionario per il
ripristino.

Derivato da [Rizzo-AI-Academy/rizzo-pii](https://github.com/Rizzo-AI-Academy/rizzo-pii)
(MIT): stesso modello e stessa rete regex+checksum, senza UI, app desktop,
training ed export PDF.

- **Modello**: [`rizzoaiacademy/rizzo-pii-0.3B`](https://huggingface.co/rizzoaiacademy/rizzo-pii-0.3B)
  v1.5.0 (mmBERT, ≈0.3B parametri, CPU), incluso nell'immagine in fase di build.
- **Rete regex + checksum** affiancata al modello: CF (anche omocodico), P.IVA,
  IBAN, carte (Luhn), email, telefono, importi, targhe, URL, IP, codici d'atto, date.
- **24 tag** — elenco in [docs/TASSONOMIA_TAG.md](docs/TASSONOMIA_TAG.md) o da `GET /v1/tags`.
- **Nessuno stato e nessun testo nei log**: il dizionario lo conserva il chiamante.

## Struttura

```
service/
  main.py        API Flask (factory create_app, avviata da gunicorn)
  engine.py      chunking, inferenza, fusione modello+regex, placeholder, ripristino
  detectors.py   rete regex + checksum (nessuna dipendenza)
tests/           unittest (solo libreria standard; test_api.py usa flask)
Dockerfile       immagine CPU con modello incluso
cloudbuild.yaml  build + push + deploy su Cloud Run
```

## API

### `POST /v1/anonymize`

```json
{
  "text": "Il sig. Mario Rossi, C.F. RSSMRA85H12F205Y, IBAN IT60X0542811101000000123456",
  "exclude_tags": ["AMOUNT"],
  "include_mapping": true
}
```

| Campo | Default | Note |
|---|---|---|
| `text` **o** `texts` | — | stringa singola, oppure lista (max `PII_MAX_BATCH`) elaborata in un'unica passata del modello |
| `exclude_tags` | `PII_EXCLUDE_TAGS` | tag rilevati ma lasciati in chiaro |
| `include_mapping` | `true` | `false` = mascheratura **definitiva**: nessun `mapping` nella risposta |

Risposta (per `texts`: `{"results": [ ... ]}` nello stesso ordine):

```json
{
  "anonymized_text": "Il sig. [FULLNAME_1], C.F. [CF_1], IBAN [IBAN_1]",
  "mapping": {"[FULLNAME_1]": "Mario Rossi", "[CF_1]": "RSSMRA85H12F205Y", "[IBAN_1]": "IT60X0542811101000000123456"},
  "entities": [{"label": "FULLNAME", "placeholder": "[FULLNAME_1]", "start": 8, "end": 19,
                "source": "modello", "validated": false, "score": 0.998}, ...],
  "stats": {"n_chars": 76, "n_chunks": 1, "n_entities": 3, "n_unique": 3, "by_label": {...}}
}
```

`start`/`end` sono offset di carattere nel testo **inviato**. Lo stesso valore
(case e spazi normalizzati) riceve sempre lo stesso placeholder. La numerazione
riparte da 1 per ogni testo.

### `POST /v1/restore`

```json
{"text": "Risposta dell'LLM con [FULLNAME_1] e **CF_1**", "mapping": {"[FULLNAME_1]": "Mario Rossi", "[CF_1]": "RSSMRA85H12F205Y"}}
```
→ `{"text": "Risposta dell'LLM con Mario Rossi e RSSMRA85H12F205Y"}`

Tollera i placeholder senza parentesi e in grassetto markdown. Un indice inventato
dall'LLM (`[CF_12]` con solo `CF_1` nel dizionario) resta visibile invece di
diventare un valore sbagliato.

### Altri endpoint

- `GET /v1/tags`: legenda dei tag e tag esclusi di default.
- `GET /health`: `200` quando il modello è caricato.

Errori: JSON `{"error": "..."}` con `400` (input non valido), `413` (troppo
grande), `500` (interno, senza dettagli del testo).

## Configurazione

| Variabile | Default | |
|---|---|---|
| `PORT` | `8080` | la imposta Cloud Run |
| `PII_MODEL_DIR` | `/app/model` | cartella del modello |
| `PII_EXCLUDE_TAGS` | — | es. `AMOUNT,GENDER` |
| `PII_MAX_CHARS` | `200000` | caratteri per testo |
| `PII_MAX_BATCH` | `64` | testi per richiesta |
| `PII_INFER_BATCH` | `8` | chunk per passata del modello |
| `GUNICORN_THREADS` | `4` | richieste parallele per istanza (= `--concurrency`) |
| `OMP_NUM_THREADS` | — | thread torch; impostalo ≈ al numero di vCPU |

Limite di Cloud Run: body della richiesta ≤ 32 MiB.

## Deploy su Cloud Run

Una tantum:

```bash
gcloud config set project TUO_PROGETTO
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
gcloud artifacts repositories create rizzo --repository-format=docker --location=europe-west8
# il service account usato da Cloud Build deve avere roles/run.admin e roles/iam.serviceAccountUser
```

Build e deploy (servizio **privato**, regione Milano):

```bash
gcloud builds submit --config cloudbuild.yaml --region europe-west8
```

Parametri in `cloudbuild.yaml`: 2 vCPU, 4 GiB, concurrency 4, 0–5 istanze,
`--cpu-boost` per accorciare l'avvio a freddo (caricamento del modello: decine di
secondi). Per eliminare l'avvio a freddo: `--min-instances=1` (istanza sempre
accesa e quindi sempre pagata).

## Chiamarlo dai tuoi servizi Google Cloud

Il servizio è protetto da IAM. Concedi al service account del chiamante
(Cloud Run, Cloud Functions, GKE, Workflows…) il ruolo `roles/run.invoker`:

```bash
gcloud run services add-iam-policy-binding rizzo-pii --region europe-west8 \
  --member serviceAccount:CHIAMANTE@TUO_PROGETTO.iam.gserviceaccount.com --role roles/run.invoker
```

Python (`pip install google-auth requests`):

```python
import google.auth.transport.requests
import google.oauth2.id_token
import requests

URL = "https://rizzo-pii-xxxxx.europe-west8.run.app"
token = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), URL)
h = {"Authorization": f"Bearer {token}"}

anon = requests.post(f"{URL}/v1/anonymize", json={"text": testo}, headers=h, timeout=300).json()
risposta_llm = chiama_llm(anon["anonymized_text"])          # solo placeholder escono
finale = requests.post(f"{URL}/v1/restore",
                       json={"text": risposta_llm, "mapping": anon["mapping"]},
                       headers=h, timeout=60).json()["text"]
```

Da terminale:

```bash
curl -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
     -H "Content-Type: application/json" \
     -d '{"text":"Mario Rossi, CF RSSMRA85H12F205Y"}' \
     https://rizzo-pii-xxxxx.europe-west8.run.app/v1/anonymize
```

Il `mapping` contiene i dati in chiaro: conservalo solo dove serve per il
ripristino (memoria della richiesta, oppure Secret Manager/Firestore cifrato se
il ripristino avviene più tardi).

## Sviluppo locale

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0"
pip install -r requirements.txt
hf download rizzoaiacademy/rizzo-pii-0.3B --revision v1.5.0 --local-dir models/rizzo-pii-0.3B-v1.5.0
cd service && gunicorn --bind 127.0.0.1:8080 --threads 4 'main:create_app()'
```

Test (senza modello): `python -m unittest discover -v tests`

Docker: `docker build --platform linux/amd64 -t rizzo-pii . && docker run --rm -p 8080:8080 rizzo-pii`

## Limiti noti

- Solo testo: per PDF/DOCX estrai il testo a monte (es. Document AI) e passalo a `/v1/anonymize`.
- Il modello è addestrato sull'italiano legale; le altre lingue non sono validate.
- Escludere un tag non disattiva le regole regex vicine: con `EMAIL` escluso, il
  dominio dell'email viene comunque mascherato come `URL`.
- Nessuna rilevazione è perfetta: per dati ad alto rischio verifica a campione l'output.

## Licenza

MIT (vedi [LICENSE](LICENSE)). Modello: vedi la scheda su Hugging Face.

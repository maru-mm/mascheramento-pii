# ============================================================================
# Rizzo PII — API REST di mascheratura per Cloud Run (CPU, stateless).
#
#   docker build -t rizzo-pii .
#   docker run --rm -p 8080:8080 rizzo-pii        # -> http://127.0.0.1:8080/health
#
# L'immagine e' AUTOSUFFICIENTE: il modello si scarica da Hugging Face in fase di
# build e a runtime non si esce verso internet (HF_HUB_OFFLINE=1).
# Su Mac Apple Silicon: docker build --platform linux/amd64 (Cloud Run e' amd64),
# oppure si builda direttamente con Cloud Build (cloudbuild.yaml).
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: runtime OpenMP richiesto dai kernel CPU di torch.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# torch dall'indice CPU: la wheel default tira dietro ~2.5 GB di CUDA inutili qui.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0" \
 && pip install -r /tmp/requirements.txt

# --- modello -----------------------------------------------------------------
ARG MODEL_REPO=rizzoaiacademy/rizzo-pii-0.3B
ARG MODEL_REVISION=v1.5.0
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('$MODEL_REPO', revision='$MODEL_REVISION', local_dir='/app/model', \
allow_patterns=['*.json','*.safetensors','*.txt','*.model'])" \
 && rm -rf /app/model/.cache

# --- applicazione ------------------------------------------------------------
WORKDIR /app
COPY service/ /app/service/

RUN useradd --uid 10001 --create-home app
USER app
ENV HOME=/home/app

ENV PORT=8080 \
    PII_MODEL_DIR=/app/model \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    GUNICORN_THREADS=4

EXPOSE 8080

# 1 worker: il modello sta in memoria una volta sola (~1.2 GB); il parallelismo lo
# danno i thread (allineare --concurrency di Cloud Run a GUNICORN_THREADS).
# --timeout 0: il timeout delle richieste lo gestisce Cloud Run.
CMD exec gunicorn --chdir /app/service --bind 0.0.0.0:${PORT} \
    --workers 1 --threads ${GUNICORN_THREADS} --timeout 0 'main:create_app()'

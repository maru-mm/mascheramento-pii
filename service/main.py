# -*- coding: utf-8 -*-
"""
API REST di mascheratura PII, pensata per Cloud Run (stateless, solo JSON).

    POST /v1/anonymize   {"text": "..."} oppure {"texts": ["...", ...]}
    POST /v1/restore     {"text": "...", "mapping": {"[FULLNAME_1]": "Mario Rossi"}}
    GET  /v1/tags        legenda dei tag
    GET  /health         200 quando il modello e' caricato

Nessuno stato fra le richieste: il dizionario placeholder -> valore lo restituisce
/v1/anonymize e lo conserva il chiamante, che lo ripassa a /v1/restore. Il testo
delle richieste non viene mai scritto nei log.

Configurazione (variabili d'ambiente):
    PII_MODEL_DIR       cartella del modello (default /app/model, poi models/ della repo)
    PII_EXCLUDE_TAGS    tag da non mascherare di default, separati da virgola
    PII_MAX_CHARS       caratteri massimi per testo            (default 200000)
    PII_MAX_BATCH       testi massimi per richiesta batch      (default 64)
    PII_INFER_BATCH     chunk per passata del modello          (default 8)
    PORT                porta di ascolto (la imposta Cloud Run; usata da gunicorn)

Avvio: gunicorn 'main:create_app()'  (vedi Dockerfile)
"""

import logging
import os
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

import engine

APP_VERSION = "2.0.0-cloudrun"
MODEL_VERSION = "1.5.0"

log = logging.getLogger("rizzo-pii")


def _default_model_dir():
    if os.environ.get("PII_MODEL_DIR"):
        return os.environ["PII_MODEL_DIR"]
    if os.path.isdir("/app/model"):
        return "/app/model"
    return str(Path(__file__).resolve().parents[1] / "models" / f"rizzo-pii-0.3B-v{MODEL_VERSION}")


def _int_env(name, default):
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


def _tag_list(raw):
    """Lista di tag da JSON (lista) o da env (stringa separata da virgole)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.split(",")
    if not isinstance(raw, list) or not all(isinstance(t, str) for t in raw):
        raise ValueError("exclude_tags deve essere una lista di stringhe.")
    tags = [t.strip().upper() for t in raw if t.strip()]
    unknown = [t for t in tags if t not in engine.TAG_NAMES]
    if unknown:
        raise ValueError(f"Tag sconosciuti: {', '.join(unknown)}")
    return tags


class ApiError(Exception):
    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.msg, self.status = msg, status


def create_app(nlp=None):
    """Factory dell'app. `nlp` iniettabile nei test; altrimenti carica il modello."""
    model_dir = _default_model_dir()
    max_chars = _int_env("PII_MAX_CHARS", 200_000)
    max_batch = _int_env("PII_MAX_BATCH", 64)
    default_excluded = _tag_list(os.environ.get("PII_EXCLUDE_TAGS"))

    if nlp is None:
        if not os.path.isdir(model_dir):
            raise SystemExit(f"Modello non trovato in {model_dir} (imposta PII_MODEL_DIR).")
        log.warning("Carico il modello da %s ...", model_dir)
        nlp = engine.load_pipeline(model_dir, batch_size=_int_env("PII_INFER_BATCH", 8))
        log.warning("Modello pronto.")

    app = Flask(__name__)
    # tetto grezzo sul body: il limite vero e' per testo (max_chars), questo ferma
    # prima del parsing i payload assurdi
    app.config["MAX_CONTENT_LENGTH"] = max_chars * max_batch * 4 + 1_000_000

    def _payload():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ApiError("Body JSON non valido (serve Content-Type: application/json).")
        return data

    def _check_text(t, field):
        if not isinstance(t, str):
            raise ApiError(f"'{field}' deve essere una stringa.")
        if len(t) > max_chars:
            raise ApiError(f"'{field}' supera {max_chars} caratteri.", 413)
        return t

    @app.errorhandler(ApiError)
    def _bad_request(e):
        return jsonify({"error": e.msg}), e.status

    @app.errorhandler(HTTPException)
    def _http_error(e):
        return jsonify({"error": e.description}), e.code

    @app.errorhandler(Exception)
    def _internal(e):
        # solo il tipo dell'errore: il messaggio potrebbe contenere pezzi del testo
        log.error("Errore interno: %s", type(e).__name__)
        return jsonify({"error": "Errore interno."}), 500

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "app_version": APP_VERSION,
                        "model_version": MODEL_VERSION})

    @app.get("/v1/tags")
    def tags():
        return jsonify({"tags": [{"tag": t, "description": d} for t, d in engine.TAGS],
                        "default_excluded": default_excluded})

    @app.post("/v1/anonymize")
    def anonymize():
        data = _payload()
        try:
            excluded = (_tag_list(data["exclude_tags"]) if "exclude_tags" in data
                        else default_excluded)
        except ValueError as e:
            raise ApiError(str(e))
        include_mapping = data.get("include_mapping", True)
        if not isinstance(include_mapping, bool):
            raise ApiError("'include_mapping' deve essere true/false.")

        if "texts" in data:
            texts = data["texts"]
            if not isinstance(texts, list) or not texts:
                raise ApiError("'texts' deve essere una lista non vuota di stringhe.")
            if len(texts) > max_batch:
                raise ApiError(f"'texts' supera {max_batch} elementi.", 413)
            texts = [_check_text(t, "texts[]") for t in texts]
            results = engine.anonymize_many(texts, nlp, excluded, include_mapping)
            return jsonify({"results": results})

        if "text" not in data:
            raise ApiError("Serve 'text' oppure 'texts'.")
        text = _check_text(data["text"], "text")
        return jsonify(engine.anonymize(text, nlp, excluded, include_mapping))

    @app.post("/v1/restore")
    def restore():
        data = _payload()
        text = _check_text(data.get("text"), "text")
        mapping = data.get("mapping")
        if (not isinstance(mapping, dict)
                or not all(isinstance(k, str) and k.strip() and isinstance(v, str)
                           for k, v in mapping.items())):
            raise ApiError("'mapping' deve essere un oggetto {placeholder: valore} di stringhe.")
        return jsonify({"text": engine.restore(text, mapping)})

    return app

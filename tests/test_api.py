# -*- coding: utf-8 -*-
"""API REST (`service/main.py`) col test client di Flask e un modello FINTO.

Richiede flask; senza, la suite si salta. Tutti i valori sono SINTETICI.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))
sys.path.insert(0, str(ROOT / "tests"))

from test_engine import fake_nlp  # noqa: E402

try:
    import main
except ImportError:                     # flask assente
    main = None


@unittest.skipIf(main is None, "flask non installato")
class Api(unittest.TestCase):
    def setUp(self):
        with patch.dict(os.environ, {"PII_MAX_CHARS": "1000", "PII_MAX_BATCH": "3"}):
            app = main.create_app(nlp=fake_nlp(("Mario Rossi", "FULLNAME", 0.99)))
        self.c = app.test_client()

    def test_health(self):
        r = self.c.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_tags(self):
        tags = [t["tag"] for t in self.c.get("/v1/tags").get_json()["tags"]]
        self.assertIn("CF", tags)
        self.assertEqual(len(tags), 24)

    def test_anonymize_e_restore(self):
        t = "Mario Rossi, C.F. RSSMRA85H12F205Y"
        r = self.c.post("/v1/anonymize", json={"text": t})
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(d["anonymized_text"], "[FULLNAME_1], C.F. [CF_1]")
        r = self.c.post("/v1/restore", json={"text": "Il [FULLNAME_1] (CF_1).",
                                             "mapping": d["mapping"]})
        self.assertEqual(r.get_json()["text"], "Il Mario Rossi (RSSMRA85H12F205Y).")

    def test_batch(self):
        r = self.c.post("/v1/anonymize", json={"texts": ["Mario Rossi", "niente"],
                                               "include_mapping": False})
        res = r.get_json()["results"]
        self.assertEqual([x["anonymized_text"] for x in res], ["[FULLNAME_1]", "niente"])
        self.assertNotIn("mapping", res[0])

    def test_exclude_tags(self):
        r = self.c.post("/v1/anonymize", json={"text": "Mario Rossi", "exclude_tags": ["FULLNAME"]})
        self.assertEqual(r.get_json()["anonymized_text"], "Mario Rossi")

    def test_errori(self):
        casi = [
            ({"data": "non json"}, 400),
            ({"json": {}}, 400),
            ({"json": {"text": 5}}, 400),
            ({"json": {"text": "x" * 1001}}, 413),
            ({"json": {"texts": ["a"] * 4}}, 413),
            ({"json": {"texts": []}}, 400),
            ({"json": {"text": "a", "exclude_tags": ["BOH"]}}, 400),
            ({"json": {"text": "a", "include_mapping": "no"}}, 400),
        ]
        for kwargs, status in casi:
            with self.subTest(kwargs=kwargs):
                r = self.c.post("/v1/anonymize", **kwargs)
                self.assertEqual(r.status_code, status)
                self.assertIn("error", r.get_json())
        r = self.c.post("/v1/restore", json={"text": "a", "mapping": {"[X_1]": 3}})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.c.get("/inesistente").status_code, 404)


if __name__ == "__main__":
    unittest.main()

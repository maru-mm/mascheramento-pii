# -*- coding: utf-8 -*-
"""Codice fiscale omocodico: riconoscimento nella rete regex.

L'omocodia e' la procedura con cui l'Agenzia delle Entrate differenzia due codici
fiscali identici: sostituisce le cifre con una lettera partendo da destra
(0=L 1=M 2=N 3=P 4=Q 5=R 6=S 7=T 8=U 9=V) e ricalcola il carattere di controllo.
Il codice che ne esce e' un codice fiscale valido a tutti gli effetti.

I detector si importano da `service/detectors.py`, che non tira dentro
torch/transformers: provare una regex non deve costare 0.3B di pesi scaricati.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "service"))

import detectors  # noqa: E402


# --------------------------------------------------------------------------- #
# Verificatore di checksum indipendente.
# Riscritto qui dalla specifica e non importato dal generatore: se lo prendessimo
# in prestito da lui, il test direbbe solo che il generatore concorda con se stesso.
# --------------------------------------------------------------------------- #
_ODD = {"0": 1, "1": 0, "2": 5, "3": 7, "4": 9, "5": 13, "6": 15, "7": 17, "8": 19,
        "9": 21, "A": 1, "B": 0, "C": 5, "D": 7, "E": 9, "F": 13, "G": 15, "H": 17,
        "I": 19, "J": 21, "K": 2, "L": 4, "M": 18, "N": 20, "O": 11, "P": 3, "Q": 6,
        "R": 8, "S": 12, "T": 14, "U": 16, "V": 10, "W": 22, "X": 25, "Y": 24, "Z": 23}


def cf_checksum_ok(code):
    code = code.strip().upper()
    if len(code) != 16 or not code.isalnum():
        return False
    total = 0
    for i, ch in enumerate(code[:15]):
        if i % 2 == 0:
            if ch not in _ODD:
                return False
            total += _ODD[ch]
        else:
            total += int(ch) if ch.isdigit() else ord(ch) - 65
    return chr(65 + total % 26) == code[15]


# --------------------------------------------------------------------------- #
# I detector CF veri, importati.
# --------------------------------------------------------------------------- #
def _load_cf_detectors():
    """[(regex compilata, strict), ...] per le voci CF di DETECTORS."""
    return [(rx, strict) for label, rx, _validator, strict in detectors.DETECTORS
            if label == "CF"]


CF_DETECTORS = _load_cf_detectors()


def app_would_redact(text):
    """True se la rete regex dell'app sostituirebbe `text` come CF.

    Riproduce la sola regola di `detect_regex`: con strict=True il match si scarta
    quando il checksum fallisce, con strict=False si redige comunque.
    """
    for rx, strict in CF_DETECTORS:
        match = rx.search(text)
        if not match:
            continue
        if strict and not cf_checksum_ok(match.group(0)):
            continue
        return True
    return False


# CF di base sintetico e posizioni delle cifre sostituite dall'omocodia, da destra.
_BASE = "RSSMRA85H12F205"
_OMOCODIA_POS = (14, 13, 12, 10, 9, 7, 6)


def _check_char(body):
    total = sum(_ODD[ch] if i % 2 == 0 else (int(ch) if ch.isdigit() else ord(ch) - 65)
                for i, ch in enumerate(body))
    return chr(65 + total % 26)


def make_cf(omocodia=0):
    body = list(_BASE)
    for pos in _OMOCODIA_POS[:omocodia]:
        body[pos] = "LMNPQRSTUV"[int(body[pos])]
    body = "".join(body)
    return body + _check_char(body)


class OmocodiaDetection(unittest.TestCase):
    def test_app_has_a_detector_for_the_omocodic_shape(self):
        self.assertGreaterEqual(len(CF_DETECTORS), 2,
                                "manca il detector per la forma omocodica")

    def test_ordinary_code_is_still_redacted(self):
        self.assertTrue(app_would_redact(f"C.F. {make_cf(omocodia=0)} residente in Roma"))

    def test_omocodic_code_is_redacted(self):
        for n in range(1, 8):
            code = make_cf(omocodia=n)
            with self.subTest(sostituzioni=n, cf=code):
                self.assertTrue(app_would_redact(f"C.F. {code} residente in Roma"),
                                f"CF omocodico lasciato in chiaro: {code}")

    def test_malformed_code_of_the_same_shape_is_not_redacted(self):
        """La forma omocodica da sola non basta: senza checksum non si redige.

        Serve a tenere onesto l'allargamento della regex — una parola di sedici
        lettere che combacia con la forma non deve diventare un CF.
        """
        code = make_cf(omocodia=7)
        wrong = code[:15] + ("A" if code[15] != "A" else "B")
        self.assertFalse(cf_checksum_ok(wrong))
        self.assertFalse(app_would_redact(f"riferimento {wrong} in atti"))

    def test_ordinary_code_with_broken_checksum_is_still_redacted(self):
        """Comportamento storico invariato: sulla forma ordinaria si redige comunque."""
        code = make_cf(omocodia=0)
        wrong = code[:15] + ("A" if code[15] != "A" else "B")
        self.assertFalse(cf_checksum_ok(wrong))
        self.assertTrue(app_would_redact(f"C.F. {wrong} in atti"))


if __name__ == "__main__":
    unittest.main()

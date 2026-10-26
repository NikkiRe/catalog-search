"""Shared text normalization and label-blind query-family grouping."""

import hashlib
import html
import re
import unicodedata

TOKEN = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
NUMBER = re.compile(r"(?:\d+\s+)?\d+\s*[/⁄]\s*\d+|\d+(?:[.,]\d+)?")
HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")


def clean(text):
    value = html.unescape(str(text or ""))
    value = HTML_TAG.sub(" ", value)
    # NFKC alone fuses mixed fractions: 1½ becomes 11⁄2, changing the value.
    value = re.sub(r"(?<=\d)(?=[¼½¾⅐⅑⅒⅓⅔⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞])", " ", value)
    value = unicodedata.normalize("NFKC", value).replace("⁄", "/")
    return " ".join(value.split())


def tokens(text):
    return TOKEN.findall(clean(text).lower())


def family(text):
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = NUMBER.sub(" number ", value)
    return " ".join(re.findall(r"\w+", value))


def stable_hash(value, seed=20261026):
    return int.from_bytes(hashlib.sha256(f"{seed}|{value}".encode()).digest()[:8], "big")


def document(title, brand="", description="", attributes=""):
    return clean(title)[:512], clean(brand)[:128], (clean(attributes)[:1536] + " " + clean(description)[:1536]).strip()

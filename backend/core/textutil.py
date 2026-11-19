"""Deterministic text utilities: normalization, tokens, and similarity."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_STOPWORDS = frozenset(
    """
    a o as os um uma uns umas de do da dos das em no na nos nas por para com sem sob sobre entre ate ao aos
    e ou mas que se como mais menos muito muita ja nao sim ser foi sao era tem tem ter tinha isso isto esse essa
    este esta eles elas seu sua seus suas lhe nos pelo pela pelos pelas quando onde qual quais porque pois
    the and for are was were with that this from have has had not but you your can will its into than then
    they them their there which what when where who how all any more also been being
    """.split()
)
_WORD = re.compile(r"[a-z0-9][a-z0-9_.+#-]*", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9\"“(\[])|\n+")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", strip_accents(text).lower()).strip()


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def content_hash(text: str) -> str:
    """Stable hash for exact deduplication (ignores case, accents, and spacing)."""
    return sha256_hex(normalize(text))


def tokens(text: str) -> set[str]:
    out = set()
    for raw in _WORD.findall(normalize(text)):
        word = raw.strip(".-_+")
        if len(word) >= 3 and word not in _STOPWORDS:
            out.add(word)
    return out


def containment(a: set[str], b: set[str]) -> float:
    """Fraction of `a` present in `b` (asymmetric: 'a is contained in b')."""
    return len(a & b) / len(a) if a else 0.0


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def first_sentences(text: str, limit: int = 240) -> str:
    out = ""
    for sentence in sentences(text):
        if out and len(out) + len(sentence) > limit:
            break
        out = f"{out} {sentence}".strip()
        if len(out) >= limit // 2:
            break
    return (out or text.strip())[:limit].rstrip()


def clip(text: str, limit: int) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"

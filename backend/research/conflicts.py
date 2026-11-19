"""Conflict detection between external web/research snippets on the same topic."""

from __future__ import annotations

import re

from backend.core.textutil import jaccard, tokens

_VERSION = re.compile(r"\bv?\d+(?:\.\d+){1,3}\b", re.IGNORECASE)
_NUMBER_UNIT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s?(?:%|gb|mb|kb|tb|ms|s|seg|min|h|dias?|days?|anos?|years?|tokens?|ghz|mhz|vram|ram|"
    r"usd|brl|r\$|\$|€|px|mm|cm|kg|x)\b",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"\b(n[aã]o|nunca|jamais|sem|not|never|no longer|deprecated|obsolet[oa]|removed|removido|desativad[oa]|inexistente)\b",
    re.IGNORECASE,
)


def _facts(text: str) -> set[str]:
    return {re.sub(r"\s+", "", m.group(0).lower().replace(",", ".")) for pat in (_VERSION, _NUMBER_UNIT) for m in pat.finditer(text)}


def _context(text: str) -> set[str]:
    return {t for t in tokens(text) if not re.search(r"\d", t)}


def detect_conflicts(candidate: str, existing: str, *, language: str = "en-US") -> list[str]:
    """Detects possible disagreements between two texts on the same subject (values or polarity)."""
    ctx = jaccard(_context(candidate), _context(existing))
    found: list[str] = []
    a, b = _facts(candidate), _facts(existing)
    only_a, only_b = a - b, b - a
    if ctx >= 0.25 and a and b and only_a and only_b:
        if language == "pt-BR":
            found.append(f"Valores diferentes: novo cita {', '.join(sorted(only_a)[:4])}; biblioteca cita {', '.join(sorted(only_b)[:4])}.")
        else:
            found.append(f"Conflicting values: one cites {', '.join(sorted(only_a)[:4])}; another cites {', '.join(sorted(only_b)[:4])}.")
    neg_a, neg_b = bool(_NEGATION.search(candidate)), bool(_NEGATION.search(existing))
    if ctx >= 0.5 and neg_a != neg_b:
        if language == "pt-BR":
            found.append("Polaridade oposta: um trecho afirma e o outro nega/descontinua o mesmo assunto.")
        else:
            found.append("Opposite polarity: one excerpt affirms and the other negates/deprecates the same subject.")
    return found

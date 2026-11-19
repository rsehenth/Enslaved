"""Prompt-injection sanitization for external content and conversation context summaries."""

from __future__ import annotations

import re

_ROLE_TAGS = re.compile(
    r"</?\s*(system|assistant|user|tool|ferramenta|memoria_recuperada|conhecimento_aprovado|resumo_contexto|evidencia_web|contexto_conversa)\b[^>]*>",
    re.IGNORECASE,
)
_ROLE_LINE = re.compile(r"^\s*(system|assistant|user|sistema|assistente)\s*:", re.IGNORECASE | re.MULTILINE)
_DELIMITER_TAG_RE = re.compile(
    r"</?(?:memoria_recuperada|resumo_contexto|ferramenta|skill_ativa|evidencia_web|contexto_conversa)\b[^>]*>",
    re.IGNORECASE,
)

_INJECTION_RE = re.compile(
    r"(?:"
    r"ignore\s+(?:todas?\s+as\s+|suas\s+|all\s+|previous\s+|prior\s+)?instru(?:ções|coes|ctions)"
    r"|(?:desconsidere|esqueça|esqueca)\s+(?:todas?\s+as\s+|suas\s+)?instru(?:ções|coes)"
    r"|(?:concede|conceda|conceder)\s+permiss(?:ão|ao)"
    r"|aprov(?:e|ar)\s+(?:automaticamente|este\s+conhecimento|esta\s+inclus(?:ão|ao))"
    r"|(?:execute|executa|rode|chame)\s+[^\n.!?]{0,80}\bsem\s+pedir\s+autoriza(?:ção|cao)"
    r"|\b(?:envie|send)\s+(?:o\s+conte[uú]do|os\s+dados|secrets?|data)?\s*(?:para|to)\s+https?://"
    r")",
    re.IGNORECASE,
)


def looks_like_prompt_injection(text: str) -> bool:
    """Detects obvious prompt-injection attempts in external content."""
    raw = text or ""
    return bool(_ROLE_TAGS.search(raw) or _ROLE_LINE.search(raw) or _INJECTION_RE.search(raw))


def sanitize_knowledge_text(text: str) -> str:
    """Neutralizes role/delimiter markers in external content so it is treated strictly as data."""
    return _ROLE_LINE.sub(
        lambda m: "[papel]" + m.group(0).split(":", 1)[0].strip() + ":",
        _ROLE_TAGS.sub("[tag]", text or ""),
    ).strip()


def sanitize_memory_text(text: str) -> str:
    """Neutralizes delimiter markers in repack/recall summaries."""
    cleaned = _DELIMITER_TAG_RE.sub("[tag]", text or "")
    return " ".join(cleaned.split()).strip()

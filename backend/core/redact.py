"""Redação simples de segredos em logs, erros e trilha de auditoria."""

import re

_QUERY_SECRET_RE = re.compile(
    r"([?&](?:token|api_?key|key|secret|password|passwd|auth|access_token|session)=)([^&\s\"']+)",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"\b(Bearer\s+)[A-Za-z0-9._\-+/=]{8,}", re.IGNORECASE)
_TOKEN_PREFIX_RE = re.compile(
    r"\b(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|AIza[A-Za-z0-9_-]{20,})\b"
)


def redact_secrets(text: str | None) -> str:
    """Mascara tokens/chaves comuns e parâmetros sensíveis de URL sem alterar texto comum."""
    if not text:
        return ""
    masked = _QUERY_SECRET_RE.sub(r"\1[REDACTED]", text)
    masked = _BEARER_RE.sub(r"\1[REDACTED]", masked)
    masked = _TOKEN_PREFIX_RE.sub("[REDACTED]", masked)
    return masked

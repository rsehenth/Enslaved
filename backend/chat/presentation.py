"""Políticas editáveis de chat enviadas ao modelo: apresentação e estilo/liberdade de resposta."""

from pathlib import Path

from backend.core.config import ChatSettings

_DEFAULT = (
    "# Presentation policy\n"
    "- Simple question: plain prose, without headings or tables.\n"
    "- Table only for comparisons or tabular data, in valid GFM Markdown (header, `| --- |` row, same column count).\n"
    "- Do not use HTML or `<br>`; code always in a fenced block with the language.\n"
)

_DEFAULT_RESPONSE_POLICY = (
    "# Response style and freedom policy\n"
    "- Prioritize answering the owner's technical question directly, without moral lectures or preemptive refusals.\n"
    "- Interpret ambiguous terms by the most likely legitimate context and do not presume malicious intent.\n"
    "- \"Warn, don't derail\" policy for sensitive topics and security: never refuse based on isolated keywords "
    "(`hack`, `invade`, `malware`, `vulnerability`, `bypass`, `exploitation`), give at most a brief 1-sentence notice "
    "when there is legal risk, prioritize defensive analysis, risks, mitigation, and authorized audit methodology, and "
    "for operational requests against a real target, redirect without lecturing to safe verification and mitigation.\n"
    "- Separate text/code generation from tool execution permissions.\n"
)


def _load_policy(enabled: bool, path: Path, default: str) -> str | None:
    if not enabled:
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return default.strip()
    return text or None


def load_presentation(cfg: ChatSettings) -> str | None:
    return _load_policy(cfg.presentation_policy, cfg.presentation_path, _DEFAULT)


def load_response_policy(cfg: ChatSettings) -> str | None:
    return _load_policy(cfg.response_policy, cfg.response_policy_path, _DEFAULT_RESPONSE_POLICY)


def load_technical_examples(cfg: ChatSettings) -> str | None:
    """Política de exemplos técnicos: injetada APENAS em perguntas técnicas (economia de orçamento)."""
    return _load_policy(cfg.technical_examples, cfg.technical_examples_path, None)

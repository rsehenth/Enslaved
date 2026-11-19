import re

import yaml

_PATTERN = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Separa o frontmatter YAML inicial do corpo markdown. Levanta ValueError se inválido."""
    match = _PATTERN.match(text)
    if match is None:
        return {}, text.strip()
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"frontmatter YAML inválido: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError("frontmatter deve ser um mapa YAML")
    return meta, text[match.end() :].strip()

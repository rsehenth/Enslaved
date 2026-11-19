from pathlib import Path

from pydantic import BaseModel

from backend.core.frontmatter import parse_frontmatter


class DirectiveError(Exception):
    """Diretrizes ausentes ou inválidas."""


class Directives(BaseModel):
    name: str = "Enslaved"
    language: str = "en-US"
    version: str | None = None
    source_path: Path | None = None
    system_prompt: str = ""


def load_directives(path: Path) -> Directives:
    """Carrega config/directives.md (editável pelo proprietário) como prompt de sistema."""
    if not path.is_file():
        raise DirectiveError(f"Arquivo de diretrizes não encontrado: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        meta, body = parse_frontmatter(text)
    except ValueError as exc:
        raise DirectiveError(f"Diretrizes inválidas em {path}: {exc}") from exc
    if not body:
        raise DirectiveError(f"Diretrizes vazias em {path}")
    return Directives(
        name=str(meta.get("name") or "Enslaved"),
        language=str(meta.get("language") or "en-US"),
        version=str(meta["version"]) if meta.get("version") is not None else None,
        source_path=path,
        system_prompt=body,
    )

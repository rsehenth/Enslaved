import re
from dataclasses import dataclass, field
from pathlib import Path

from backend.core.frontmatter import parse_frontmatter

SKILL_FILE = "SKILL.md"
_SKILL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class SkillRecord:
    id: str
    name: str
    description: str
    path: Path
    version: str | None = None
    metadata: dict = field(default_factory=dict)
    instructions: str = ""
    has_scripts: bool = False
    has_references: bool = False
    keywords: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)


def _string_list(meta: dict, key: str, path: Path, errors: list) -> list[str] | None:
    """Extrai e valida uma lista de strings opcional do frontmatter. None = ausente."""
    value = meta.pop(key, None)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    errors.append({"path": str(path), "reason": f"metadata '{key}' deve ser uma lista de strings"})
    return None


class SkillRegistry:
    """Descobre e valida Skills (SKILL.md + metadata) em diretórios configurados.

    Nesta fase nada é executado: scripts/ e references/ são apenas detectados.
    """

    def __init__(self) -> None:
        self._skills: dict[str, SkillRecord] = {}
        self.errors: list[dict] = []

    def discover(self, directories: list[Path]) -> None:
        self._skills.clear()
        self.errors.clear()
        for directory in directories:
            if not directory.is_dir():
                self.errors.append({"path": str(directory), "reason": "diretório de skills não encontrado"})
                continue
            resolved_dir = directory.resolve()
            for child in sorted(directory.iterdir()):
                if child.is_dir():
                    self._load(child, resolved_dir)

    def _load(self, path: Path, root_dir: Path | None = None) -> None:
        skill_id = path.name
        if not _SKILL_ID.match(skill_id):
            self._error(path, f"id de skill inválido: {skill_id!r}")
            return
        skill_md = path / SKILL_FILE
        if not skill_md.is_file():
            self._error(path, f"{SKILL_FILE} não encontrado")
            return
        if root_dir is not None:
            resolved_md = skill_md.resolve()
            if not resolved_md.is_relative_to(root_dir):
                self._error(path, "SKILL.md aponta para fora do diretório de skills")
                return
        try:
            meta, body = parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        except ValueError as exc:
            self._error(path, str(exc))
            return
        name = meta.pop("name", None)
        description = meta.pop("description", None)
        version = meta.pop("version", None)
        if not name or not isinstance(name, str):
            self._error(path, "metadata obrigatória ausente: 'name'")
            return
        if not description or not isinstance(description, str):
            self._error(path, "metadata obrigatória ausente: 'description'")
            return
        keywords = _string_list(meta, "keywords", path, self.errors)
        requires = _string_list(meta, "requires", path, self.errors)
        if keywords is None or requires is None:
            return
        if skill_id in self._skills:
            self._error(path, f"id de skill duplicado: {skill_id}")
            return
        self._skills[skill_id] = SkillRecord(
            id=skill_id,
            name=name,
            description=description,
            path=path,
            version=str(version) if version is not None else None,
            metadata=dict(meta),
            instructions=body,
            has_scripts=(path / "scripts").is_dir(),
            has_references=(path / "references").is_dir(),
            keywords=keywords,
            requires=requires,
        )

    def _error(self, path: Path, reason: str) -> None:
        self.errors.append({"path": str(path), "reason": reason})

    def list_skills(self) -> list[SkillRecord]:
        return sorted(self._skills.values(), key=lambda s: s.name.lower())

    def get(self, skill_id: str) -> SkillRecord | None:
        if not skill_id or not _SKILL_ID.match(skill_id):
            return None
        return self._skills.get(skill_id)

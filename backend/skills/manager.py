"""Motor de Skills sobre o registro de descoberta.

Responsabilidades da Fase 2A:
- estado habilitada/desabilitada persistido (data/skills-state.json);
- verificação de compatibilidade (`requires`: ferramentas/capacidades presentes);
- catálogo compacto para o prompt (uma linha por skill; nada é carregado à toa);
- seleção contextual por palavras-chave (sugestão de ativação);
- injeção progressiva: apenas skills explicitamente ATIVADAS têm o corpo das
  instruções carregado no contexto.

Carregar instruções ≠ executar ferramentas: skills não executam nada aqui;
`scripts/` permanece apenas sinalizado (decisão da Fase 1, mantida).
"""

import json
import re
from pathlib import Path

from backend.skills.discovery import SkillRecord, SkillRegistry

_TOKEN = re.compile(r"[a-záàâãéêíóôõúüçñ0-9]+", re.IGNORECASE)
_DELIMITER_TAG_RE = re.compile(
    r"</?(?:ferramenta|memoria_recuperada|resumo_contexto|skill_ativa)\b[^>]*>",
    re.IGNORECASE,
)


def _sanitize_skill_text(text: str) -> str:
    return _DELIMITER_TAG_RE.sub("[tag]", text or "")


def _tokens(text: str) -> set[str]:
    return {token.group(0).lower() for token in _TOKEN.finditer(text or "")}


class SkillManager:
    def __init__(self, registry: SkillRegistry, state_path: Path | None = None):
        self.registry = registry
        self.state_path = state_path
        self._disabled: set[str] = set()
        self.load_state()

    # ------------------------------------------------------------- estado

    def load_state(self) -> None:
        if self.state_path is None or not self.state_path.is_file():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # estado corrompido: segue com padrão (todas habilitadas)
        disabled = data.get("disabled", [])
        if isinstance(disabled, list):
            self._disabled = {str(item) for item in disabled}

    def save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"disabled": sorted(self._disabled)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def is_enabled(self, skill_id: str) -> bool:
        return self.registry.get(skill_id) is not None and skill_id not in self._disabled

    def set_enabled(self, skill_id: str, enabled: bool) -> bool:
        """Alterna o estado. Retorna False se a skill não existe."""
        if self.registry.get(skill_id) is None:
            return False
        if enabled:
            self._disabled.discard(skill_id)
        else:
            self._disabled.add(skill_id)
        self.save_state()
        return True

    # ------------------------------------------------------ compatibilidade

    def missing_requirements(self, skill: SkillRecord, tool_ids: set[str], capabilities: set[str]) -> list[str]:
        """Requisitos ausentes. Entradas: 'tool:<id>', 'cap:<capacidade>' ou nome simples
        (tentado como id de ferramenta e depois como capacidade)."""
        missing = []
        for requirement in skill.requires:
            if requirement.startswith("tool:"):
                target = requirement[5:]
                if target not in tool_ids:
                    missing.append(requirement)
            elif requirement.startswith("cap:"):
                target = requirement[4:]
                if target not in capabilities:
                    missing.append(requirement)
            elif requirement not in tool_ids and requirement not in capabilities:
                missing.append(requirement)
        return missing

    def is_compatible(self, skill: SkillRecord, tool_ids: set[str], capabilities: set[str]) -> bool:
        return not self.missing_requirements(skill, tool_ids, capabilities)

    def usable_skills(self, tool_ids: set[str], capabilities: set[str]) -> list[SkillRecord]:
        return [
            skill
            for skill in self.registry.list_skills()
            if self.is_enabled(skill.id) and self.is_compatible(skill, tool_ids, capabilities)
        ]

    # -------------------------------------------------------------- prompt

    def catalog_block(self, tool_ids: set[str], capabilities: set[str]) -> str | None:
        """Catálogo compacto (id + descrição) das skills utilizáveis, ou None."""
        skills = self.usable_skills(tool_ids, capabilities)
        if not skills:
            return None
        lines = []
        for skill in skills:
            desc = " ".join(_sanitize_skill_text(skill.description).split())[:300]
            lines.append(f"- {skill.id}: {desc}")
        return "\n".join(lines)

    def instructions_block(
        self,
        activated_ids: list[str],
        tool_ids: set[str],
        capabilities: set[str],
    ) -> tuple[str | None, list[str]]:
        """Corpo completo das skills ativadas (carregamento progressivo) + ids ignorados."""
        blocks: list[str] = []
        skipped: list[str] = []
        for skill_id in activated_ids:
            skill = self.registry.get(skill_id)
            if (
                skill is None
                or not self.is_enabled(skill_id)
                or not self.is_compatible(skill, tool_ids, capabilities)
            ):
                skipped.append(skill_id)
                continue
            body = _sanitize_skill_text(skill.instructions.strip())
            blocks.append(
                f"### Skill ativada: {skill.name} (id: {skill.id})\n"
                f"<skill_ativa id=\"{skill.id}\">\n{body}\n</skill_ativa>"
            )
        if not blocks:
            return None, skipped
        return "\n\n---\n\n".join(blocks), skipped

    # ------------------------------------------------- seleção contextual

    def select(self, user_message: str, limit: int = 3, tool_ids: set[str] | None = None, capabilities: set[str] | None = None) -> list[SkillRecord]:
        """Skills candidatas pela interseção de palavras-chave (sugestão, não execução)."""
        tool_ids = tool_ids or set()
        capabilities = capabilities or set()
        message_tokens = _tokens(user_message)
        if not message_tokens:
            return []
        scored: list[tuple[int, SkillRecord]] = []
        for skill in self.usable_skills(tool_ids, capabilities):
            skill_tokens = (
                _tokens(skill.id)
                | _tokens(skill.name)
                | _tokens(skill.description)
                | {keyword.lower() for keyword in skill.keywords}
            )
            overlap = message_tokens & skill_tokens
            if overlap:
                scored.append((len(overlap), skill))
        scored.sort(key=lambda pair: (-pair[0], pair[1].id))
        return [skill for _score, skill in scored[: max(1, limit)]]

    # ------------------------------------------------------------------ resumo

    def summary(self, tool_ids: set[str] | None = None, capabilities: set[str] | None = None) -> dict:
        tool_ids = tool_ids or set()
        capabilities = capabilities or set()
        incompatible = [
            {"id": skill.id, "missing": self.missing_requirements(skill, tool_ids, capabilities)}
            for skill in self.registry.list_skills()
            if not self.is_compatible(skill, tool_ids, capabilities)
        ]
        return {
            "total": len(self.registry.list_skills()),
            "enabled": sum(1 for skill in self.registry.list_skills() if skill.id not in self._disabled),
            "disabled_ids": sorted(self._disabled),
            "incompatible": incompatible,
        }

from pathlib import Path

from backend.core.config import PROJECT_ROOT, Settings

DIRECTIVES_TEXT = """---
name: Enslaved
language: en-US
version: 1
---

# Goal
Reply directly and execute tasks permitted by the owner.
"""


def make_settings(tmp_path: Path, **overrides) -> Settings:
    """Settings de teste: provider simulado, skills temporárias + skill demo do projeto."""
    skills_dir = tmp_path / "skills"
    valid = skills_dir / "skill-teste"
    valid.mkdir(parents=True, exist_ok=True)
    (valid / "SKILL.md").write_text(
        "---\nname: skill-teste\ndescription: Skill de teste para validação.\n---\n\n# Instruções\nUse a skill de teste.\n",
        encoding="utf-8",
    )
    broken = skills_dir / "skill-quebrada"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "SKILL.md").write_text("---\nname: skill-quebrada\n---\n\nSem description.\n", encoding="utf-8")

    directives_path = tmp_path / "directives.md"
    directives_path.write_text(DIRECTIVES_TEXT, encoding="utf-8")

    data = {
        "app": {"name": "Enslaved Teste"},
        "provider": {"name": "simulated", "model": "simulado-1b"},
        "storage": {"database_path": tmp_path / "data" / "test.db"},
        "memory": {"base_dir": tmp_path / "data" / "memory"},
        "directives": {"path": directives_path},
        "skills": {"directories": [skills_dir, PROJECT_ROOT / "skills"]},
        # testes legados exercitam o caminho de streaming simples; testes de
        # ferramentas habilitam explicitamente
        "tools": {"enabled": False},
    }
    data.update(overrides)
    return Settings.model_validate(data)

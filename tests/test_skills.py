from backend.core.config import PROJECT_ROOT
from backend.skills.discovery import SkillRegistry


def _write_skill(base, name, frontmatter, body="# Instruções\nConteúdo."):
    path = base / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"---\n{frontmatter}---\n\n{body}\n", encoding="utf-8")
    return path


def test_discovery_finds_valid_skill(tmp_path):
    _write_skill(tmp_path, "minha-skill", "name: minha-skill\ndescription: Uma skill válida.\n")
    registry = SkillRegistry()
    registry.discover([tmp_path])
    skills = registry.list_skills()
    assert [s.id for s in skills] == ["minha-skill"]
    assert skills[0].description == "Uma skill válida."
    assert "Instruções" in skills[0].instructions
    assert registry.errors == []


def test_discovery_reports_invalid_skills(tmp_path):
    _write_skill(tmp_path, "sem-description", "name: sem-description\n")
    no_md = tmp_path / "sem-skill-md"
    no_md.mkdir()
    _write_skill(tmp_path, "id invalido", "name: x\ndescription: y\n")
    registry = SkillRegistry()
    registry.discover([tmp_path])
    assert registry.list_skills() == []
    reasons = " ".join(error["reason"] for error in registry.errors)
    assert "description" in reasons
    assert "SKILL.md" in reasons
    assert "inválido" in reasons


def test_discovery_detects_scripts_and_references(tmp_path):
    path = _write_skill(tmp_path, "com-recursos", "name: com-recursos\ndescription: tem arquivos.\n")
    (path / "scripts").mkdir()
    (path / "references").mkdir()
    registry = SkillRegistry()
    registry.discover([tmp_path])
    skill = registry.get("com-recursos")
    assert skill.has_scripts is True
    assert skill.has_references is True


def test_missing_skill_directory_reported(tmp_path):
    registry = SkillRegistry()
    registry.discover([tmp_path / "não-existe"])
    assert registry.list_skills() == []
    assert "não encontrado" in registry.errors[0]["reason"]


def test_demo_skill_discovered_in_project():
    registry = SkillRegistry()
    registry.discover([PROJECT_ROOT / "skills"])
    skill = registry.get("hello-enslaved")
    assert skill is not None
    assert "Demo skill" in skill.description
    assert skill.has_references is True


def test_get_rejects_malicious_ids(tmp_path):
    registry = SkillRegistry()
    registry.discover([tmp_path])
    assert registry.get("../fora") is None
    assert registry.get("") is None

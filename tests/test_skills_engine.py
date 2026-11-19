import json

from backend.skills.discovery import SkillRegistry
from backend.skills.manager import SkillManager


def _write_skill(base, name, frontmatter, body="# Instruções\nConteúdo da skill."):
    path = base / name
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"---\n{frontmatter}---\n\n{body}\n", encoding="utf-8")
    return path


def make_manager(tmp_path, skills=None):
    if skills:
        for name, frontmatter in skills.items():
            _write_skill(tmp_path / "skills", name, frontmatter)
    registry = SkillRegistry()
    registry.discover([tmp_path / "skills"])
    return SkillManager(registry, tmp_path / "estado" / "skills-state.json")


DEMO = "name: demo\ndescription: Uma skill de demonstração.\nkeywords:\n  - apresentacao\n  - saudacao\n"
WITH_TOOL = "name: com-fs\ndescription: Usa leitura de arquivos.\nrequires:\n  - tool:fs.read\n"


def test_state_persists_disabled_skills(tmp_path):
    manager = make_manager(tmp_path, {"demo": DEMO})
    assert manager.is_enabled("demo") is True
    assert manager.set_enabled("demo", False) is True
    assert manager.is_enabled("demo") is False

    reloaded = SkillManager(manager.registry, manager.state_path)
    assert reloaded.is_enabled("demo") is False

    reloaded.set_enabled("demo", True)
    assert SkillManager(manager.registry, manager.state_path).is_enabled("demo") is True


def test_set_enabled_unknown_skill_returns_false(tmp_path):
    manager = make_manager(tmp_path)
    assert manager.set_enabled("inexistente", False) is False
    assert manager.state_path.exists() is False


def test_corrupted_state_falls_back_to_default(tmp_path):
    manager = make_manager(tmp_path, {"demo": DEMO})
    manager.state_path.parent.mkdir(parents=True, exist_ok=True)
    manager.state_path.write_text("{json quebrado", encoding="utf-8")
    reloaded = SkillManager(manager.registry, manager.state_path)
    assert reloaded.is_enabled("demo") is True


def test_requirements_check_tool_and_capability(tmp_path):
    manager = make_manager(tmp_path, {"com-fs": WITH_TOOL})
    skill = manager.registry.get("com-fs")
    assert manager.missing_requirements(skill, {"fs.read"}, set()) == []
    assert manager.missing_requirements(skill, {"fs.read"}, {"fs.read"}) == []
    assert manager.missing_requirements(skill, set(), {"outra"}) == ["tool:fs.read"]
    # requisito explícito de ferramenta não é satisfeito por capacidade homônima
    assert manager.missing_requirements(skill, set(), {"fs.read"}) == ["tool:fs.read"]


def test_catalog_block_lists_only_enabled_and_compatible(tmp_path):
    manager = make_manager(tmp_path, {"demo": DEMO, "com-fs": WITH_TOOL})
    sem_ferramenta = manager.catalog_block(set(), set())
    assert "demo:" in sem_ferramenta
    assert "com-fs" not in sem_ferramenta
    com_ferramenta = manager.catalog_block({"fs.read"}, set())
    assert "com-fs" in com_ferramenta
    manager.set_enabled("demo", False)
    assert "demo" not in manager.catalog_block({"fs.read"}, set())


def test_catalog_block_none_when_nothing_usable(tmp_path):
    manager = make_manager(tmp_path)
    assert manager.catalog_block(set(), set()) is None


def test_select_ranks_by_keyword_overlap(tmp_path):
    manager = make_manager(
        tmp_path,
        {
            "demo": "name: demo\ndescription: Apresentação e saudação do assistente.\nkeywords:\n  - saudacao\n",
            "outra": "name: outra\ndescription: Documenta APIs.\nkeywords:\n  - api\n",
        },
    )
    selecionadas = manager.select("Quero uma saudacao bonita para minha apresentacao")
    assert [skill.id for skill in selecionadas] == ["demo"]
    assert [skill.id for skill in manager.select("use a api rest")] == ["outra"]
    assert manager.select("xyz nada a ver") == []


def test_select_excludes_disabled_and_incompatible(tmp_path):
    manager = make_manager(tmp_path, {"com-fs": WITH_TOOL})
    assert [skill.id for skill in manager.select("leitura de arquivos", tool_ids={"fs.read"})] == ["com-fs"]
    manager.set_enabled("com-fs", False)
    assert manager.select("leitura de arquivos", tool_ids={"fs.read"}) == []
    manager.set_enabled("com-fs", True)
    # sem a ferramenta exigida, a skill não é utilizável
    assert manager.select("leitura de arquivos", tool_ids=set()) == []


def test_instructions_block_injects_only_activated(tmp_path):
    manager = make_manager(tmp_path, {"demo": DEMO})
    block, skipped = manager.instructions_block(["demo"], set(), set())
    assert block is not None and "Instruções" in block and "demo" in block
    assert skipped == []
    block, skipped = manager.instructions_block([], set(), set())
    assert block is None
    block, skipped = manager.instructions_block(["inexistente", "demo"], set(), set())
    assert skipped == ["inexistente"]


def test_instructions_block_skips_disabled_or_incompatible(tmp_path):
    manager = make_manager(tmp_path, {"com-fs": WITH_TOOL})
    block, skipped = manager.instructions_block(["com-fs"], set(), set())
    assert block is None and skipped == ["com-fs"]
    block, skipped = manager.instructions_block(["com-fs"], {"fs.read"}, set())
    assert block is not None and skipped == []
    manager.set_enabled("com-fs", False)
    block, skipped = manager.instructions_block(["com-fs"], {"fs.read"}, set())
    assert block is None and skipped == ["com-fs"]


def test_summary_reports_counts_and_incompatibilities(tmp_path):
    manager = make_manager(tmp_path, {"demo": DEMO, "com-fs": WITH_TOOL})
    summary = manager.summary(set(), set())
    assert summary["total"] == 2
    # ambas habilitadas; com-fs é incompatível, mas isso não é "desabilitada"
    assert summary["enabled"] == 2
    assert summary["disabled_ids"] == []
    assert summary["incompatible"] == [{"id": "com-fs", "missing": ["tool:fs.read"]}]
    manager.set_enabled("demo", False)
    assert SkillManager(manager.registry, manager.state_path).summary(set(), set())["enabled"] == 1


def test_discovery_parses_keywords_and_requires(tmp_path):
    manager = make_manager(tmp_path, {"demo": DEMO, "com-fs": WITH_TOOL})
    assert manager.registry.get("demo").keywords == ["apresentacao", "saudacao"]
    assert manager.registry.get("com-fs").requires == ["tool:fs.read"]


def test_discovery_reports_invalid_keyword_type(tmp_path):
    base = tmp_path / "skills"
    _write_skill(base, "ruim", "name: ruim\ndescription: x\nkeywords: 42\n")
    registry = SkillRegistry()
    registry.discover([base])
    assert registry.get("ruim") is None
    assert "keywords" in registry.errors[0]["reason"]


def test_discovery_accepts_single_keyword_string_shorthand(tmp_path):
    base = tmp_path / "skills"
    _write_skill(base, "simples", "name: simples\ndescription: x\nkeywords: unica\n")
    registry = SkillRegistry()
    registry.discover([base])
    assert registry.get("simples").keywords == ["unica"]

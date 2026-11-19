import pytest

from backend.core.config import ConfigError, Settings, load_settings
from backend.core.directives import DirectiveError, load_directives


def test_defaults_without_config_file(tmp_path):
    settings = load_settings(config_path=tmp_path / "inexistente.yaml", root=tmp_path)
    assert settings.app.host == "127.0.0.1"
    assert settings.provider.model == "hf.co/bartowski/Qwen_Qwen3.5-9B-GGUF:Q6_K"
    assert settings.ollama.num_ctx == 16384
    assert settings.ollama.temperature == 0.7
    assert settings.ollama.think is False


def test_rollback_to_4b_via_env_or_yaml(tmp_path, monkeypatch):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("provider:\n  model: qwen3.5:4b\n", encoding="utf-8")
    assert load_settings(config_path=cfg, root=tmp_path).provider.model == "qwen3.5:4b"
    monkeypatch.setenv("ENSLAVED_MODEL", "qwen3.5:4b")
    assert load_settings(config_path=tmp_path / "nada.yaml", root=tmp_path).provider.model == "qwen3.5:4b"


def test_loads_yaml_and_merges_defaults(tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text(
        "app:\n  port: 9999\nprovider:\n  model: outro-modelo\nollama:\n  temperature: 0.3\n",
        encoding="utf-8",
    )
    settings = load_settings(config_path=config, root=tmp_path)
    assert settings.app.port == 9999
    assert settings.provider.model == "outro-modelo"
    assert settings.ollama.temperature == 0.3
    assert settings.ollama.base_url == "http://127.0.0.1:11434"
    assert settings.provider.name == "ollama"


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("ENSLAVED_PORT", "9001")
    monkeypatch.setenv("ENSLAVED_MODEL", "modelo-env")
    settings = load_settings(config_path=tmp_path / "nada.yaml", root=tmp_path)
    assert settings.app.port == 9001
    assert settings.provider.model == "modelo-env"


def test_invalid_env_value_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("ENSLAVED_PORT", "não-numérico")
    with pytest.raises(ConfigError):
        load_settings(config_path=tmp_path / "nada.yaml", root=tmp_path)


def test_relative_paths_resolved_against_root(tmp_path):
    settings = load_settings(config_path=tmp_path / "nada.yaml", root=tmp_path)
    assert settings.storage.database_path == tmp_path / "data" / "enslaved.db"
    assert settings.directives.path == tmp_path / "config" / "directives.md"
    assert settings.skills.directories == [tmp_path / "skills"]


def test_invalid_yaml_raises(tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text("- apenas\n- lista\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(config_path=config, root=tmp_path)


def test_load_directives_from_file(tmp_path):
    path = tmp_path / "directives.md"
    path.write_text(
        "---\nname: Enslaved\nlanguage: pt-BR\n---\n\n# Objetivo\nResponder direto.\n",
        encoding="utf-8",
    )
    directives = load_directives(path)
    assert directives.name == "Enslaved"
    assert directives.language == "pt-BR"
    assert "# Objetivo" in directives.system_prompt


def test_directives_without_frontmatter(tmp_path):
    path = tmp_path / "directives.md"
    path.write_text("# Apenas corpo\nTexto livre.\n", encoding="utf-8")
    directives = load_directives(path)
    assert directives.system_prompt.startswith("# Apenas corpo")
    assert directives.name == "Enslaved"


def test_empty_directives_raises(tmp_path):
    path = tmp_path / "directives.md"
    path.write_text("---\nname: x\n---\n\n", encoding="utf-8")
    with pytest.raises(DirectiveError):
        load_directives(path)


def test_missing_directives_raises(tmp_path):
    with pytest.raises(DirectiveError):
        load_directives(tmp_path / "inexistente.md")

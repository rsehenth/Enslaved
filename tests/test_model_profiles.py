"""Perfis por modelo (Qwen3.8 experimental / Qwen3.5 intacto), troca com reinício,
orçamento de contexto adaptativo e bloqueio multimodal para modelo somente texto.

Todos os testes são offline (provider simulado / provider Ollama apenas construído, sem rede)."""

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.core.config import load_settings
from backend.main import create_app
from backend.memory.manager import MemoryManager
from backend.providers.ollama import OllamaProvider
from backend.providers.registry import create_provider
from backend.storage.attachments import AttachmentRejected
from backend.storage.conversations import Message
from backend.storage.database import Database
from tests.helpers import make_settings

QWEN38 = "hf.co/cognitivers/Qwen3.8-27B-OBLITERATED-12GB-GGUF:IQ2_M"
QWEN35 = "hf.co/bartowski/Qwen_Qwen3.5-9B-GGUF:Q6_K"


def catalog(active: str = QWEN38, **overrides) -> dict:
    data = {
        "name": "simulated",
        "model": active,
        "models": {
            QWEN38: {
                "num_ctx": 2048,
                "num_predict": 512,
                "temperature": 0.7,
                "think": False,
                "supports_vision": False,
                "experimental": True,
            },
            QWEN35: {"num_ctx": 16384, "num_predict": 1024, "supports_vision": True},
        },
    }
    data.update(overrides)
    return {"provider": data}


# ---------------------------------------------------------------- perfis por modelo


def test_active_profile_resolves_per_model(tmp_path):
    settings = make_settings(tmp_path, **catalog())
    assert settings.active_profile().num_ctx == 2048
    assert settings.active_profile().num_predict == 512
    assert settings.active_profile().experimental is True
    assert settings.active_profile().supports_vision is False

    settings.provider.model = QWEN35
    assert settings.active_profile().num_ctx == 16384
    assert settings.active_profile().num_predict == 1024
    assert settings.active_profile().experimental is False
    assert settings.active_profile().supports_vision is True
    # campo ausente no perfil herda de ollama.*
    assert settings.active_profile().temperature == settings.ollama.temperature


def test_profile_without_entry_falls_back_to_globals(tmp_path):
    settings = make_settings(tmp_path)
    assert QWEN38 not in settings.provider.models
    profile = settings.active_profile()
    assert profile.num_ctx == settings.ollama.num_ctx
    assert profile.supports_vision is True
    assert profile.experimental is False


def test_create_provider_applies_profile(tmp_path):
    settings = make_settings(tmp_path, **catalog())
    provider = create_provider(settings)
    assert provider.capabilities == {"vision": False, "tools": None, "experimental": True}

    settings.provider.name = "ollama"
    ollama = create_provider(settings)
    assert isinstance(ollama, OllamaProvider)
    assert ollama.default_options["num_ctx"] == 2048
    assert ollama.default_options["num_predict"] == 512
    assert ollama.default_options["temperature"] == 0.7
    assert ollama.capabilities["vision"] is False


def test_extra_options_empty_sends_nothing_and_populated_merges(tmp_path):
    settings = make_settings(tmp_path, **catalog(active=QWEN38, name="ollama"))
    settings.provider.models[QWEN38].extra_options = {"min_p": 0.05}
    ollama = create_provider(settings)
    assert ollama.default_options["min_p"] == 0.05
    # samplers padrão do projeto permanecem ausentes até validação manual
    assert "top_p" not in ollama.default_options
    assert "repeat_penalty" not in ollama.default_options


# ------------------------------------------------------- seleção runtime com reinício


def test_runtime_model_selection_file_applied(tmp_path, monkeypatch):
    monkeypatch.delenv("ENSLAVED_MODEL", raising=False)
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(f"provider:\n  model: {QWEN35}\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "model-selection.json").write_text(json.dumps({"model": QWEN38}), encoding="utf-8")

    settings = load_settings(config_path=cfg, root=tmp_path)
    assert settings.provider.model == QWEN38


def test_env_model_wins_over_runtime_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("ENSLAVED_MODEL", "qwen3.5:4b")
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(f"provider:\n  model: {QWEN35}\n", encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "model-selection.json").write_text(json.dumps({"model": QWEN38}), encoding="utf-8")

    settings = load_settings(config_path=cfg, root=tmp_path)
    assert settings.provider.model == "qwen3.5:4b"


def test_endpoint_select_model_persists_and_requires_restart(tmp_path):
    settings = make_settings(tmp_path, **catalog())
    with TestClient(create_app(settings)) as client:
        res = client.post("/api/model", json={"model": QWEN35})
        assert res.status_code == 200
        body = res.json()
        assert body["restart_required"] is True
        assert body["selected"] == QWEN35
        assert body["active"] == QWEN38

        # persistido para a próxima inicialização (data/ é ignorado pelo git)
        saved = json.loads((settings.storage.database_path.parent / "model-selection.json").read_text(encoding="utf-8"))
        assert saved == {"model": QWEN35}

        same = client.post("/api/model", json={"model": QWEN35})
        assert same.json()["restart_required"] is True  # continua pendente até reiniciar

        # selecionar o modelo JÁ ativo no processo não exige reinício
        active = client.post("/api/model", json={"model": QWEN38})
        assert active.json()["restart_required"] is False

        unknown = client.post("/api/model", json={"model": "modelo-fantasma"})
        assert unknown.status_code == 404

        missing = client.post("/api/model", json={})
        assert missing.status_code == 400


def test_config_endpoint_exposes_catalog_and_flags(tmp_path):
    settings = make_settings(tmp_path, **catalog())
    with TestClient(create_app(settings)) as client:
        config = client.get("/api/config").json()
    provider = config["provider"]
    assert provider["model"] == QWEN38
    assert provider["experimental"] is True
    assert provider["supports_vision"] is False
    assert provider["num_ctx"] == 2048
    ids = {entry["id"] for entry in provider["models"]}
    assert ids == {QWEN38, QWEN35}
    by_id = {entry["id"]: entry for entry in provider["models"]}
    assert by_id[QWEN35]["experimental"] is False
    assert by_id[QWEN35]["num_ctx"] == 16384
    # defaults refletidos vêm do perfil ativo (controles compatíveis com o modelo)
    assert config["defaults"]["num_predict"] == 512


# ------------------------------------------------- orçamento de contexto adaptativo


def _populate(manager: MemoryManager, turns: list[tuple[str, str]]) -> str:
    conv = manager.conv_repo.create_conversation("Orçamento")
    for user, assistant in turns:
        manager.conv_repo.add_message(conv.id, "user", user)
        manager.conv_repo.add_message(conv.id, "assistant", assistant)
    manager.sync_conversation_markdown(conv.id)
    return conv.id


def test_budget_drops_optional_blocks_in_priority_order(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=2048, num_predict=512)  # input_budget=1536
        conv_id = _populate(manager, [("Pergunta anterior sobre arquitetura", "Resposta anterior detalhada")] * 2)
        manager.conv_repo.add_message(conv_id, "user", "Pergunta atual")  # mensagem atual entra no histórico (fluxo real)
        history = manager.conv_repo.get_messages(conv_id)

        ctx = asyncio.run(
            manager.prepare_context(
                conversation_id=conv_id,
                query_text="Pergunta atual",
                history=history,
                system_base_text="Diretrizes essenciais de segurança",
                tools_block="T" * 3000,
                skills_block="S" * 3000,
                tools_schema=[{"type": "function", "function": {"name": "fs.read", "description": "d" * 3000}}],
                optional_blocks={"response_policy": "P" * 3000, "presentation": "A" * 3000},
            )
        )
        # Tudo que era opcional foi descartado para caber; diretrizes e mensagem atual permanecem
        assert ctx.dropped_blocks == {"presentation", "response_policy", "skills", "tools"}
        assert ctx.budget.over_hard_limit is False
        assert ctx.budget.usage_ratio <= 1.0
        assert ctx.budget.system_tokens > 0
        last_user = next(m for m in reversed(history) if m.role == "user")
        assert any(m.id == last_user.id for m in ctx.history)
    finally:
        db.close()


def test_budget_partial_drop_keeps_tools_when_they_fit(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=2048, num_predict=512)
        conv_id = _populate(manager, [("Pergunta anterior", "Resposta anterior")])
        history = manager.conv_repo.get_messages(conv_id)

        # Blocos opcionais pequenos: apenas a apresentação (a mais descartável) sai
        ctx = asyncio.run(
            manager.prepare_context(
                conversation_id=conv_id,
                query_text="Pergunta atual",
                history=history,
                system_base_text="Diretrizes",
                tools_block="Ferramentas curtas",
                skills_block="Skill curta",
                tools_schema=[{"type": "function", "function": {"name": "app.diagnostics"}}],
                optional_blocks={"response_policy": "P" * 1400, "presentation": "A" * 3300},
            )
        )
        assert ctx.dropped_blocks == {"presentation"}
        assert "tools" not in ctx.dropped_blocks
        assert ctx.budget.over_hard_limit is False
    finally:
        db.close()


def test_budget_unfit_minimum_signals_instead_of_fake_completeness(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=2048, num_predict=512)
        conv = manager.conv_repo.create_conversation("Estouro")
        huge_user = "X" * 9000  # ~3000 tokens estimados > input_budget 1536
        manager.conv_repo.add_message(conv.id, "user", huge_user)
        manager.sync_conversation_markdown(conv.id)
        history = manager.conv_repo.get_messages(conv.id)

        ctx = asyncio.run(
            manager.prepare_context(
                conversation_id=conv.id,
                query_text=huge_user,
                history=history,
                system_base_text="Diretrizes",
                optional_blocks={"response_policy": "P" * 1500, "presentation": "A" * 900},
            )
        )
        assert ctx.dropped_blocks == {"presentation", "response_policy"}
        assert ctx.budget.over_hard_limit is True
        # a mensagem do usuário NÃO é truncada nem descartada: a limitação é sinalizada
        assert ctx.history[-1].content == huge_user
    finally:
        db.close()


def test_budget_flags_surface_notice_for_stream_meta(tmp_path):
    from backend.chat.service import ChatService
    from backend.core.directives import load_directives
    from backend.providers.simulated import SimulatedProvider

    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=2048, num_predict=512)
        provider = SimulatedProvider()
        provider.capabilities = {"vision": False, "experimental": True}
        service = ChatService(db, provider, load_directives(settings.directives.path), memory_manager=manager)
        over_ctx = type(
            "Ctx",
            (),
            {"dropped_blocks": {"presentation"}, "budget": manager.context.assess(system_base_text="x" * 9000, num_predict_override=512)},
        )()
        assert over_ctx.budget.over_hard_limit is True
        flags = service._budget_flags(over_ctx, "pt-BR")
        assert flags["over_budget"] is True
        assert "orçamento" in flags["notice"]
        assert flags["dropped_blocks"] == ["presentation"]
        ok_ctx = type("Ctx2", (), {"dropped_blocks": set(), "budget": manager.context.assess(system_base_text="ok")})()
        assert service._budget_flags(ok_ctx, "en-US") == {}
    finally:
        db.close()


# --------------------------------------------------------------- multimodal bloqueado


def test_text_only_model_rejects_attachments(tmp_path):
    from backend.chat.service import ChatService
    from backend.core.directives import load_directives
    from backend.providers.simulated import SimulatedProvider

    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=2048, num_predict=512)
        provider = SimulatedProvider()
        provider.capabilities = {"vision": False, "experimental": True}
        service = ChatService(db, provider, load_directives(settings.directives.path), memory_manager=manager)
        with pytest.raises(AttachmentRejected) as exc:
            asyncio.run(service.send(None, "Analise a imagem", attachment_ids=["id-falso"], language="pt-BR"))
        assert exc.value.status == 415
        assert "somente texto" in str(exc.value)
    finally:
        db.close()


def test_text_only_model_never_emits_images_in_payload(tmp_path):
    from backend.chat.service import ChatService
    from backend.core.directives import load_directives
    from backend.providers.simulated import SimulatedProvider

    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=4096, num_predict=1024)
        provider = SimulatedProvider()
        provider.capabilities = {"vision": False, "experimental": True}
        service = ChatService(db, provider, load_directives(settings.directives.path), memory_manager=manager)
        history = [
            Message(
                id=1,
                conversation_id="c1",
                seq=1,
                role="user",
                content="olha a imagem",
                created_at="2026-11-18T00:00:00Z",
                meta={"attachments": [{"id": "att-1", "name": "print.png"}]},
            ),
            Message(id=2, conversation_id="c1", seq=2, role="user", content="e agora?", created_at="2026-11-18T00:01:00Z"),
        ]
        payload = service.build_payload(history, active_image_msg_ids={1})
        assert all("images" not in entry for entry in payload)
        # referência textual do anexo do histórico é preservada (nada é apagado)
        assert any("att-1" in entry["content"] for entry in payload if entry["role"] == "user")
    finally:
        db.close()


def test_vision_capable_model_keeps_selecting_images(tmp_path):
    from backend.chat.service import ChatService
    from backend.core.directives import load_directives
    from backend.providers.simulated import SimulatedProvider

    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=4096, num_predict=1024)
        provider = SimulatedProvider()  # capabilities padrão: vision True
        service = ChatService(db, provider, load_directives(settings.directives.path), memory_manager=manager)
        history = [
            Message(
                id=1,
                conversation_id="c1",
                seq=1,
                role="user",
                content="olha a imagem",
                created_at="2026-11-18T00:00:00Z",
                meta={"attachments": [{"id": "att-1", "name": "print.png"}]},
            ),
        ]
        selected = service._select_active_image_msg_ids(history)
        assert selected == {1}
    finally:
        db.close()

"""Capacidade de tool calling por modelo (Qwen3.8 sem tools nativas / Qwen3.5 com tools).

Causa investigada: Ollama responde HTTP 400 ("does not support tools") quando o payload
leva `tools` para um modelo sem tool calling nativo. Cobertura offline (mocks/simulado,
sem inferência real): detecção via /api/show, fallback conservador, payload efetivamente
enviado pelo provider, Deep Research independente de tool calling e preservação do Qwen3.5.
"""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend.chat.service import ChatService, _ResearchOutcome
from backend.core.config import PROJECT_ROOT, ToolsSettings, load_settings
from backend.core.directives import load_directives
from backend.main import create_app
from backend.memory.manager import MemoryManager
from backend.providers.base import ChatTurn, ModelProvider
from backend.providers.ollama import OllamaProvider
from backend.providers.registry import create_provider
from backend.providers.simulated import SimulatedProvider
from backend.storage.database import Database
from backend.tools.base import Tool, ToolResult, ToolSpec
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry
from tests.helpers import make_settings

QWEN38 = "hf.co/cognitivers/Qwen3.8-27B-OBLITERATED-12GB-GGUF:IQ2_M"
QWEN35 = "hf.co/bartowski/Qwen_Qwen3.5-9B-GGUF:Q6_K"
RANA = "abliterant/qwen3.5-rana:9b-q6_K"

DIRECTIVES = """---
name: Enslaved
language: en-US
version: 1
---

# Goal
Reply directly and execute tasks permitted by the owner.
"""


# ------------------------------------------------------------------ providers


def _ollama(handler, tools=None) -> OllamaProvider:
    provider = OllamaProvider(
        model="teste:1b", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    if tools is not None:
        provider.capabilities["tools"] = tools
    return provider


def test_ollama_reads_tools_capability_from_show_metadata():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        assert json.loads(request.content) == {"model": "teste:1b"}
        return httpx.Response(200, json={"capabilities": ["tools", "thinking", "completion"]})

    provider = _ollama(handler)
    assert asyncio.run(provider.supports_native_tools()) is True
    assert calls == ["/api/show"]


def test_ollama_model_without_tools_is_detected_and_cached():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"capabilities": ["thinking", "completion"]})

    provider = _ollama(handler)
    assert asyncio.run(provider.supports_native_tools()) is False
    assert asyncio.run(provider.supports_native_tools()) is False
    assert calls["n"] == 1  # segunda chamada usa o cache


@pytest.mark.parametrize("handler", [
    lambda request: httpx.Response(500, json={"error": "boom"}),
    lambda request: (_ for _ in ()).throw(httpx.ConnectError("conexão recusada")),
    lambda request: httpx.Response(200, json={"capabilities": "não-lista"}),
])
def test_ollama_without_metadata_falls_back_conservative(handler):
    # Sem metadados (Ollama fora, modelo ausente, resposta inválida): sem tools — evita HTTP 400.
    assert asyncio.run(_ollama(handler).supports_native_tools()) is False


def test_explicit_profile_setting_wins_and_skips_metadata():
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("/api/show não deveria ser consultado com perfil explícito")

    assert asyncio.run(_ollama(handler, tools=False).supports_native_tools()) is False
    assert asyncio.run(_ollama(handler, tools=True).supports_native_tools()) is True


def test_provider_without_detection_defaults_to_tools_supported():
    # Providers sem fonte de metadados (ex.: simulado) mantêm o comportamento atual.
    assert asyncio.run(SimulatedProvider().supports_native_tools()) is True


def test_stream_turn_payload_includes_tools_only_when_provided():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "ok"}, "done": True, "done_reason": "stop"})

    provider = _ollama(handler)

    async def run(tools):
        items = []
        async for item in provider.stream_turn([{"role": "user", "content": "oi"}], tools=tools):
            items.append(item)
        return items

    asyncio.run(run(None))
    assert "tools" not in captured["body"]  # payload sem tools: nenhum HTTP 400 possível
    tools = [{"type": "function", "function": {"name": "fs.read", "parameters": {}}}]
    asyncio.run(run(tools))
    assert captured["body"]["tools"] == tools


# --------------------------------------------------------------- chat service


class EchoArgs(BaseModel):
    text: str


class EchoTool(Tool):
    args_model = EchoArgs

    def __init__(self, tool_id: str = "echo", capability: str = "demo.execute"):
        self.spec = ToolSpec(
            id=tool_id,
            name="Echo",
            description="Repete o texto informado (teste).",
            required_permissions=[capability],
        )

    async def run(self, args, permission=None) -> ToolResult:
        if permission is None or not permission.granted:
            return ToolResult(ok=False, error="permissão ausente")
        return ToolResult(ok=True, output=args.get("text", ""))


class RecordingProvider(ModelProvider):
    """Registra se o fluxo chegou como stream puro ou como turno com `tools`."""

    name = "recording"

    def __init__(self, supports_tools: bool | None = None):
        super().__init__("rec-1b")
        if supports_tools is not None:
            self.capabilities["tools"] = supports_tools
        self.stream_calls: list[dict] = []
        self.turn_calls: list[dict] = []

    async def chat(self, messages, **options):
        self.stream_calls.append({"messages": [dict(m) for m in messages]})
        return "Resposta completa."

    async def stream_chat(self, messages, **options):
        self.stream_calls.append({"messages": [dict(m) for m in messages]})
        text = "Resposta textual direta."
        for start in range(0, len(text), 6):
            yield text[start : start + 6]

    async def stream_turn(self, messages, tools=None, **options):
        self.turn_calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        text = "Resposta com ferramentas."
        for start in range(0, len(text), 6):
            yield text[start : start + 6]
        yield ChatTurn(content=text)

    async def health(self):
        return {"available": True}


def make_service(tmp_path, provider, extra_tool_ids=()):
    directives_path = tmp_path / "directives.md"
    directives_path.write_text(DIRECTIVES, encoding="utf-8")
    db = Database(tmp_path / "tools-cap.db")
    db.connect()
    registry = ToolRegistry()
    registry.register(EchoTool("echo", "demo.execute"))
    for tool_id in extra_tool_ids:
        registry.register(EchoTool(tool_id, "demo.execute"))
    tools = ToolsSettings(enabled=True, permission_modes={"demo.execute": "allow"})
    engine = ToolEngine(
        registry, ToolPolicy(tools), PermissionManager(ToolPolicy(tools)),
        AuditLog(tmp_path / "audit.jsonl"), tools,
    )
    settings = make_settings(tmp_path)
    manager = MemoryManager(db, settings.memory, num_ctx=8192, num_predict=1024)
    service = ChatService(db, provider, load_directives(directives_path), tool_engine=engine, memory_manager=manager)
    return service, db


def _types(events):
    return [event["type"] for event in events]


def test_model_without_native_tools_streams_plain_even_with_tools_on(tmp_path):
    provider = RecordingProvider(supports_tools=False)
    service, db = make_service(tmp_path, provider)
    try:
        events = asyncio.run(_drain(service.send_stream(None, "olá", use_tools=True)))
        assert "error" not in _types(events)
        assert "done" in _types(events)
        assert provider.turn_calls == []  # fluxo de ferramentas nunca selecionado
        assert len(provider.stream_calls) == 1  # chat/streaming normal
        payload = provider.stream_calls[0]["messages"]
        # nenhum anúncio de ferramentas que o modelo não pode chamar
        assert all("# Tools" not in str(m.get("content")) for m in payload)
    finally:
        db.close()


async def _drain(stream):
    return [event async for event in stream]


def test_model_with_native_tools_keeps_tool_flow(tmp_path):
    provider = RecordingProvider(supports_tools=True)
    service, db = make_service(tmp_path, provider)
    try:
        events = asyncio.run(_drain(service.send_stream(None, "olá", use_tools=True)))
        assert "error" not in _types(events)
        assert "done" in _types(events)
        assert provider.stream_calls == []  # sem stream puro
        assert len(provider.turn_calls) == 1
        tools = provider.turn_calls[0]["tools"]
        assert tools and {tool["function"]["name"] for tool in tools} >= {"echo"}
        system = provider.turn_calls[0]["messages"][0]["content"]
        assert "# Tools" in system  # ferramentas anunciadas só a quem pode chamá-las
    finally:
        db.close()


def test_tools_toggle_off_skips_tools_for_capable_model(tmp_path):
    provider = RecordingProvider(supports_tools=True)
    service, db = make_service(tmp_path, provider)
    try:
        events = asyncio.run(_drain(service.send_stream(None, "olá", use_tools=False)))
        assert "error" not in _types(events)
        assert provider.turn_calls == []
        assert len(provider.stream_calls) == 1
    finally:
        db.close()


def test_research_phase_runs_with_frontend_state_use_tools_false(tmp_path, monkeypatch):
    # Estado real do frontend no Qwen3.8 (checkbox oculto -> use_tools=false): a investigação
    # explícita não pode depender do checkbox; quem filtra mensagens comuns é o classificador.
    provider = RecordingProvider(supports_tools=False)
    service, db = make_service(tmp_path, provider, extra_tool_ids=("net.inspect",))
    try:
        entered: list[str] = []

        async def fake_phase(self, conversation, content, mode, turn, holder, *, language=None, decision_retry=False):
            entered.append(content)
            yield _ResearchOutcome()

        monkeypatch.setattr(ChatService, "_research_phase", fake_phase)
        events = asyncio.run(_drain(service.send_stream(None, "olá", use_tools=False, research_mode="deep")))
        assert entered == ["olá"]  # fase de investigação alcançada com use_tools=false
        assert "error" not in _types(events)
        assert "done" in _types(events)
        assert provider.turn_calls == []  # relatório redigido em fluxo textual puro
        assert len(provider.stream_calls) == 1
    finally:
        db.close()


def test_qwen38_chat_normal_with_frontend_state_use_tools_false(tmp_path):
    provider = RecordingProvider(supports_tools=False)
    service, db = make_service(tmp_path, provider)
    try:
        events = asyncio.run(_drain(service.send_stream(None, "olá", use_tools=False)))
        assert "error" not in _types(events)
        assert "done" in _types(events)
        assert provider.turn_calls == []  # nunca envia tools ao provider
        assert len(provider.stream_calls) == 1
        assert all("# Tools" not in str(m.get("content")) for m in provider.stream_calls[0]["messages"])
    finally:
        db.close()


def test_history_and_language_preserved_without_tools(tmp_path):
    provider = RecordingProvider(supports_tools=False)
    service, db = make_service(tmp_path, provider)
    try:
        first = asyncio.run(_drain(service.send_stream(None, "olá", use_tools=True, language="en-US")))
        conversation_id = first[-1]["conversation_id"]
        second = asyncio.run(_drain(service.send_stream(conversation_id, "e agora?", use_tools=True, language="en-US")))
        assert "error" not in _types(second)
        payload = provider.stream_calls[-1]["messages"]
        assert [m["role"] for m in payload].count("user") == 2  # histórico preservado
        assert "# Goal" in payload[0]["content"]  # diretrizes (en-US) seguem no sistema
    finally:
        db.close()


# ------------------------------------------------------------------- API/perfis


def test_registry_wires_profile_supports_tools(tmp_path):
    settings = make_settings(tmp_path, provider={
        "name": "simulated",
        "model": "m1",
        "models": {"m1": {"supports_tools": False}},
    })
    assert create_provider(settings).capabilities["tools"] is False
    settings.provider.models["m1"].supports_tools = True
    assert create_provider(settings).capabilities["tools"] is True
    settings.provider.models["m1"].supports_tools = None  # auto-detecção
    assert create_provider(settings).capabilities["tools"] is None


def test_qwen38_profile_stream_endpoint_never_sends_tools(tmp_path):
    settings = make_settings(tmp_path, provider={
        "name": "simulated",
        "model": QWEN38,
        "models": {
            QWEN38: {"num_ctx": 4096, "num_predict": 512, "supports_vision": False, "supports_tools": False, "experimental": True},
            QWEN35: {"num_ctx": 16384, "num_predict": 1024, "supports_vision": True, "supports_tools": True},
        },
    }, tools={"enabled": True})
    with TestClient(create_app(settings)) as client:
        config = client.get("/api/config").json()
        assert config["provider"]["supports_tools"] is False
        with client.stream("POST", "/api/chat/stream", json={"message": "olá", "use_tools": True}) as response:
            assert response.status_code == 200
            events = [
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
        assert "error" not in _types(events)  # SimulatedProvider recusaria tools (chat_turn)
        assert _types(events)[-1] == "done"

        # troca de modelo (Settings) continua disponível e persiste a escolha
        switch = client.post("/api/model", json={"model": QWEN35})
        assert switch.status_code == 200
        assert switch.json()["restart_required"] is True


def test_research_endpoint_with_frontend_state_use_tools_false(tmp_path):
    """Fluxo real do Qwen3.8: checkbox oculto -> use_tools=false; investigação explícita
    chega ao cartão de permissão, é NEGADA pelo proprietário e não executa rede nenhuma."""
    settings = make_settings(tmp_path, provider={
        "name": "simulated",
        "model": QWEN38,
        "models": {QWEN38: {"num_ctx": 4096, "num_predict": 512, "supports_vision": False, "supports_tools": False, "experimental": True}},
    }, tools={"enabled": True})
    with TestClient(create_app(settings)) as client:
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "Faça uma investigação profunda da segurança do site https://loja.example",
                  "use_tools": False, "research_mode": "deep", "language": "pt-BR"},
        ) as response:
            assert response.status_code == 200
            events = [
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
        card = next(event for event in events if event["type"] == "permission_request")
        assert card.get("research") is True  # cartão de investigação, não de ferramentas nativas
        conversation_id = card["conversation_id"]
        assert events[-1]["type"] == "done"
        assert events[-1]["status"] == "awaiting_permission"

        # decisão do proprietário: NEGAR (mesma chamada que o frontend faz no card)
        decision = client.post(
            f"/api/permissions/{card['request_id']}/decision",
            json={"granted": False, "conversation_id": conversation_id},
        )
        assert decision.status_code == 200
        assert decision.json()["granted"] is False

        # retomada (retry) exatamente como o frontend faz após o cartão (resumeConversation(id, true))
        with client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "", "conversation_id": conversation_id, "retry": True,
                  "use_tools": False, "research_mode": "deep", "language": "pt-BR"},
        ) as response:
            assert response.status_code == 200
            events = [
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
        assert "error" not in [event["type"] for event in events]
        final = events[-1]
        assert final["type"] == "done"
        # resposta determinística da política de evidência: nada foi executado sem autorização
        assert final["model"] == "politica-de-evidencia"


def test_shipped_settings_declare_model_tool_capability():
    settings = load_settings(config_path=PROJECT_ROOT / "config" / "settings.yaml", root=PROJECT_ROOT)
    # RANA (ativo): com tool calling nativo (verificado em /api/show: completion, vision, tools, thinking)
    assert settings.provider.models[RANA].supports_tools is True
    # Qwen3.5: com tool calling nativo (verificado em /api/show: tools, thinking, completion, vision)
    assert settings.provider.models[QWEN35].supports_tools is True

import asyncio
import json

import httpx
import pytest

from backend.core.config import ConfigError, Settings
from backend.providers.base import ProviderError, ProviderUnavailableError
from backend.providers.ollama import OllamaProvider
from backend.providers.registry import create_provider
from backend.providers.simulated import SimulatedProvider


def run(coro):
    return asyncio.run(coro)


def test_registry_selects_ollama():
    settings = Settings.model_validate({"provider": {"name": "ollama", "model": "qwen3.5:4b"}})
    provider = create_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "qwen3.5:4b"
    assert provider.default_options["num_ctx"] == 16384
    assert provider.default_options["num_predict"] == 2048
    assert provider.think is False


def test_registry_ollama_think_configurable():
    settings = Settings.model_validate(
        {"provider": {"name": "ollama"}, "ollama": {"think": True}}
    )
    assert create_provider(settings).think is True


def test_registry_selects_simulated():
    settings = Settings.model_validate({"provider": {"name": "simulated"}})
    assert isinstance(create_provider(settings), SimulatedProvider)


def test_registry_rejects_unknown_provider():
    settings = Settings.model_validate({"provider": {"name": "fantasma"}})
    with pytest.raises(ConfigError):
        create_provider(settings)


def test_simulated_provider_chat_and_stream():
    provider = SimulatedProvider(model="simulado-1b")
    messages = [
        {"role": "system", "content": "diretrizes"},
        {"role": "user", "content": "oi"},
    ]
    reply = run(provider.chat(messages))
    assert reply.startswith("[simulado]")
    assert "oi" in reply
    assert provider.calls[0][0]["role"] == "system"

    async def collect():
        return [chunk async for chunk in provider.stream_chat(messages)]

    assert "".join(run(collect())) == reply


def _ollama_with_handler(handler) -> OllamaProvider:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    return OllamaProvider(model="qwen3.5:4b", base_url="http://ollama.test", client=client)


def test_ollama_chat_success():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3.5:4b"
        assert payload["stream"] is False
        assert payload["think"] is False
        assert payload["options"]["num_ctx"] == 4096
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "olá!"}, "done": True})

    provider = _ollama_with_handler(handler)
    try:
        reply = run(provider.chat([{"role": "user", "content": "oi"}], num_ctx=4096))
    finally:
        run(provider.aclose())
    assert reply == "olá!"


def test_ollama_stream_success():
    def handler(request: httpx.Request) -> httpx.Response:
        lines = [
            json.dumps({"message": {"content": "olá"}, "done": False}),
            json.dumps({"message": {"content": "!"}, "done": True}),
        ]
        return httpx.Response(200, content="\n".join(lines) + "\n")

    async def collect():
        provider = _ollama_with_handler(handler)
        try:
            return [chunk async for chunk in provider.stream_chat([{"role": "user", "content": "oi"}])]
        finally:
            await provider.aclose()

    assert "".join(run(collect())) == "olá!"


def test_ollama_unavailable_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conexão recusada", request=request)

    provider = _ollama_with_handler(handler)
    try:
        with pytest.raises(ProviderUnavailableError, match="indisponível"):
            run(provider.chat([{"role": "user", "content": "oi"}]))
    finally:
        run(provider.aclose())


def test_ollama_model_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen3.5:4b' not found"})

    provider = _ollama_with_handler(handler)
    try:
        with pytest.raises(ProviderError, match="não encontrado"):
            run(provider.chat([{"role": "user", "content": "oi"}]))
    finally:
        run(provider.aclose())


def test_ollama_health_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conexão recusada", request=request)

    provider = _ollama_with_handler(handler)
    try:
        status = run(provider.health())
    finally:
        run(provider.aclose())
    assert status["available"] is False


def test_ollama_health_available():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "qwen3.5:4b"}]})

    provider = _ollama_with_handler(handler)
    try:
        status = run(provider.health())
    finally:
        run(provider.aclose())
    assert status["available"] is True
    assert "qwen3.5:4b" in status["models"]


def test_ollama_loaded_models_reports_vram():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/ps"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3.5:4b",
                        "size": 3300000000,
                        "size_vram": 4400000000,
                        "expires_at": "2026-11-16T18:00:00Z",
                    }
                ]
            },
        )

    provider = _ollama_with_handler(handler)
    try:
        loaded = run(provider.loaded_models())
    finally:
        run(provider.aclose())
    assert loaded == [
        {
            "name": "qwen3.5:4b",
            "size_mb": 3147,
            "vram_mb": 4196,
            "expires_at": "2026-11-16T18:00:00Z",
        }
    ]


def test_ollama_loaded_models_unavailable_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conexão recusada", request=request)

    provider = _ollama_with_handler(handler)
    try:
        assert run(provider.loaded_models()) == []
    finally:
        run(provider.aclose())


def test_simulated_provider_loaded_models_defaults_empty():
    assert run(SimulatedProvider().loaded_models()) == []


def test_ollama_stream_turn_streams_thinking_content_and_tool_calls():
    from backend.providers.base import ChatTurn, Thinking

    lines = [
        {"message": {"thinking": "hmm"}},
        {"message": {"content": "Ol"}},
        {"message": {"content": "á", "tool_calls": [{"function": {"name": "fs.read", "arguments": {"path": "a"}}}]}},
        {"message": {"content": ""}, "done": True, "prompt_eval_count": 3, "eval_count": 2},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True and payload["think"] is True
        assert payload["options"]["num_predict"] == 100 + 2048  # folga para o raciocínio
        assert "think" not in payload["options"]
        return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())

    provider = _ollama_with_handler(handler)

    async def collect():
        return [i async for i in provider.stream_turn([{"role": "user", "content": "oi"}], tools=[{}], think=True, num_predict=100)]

    items = run(collect())
    assert [type(i) for i in items] == [Thinking, str, str, ChatTurn]
    turn = items[-1]
    assert turn.content == "Olá" and turn.tool_calls[0].name == "fs.read"


def test_registry_and_ollama_support_hf_qwen35_9b_q6k():
    model_id = "hf.co/bartowski/Qwen_Qwen3.5-9B-GGUF:Q6_K"
    settings = Settings.model_validate({"provider": {"name": "ollama", "model": model_id}})
    provider = create_provider(settings)
    assert isinstance(provider, OllamaProvider)
    assert provider.model == model_id
    assert provider.default_options["num_ctx"] == 16384

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": model_id, "size": 7830765568, "size_vram": 7830765568}]})
        payload = json.loads(request.content)
        assert payload["model"] == model_id
        return httpx.Response(
            200,
            json={
                "message": {
                    "content": "",
                    "tool_calls": [{"function": {"name": "text.analyze", "arguments": {"text": "abc"}}}],
                },
                "done": True,
                "prompt_eval_count": 20,
                "eval_count": 10,
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False)
    mocked = OllamaProvider(model=model_id, base_url="http://ollama.test", client=client)
    try:
        turn = run(mocked.chat_turn([{"role": "user", "content": "analise abc"}], tools=[{"type": "function"}]))
        loaded = run(mocked.loaded_models())
    finally:
        run(mocked.aclose())
    assert turn.tool_calls[0].name == "text.analyze"
    assert json.loads(turn.tool_calls[0].arguments) == {"text": "abc"}
    assert loaded[0]["name"] == model_id and loaded[0]["vram_mb"] == loaded[0]["size_mb"]

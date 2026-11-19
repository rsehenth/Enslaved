import asyncio
import json

import httpx
import pytest

from backend.providers.base import ProviderError, ProviderUnavailableError
from backend.providers.ollama import OllamaProvider
from backend.providers.simulated import SimulatedProvider

BASE = "http://127.0.0.1:11434"


def make_provider(handler) -> OllamaProvider:
    return OllamaProvider(model="teste:4b", base_url=BASE, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_chat_turn_without_tools_returns_content():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is False
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "resposta direta"}})

    provider = make_provider(handler)
    turn = asyncio.run(provider.chat_turn([{"role": "user", "content": "oi"}]))
    assert turn.content == "resposta direta"
    assert turn.tool_calls == []


def test_chat_turn_sends_tools_and_parses_tool_calls():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"function": {"name": "fs.read", "arguments": {"path": "docs/a.md"}}},
                        {"function": {"name": "net.fetch", "arguments": "{\"url\": \"https://exemplo.org\"}"}},
                    ],
                }
            },
        )

    provider = make_provider(handler)
    tools = [{"type": "function", "function": {"name": "fs.read", "parameters": {}}}]
    turn = asyncio.run(provider.chat_turn([{"role": "user", "content": "leia"}], tools=tools))
    assert captured["body"]["tools"] == tools
    assert turn.content == ""
    assert [call.name for call in turn.tool_calls] == ["fs.read", "net.fetch"]
    first, second = turn.tool_calls
    assert json.loads(first.arguments) == {"path": "docs/a.md"}
    assert second.arguments == "{\"url\": \"https://exemplo.org\"}"


def test_chat_turn_without_tool_calls_key_yields_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "texto"}})

    provider = make_provider(handler)
    turn = asyncio.run(provider.chat_turn([{"role": "user", "content": "x"}], tools=[]))
    # tools=[] (falsy) não é enviado no payload
    assert turn.tool_calls == []


def test_chat_turn_transport_error_maps_to_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("conexão recusada")

    provider = make_provider(handler)
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(provider.chat_turn([{"role": "user", "content": "x"}]))


def test_chat_turn_http_404_maps_to_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    provider = make_provider(handler)
    with pytest.raises(ProviderError):
        asyncio.run(provider.chat_turn([{"role": "user", "content": "x"}]))


def test_base_provider_refuses_tools():
    provider = SimulatedProvider()
    with pytest.raises(ProviderError):
        asyncio.run(provider.chat_turn([{"role": "user", "content": "x"}], tools=[{"type": "function"}]))


def test_base_provider_chat_turn_wraps_chat():
    provider = SimulatedProvider(reply="ok simulado")
    turn = asyncio.run(provider.chat_turn([{"role": "user", "content": "x"}]))
    assert turn.content == "ok simulado"
    assert turn.tool_calls == []


def test_turns_carry_their_own_usage_and_finish_reason():
    """Uso/término vão no próprio ChatTurn (last_usage do provider é compartilhado entre requisições)."""
    done = {"done": True, "done_reason": "length", "prompt_eval_count": 12, "eval_count": 34}

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["stream"]:
            lines = [{"message": {"content": "a"}}, {"message": {"content": "b"}, **done}]
            return httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
        return httpx.Response(200, json={"message": {"content": "ab"}, **done})

    provider = make_provider(handler)
    expected = {"prompt_eval_count": 12, "eval_count": 34}
    turn = asyncio.run(provider.chat_turn([{"role": "user", "content": "oi"}]))
    assert (turn.usage, turn.finish_reason) == (expected, "length")

    async def collect():
        return [item async for item in provider.stream_turn([{"role": "user", "content": "oi"}])]

    final = asyncio.run(collect())[-1]
    assert (final.content, final.usage, final.finish_reason) == ("ab", expected, "length")
    assert provider.last_usage == expected  # chat da UI continua lendo o atributo

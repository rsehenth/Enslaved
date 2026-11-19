"""Orçamento de contexto da API /v1: estimador por pré-tokens, limite exato do Ollama e bind local.

Calibração (homologação 2026-11-18, qwen3.5-rana 9B Q6_K, prompt_eval_count real do Ollama 0.40.16):
payload inicial real do OpenCode 2.2.5 = 11.950 tokens (35,3K chars de system + 12 tools, 13,9K chars);
o estimador antigo (3,0 chars/token) dava 16.894 e rejeitava. Prosa/código/schemas reais: ~4,0 chars/token;
dígitos: 1 token cada; hex 1,13; base64 1,35; emoji ~2,6 tokens por emoji.
"""

import asyncio
import base64
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.api.routes_v1 import convert_messages, normalize_tools
from backend.main import create_app
from backend.memory.context_manager import ContextManager, pretoken_estimate
from backend.providers.base import ProviderError, ToolCall
from backend.providers.ollama import OllamaProvider
from tests.helpers import make_settings
from tests.test_localapi import ToolsStubProvider, auth_headers, chat_body, create_key, enable_api, parse_sse

OLLAMA_OVERFLOW = (
    'Ollama respondeu HTTP 400: {"error":"request (17020 tokens) exceeds the available context size '
    '(16384 tokens), try increasing it"}'
)

_SENTENCES = [
    "You are an AI agent running in a coding harness; help the user accomplish their goals using the tools available.",
    "Prefer reading the relevant files before editing them, and keep every change focused on the requested task.",
    "When a command fails, inspect the error output carefully and fix the underlying cause instead of retrying blindly.",
    "Use the shell tool for builds and tests, the read tool for files, and the edit tool for precise replacements.",
    "Never invent file paths or APIs; search the repository with glob and grep when you are unsure where code lives.",
    "Summarize what you changed at the end, including tests that were run and anything that still needs attention.",
]


def system_prompt(chars: int) -> str:
    text, i = "", 0
    while len(text) < chars:
        text += f"{_SENTENCES[i % len(_SENTENCES)]}\n" if i % 7 else f"\n## Section {i // 7}\n\n"
        i += 1
    return text[:chars]


def opencode_tools() -> list[dict]:
    names = ["edit", "glob", "grep", "question", "read", "shell", "skill", "subagent", "webfetch", "websearch", "write", "execute"]
    return normalize_tools([
        {
            "type": "function",
            "function": {
                "name": name,
                "description": " ".join(_SENTENCES[: 4 + n % 3]) + f" The {name} tool returns plain text output.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filePath": {"type": "string", "description": "Absolute path to the file to operate on."},
                        "pattern": {"type": "string", "description": "Glob or regex pattern used to filter results."},
                        "timeout": {"type": "number", "description": "Optional timeout in milliseconds (max 600000)."},
                    },
                    "required": ["filePath"],
                    "additionalProperties": False,
                },
            },
        }
        for n, name in enumerate(names)
    ])


def opencode_messages(system_chars: int = 35_000) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt(system_chars)},
        {"role": "user", "content": "Crie um projeto Python com uma calculadora e testes com pytest."},
    ]


def tool_step(i: int, name: str, args: dict, result: str) -> list[dict]:
    call_id = f"call_{i}"
    return [
        {"role": "assistant", "content": "", "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": result},
    ]


PY_CODE = "\n".join(
    f"{n}: def soma_{n}(a: int, b: int) -> int:\n{n}:     \"\"\"Soma dois inteiros.\"\"\"\n{n}:     return a + b\n" for n in range(1, 60)
)


def v1_context(**overrides) -> ContextManager:
    params = {"num_ctx": 16384, "output_reserve": 2048, "safety_margin_ratio": 0.06, "pretoken": True}
    params.update(overrides)
    return ContextManager(**params)


# ------------------------------------------------------------------ 1. payload realista que cabe


def test_realistic_opencode_payload_fits_where_old_estimator_rejected():
    messages, tools = convert_messages(opencode_messages()), opencode_tools()
    new = v1_context().validate_payload(messages, tools=tools, num_predict=2048)
    old = ContextManager(num_ctx=16384, output_reserve=2048).validate_payload(messages, tools=tools, num_predict=2048)
    assert new.fits, new.reason
    assert not old.fits  # falso positivo da homologação: 3,0 chars/token superconta prosa/schemas em ~40%
    chars = sum(len(m["content"]) for m in messages) + len(json.dumps(tools, ensure_ascii=False))
    # real: 4,1 chars/token no payload do OpenCode; prosa sintética de palavras comuns rende um pouco mais
    assert 3.6 <= chars / (new.messages_tokens + new.tools_tokens) <= 5.0


# ------------------------------------------------------------------ 2. payload genuinamente grande


def test_payload_above_window_is_rejected_with_diagnostic():
    v = v1_context().validate_payload(convert_messages(opencode_messages(system_chars=80_000)), tools=opencode_tools(), num_predict=2048)
    assert not v.fits
    assert v.total_input_tokens > v.input_budget == 14336


# ------------------------------------------------------------------ 3. histórico crescente de ferramentas


def test_growing_tool_history_grows_monotonically_until_rejected():
    ctx, tools = v1_context(), opencode_tools()
    raw = opencode_messages(system_chars=20_000)
    totals, fits = [], []
    for i in range(1, 30):
        raw += tool_step(i, "read", {"filePath": f"/proj/calc_{i}.py"}, PY_CODE)
        v = ctx.validate_payload(convert_messages(raw), tools=tools, num_predict=2048)
        totals.append(v.total_input_tokens)
        fits.append(v.fits)
    assert totals == sorted(totals) and len(set(totals)) == len(totals)
    assert fits[0] and not fits[-1]
    assert fits == sorted(fits, reverse=True)  # uma vez fora, não volta a caber


# ------------------------------------------------------------------ 4. código, JSON e Unicode


def test_dense_content_is_not_undercounted():
    digits = "\n".join("12345,67890,24680,13579" for _ in range(200))
    assert pretoken_estimate(digits) >= sum(c.isdigit() for c in digits)  # Qwen: 1 token por dígito
    hexes = "\n".join(f"{n:064x}" for n in range(1, 120))
    assert pretoken_estimate(hexes) >= len(hexes) / 1.2  # real: 1,13 chars/token
    b64 = base64.b64encode(bytes(range(256)) * 24).decode()
    assert pretoken_estimate(b64) >= len(b64) / 1.4  # real: 1,35 chars/token
    emoji = "🚀🔥🧪✅❌🎉👍💡📦🐍" * 50
    assert pretoken_estimate(emoji) >= 2.4 * len(emoji)  # real: ~2,6 tokens por emoji
    accents = "àáâãçéêíóôõúü ∑∫√≈≠≤≥→←↑↓ “aspas” — … " * 40
    assert pretoken_estimate(accents) >= len(accents) / 1.2  # real: 1,16 chars/token
    # e o estimador antigo subcontava todos eles (o guard deixava passar payloads maiores que a janela)
    old = ContextManager()
    assert old.estimate_dense_text_tokens(digits) < pretoken_estimate(digits) / 2
    assert old.estimate_dense_text_tokens(emoji) < pretoken_estimate(emoji) / 3


def test_prose_code_and_json_are_not_overcounted():
    prose = system_prompt(20_000)
    code = PY_CODE
    schema = json.dumps(opencode_tools(), ensure_ascii=False)
    for text, lo, hi in ((prose, 3.6, 5.0), (code, 2.0, 3.6), (schema, 3.6, 5.0)):
        ratio = len(text) / pretoken_estimate(text)
        assert lo <= ratio <= hi, (ratio, text[:40])


# ------------------------------------------------------------------ 5. reservas de saída e templates


def test_output_reserve_and_template_overheads():
    ctx = v1_context()
    messages = convert_messages(opencode_messages(system_chars=40_000))
    full = ctx.validate_payload(messages, tools=opencode_tools(), num_predict=2048)
    small = ctx.validate_payload(messages, tools=opencode_tools(), num_predict=512)
    assert full.input_budget == 14336 and small.input_budget == 16384 - 512
    assert small.total_input_tokens == full.total_input_tokens  # a reserva não muda a entrada
    tools_tokens = ctx.estimate_tools_schema_tokens(opencode_tools())
    assert tools_tokens == 120 + pretoken_estimate(json.dumps(opencode_tools(), ensure_ascii=False))
    assert ctx.estimate_message_tokens({"role": "user", "content": "ok"}) == 6 + 1  # delimitadores do template
    assert full.safety_margin_tokens == pytest.approx(0.06 * (full.messages_tokens + full.tools_tokens), abs=1)


# ------------------------------------------------------------------ 6/7/8. sub/supercontagem, context_overflow, stream e tools


class OverflowProvider(ToolsStubProvider):
    """Simula o Ollama (truncate=False) recusando um prompt que o estimador subcontou."""

    async def chat_turn(self, messages, tools=None, **options):
        self._record(messages, tools, options)
        raise ProviderError(OLLAMA_OVERFLOW)

    async def stream_turn(self, messages, tools=None, **options):
        self._record(messages, tools, options)
        raise ProviderError(OLLAMA_OVERFLOW)
        yield  # pragma: no cover


def test_estimated_overflow_keeps_context_overflow_code_and_diagnostic(client):
    enable_api(client)
    key = create_key(client)
    stub = ToolsStubProvider()
    client.app.state.provider = stub
    body = chat_body(messages=opencode_messages(system_chars=80_000), tools=opencode_tools())
    response = client.post("/v1/chat/completions", headers=auth_headers(key), json=body)
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_overflow"
    message = error["message"]
    for part in ("mensagens", "ferramentas", "margem de segurança", "reserva de saída 2048", "orçamento de entrada de 14336"):
        assert part in message
    assert "maximum context length is 16384 tokens" in message  # reconhecido pelo OpenCode -> compactação
    assert stub.calls == []  # nada chega ao modelo


def test_overcounted_payload_now_reaches_model_with_hard_limit_options(client):
    enable_api(client)
    key = create_key(client)
    stub = ToolsStubProvider()
    client.app.state.provider = stub
    body = chat_body(messages=opencode_messages(), tools=opencode_tools())
    response = client.post("/v1/chat/completions", headers=auth_headers(key), json=body)
    assert response.status_code == 200, response.text
    options = stub.calls[0]["options"]
    assert options["truncate"] is False and options["shift"] is False
    assert options["num_ctx"] == 16384 and options["num_predict"] == 2048


def test_undercounted_payload_is_rejected_by_ollama_exact_count_non_stream(client):
    enable_api(client)
    key = create_key(client)
    client.app.state.provider = OverflowProvider()
    response = client.post("/v1/chat/completions", headers=auth_headers(key), json=chat_body())
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["code"] == "context_overflow"
    assert "17020 tokens" in error["message"] and "maximum context length is 16384 tokens" in error["message"]


def test_undercounted_payload_is_rejected_by_ollama_exact_count_stream(client):
    enable_api(client)
    key = create_key(client)
    client.app.state.provider = OverflowProvider()
    response = client.post("/v1/chat/completions", headers=auth_headers(key), json=chat_body(stream=True))
    assert response.status_code == 200
    errors = [line for line in response.text.splitlines() if '"error"' in line]
    event = json.loads(errors[0][len("data: "):])["error"]
    assert event["code"] == "context_overflow" and event["type"] == "invalid_request_error"
    assert "17020 tokens" in event["message"]
    assert "[DONE]" not in response.text


def test_streaming_tool_calls_still_work_with_long_opencode_context(client):
    enable_api(client)
    key = create_key(client)
    stub = ToolsStubProvider(tool_calls=[ToolCall(name="read", arguments='{"filePath": "calc.py"}', id="call_1")])
    client.app.state.provider = stub
    raw = opencode_messages() + tool_step(1, "glob", {"pattern": "**/*.py"}, "calc.py\ntest_calc.py")
    response = client.post(
        "/v1/chat/completions", headers=auth_headers(key), json=chat_body(messages=raw, tools=opencode_tools(), stream=True)
    )
    chunks, done = parse_sse(response.text)
    assert done
    calls = [c for chunk in chunks for c in chunk["choices"][0]["delta"].get("tool_calls", [])] if chunks else []
    assert any(c.get("function", {}).get("name") == "read" for c in calls)
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    sent = stub.calls[0]["messages"]
    assert sent[2]["tool_calls"][0]["function"]["name"] == "glob" and sent[3]["role"] == "tool"


# ------------------------------------------------------------------ 9. isolamento API externa x chat interno


def test_internal_context_manager_keeps_chars_per_token_estimator():
    internal = ContextManager()
    assert internal.pretoken is False
    assert internal.estimate_text_tokens("x" * 300) == 100  # 3,0 chars/token inalterado
    assert internal.estimate_tools_schema_tokens([{"a": 1}]) == internal.estimate_text_tokens('[{"a": 1}]')


def test_ollama_payload_only_sends_truncate_when_requested():
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "ok"}, "done": True, "done_reason": "stop"})

    provider = OllamaProvider("m", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    msgs = [{"role": "user", "content": "oi"}]
    asyncio.run(provider.chat_turn(msgs, num_ctx=16384, num_predict=2048))  # chat interno / Deep Research
    asyncio.run(provider.chat_turn(msgs, num_ctx=16384, num_predict=2048, truncate=False, shift=False))  # /v1
    assert "truncate" not in seen[0] and "shift" not in seen[0]
    assert seen[1]["truncate"] is False and seen[1]["shift"] is False
    assert "truncate" not in seen[1]["options"] and "shift" not in seen[1]["options"]


def test_ollama_overflow_error_surfaces_exact_count():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "request (17020 tokens) exceeds the available context size (16384 tokens), try increasing it"})

    provider = OllamaProvider("m", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async def consume_stream():
        async for _ in provider.stream_turn([{"role": "user", "content": "oi"}], truncate=False):
            pass

    with pytest.raises(ProviderError, match="exceeds the available context size"):
        asyncio.run(provider.chat_turn([{"role": "user", "content": "oi"}], truncate=False))
    with pytest.raises(ProviderError, match="17020 tokens"):
        asyncio.run(consume_stream())


# ------------------------------------------------------------------ 10. bind local da API externa


def _lan_app(tmp_path):
    settings = make_settings(tmp_path)
    settings.app.allow_lan = True  # o que o start.bat faz (ENSLAVED_LAN=1, ENSLAVED_HOST=0.0.0.0)
    settings.app.host = "0.0.0.0"
    return create_app(settings)


def test_v1_is_loopback_only_even_when_server_listens_on_lan(tmp_path):
    app = _lan_app(tmp_path)
    with TestClient(app) as ui:
        enable_api(ui)
        key = create_key(ui)
        status = ui.get("/api/localapi", headers={"Sec-Fetch-Site": "same-origin"}).json()
        assert status["endpoint"] == "http://127.0.0.1:8000/v1"  # nunca 0.0.0.0
    with TestClient(app, client=("192.168.0.10", 50000)) as phone:
        lan = {"Host": "192.168.0.10:8000"}
        assert phone.get("/api/health", headers=lan).status_code == 200  # UI no celular continua
        denied = phone.post("/v1/chat/completions", headers={**lan, **auth_headers(key)}, json=chat_body())
        assert denied.status_code == 403 and denied.json()["error"]["code"] == "local_only"
        assert phone.get("/v1/models", headers={**lan, **auth_headers(key)}).status_code == 403
    with TestClient(app, client=("127.0.0.1", 50000)) as local:
        # Host de LAN mesmo vindo do loopback: bloqueado (só 127.0.0.1/localhost/::1)
        assert local.get("/v1/models", headers={"Host": "192.168.0.10:8000", **auth_headers(key)}).status_code == 403
        ok = local.get("/v1/models", headers={"Host": "127.0.0.1:8000", **auth_headers(key)})
        assert ok.status_code == 200
        assert local.get("/v1/models", headers={"Host": "127.0.0.1:8000"}).status_code == 401  # chave segue exigida


def test_default_bind_is_loopback(tmp_path):
    settings = make_settings(tmp_path)
    assert settings.app.host == "127.0.0.1" and settings.app.allow_lan is False

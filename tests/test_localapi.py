"""API local OpenAI-compatible (/v1): chaves, autenticação, chat, streaming, tool calling,
isolamento e segurança. Sem inferência real (providers simulados)."""

import asyncio
import hashlib
import json
import sqlite3
import threading
import time

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.providers.base import ChatTurn, ModelProvider, ProviderError, ProviderUnavailableError, ToolCall
from tests.helpers import make_settings

UI_HEADERS = {"Sec-Fetch-Site": "same-origin"}


class ToolsStubProvider(ModelProvider):
    """Provider determinístico com tool calling, atraso e falhas controláveis."""

    name = "stub"

    def __init__(self, model="simulado-1b", content="resposta do stub", tool_calls=None, delay=0.0, fail=None):
        super().__init__(model)
        self.content = content
        self._tool_calls = list(tool_calls or [])
        self.delay = delay
        self.fail = fail
        self.calls: list[dict] = []

    def _record(self, messages, tools, options):
        self.calls.append({"messages": messages, "tools": tools, "options": options})
        self.last_usage = {"prompt_eval_count": 11, "eval_count": 7}

    async def _maybe_fail(self):
        if self.delay:
            await asyncio.sleep(self.delay)  # não bloquear o event loop (concorrência real)
        if self.fail == "unavailable":
            raise ProviderUnavailableError("Ollama indisponível em http://127.0.0.1:11434 (stub).")
        if self.fail == "provider":
            raise ProviderError("Erro do Ollama: modelo não encontrado (stub).")

    async def chat(self, messages, **options):
        self._record(messages, None, options)
        await self._maybe_fail()
        return self.content

    async def stream_chat(self, messages, **options):
        yield await self.chat(messages, **options)

    async def chat_turn(self, messages, tools=None, **options):
        self._record(messages, tools, options)
        await self._maybe_fail()
        return ChatTurn(content=self.content, tool_calls=list(self._tool_calls))

    async def stream_turn(self, messages, tools=None, **options):
        self._record(messages, tools, options)
        await self._maybe_fail()
        yield "Olá "
        yield "mundo"
        yield ChatTurn(content="Olá mundo", tool_calls=list(self._tool_calls))

    async def health(self):
        return {"available": True, "detail": "stub", "models": [self.model]}

    async def supports_native_tools(self):
        return True


# ------------------------------------------------------------------ helpers


def enable_api(client):
    response = client.post("/api/localapi/enabled", json={"enabled": True}, headers=UI_HEADERS)
    assert response.status_code == 200, response.text


def create_key(client, name="OpenCode"):
    response = client.post("/api/localapi/keys", json={"name": name}, headers=UI_HEADERS)
    assert response.status_code == 200, response.text
    return response.json()


def auth_headers(key):
    return {"Authorization": f"Bearer {key['secret']}"}


def parse_sse(text):
    """Retorna (chunks_json, done)."""
    chunks, done = [], False
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            done = True
            continue
        chunks.append(json.loads(payload))
    return chunks, done


def chat_body(**overrides):
    body = {"messages": [{"role": "user", "content": "olá"}]}
    body.update(overrides)
    return body


# ------------------------------------------------------------------ chaves (1-10)


def test_first_key_created_and_shown_once(client):
    enable_api(client)
    key = create_key(client, "OpenCode")
    assert key["secret"].startswith("ens_")
    assert len(key["secret"]) >= 40  # ens_ + 43 chars base64url (256 bits)
    assert key["masked"] != key["secret"] and key["secret"] not in key["masked"]
    assert key["status"] == "active"
    listed = client.get("/api/localapi", headers=UI_HEADERS).json()
    assert listed["active_keys"] == 1
    assert listed["keys"][0]["masked"] == key["masked"]
    # o segredo nunca volta em listagens
    assert all(key["secret"] not in json.dumps(k) for k in listed["keys"])


def test_five_keys_then_sixth_blocked(client):
    enable_api(client)
    for i in range(5):
        assert create_key(client, f"chave-{i}")["status"] == "active"
    sixth = client.post("/api/localapi/keys", json={"name": "sexta"}, headers=UI_HEADERS)
    assert sixth.status_code == 409
    assert "Limite" in sixth.json()["detail"]
    assert client.get("/api/localapi", headers=UI_HEADERS).json()["active_keys"] == 5


def test_revoke_then_create_new_key(client):
    enable_api(client)
    first = create_key(client, "Aider")
    assert client.delete(f"/api/localapi/keys/{first['id']}", headers=UI_HEADERS).json()["revoked"] is True
    keys = client.get("/api/localapi", headers=UI_HEADERS).json()["keys"]
    assert keys[0]["status"] == "revoked" and keys[0]["revoked_at"]
    second = create_key(client, "VS Code")  # vagou um slot: nova chave é aceita
    assert second["status"] == "active"
    assert second["secret"] != first["secret"]


def test_key_uniqueness_and_entropy(client):
    enable_api(client)
    secrets = [create_key(client, f"k{i}")["secret"] for i in range(5)]
    assert len(set(secrets)) == 5
    assert all(s.startswith("ens_") and len(s) >= 40 for s in secrets)


def test_only_hashes_persisted(client, settings):
    enable_api(client)
    key = create_key(client, "OpenCode")
    conn = sqlite3.connect(settings.storage.database_path)
    rows = conn.execute("SELECT key_hash, key_prefix, name FROM api_keys").fetchall()
    conn.close()
    assert len(rows) == 1
    key_hash, key_prefix, _name = rows[0]
    assert key_hash == hashlib.sha256(key["secret"].encode("utf-8")).hexdigest()
    assert len(key_hash) == 64 and all(c in "0123456789abcdef" for c in key_hash)
    assert key["secret"] not in key_prefix  # prefixo é só a parte exibível


def test_invalid_missing_and_revoked_keys_rejected(client):
    enable_api(client)
    key = create_key(client, "OpenCode")
    assert client.get("/v1/models").status_code == 401  # sem header
    assert client.get("/v1/models", headers={"Authorization": "Basic abc"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer ens_lixo123456"}).status_code == 401
    client.delete(f"/api/localapi/keys/{key['id']}", headers=UI_HEADERS)
    revoked = client.get("/v1/models", headers=auth_headers(key))
    assert revoked.status_code == 401
    assert revoked.json()["error"]["code"] == "invalid_api_key"


def test_api_key_cannot_manage_keys(client):
    enable_api(client)
    key = create_key(client, "OpenCode")
    auth = auth_headers(key)  # chave /v1, sem cabeçalho de navegador: nada de gerenciamento
    assert client.get("/api/localapi", headers=auth).status_code == 403
    assert client.post("/api/localapi/keys", json={"name": "x"}, headers=auth).status_code == 403
    assert client.delete(f"/api/localapi/keys/{key['id']}", headers=auth).status_code == 403
    assert client.post("/api/localapi/enabled", json={"enabled": False}, headers=auth).status_code == 403


def test_keys_and_enabled_state_persist_after_restart(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        enable_api(client)
        key = create_key(client, "OpenCode")
    with TestClient(create_app(settings)) as client:  # mesma base SQLite, novo processo
        status = client.get("/api/localapi", headers=UI_HEADERS).json()
        assert status["enabled"] is True  # toggle persistiu em data/local-api.json
        assert len(status["keys"]) == 1
        assert client.get("/v1/models", headers=auth_headers(key)).status_code == 200


# ------------------------------------------------------------------ /v1 (11-17)


def test_disabled_api_returns_503(client):
    response = client.get("/v1/models", headers={"Authorization": "Bearer ens_x"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "api_disabled"


def test_models_requires_auth_and_lists_active_model(client):
    enable_api(client)
    key = create_key(client)
    response = client.get("/v1/models", headers=auth_headers(key))
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert [m["id"] for m in data["data"]] == ["simulado-1b"]
    assert data["data"][0]["object"] == "model"


def test_chat_completion_non_streaming(client):
    enable_api(client)
    key = create_key(client)
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers(key),
        json=chat_body(
            model="simulado-1b",
            messages=[
                {"role": "system", "content": "Seja direto."},
                {"role": "developer", "content": "Responda em pt-BR."},
                {"role": "user", "content": "olá"},
            ],
            temperature=0.2,
            top_p=0.9,
            max_tokens=128,
        ),
    )
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "simulado-1b"
    assert data["id"].startswith("chatcmpl-")
    choice = data["choices"][0]
    assert choice["index"] == 0
    assert choice["finish_reason"] == "stop"
    assert choice["message"]["role"] == "assistant"
    assert "[simulado]" in choice["message"]["content"]
    assert data["usage"] is None  # provider simulado não reporta uso


def test_chat_completion_streaming(client):
    enable_api(client)
    key = create_key(client)
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers(key),
        json=chat_body(stream=True, stream_options={"include_usage": True}),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    chunks, done = parse_sse(response.text)
    assert done
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert len({c["id"] for c in chunks}) == 1  # mesmo id para todos os chunks
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert "[simulado]" in content
    finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"][0]["finish_reason"]]
    assert finish == ["stop"]
    # provider simulado não reporta uso: nenhum chunk de usage é emitido
    assert [c for c in chunks if "usage" in c] == []


def test_tool_calling_native_and_tool_result_round_trip(client):
    enable_api(client)
    key = create_key(client)
    stub = ToolsStubProvider(content="", tool_calls=[ToolCall(name="fs.read", arguments='{"path": "docs/a.md"}', id="call_abc")])
    client.app.state.provider = stub
    tools = [
        {
            "type": "function",
            "function": {
                "name": "fs.read",
                "description": "Lê um arquivo",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers(key),
        json=chat_body(messages=[{"role": "user", "content": "leia docs/a.md"}], tools=tools),
    )
    assert response.status_code == 200
    data = response.json()
    choice = data["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    call = choice["message"]["tool_calls"][0]
    assert call["id"] == "call_abc"  # id preservado
    assert call["type"] == "function"
    assert call["function"]["name"] == "fs.read"
    assert json.loads(call["function"]["arguments"]) == {"path": "docs/a.md"}
    assert choice["message"]["content"] == ""
    assert stub.calls[0]["tools"] == tools  # schemas encaminhados ao modelo

    # cliente devolve o resultado com role:tool e a conversa continua
    follow_up = client.post(
        "/v1/chat/completions",
        headers=auth_headers(key),
        json=chat_body(
            messages=[
                {"role": "user", "content": "leia docs/a.md"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "call_abc", "type": "function", "function": {"name": "fs.read", "arguments": "{\"path\": \"docs/a.md\"}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_abc", "content": "conteúdo do arquivo"},
            ]
        ),
    )
    assert follow_up.status_code == 200
    converted = stub.calls[-1]["messages"]
    assert converted[1]["role"] == "assistant"
    assert converted[1]["tool_calls"][0]["function"]["name"] == "fs.read"
    assert converted[1]["tool_calls"][0]["function"]["arguments"] == {"path": "docs/a.md"}
    assert converted[2]["role"] == "tool"
    assert converted[2]["tool_name"] == "fs.read"  # mapeado do tool_call_id
    assert converted[2]["content"] == "conteúdo do arquivo"


def test_streaming_tool_call_deltas_and_finish_reason(client):
    enable_api(client)
    key = create_key(client)
    stub = ToolsStubProvider(
        tool_calls=[
            ToolCall(name="fs.read", arguments='{"path": "x.md"}', id="call_1"),
            ToolCall(name="net.fetch", arguments="{}", id=None),
        ]
    )
    client.app.state.provider = stub
    response = client.post(
        "/v1/chat/completions",
        headers=auth_headers(key),
        json=chat_body(stream=True, tools=[{"type": "function", "function": {"name": "fs.read"}}]),
    )
    chunks, done = parse_sse(response.text)
    assert done
    tool_deltas = [c["choices"][0]["delta"]["tool_calls"] for c in chunks if c["choices"][0]["delta"].get("tool_calls")]
    assert tool_deltas, "esperado delta de tool_calls"
    first = tool_deltas[0][0]
    assert first["index"] == 0 and first["id"] == "call_1" and first["function"]["name"] == "fs.read"
    args = "".join(
        d[0]["function"].get("arguments", "") for d in tool_deltas if d[0]["index"] == 0
    )
    assert json.loads(args) == {"path": "x.md"}
    second = [d[0] for d in tool_deltas if d[0]["index"] == 1]
    assert second[0]["id"]  # id gerado quando o provider não envia
    assert second[0]["function"]["name"] == "net.fetch"
    finish = [c["choices"][0]["finish_reason"] for c in chunks if c["choices"][0]["finish_reason"]]
    assert finish == ["tool_calls"]


def test_unknown_model_and_provider_errors(client):
    enable_api(client)
    key = create_key(client)
    unknown = client.post("/v1/chat/completions", headers=auth_headers(key), json=chat_body(model="gpt-4o"))
    assert unknown.status_code == 404
    assert unknown.json()["error"]["code"] == "model_not_found"

    stub = ToolsStubProvider(fail="unavailable")
    client.app.state.provider = stub
    down = client.post("/v1/chat/completions", headers=auth_headers(key), json=chat_body())
    assert down.status_code == 503
    assert down.json()["error"]["code"] == "provider_unavailable"

    stub_err = ToolsStubProvider(fail="provider")
    client.app.state.provider = stub_err
    erro = client.post("/v1/chat/completions", headers=auth_headers(key), json=chat_body())
    assert erro.status_code == 502


def test_stream_provider_error_yields_error_event(client):
    enable_api(client)
    key = create_key(client)
    client.app.state.provider = ToolsStubProvider(fail="unavailable")
    response = client.post(
        "/v1/chat/completions", headers=auth_headers(key), json=chat_body(stream=True)
    )
    assert response.status_code == 200  # stream já começou: erro vem como evento
    assert '"error"' in response.text
    _, done = parse_sse(response.text)
    assert not done  # sem [DONE] após erro


def test_output_and_context_budgets(client):
    enable_api(client)
    key = create_key(client)
    over = client.post(
        "/v1/chat/completions",
        headers=auth_headers(key),
        json=chat_body(max_tokens=9999),  # orçamento do perfil: 2048
    )
    assert over.status_code == 400
    assert over.json()["error"]["code"] == "output_budget_exceeded"

    stub = ToolsStubProvider()
    client.app.state.provider = stub
    ok = client.post(
        "/v1/chat/completions",
        headers=auth_headers(key),
        json=chat_body(max_tokens=512, temperature=0.3, top_p=0.8),
    )
    assert ok.status_code == 200
    options = stub.calls[0]["options"]
    assert options["num_predict"] == 512
    assert options["temperature"] == 0.3 and options["top_p"] == 0.8
    assert options["num_ctx"] == 16384  # orçamento do perfil ativo
    assert ok.json()["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}

    overflow = client.post(
        "/v1/chat/completions",
        headers=auth_headers(key),
        json=chat_body(messages=[{"role": "user", "content": "x" * 200_000}]),
    )
    assert overflow.status_code == 400
    assert overflow.json()["error"]["code"] == "context_overflow"


def test_request_limits(client):
    enable_api(client)
    key = create_key(client)
    auth = auth_headers(key)
    too_many = client.post("/v1/chat/completions", headers=auth, json=chat_body(messages=[{"role": "user", "content": "x"}] * 161))
    assert too_many.status_code == 400
    huge = client.post("/v1/chat/completions", headers=auth, json=chat_body(messages=[{"role": "user", "content": "y" * (4 * 1024 * 1024 + 1)}]))
    assert huge.status_code == 413
    invalid_params = [
        {"messages": []},
        {"messages": [{"role": "robot", "content": "oi"}]},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,x"}}]}]},
        {"messages": [{"role": "user", "content": "oi"}], "temperature": 5},
        {"messages": [{"role": "user", "content": "oi"}], "n": 2},
        {"messages": [{"role": "user", "content": "oi"}], "max_tokens": "muito"},
        {"messages": [{"role": "user", "content": "oi"}], "tools": []},
        {"messages": [{"role": "user", "content": "oi"}], "stream": "sim"},
    ]
    for payload in invalid_params:
        response = client.post("/v1/chat/completions", headers=auth, json=chat_body(**payload))
        assert response.status_code == 400, payload
        assert "error" in response.json()
    malformed = client.post(
        "/v1/chat/completions", headers={**auth, "Content-Type": "application/json"}, content=b"{invalid"
    )
    assert malformed.status_code == 400


# ------------------------------------------------------------------ concorrência e cancelamento (18)


@pytest.fixture
def busy_client(tmp_path):
    settings = make_settings(tmp_path, local_api={"max_concurrent": 1, "queue_timeout_seconds": 0.3})
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_concurrency_limit_returns_429(busy_client):
    enable_api(busy_client)
    key = create_key(busy_client)
    auth = auth_headers(key)
    busy_client.app.state.provider = ToolsStubProvider(delay=0.8)

    first_result = {}

    def first_request():
        first_result["response"] = busy_client.post("/v1/chat/completions", headers=auth, json=chat_body())

    thread = threading.Thread(target=first_request)
    thread.start()
    time.sleep(0.15)  # primeira requisição segura o slot
    second = busy_client.post("/v1/chat/completions", headers=auth, json=chat_body())
    thread.join()
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "busy"
    assert first_result["response"].status_code == 200


def test_stream_releases_slot_after_completion(busy_client):
    enable_api(busy_client)
    key = create_key(busy_client)
    auth = auth_headers(key)
    busy_client.app.state.provider = ToolsStubProvider(delay=0.1)
    first = busy_client.post("/v1/chat/completions", headers=auth, json=chat_body(stream=True))
    assert first.status_code == 200
    _, done = parse_sse(first.text)
    assert done
    second = busy_client.post("/v1/chat/completions", headers=auth, json=chat_body())
    assert second.status_code == 200  # slot liberado após o stream terminar


# ------------------------------------------------------------------ isolamento (19) e segurança (20)


def test_v1_stateless_isolated_from_conversations(client):
    enable_api(client)
    key = create_key(client)
    stub = ToolsStubProvider()
    client.app.state.provider = stub
    response = client.post(
        "/v1/chat/completions", headers=auth_headers(key), json=chat_body(messages=[{"role": "user", "content": "segredo da IDE"}])
    )
    assert response.status_code == 200
    assert client.get("/api/conversations").json()["conversations"] == []  # nada persistido
    sent = stub.calls[0]["messages"]
    assert sent == [{"role": "user", "content": "segredo da IDE"}]  # sem system/directives/memórias injetados


def test_host_and_cross_origin_blocked(client):
    enable_api(client)
    key = create_key(client)
    auth = auth_headers(key)
    bad_host = client.get("/v1/models", headers={**auth, "Host": "evil.example"})
    assert bad_host.status_code == 403
    cross_origin = client.post(
        "/v1/chat/completions",
        headers={**auth, "Origin": "http://evil.example"},
        json=chat_body(),
    )
    assert cross_origin.status_code == 403


# ------------------------------------------------------------------ compatibilidade (21, 22)


def test_existing_routes_still_work(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/config").status_code == 200
    chat = client.post("/api/chat", json={"message": "olá interno"})
    assert chat.status_code == 200
    conversation_id = chat.json()["conversation_id"]
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 200
    assert client.delete(f"/api/conversations/{conversation_id}").json()["deleted"] is True


def test_internal_sse_format_preserved(client):
    response = client.post("/api/chat/stream", json={"message": "olá interno"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events
    assert all("type" in event for event in events)  # formato próprio da interface, não OpenAI
    assert any(event["type"] == "token" for event in events)


# ------------------------------------------------------------------ regressões da auditoria final


def test_chunked_body_without_content_length_respects_limit(tmp_path):
    settings = make_settings(tmp_path, local_api={"max_payload_bytes": 1024})
    with TestClient(create_app(settings)) as client:
        enable_api(client)
        key = create_key(client)
        client.app.state.provider = stub = ToolsStubProvider()
        body = json.dumps(chat_body(messages=[{"role": "user", "content": "x" * 20_000}])).encode()

        def chunked():  # gerador => Transfer-Encoding: chunked, sem Content-Length
            for i in range(0, len(body), 512):
                yield body[i:i + 512]

        response = client.post("/v1/chat/completions", content=chunked(), headers={**auth_headers(key), "Content-Type": "application/json"})
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"
        assert stub.calls == []


def test_stream_closed_at_first_chunk_releases_slot():
    from backend.api.routes_v1 import _Slot, _stream_sse

    class State:
        provider = ToolsStubProvider()

    async def scenario():
        semaphore = asyncio.Semaphore(1)
        await semaphore.acquire()
        flow = _stream_sse(State(), [{"role": "user", "content": "oi"}], [], {}, "m", "id", False, _Slot(semaphore))
        await flow.__anext__()  # cliente desconecta logo após o chunk de role
        await flow.aclose()
        return semaphore.locked()

    assert asyncio.run(scenario()) is False


def test_stream_usage_is_per_request_not_shared_provider_state():
    from backend.api.routes_v1 import _Slot, _stream_sse

    class SharedProvider(ToolsStubProvider):
        async def stream_turn(self, messages, tools=None, **options):
            yield "A"
            yield ChatTurn(content="A", usage={"prompt_eval_count": 10, "eval_count": 5}, finish_reason="stop")

    class State:
        provider = SharedProvider()

    async def scenario():
        semaphore = asyncio.Semaphore(2)
        await semaphore.acquire()
        out = []
        async for chunk in _stream_sse(State(), [{"role": "user", "content": "oi"}], [], {}, "m", "id", True, _Slot(semaphore)):
            out.append(chunk)
            if '"finish_reason": "stop"' in chunk:  # outra geração termina e sobrescreve o estado do provider
                State.provider.last_usage = {"prompt_eval_count": 9999, "eval_count": 2048}
                State.provider.last_finish_reason = "length"
        return "".join(out)

    chunks, done = parse_sse(asyncio.run(scenario()))
    assert done
    usage_chunks = [c for c in chunks if "usage" in c]
    assert len(usage_chunks) == 1 and usage_chunks[0]["choices"] == []
    assert usage_chunks[0]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

"""API local /v1: ciclo de vida da vaga SSE (cancelamento, desconexão, erro) e isolamento
dos metadados de geração entre requisições concorrentes (/v1 × /v1 e chat da UI × /v1).
Somente providers simulados, sem inferência real."""

import asyncio
import threading

import pytest
from fastapi.testclient import TestClient
from starlette.requests import ClientDisconnect

from backend.api.routes_v1 import SlotStreamingResponse, _Slot, _stream_sse
from backend.main import create_app
from backend.providers.base import ChatTurn, begin_request_generation_scope
from tests.helpers import make_settings
from tests.test_deep_research import DEEP_MSG, collect, last_assistant, public_dns  # noqa: F401
from tests.test_localapi import ToolsStubProvider, auth_headers, create_key, enable_api, parse_sse
from tests.test_report_quality import Cutter, _service_with


class TrackingProvider(ToolsStubProvider):
    """Stream de N trechos que registra início e fechamento (o aclosing deve fechá-lo sempre)."""

    def __init__(self, chunks=5, fail=None):
        super().__init__(fail=fail)
        self.chunks, self.started, self.closed = chunks, False, False

    async def stream_turn(self, messages, tools=None, **options):
        self.started = True
        try:
            await self._maybe_fail()
            for i in range(self.chunks):
                await asyncio.sleep(0)
                yield f"t{i} "
            yield ChatTurn(content="ok", usage={"prompt_eval_count": 1, "eval_count": 1}, finish_reason="stop")
        finally:
            self.closed = True


def _response(provider, semaphore):
    class State:
        pass

    State.provider = provider
    slot = _Slot(semaphore)
    flow = _stream_sse(State, [{"role": "user", "content": "oi"}], [], {}, "m", "id", False, slot)
    return SlotStreamingResponse(flow, slot, media_type="text/event-stream")


def _scope(spec="2.4"):
    return {"type": "http", "asgi": {"spec_version": spec}, "method": "POST", "path": "/v1/chat/completions", "headers": []}


async def _never_disconnect():
    await asyncio.Event().wait()
    return {"type": "http.disconnect"}


async def _acquired(value=1):
    semaphore = asyncio.Semaphore(value)
    await semaphore.acquire()
    return semaphore


def _run(coro):
    return asyncio.run(asyncio.wait_for(coro, 10))


# ------------------------------------------------------------------ 1-4, 8: ciclo de vida da vaga


def test_cancel_before_stream_starts_releases_slot_once():
    """Cliente já desconectado (ex.: desistiu na fila): o Starlette cancela o envio antes da 1ª iteração."""

    async def scenario():
        semaphore, provider = await _acquired(), TrackingProvider()

        async def disconnected():
            return {"type": "http.disconnect"}

        async def send(message):
            await asyncio.sleep(0)  # ponto de cancelamento antes do 1º chunk

        await _response(provider, semaphore)(_scope("2.3"), disconnected, send)
        return semaphore._value, provider.started

    assert _run(scenario()) == (1, False)  # vaga devolvida exatamente uma vez; Ollama nunca chamado


def test_send_failure_before_first_response_releases_slot():
    async def scenario():
        semaphore, provider = await _acquired(), TrackingProvider()

        async def send(message):
            raise OSError("conexão encerrada")

        with pytest.raises(ClientDisconnect):
            await _response(provider, semaphore)(_scope(), _never_disconnect, send)
        return semaphore._value, provider.started

    assert _run(scenario()) == (1, False)


def test_disconnect_on_first_chunk_releases_slot():
    async def scenario():
        semaphore, provider = await _acquired(), TrackingProvider()

        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("cliente saiu")

        with pytest.raises(ClientDisconnect):
            await _response(provider, semaphore)(_scope(), _never_disconnect, send)
        return semaphore._value

    assert _run(scenario()) == 1


def test_disconnect_after_several_chunks_releases_slot_and_closes_provider_stream():
    async def scenario():
        semaphore, provider = await _acquired(), TrackingProvider(chunks=20)
        sent = []

        async def send(message):
            if message["type"] == "http.response.body":
                sent.append(message)
                if len(sent) > 3:
                    raise OSError("cliente saiu")

        with pytest.raises(ClientDisconnect):
            await _response(provider, semaphore)(_scope(), _never_disconnect, send)
        return semaphore._value, provider.started, provider.closed

    assert _run(scenario()) == (1, True, True)


def test_task_cancelled_mid_stream_releases_slot_and_closes_provider_stream():
    async def scenario():
        semaphore, provider = await _acquired(), TrackingProvider(chunks=1000)
        three = asyncio.Event()
        sent = []

        async def send(message):
            sent.append(message)
            if len(sent) >= 4:
                three.set()
            await asyncio.sleep(0)

        task = asyncio.create_task(_response(provider, semaphore)(_scope(), _never_disconnect, send))
        await three.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return semaphore._value, provider.closed

    assert _run(scenario()) == (1, True)


def test_provider_error_and_normal_end_release_slot_once():
    async def scenario():
        results = []
        for provider in (TrackingProvider(fail="provider"), TrackingProvider()):
            semaphore = await _acquired()
            sent = []

            async def send(message):
                sent.append(message)

            await _response(provider, semaphore)(_scope(), _never_disconnect, send)
            body = b"".join(m.get("body", b"") for m in sent).decode()
            results.append((semaphore._value, "[DONE]" in body, '"error"' in body))
        return results

    assert _run(scenario()) == [(1, False, True), (1, True, False)]


def test_many_concurrent_disconnects_and_cancellations_leak_no_slot():
    async def scenario():
        semaphore = asyncio.Semaphore(4)
        tasks = []
        for i in range(24):
            await semaphore.acquire()
            provider = TrackingProvider(chunks=50, fail="provider" if i % 6 == 5 else None)

            async def send(message, i=i, count=[0]):
                await asyncio.sleep(0)
                if i % 3 == 1 and message["type"] == "http.response.body":
                    raise OSError("cliente saiu")

            async def receive(i=i):
                if i % 3 == 2:
                    return {"type": "http.disconnect"}
                return await _never_disconnect()

            spec = "2.3" if i % 3 == 2 else "2.4"
            task = asyncio.create_task(_response(provider, semaphore)(_scope(spec), receive, send))
            if i % 4 == 0:
                await asyncio.sleep(0)
                task.cancel()
            tasks.append(task)
        await asyncio.gather(*tasks, return_exceptions=True)
        return semaphore._value

    assert _run(scenario()) == 4  # capacidade total de volta, sem vaga perdida nem liberada em dobro


@pytest.fixture
def one_slot_client(tmp_path):
    settings = make_settings(tmp_path, local_api={"max_concurrent": 1, "queue_timeout_seconds": 0.3})
    with TestClient(create_app(settings)) as client:
        enable_api(client)
        yield client, auth_headers(create_key(client))


def test_capacity_recovers_end_to_end_after_provider_error(one_slot_client):
    client, auth = one_slot_client
    client.app.state.provider = ToolsStubProvider(fail="provider")
    body = {"messages": [{"role": "user", "content": "oi"}], "stream": True}
    for _ in range(3):  # com 1 vaga, um vazamento faria a próxima requisição virar 429
        response = client.post("/v1/chat/completions", headers=auth, json=body)
        assert response.status_code == 200 and parse_sse(response.text)[0][-1]["error"]["code"] == "provider_error"
    client.app.state.provider = ToolsStubProvider()
    response = client.post("/v1/chat/completions", headers=auth, json=body)
    assert response.status_code == 200 and parse_sse(response.text)[1]
    assert client.app.state.local_api_semaphore._value == 1


# ------------------------------------------------------------------ 5: duas /v1 simultâneas


class CrossingProvider(ToolsStubProvider):
    """Duas gerações sobrepostas que gravam finish/usage diferentes SÓ no estado do provider
    (sem metadados no ChatTurn), uma esperando a outra: sem isolamento, uma leria a da outra."""

    def __init__(self):
        super().__init__()
        self.marks = {"curta": asyncio.Event(), "longa": asyncio.Event()}

    async def stream_turn(self, messages, tools=None, **options):
        mine = messages[-1]["content"]
        other = "longa" if mine == "curta" else "curta"
        self.last_finish_reason = "length" if mine == "longa" else "stop"
        self.last_usage = {"prompt_eval_count": len(mine), "eval_count": 100 if mine == "longa" else 1}
        self.marks[mine].set()
        await self.marks[other].wait()  # a outra requisição gravou depois desta
        yield mine
        yield ChatTurn(content=mine)


def test_two_concurrent_v1_requests_keep_their_own_finish_reason_and_usage(tmp_path):
    settings = make_settings(tmp_path, local_api={"max_concurrent": 2})
    with TestClient(create_app(settings)) as client:
        enable_api(client)
        auth = auth_headers(create_key(client))
        client.app.state.provider = CrossingProvider()
        results = {}

        def call(text):
            body = {"messages": [{"role": "user", "content": text}], "stream": True, "stream_options": {"include_usage": True}}
            results[text] = parse_sse(client.post("/v1/chat/completions", headers=auth, json=body).text)

        threads = [threading.Thread(target=call, args=(t,)) for t in ("curta", "longa")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)

        for text, finish, completion in (("curta", "stop", 1), ("longa", "length", 100)):
            chunks, done = results[text]
            assert done
            assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]] == [finish]
            assert chunks[-1]["usage"] == {"prompt_tokens": len(text), "completion_tokens": completion, "total_tokens": len(text) + completion}
        assert client.app.state.local_api_semaphore._value == 2


# ------------------------------------------------------------------ 6-7: chat da UI × /v1


class RacingCutter(Cutter):
    """O 1º relatório da UI é cortado (length) e só termina depois que uma /v1 simultânea
    grava 'stop' no mesmo provider — exatamente a janela da condição de corrida."""

    def __init__(self, script):
        super().__init__(script)
        self.ui_marked, self.v1_done, self.ui_streams = asyncio.Event(), asyncio.Event(), 0

    async def stream_chat(self, messages, **options):
        self.ui_streams += 1
        first = self.ui_streams == 1
        async for chunk in super().stream_chat(messages, **options):
            yield chunk
        if first:
            self.ui_marked.set()
            await self.v1_done.wait()

    async def stream_turn(self, messages, tools=None, **options):
        await self.ui_marked.wait()
        self.last_finish_reason = "stop"
        self.last_usage = {"prompt_eval_count": 5, "eval_count": 3}
        yield "resposta externa"
        yield ChatTurn(content="resposta externa")
        self.v1_done.set()


BODY = "## Resumo executivo\n\nA CSP está ausente [F1] conforme [E1].\n\n## Recomendações\n\n- Implementar uma CSP restritiva (Content-Security-P"
TAIL = "olicy) e revisar a política regularmente.\n"


def _race(service, scoped):
    provider = service.provider

    class State:
        pass

    State.provider = provider

    async def ui():
        if scoped:
            begin_request_generation_scope()  # o que o middleware HTTP faz em cada requisição
        return await collect(service, DEEP_MSG)

    async def v1():
        if scoped:
            begin_request_generation_scope()
        semaphore = await _acquired()
        chunks = [c async for c in _stream_sse(State, [{"role": "user", "content": "x"}], [], {}, "m", "id", True, _Slot(semaphore))]
        return "".join(chunks), semaphore._value

    async def both():
        return await asyncio.gather(ui(), v1())

    return asyncio.run(asyncio.wait_for(both(), 20))


def test_ui_truncated_report_still_continues_while_v1_finishes_concurrently(db, settings, public_dns):
    service = _service_with(db, settings, [(BODY, "length"), (TAIL, "stop")])
    service.provider = RacingCutter([(BODY, "length"), (TAIL, "stop")])
    events, (v1_text, v1_slot) = _race(service, scoped=True)

    message = last_assistant(service, events)
    assert message.meta["research"]["completion"]["continuations"] == 1  # leu o próprio "length", não o "stop" da /v1
    assert TAIL.strip() in message.content
    chunks, done = parse_sse(v1_text)
    assert done and v1_slot == 1
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"] and c["choices"][0]["finish_reason"]] == ["stop"]
    assert chunks[-1]["usage"] == {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}


def test_without_request_scope_the_race_is_reproduced(db, settings, public_dns):
    """Controle: sem o escopo por requisição (estado compartilhado), a UI lê o 'stop' da /v1 e não continua."""
    service = _service_with(db, settings, [(BODY, "length"), (TAIL, "stop")])
    service.provider = RacingCutter([(BODY, "length"), (TAIL, "stop")])
    events, _ = _race(service, scoped=False)
    assert last_assistant(service, events).meta["research"]["completion"]["continuations"] == 0

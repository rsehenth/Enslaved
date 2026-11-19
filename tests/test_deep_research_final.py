"""Regressões da homologação integrada: opções da UI, término, falhas e revisão."""

import asyncio
import hashlib
import json
import socket
import threading
import time
from contextlib import asynccontextmanager

import httpx
import pytest

from backend.api.routes_chat import research_cancel
from backend.main import create_app
from backend.providers.base import ProviderError
from backend.providers.ollama import OllamaProvider
from backend.research.report import review_report
from backend.research.state import ResearchRegistry
from tests.test_deep_research import (
    AMBIG_MSG, DEEP_MSG, HOME, Reporter, Site, collect, last_assistant, make_env, public_dns, run,  # noqa: F401
)
from tests.test_report_quality import Cutter
from tests.helpers import make_settings
from tests.test_research_ui import browser  # noqa: F401


def test_extended_after_depth_and_permission_uses_report_budget_despite_ui_defaults(db, settings, public_dns):
    service, _, permissions, _ = make_env(db, settings, Site(), net_mode="ask")
    provider = service.provider = Cutter([("Resumo completo com evidências [F1] [E1].", "stop")])
    depth = next(e for e in run(collect(service, AMBIG_MSG)) if e["type"] == "research_depth_request")
    conv = depth["conversation_id"]
    permission = next(e for e in run(collect(service, "", conv, retry=True, research_mode="extended")) if e["type"] == "permission_request")
    permissions.resolve(permission["request_id"], granted=True)
    service.record_web_decision(conv, True, permission["arguments"]["consulta"])
    events = run(collect(service, "", conv, retry=True, options={"temperature": 0.7, "num_predict": 1024, "think": True}))
    assert provider.seen[0]["num_predict"] == 4608
    assert provider.seen[0]["think"] is False
    assert [e["phase"] for e in events if e["type"] == "research_progress"][:5] == ["planning", "collecting", "collecting", "analyzing", "verifying"]
    assert last_assistant(service, events).meta["research"]["mode"] == "extended"
    assert service.research.mode(conv) == "quick"


def test_ollama_eof_without_done_is_an_error():
    provider = OllamaProvider("m", client=httpx.AsyncClient(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, text=json.dumps({"message": {"content": "Frase cortada ao mei"}, "done": False}) + "\n")
    )))
    async def go():
        try:
            return [x async for x in provider.stream_chat([])]
        finally:
            await provider.aclose()
    with pytest.raises(ProviderError, match="sem confirmar"):
        run(go())


@pytest.mark.parametrize("phase", ["collecting", "reporting"])
def test_cancel_endpoint_interrupts_inflight_collection_or_generation(db, settings, public_dns, phase):
    service, _, _, _ = make_env(db, settings, Site())
    async def scenario():
        entered, stopped = asyncio.Event(), asyncio.Event()
        async def wait():
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
        if phase == "reporting":
            class WaitingReporter(Reporter):
                async def stream_chat(self, *args, **kwargs):
                    await wait()
                    yield "nunca"
            service.provider = WaitingReporter()
        else:
            async def inspect(*args):
                await wait()
            service._inspect_tool = inspect
        pending = asyncio.create_task(collect(service, DEEP_MSG))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            conv = next(iter(service.research._current))
            decision = research_cancel(conv, service)
            if asyncio.iscoroutine(decision):
                decision = await decision
            assert decision == {"cancelled": True}
            events = await asyncio.wait_for(asyncio.shield(pending), 0.5)
            assert stopped.is_set() and not service._active_turns
            assert not any(e["type"] == "done" for e in events)
            assert service.research.mode(conv) == "quick"
            assert service.research.current(conv).research_status == "cancelled"
            assert not [m for m in service.repo.get_messages(conv) if m.role == "assistant"]
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    run(scenario())


def test_network_failure_keeps_failed_ledger_in_live_and_persisted_panel(db, settings, public_dns):
    def offline(request):
        raise httpx.ConnectError("offline")
    service, provider, _, _ = make_env(db, settings, offline)
    events = run(collect(service, DEEP_MSG))
    message = last_assistant(service, events)
    research = message.meta["research"]
    assert research["completion"]["status"] == "partial"
    assert research["view"]["ledger"] and not research["view"]["ledger"][0]["ok"]
    assert next(e for e in events if e["type"] == "done")["research"]["view"] == research["view"]
    assert not provider.calls


def task_with_absent_hsts():
    task = ResearchRegistry().begin("c", "extended", {"goal": "g", "profile": "security", "targets": ["https://example.com/"]})
    evidence = task.ledger.add_evidence("http", "https://example.com/", "HTTP 200 text/html · server: cloudflare")
    task.ledger.add_finding("H", "HSTS ausente", "Resposta HTTPS sem Strict-Transport-Security.", [evidence.id])
    return task


@pytest.mark.parametrize("text", [
    "| Falha | Fonte |\n| --- | --- |\n| CVE-2026-99999 confirmada | E1 |",
    "## CVE-2026-99999 confirmada no alvo",
])
def test_cve_without_source_cannot_bypass_review_in_markdown_structure(text):
    revision = review_report(task_with_absent_hsts(), text)
    assert "CVE-2026-99999" not in revision.text and revision.removed == 1


def test_review_moves_contradiction_of_referenced_header_absence_to_hypotheses():
    text = "O cabeçalho HSTS está presente na resposta inspecionada [F1] [E1]."
    revision = review_report(task_with_absent_hsts(), text)
    assert not revision.text and revision.hypotheses == [text]
    valid = "O cabeçalho HSTS não está presente na resposta inspecionada [F1] [E1]."
    assert review_report(task_with_absent_hsts(), valid).text == valid
    task = task_with_absent_hsts()
    task.ledger.add_evidence("http", "https://example.com/login", "HTTP 200 text/html · strict-transport-security: max-age=31536000")
    comparison = "O HSTS está presente em https://example.com/login [E2], em contraste com o achado [F1]."
    assert review_report(task, comparison).text == comparison


def test_recommendation_does_not_make_an_unobserved_waf_a_fact():
    text = ("## Recomendações\n"
            "Manter verificações periódicas, pois a presença de um WAF (Cloudflare) pode ocultar configurações do servidor de origem.")
    revision = review_report(task_with_absent_hsts(), text)
    assert "presença de um WAF" not in revision.text and revision.hypotheses
    recommendation = "## Recomendações\nAvaliar a adoção de um WAF conforme o risco do serviço."
    assert review_report(task_with_absent_hsts(), recommendation).text == recommendation


def test_ledger_preserves_tool_body_hash_and_records_every_discovery_file(db, settings, public_dns):
    service, _, _, _ = make_env(db, settings, Site())
    message = last_assistant(service, run(collect(service, DEEP_MSG)))
    evidence = {e["url"]: e for e in message.meta["research"]["view"]["ledger"] if e["kind"] != "tls"}
    assert evidence["https://loja.example"]["sha256"] == hashlib.sha256(HOME.encode()).hexdigest()
    for path in ("/.well-known/security.txt", "/sitemap.xml"):
        assert evidence["https://loja.example" + path]["sha256"] == hashlib.sha256("não encontrado".encode()).hexdigest()


def test_review_preserves_description_of_passive_scope():
    text = "A análise focou em vulnerabilidades passivas configuráveis."
    assert review_report(task_with_absent_hsts(), text).text == text


@pytest.mark.parametrize("failure", ["unavailable", "invalid_json", "eof", "http_error"])
def test_provider_failures_never_persist_a_complete_research_report(db, settings, public_dns, failure):
    def handler(request):
        if failure == "unavailable":
            raise httpx.ConnectError("Ollama offline", request=request)
        if failure == "invalid_json":
            return httpx.Response(200, text="not-json\n")
        if failure == "http_error":
            return httpx.Response(500, text="failed")
        return httpx.Response(200, text=json.dumps({"message": {"content": "Texto sem fim"}, "done": False}) + "\n")
    service, _, _, _ = make_env(db, settings, Site())
    provider = service.provider = OllamaProvider("m", client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    async def go():
        try:
            return await collect(service, DEEP_MSG)
        finally:
            await provider.aclose()
    events = run(go())
    conv = next(e["conversation_id"] for e in events if e.get("conversation_id"))
    assert events[-1]["type"] == "error" and not any(e["type"] == "done" for e in events)
    assert not [m for m in service.repo.get_messages(conv) if m.role == "assistant"]
    assert service.research.mode(conv) == "quick" and service.research.current(conv).research_status == "error"


def test_continuation_error_is_not_masked_as_complete(db, settings, public_dns):
    class FailingContinuation(Cutter):
        async def stream_chat(self, messages, **options):
            if self.calls:
                raise ProviderError("falha na retomada")
            async for token in super().stream_chat(messages, **options):
                yield token
    service, _, _, _ = make_env(db, settings, Site())
    service.provider = FailingContinuation([("Resumo completo. Frase cortada ao mei", "length")])
    events = run(collect(service, DEEP_MSG))
    conv = next(e["conversation_id"] for e in events if e.get("conversation_id"))
    assert events[-1] == {"type": "error", "message": "falha na retomada"}
    assert not [m for m in service.repo.get_messages(conv) if m.role == "assistant"]
    assert service.research.mode(conv) == "quick"


def test_reloading_real_page_cancels_inflight_report(browser, tmp_path):
    import uvicorn
    from backend.tools.base import ToolResult

    settings = make_settings(tmp_path, tools={"enabled": True, "permission_modes": {"net.fetch": "allow"}})
    app = create_app(settings)
    lifespan = app.router.lifespan_context
    state = {}
    class WaitingReporter(Reporter):
        async def stream_chat(self, *args, **kwargs):
            state["generating"] = True
            try:
                await asyncio.Event().wait()
                yield "nunca"
            finally:
                state["stopped"] = True
    @asynccontextmanager
    async def controlled(application):
        async with lifespan(application):
            state["loop"] = asyncio.get_running_loop()
            service = state["service"] = application.state.chat_service
            service.provider = WaitingReporter()
            async def inspect(_conv, _task, url):
                return ToolResult(ok=True, data={"final_url": url, "status": 200, "content_type": "text/html", "headers": {}, "signals": {}, "body": "test"})
            service._inspect_tool = inspect
            yield
    app.router.lifespan_context = controlled
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    page = browser.new_page()
    conv = None
    try:
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        page.goto(f"http://127.0.0.1:{port}")
        page.wait_for_function("() => state.tools.length > 0")
        page.fill("#input", "Faça uma investigação estendida da segurança do site https://example.com/")
        page.keyboard.press("Enter")
        page.wait_for_function("() => document.querySelector('.research-card .rc-step[aria-current=step]')?.textContent === 'Preparing report'", timeout=5000)
        for _ in range(100):
            if state.get("generating"):
                break
            page.wait_for_timeout(10)
        assert state.get("generating")
        service = state["service"]
        conv = next(iter(service.research._current))
        page.reload()
        for _ in range(30):
            if service.research.mode(conv) == "quick" and state.get("stopped"):
                break
            page.wait_for_timeout(50)
        assert service.research.mode(conv) == "quick" and state.get("stopped")
        assert not service._active_turns and not [m for m in service.repo.get_messages(conv) if m.role == "assistant"]
    finally:
        if state.get("service"):
            for active_conv in tuple(state["service"].research._current):
                state["loop"].call_soon_threadsafe(state["service"].cancel_research, active_conv)
        page.close()
        server.should_exit = True
        thread.join(timeout=5)

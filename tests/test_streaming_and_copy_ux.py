"""Streaming progressivo (chat normal e Deep Research) e botão "Copiar resposta" no rodapé da mensagem."""

import asyncio
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from backend.main import create_app
from backend.providers.base import ModelProvider, ProviderError
from backend.research.progressive import ProgressiveReport, safe_cut
from tests.helpers import make_settings
from tests.test_deep_research import DEEP_MSG, Site, collect, last_assistant, make_env, public_dns, run  # noqa: F401
from tests.test_deep_research_final import task_with_absent_hsts
from tests.test_report_quality import Cutter, _service_with
from tests.test_calibration_and_chat_ux import FAKE_STREAM, sample_view  # noqa: F401

sync_api = pytest.importorskip("playwright.sync_api")
SHOTS = Path("docs/screenshots")
GATE = object()


def apply(events):
    """Reconstrói o texto exibido ao usuário: `token` acrescenta, `replace` substitui."""
    shown = ""
    for event in events:
        if event["type"] == "token":
            shown += event["content"]
        elif event["type"] == "replace":
            shown = event["content"]
    return shown


# ------------------------------------------------------------------ unidades: o que pode ser liberado

def test_safe_cut_only_releases_complete_blocks_outside_code_and_never_a_lone_heading():
    assert safe_cut("Primeiro parágrafo incompleto") == 0
    assert safe_cut("Parágrafo.\n\nSegundo em curso") == len("Parágrafo.\n\n")
    assert safe_cut("## Título\n\n") == 0  # título sozinho espera o seu parágrafo
    assert safe_cut("## Título\n\nCorpo.\n\n") == len("## Título\n\nCorpo.\n\n")
    code = "Antes.\n\n```py\na = 1\n\nb = 2\n"
    assert safe_cut(code) == len("Antes.\n\n")  # linha em branco dentro do código não é limite
    assert safe_cut(code + "```\n\n") == len(code + "```\n\n")


def test_progressive_report_reviews_before_release_and_replaces_explicitly():
    live = ProgressiveReport(task_with_absent_hsts())
    assert live.feed("## Resumo\n\nA análise inspecionou a página [E1].") == []  # ainda sem bloco fechado
    events = live.feed("## Resumo\n\nA análise inspecionou a página [E1].\n\nA seguir.")
    assert apply(events) == "## Resumo\n\nA análise inspecionou a página [E1]."
    assert all(e["type"] == "token" for e in events)
    hostile = "## Resumo\n\nA análise inspecionou a página [E1].\n\nCVE-2026-99999 foi confirmada no alvo em produção [E1].\n\n"
    assert "CVE-2026-99999" not in apply(live.feed(hostile))  # afirmação sem suporte nunca é liberada
    assert [e["type"] for e in live.show("Outro texto")] == ["replace"]  # divergência: substituição explícita, nunca silenciosa
    assert live.shown == "Outro texto"


# ------------------------------------------------------------------ Deep Research: serviço

REPORT = (
    "## Resumo executivo\n\nA CSP está ausente [F1] conforme [E1].\n\n"
    "## Detalhes\n\nO site foi inspecionado de forma passiva [E1].\n\n"
    "## Recomendações\n\n- Implementar uma CSP restritiva.\n"
)


class Gated(Cutter):
    """Gera o relatório em duas partes: a segunda só sai quando o teste libera (prova a geração em andamento)."""

    def __init__(self, first, second):
        super().__init__([(first + second, "stop")])
        self.first, self.second, self.gate, self.finished = first, second, asyncio.Event(), False

    async def stream_chat(self, messages, **options):
        self.calls.append([dict(m) for m in messages])
        self.seen.append(options)
        self.last_finish_reason = "stop"
        for i in range(0, len(self.first), 16):
            yield self.first[i : i + 16]
        await self.gate.wait()
        for i in range(0, len(self.second), 16):
            yield self.second[i : i + 16]
        self.finished = True


def test_deep_research_releases_reviewed_blocks_while_the_model_is_still_generating(db, settings, public_dns):
    service, _, _, _ = make_env(db, settings, Site())
    provider = service.provider = Gated("## Resumo executivo\n\nA CSP está ausente [F1] conforme [E1].\n\nSegue o detalhe", " do achado [E1].\n")

    async def scenario():
        events, early = [], []
        async for event in service.send_stream(None, DEEP_MSG):
            events.append(event)
            if event["type"] == "token" and not provider.gate.is_set():
                early.append(event)
                assert not provider.finished  # o primeiro trecho chegou ANTES do fim da geração
                provider.gate.set()
        return events, early

    events, early = run(asyncio.wait_for(scenario(), 20))
    assert early and "CSP está ausente" in apply(early) and "Segue o detalhe" not in apply(early)
    message = last_assistant(service, events)
    assert apply(events) == message.content  # texto exibido == texto salvo
    assert [e["type"] for e in events][-1] == "done"


def test_unreviewed_claims_are_never_published_nor_saved(db, settings, public_dns):
    bad = "CVE-2026-99999 foi confirmada no alvo em produção [E1].\n\n"
    service, _, _, _ = make_env(db, settings, Site())
    service.provider = Cutter([("## Resumo\n\nA CSP está ausente [F1] conforme [E1].\n\n" + bad + "Conclusão final do relatório [E1].\n", "stop")])
    events = run(collect(service, DEEP_MSG))
    shown = apply(events)
    assert "CVE-2026-99999" not in "".join(e.get("content", "") for e in events if e["type"] in ("token", "replace"))
    assert shown == last_assistant(service, events).content and "Conclusão final" in shown


def test_truncated_report_continues_and_displayed_text_equals_saved(db, settings, public_dns):
    body = "## Resumo executivo\n\nA CSP está ausente [F1] conforme [E1].\n\n## Recomendações\n\n- Implementar uma CSP restritiva (Content-Security-P"
    service = _service_with(db, settings, [(body, "length"), ("olicy) e revisar a política regularmente.\n", "stop")])
    events = run(collect(service, DEEP_MSG))
    message = last_assistant(service, events)
    assert "olicy) e revisar" in message.content and apply(events) == message.content
    assert message.meta["research"]["completion"]["continuations"] == 1


def test_partial_report_shows_the_same_text_that_is_saved_with_explicit_replacement(db, settings, public_dns):
    body = "## Resumo executivo\n\nA CSP está ausente [F1] conforme [E1].\n\n## Detalhes\n\nA configuração observada foi considerada e depois a frase corta no mei"
    service = _service_with(db, settings, [(body, "length")], report_continuations=0)
    events = run(collect(service, DEEP_MSG))
    message = last_assistant(service, events)
    assert message.meta["research"]["completion"]["status"] == "partial"
    assert apply(events) == message.content and "corta no mei" not in message.content


def test_cancel_during_report_keeps_nothing_and_publishes_only_reviewed_text(db, settings, public_dns):
    service, _, _, _ = make_env(db, settings, Site())
    provider = service.provider = Gated("## Resumo\n\nA CSP está ausente [F1] conforme [E1].\n\nTexto em curso", " fim.\n")

    async def scenario():
        events = []
        async for event in service.send_stream(None, DEEP_MSG):
            events.append(event)
            if event["type"] == "token":
                conv = next(iter(service.research._current))
                service.cancel_research(conv)
        return events, conv

    events, conv = run(asyncio.wait_for(scenario(), 20))
    assert not any(e["type"] == "done" for e in events)
    assert "Texto em curso" not in apply(events)
    assert not [m for m in service.repo.get_messages(conv) if m.role == "assistant"]


def test_provider_error_midway_keeps_error_event_and_saves_no_report(db, settings, public_dns):
    class Failing(Cutter):
        async def stream_chat(self, messages, **options):
            yield "## Resumo\n\nA CSP está ausente [F1] conforme [E1].\n\nSegue"
            raise ProviderError("Ollama caiu")

    service, _, _, _ = make_env(db, settings, Site())
    service.provider = Failing([])
    events = run(collect(service, DEEP_MSG))
    conv = next(e["conversation_id"] for e in events if e.get("conversation_id"))
    assert events[-1] == {"type": "error", "message": "Ollama caiu"}
    assert not [m for m in service.repo.get_messages(conv) if m.role == "assistant"]


# ------------------------------------------------------------------ navegador real

class GatedProvider(ModelProvider):
    """Provider de chat normal: devolve as partes e fica parado em cada GATE até o teste liberar."""

    name = "gated"

    def __init__(self, parts, fail_after=None):
        super().__init__("gated-1b")
        self.parts, self.fail_after = parts, fail_after
        self.gate, self.finished = threading.Event(), False

    async def chat(self, messages, **options):
        return "".join(p for p in self.parts if p is not GATE)

    async def stream_chat(self, messages, **options):
        self.finished = False
        for part in self.parts:
            if part is GATE:
                for _ in range(1000):
                    if self.gate.is_set():
                        break
                    await asyncio.sleep(0.02)
                if self.fail_after:
                    raise ProviderError("Ollama indisponível (teste)")
                continue
            await asyncio.sleep(0.005)
            yield part
        self.finished = True

    async def health(self):
        return {"available": True}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    app = create_app(make_settings(tmp_path_factory.mktemp("streamweb")))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    yield app, f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as pw:
        try:
            br = pw.chromium.launch(channel="chrome")
        except Exception:
            pytest.skip("Chrome indisponível para Playwright")
        yield br
        br.close()


CLIPBOARD = """
    window.__copied = [];
    Object.defineProperty(navigator, 'clipboard', { configurable: true, value: { writeText: async (t) => { window.__copied.push(t); } } });
"""
VIEWPORTS = [(1366, 768), (1920, 1080), (390, 844)]


def open_chat(browser, server, provider, viewport):
    app, url = server
    app.state.chat_service.provider = provider
    page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
    page.add_init_script(CLIPBOARD)
    page.goto(url)
    page.wait_for_function("() => window.EnslavedMarkdown")
    return page


def ask(page, text="Pergunta"):
    page.fill("#input", text)
    page.keyboard.press("Enter")


def footer_is_last(page):
    return page.evaluate("""() => { const m = [...document.querySelectorAll('#messages .message.assistant')].pop();
        return m.lastElementChild.classList.contains('msg-footer') && m.querySelectorAll('.copy-btn').length === 1; }""")


@pytest.mark.parametrize("viewport", VIEWPORTS, ids=["1366x768", "1920x1080", "390x844"])
def test_chat_streams_real_deltas_and_copy_button_stays_in_the_footer(browser, server, viewport):
    long_tail = "\n".join(f"- item {i} da lista" for i in range(60))
    parts = ["Olá, ", "**mundo**!\n\n", "```py\nprint(1)", GATE, "\n```\n\n", long_tail, "\n\nFim."]
    full = "".join(p for p in parts if p is not GATE)
    provider = GatedProvider(parts)
    page = open_chat(browser, server, provider, viewport)
    try:
        ask(page)
        page.wait_for_function("() => document.querySelector('#messages .message.assistant .content pre code')")
        assert not provider.finished  # o primeiro delta apareceu antes do fim da geração
        content = page.locator("#messages .message.assistant .content").last
        assert "Olá, mundo!" in content.inner_text() and content.locator("strong").inner_text() == "mundo"  # Markdown já durante o stream
        assert "streaming" in content.get_attribute("class")
        button = page.locator("#messages .message.assistant .copy-btn").last
        assert button.is_disabled() and footer_is_last(page)  # desabilitado enquanto responde; já no rodapé
        assert page.evaluate("document.querySelectorAll('#messages .message.assistant > header .copy-btn').length") == 0
        provider.gate.set()
        page.wait_for_function("() => !document.querySelector('.content.streaming')", timeout=15000)
        assert provider.finished and not button.is_disabled() and footer_is_last(page)
        button.click()
        page.wait_for_function("() => window.__copied.length === 1")
        assert page.evaluate("window.__copied[0]") == full  # Markdown completo, sem metadados
        box = page.evaluate("""() => { const f = document.querySelector('#messages .message.assistant:last-child .msg-footer').getBoundingClientRect();
            const c = document.querySelector('#messages .message.assistant:last-child .content').getBoundingClientRect();
            return {fy: f.y, cb: c.bottom, fr: f.right, w: innerWidth}; }""")
        assert box["fy"] >= box["cb"] - 1 and box["fr"] <= box["w"] + 1  # abaixo do texto e dentro da tela
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        page.screenshot(path=str(SHOTS / f"streaming-copy-footer-{viewport[0]}x{viewport[1]}.png"))
        page.reload()  # histórico recarregado: mesmo rodapé, mesmo texto
        page.locator("#conversation-list button, #conversation-list li, #conversation-list a").first.dispatch_event("click")
        page.wait_for_selector("#messages .message.assistant .copy-btn")
        assert footer_is_last(page) and page.locator("#messages .message.assistant .content").last.get_attribute("data-raw") == full
        page.locator("#messages .message.assistant .copy-btn").last.click()
        page.wait_for_function("() => window.__copied.length === 1")
        assert page.evaluate("window.__copied[0]") == full
    finally:
        page.close()


def test_user_copy_button_stays_independent(browser, server):
    provider = GatedProvider(["Resposta."])
    page = open_chat(browser, server, provider, (1366, 768))
    try:
        ask(page, "texto do usuário")
        page.wait_for_function("() => document.querySelector('.message.assistant .copy-btn') && !document.querySelector('.content.streaming')")
        page.locator("#messages .message.user .copy-btn").last.click()
        page.wait_for_function("() => window.__copied.length === 1")
        assert page.evaluate("window.__copied[0]") == "texto do usuário"
    finally:
        page.close()


def test_autoscroll_follows_only_while_the_user_is_at_the_bottom(browser, server):
    block = "\n\n".join(f"Parágrafo {i} da resposta longa para forçar rolagem." for i in range(40))
    for reading_above in (False, True):
        provider = GatedProvider([block, "\n\n", GATE, block.replace("Parágrafo", "Continuação")])
        page = open_chat(browser, server, provider, (1366, 768))
        try:
            ask(page)
            page.wait_for_function("() => document.querySelector('#messages .message.assistant .content')?.dataset.raw?.includes('Parágrafo 39')")
            page.wait_for_function("() => (() => { const m = document.getElementById('messages'); return m.scrollHeight - m.scrollTop - m.clientHeight < 80; })()")
            if reading_above:
                page.evaluate("document.getElementById('messages').scrollTop = 0")
            provider.gate.set()
            page.wait_for_function("() => !document.querySelector('.content.streaming')", timeout=15000)
            top = page.evaluate("document.getElementById('messages').scrollTop")
            near = page.evaluate("(() => { const m = document.getElementById('messages'); return m.scrollHeight - m.scrollTop - m.clientHeight < 80; })()")
            assert (top < 50 and not near) if reading_above else near  # leitura anterior não é arrastada; no fim, acompanha
            assert page.evaluate("document.getElementById('composer-wrap').getBoundingClientRect().bottom <= innerHeight + 1")  # composer ancorado
        finally:
            page.close()


def test_cancel_stops_the_stream_and_releases_the_composer(browser, server):
    provider = GatedProvider(["Início da resposta", GATE, " que nunca chega."])
    page = open_chat(browser, server, provider, (1366, 768))
    try:
        ask(page)
        page.wait_for_function("() => document.querySelector('#messages .message.assistant .content')?.dataset.raw?.includes('Início')")
        page.click("#stop")
        page.wait_for_function("() => !document.getElementById('composer').classList.contains('busy')", timeout=10000)
        assert not page.locator("#input").is_disabled()
        assert "nunca chega" not in page.inner_text("#messages")
        assert page.locator("#messages .message.assistant .content.streaming").count() == 0
        provider.gate.set()
    finally:
        page.close()


def test_provider_error_midway_shows_error_and_no_partial_bubble(browser, server):
    provider = GatedProvider(["Começo da resposta", GATE], fail_after=True)
    page = open_chat(browser, server, provider, (1366, 768))
    try:
        ask(page)
        page.wait_for_function("() => document.querySelector('#messages .message.assistant .content')?.dataset.raw?.includes('Começo')")
        provider.gate.set()
        page.wait_for_selector("#messages .message.error", timeout=10000)
        assert "Ollama indisponível" in page.inner_text("#messages .message.error")
        assert page.locator("#messages .message.assistant").count() == 0
        assert not page.locator("#input").is_disabled()
    finally:
        page.close()


def test_replace_event_swaps_the_displayed_text_and_footer_follows_the_evidence_panel(browser, server):
    app, url = server
    page = browser.new_page(viewport={"width": 1366, "height": 768})
    page.add_init_script(FAKE_STREAM)
    page.add_init_script(CLIPBOARD)
    try:
        page.goto(url)
        page.wait_for_function("() => Boolean(window.EnslavedResearch)")
        page.fill("#input", "Investigue https://alvo.example/")
        page.keyboard.press("Enter")
        page.wait_for_function("() => Boolean(window.__ctl)")
        page.evaluate("window.__ctl.push({type: 'token', content: '## Parcial\\n\\nTexto antigo.'})")
        page.wait_for_function("() => document.querySelector('#messages .message.assistant .content')?.innerText.includes('Texto antigo')")
        btn = page.locator("#messages .message.assistant .msg-footer .copy-btn")
        assert btn.is_disabled()  # relatório em andamento não pode ser copiado como se estivesse completo
        page.evaluate("window.__ctl.push({type: 'replace', content: '## Final\\n\\nTexto corrigido.'})")
        page.wait_for_function("() => document.querySelector('#messages .message.assistant .content')?.innerText.includes('Texto corrigido')")
        assert "Texto antigo" not in page.inner_text("#messages .message.assistant .content")
        view = sample_view()
        page.evaluate("([v]) => { window.__ctl.push({type: 'done', conversation_id: 'c1', message_id: 'm1', research: {view: v, completion: v.completion}}); window.__ctl.close(); }", [view])
        page.wait_for_selector("#messages .research-panel")
        assert page.evaluate("""() => { const m = document.querySelector('#messages .message.assistant');
            const kids = [...m.children]; return kids.indexOf(m.querySelector('.research-panel')) < kids.indexOf(m.querySelector('.msg-footer')) && m.lastElementChild.classList.contains('msg-footer'); }""")
        btn.click()
        page.wait_for_function("() => window.__copied.length === 1")
        assert page.evaluate("window.__copied[0]") == "## Final\n\nTexto corrigido."
        page.screenshot(path=str(SHOTS / "streaming-copy-footer-research-1366x768.png"))
    finally:
        page.close()


def test_streaming_render_keeps_finished_blocks_untouched(browser, server):
    page = open_chat(browser, server, GatedProvider(["x"]), (1366, 768))
    try:
        same = page.evaluate(r"""() => {
            const el = document.createElement('div');
            patchContent(el, 'Primeiro.\n\n```py\na = 1');
            const [first, code] = [el.children[0], el.children[1]];
            patchContent(el, 'Primeiro.\n\n```py\na = 1\nb = 2\n```\n\nTerceiro.');
            return [el.children[0] === first, el.children[1] === code, el.children.length, el.dataset.raw.endsWith('Terceiro.')];
        }""")
        assert same == [True, False, 3, True]  # parágrafo concluído intacto; só o bloco em curso é trocado
    finally:
        page.close()

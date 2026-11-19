"""UI da investigação no navegador real (Chrome/Playwright): cartão de progresso centralizado no chat e painel de evidências navegável.

O stream do chat é substituído por um controlador local (eventos empurrados pelo teste); o `view` das evidências vem do backend real."""

import json
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from backend.main import create_app
from backend.research.report import evidence_view
from backend.research.state import ResearchRegistry
from tests.helpers import make_settings

sync_api = pytest.importorskip("playwright.sync_api")

FAKE_STREAM = """
(() => {
  const realFetch = window.fetch.bind(window);
  window.fetch = (url, opts = {}) => {
    if (!String(url).includes('/api/chat/stream')) return realFetch(url, opts);
    const enc = new TextEncoder();
    let ctl;
    const body = new ReadableStream({ start(c) { ctl = c; } });
    if (opts.signal) opts.signal.addEventListener('abort', () => { try { ctl.error(new DOMException('Aborted', 'AbortError')); } catch (e) {} });
    window.__ctl = { push: (o) => ctl.enqueue(enc.encode('data: ' + JSON.stringify(o) + '\\n\\n')), close: () => ctl.close() };
    return Promise.resolve(new Response(body, { status: 200, headers: { 'Content-Type': 'text/event-stream' } }));
  };
})();
"""

REPORT = ("## Resumo executivo\nForam examinadas 9 páginas; nenhuma vulnerabilidade foi comprovada.\n\n"
          "## Principais descobertas\n[H][F1] HSTS ausente — sem Strict-Transport-Security na resposta de https://a.example/ [E10]\n"
          "- **Limitações:** só a ausência do cabeçalho foi observada (veja F1 e E1).\n")


def sample_view(partial=False):
    task = ResearchRegistry().begin("c", "deep", {"goal": "g", "profile": "security", "targets": ["https://a.example"]})
    led = task.ledger
    for i in range(9):  # 9 observações TLS equivalentes (uma por página)
        led.add_evidence("tls", f"https://a.example/p{i}", "TLS TLSv1.3 TLS_AES_256_GCM_SHA384 · verificado=True · emissor=CA Teste · expira=2031-01-01 · ok")
        led.pages.add(f"https://a.example/p{i}")
    e = led.add_evidence("http", "https://a.example/", "HTTP 200 text/html · server: nginx; content-security-policy: default-src 'self'", body=b"<html>")
    led.add_evidence("http", "https://a.example/x", "<img src=x onerror=window.__xss=1> HTTP 200 text/html", body=b"y")
    led.add_finding("H", "HSTS ausente", "Resposta HTTPS sem Strict-Transport-Security.", [e.id], severity="baixa", recommendation="Enviar HSTS.")
    led.add_finding("D", "<img src=x onerror=window.__xss=1>", "Item não verificado.", [])
    return evidence_view(task, {"status": "partial" if partial else "complete", "finish_reason": "length" if partial else "stop", "continuations": 2 if partial else 0})


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    app = create_app(make_settings(tmp_path_factory.mktemp("rweb")))
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
    yield f"http://127.0.0.1:{port}"
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


def open_page(browser, url, width, height):
    page = browser.new_page(viewport={"width": width, "height": height})
    page.add_init_script(FAKE_STREAM)
    page.goto(url)
    page.wait_for_function("window.EnslavedResearch && window.EnslavedMarkdown")
    return page


def start(page, text="Faça uma investigação profunda do site https://a.example"):
    page.fill("#input", text)
    page.keyboard.press("Enter")
    page.wait_for_function("window.__ctl")
    page.evaluate("window.__ctl.push({type: 'meta', conversation_id: 'c1', conversation_title: 'Teste', context: {estimated_tokens: 10, input_budget: 100, usage_ratio: 0.1}})")


def progress(page, phase, message, pages=0, requests=0, evidence=0):
    page.evaluate("(e) => window.__ctl.push(Object.assign({type: 'research_progress', task_id: 't1'}, e))",
                  {"phase": phase, "message": message, "pages": pages, "requests": requests, "evidence": evidence})


def box(page, selector):
    return page.evaluate("(s) => { const r = document.querySelector(s).getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height, right: r.right, bottom: r.bottom}; }", selector)


@pytest.mark.parametrize("viewport", [(1280, 800), (375, 812)], ids=["desktop", "mobile"])
def test_progress_card_is_centered_in_the_message_column_and_never_overlaps(browser, server, viewport):
    page = open_page(browser, server, *viewport)
    try:
        start(page)
        progress(page, "collecting", "Consultando loja.example", 3, 5, 6)
        page.wait_for_selector("#messages .research-card")
        card, user, composer, sidebar = box(page, "#messages .research-card"), box(page, "#messages .message.user"), box(page, "#composer"), box(page, "#sidebar")
        assert abs(card["x"] - user["x"]) <= 1 and abs(card["w"] - user["w"]) <= 1  # mesma coluna e largura das mensagens
        assert page.evaluate("getComputedStyle(document.querySelector('.research-card')).position") != "fixed"
        assert card["bottom"] <= composer["y"] + 1  # não cobre o composer
        if viewport[0] > 700:  # desktop: o cartão não invade a barra lateral e fica centralizado na área do chat
            assert card["x"] >= sidebar["right"] - 1
            chat = box(page, "#messages")
            assert abs((card["x"] + card["w"] / 2) - (chat["x"] + chat["w"] / 2)) <= 12
        else:
            assert card["x"] >= 0 and card["right"] <= viewport[0] + 1 and page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        assert not page.locator("#input").is_disabled()  # composer livre
        page.fill("#input", "digitando durante a investigação")
        assert page.input_value("#input") == "digitando durante a investigação"
        assert "Querying sources" in page.inner_text("#messages .research-card")
        assert "3 pages" in page.inner_text("#messages .research-card") and "5 requests" in page.inner_text("#messages .research-card")
        assert page.evaluate("document.getElementById('infer-state').textContent").find("Investigation") < 0  # nada de aviso solto fora do fluxo
        page.screenshot(path=str(Path("docs/screenshots") / f"research-card-{'mobile' if viewport[0] < 500 else 'desktop'}.png"))
    finally:
        page.close()


def test_progress_updates_the_same_card_in_order_and_ends_cleanly(browser, server):
    page = open_page(browser, server, 1280, 800)
    try:
        start(page)
        state = lambda: page.evaluate("[...document.querySelectorAll('.research-card .rc-step')].map(li => li.dataset.state)")  # noqa: E731
        progress(page, "planning", "Planejando 1 alvo(s)")
        page.wait_for_selector(".research-card")
        assert state() == ["active", "pending", "pending", "pending", "pending"]
        progress(page, "collecting", "Consultando página(s)", 2, 4, 5)
        progress(page, "analyzing", "Analisando evidências", 9, 12, 20)
        progress(page, "collecting", "evento atrasado não regride a etapa")
        page.wait_for_function("() => document.querySelector('.research-card .rc-step[data-state=active]')?.textContent === 'Analyzing evidence'")
        assert state() == ["done", "done", "active", "pending", "pending"]
        assert page.locator("#messages .research-card").count() == 1  # nunca duplica
        progress(page, "verifying", "Verificando versões e fontes públicas", 9, 12, 20)
        page.evaluate("window.__ctl.push({type: 'research_summary', pages: 9, requests: 12, evidence: 20, findings: {A: 0, H: 3, B: 0, C: 1, D: 1}})")
        progress(page, "reporting", "Preparando resposta", 9, 12, 20)
        page.wait_for_function("() => document.querySelector('.research-card .rc-step[data-state=active]')?.textContent === 'Preparing report'")
        assert state() == ["done", "done", "done", "done", "active"]
        assert page.get_attribute(".research-card", "aria-busy") == "true" and page.get_attribute(".research-card .rc-step[data-state=active]", "aria-current") == "step"
        assert "9 pages · 12 requests · 20 pieces of evidence" in page.inner_text(".research-card")
        assert page.get_attribute(".research-card [role=status]", "aria-live") == "polite"
        page.evaluate("window.__ctl.push({type: 'token', content: 'Relatório pronto.'}); window.__ctl.push({type: 'done', conversation_id: 'c1', message_id: 'm1'}); window.__ctl.close()")
        page.wait_for_function("() => document.querySelector('.research-card')?.dataset.state === 'done'")
        assert state() == ["done"] * 5 and page.get_attribute(".research-card", "aria-busy") == "false" and page.locator("#messages .research-card").count() == 1
        assert page.locator("#messages .message.assistant .content").last.get_attribute("data-raw") == "Relatório pronto."
    finally:
        page.close()


def test_cancelled_or_dropped_stream_never_leaves_the_card_running(browser, server):
    page = open_page(browser, server, 1280, 800)
    try:
        start(page)
        progress(page, "collecting", "Consultando fontes", 1, 2, 2)
        page.wait_for_selector(".research-card")
        page.click("#stop")
        page.wait_for_function("() => document.querySelector('.research-card')?.dataset.state === 'interrupted'")
        assert "interrupted" in page.inner_text(".research-card").lower() and page.get_attribute(".research-card", "aria-busy") == "false"
        assert page.locator(".research-card .rc-step[data-state=stopped]").count() == 1 and not page.locator("#input").is_disabled()
    finally:
        page.close()


def finish_with_report(page, view, content=REPORT):
    page.evaluate("([c, v]) => { window.__ctl.push({type: 'token', content: c}); window.__ctl.push({type: 'done', conversation_id: 'c1', message_id: 'm1', research: {view: v, completion: v.completion}}); window.__ctl.close(); }", [content, view])
    page.wait_for_selector("#messages .research-panel")


def test_evidence_panel_groups_data_keeps_ledger_collapsed_and_links_findings_to_evidence(browser, server):
    page = open_page(browser, server, 1280, 800)
    try:
        start(page)
        progress(page, "reporting", "Preparando resposta", 9, 12, 11)
        view = sample_view()
        finish_with_report(page, view)
        panel = page.locator("#messages .research-panel")
        text = panel.inner_text()
        for label in ("HTTP evidence", "Certificates and TLS", "Headers", "Pages consulted", "Related findings"):
            assert label in text
        assert "equivalent across pages" in text  # 9 observações TLS → 1 item agrupado
        assert page.locator(".rp-group").first.get_attribute("open") is None and page.locator("details.rp-ledger").get_attribute("open") is None  # ledger recolhido
        page.click("details.rp-ledger > summary")
        assert page.locator(".rp-ledger-table tbody tr").count() == len(view["ledger"]) == 11  # nada descartado
        first = page.locator(".rp-ledger-table tbody tr").first
        assert first.locator(".rp-hash").inner_text() == "—" and "https://a.example/p0" in first.inner_text()
        assert page.locator(".rp-ledger-table tr[data-ref=E10] .rp-hash").inner_text() != "—"  # SHA-256 completo disponível
        # links no texto do relatório → painel
        content = page.locator("#messages .message.assistant .content").last
        links = content.locator("a.ref-link")
        assert [links.nth(i).inner_text() for i in range(links.count())] == ["F1", "E10", "F1", "E1"]  # F# e E# do texto ficaram navegáveis
        page.click("details.rp-ledger > summary")  # recolhe para provar que o link reabre
        content.locator("a.ref-link", has_text="E10").click()
        assert page.get_attribute("details.rp-ledger", "open") is not None
        page.wait_for_function("() => document.querySelector('.rp-ledger-table tr[data-ref=E10]').classList.contains('flash')")
        content.locator("a.ref-link", has_text="F1").first.click()
        page.wait_for_function("() => document.querySelector('.rp-finding[data-ref=F1]').classList.contains('flash')")
        # do achado para a evidência e da evidência para o achado
        page.locator(".rp-finding[data-ref=F1] a.ref-link", has_text="E10").click()
        page.wait_for_function("() => document.activeElement && document.activeElement.dataset.ref === 'E10'")
        page.locator(".rp-ledger-table tr[data-ref=E10] a.ref-link", has_text="F1").click()
        page.wait_for_function("() => document.activeElement && document.activeElement.dataset.ref === 'F1'")
        assert page.locator("#messages .research-panel table.rp-matrix").count() == 1  # tabela de cabeçalhos legível
        assert "{" not in page.inner_text(".rp-summary") and "<html" not in text  # sem JSON/HTML bruto
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")
        page.screenshot(path=str(Path("docs/screenshots") / "research-evidence-desktop.png"), full_page=False)
    finally:
        page.close()


def test_panel_is_safe_against_hostile_evidence_and_marks_partial_reports(browser, server):
    page = open_page(browser, server, 375, 812)
    try:
        start(page)
        progress(page, "reporting", "Preparando resposta", 1, 1, 1)
        finish_with_report(page, sample_view(partial=True), "Texto <script>window.__xss=1</script> [H][F1] interrompido.")
        page.click("details.rp-ledger > summary")
        assert page.evaluate("window.__xss") is None and page.locator("#messages .research-panel img, #messages .message.assistant .content script").count() == 0
        assert "<img src=x" in page.inner_text(".rp-ledger-table")  # exibido como TEXTO, nunca interpretado
        assert "Partial report" in page.inner_text(".rp-title") and page.get_attribute(".research-panel", "data-status") == "partial"
        assert page.get_attribute(".research-card", "data-state") == "partial" and "Partial" in page.inner_text(".research-card .rc-state")
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")  # tabelas rolam dentro do painel no celular
        page.screenshot(path=str(Path("docs/screenshots") / "research-evidence-mobile.png"))
    finally:
        page.close()


def test_saved_message_renders_the_same_panel_from_history_meta(browser, server):
    page = open_page(browser, server, 1280, 800)
    try:
        view = sample_view()
        page.evaluate("([c, v]) => { const b = messageBubble('assistant', c, { research: { view: v } }, 'c1'); document.getElementById('messages').append(b); }", [REPORT, view])
        assert page.locator("#messages .research-panel").count() == 1 and page.locator("#messages a.ref-link").count() >= 4
        assert page.locator("#messages .message.assistant .content").last.get_attribute("data-raw") == REPORT  # texto original intacto
        assert page.locator(".rp-ledger-table tbody tr").count() == len(view["ledger"])
        assert json.dumps(view)  # o mesmo objeto que o backend salvou em meta.research.view
    finally:
        page.close()

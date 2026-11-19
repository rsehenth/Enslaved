"""Regressões dirigidas: normalização de URLs, consistência F#->E#/404/8-de-9/prioridade/H/cobertura, scroll, composer e cópia."""

import json
import socket
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

from backend.main import create_app
from backend.providers.base import ModelProvider
from backend.research.evidence import Ledger
from backend.research.intent import canonicalize_url, classify, extract_targets, extract_urls
from backend.research.report import appendix, check_coverage, evidence_view, review_report
from backend.research.state import ResearchRegistry
from tests.helpers import make_settings
from tests.test_deep_research import Reporter, Site, collect, last_assistant, make_env, public_dns, run  # noqa: F401
from tests.test_research_ui import FAKE_STREAM, REPORT, sample_view

sync_api = pytest.importorskip("playwright.sync_api")


# ====================================================================== 1. URLs & Markdown


def test_url_extraction_strips_outer_markdown_and_deduplicates_quatorzeweb():
    variants = [
        "Investigue **https://alvo.example/** passivamente.",
        "Investigue https://alvo.example/ passivamente.",
        "Investigue [Site](https://alvo.example/) passivamente.",
        "Investigue `https://alvo.example/` passivamente.",
        "**Investigue https://alvo.example/**.",
    ]
    for msg in variants:
        assert extract_targets(msg) == ("https://alvo.example/",)
        assert classify(msg).targets == ("https://alvo.example/",)

    combined = (
        "Compare **https://alvo.example/** com https://alvo.example/ "
        "e [Site](https://alvo.example/) e `https://alvo.example/` "
        "e HTTPS://Alvo.Example:443/"
    )
    assert extract_targets(combined) == ("https://alvo.example/",)


def test_url_canonicalization_preserves_valid_queries_and_special_characters():
    msg = (
        "Veja **https://alvo.example/busca?q=a*b&tag=foo_bar&v=~1** "
        "e [Wiki](https://en.wikipedia.org/wiki/Foo_(bar)) "
        "e `https://alvo.example/rota_1/a*b?x=1%202`"
    )
    urls = extract_urls(msg)
    assert urls == (
        "https://alvo.example/busca?q=a*b&tag=foo_bar&v=~1",
        "https://en.wikipedia.org/wiki/Foo_(bar)",
        "https://alvo.example/rota_1/a*b?x=1%202",
    )
    assert canonicalize_url("**HTTPS://Alvo.Example:443/a/../b?q=x*y_z#frag**") == "https://alvo.example/b?q=x*y_z"


def test_bold_markdown_prompt_does_not_request_404_star_star_route(db, settings, public_dns):
    site = Site(
        pages={"/": '<html><head><title>Quatorze</title></head><body><a href="/sobre">Sobre</a></body></html>'},
        headers={"x-content-type-options": "nosniff", "referrer-policy": "no-referrer"},
    )
    service, _prov, _p, _e = make_env(db, settings, site)
    msg = last_assistant(service, run(collect(service, "Faça uma investigação profunda da segurança do site **https://alvo.example/**")))
    assert "/**" not in site.paths()
    ledger = msg.meta["research"]["view"]["ledger"]
    assert all(not e["url"].endswith("/**") for e in ledger)
    assert ledger[0]["url"] == "https://alvo.example/"
    assert all("E1" in f["evidence"] for f in msg.meta["research"]["view"]["findings"] if f["category"] == "H")


# ====================================================================== 2. 404 vs 200 & 8/9 headers & consistency


def test_404_page_is_separated_from_200_and_never_receives_global_findings(db, settings, public_dns):
    base_headers = {
        "content-type": "text/html; charset=utf-8",
        "x-content-type-options": "nosniff",
        "referrer-policy": "strict-origin-when-cross-origin",
    }
    pages = {
        "/": '<html><body><a href="/sobre">Sobre</a> <a href="/faltando">Faltando</a></body></html>',
        "/sobre": "<html><body><p>Sobre</p></body></html>",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path in pages:
            return httpx.Response(200, text=pages[path], headers=base_headers)
        return httpx.Response(404, text="<html><body>Não encontrado</body></html>", headers={"content-type": "text/html"})

    service, _prov, _p, _e = make_env(db, settings, handler)
    msg = last_assistant(service, run(collect(service, "Faça uma investigação profunda da segurança de https://alvo.example/")))
    view = msg.meta["research"]["view"]
    led = next(iter(service.research._tasks.values())).ledger
    # Somente as 2 páginas HTTP 200 contam como páginas analisadas
    assert len(led.pages) == 2
    assert "https://alvo.example/faltando" not in led.pages
    assert any(p["url"].endswith("/faltando") and p["status"] == 404 for p in led.coverage["error_pages"])
    # Nenhuma descoberta H aponta para a evidência da rota 404
    e_404 = {e["id"] for e in view["ledger"] if e["url"].endswith("/faltando")}
    assert e_404
    for f in view["findings"]:
        if f["category"] == "H":
            assert not (set(f["evidence"]) & e_404)
            assert set(f["evidence"]) == {"E1", "E5"} or "E1" in f["evidence"]


def test_all_404_target_does_not_fabricate_global_header_findings(db, settings, public_dns):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="<html><body>404</body></html>", headers={"content-type": "text/html"})

    service, _prov, _p, _e = make_env(db, settings, handler)
    msg = last_assistant(service, run(collect(service, "Faça uma investigação profunda da segurança de https://alvo.example/nao-existe")))
    view = msg.meta["research"]["view"]
    cats = [f["category"] for f in view["findings"]]
    assert "H" not in cats
    assert any(f["category"] == "D" and "200" in f["title"] for f in view["findings"])


def test_partial_headers_8_of_9_pages_matches_matrix_and_reports_affected_endpoints(db, settings, public_dns):
    links = "".join(f'<a href="/p{i}">P{i}</a> ' for i in range(1, 9))
    full_headers = {
        "content-type": "text/html; charset=utf-8",
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
    }
    legacy_headers = {"content-type": "text/html; charset=utf-8"}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/":
            return httpx.Response(200, text=f"<html><body>{links}</body></html>", headers=full_headers)
        if path == "/p8":
            return httpx.Response(200, text="<html><body>Legado</body></html>", headers=legacy_headers)
        if path.startswith("/p") and path[2:].isdigit() and 1 <= int(path[2:]) <= 7:
            return httpx.Response(200, text=f"<html><body>P{path[2:]}</body></html>", headers=full_headers)
        return httpx.Response(404, text="404", headers={"content-type": "text/plain"})

    service, _prov, _p, _e = make_env(db, settings, handler, research=None)
    # Garantir max_pages >= 9 para cobrir todas as 9 páginas (home + /p1../p8)
    service.research_settings.deep.max_pages = 10
    msg = last_assistant(service, run(collect(service, "Faça uma investigação profunda da segurança de https://alvo.example/")))
    view = msg.meta["research"]["view"]
    matrix = view["header_matrix"]
    assert len(matrix["rows"]) == 9
    assert sum(r["present"]["xcto"] for r in matrix["rows"]) == 8
    assert sum(r["present"]["referrer"] for r in matrix["rows"]) == 8
    hard = next(f for f in view["findings"] if f["title"] == "Cabeçalhos de hardening ausentes")
    assert "8/9" in hard["detail"] and "/p8" in hard["detail"]
    # Consistente também como achado B explícito para X-Content-Type-Options e Referrer-Policy
    b_titles = {f["title"]: f for f in view["findings"] if f["category"] == "B"}
    assert "X-Content-Type-Options aplicado de forma inconsistente" in b_titles
    assert "Referrer-Policy aplicado de forma inconsistente" in b_titles
    assert "/p8" in b_titles["X-Content-Type-Options aplicado de forma inconsistente"]["detail"]
    assert "8/9" in b_titles["Referrer-Policy aplicado de forma inconsistente"]["detail"]


# ====================================================================== 3. Prioridade, H como observação, metodologia e cobertura


def test_review_enforces_consistent_priority_and_H_as_observation_and_keeps_methodology():
    task = ResearchRegistry().begin("c", "deep", {"goal": "g", "profile": "security", "targets": ["https://alvo.example/"]})
    ev = task.ledger.add_evidence("http", "https://alvo.example/", "HTTP 200 text/html · server: nginx")
    task.ledger.add_finding("H", "HSTS ausente", "Resposta HTTPS sem Strict-Transport-Security.", [ev.id], severity="baixa", recommendation="Enviar HSTS.")
    task.ledger.add_finding("H", "CSP ausente", "Sem CSP.", [ev.id], severity="baixa", recommendation="Enviar CSP.")
    draft = "\n".join([
        "## Metodologia",
        "A metodologia adotada baseou-se exclusivamente em coleta passiva e somente leitura, deixando testes ativos de XSS, CSRF e injeção de SQL fora do escopo.",
        "",
        "## Principais descobertas",
        "[H][F1] HSTS ausente — sem Strict-Transport-Security [E1]",
        "- **O que significa:** A ausência do cabeçalho foi observada na resposta e trata-se de uma hipótese.",
        "- **Impacto potencial:** Um cenário hipotético de downgrade poderia ocorrer.",
        "- **Prioridade:** baixa.",
        "",
        "## Recomendações",
        "- **Prioridade alta:** Implementar HSTS imediatamente [F1].",
        "- Configurar CSP com prioridade média [F2].",
    ])
    rev = review_report(task, draft)
    # Metodologia válida preservada no corpo (não movida para hipóteses)
    assert "fora do escopo" in rev.text and rev.hypotheses == []
    # H não é descrita como hipótese, mas impacto hipotético é preservado
    assert "trata-se de uma hipótese" not in rev.text
    assert "configuração ausente ou fraca observada" in rev.text
    assert "cenário hipotético" in rev.text
    # Prioridade do HSTS e CSP alinhadas à do dossiê (baixa) em todas as seções
    assert "Prioridade alta" not in rev.text and "prioridade média" not in rev.text
    assert "Prioridade baixa" in rev.text and "prioridade baixa" in rev.text
    assert rev.priority_fixed == 2


def test_coverage_matrix_accurately_reflects_partial_and_unverified_items(db, settings, public_dns):
    def unversioned_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/":
            return httpx.Response(200, text='<html><body><form method="post" action="/login"><input type="password" name="p"></form></body></html>', headers={"content-type": "text/html"})
        return httpx.Response(404, text="404", headers={"content-type": "text/plain"})

    service, _prov, _p, _e = make_env(db, settings, unversioned_handler)
    msg = last_assistant(service, run(collect(service, "Faça uma investigação profunda da segurança de https://alvo.example/")))
    matrix = msg.meta["research"]["view"]["coverage_matrix"]
    by_area = {m["area"]: m for m in matrix}
    assert by_area["Formulários"]["status"] == "PARCIALMENTE VERIFICADO"
    assert by_area["Bibliotecas e versões"]["status"] == "PARCIALMENTE VERIFICADO"
    assert by_area["CVEs aplicáveis"]["status"] == "NÃO VERIFICADO"
    assert by_area["Configurações e exposição de informações"]["status"] == "PARCIALMENTE VERIFICADO"
    assert by_area["Testes ativos (exploração, fuzzing, autenticação)"]["status"] == "NÃO VERIFICADO"
    assert "### Matriz de cobertura" in msg.content
    assert "PARCIALMENTE VERIFICADO" in msg.content and "NÃO VERIFICADO" in msg.content

    # Quando há versão identificada (ex.: nginx/1.14.0) e busca ativa, CVEs aparecem como PARCIALMENTE VERIFICADO
    site2 = Site(pages={"/": '<html><body><p>ok</p></body></html>'})
    service2, _prov2, _p2, _e2 = make_env(db, settings, site2)
    msg2 = last_assistant(service2, run(collect(service2, "Faça uma investigação profunda da segurança de https://alvo.example/")))
    by_area2 = {m["area"]: m for m in msg2.meta["research"]["view"]["coverage_matrix"]}
    assert by_area2["CVEs aplicáveis"]["status"] == "PARCIALMENTE VERIFICADO"


# ====================================================================== 4 & 5. Browser UI: Scroll, Composer, Copiar, Reload, Repack, Clipboard Failure, XSS


@pytest.fixture(scope="module")
def ui_server(tmp_path_factory):
    app = create_app(make_settings(tmp_path_factory.mktemp("uxweb")))
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


@pytest.mark.parametrize("viewport", [(1920, 1080), (1366, 768), (390, 844)], ids=["1920x1080", "1366x768", "mobile-390x844"])
def test_chat_scroll_anchored_composer_and_expanded_panel_no_extra_space(browser, ui_server, viewport):
    _app, url = ui_server
    page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
    try:
        page.goto(url)
        page.wait_for_function("window.EnslavedResearch && window.EnslavedMarkdown")
        page.evaluate(
            """([rep, view]) => {
                const msgs = document.getElementById('messages');
                for (let i = 0; i < 6; i++) {
                    msgs.append(messageBubble('user', 'Prompt ' + i + ' com **https://alvo.example/**\\nlinha 2', null, 'c1'));
                    msgs.append(messageBubble('assistant', rep + '\\n\\n' + ('Bloco de texto longo. '.repeat(20)), { research: { view } }, 'c1'));
                }
                document.getElementById('chat').classList.remove('empty');
                document.querySelectorAll('.rp-ledger, .rp-group').forEach(d => d.open = true);
                const m = document.getElementById('messages');
                m.scrollTop = m.scrollHeight;
            }""",
            [REPORT, sample_view()],
        )
        # Tentar rolar a página e o wheel além do composer
        page.evaluate("window.scrollTo(0, 99999)")
        page.mouse.move(viewport[0] // 2, viewport[1] - 40)
        page.mouse.wheel(0, 1000)
        page.wait_for_timeout(100)
        metrics = page.evaluate("""() => {
            const cw = document.getElementById('composer-wrap').getBoundingClientRect();
            const m = document.getElementById('messages');
            const last = m.lastElementChild.getBoundingClientRect();
            return {
                winH: window.innerHeight,
                docSh: document.documentElement.scrollHeight,
                winScrollY: window.scrollY,
                cwBottom: Math.round(cw.bottom),
                cwTop: Math.round(cw.top),
                mBottom: Math.round(m.getBoundingClientRect().bottom),
                lastBottom: Math.round(last.bottom),
                scrollGap: Math.round(m.scrollHeight - m.scrollTop - m.clientHeight),
            };
        }""")
        assert metrics["docSh"] == metrics["winH"]
        assert metrics["winScrollY"] == 0
        assert abs(metrics["cwBottom"] - metrics["winH"]) <= 1
        assert metrics["scrollGap"] <= 2
        assert metrics["lastBottom"] <= metrics["cwTop"] + 2

        # Alternar sidebar (colapsar desktop ou abrir drawer mobile) também mantém o composer ancorado sem scroll extra
        tag = f"{viewport[0]}x{viewport[1]}"
        page.screenshot(path=str(Path("docs/screenshots") / f"calibration-chat-{tag}.png"))
        page.click("#toggle-sidebar")
        page.wait_for_timeout(220)
        after_toggle = page.evaluate("() => ({docSh: document.documentElement.scrollHeight, winH: window.innerHeight, cwBottom: Math.round(document.getElementById('composer-wrap').getBoundingClientRect().bottom)})")
        assert after_toggle["docSh"] == after_toggle["winH"]
        assert abs(after_toggle["cwBottom"] - after_toggle["winH"]) <= 1
    finally:
        page.close()


def test_auto_scroll_respects_user_reading_older_messages(browser, ui_server):
    _app, url = ui_server
    page = browser.new_page(viewport={"width": 1366, "height": 768})
    page.add_init_script(FAKE_STREAM)
    try:
        page.goto(url)
        page.wait_for_function("window.EnslavedResearch")
        page.evaluate("""() => {
            const msgs = document.getElementById('messages');
            for (let i = 0; i < 12; i++) {
                msgs.append(messageBubble('user', 'Antiga ' + i, null, 'c1'));
                msgs.append(messageBubble('assistant', 'Resposta antiga ' + i + '\\n\\n' + ('Texto. '.repeat(30)), null, 'c1'));
            }
            document.getElementById('chat').classList.remove('empty');
            msgs.scrollTop = msgs.scrollHeight;
        }""")
        page.fill("#input", "Nova pergunta")
        page.keyboard.press("Enter")
        page.wait_for_function("window.__ctl")
        # Usuário sobe o scroll para ler mensagens antigas durante o streaming
        page.evaluate("const m = document.getElementById('messages'); m.scrollTop = 100; m.dispatchEvent(new Event('scroll'));")
        page.wait_for_timeout(50)
        page.evaluate("""() => {
            window.__ctl.push({type: 'thinking', content: 'pensando bastante...'});
            window.__ctl.push({type: 'token', content: 'Primeiro trecho. '});
            window.__ctl.push({type: 'token', content: 'Segundo trecho longo.\\n\\n' + 'Linha '.repeat(40)});
            window.__ctl.push({type: 'done', conversation_id: 'c1', message_id: 'm1'});
            window.__ctl.close();
        }""")
        page.wait_for_function("() => !document.querySelector('.content.streaming')")
        assert page.evaluate("document.getElementById('messages').scrollTop") < 300
    finally:
        page.close()


def test_copy_buttons_user_and_assistant_full_markdown_reload_repack_and_clipboard_failure(browser, ui_server):
    app, url = ui_server
    long_markdown = (
        "# Relatório Técnico\n\n"
        "Resumo com [Link](https://alvo.example/busca?q=a*b) e texto longo:\n\n"
        + ("- Item de lista detalhado com observação completa.\n" * 35)
        + "\n| Coluna A | Coluna B |\n| --- | --- |\n| 1 | `valor_interno` |\n\n"
        "```python\ndef check():\n    return 'ok'\n```\n"
    )
    user_prompt = "Investigue **https://alvo.example/**\n\n```bash\necho \"linha 2\"\n```"

    class FixedProvider(ModelProvider):
        name = "fixed"

        def __init__(self):
            super().__init__("fixed-1b")

        async def chat(self, messages, **options):
            return "MESMO"

        async def stream_chat(self, messages, **options):
            for i in range(0, len(long_markdown), 32):
                yield long_markdown[i : i + 32]

        async def health(self):
            return {"available": True}

    app.state.chat_service.provider = FixedProvider()
    page = browser.new_page(viewport={"width": 1366, "height": 768})
    # Mock de clipboard controlável para validar exatamente o texto copiado e também simular falha
    page.add_init_script("""
        window.__copied = [];
        window.__failCopy = false;
        Object.defineProperty(navigator, 'clipboard', {
            configurable: true,
            value: {
                writeText: async (text) => {
                    if (window.__failCopy) throw new DOMException('Denied', 'NotAllowedError');
                    window.__copied.push(text);
                }
            }
        });
        document.execCommand = () => !window.__failCopy;
    """)
    try:
        page.goto(url)
        page.wait_for_function("() => Boolean(window.EnslavedCopy)")
        page.fill("#input", user_prompt)
        page.click("#send")
        page.wait_for_function("() => !document.querySelector('.content.streaming') && !document.getElementById('composer').classList.contains('busy')", timeout=15000)

        user_btn = page.locator("#messages .message.user .copy-btn").last
        asst_btn = page.locator("#messages .message.assistant .copy-btn").last
        assert user_btn.is_visible() and asst_btn.is_visible()
        assert user_btn.get_attribute("aria-label") and user_btn.get_attribute("title")
        assert asst_btn.get_attribute("aria-label") and asst_btn.get_attribute("title")

        # Cópia da mensagem do usuário preserva Markdown, URL e quebras de linha
        user_btn.click()
        page.wait_for_function("() => window.__copied.length === 1")
        assert page.evaluate("window.__copied[0]") == user_prompt
        assert "Copied" in user_btn.inner_text()

        # Cópia da resposta da IA via teclado (Enter) copia TODO o Markdown longo (não só o visível)
        asst_btn.focus()
        page.keyboard.press("Enter")
        page.wait_for_function("() => window.__copied.length === 2")
        copied_ai = page.evaluate("window.__copied[1]")
        assert copied_ai == long_markdown
        assert "# Relatório Técnico" in copied_ai and "| Coluna A | Coluna B |" in copied_ai and "def check():" in copied_ai

        # Reload da página e reabertura da conversa preserva cópia exata (e após Repack também)
        conv_id = page.evaluate("state.currentId")
        repack_resp = page.request.post(f"{url}/api/memory/conversations/{conv_id}/repack", data={})
        assert repack_resp.ok
        page.reload()
        page.wait_for_selector("#conversation-list .conversation-item .title")
        page.locator("#conversation-list .conversation-item .title").first.click()
        page.wait_for_selector("#messages .message.assistant .copy-btn")
        page.locator("#messages .message.user .copy-btn").last.click()
        page.locator("#messages .message.assistant .copy-btn").last.click()
        page.wait_for_function("() => window.__copied.length === 2")
        assert page.evaluate("window.__copied[0]") == user_prompt
        assert page.evaluate("window.__copied[1]") == long_markdown

        # Seleção de texto existente no conteúdo não é quebrada e não inclui o botão Copiar
        selected = page.evaluate("""() => {
            const p = document.querySelector('#messages .message.assistant .content p');
            const range = document.createRange();
            range.selectNodeContents(p);
            const sel = window.getSelection();
            sel.removeAllRanges();
            sel.addRange(range);
            return sel.toString();
        }""")
        assert "Resumo com Link" in selected and "Copy" not in selected

        # Falha da Clipboard API não mostra Copied e exibe erro
        page.evaluate("window.__failCopy = true")
        asst_btn2 = page.locator("#messages .message.assistant .copy-btn").last
        asst_btn2.click()
        page.wait_for_selector("#toast.error:not([hidden])")
        assert "Copied" not in asst_btn2.inner_text()
        assert "Could not copy" in page.inner_text("#toast")
    finally:
        page.close()


def test_incomplete_streaming_report_cannot_be_copied_until_done_and_panel_has_own_copy(browser, ui_server):
    _app, url = ui_server
    page = browser.new_page(viewport={"width": 1366, "height": 768})
    page.add_init_script(FAKE_STREAM)
    page.add_init_script("""
        window.__copied = [];
        Object.defineProperty(navigator, 'clipboard', {
            configurable: true,
            value: { writeText: async (text) => { window.__copied.push(text); } }
        });
    """)
    try:
        page.goto(url)
        page.wait_for_function("() => Boolean(window.EnslavedResearch)")
        page.fill("#input", "Investigue https://alvo.example/")
        page.keyboard.press("Enter")
        page.wait_for_function("() => Boolean(window.__ctl)")
        page.evaluate("window.__ctl.push({type: 'token', content: 'Relatório ainda em andamento...'})")
        page.wait_for_selector("#messages .message.assistant > .msg-footer > .copy-btn")
        btn = page.locator("#messages .message.assistant > .msg-footer > .copy-btn").last
        assert btn.is_disabled()  # não permite copiar relatório incompleto durante streaming como se estivesse finalizado
        view = sample_view()
        page.evaluate("([v]) => { window.__ctl.push({type: 'done', conversation_id: 'c1', message_id: 'm1', research: {view: v, completion: v.completion}}); window.__ctl.close(); }", [view])
        page.wait_for_selector("#messages .research-panel .rp-copy-btn")
        assert not btn.is_disabled()
        btn.click()
        page.locator("#messages .research-panel .rp-copy-btn").click()
        page.wait_for_function("() => window.__copied.length === 2")
        # O botão Copy da IA não inclui o ledger técnico separado do painel; o botão do painel copia as evidências
        assert "Technical evidence" not in page.evaluate("window.__copied[0]")
        assert "Technical evidence" in page.evaluate("window.__copied[1]") and "E10" in page.evaluate("window.__copied[1]")
        # Conteúdo hostil/XSS no painel continua inofensivo ao copiar
        assert page.evaluate("window.__xss") is None
        page.screenshot(path=str(Path("docs/screenshots") / "calibration-evidence-copy-1366.png"))
    finally:
        page.close()


def test_copy_handles_untrusted_xss_content_and_clipboard_fallback(browser, ui_server):
    _app, url = ui_server
    page = browser.new_page(viewport={"width": 390, "height": 844})
    page.add_init_script("""
        window.__fallbackCopied = [];
        Object.defineProperty(navigator, 'clipboard', {
            configurable: true,
            value: { writeText: async () => { throw new Error('clipboard blocked'); } }
        });
        document.execCommand = (cmd) => {
            if (cmd === 'copy') {
                const active = document.activeElement;
                window.__fallbackCopied.push(active ? active.value : '');
                return true;
            }
            return false;
        };
    """)
    try:
        page.goto(url)
        page.wait_for_function("() => Boolean(window.EnslavedCopy)")
        hostile_user = '<script>window.__xss=1</script> **https://alvo.example/**'
        hostile_ai = 'Texto seguro <img src=x onerror="window.__xss=1"> [link](javascript:window.__xss=1)\n\n```html\n<script>alert(1)</script>\n```'
        page.evaluate(
            """([u, a]) => {
                const msgs = document.getElementById('messages');
                msgs.append(messageBubble('user', u, null, 'c1'));
                msgs.append(messageBubble('assistant', a, null, 'c1'));
                document.getElementById('chat').classList.remove('empty');
            }""",
            [hostile_user, hostile_ai],
        )
        assert page.evaluate("window.__xss") is None
        user_btn = page.locator("#messages .message.user .copy-btn").last
        asst_btn = page.locator("#messages .message.assistant > .msg-footer > .copy-btn").last
        assert user_btn.is_visible() and asst_btn.is_visible()
        user_btn.click()
        asst_btn.click()
        page.wait_for_function("() => window.__fallbackCopied.length === 2")
        assert page.evaluate("window.__fallbackCopied[0]") == hostile_user
        assert page.evaluate("window.__fallbackCopied[1]") == hostile_ai
        assert page.evaluate("window.__xss") is None
    finally:
        page.close()

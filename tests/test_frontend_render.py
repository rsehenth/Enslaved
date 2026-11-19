"""Renderização das respostas no navegador real (Chrome via Playwright): Markdown, tabelas, XSS, streaming e layout."""

import socket
import threading
import time

import pytest
import uvicorn

from backend.main import create_app
from backend.providers.base import ModelProvider
from tests.helpers import make_settings

sync_api = pytest.importorskip("playwright.sync_api")

TABLE = (
    "## Comparação\n\nSegue o resumo:\n\n"
    "| Item | Detalhe | Observação muito longa para forçar rolagem horizontal da tabela em telas pequenas |\n"
    "| --- | --- | --- |\n"
    "| A | linha 1<br>linha 2 | **negrito** e `código` |\n"
    "| B | [site](https://exemplo.com) | fim |\n\n"
    "```python\nprint('oi')\n```\n"
)


class Scripted(ModelProvider):
    name = "scripted"

    def __init__(self, reply):
        super().__init__("scripted-1b")
        self.reply = reply

    async def chat(self, messages, **options):
        return "MESMO"

    async def stream_chat(self, messages, **options):
        import asyncio

        for start in range(0, len(self.reply), 7):
            await asyncio.sleep(0.005)
            yield self.reply[start : start + 7]

    async def health(self):
        return {"available": True}


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    app = create_app(make_settings(tmp_path_factory.mktemp("web")))
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
        except Exception:  # sem Chrome instalado
            pytest.skip("Chrome indisponível para Playwright")
        yield br
        br.close()


@pytest.fixture
def page(browser, server):
    pg = browser.new_page(viewport={"width": 1100, "height": 800})
    pg.goto(server[1])
    pg.wait_for_function("window.EnslavedMarkdown")
    yield pg
    pg.close()


def render(page, source, partial=False):
    return page.evaluate(
        "([s, p]) => { const d = document.createElement('div'); d.append(window.EnslavedMarkdown.render(s, { partial: p })); return d.innerHTML; }",
        [source, partial],
    )


def test_markdown_blocks_render_as_html(page):
    html = render(page, "# T\n\nUm **forte** e *itálico* com `código`.\n\n- a\n- b\n\n1. x\n2. y\n\n> citação\n\n---\n\n[l](https://a.com)")
    for needle in ("<h1>", "<strong>forte</strong>", "<em>itálico</em>", "<code>código</code>", "<ul>", "<ol>", "<blockquote>", "<hr>"):
        assert needle in html
    assert 'target="_blank"' in html and 'rel="noopener noreferrer"' in html


def test_gfm_table_with_br_becomes_real_table_not_text(page):
    html = render(page, TABLE)
    assert '<div class="table-wrap"><table>' in html and "<th>Item</th>" in html
    assert "linha 1<br>linha 2" in html  # <br> vira quebra de linha real
    assert "&lt;br" not in html and 'data-lang="python"' in html and 'class="code"' in html


def test_malformed_table_gets_safe_literal_fallback(page):
    broken = "| a | b |\n| 1 | 2 | 3 |\ntexto solto"
    html = render(page, broken)
    assert "<table" not in html and "<pre" in html and "| 1 | 2 | 3 |" in html  # nada inventado nem removido


@pytest.mark.parametrize(
    "payload",
    [
        "<script>window.__xss=1</script>",
        '<img src=x onerror="window.__xss=1">',
        "[x](javascript:window.__xss=1)",
        '<a href="javascript:window.__xss=1">x</a>',
        "<svg onload=\"window.__xss=1\"></svg>",
        '<iframe src="https://evil.example"></iframe>',
        '<style>body{display:none}</style>',
        '<form action="https://evil.example"><input name=p></form>',
        '<p onclick="window.__xss=1" style="position:fixed">x</p>',
    ],
)
def test_malicious_content_is_neutralized(page, payload):
    html = render(page, payload)
    page.evaluate("h => { const d = document.createElement('div'); d.innerHTML = h; document.body.append(d); }", html)
    lowered = html.lower()
    for bad in ("<script", "onerror", "onload", "onclick", "javascript:", "<iframe", "<style", "<form", "<svg", "<img", "style="):
        assert bad not in lowered
    assert page.evaluate("window.__xss") is None


def test_partial_markdown_never_breaks_rendering(page):
    assert "<pre" in render(page, "texto\n\n```js\nconst a = 1;", partial=True)  # cerca aberta
    assert "<table" in render(page, "| a | b |\n| --- | --- |\n| 1 |", partial=True)  # linha incompleta
    assert render(page, "**negrito sem fechar", partial=True)  # não lança


def send(page, text):
    page.fill("#input", text)
    page.keyboard.press("Enter")
    page.wait_for_selector("#messages .message.assistant .table-wrap", timeout=15000)
    page.wait_for_function("!document.querySelector('.content.streaming') && !document.getElementById('composer').classList.contains('busy')", timeout=15000)


@pytest.mark.parametrize("viewport", [(1100, 800), (375, 812)], ids=["desktop", "mobile"])
def test_streamed_table_renders_and_fits_the_chat(browser, server, viewport):
    app, url = server
    app.state.chat_service.provider = Scripted(TABLE)
    page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
    try:
        page.goto(url)
        send(page, "Compare A e B")
        message = page.locator("#messages .message.assistant").last
        assert message.locator("table th").first.inner_text() == "Item"
        assert "<br" not in message.locator(".content").inner_text()
        assert page.locator("#messages .message.assistant .content").last.get_attribute("data-raw") == TABLE  # texto original intacto
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")  # sem rolagem horizontal da página
        box = message.locator(".content").bounding_box()
        wrap = message.locator(".table-wrap").bounding_box()
        assert wrap["x"] + wrap["width"] <= box["x"] + box["width"] + 1  # tabela dentro da coluna
        assert not page.locator("#input").is_disabled()  # composer liberado
        page.screenshot(path=str(__import__("pathlib").Path("docs/screenshots") / f"response-table-{'mobile' if viewport[0] < 500 else 'desktop'}.png"))
        if viewport[0] < 500:
            assert page.evaluate("(() => { const w = document.querySelector('.table-wrap'); return w.scrollWidth > w.clientWidth; })()")  # rolagem própria
        # recarregar a conversa renderiza o mesmo conteúdo a partir do histórico salvo
        page.reload()
        page.locator("#conversation-list button, #conversation-list li, #conversation-list a").first.dispatch_event("click")
        page.wait_for_selector("#messages .message.assistant .table-wrap", timeout=10000)
        assert page.locator("#messages .message.assistant .content").last.get_attribute("data-raw") == TABLE
    finally:
        page.close()


def test_scene_stays_done_until_user_types_or_leaves_the_conversation(browser, server):
    app, url = server
    app.state.chat_service.provider = Scripted("Resposta curta.")
    page = browser.new_page()
    mood = lambda: page.evaluate("document.getElementById('scene').dataset.mood")  # noqa: E731
    try:
        page.goto(url)
        page.fill("#input", "oi")
        page.keyboard.press("Enter")
        page.wait_for_selector("#messages .message.assistant .content")
        time.sleep(0.8)
        assert mood() == "done"
        time.sleep(3)  # antes voltava a idle em 2 s
        assert mood() == "done"
        page.keyboard.type("x")  # digitar retoma o comportamento normal
        assert mood() == "typing"
        page.fill("#input", "")
        page.dispatch_event("#input", "input")
        assert mood() == "idle"
        page.fill("#input", "outra")
        page.keyboard.press("Enter")
        page.wait_for_function("document.querySelectorAll('#messages .message.assistant .content').length >= 2")
        time.sleep(0.8)
        assert mood() == "done"  # sempre termina em joinha
        page.click("#new-chat")
        assert mood() == "idle"  # sair da conversa volta ao repouso
    finally:
        page.close()

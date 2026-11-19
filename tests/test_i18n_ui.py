"""Troca pt-BR/en-US no navegador real (Chrome via Playwright): interface, persistência, fluxos de chat, pesquisa profunda, Learning Center."""

import asyncio
import socket
import threading
import time

import pytest
import uvicorn

from backend.main import create_app
from backend.providers.base import ModelProvider, ProviderError
from tests.helpers import make_settings

sync_api = pytest.importorskip("playwright.sync_api")


class Scripted(ModelProvider):
    name = "scripted"

    def __init__(self, reply="Resposta simulada."):
        super().__init__("scripted-1b")
        self.reply = reply

    async def chat(self, messages, **options):
        return self.reply

    async def stream_chat(self, messages, **options):
        for start in range(0, len(self.reply), 9):
            await asyncio.sleep(0.003)
            yield self.reply[start : start + 9]

    async def health(self):
        return {"available": True}


class Broken(Scripted):
    async def stream_chat(self, messages, **options):
        raise ProviderError("falha simulada")
        yield  # pragma: no cover

    async def chat(self, messages, **options):
        raise ProviderError("falha simulada")


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    app = create_app(make_settings(tmp_path_factory.mktemp("i18n")))
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


@pytest.fixture
def page(browser, server):
    app, url = server
    app.state.chat_service.provider = Scripted()
    pg = browser.new_page(viewport={"width": 1100, "height": 800})
    pg.console_log = []
    pg.on("console", lambda m: pg.console_log.append(m.text))
    pg.goto(url)
    pg.wait_for_function("() => window.i18n && window.EnslavedResearch")
    yield pg
    pg.close()


def set_language(page, code):
    page.click("#open-settings")
    page.select_option("#setting-language", code)
    page.click("#settings-close")


def text(page, selector):
    return page.inner_text(selector).strip()


def test_default_is_english_for_installs_without_preference(page):
    assert page.evaluate("document.documentElement.lang") == "en-US"
    assert text(page, "#new-chat") == "New chat"
    assert page.evaluate("i18n.lang") == "en-US"
    assert page.evaluate("localStorage.getItem('enslaved.language')") is None  # nada gravado até o usuário escolher


def test_language_selector_names_each_option_in_its_own_language(page):
    page.click("#open-settings")
    options = page.eval_on_selector_all("#setting-language option", "els => els.map(e => [e.value, e.textContent])")
    assert options == [["en-US", "English"], ["pt-BR", "Português (Brasil)"]]
    assert page.inner_text("label.field:has(#setting-language) span").strip() == "Idioma / Language"  # igual nos dois idiomas


def test_switch_to_portuguese_is_immediate_and_persists_after_reload(page):
    set_language(page, "pt-BR")
    assert text(page, "#new-chat") == "Nova conversa"
    assert text(page, "#open-settings") == "Configurações"
    assert page.get_attribute("#input", "placeholder") == "Envie uma mensagem para o Enslaved"
    assert page.get_attribute("#send", "aria-label") == "Enviar mensagem"
    assert page.evaluate("document.documentElement.lang") == "pt-BR"
    assert page.evaluate("localStorage.getItem('enslaved.language')") == "pt-BR"  # só o código do idioma
    page.reload()
    page.wait_for_function("() => window.i18n")
    assert text(page, "#new-chat") == "Nova conversa"
    assert page.evaluate("document.querySelector('#setting-language').value") == "pt-BR"
    set_language(page, "en-US")
    assert text(page, "#new-chat") == "New chat"
    assert page.evaluate("localStorage.getItem('enslaved.language')") == "en-US"


def test_unknown_saved_language_falls_back_to_english(browser, server):
    pg = browser.new_page()
    pg.add_init_script("localStorage.setItem('enslaved.language', 'xx-XX')")
    pg.goto(server[1])
    pg.wait_for_function("() => window.i18n")
    assert pg.evaluate("i18n.lang") == "en-US"
    pg.close()


def test_translation_engine_interpolation_plural_fallback_and_missing_keys(page):
    out = page.evaluate("""() => {
      i18n.setLanguage('pt-BR');
      const r = {};
      r.one = tr('sel.count', { count: 1 });
      r.many = tr('sel.count', { count: 3 });
      r.interp = tr('chat.maxImages', { limit: 4 });
      r.untouched = tr('chat.maxImages');           // parâmetro ausente: o marcador permanece visível
      r.missing = tr('no.such.key');
      r.logged = i18n.missing.includes('pt-BR:no.such.key');
      const cat = i18n.catalog('pt-BR'); const saved = cat['chat.new']; delete cat['chat.new'];
      r.fallback = tr('chat.new');                  // chave só no catálogo-padrão: cai em en-US, nunca fica vazia
      cat['chat.new'] = saved;
      r.numberPt = i18n.formatNumber(1234.5, { minimumFractionDigits: 1 });
      r.datePt = i18n.formatDate('2026-03-05T12:00:00Z', { dateStyle: 'medium', timeZone: 'UTC' });
      i18n.setLanguage('en-US');
      r.number = i18n.formatNumber(1234.5, { minimumFractionDigits: 1 });
      r.date = i18n.formatDate('2026-03-05T12:00:00Z', { dateStyle: 'medium', timeZone: 'UTC' });
      return r;
    }""")
    assert out["one"] == "1 selecionada" and out["many"] == "3 selecionadas"
    assert out["interp"] == "Máximo de 4 imagens por mensagem." and "{limit}" in out["untouched"]
    assert out["missing"] == "no.such.key" and out["logged"] is True
    assert out["fallback"] == "New chat"
    assert out["number"] == "1,234.5" and out["numberPt"] == "1.234,5"
    assert "Mar" in out["date"] and "mar" in out["datePt"]


def test_backend_texts_are_translated_only_when_catalogued(page):
    out = page.evaluate("""() => {
      i18n.setLanguage('en-US');
      const r = [tm('Preparando resposta'), tm('Planejando 3 alvo(s), nível deep'), tm('Consultando example.com'),
                 tm('Texto livre do modelo que não consta'), tm('2 resposta(s) em 1 página(s)'), tm('CSP em 1/2; HSTS em 0/2')];
      i18n.setLanguage('pt-BR');
      r.push(tm('Preparando resposta'));
      return r;
    }""")
    assert out == ["Preparing answer", "Planning 3 target(s), level deep", "Querying example.com", "Texto livre do modelo que não consta",
                   "2 response(s) on 1 page(s)", "CSP in 1/2; HSTS in 0/2", "Preparando resposta"]


def test_chat_flow_in_english_sends_language_and_keeps_history_on_switch(page, server):
    app, url = server
    bodies = []
    page.on("request", lambda r: bodies.append(r.post_data) if r.url.endswith("/api/chat/stream") else None)
    set_language(page, "en-US")
    page.fill("#input", "Pergunta antiga em português")
    page.keyboard.press("Enter")
    page.wait_for_selector("#messages .message.assistant .copy-btn:not([disabled])")
    assert '"language":"en-US"' in bodies[-1]
    assert text(page, "#messages .message.assistant .copy-btn") == "Copy"
    assert page.get_attribute("#messages .message.assistant .copy-btn", "aria-label") == "Copy Enslaved's reply"
    assert text(page, "#messages .message.user .copy-btn") == "Copy"
    assert text(page, "#chat-title") != "New chat"

    conv_id = page.evaluate("state.currentId")
    before = page.evaluate("fetch(`/api/conversations/${state.currentId}`).then(r => r.json()).then(d => d.messages.map(m => m.content))")
    draft = "rascunho que não pode sumir"
    page.fill("#input", draft)
    shown_before = page.eval_on_selector_all("#messages .message .content", "els => els.map(e => e.dataset.raw)")
    page.evaluate("window.__marker = true")

    set_language(page, "pt-BR")
    page.wait_for_function("() => document.querySelector('#messages .message.assistant .copy-btn span')?.textContent === 'Copiar'")
    assert page.evaluate("window.__marker") is True  # sem recarregar a página
    assert page.evaluate("state.currentId") == conv_id
    assert page.input_value("#input") == draft  # rascunho preservado
    assert page.eval_on_selector_all("#messages .message .content", "els => els.map(e => e.dataset.raw)") == shown_before
    after = page.evaluate("fetch(`/api/conversations/${state.currentId}`).then(r => r.json()).then(d => d.messages.map(m => m.content))")
    assert after == before  # o histórico salvo não é tocado pela troca de idioma
    assert not [m for m in page.console_log if "chave ausente" in m]


def test_error_state_is_localized(page, server):
    app, _ = server
    app.state.chat_service.provider = Broken()
    set_language(page, "en-US")
    page.fill("#input", "teste")
    page.keyboard.press("Enter")
    page.wait_for_selector("#messages .message.error")
    assert text(page, "#messages .message.error").startswith("Error:")


def test_permission_and_depth_cards_and_dialogs(page):
    page.evaluate("i18n.setLanguage('en-US')")
    perm = page.evaluate("""() => { const n = permissionCard({ tool: 'fs.read', arguments: { path: 'a' }, request_id: 'r1', reason: 'Pedido expirado.' });
      return [n.innerText, [...n.querySelectorAll('button')].map(b => b.textContent)]; }""")
    assert "Permission requested" in perm[0] and "Tool fs.read is asking for authorization to run." in perm[0]
    assert "Request expired." in perm[0] and "Remember in this conversation" in perm[0]
    assert perm[1] == ["Approve", "Deny"]
    passive = page.evaluate("permissionCard({ tool: 'Investigação passiva', research: true, arguments: {}, request_id: 'r2' }).innerText")
    assert "passively inspect this target" in passive  # a lógica de segurança compara o identificador original do backend
    depth = page.evaluate("""() => depthCard({ question: 'Essa investigação pode ser realizada com diferentes níveis de profundidade. Como você prefere que eu analise?',
      options: [{ mode: 'quick', label: 'Rápida', hint: 'Visão geral e informações essenciais.' }, { mode: 'deep', label: 'Profunda', hint: 'Múltiplas fontes, verificações e evidências.' }], conversation_id: 'c1' }).innerText""")
    assert "How would you like me to analyze it?" in depth and "Quick" in depth and "Deep" in depth
    page.evaluate("i18n.setLanguage('pt-BR')")
    assert "Permissão solicitada" in page.evaluate("permissionCard({ tool: 'x', arguments: {}, request_id: 'r3' }).innerText")
    # modal de confirmação de exclusão (diálogo próprio) em inglês
    page.evaluate("i18n.setLanguage('en-US')")
    page.evaluate("() => { askDialog({ title: tr('del.titleOne'), message: buildDeleteConfirmationMessage(2), okLabel: tr('del.ok'), danger: true }); }")  # a promessa só resolve ao confirmar
    assert text(page, "#dialog-message").startswith("You are about to delete 2 conversations")
    assert text(page, "#dialog-ok") == "Delete permanently"
    assert text(page, "#dialog-form footer .btn.ghost") == "Cancel"


def test_verification_note_and_tool_badges_follow_language(page):
    page.evaluate("i18n.setLanguage('en-US')")
    note = page.evaluate("verificationNote({ status: 'unverified', sources: [] }, 'q').innerText")
    assert note.startswith("NOT VERIFIED")
    assert page.evaluate("modeBadge('ask').textContent") == "asks permission"
    page.evaluate("i18n.setLanguage('pt-BR')")
    assert page.evaluate("modeBadge('ask').textContent") == "pede permissão"


VIEW = {
    "summary": [{"key": "pages", "label": "Páginas consultadas", "value": "1"}, {"key": "http", "label": "Evidências HTTP", "value": "2 resposta(s) em 1 página(s)"}],
    "groups": [{"key": "http", "label": "Respostas HTTP e cabeçalhos", "evidence_count": 2,
                "items": [{"ids": ["E1", "E2"], "kind": "http", "ok": True, "what": "Resposta HTTP e cabeçalhos de segurança", "result": "HTTP 200", "pages": ["https://a.example/"], "supports": [], "limit": None}]}],
    "header_matrix": None, "findings": [{"id": "F1", "category": "H", "title": "CSP ausente", "detail": "", "evidence": ["E1"], "pages": ["https://a.example/"], "severity": "média",
                                         "severity_label": "prioridade", "proof": "", "recommendation": ""}],
    "ledger": [{"id": "E1", "kind": "http", "url": "https://a.example/", "observation": "HTTP 200", "sha256": "ab", "ok": True, "tool": "t", "findings": ["F1"]},
               {"id": "E2", "kind": "http", "url": "https://a.example/x", "observation": "HTTP 500", "sha256": None, "ok": False, "tool": "t", "findings": []}],
    "coverage_matrix": [{"area": "Formulários", "status": "NÃO VERIFICADO", "detail": "Nenhuma página HTML HTTP 200 coletada."}],
    "limitations": [], "completion": {"status": "partial"},
}


def test_deep_research_card_and_evidence_panel_in_both_languages(page):
    page.evaluate("i18n.setLanguage('en-US')")
    page.evaluate("""(view) => { window.__card = EnslavedResearch.createCard(); document.querySelector('#messages').append(__card.node);
      __card.update({ type: 'research_progress', phase: 'collecting', message: 'Consultando example.com', pages: 1, requests: 3, evidence: 1 });
      const bubble = messageBubble('assistant', 'Relatório com [F1] e [E1].');
      document.querySelector('#messages').append(bubble); EnslavedResearch.attach(bubble, view).link(); }""", VIEW)
    card = text(page, ".research-card")
    assert "Investigation" in card and "Querying sources" in card and "Querying example.com" in card
    assert "1 page · 3 requests · 1 piece of evidence" in card
    panel = text(page, ".research-panel")
    for needle in ("Investigation evidence", "Partial report", "Pages consulted", "HTTP evidence", "2 response(s) on 1 page(s)",
                   "Findings and their evidence", "priority média", "Investigation coverage", "NOT VERIFIED", "Forms", "No HTML HTTP 200 page collected.",
                   "View technical evidence (2)", "HTTP responses and headers"):
        assert needle in panel, needle
    assert page.get_attribute(".research-panel .rp-copy-btn", "aria-label") == "Copy the panel's technical evidence"
    assert page.get_attribute(".research-panel .ref-link", "aria-label") in ("View finding F1", "View evidence E1")
    page.evaluate("i18n.setLanguage('pt-BR')")  # troca ao vivo: o cartão em andamento se retraduz sem perder o estado
    card_pt = text(page, ".research-card")
    assert "Consultando fontes" in card_pt and "Consultando example.com" in card_pt and "1 página · 3 requisições · 1 evidência" in card_pt
    assert page.get_attribute(".research-panel", "aria-label") == "Evidências da investigação"
    page.evaluate("i18n.setLanguage('en-US')")
    page.evaluate("__card.finish('partial')")
    assert "Partial" in text(page, ".research-card .rc-state") and "report is partial" in text(page, ".research-card .rc-detail")


def test_learning_center_and_persistent_memory_ui_are_absent(page):
    assert page.locator("#open-learning").count() == 0
    assert page.locator("#learning-modal").count() == 0
    assert page.locator("#learn-source-modal").count() == 0
    page.click("#open-memory")
    assert text(page, "#memory-title") == "Context & history"
    assert page.locator("#memory-items").count() == 0
    assert page.locator("#memory-search").count() == 0
    assert page.locator("#memory-add-btn").count() == 0


def test_english_layout_has_no_horizontal_overflow_on_phone_and_desktop(browser, server):
    for width, height in ((390, 844), (1100, 800)):
        pg = browser.new_page(viewport={"width": width, "height": height})
        pg.add_init_script("localStorage.setItem('enslaved.language', 'en-US')")
        pg.goto(server[1])
        pg.wait_for_function("() => window.i18n")
        assert pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        for modal_open in ("#open-settings", "#open-memory"):
            if width < 820:
                pg.click("#toggle-sidebar")
            pg.click(modal_open)
            overflow = pg.evaluate("""() => [...document.querySelectorAll('.modal:not(.hidden) .btn')]
              .filter(b => b.offsetParent && b.scrollWidth > b.clientWidth + 1).map(b => b.textContent.trim())""")
            assert not overflow, (width, modal_open, overflow)
            assert pg.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
            pg.keyboard.press("Escape")
        pg.close()


def test_language_selector_is_keyboard_operable(page):
    page.click("#open-settings")
    page.focus("#setting-language")
    page.keyboard.press("ArrowDown")  # em um <select> focado troca de en-US para pt-BR e dispara change
    page.wait_for_function("() => document.documentElement.lang === 'pt-BR'")
    assert text(page, "#settings-save") == "Salvar"

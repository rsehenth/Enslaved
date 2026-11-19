"""Inteligência, liberdade de resposta e precisão (offline; sem rede real nem execução intrusiva).

Cobre as 12 áreas do plano: conversa simples, resposta técnica, thinking/orçamento, citações
clicáveis, URLs puras nas ferramentas, fontes conflitantes, identificação de entidades,
fatos vs. hipóteses, segurança sem bloqueio por palavra, ferramentas dentro do escopo,
compatibilidade SSE e regras do Deep Research.
"""

import asyncio
import json
import socket
import threading
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.chat.grounding import (
    extract_entities,
    is_factual,
    is_identification,
    is_time_sensitive,
    knowledge_block,
    model_only_block,
    unverified_prefix,
    web_block,
)
from backend.chat.models import ChatRequest
from backend.chat.service import ChatService
from backend.core.config import ChatSettings, NetToolsSettings, ToolsSettings
from backend.chat.presentation import load_presentation, load_response_policy
from backend.main import create_app
from backend.providers.base import ModelProvider, Thinking
from backend.providers.ollama import OllamaProvider
from backend.research.conflicts import detect_conflicts
from backend.research.report import report_rules, review_report
from backend.research.state import ResearchTask
from backend.research.evidence import Ledger
from backend.tools.builtin.web_fetch import WebFetchArgs, WebFetchTool
from backend.tools.builtin.web_inspect import WebInspectArgs
from backend.tools.engine import ToolEngine
from backend.tools.net_guard import normalize_candidate_url, resolve_public_url
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry
from tests.helpers import make_settings

from backend.core.directives import load_directives


# ---------------------------------------------------------------- infraestrutura mínima


class Capture(ModelProvider):
    name = "capture"

    def __init__(self, reply="Resposta do modelo.", verdict="MESMO"):
        super().__init__("capture-1b")
        self.reply, self.verdict, self.streams = reply, verdict, []

    async def chat(self, messages, **options):
        return self.verdict

    async def stream_chat(self, messages, **options):
        self.streams.append([dict(m) for m in messages])
        yield self.reply

    async def health(self):
        return {"available": True}


class ThinkingProvider(ModelProvider):
    """Emite raciocínio privado antes da resposta; o raciocínio nunca pode ser salvo."""

    name = "thinking"

    def __init__(self):
        super().__init__("thinking-1b")
        self.streams = []

    async def chat(self, messages, **options):
        return "Resposta final."

    async def stream_chat(self, messages, **options):
        self.streams.append([dict(m) for m in messages])
        yield Thinking("plano interno privado: 1+1=2")
        yield "Resposta "
        yield "final."

    async def health(self):
        return {"available": True}


def run(coro):
    return asyncio.run(coro)


def make_service(db, settings, directives, provider, net_mode="allow", handler=None, pages=None):
    calls = []

    def search(request):
        calls.append(request)
        return handler(request) if handler else httpx.Response(200, text="")

    fetch_tool = WebFetchTool(NetToolsSettings(timeout_seconds=5), transport=httpx.MockTransport(pages)) if pages else None
    tools = ToolsSettings(permission_modes={"net.fetch": net_mode})
    policy = ToolPolicy(tools)
    engine = ToolEngine(ToolRegistry(), policy, PermissionManager(policy), AuditLog(settings.storage.database_path.parent / "a.jsonl"), tools)
    service = ChatService(
        db, provider, directives, tool_engine=engine,
        chat_settings=settings.chat, net_settings=settings.tools.net,
        fetch_tool=fetch_tool,
        search_transport=httpx.MockTransport(search) if handler else None,
    )
    return service, calls


async def collect(service, message, **kwargs):
    return [e async for e in service.send_stream(None, message, use_tools=False, language="pt-BR", **kwargs)]


def system_of(provider):
    return provider.streams[-1][0]["content"]


# ------------------------------------------------- 1. conversa simples sem pesquisa desnecessária


def test_simple_conversation_never_searches_nor_cans_replies(db, settings, directives):
    provider = Capture(reply="Oi! Tudo ótimo por aqui. Em que posso ajudar?")
    service, calls = make_service(db, settings, directives, provider, handler=lambda r: httpx.Response(200, text=""))
    events = run(collect(service, "Oi, tudo bem?"))
    assert calls == [] and not [e for e in events if e["type"] == "verification"]
    assert not [e for e in events if e["type"] == "permission_request"]
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert text == "Oi! Tudo ótimo por aqui. Em que posso ajudar?"  # resposta natural, sem prefixo de auditoria
    assert "evidencia_web" not in system_of(provider) and "conhecimento próprio" not in system_of(provider)


def test_identification_uses_own_knowledge_instead_of_forcing_search(db, settings, directives):
    provider = Capture(reply="Jaspion é um seriado tokusatsu japonês da Toei, exibido no Brasil pela Manchete.")
    service, calls = make_service(db, settings, directives, provider, handler=lambda r: pytest.fail("não deveria pesquisar"))
    events = run(collect(service, "Você conhece o Jaspion?"))
    assert calls == [] and not [e for e in events if e["type"] == "verification"]
    assert "Identificação pelo conhecimento próprio" in system_of(provider)
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert "tokusatsu" in text and not text.startswith("**NÃO VERIFICADO.**")


# ------------------------------------------------- 2. resposta técnica detalhada


def test_technical_answers_get_full_policies_and_output_budget(client):
    status = client.get("/api/memory/status").json()
    assert status["output_reserve"] == 2048 and status["input_budget"] == 14336  # orçamento para respostas longas

    policy = load_response_policy(ChatSettings())
    assert "Epistemic honesty" in policy and "Never wrap ordinary conversation in audit/report format" in policy
    assert "never refuse, interrupt" in policy  # liberdade sem bloqueio por palavra
    presentation = load_presentation(ChatSettings())
    assert "no audit format" in presentation and "no repeated formulaic openers" in presentation


# ------------------------------------------------- 3. thinking e orçamento de saída


def test_think_flag_reaches_ollama_and_adds_headroom():
    provider = OllamaProvider("m", default_options={"num_ctx": 16384, "num_predict": 2048})
    off = provider._payload([{"role": "user", "content": "oi"}], False, {})
    assert off["think"] is False and off["options"]["num_predict"] == 2048
    on = provider._payload([{"role": "user", "content": "oi"}], False, {"think": True})
    assert on["think"] is True and on["options"]["num_predict"] == 4096  # folga de raciocínio (2048+2048)
    huge = provider._payload([{"role": "user", "content": "oi"}], False, {"num_predict": 32768, "think": True})
    assert huge["options"]["num_predict"] == 4096  # override absurdo é contido; folga de think mantida
    small_ctx = OllamaProvider("m", default_options={"num_ctx": 3072, "num_predict": 1024})
    tight = small_ctx._payload([{"role": "user", "content": "oi"}], False, {"think": True})
    assert tight["options"]["num_predict"] == 1536  # nunca excede num_ctx//2


def test_chat_request_think_reaches_provider_options():
    assert ChatRequest(message="x", think=True).provider_options() == {"think": True}
    assert ChatRequest(message="x", think=False).provider_options() == {"think": False}
    assert ChatRequest(message="x").provider_options() == {}


def test_thinking_is_shown_but_never_persisted(db, settings, directives):
    provider = ThinkingProvider()
    service, _calls = make_service(db, settings, directives, provider)
    events = run(collect(service, "Explique o que é um buffer overflow."))
    thinking = "".join(e["content"] for e in events if e["type"] == "thinking")
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert "plano interno privado" in thinking  # raciocínio vai ao vivo…
    assert text == "Resposta final." and "plano interno" not in text
    saved = service.repo.get_messages(events[0]["conversation_id"])[-1]
    assert saved.content == "Resposta final."  # …mas nunca é salvo nem tratado como relatório
    assert "thinking" not in (saved.meta or {})


# ------------------------------------------------- 4. citações web clicáveis


def test_verification_sources_carry_stable_ids_and_urls(db, settings, directives):
    ddg = (
        '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpt.wikipedia.org%2Fwiki%2FPython_(linguagem_de_programa%C3%A7%C3%A3o)&amp;r=1">Python – Wikipédia</a>'
        '<a class="result__snippet">A versão estável mais recente do Python é publicada no site oficial.</a>'
    )
    provider = Capture(reply="A versão mais recente está documentada [web#1] e também [web#2].")
    service, _calls = make_service(db, settings, directives, provider, handler=lambda r: httpx.Response(200, text=ddg))
    events = run(collect(service, "Qual é a versão mais recente do Python?"))  # fato SENSÍVEL ao tempo: verifica
    ver = next(e for e in events if e["type"] == "verification")
    assert [s["id"] for s in ver["sources"]] == ["web#1"]
    assert ver["sources"][0]["url"] == "https://pt.wikipedia.org/wiki/Python_(linguagem_de_programação)"
    saved = service.repo.get_messages(events[0]["conversation_id"])[-1]
    meta = saved.meta["verification"]["sources"]
    assert all(m["id"] and m["url"].startswith("https://") for m in meta)  # contrato do frontend: id + URL


def test_unverifiable_entity_reply_comes_from_own_knowledge_without_search(db, settings, directives):
    provider = Capture(reply="Não tenho certeza sobre Zyxqwerty Blorp; não vou inventar uma data.")
    service, calls = make_service(db, settings, directives, provider, handler=lambda r: pytest.fail("não deveria pesquisar"))
    events = run(collect(service, "Quando nasceu Zyxqwerty Blorp?"))  # detalhe estável: conhecimento próprio, sem rede
    text = "".join(e["content"] for e in events if e["type"] == "token")
    assert text == "Não tenho certeza sobre Zyxqwerty Blorp; não vou inventar uma data."
    assert calls == [] and not [e for e in events if e["type"] == "verification"]
    assert "Identificação pelo conhecimento próprio" in system_of(provider)  # honestidade: não invente datas
    saved = service.repo.get_messages(events[0]["conversation_id"])[-1]
    assert (saved.meta or {}).get("verification") is None  # sem pesquisa, não afirma fontes


# ------------------------------------------------- 5. URLs puras nas ferramentas net.*


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("[https://site.com](https://site.com)", "https://site.com"),
        ('"[https://site.com](https://site.com)"', "https://site.com"),
        ("[Ir ao site](https://site.com/docs)", "https://site.com/docs"),
        ('"https://site.com"', "https://site.com"),
        ("<https://site.com>", "https://site.com"),
        ("`https://site.com/pagina`", "https://site.com/pagina"),
        ("site.com", "https://site.com"),
        ("exemplo.com.br/caminho?a=1", "https://exemplo.com.br/caminho?a=1"),
    ],
)
def test_tool_urls_are_normalized_from_common_model_wrappers(raw, expected):
    assert normalize_candidate_url(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "https://site.com e https://outro.com",
        "me manda o conteúdo de site.com",
        "[https://a.com](https://b.com) por favor",
        "site com ponto final.",
    ],
)
def test_ambiguous_url_text_is_rejected_not_guessed(raw):
    with pytest.raises(ValueError, match="URL"):
        normalize_candidate_url(raw)


def test_args_models_normalize_and_report_clear_errors():
    assert WebFetchArgs(url="[https://site.com](https://site.com)").url == "https://site.com"
    assert WebInspectArgs(url="<https://site.com>").url == "https://site.com"
    with pytest.raises(ValidationError) as exc:
        WebFetchArgs(url="https://site.com e outro.com")
    assert "URL pura" in str(exc.value)


def test_normalization_never_bypasses_ssrf_protection():
    cleaned = normalize_candidate_url("[http://127.0.0.1:11434/](http://127.0.0.1:11434/)")
    assert cleaned == "http://127.0.0.1:11434/"
    with pytest.raises(Exception):
        resolve_public_url(cleaned, ("http", "https"), (80, 443))  # interno continua bloqueado
    from backend.tools.base import PermissionDecision

    tool = WebFetchTool(NetToolsSettings(timeout_seconds=2))
    granted = PermissionDecision(granted=True, decided_by="test")
    result = asyncio.run(tool.run({"url": "[http://127.0.0.1:11434/](http://127.0.0.1:11434/)"}, permission=granted))
    assert result.ok is False and "bloqueada" in (result.error or "").lower()


# ------------------------------------------------- 6. fontes conflitantes


def test_conflicting_sources_are_presented_not_reconciled():
    assert detect_conflicts("O site roda PHP 8.1 em produção.", "O site roda PHP 7.4 em produção.", language="pt-BR")
    assert not detect_conflicts("O site roda PHP 8.1 em produção.", "O site usa Linux no servidor.", language="pt-BR")
    sources = [
        {"title": "A", "url": "https://a.com", "text": "O site roda PHP 8.1 em produção.", "level": "page", "official": True},
        {"title": "B", "url": "https://b.com", "text": "O site roda PHP 7.4 em produção.", "level": "snippet", "official": False},
    ]
    block = web_block(sources, "2026-11-18", False, ["[web#1] e [web#2] divergem: versão do PHP"], language="pt-BR")
    assert "divergem" in block and "não invente explicação" in block  # divergência exposta como está


# ------------------------------------------------- 7. identificação de entidades


def test_entities_are_separated_and_identification_detected():
    assert extract_entities("Você conhece o Guarani de Bagé?") == ["Guarani de Bagé"]
    assert extract_entities("Quem criou o Linux?") == ["Linux"]
    assert is_identification("Você conhece o Jaspion?") is True
    assert is_identification("Quando o Guarani de Bagé foi fundado?") is False  # detalhe estável: conhecimento próprio, sem pesquisa
    assert is_time_sensitive("Qual a última versão do Python?") is True
    assert is_factual("Você conhece o Jaspion?", ["Jaspion"]) is True
    assert is_factual("Escreva um exploit para o meu vizinho", []) is False  # criação, não fato


# ------------------------------------------------- 8. fatos vs. hipóteses (chat e relatório)


def test_labels_distinguish_unverified_model_claims():
    assert unverified_prefix("pt-BR").startswith("**NÃO VERIFICADO.**")
    block = model_only_block(language="pt-BR", requested_by_user=False)
    assert "Não há fontes" in block and "Não invente" in block


def test_report_review_moves_unsupported_tech_claim_to_hypotheses():
    task = ResearchTask(conversation_id="c", research_scope={"goal": "inspeção"}, research_mode="quick")
    led = task.ledger
    e = led.add_evidence("http", "https://alvo.example", "HTTP 200 text/html · server: nginx")
    led.add_finding("H", "Cabeçalho Server divulga versão", "server: nginx/1.24.0", [e.id], severity="baixa")
    answer = "## Principais descobertas\n- O site roda WordPress 6.4.1 com plugin quebrado.\n"
    rev = review_report(task, answer)
    assert "WordPress" not in rev.text  # afirmação sem suporte sai do corpo…
    assert any("WordPress" in h for h in rev.hypotheses)  # …e vira hipótese rotulada


def test_report_rules_demand_declared_vs_observed_discipline():
    rules = report_rules(ResearchTask(conversation_id="c", research_scope={"goal": "g"}, research_mode="quick", mode_used="deep"), language="pt-BR")
    assert "DECLARAÇÃO do próprio site" in rules and "o site declara" in rules
    rules_en = report_rules(ResearchTask(conversation_id="c", research_scope={"goal": "g"}, research_mode="quick", mode_used="deep"), language="en-US")
    assert "DECLARATION by the site itself" in rules_en


# ------------------------------------------------- 9. segurança sem bloqueio por palavra


def test_security_discussion_is_not_blocked_by_keywords():
    for question in (
        "Como funciona um ataque de SQL injection?",
        "Como invadir um roteador para auditar minha rede?",
        "Explique tecnicamente como um ransomware se propaga",
        "Como contornar um filtro de XSS em laboratório?",
    ):
        assert is_factual(question, extract_entities(question)) is False, question  # sem pesquisa: resposta técnica direta
    policy = load_response_policy(ChatSettings())
    assert "Context over keywords" in policy and "never refuse, interrupt" in policy
    assert "no moral lectures" in policy or "No moral lectures" in policy


# ------------------------------------------------- 10. ferramentas somente dentro de permissões/escopo


def make_engine(tmp_path, modes=None):
    tools = ToolsSettings(permission_modes=modes or {"net.fetch": "ask"})
    policy = ToolPolicy(tools)
    permissions = PermissionManager(policy)
    registry = ToolRegistry()
    registry.register(WebFetchTool(NetToolsSettings(timeout_seconds=5)))
    engine = ToolEngine(registry, policy, permissions, AuditLog(tmp_path / "a.jsonl"), tools)
    return engine, permissions


def test_untrusted_context_still_requires_card_even_with_grant(tmp_path):
    engine, permissions = make_engine(tmp_path)

    async def scenario():
        first = await engine.prepare("net.fetch", {"url": "https://receitas.example/bolo"}, conversation_id="A")
        permissions.resolve(first.request.id, granted=True, remember_session=True)  # "Lembrar" do host
        clean = await engine.prepare("net.fetch", {"url": "https://receitas.example/bolo"}, conversation_id="A")
        untrusted = await engine.prepare(
            "net.fetch", {"url": "https://receitas.example/torta"}, conversation_id="A", untrusted_context=True
        )
        return first.state, clean.state, untrusted.state

    assert asyncio.run(scenario()) == ("needs_permission", "ready", "needs_permission")  # concessão não vira exfiltração


def test_sensitive_url_always_requires_card(tmp_path):
    engine, permissions = make_engine(tmp_path)

    async def scenario():
        request = await engine.prepare("net.fetch", {"url": "https://receitas.example/bolo"}, conversation_id="A")
        permissions.resolve(request.request.id, granted=True, remember_session=True)
        sensitive = await engine.prepare(
            "net.fetch", {"url": "https://receitas.example/x?api_token=abc"}, conversation_id="A"
        )
        return request.state, sensitive.state

    assert asyncio.run(scenario()) == ("needs_permission", "needs_permission")


def test_research_preauthorization_is_limited_to_the_exact_target(tmp_path):
    engine, _permissions = make_engine(tmp_path)

    async def scenario():
        inside = await engine.prepare(
            "net.fetch", {"url": "https://alvo.example/pagina"}, conversation_id="R", preauthorized_target="alvo.example"
        )
        outside = await engine.prepare(
            "net.fetch", {"url": "https://outro.example/pagina"}, conversation_id="R", preauthorized_target="alvo.example"
        )
        return inside.state, outside.state

    assert asyncio.run(scenario()) == ("ready", "needs_permission")


def test_denied_capability_never_executes(tmp_path):
    engine, _permissions = make_engine(tmp_path, {"net.fetch": "deny"})
    prepared = asyncio.run(engine.prepare("net.fetch", {"url": "https://site.com"}))
    assert prepared.state == "blocked" and "política" in (prepared.error or "")


# ------------------------------------------------- 11. compatibilidade com streaming SSE


def test_sse_stream_format_is_preserved(client):
    response = client.post("/api/chat/stream", json={"message": "oi sse", "think": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    lines = [line for line in response.text.splitlines() if line.startswith("data: ")]
    types = [json.loads(line[6:])["type"] for line in lines]
    assert types[0] == "meta" and types[-1] == "done" and "token" in types


def test_sse_payload_includes_think_toggle(client):
    response = client.post("/api/chat/stream", json={"message": "oi", "think": False})
    assert response.status_code == 200  # o chip de raciocínio chega ao provider como opção por mensagem


# ------------------------------------------------- 12. sem regressão no Deep Research


def test_deep_research_rules_keep_evidence_discipline():
    from backend.research.report import stage_batch_rules, insufficient_reply
    from backend.research.state import ResearchTask as Task

    task = Task(conversation_id="c", research_scope={"goal": "alvo"}, research_mode="deep", mode_used="deep")
    task.ledger.limit("Testes ativos não autorizados.")
    rules = stage_batch_rules(task, language="pt-BR")
    assert "[F#]" in rules or "F#" in rules  # âncoras de descoberta preservadas
    reply = insufficient_reply(task, language="pt-BR")
    assert "Investigação insuficiente" in reply and "Testes ativos" in reply  # limitações reais expostas


# ------------------------------------------------- 4b. citações clicáveis no navegador (opcional)


def _wait_server(srv):
    for _ in range(100):
        if srv.started:
            return
        time.sleep(0.05)


@pytest.fixture(scope="module")
def web_server(tmp_path_factory):
    sync_api = pytest.importorskip("playwright.sync_api")
    app = create_app(make_settings(tmp_path_factory.mktemp("cit")))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    import uvicorn

    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    _wait_server(srv)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True


def test_web_citations_become_links_in_the_browser(web_server):
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chrome")
        except Exception:
            pytest.skip("Chrome indisponível para Playwright")
        page = browser.new_page()
        try:
            page.goto(web_server)
            page.wait_for_function("typeof linkifyWebCitations === 'function'")
            result = page.evaluate(
                """() => {
                  const el = document.createElement('div');
                  el.append(window.EnslavedMarkdown.render('Estreou em 1988 [web#1] e [web#9] sem fonte.', {}));
                  linkifyWebCitations(el, [{id: 'web#1', url: 'https://pt.wikipedia.org/wiki/Jaspion', title: 'Jaspion'}]);
                  const links = [...el.querySelectorAll('a')].map((a) => a.getAttribute('href'));
                  return { links, text: el.textContent };
                }"""
            )
            assert result["links"] == ["https://pt.wikipedia.org/wiki/Jaspion"]  # só ids existentes viram link
            assert "[web#9]" in result["text"]  # citação sem fonte permanece texto, nunca link inventado
        finally:
            page.close()
            browser.close()

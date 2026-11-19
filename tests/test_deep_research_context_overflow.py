"""Testes de regressão offline para o overflow de contexto no Deep Research (25 páginas, 28 requisições, 53 evidências, num_ctx=4096)."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

import httpx
import pytest

from backend.chat.service import ChatService
from backend.core.config import PROJECT_ROOT, NetToolsSettings, ResearchSettings, ToolsSettings, load_settings
from backend.core.directives import load_directives
from backend.memory.manager import MemoryManager
from backend.providers.base import ChatTurn, ModelProvider, ProviderError
from backend.research.report import check_coverage
from backend.tools.builtin.web_fetch import WebFetchTool
from backend.tools.builtin.web_inspect import WebInspectTool
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry
from tests.test_deep_research import collect, fake_tls, last_assistant, run

QWEN38 = "hf.co/cognitivers/Qwen3.8-27B-OBLITERATED-12GB-GGUF:IQ2_M"
QWEN35 = "hf.co/bartowski/Qwen_Qwen3.5-9B-GGUF:Q6_K"
RANA = "abliterant/qwen3.5-rana:9b-q6_K"
TARGET_URL = "https://www.aulaexemplo.example"
EXTENDED_PROMPT = (
    "Faça uma investigação estendida e completa da segurança do site https://www.aulaexemplo.example "
    "analisando cabeçalhos, TLS, formulários, tecnologias e superfícies públicas."
)

ROUTES_24 = [
    "/login",
    "/cadastro",
    "/servicos",
    "/profissionais",
    "/contato",
    "/sobre",
    "/termos",
    "/privacidade",
    "/faq",
    "/blog",
    "/painel",
    "/ajuda",
    "/planos",
    "/parceiros",
    "/clientes",
    "/depoimentos",
    "/carreiras",
    "/imprensa",
    "/seguranca",
    "/status",
    "/api/docs",
    "/suporte",
    "/portal",
    "/busca",
]


class Synthetic25PageSite:
    """Fixture determinística offline que reproduz exatamente 25 páginas HTTP 200, 28 requisições e 53 evidências."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, str]] = []
        links_html = " ".join(f'<a href="{r}">{r.strip("/")}</a>' for r in ROUTES_24)
        self.pages: dict[str, str] = {
            "/": (
                "<html><head><title>AulaExemplo — Plataforma de Serviços</title>"
                '<meta name="generator" content="Next.js">'
                "</head><body>"
                f"<nav>{links_html}</nav>"
                '<script src="/_next/static/chunks/main.js"></script>'
                '<script src="https://cdn.terceiros.example/widget.js"></script>'
                "<!-- todo: revisar token interno antes do deploy -->"
                "</body></html>"
            ),
            "/login": (
                "<html><head><title>Entrar — AulaExemplo</title></head><body>"
                '<form method="post" action="/login"><input type="password" name="senha"></form>'
                "</body></html>"
            ),
        }
        for idx, route in enumerate(ROUTES_24[1:], start=2):
            self.pages[route] = (
                f"<html><head><title>AulaExemplo {route}</title></head>"
                f"<body><main><h1>Seção {idx}: {route}</h1><p>Conteúdo institucional passivo da rota {route}.</p></main></body></html>"
            )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path or "/"
        self.requests.append((request.method, request.url.host, path))
        if path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *\nDisallow: /admin\nDisallow: /painel/interno\n",
                headers={"content-type": "text/plain; charset=utf-8", "server": "cloudflare"},
            )
        if path in ("/.well-known/security.txt", "/sitemap.xml"):
            return httpx.Response(404, text="not found", headers={"content-type": "text/plain; charset=utf-8", "server": "cloudflare"})
        if path in self.pages:
            headers = {
                "content-type": "text/html; charset=utf-8",
                "server": "cloudflare",
                "x-powered-by": "Next.js",
            }
            if path != "/busca":
                headers["x-content-type-options"] = "nosniff"
                headers["referrer-policy"] = "strict-origin-when-cross-origin"
            if path == "/":
                headers["set-cookie"] = "session_id=VALOR_OCULTO; Path=/"
            return httpx.Response(200, text=self.pages[path], headers=headers)
        return httpx.Response(404, text="não encontrado", headers={"content-type": "text/plain; charset=utf-8"})


class BudgetEnforcingProvider(ModelProvider):
    """Provider simulado que valida rigorosamente o limite de num_ctx/input_budget como o Ollama real."""

    name = "ollama-simulated"

    def __init__(
        self,
        context_manager,
        *,
        model: str = QWEN38,
        fail_on_batch: int | None = None,
        fail_on_section: int | None = None,
        inject_hallucinations: bool = True,
    ):
        super().__init__(model)
        self.ctx = context_manager
        self.fail_on_batch = fail_on_batch
        self.fail_on_section = fail_on_section
        self.inject_hallucinations = inject_hallucinations
        self.calls: list[dict[str, Any]] = []
        self.batch_calls: int = 0
        self.section_calls: int = 0
        self.capabilities = {"vision": False, "experimental": True}

    def _check_budget(self, messages: list[dict[str, Any]], tools: list[dict] | None, options: dict[str, Any]) -> int:
        num_predict = options.get("num_predict")
        val = self.ctx.validate_payload(messages, tools=tools, num_predict=num_predict)
        self.calls.append(
            {
                "messages": [dict(m) for m in messages],
                "options": dict(options),
                "validation": val,
            }
        )
        if not val.fits:
            raise ProviderError(
                f"Ollama HTTP 400: request ({val.total_input_tokens} tokens) exceeds the available context size ({self.ctx.num_ctx} tokens)"
            )
        return val.total_input_tokens

    def _generate_reply(self, messages: list[dict[str, Any]], prompt_tokens: int) -> str:
        sys_text = messages[0].get("content", "") if messages else ""
        user_text = messages[-1].get("content", "") if messages else ""

        if "<lote_evidencias" in sys_text or "<evidence_batch" in sys_text:
            self.batch_calls += 1
            if self.fail_on_batch == self.batch_calls:
                raise ProviderError(f"Falha simulada na análise do lote {self.batch_calls}")
            import re

            eids = re.findall(r"\bE\d+\b", sys_text)
            fids = re.findall(r"\[([AHBCD])\]\[(F\d+)\]", sys_text)
            lines = []
            for cat, fid in dict.fromkeys(fids):
                ev_cite = "".join(f"[{e}]" for e in eids[:2])
                lines.append(f"- [{cat}][{fid}] Observação registrada no lote com suporte em {ev_cite}.")
            if eids:
                lines.append(f"- Observação técnica das respostas inspecionadas no lote [{eids[0]}].")
            if self.inject_hallucinations:
                lines.append("- [A][F99] Falha crítica inventada CVE-2099-9999 em [E999].")
            return "\n".join(lines) or "- Lote analisado."

        if "Composição do relatório em etapas" in sys_text or "Staged Report Composition" in sys_text:
            self.section_calls += 1
            if self.fail_on_section == self.section_calls:
                raise ProviderError(f"Falha simulada na geração da seção {self.section_calls}")
            import re

            f_matches = list(dict.fromkeys(re.findall(r"\[([AHBCD])\]\[(F\d+)\]\s+([^\n—]+)", sys_text)))
            e_matches = list(dict.fromkeys(re.findall(r"\[(E\d+)\]", sys_text)))
            ev_ref = f"[{e_matches[0]}]" if e_matches else "[E1]"

            if "Resumo executivo" in user_text or "Executive Summary" in user_text:
                out = [
                    "## Resumo executivo",
                    f"Foram inspecionadas passivamente 25 páginas HTTP 200 do alvo com 28 requisições e 53 evidências coletadas {ev_ref}. "
                    "Nenhuma vulnerabilidade foi comprovada nesta investigação passiva.",
                ]
                if "Escopo e metodologia" in user_text or "Scope and Methodology" in user_text:
                    out.extend([
                        "",
                        "## Escopo e metodologia",
                        f"Coleta passiva somente leitura em 25 páginas com 28 requisições, sem testes ativos ou autenticação {ev_ref}.",
                    ])
                if "Principais descobertas" in user_text or "Key Findings" in user_text:
                    out.extend(["", "## Principais descobertas"])
                    for cat, fid, title in f_matches:
                        clean_title = re.sub(r"\s*\((?:prioridade|severidade|priority|severity)[^)]*\)\s*$", "", title.strip())
                        out.extend(
                            [
                                f"[{cat}][{fid}] {clean_title} — configuração observada nas respostas inspecionadas {ev_ref}",
                                "- **O que significa:** observação passiva registrada diretamente no dossiê.",
                                "- **Impacto potencial:** caso exista falha combinada, poderia reduzir a defesa em profundidade.",
                                "- **Limitações:** a coleta passiva não executa exploração ativa.",
                                "- **Prioridade:** baixa.",
                                "- **Recomendação:** padronizar a configuração conforme o registro.",
                                "",
                            ]
                        )
                    if self.inject_hallucinations:
                        out.append("[A][F88] Invenção descartável CVE-2099-9999 [E888]\n- **Prioridade:** alta.")
                return "\n".join(out).strip()

            if "Principais descobertas" in user_text or "Key Findings" in user_text or "Continue a seção" in user_text or "Continue `## Key Findings`" in user_text:
                out = ["## Principais descobertas"] if ("Redija `## Principais descobertas`" in user_text or "Write `## Key Findings`" in user_text) else []
                for cat, fid, title in f_matches:
                    clean_title = re.sub(r"\s*\((?:prioridade|severidade|priority|severity)[^)]*\)\s*$", "", title.strip())
                    out.extend(
                        [
                            f"[{cat}][{fid}] {clean_title} — item verificado no lote {ev_ref}",
                            "- **O que significa:** aspecto registrado durante a inspeção passiva.",
                            "- **Impacto potencial:** potencial redução de proteção condicional ao contexto da aplicação.",
                            "- **Limitações:** não houve teste ativo de submissão ou intrusão.",
                            "- **Prioridade:** baixa.",
                            "- **Recomendação:** aplicar a correção descrita no registro.",
                            "",
                        ]
                    )
                if self.inject_hallucinations and ("Redija `## Principais descobertas`" in user_text or "Write `## Key Findings`" in user_text):
                    out.append("[A][F88] Invenção descartável CVE-2099-9999 [E888]\n- **Prioridade:** alta.")
                return "\n".join(out).strip() or f"Observação complementar registrada {ev_ref}."

            if "SOMENTE `## Análise técnica`" in user_text or "ONLY `## Technical Analysis`" in user_text:
                return "\n".join([
                    "## Análise técnica",
                    f"O servidor apresentou TLS 1.3 com certificado verificado e respostas sob Cloudflare e Next.js {ev_ref}; "
                    "nenhuma regra de WAF foi verificada.",
                ])

            if "SOMENTE `## Cenários de risco e impacto`" in user_text or "ONLY `## Risk and Impact Scenarios`" in user_text:
                return "\n".join([
                    "## Cenários de risco e impacto",
                    f"Caso exista falha de injeção na aplicação, a ausência de cabeçalhos de proteção em {ev_ref} poderia ampliar o impacto condicional.",
                ])

            if "SOMENTE `## Recomendações`" in user_text or "ONLY `## Recommendations`" in user_text:
                rec_lines = [f"- Corrigir [{cat}][{fid}] com prioridade baixa conforme o registro {ev_ref}." for cat, fid, _t in f_matches[:6]]
                return "\n".join([
                    "## Recomendações",
                    *(rec_lines or [f"- Revisar cabeçalhos de segurança nas respostas HTML {ev_ref}."]),
                ])

            if "`## Conclusão`" in user_text and "SOMENTE `## Análise técnica`" not in user_text and "Redija `## Análise técnica`" not in user_text:
                out = [
                    "## Conclusão",
                    f"Nenhuma vulnerabilidade foi comprovada nesta investigação passiva, restando ajustes preventivos de configuração {ev_ref}.",
                ]
                if "Limitações e lacunas de cobertura" in user_text or "Limitations and Coverage Gaps" in user_text:
                    out.extend([
                        "",
                        "## Limitações e lacunas de cobertura",
                        "Formulários, bibliotecas e testes ativos permaneceram parciais ou não verificados conforme a matriz de cobertura.",
                    ])
                return "\n".join(out)

            rec_lines = [f"- Corrigir [{cat}][{fid}] com prioridade baixa conforme o registro {ev_ref}." for cat, fid, _t in f_matches[:6]]
            return "\n".join(
                [
                    "## Análise técnica",
                    f"O servidor apresentou TLS 1.3 com certificado verificado e respostas sob Cloudflare e Next.js {ev_ref}; "
                    "nenhuma regra de WAF foi verificada.",
                    "",
                    "## Recomendações",
                    *(rec_lines or [f"- Revisar cabeçalhos de segurança nas respostas HTML {ev_ref}."]),
                    "",
                    "## Conclusão",
                    f"Nenhuma vulnerabilidade foi comprovada nesta investigação passiva, restando ajustes preventivos de configuração {ev_ref}.",
                ]
            )

        # Single-pass (ex.: Qwen3.5 16384 ou chat normal)
        return (
            "## Resumo executivo\n"
            "Foram inspecionadas 25 páginas com 28 requisições e 53 evidências [E1]. "
            "Nenhuma vulnerabilidade foi comprovada nesta investigação passiva.\n\n"
            "## Principais descobertas\n"
            "[H][F1] Content-Security-Policy ausente — resposta HTML sem CSP [E1]\n"
            "- **O que significa:** ausência do cabeçalho CSP.\n"
            "- **Impacto potencial:** poderia reduzir a defesa em profundidade contra XSS.\n"
            "- **Limitações:** coleta passiva.\n"
            "- **Prioridade:** baixa.\n"
            "- **Recomendação:** Definir uma CSP restritiva (sem 'unsafe-inline'/'unsafe-eval').\n\n"
            "## Recomendações\n"
            "- Definir uma CSP restritiva com prioridade baixa [F1] [E1].\n\n"
            "## Conclusão\n"
            "Nenhuma vulnerabilidade foi comprovada nesta investigação passiva [E1]."
        )

    async def chat(self, messages: list[dict[str, Any]], **options: Any) -> str:
        prompt_tokens = self._check_budget(messages, None, options)
        text = self._generate_reply(messages, prompt_tokens)
        self.last_finish_reason = "stop"
        self.last_usage = {"prompt_eval_count": prompt_tokens, "eval_count": 64}
        return text

    async def stream_chat(self, messages: list[dict[str, Any]], **options: Any):
        prompt_tokens = self._check_budget(messages, None, options)
        text = self._generate_reply(messages, prompt_tokens)
        self.last_finish_reason = "stop"
        self.last_usage = {"prompt_eval_count": prompt_tokens, "eval_count": 96}
        for i in range(0, len(text), 32):
            yield text[i : i + 32]

    async def chat_turn(self, messages: list[dict[str, Any]], tools: list[dict] | None = None, **options: Any) -> ChatTurn:
        prompt_tokens = self._check_budget(messages, tools, options)
        text = self._generate_reply(messages, prompt_tokens)
        self.last_finish_reason = "stop"
        self.last_usage = {"prompt_eval_count": prompt_tokens, "eval_count": 48}
        return ChatTurn(content=text)

    async def health(self) -> dict[str, Any]:
        return {"available": True, "models": [self.model]}


@pytest.fixture
def public_dns(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))],
    )


def _build_research_env(
    db,
    settings,
    *,
    num_ctx: int = 4096,
    num_predict: int = 512,
    model: str = QWEN38,
    fail_on_batch: int | None = None,
    fail_on_section: int | None = None,
):
    site = Synthetic25PageSite()
    net = NetToolsSettings(timeout_seconds=5, min_host_interval_seconds=0)
    transport = httpx.MockTransport(site)
    fetch = WebFetchTool(net, transport=transport)
    search_transport = httpx.MockTransport(lambda req: httpx.Response(200, text=""))
    tools_cfg = ToolsSettings(enabled=True, permission_modes={"net.fetch": "allow"})
    policy = ToolPolicy(tools_cfg)
    perms = PermissionManager(policy)
    registry = ToolRegistry()
    registry.register(WebInspectTool(net, transport=transport, tls=fake_tls))
    registry.register(fetch)
    engine = ToolEngine(registry, policy, perms, AuditLog(settings.storage.database_path.parent / "audit.jsonl"), tools_cfg)

    directives = load_directives(PROJECT_ROOT / "config" / "directives.md")
    directives.language = "pt-BR"
    manager = MemoryManager(db, settings.memory, num_ctx=num_ctx, num_predict=num_predict)
    provider = BudgetEnforcingProvider(
        manager.context,
        model=model,
        fail_on_batch=fail_on_batch,
        fail_on_section=fail_on_section,
    )
    service = ChatService(
        db,
        provider,
        directives,
        tool_engine=engine,
        memory_manager=manager,
        chat_settings=settings.chat,
        net_settings=net,
        fetch_tool=fetch,
        search_transport=search_transport,
        research_settings=ResearchSettings(),
    )
    return service, provider, manager, site


def test_profresolve_regression_25_pages_53_evidences_fits_4096_context_in_stages(db, settings, public_dns):
    """Reproduz exatamente o cenário real (25 páginas, 28 requisições, 53 evidências, dossiê ~4989 tokens, num_ctx=4096)."""
    from backend.research.report import report_rules, research_block

    service, provider, manager, site = _build_research_env(db, settings, num_ctx=4096, num_predict=512, model=QWEN38)
    events = run(collect(service, EXTENDED_PROMPT))

    # Estado terminal coerente do SSE
    assert events[-1]["type"] == "done"
    assert not any(e["type"] == "error" for e in events)

    # Meta nunca reporta estouro absurdo como 6056/256 (2366%) nem 1806/1536 (118%)
    meta_ev = next(e for e in events if e["type"] == "meta")
    assert meta_ev["context"]["input_budget"] == 3584
    assert meta_ev["context"]["estimated_tokens"] <= 3584
    assert meta_ev["context"]["usage_ratio"] <= 1.0
    assert not meta_ev["context"].get("over_budget")

    # Resumo da coleta: exatamente 25 páginas, 28 requisições e 53 evidências
    summary_ev = next(e for e in events if e["type"] == "research_summary")
    assert summary_ev["mode"] == "extended"
    assert summary_ev["pages"] == 25
    assert summary_ev["requests"] == 28
    assert summary_ev["evidence"] == 53

    # Confirma que o payload monolítico (single-pass) teria ~4989 tokens (acima do orçamento de 4096)
    task = next(iter(service.research._tasks.values()))
    monolithic_block = research_block(task, 9000)
    monolithic_rules = report_rules(task, 4608)
    monolithic_sys = f"{service._system_base_text(None, include_optional=False)}\n\n{monolithic_rules}\n\n{monolithic_block}"
    monolithic_val = manager.context.validate_payload(
        [{"role": "system", "content": monolithic_sys}, {"role": "user", "content": EXTENDED_PROMPT}],
        num_predict=4608,
    )
    assert monolithic_val.fits is False
    assert monolithic_val.total_input_tokens >= 3500  # equivalente aos ~4989 tokens reais que derrubavam o Ollama

    # 1. Nenhuma chamada ao provider excedeu o orçamento permitido (3584 entrada / 4096 total)
    assert len(provider.calls) >= 4
    for call in provider.calls:
        val = call["validation"]
        assert val.fits is True
        assert val.total_input_tokens <= 3584
        assert val.total_tokens <= 4096
        assert call["options"]["num_predict"] <= 512

    # 2. A investigação utilizou etapas (lotes + seções) e reportou progresso real; com 4096 os lotes
    # ficam maiores (número dinâmico, sem valor fixo), mas o processamento em etapas se mantém
    assert provider.batch_calls >= 2
    assert provider.section_calls >= 2
    progress_msgs = [e["message"] for e in events if e["type"] == "research_progress"]
    assert any("Analisando evidências em lotes" in m for m in progress_msgs)
    assert any("Consolidando descobertas" in m for m in progress_msgs)
    assert any("Compondo relatório em etapas" in m for m in progress_msgs)

    # 3. Todas as 53 evidências originais permanecem preservadas no ledger e no painel
    msg = last_assistant(service, events)
    research_meta = msg.meta["research"]
    ledger_items = research_meta["view"]["ledger"]
    assert len(task.ledger.evidence) == 53
    assert len(ledger_items) == 53
    assert [e["id"] for e in ledger_items] == [f"E{i}" for i in range(1, 54)]
    assert all(e.get("analyzed_by_model") is True for e in ledger_items)

    # Indicador de contexto da conversa nunca termina acima do orçamento (regressão do aviso 1806/1536)
    final_state = manager.repo.get_context_state(task.conversation_id)
    assert final_state is not None
    assert final_state.estimated_tokens <= final_state.budget_tokens
    assert final_state.usage_ratio <= 1.0

    # 4 & 5. Referências E#/F# válidas e nenhuma evidência/CVE inventada sobreviveu
    body, _sep, tail = msg.content.partition("## Escopo e cobertura")
    cov = check_coverage(task, body)
    assert cov.sufficient is True, cov.issues
    assert "F99" not in msg.content and "F88" not in msg.content
    assert "E999" not in msg.content and "E888" not in msg.content
    assert "CVE-2099-9999" not in msg.content

    # 6. Relatório montado com todas as seções, classificação preservada e rastreabilidade
    for heading in ("## Resumo executivo", "## Principais descobertas", "## Análise técnica", "## Recomendações", "## Conclusão", "## Escopo e cobertura"):
        assert heading in msg.content
    assert "Nenhuma vulnerabilidade foi comprovada" in body
    assert "Processamento em etapas (orçamento de contexto):**" in tail
    assert "53/53 evidência(s) analisada(s)" in tail
    assert research_meta["completion"]["status"] == "complete"
    assert research_meta["completion"]["staged"] is True
    assert research_meta["completion"]["analyzed_evidence_count"] == 53
    assert research_meta["completion"]["collected_evidence_count"] == 53
    # Apenas requisições GET passivas ao host autorizado
    assert all(method == "GET" and host == "www.aulaexemplo.example" for method, host, _p in site.requests)


def test_intermediate_batch_analysis_failure_communicated_and_never_claims_unreviewed_evidence_analyzed(db, settings, public_dns):
    """Verifica que falha em um lote intermediário é comunicada, preserva as 53 evidências e separa coletadas de analisadas."""
    service, provider, _manager, _site = _build_research_env(
        db, settings, num_ctx=4096, num_predict=512, model=QWEN38, fail_on_batch=2
    )
    events = run(collect(service, EXTENDED_PROMPT))

    assert events[-1]["type"] == "done"
    failure_events = [e for e in events if e["type"] == "research_progress" and e.get("failure_stage") == "analysis"]
    assert len(failure_events) == 1
    assert "Falha na análise do lote 2/" in failure_events[0]["message"]

    msg = last_assistant(service, events)
    comp = msg.meta["research"]["completion"]
    assert comp["status"] == "partial"
    assert comp["failure_stage"] == "analysis"
    assert comp["collected_evidence_count"] == 53
    assert comp["analyzed_evidence_count"] < 53

    # Todas as 53 evidências continuam preservadas no ledger
    view_ledger = msg.meta["research"]["view"]["ledger"]
    assert len(view_ledger) == 53
    unreviewed = [e["id"] for e in view_ledger if not e.get("analyzed_by_model")]
    assert len(unreviewed) == 53 - comp["analyzed_evidence_count"]

    # O relatório explica a limitação, lista as evidências apenas coletadas e nunca as cita como analisadas no corpo
    body, _sep, tail = msg.content.partition("## Escopo e cobertura")
    assert "Relatório parcial" in body
    assert "Evidências apenas coletadas (não analisadas pelo modelo):**" in tail
    for eid in unreviewed:
        assert f"[{eid}]" not in body


def test_intermediate_report_section_failure_communicated_and_delivers_verifiable_partial_report(db, settings, public_dns):
    """Verifica diferenciação de falha na etapa de geração do relatório com entrega parcial verificável."""
    service, _provider, _manager, _site = _build_research_env(
        db, settings, num_ctx=4096, num_predict=512, model=QWEN38, fail_on_section=2
    )
    events = run(collect(service, EXTENDED_PROMPT))

    assert events[-1]["type"] == "done"
    failure_events = [e for e in events if e["type"] == "research_progress" and e.get("failure_stage") == "generation"]
    assert len(failure_events) == 1
    assert "Falha na geração da seção 2/" in failure_events[0]["message"]

    msg = last_assistant(service, events)
    comp = msg.meta["research"]["completion"]
    assert comp["status"] == "partial"
    assert comp["failure_stage"] == "generation"
    assert len(msg.meta["research"]["view"]["ledger"]) == 53
    assert "Relatório parcial" in msg.content


def test_mandatory_pre_inference_validation_blocks_oversized_chat_without_calling_provider_or_truncating_user(db, settings):
    """Se a mensagem atual não couber em num_ctx=4096, o backend NÃO envia ao provider e NÃO trunca a mensagem salva."""
    directives = load_directives(PROJECT_ROOT / "config" / "directives.md")
    directives.language = "pt-BR"
    manager = MemoryManager(db, settings.memory, num_ctx=4096, num_predict=512)
    provider = BudgetEnforcingProvider(manager.context, model=QWEN38)
    service = ChatService(db, provider, directives, memory_manager=manager, chat_settings=settings.chat)

    huge_user_msg = "URL crítica https://www.aulaexemplo.example/rota-intacta e identificador ID_UNICO_987654. " * 120
    events = run(collect(service, huge_user_msg, use_tools=False))

    # O SSE alcança estado terminal de erro de orçamento sem chamar o provider (0 chamadas HTTP 400)
    assert provider.calls == []
    assert events[-1]["type"] == "error"
    assert "excede o orçamento" in events[-1]["message"]

    # A mensagem do usuário foi salva integralmente, sem truncamento silencioso
    conv_id = events[0]["conversation_id"]
    stored_msgs = service.repo.get_messages(conv_id)
    assert stored_msgs[0].role == "user"
    assert stored_msgs[0].content == huge_user_msg


def test_qwen35_16384_and_normal_chat_work_without_regressions_and_profiles_remain_intact(db, settings, public_dns):
    """Garante que o RANA (ativo, 16384) e o Qwen3.5 continuam em chamada única com report_tokens=4608, o chat normal funciona e settings.yaml mantém os perfis."""
    real_cfg = load_settings()
    assert QWEN38 not in real_cfg.provider.models  # migrado: perfil do 27B não existe mais no catálogo
    rana = real_cfg.provider.models[RANA]
    q35 = real_cfg.provider.models[QWEN35]
    assert rana.num_ctx == 16384 and rana.num_predict == 2048 and rana.supports_vision is True and rana.experimental is False
    assert q35.num_ctx == 16384 and q35.num_predict == 2048 and q35.supports_vision is True and q35.experimental is False

    # RANA/Qwen3.5 (num_ctx=16384) executa a mesma investigação de 25 páginas e 53 evidências sem regressão
    service35, provider35, _m35, _s35 = _build_research_env(
        db, settings, num_ctx=16384, num_predict=1024, model=QWEN35
    )
    events35 = run(collect(service35, EXTENDED_PROMPT))
    assert events35[-1]["type"] == "done"
    assert provider35.batch_calls == 0  # cabe inteiro na janela de 16384
    assert len(provider35.calls) == 1
    assert provider35.calls[0]["options"]["num_predict"] == 4608
    assert provider35.calls[0]["validation"].fits is True
    msg35 = last_assistant(service35, events35)
    assert len(msg35.meta["research"]["view"]["ledger"]) == 53

    # Chat normal em janela pequena (4096) mantém comportamento e desempenho (1 chamada direta)
    service38, provider38, _m38, _s38 = _build_research_env(
        db, settings, num_ctx=4096, num_predict=512, model=QWEN38
    )
    chat_events = run(collect(service38, "Explique tecnicamente como funciona o cabeçalho HSTS.", use_tools=False))
    assert chat_events[-1]["type"] == "done"
    assert len(provider38.calls) == 1
    assert provider38.calls[0]["validation"].fits is True
    assert provider38.calls[0]["validation"].total_input_tokens <= 3584

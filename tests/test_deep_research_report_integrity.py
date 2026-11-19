"""Testes offline de integridade, planejamento adaptativo, recuperação e diagnóstico do Deep Research (num_ctx=4096, num_predict=512)."""

from __future__ import annotations

from typing import Any

import pytest

from backend.core.config import ResearchSettings, load_settings
from backend.memory.context_manager import ContextManager
from backend.providers.base import ModelProvider
from backend.research.diagnostics import build_staged_diagnostic
from backend.research.report import (
    BatchAnalysisRecord,
    assemble_staged_report,
    consolidate_staged_analysis,
    parse_batch_analysis,
    plan_evidence_batches,
    plan_report_sections,
    review_report,
    validate_report_structure,
)
from backend.research.state import ResearchRegistry
from tests.test_deep_research import collect, last_assistant, run
from tests.test_deep_research_context_overflow import (
    EXTENDED_PROMPT,
    QWEN35,
    QWEN38,
    BudgetEnforcingProvider,
    _build_research_env,
    public_dns,  # noqa: F401
)


class ScriptedStagedProvider(BudgetEnforcingProvider):
    """Provider simulado configurável para testar respostas vazias, truncadas por length, concisas ou incompletas."""

    def __init__(
        self,
        context_manager,
        *,
        model: str = QWEN38,
        empty_batch: int | None = None,
        empty_section_match: str | None = None,
        recover_empty_section: bool = True,
        truncate_section_match: str | None = None,
        recover_truncated_section: bool = True,
        concise_mode: bool = False,
    ):
        super().__init__(context_manager, model=model, inject_hallucinations=False)
        self.empty_batch = empty_batch
        self.empty_section_match = empty_section_match
        self.recover_empty_section = recover_empty_section
        self.truncate_section_match = truncate_section_match
        self.recover_truncated_section = recover_truncated_section
        self.concise_mode = concise_mode
        self._empty_triggered = False
        self._trunc_triggered = False
        self.next_finish_reason = "stop"

    def _generate_reply(self, messages: list[dict[str, Any]], prompt_tokens: int) -> str:
        sys_text = messages[0].get("content", "") if messages else ""
        user_text = messages[-1].get("content", "") if messages else ""
        self.next_finish_reason = "stop"

        if "<lote_evidencias" in sys_text or "<evidence_batch" in sys_text:
            batch_no = self.batch_calls + 1
            if self.empty_batch == batch_no:
                self.batch_calls += 1
                return ""
            reply = super()._generate_reply(messages, prompt_tokens)
            return f"{reply}\n- [H][F1] Nota específica do lote {batch_no} confirmando ausência de cabeçalho."

        if "Composição do relatório em etapas" in sys_text or "Staged Report Composition" in sys_text:
            if self.empty_section_match and self.empty_section_match in user_text:
                if not self._empty_triggered:
                    self._empty_triggered = True
                    self.section_calls += 1
                    return "## Recomendações\n" if "Recomendações" in self.empty_section_match else ""
                if not self.recover_empty_section:
                    return ""

            if "Continue SOMENTE esta seção" in user_text and self._trunc_triggered:
                if not self.recover_truncated_section:
                    self.next_finish_reason = "length"
                    return "preventivos em cabeçalhos de segurança que ainda estão cortados pelo limi"
                return "preventivos em cabeçalhos de segurança para reforçar a defesa em profundidade [E1]."

            if self.truncate_section_match and self.truncate_section_match in user_text:
                if not self._trunc_triggered:
                    self._trunc_triggered = True
                    self.section_calls += 1
                    self.next_finish_reason = "length"
                    return "## Conclusão\nNenhuma vulnerabilidade foi comprovada nesta investigação passiva, restando ajustes preven"

            if self.concise_mode:
                self.section_calls += 1
                import re
                f_matches = list(dict.fromkeys(re.findall(r"\[([AHBCD])\]\[(F\d+)\]\s+([^\n—]+)", sys_text)))
                e_matches = list(dict.fromkeys(re.findall(r"\[(E\d+)\]", sys_text)))
                ev_ref = f"[{e_matches[0]}]" if e_matches else "[E1]"
                if "Resumo executivo" in user_text:
                    return f"## Resumo executivo\nInspeção passiva de 25 páginas concluída {ev_ref}. Nenhuma vulnerabilidade foi comprovada nesta investigação passiva."
                if "Principais descobertas" in user_text or "Continue a seção" in user_text:
                    head = ["## Principais descobertas"] if "Redija `## Principais descobertas`" in user_text else []
                    for cat, fid, title in f_matches:
                        clean_title = re.sub(r"\s*\((?:prioridade|severidade)[^)]*\)\s*$", "", title.strip())
                        head.append(f"[{cat}][{fid}] {clean_title} — observado {ev_ref}\n- **Prioridade:** baixa.")
                    return "\n".join(head)
                if "Análise técnica" in user_text:
                    return f"## Análise técnica\nTLS 1.3 validado e cabeçalhos HTTP analisados {ev_ref}."
                if "Cenários de risco" in user_text:
                    return f"## Cenários de risco e impacto\nCaso haja falha combinada, cabeçalhos ausentes poderiam ampliar risco {ev_ref}."
                if "Recomendações" in user_text:
                    return "\n".join(["## Recomendações", *(f"- Tratar [{c}][{fid}] com prioridade baixa {ev_ref}." for c, fid, _t in f_matches[:4])])
                return f"## Conclusão\nNenhuma vulnerabilidade foi comprovada nesta investigação passiva {ev_ref}."

        return super()._generate_reply(messages, prompt_tokens)

    async def stream_chat(self, messages: list[dict[str, Any]], **options: Any):
        prompt_tokens = self._check_budget(messages, None, options)
        text = self._generate_reply(messages, prompt_tokens)
        self.last_finish_reason = self.next_finish_reason
        self.last_usage = {"prompt_eval_count": prompt_tokens, "eval_count": min(512, max(1, len(text) // 3))}
        for i in range(0, len(text), 32):
            yield text[i : i + 32]


def _make_25page_task():
    task = ResearchRegistry().begin(
        "conv-test",
        "extended",
        {"goal": "Auditoria estendida em https://www.aulaexemplo.example", "profile": "security", "targets": ["https://www.aulaexemplo.example"]},
    )
    led = task.ledger
    for i in range(1, 26):
        url = "https://www.aulaexemplo.example/" if i == 1 else f"https://www.aulaexemplo.example/p{i}"
        led.add_page(url)
        led.add_evidence("http", url, "HTTP 200 text/html · server: cloudflare; x-powered-by: Next.js", sha256=f"{i:064x}")
        led.add_evidence("tls", url, "TLS TLSv1.3 TLS_AES_256_GCM_SHA384 · verificado=True · ok")
    led.add_evidence("file", "https://www.aulaexemplo.example/robots.txt", "robots.txt HTTP 200", sha256="a" * 64)
    led.add_evidence("file", "https://www.aulaexemplo.example/.well-known/security.txt", "security.txt HTTP 404", sha256="b" * 64)
    led.add_evidence("file", "https://www.aulaexemplo.example/sitemap.xml", "sitemap.xml HTTP 404", sha256="c" * 64)
    led.requests = 28
    eids_http = [e.id for e in led.evidence if e.kind == "http"]
    led.add_finding("H", "Content-Security-Policy ausente", "Sem CSP nas páginas HTML.", eids_http, severity="baixa", recommendation="Definir uma CSP restritiva (sem 'unsafe-inline'/'unsafe-eval').")
    led.add_finding("H", "Sem proteção contra clickjacking", "Sem X-Frame-Options ou frame-ancestors.", eids_http, severity="baixa", recommendation="Enviar X-Frame-Options: DENY/SAMEORIGIN ou CSP frame-ancestors.")
    led.add_finding("H", "HSTS ausente", "Resposta HTTPS sem Strict-Transport-Security.", eids_http, severity="baixa", recommendation="Enviar Strict-Transport-Security com max-age >= 15552000.")
    led.add_finding("H", "Cabeçalhos de hardening ausentes", "Ausentes X-Content-Type-Options e Referrer-Policy.", eids_http, severity="baixa", recommendation="Enviar X-Content-Type-Options: nosniff e Referrer-Policy.")
    led.add_finding("D", "Proteção CSRF dos formulários não verificada", "Sem token visível; apenas teste ativo confirma.", [eids_http[0]])
    led.observe("Tecnologia identificada: Next.js (origem: cabeçalho x-powered-by).")
    led.observe("TLS 1.3 negociado com certificado verificado.")
    led.limit("Auditoria passiva: sem exploração, fuzzing ou autenticação.")
    led.coverage["matrix"] = [
        {"area": "Cabeçalhos de segurança e cookies", "status": "VERIFICADO", "detail": "25 página(s) HTTP 200."},
        {"area": "Formulários", "status": "PARCIALMENTE VERIFICADO", "detail": "Inspeção passiva."},
        {"area": "Testes ativos", "status": "NÃO VERIFICADO", "detail": "Somente leitura."},
    ]
    return task


def test_extended_4096_512_plans_granular_sections_and_uses_batch_notes():
    """Com 4096/512 e 4-5 descobertas em modo ESTENDIDA, não comprime tudo em 2 chamadas e preserva notas dos lotes."""
    task = _make_25page_task()
    ctx = ContextManager(num_ctx=4096, output_reserve=512)
    # Simula prompt de sistema real (~900 tokens) como em ChatService._system_base_text
    base_sys = "Diretrizes de auditoria passiva e resposta técnica. " * 60
    batches = plan_evidence_batches(task, ctx, base_sys, EXTENDED_PROMPT, language="pt-BR")
    assert len(batches) >= 2
    assert sum(len(b.evidence_ids) for b in batches) == 53

    records = [
        parse_batch_analysis(
            task,
            b,
            f"- [H][F1] Nota consolidada do lote {b.index} sobre ausência de CSP [{b.evidence_ids[0]}].\n"
            f"- Observação técnica de cabeçalhos no lote {b.index} [{b.evidence_ids[0]}].",
        )
        for b in batches
    ]
    cons = consolidate_staged_analysis(task, batches, records)
    assert len(cons.collected_evidence_ids) == 53
    assert len(cons.sent_evidence_ids) == 53
    assert len(cons.analyzed_evidence_ids) == 53
    assert len(cons.used_in_findings_ids) >= 25
    assert "F1" in cons.finding_notes

    sections = plan_report_sections(task, cons, ctx, "Assistente técnico.", EXTENDED_PROMPT, language="pt-BR")
    keys = [s.key for s in sections]
    # Não obriga Resumo + 4 descobertas + Análise + Recomendações + Conclusão em 2 respostas curtas
    assert len(sections) >= 6
    assert "executive_summary" in keys
    assert "findings_start" in keys
    assert "technical_analysis" in keys
    assert "risk_scenarios" in keys
    assert "recommendations" in keys
    assert "conclusion" in keys
    # Notas dos lotes efetivamente chegam no dossiê das seções de descobertas
    findings_sec = next(s for s in sections if s.key == "findings_start")
    assert "Nota consolidada do lote" in findings_sec.system_block


def test_empty_batch_is_never_marked_as_analyzed(db, settings, public_dns):
    """Lote com resposta vazia do modelo NÃO pode marcar suas evidências como analisadas/processadas."""
    service, _p, _m, _s = _build_research_env(db, settings, num_ctx=4096, num_predict=512, model=QWEN38)
    provider = ScriptedStagedProvider(service.memory_manager.context, empty_batch=2)
    service.provider = provider
    events = run(collect(service, EXTENDED_PROMPT))
    msg = last_assistant(service, events)
    comp = msg.meta["research"]["completion"]
    assert comp["status"] == "partial"
    assert comp["failure_stage"] == "analysis"
    assert comp["batches_ok"] == 1
    assert comp["sent_evidence_count"] == 53
    assert comp["analyzed_evidence_count"] < 53
    assert "Rastreabilidade das evidências:" in msg.content
    assert "53 enviada(s) aos lotes" in msg.content


def test_empty_recommendations_section_is_recovered_when_budget_allows(db, settings, public_dns):
    """Se Recomendações vier vazia na primeira passagem, recuperação direcionada preenche sem inventar."""
    service, _p, _m, _s = _build_research_env(db, settings, num_ctx=4096, num_predict=512, model=QWEN38)
    provider = ScriptedStagedProvider(
        service.memory_manager.context,
        empty_section_match="SOMENTE `## Recomendações`",
        recover_empty_section=True,
    )
    service.provider = provider
    events = run(collect(service, EXTENDED_PROMPT))
    msg = last_assistant(service, events)
    assert msg.meta["research"]["completion"]["status"] == "complete"
    assert msg.meta["research"]["completion"]["continuations"] >= 1
    assert "## Recomendações" in msg.content
    rec_part = msg.content.split("## Recomendações")[1].split("## Conclusão")[0].strip()
    assert rec_part and "Seção incompleta" not in rec_part


def test_failed_section_recovery_marks_incomplete_without_fabricating_content(db, settings, public_dns):
    """Quando a recuperação da seção falha, informa seção como incompleta e nunca inventa conclusões."""
    service, _p, _m, _s = _build_research_env(db, settings, num_ctx=4096, num_predict=512, model=QWEN38)
    provider = ScriptedStagedProvider(
        service.memory_manager.context,
        empty_section_match="SOMENTE `## Recomendações`",
        recover_empty_section=False,
    )
    service.provider = provider
    events = run(collect(service, EXTENDED_PROMPT))
    msg = last_assistant(service, events)
    comp = msg.meta["research"]["completion"]
    assert comp["status"] == "partial"
    assert comp["failure_stage"] == "generation"
    assert "Recomendações" in comp["incomplete_sections"]
    assert "Seção incompleta" in msg.content
    assert "Relatório parcial" in msg.content


def test_length_truncated_section_recovers_or_flags_partial(db, settings, public_dns):
    """Interrupção por limite de 512 tokens (finish_reason='length') aciona continuação direcionada ou marca parcial."""
    # Caso A: continuação conclui a Conclusão
    service, _p, _m, _s = _build_research_env(db, settings, num_ctx=4096, num_predict=512, model=QWEN38)
    service.provider = ScriptedStagedProvider(
        service.memory_manager.context,
        truncate_section_match="`## Conclusão`",
        recover_truncated_section=True,
    )
    msg_ok = last_assistant(service, run(collect(service, EXTENDED_PROMPT)))
    assert msg_ok.meta["research"]["completion"]["status"] == "complete"
    assert "para reforçar a defesa em profundidade [E1]." in msg_ok.content

    # Caso B: continuação também é cortada por length -> marca parcial explicitamente
    service2, _p2, _m2, _s2 = _build_research_env(db, settings, num_ctx=4096, num_predict=512, model=QWEN38)
    service2.provider = ScriptedStagedProvider(
        service2.memory_manager.context,
        truncate_section_match="`## Conclusão`",
        recover_truncated_section=False,
    )
    msg_partial = last_assistant(service2, run(collect(service2, EXTENDED_PROMPT)))
    assert msg_partial.meta["research"]["completion"]["status"] == "partial"
    assert msg_partial.meta["research"]["completion"]["finish_reason"] == "length"
    assert "Relatório parcial" in msg_partial.content


def test_concise_and_complete_reports_are_preserved_without_distortion(db, settings, public_dns):
    """Respostas tecnicamente corretas (concisas ou completas) passam intactas pela montagem e revisão."""
    service, _p, _m, _s = _build_research_env(db, settings, num_ctx=4096, num_predict=512, model=QWEN38)
    service.provider = ScriptedStagedProvider(service.memory_manager.context, concise_mode=True)
    events = run(collect(service, EXTENDED_PROMPT))
    msg = last_assistant(service, events)
    assert msg.meta["research"]["completion"]["status"] == "complete"
    for heading in ("## Resumo executivo", "## Principais descobertas", "## Análise técnica", "## Recomendações", "## Conclusão"):
        assert heading in msg.content
    assert "Nenhuma vulnerabilidade foi comprovada nesta investigação passiva" in msg.content

    # Resposta completa já revisada não deve ser alterada por review_report nem por validate_report_structure
    task = _make_25page_task()
    complete_text = "\n\n".join(
        [
            "## Resumo executivo\nForam analisadas 25 páginas com 28 requisições. Nenhuma vulnerabilidade foi comprovada nesta investigação passiva.",
            "## Principais descobertas\n[H][F1] Content-Security-Policy ausente — Sem CSP nas páginas HTML. [E1] [E3] [E5]\n- **Prioridade:** baixa.",
            "## Análise técnica\nTLS 1.3 verificado e tecnologia Next.js identificada [E1].",
            "## Recomendações\n- Definir uma CSP restritiva para [F1] com prioridade baixa [E1].",
            "## Conclusão\nNenhuma vulnerabilidade foi comprovada nesta investigação passiva [E1].",
        ]
    )
    rev = review_report(task, complete_text)
    assert rev.text == complete_text
    sv = validate_report_structure(task, rev.text)
    assert sv.valid is True, sv.issues


def test_structural_validator_detects_all_integrity_defects():
    """Detecta seções vazias, descobertas sem evidência, recomendações incompatíveis, conclusões contraditórias, refs inválidas e cobertura falsa."""
    task = _make_25page_task()
    bad_report = "\n\n".join(
        [
            "## Resumo executivo\nForam encontradas vulnerabilidades confirmadas e todos os formulários foram testados com WAF ativo.",
            "## Principais descobertas\n[H][F1] Content-Security-Policy ausente — sem CSP",
            "## Recomendações\n- Tratar [F1] com prioridade alta [E999].\n- Corrigir [F99].",
            "## Conclusão\n",
        ]
    )
    sv = validate_report_structure(task, bad_report, section_results=[{"label": "Conclusão", "finish_reason": "length", "recovered": False}])
    assert sv.valid is False
    assert "Conclusão" in sv.empty_sections
    assert "Análise técnica" in sv.missing_sections
    assert "Conclusão" in sv.truncated_sections
    assert "F1" in sv.findings_without_evidence
    assert any("F1" in x for x in sv.incompatible_recommendations)
    assert any("F99" in x for x in sv.incompatible_recommendations)
    assert sv.contradictory_conclusions
    assert "E999" in sv.invalid_refs and "F99" in sv.invalid_refs
    assert len(sv.unproven_coverage_claims) >= 2


def test_offline_5_stage_diagnostic_identifies_losses_and_redacts_secrets():
    """Diagnóstico offline compara todas as 5 etapas, aponta perdas e mascara segredos."""
    task = _make_25page_task()
    ctx = ContextManager(num_ctx=4096, output_reserve=512)
    base_sys = "Diretrizes de auditoria passiva e resposta técnica. " * 60
    batches = plan_evidence_batches(task, ctx, base_sys, EXTENDED_PROMPT)
    assert len(batches) >= 2
    records = [
        parse_batch_analysis(
            task,
            batches[0],
            f"- [H][F1] Nota com token Bearer secreto_xyz_1234567890 [{batches[0].evidence_ids[0]}].",
        ),
        BatchAnalysisRecord(
            batch_index=2,
            evidence_ids=list(batches[1].evidence_ids),
            finding_ids=list(batches[1].finding_ids),
            raw_text="",
            ok=False,
            error="Resposta vazia",
        ),
    ]
    cons = consolidate_staged_analysis(task, batches, records)
    sec_specs = plan_report_sections(task, cons, ctx, "Assistente.", EXTENDED_PROMPT)
    sec_outputs = [
        (sec_specs[0], "## Resumo executivo\nResumo com ?api_key=SEGREDO_SENSIVEL [E1]."),
        (sec_specs[1], "## Principais descobertas\n[H][F1] Content-Security-Policy ausente — sem CSP [E1]."),
        (sec_specs[-1], "## Conclusão\n"),
    ]
    sec_records = [
        {"index": 1, "key": sec_specs[0].key, "label": sec_specs[0].label, "raw_text": sec_outputs[0][1], "finish_reason": "stop", "empty": False, "recovered": False, "recovery_failed": False},
        {"index": 2, "key": sec_specs[1].key, "label": sec_specs[1].label, "raw_text": sec_outputs[1][1], "finish_reason": "stop", "empty": False, "recovered": False, "recovery_failed": False},
        {"index": len(sec_specs), "key": sec_specs[-1].key, "label": sec_specs[-1].label, "raw_text": sec_outputs[2][1], "finish_reason": "length", "empty": False, "recovered": False, "recovery_failed": True},
    ]
    assembled = assemble_staged_report(task, sec_outputs, cons)
    rev = review_report(task, assembled)
    diag = build_staged_diagnostic(
        task, batches, records, cons, sec_records, assembled, rev.text, rev.text, section_specs=sec_specs, review=rev
    )
    stages = diag["stages"]
    assert len(stages["1_batches_raw"]) == len(batches)
    assert "secreto_xyz_1234567890" not in str(stages["1_batches_raw"])
    assert "SEGREDO_SENSIVEL" not in str(stages["3_sections_raw"])
    assert stages["2_consolidation"]["sent_evidence_ids"]
    assert diag["information_losses"]["evidence_sent_vs_processed"]
    assert diag["information_losses"]["truncated_sections"]
    assert diag["information_losses"]["empty_or_missing_sections"]


def test_assemble_staged_report_preserves_repeated_sub_bullets_across_findings():
    """Sub-bullets padrão iguais em F1, F2 e F3 não devem ser apagados pelo deduplicador global."""
    task = _make_25page_task()
    ctx = ContextManager(num_ctx=4096, output_reserve=512)
    batches = plan_evidence_batches(task, ctx, "Assistente.", EXTENDED_PROMPT)
    records = [parse_batch_analysis(task, b, f"- [H][F1] ok [{b.evidence_ids[0]}].") for b in batches]
    cons = consolidate_staged_analysis(task, batches, records)
    specs = plan_report_sections(task, cons, ctx, "Assistente.", EXTENDED_PROMPT)
    raw = "\n".join(
        [
            "## Principais descobertas",
            "[H][F1] Content-Security-Policy ausente — Sem CSP [E1]",
            "- **Limitações:** a coleta passiva não executa exploração ativa nem testes autenticados.",
            "- **Prioridade:** baixa.",
            "",
            "[H][F2] Sem proteção contra clickjacking — Sem XFO [E1]",
            "- **Limitações:** a coleta passiva não executa exploração ativa nem testes autenticados.",
            "- **Prioridade:** baixa.",
        ]
    )
    out = assemble_staged_report(task, [(specs[1], raw)], cons)
    assert out.count("a coleta passiva não executa exploração ativa nem testes autenticados.") == 2

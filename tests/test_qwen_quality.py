"""Regressão dos erros reais do Qwen 3.5 9B observados em investigação de segurança real (cenário-tipo passivo).

Cada teste reproduz um erro observado com fixtures locais (nenhum acesso à rede externa):
1. TLS 1.3 seguro classificado como [A];
2. item não verificado classificado como [C] (correto: [D]);
3. Next.js identificado no cabeçalho, mas matriz de cobertura com zero tecnologias;
4. recomendação NEXT_PUBLIC_CSP sem fundamento no dossiê;
5. relatório extenso sem referências [E#]/[F#];
6. follow-up genérico (checklist XSS/SQLi/Nmap) ignorando a investigação anterior;
7. vínculos [E#] incorretos em linhas de descoberta.

Fechamento de consistência (rodada 2):
8. cita [H][F2] para exposição do Next.js com F2 inexistente no Evidence Ledger;
9. verificador aponta F2 inexistente mas afirma que "todas as referências conferem";
10. CSP com prioridade baixa nas descobertas e alta nas recomendações, sem justificativa;
11. Cloudflare apresentado como prova de WAF robusto, sem evidência de regras habilitadas;
12. recomendação de CSP prescreve diretivas sem validar requisitos da aplicação;
13. cenário hipotético com lista excessiva de [E#] que só comprovam a ausência de CSP;
14. mitigação de XSS confundida com garantia absoluta de proteção.

Rodada 3 (preservação de conteúdo legítimo):
15. varredura final não pode alterar URLs, código cercado, código inline nem ids coincidentes legítimos;
16. verificador (check_coverage) trata id em URL/código como conteúdo, nunca como citação.
"""

import asyncio

from backend.chat.grounding import Grounding
from backend.research.evidence import Ledger
from backend.research.report import _strip_invalid_refs, check_coverage, followup_research_block, review_report, research_block
from backend.research.state import ResearchRegistry
from tests.test_deep_research import DEEP_MSG, Site, collect, last_assistant, make_env, public_dns, run  # noqa: F401


def _ledger(*findings, observations=(), limitations=()) -> Ledger:
    led = Ledger()
    ev = led.add_evidence("http", "https://a.example/", "HTTP 200 text/html · sem cabeçalhos de segurança")
    led.add_evidence("http", "https://a.example/login", "HTTP 200 text/html · server: nginx/1.14.0")
    for cat, title, detail in findings:
        led.add_finding(cat, title, detail, [ev.id], severity="baixa" if cat in "AH" else None,
                        proof="observação direta" if cat == "A" else None,
                        recommendation=f"Corrigir {title.split()[0].lower()}." if cat in "AH" else None)
    for obs in observations:
        led.observe(obs)
    for lim in limitations:
        led.limit(lim)
    return led


def _task(led: Ledger):
    task = ResearchRegistry().begin("c", "deep", {"goal": "investigação de segurança", "profile": "security", "targets": ["https://a.example"]})
    task.ledger = led
    return task


# ===================================================== erro 1: TLS seguro como [A]

def test_safe_tls_1_3_is_never_a_category_A_finding():
    led = _ledger(("H", "HSTS ausente", "Sem HSTS."),
                  observations=["TLS 1.3 negociado com certificado válido (observação técnica)."])
    task = _task(led)
    text = "[A] TLS 1.3 com negociação segura e certificado válido [E1]\n[H][F1] HSTS ausente — sem HSTS [E1]"
    rev = review_report(task, text)
    assert "[A]" not in rev.text  # a letra de vulnerabilidade confirmada sai: não há categoria A no registro
    assert "TLS 1.3 com negociação segura" in rev.text  # a observação verdadeira permanece, como texto neutro
    assert rev.tag_fixed == 1 and rev.hypotheses == []
    assert rev.text.endswith("[H][F1] HSTS ausente — sem HSTS [E1]")


def test_unanchored_category_A_claim_without_registry_A_goes_to_hypotheses():
    led = _ledger(("H", "HSTS ausente", "Sem HSTS."))
    task = _task(led)
    rev = review_report(task, "[A] Injeção de SQL confirmada no formulário de login [E1].")
    assert "[A]" not in rev.text and rev.hypotheses  # vulnerabilidade "confirmada" sem registro não fica no corpo


# ===================================================== erro 2: não verificado como [C]

def test_unverified_item_is_classified_D_from_the_registry():
    led = _ledger(("D", "Proteção CSRF dos formulários não verificada", "Não há campo tipo token visível; só um teste autorizado confirma."))
    task = _task(led)
    rev = review_report(task, "[C] Proteção CSRF dos formulários não verificada [E1]")
    assert "[D][F1] Proteção CSRF dos formulários não verificada [E1]" in rev.text
    assert "[C]" not in rev.text and rev.fixed == 1


def test_bare_C_tag_without_registry_match_is_never_presented_as_finding():
    led = _ledger(("H", "HSTS ausente", "Sem HSTS."))
    task = _task(led)
    rev = review_report(task, "[C] Configuração de rate limiting não verificada.")
    assert "[C]" not in rev.text  # tag avulsa sem descoberta correspondente sai; o texto permanece honesto
    assert "rate limiting" in rev.text and rev.tag_fixed == 1


# ===================================================== erro 3: Next.js x matriz de cobertura

def test_nextjs_in_header_enters_matrix_and_observations(db, settings, public_dns):
    site = Site(headers={"x-powered-by": "Next.js"})
    service, _prov, _p, _e = make_env(db, settings, site)
    events = run(collect(service, DEEP_MSG))
    msg = last_assistant(service, events)
    digest = msg.meta["research"]
    row = next(r for r in digest["view"]["coverage_matrix"] if r["area"] == "Bibliotecas e versões")
    assert "Next.js" in row["detail"]  # a matriz reflete a tecnologia identificada no cabeçalho
    assert any("Tecnologia identificada: Next.js" in o for o in digest["ledger"]["observations"])


def test_research_block_is_consistent_with_matrix_technologies(db, settings, public_dns):
    site = Site(headers={"x-powered-by": "Next.js"})
    service, _prov, _p, _e = make_env(db, settings, site)
    run(collect(service, DEEP_MSG))
    task = next(iter(service.research._tasks.values()))
    block = research_block(task, 9000)
    row = next(r for r in task.ledger.coverage["matrix"] if r["area"] == "Bibliotecas e versões")
    assert "identificadas:" in row["detail"] and "Next.js" in row["detail"]
    assert "Next.js" in block  # o dossiê enviado ao modelo contém a mesma tecnologia da matriz


# ===================================================== erro 4: recomendação NEXT_PUBLIC_CSP sem fundamento

def test_unfounded_mechanism_recommendation_is_replaced_by_the_registry_recommendation():
    led = Ledger()
    ev = led.add_evidence("http", "https://a.example/", "HTTP 200 text/html · sem cabeçalhos de segurança")
    led.add_finding("H", "Content-Security-Policy ausente", "Resposta HTML sem CSP.", [ev.id], severity="baixa",
                    recommendation="Definir uma CSP restritiva (sem 'unsafe-inline'/'unsafe-eval').")
    task = _task(led)
    text = "## Recomendações\n- Defina a variável NEXT_PUBLIC_CSP no arquivo .env para configurar a CSP."
    rev = review_report(task, text)
    assert "next_public_csp" not in rev.text.lower()  # mecanismo não evidenciado sai da recomendação
    assert rev.reco_fixed == 1
    assert "Definir uma CSP restritiva" in rev.text  # entra a recomendação fundamentada do registro


def test_grounded_recommendation_with_recommended_values_is_untouched():
    led = _ledger(("H", "HSTS ausente", "Sem HSTS."))
    task = _task(led)
    text = "## Recomendações\n- Enviar Strict-Transport-Security com max-age=31536000 e includeSubDomains em todas as respostas."
    rev = review_report(task, text)
    assert rev.text == text and rev.reco_fixed == 0 and rev.hypotheses == []


def test_unsupported_mechanism_stated_as_fact_leaves_the_body():
    led = _ledger(("H", "Content-Security-Policy ausente", "Resposta HTML sem CSP."))
    task = _task(led)
    rev = review_report(task, "O site aplica a política pela variável NEXT_PUBLIC_CSP do framework.")
    assert "next_public_csp" not in rev.text.lower() and rev.hypotheses


# ===================================================== erro 5: relatório sem referências [E#]/[F#]

def test_descriptive_bullet_without_anchor_is_relinked_to_the_registry():
    led = _ledger(("H", "HSTS ausente", "Sem HSTS."))
    task = _task(led)
    rev = review_report(task, "## Principais descobertas\n- **HSTS ausente** — observado na resposta HTML da página inicial.")
    assert "[H][F1] **HSTS ausente**" in rev.text and "[E1]" in rev.text and rev.anchored == 1


def test_prose_paragraph_is_not_forcibly_anchored():
    led = _ledger(("H", "HSTS ausente", "Sem HSTS."))
    task = _task(led)
    text = "A investigação observou a ausência do cabeçalho na resposta analisada."
    assert review_report(task, text).text == text


# ===================================================== erro 7: vínculos [E#] incorretos

def test_finding_line_citing_wrong_evidence_is_relinked_to_the_registry():
    led = _ledger(("H", "HSTS ausente", "Sem HSTS."))
    task = _task(led)
    rev = review_report(task, "[H][F1] HSTS ausente — sem HSTS na página inicial [E2]")
    assert rev.text.endswith("[E1]") and rev.evidence_linked == 1


# ===================================================== erro 6: follow-up genérico

def test_followup_question_receives_persisted_dossier_not_generic_checklist(db, settings, public_dns):
    site = Site()
    service, provider, _p, _e = make_env(db, settings, site)
    events = run(collect(service, DEEP_MSG))
    conv = next(e["conversation_id"] for e in events if "conversation_id" in e)
    calls_before = len(provider.calls)
    run(collect(service, "Que recomendações você dá para validar a segurança desse site?", conv, use_tools=False))
    system = provider.calls[calls_before][0]["content"]  # chamada do modelo na mensagem de acompanhamento
    assert "Investigação anterior" in system
    assert "[F1]" in system  # as descobertas reais da investigação fundamentam a resposta
    assert "checklist genérico" in system


def test_followup_block_is_built_only_from_persisted_registry():
    meta = {
        "mode": "deep", "goal": "Investigação de segurança do site X",
        "view": {"findings": [{"id": "F1", "category": "H", "title": "Content-Security-Policy ausente",
                               "detail": "Resposta HTML sem CSP.", "evidence": ["E1"], "recommendation": "Definir CSP restritiva."}]},
        "ledger": {"limitations": ["Auditoria passiva: sem testes ativos."]},
    }
    block = followup_research_block(meta, "Que recomendações você dá para validar a segurança desse site?")
    assert block and "[H][F1]" in block and "Correção: Definir CSP restritiva." in block
    assert "Limitações registradas" in block and "Auditoria passiva" in block
    assert followup_research_block({"mode": "deep"}) is None  # sem descobertas persistidas, não há bloco


async def _ground_block(service, conv_id, text):
    service.repo.add_message(conv_id, "user", text)  # como no fluxo real, a mensagem atual já está no histórico
    history = service.repo.get_messages(conv_id)
    async for item in service._ground_stream(text, history, resume=False, conversation_id=conv_id):
        if isinstance(item, Grounding):
            return item.context_block() or ""
    return ""


def test_topic_shift_after_research_does_not_inject_previous_dossier(db, settings, public_dns):
    site = Site()
    service, _prov, _p, _e = make_env(db, settings, site)
    events = run(collect(service, DEEP_MSG))
    conv = next(e["conversation_id"] for e in events if "conversation_id" in e)
    same_topic = run(_ground_block(service, conv, "Sobre esse site, que recomendações de validação você dá?"))
    other_topic = run(_ground_block(service, conv, "Você conhece o Jaspion?"))
    assert "Investigação anterior" in same_topic
    assert "Investigação anterior" not in other_topic  # assunto novo não herda o dossiê


# ===================================================== rodada 2: fechamento de consistência

def _csp_task() -> ResearchRegistry:  # noqa: ANN201
    """Registro típico do erro: CSP ausente (prioridade baixa), Cloudflare observado e Next.js identificado."""
    led = Ledger()
    ev = led.add_evidence("http", "https://a.example/", "HTTP 200 text/html · server: cloudflare · sem cabeçalhos de segurança")
    led.add_evidence("http", "https://a.example/login", "HTTP 200 text/html · sem cabeçalhos de segurança")
    led.add_finding("H", "Content-Security-Policy ausente", "Resposta HTML sem CSP.", [ev.id], severity="baixa",
                    recommendation="Definir uma CSP restritiva (sem 'unsafe-inline'/'unsafe-eval').")
    led.observe("Tecnologia identificada: Next.js (origem: cabeçalho x-powered-by).")
    return _task(led)


def test_inexistent_finding_reference_is_removed_and_never_reported_as_conferred():
    """Erro 8/9: cita [H][F2] para exposição do Next.js com F2 fora do registro — e o verificador não pode dizer que 'tudo confere'."""
    task = _csp_task()
    text = ("## Principais descobertas\n[H][F1] Content-Security-Policy ausente — sem CSP [E1]\n\n"
            "## Análise técnica\nA exposição do Next.js foi tratada como configuração fraca [H][F2], mas o registro só descreve a tecnologia.\n"
            "## Recomendações\n- Revisar a exposição do framework antes do lançamento [F2].\n"
            "## Conclusão\nFim da análise [F2] [E2].")
    rev = review_report(task, text)
    assert "F2" not in rev.text
    assert "exposição do Next.js" in rev.text and "Revisar a exposição do framework" in rev.text  # o texto fica; a referência inventada sai
    assert "[H][F1] Content-Security-Policy ausente" in rev.text and rev.invalid_refs == 3
    assert rev.notes and not any("todas as referências conferem" in n for n in rev.notes)
    assert any("referência(s) a id inexistente" in n for n in rev.notes)
    cov = check_coverage(task, rev.text)
    assert cov.issues == [] or not any("ids inexistentes" in i for i in cov.issues)
    # id inválido em bloco de descoberta inteiro (linha própria) continua removido com seus itens
    block = review_report(task, "[H][F2] Exposição do Next.js — tecnologia tratada como achado [E1]\n**O que significa**: texto inventado.")
    assert "F2" not in block.text and block.invented == 1


def test_validator_never_claims_all_references_confer_when_an_id_is_invalid():
    """Erro 9: a afirmação 'todas as referências conferem' só pode existir com o texto final sem nenhum id inválido."""
    task = _csp_task()
    cov = check_coverage(task, "ok [E1] e [F2] e [E9].")
    assert not cov.sufficient and any("ids inexistentes" in i for i in cov.issues)
    rev = review_report(task, "ok [E1] e [F2] e [E9].")
    assert "F2" not in rev.text and "E9" not in rev.text and rev.invalid_refs == 2
    assert not any("todas as referências conferem" in n for n in rev.notes)
    clean = review_report(task, "A falta de CSP é uma vulnerabilidade séria do servidor analisado [F1] [E1] [E2].")
    assert clean.reworded == 1 and "todas as referências conferem" in clean.notes[0]  # e a afirmação continua existindo quando é verdade


def test_recommendation_priority_without_anchor_follows_the_registry_or_is_justified():
    """Erro 10: CSP com prioridade baixa no dossiê e alta na recomendação, sem justificativa."""
    task = _csp_task()
    rev = review_report(task, "## Recomendações\n- **Prioridade alta:** Definir uma CSP restritiva em todas as respostas.")
    assert "Prioridade baixa:" in rev.text and rev.priority_fixed == 1
    justified = "## Recomendações\n- **Prioridade alta** para a CSP porque a aplicação executa scripts de terceiros."
    rev2 = review_report(task, justified)
    assert "Prioridade alta" in rev2.text and rev2.priority_fixed == 0  # diferença justificada no próprio texto permanece


def test_cloudflare_identified_never_proves_an_active_or_effective_waf():
    """Erro 11: Cloudflare apresentado como prova de WAF robusto, sem evidência de regras habilitadas."""
    task = _csp_task()
    bad = "## Análise técnica\nO site usa Cloudflare, um WAF robusto que protege o servidor de origem [E1]."
    rev = review_report(task, bad)
    assert rev.hypotheses and "robusto" not in rev.text and "protege" not in rev.text
    neutral = "O cabeçalho Server indica Cloudflare [E1]; nenhuma regra de WAF foi verificada."
    assert review_report(task, neutral).text == neutral  # tecnologia observada é descrita como observação
    reco = "## Recomendações\nAvaliar a adoção de um WAF conforme o risco do serviço."
    assert review_report(task, reco).text == reco  # recomendar avaliar um WAF continua permitido


def test_csp_recommendation_prescribing_directives_is_replaced_and_the_limitation_is_declared():
    """Erro 12: recomendação de CSP prescreve diretivas sem validar requisitos da aplicação."""
    task = _csp_task()
    text = "## Recomendações\n- Implementar CSP com script-src 'self'; object-src 'none'; frame-ancestors 'none' em todas as páginas."
    rev = review_report(task, text)
    assert "script-src" not in rev.text and "Definir uma CSP restritiva" in rev.text and rev.reco_fixed == 1
    assert any("Requisitos da aplicação para CSP" in x for x in task.ledger.limitations)  # a limitação fica declarada
    grounded = "## Recomendações\n- Definir uma CSP restritiva (sem 'unsafe-inline'/'unsafe-eval')."
    assert review_report(task, grounded).text == grounded  # recomendação fundamentada no registro não é tocada


def test_hypothetical_scenario_is_not_piled_with_every_evidence_reference():
    """Erro 13: cenário hipotético recebe lista excessiva de [E#] que só comprovam a ausência de CSP."""
    led = Ledger()
    evs = [led.add_evidence("http", f"https://a.example/p{i}", "HTTP 200 text/html · sem cabeçalhos de segurança") for i in range(6)]
    led.add_finding("H", "Content-Security-Policy ausente", "Respostas HTML sem CSP.", [e.id for e in evs], severity="baixa",
                    recommendation="Definir uma CSP restritiva (sem 'unsafe-inline'/'unsafe-eval').")
    task = _task(led)
    text = ("[H][F1] Content-Security-Policy ausente — sem CSP nas respostas [E1][E2][E3][E4][E5][E6]\n"
            "- **Impacto potencial:** um cenário hipotético de script injetado poderia ocorrer [E1][E2][E3][E4][E5][E6].")
    rev = review_report(task, text)
    assert "[E1][E2][E3]" in rev.text and "[E4]" not in rev.text and "[E6]" not in rev.text
    assert rev.evidence_linked == 1 and rev.evidence_trimmed == 3
    assert "cenário hipotético" in rev.text  # o conteúdo hipotético permanece, só as citações em excesso saem


def test_mitigation_of_xss_is_never_reported_as_an_absolute_guarantee():
    """Erro 14: mitigação de XSS confundida com garantia absoluta de proteção."""
    led = Ledger()
    ev = led.add_evidence("http", "https://a.example/", "HTTP 200 text/html · sem cabeçalhos de segurança")
    led.add_finding("H", "Content-Security-Policy ausente", "Resposta HTML sem CSP. Impacto potencial (NÃO verificado): menor defesa em profundidade contra XSS.", [ev.id], severity="baixa")
    task = _task(led)
    mitigates = "A CSP mitiga o risco de XSS em cenários de injeção [F1] [E1]."
    assert review_report(task, mitigates).text == mitigates  # mitigação correta permanece
    absolute = review_report(task, "A implementação da CSP garante proteção contra XSS [F1] [E1].")
    assert absolute.hypotheses and "garante" not in absolute.text  # garantia absoluta não tem suporte no dossiê
    negated = "Nenhum controle garante proteção absoluta contra XSS."
    assert review_report(task, negated).text == negated  # negar a garantia é honesto e permanece


# ===================================================== rodada 3: conteúdo legítimo preservado

def test_invalid_ref_sweep_preserves_urls_code_blocks_and_inline_code():
    """Erro 15: a varredura final não pode tratar id coincidente em URL/código como citação."""
    task = _csp_task()  # registro válido: E1, E2 e F1
    text = ("Cite [E1] e [F1] aqui, e também [E9] com [F2].\n"
            "```\nGET /E999 HTTP/1.1\nX-Debug: [F7]\nalerta E404 no log\n```\n"
            "Trecho `x = [E9]` reproduzido e rota https://a.example/E999 documentada [E2].\n"
            "Em prosa: conforme F3 e E7, nada foi concluído.")
    out, removed = _strip_invalid_refs(text, task.ledger)
    assert removed == 4  # só citações reais: [E9], [F2] e as menções soltas F3/E7
    assert "[E1] e [F1] aqui," in out and "[E2]." in out  # referências válidas permanecem
    assert "GET /E999 HTTP/1.1" in out and "X-Debug: [F7]" in out and "alerta E404 no log" in out  # bloco de código intacto
    assert "`x = [E9]`" in out and "https://a.example/E999" in out  # código inline e URL intactos
    assert "também com." in out and "conforme e, nada" in out  # buracos das citações removidas são recompostos
    assert "F3" not in out and "E7" not in out  # menções inválidas em prosa continuam saindo


def test_valid_references_and_technical_content_survive_byte_identical():
    """Erro 15: quando tudo é legítimo, a saída é idêntica byte a byte à entrada."""
    task = _csp_task()
    text = ("A CSP está ausente [H][F1] — observada na home [E1] e no login [E2].\n"
            "Exemplo `max-age=0` e rota https://a.example/login.\n"
            "```\nlog E999 / [F2]\n```")
    out, removed = _strip_invalid_refs(text, task.ledger)
    assert out == text and removed == 0  # [F2] dentro do bloco de código é conteúdo, não citação


def test_report_keeps_urls_and_code_while_still_flagging_invalid_citations():
    """Rodada 3 no pipeline completo: URL e código preservados; citação inválida em prosa sai e o verificador é coerente."""
    led = Ledger()
    ev = led.add_evidence("http", "https://a.example/", "HTTP 200 text/html · sem cabeçalhos de segurança")
    led.add_evidence("http", "https://a.example/E999", "HTTP 200 text/plain · log público de exemplo")
    led.add_finding("H", "Content-Security-Policy ausente", "Resposta HTML sem CSP.", [ev.id], severity="baixa",
                    recommendation="Definir uma CSP restritiva (sem 'unsafe-inline'/'unsafe-eval').")
    task = _task(led)
    text = ("## Análise técnica\n"
            "O log público em https://a.example/E999 respondeu HTTP 200 [E2].\n"
            "```\nX-Trace: [F7]\n```\n"
            "## Conclusão\nO cabeçalho está ausente conforme descrito em F2 e a evidência é [E1].")
    rev = review_report(task, text)
    assert "https://a.example/E999" in rev.text and "X-Trace: [F7]" in rev.text  # dados técnicos originais preservados
    assert "F2" not in rev.text and rev.invalid_refs == 1  # menção inválida em prosa é referência: sai
    assert rev.notes and not any("todas as referências conferem" in n for n in rev.notes)
    cov = check_coverage(task, rev.text)
    assert not any("ids inexistentes" in i for i in cov.issues)  # o verificador não contradiz a varredura


def test_check_coverage_ignores_ids_inside_urls_code_and_inline_code():
    """Erro 16: id em URL/código não é citação — o verificador não o aponta como inexistente."""
    task = _csp_task()
    text = "Log `status [E9]` reproduzido, rota https://a.example/E999 consultada [E1] e bloco:\n```\n[F7]\n```"
    cov = check_coverage(task, text)
    assert cov.sufficient and cov.issues == []
    bad = check_coverage(task, "ok [E1] e [F2] e [E9].")
    assert not bad.sufficient and any("ids inexistentes" in i for i in bad.issues)  # citação real continua detectada

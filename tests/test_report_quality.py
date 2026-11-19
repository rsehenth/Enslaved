"""Qualidade do relatório do Deep Research: truncamento, revisão de afirmações, evidências agrupadas, cobertura real e Markdown."""

import asyncio
import json

import httpx
import pytest

from backend.core.config import ResearchSettings
from backend.providers.ollama import OllamaProvider
from backend.research.evidence import Ledger
from backend.research.report import (
    appendix, check_coverage, clean_markdown, drop_own_sections, evidence_view, merge_continuation, review_report, trim_incomplete,
)
from backend.research.state import ResearchRegistry
from tests.test_deep_research import DEEP_MSG, SITE, Reporter, Site, collect, last_assistant, make_env, public_dns, run  # noqa: F401


def _task(*findings, targets=("https://a.example",), pages=0, requests=0):
    task = ResearchRegistry().begin("c", "deep", {"goal": "g", "profile": "security", "targets": list(targets)})
    ev = task.ledger.add_evidence("http", "https://a.example/", "HTTP 200 text/html · server: cloudflare; strict-transport-security: max-age=63072000")
    task.ledger.add_evidence("http", "https://a.example/login", "HTTP 200 text/html · sem cabeçalhos de segurança")
    for cat, title, detail in findings:
        task.ledger.add_finding(cat, title, detail, [ev.id], severity="baixa" if cat in "AH" else None, proof="obs direta" if cat == "A" else None)
    task.ledger.pages.update(f"https://a.example/p{i}" for i in range(pages))
    task.ledger.requests = requests
    return task


# ====================================================================== revisão de afirmações


def test_negative_statements_are_not_treated_as_claims_of_vulnerability():
    task = _task(("H", "HSTS ausente", "Sem HSTS."))
    text = ("Nenhuma vulnerabilidade foi comprovada nesta investigação passiva.\n"
            "Não foram encontradas vulnerabilidades confirmadas no alvo analisado [E1].\n"
            "Sem evidência de injeção de SQL ou XSS nas páginas inspecionadas [E1].")
    rev = review_report(task, text)
    assert rev.text == text and rev.hypotheses == [] and rev.notes == []
    # a checagem de cobertura também entende a negação (antes apontava "vulnerabilidade confirmada" sem categoria A)
    cov = check_coverage(task, "Nenhuma vulnerabilidade confirmada foi encontrada [E1].")
    assert cov.sufficient and not any("confirmada" in i for i in cov.issues)


def test_real_unsupported_hypothesis_leaves_the_body_and_is_listed_not_tagged_inline():
    task = _task(("H", "HSTS ausente", "Sem HSTS."))
    text = "## Conclusão\nO site está vulnerável a ataques de envenenamento de cache e execução de código remota [F1].\nA ausência de HSTS foi observada [E1]."
    rev = review_report(task, text)
    assert "envenenamento" not in rev.text and "hipótese do modelo" not in rev.text  # sem etiqueta frase a frase
    assert "A ausência de HSTS foi observada [E1]." in rev.text
    assert len(rev.hypotheses) == 1 and "envenenamento de cache" in rev.hypotheses[0]
    tail = appendix(task, rev.notes, rev.hypotheses)
    assert "## Limitações e hipóteses" in tail and "envenenamento de cache" in tail and "hipóteses não confirmadas" in tail


def test_scope_numbers_target_urls_and_recommended_values_are_not_inventions():
    task = _task(("H", "HSTS ausente", "Sem HSTS."), pages=9, requests=12)
    text = "\n".join([
        "## Resumo executivo",
        "Foram analisadas 9 páginas de https://a.example/ com 12 requisições, e a página https://a.example/login respondeu HTTP 200 [E2].",
        "O certificado do host a.example negociou TLS 1.3 conforme o registro de coleta.",  # TLS 1.3 é norma/protocolo
        "",
        "## Recomendações",
        "Enviar Strict-Transport-Security com max-age=31536000 e includeSubDomains em todas as respostas HTTPS.",  # valor recomendado
    ])
    rev = review_report(task, text)
    assert rev.text == text and rev.hypotheses == []
    invented = review_report(task, "O servidor também respondeu em https://invented-host.example/admin com 8443 conexões abertas [E1].")
    assert invented.text == "" and len(invented.hypotheses) == 1


def test_conditional_impact_is_kept_but_assertive_attack_claims_are_not():
    task = _task(("H", "Content-Security-Policy ausente", "Sem CSP. Impacto potencial (NÃO verificado): menor defesa em profundidade contra XSS."))
    ok = "A ausência de CSP poderia reduzir a defesa em profundidade contra XSS caso exista uma falha de injeção [F1]."
    assert review_report(task, ok).text == ok
    bad = review_report(task, "O site está vulnerável a injeção de SQL e CSRF por causa disso [F1].")
    assert bad.text == "" and bad.hypotheses


def test_explanatory_items_of_a_documented_finding_are_not_labelled_hypotheses():
    """Regressão do smoke com Qwen 9B: '**O que significa:** …scripts maliciosos injetados…' sob um achado H do dossiê foi tratado como hipótese."""
    task = _task(("H", "Cookie sem HttpOnly", "Cookie de sessão sem HttpOnly."))
    block = "[H][F1] Cookie sem HttpOnly — cookie de sessão sem o atributo [E1]\n*   **O que significa:** A falta de HttpOnly expõe o cookie a scripts maliciosos injetados na página."
    rev = review_report(task, block)
    assert rev.text == block and rev.hypotheses == []
    outside = review_report(task, "## Conclusão\nO site expõe o cookie a scripts maliciosos injetados por terceiros hoje.")  # fora do bloco de uma descoberta
    assert outside.hypotheses
    assert review_report(task, "[H][F1] Cookie sem HttpOnly — x [E1]\n**O que significa:** foi explorado por scripts maliciosos em produção.").hypotheses  # afirmar que aconteceu continua barrado


def test_vulnerability_wording_on_non_confirmed_finding_is_reworded_naturally():
    task = _task(("H", "HSTS ausente", "Sem HSTS."))
    rev = review_report(task, "A falta de HSTS é uma vulnerabilidade séria do servidor analisado [F1] [E1].")
    assert rev.text == "A falta de HSTS é uma fragilidade de configuração séria do servidor analisado [F1] [E1]." and rev.reworded == 1 and not rev.hypotheses
    confirmed = _task(("A", "Senha enviada sem HTTPS", "Formulário de senha com ação http."))
    assert review_report(confirmed, "A vulnerabilidade de senha sem HTTPS foi confirmada com prova direta [F1] [E1].").hypotheses == []


def test_wrong_category_letter_is_corrected_and_finding_blocks_inherit_the_finding():
    task = _task(("H", "CSP ausente", "Sem CSP."))
    text = "[A][F1] CSP ausente — sem CSP na resposta de https://a.example/ [E1]\n- **Significado:** isto é uma vulnerabilidade grave do site analisado."
    rev = review_report(task, text)
    assert rev.text.startswith("[H][F1] CSP ausente") and rev.fixed == 1
    assert "vulnerabilidade" not in rev.text and "fragilidade de configuração" in rev.text  # o bloco herda F1 (categoria H)


def test_finding_block_that_does_not_exist_in_the_registry_is_removed_with_its_items():
    task = _task(("H", "HSTS ausente", "Sem HSTS."))
    text = ("[H][F1] HSTS ausente — sem HSTS [E1]\n**Prioridade**: baixa.\n\n"
            "[D][F14] Duplicata inventada — copia do F1 [E1]\n**O que significa**: texto inventado sobre algo inexistente.\n**Prioridade**: baixa.\n\n## Conclusão\nFim do relatório analisado.")
    rev = review_report(task, text)
    assert "F14" not in rev.text and "inventado" not in rev.text and rev.invented == 1
    assert rev.text.startswith("[H][F1] HSTS ausente") and rev.text.endswith("## Conclusão\nFim do relatório analisado.") and "**Prioridade**: baixa." in rev.text
    assert "1 descoberta(s) inexistente(s)" in rev.notes[0]


def test_model_own_hypothesis_label_and_code_blocks_are_left_alone():
    task = _task(("H", "HSTS ausente", "Sem HSTS."))
    text = "hipótese: o proxy reverso pode estar na porta 8443 do servidor analisado.\n```\nSet-Cookie: x=1234567; max-age=99999\n```"
    assert review_report(task, text).text == text


def test_header_absence_is_classified_h_and_never_a_without_direct_proof(db, settings, public_dns):
    site = Site(pages={"/": "<html><head><title>X</title></head><body><p>oi</p></body></html>"})
    service, _p, _perm, _e = make_env(db, settings, site)
    msg = last_assistant(service, run(collect(service, DEEP_MSG)))
    cats = {f["title"]: f["category"] for f in msg.meta["research"]["view"]["findings"]}
    assert cats["Content-Security-Policy ausente"] == "H" and "A" not in cats.values()
    assert "Vulnerabilidades comprovadas (categoria A):** nenhuma" in msg.content


def test_confirmed_finding_a_is_reported_with_its_proof_in_view_and_text():
    task = _task(("A", "Protocolo TLS obsoleto aceito", "Negociado TLSv1."))
    assert "Vulnerabilidades comprovadas (categoria A):** 1." in appendix(task, [])
    view = evidence_view(task)
    fa = next(f for f in view["findings"] if f["category"] == "A")
    assert fa["proof"] == "obs direta" and fa["severity_label"] == "severidade" and fa["evidence"] == ["E1"]


# ====================================================================== evidências agrupadas


def test_equivalent_tls_observations_are_grouped_without_losing_any_reference():
    task = ResearchRegistry().begin("c", "deep", {"goal": "g", "profile": "security", "targets": ["https://a.example"]})
    led = task.ledger
    for i in range(9):
        ev = led.add_evidence("tls", f"https://a.example/p{i}", "TLS TLSv1.3 TLS_AES_256_GCM_SHA384 · verificado=True · emissor=CA Teste · expira=2031-01-01 · ok")
        led.pages.add(f"https://a.example/p{i}")
    http = led.add_evidence("http", "https://a.example/p0", "HTTP 200 text/html · content-security-policy: default-src 'self'; strict-transport-security: max-age=1", body=b"x")
    led.add_finding("H", "HSTS fraco", "max-age curto.", [http.id, "E3"], severity="baixa")
    view = evidence_view(task)
    tls = next(g for g in view["groups"] if g["key"] == "tls")
    assert tls["evidence_count"] == 9 and len(tls["items"]) == 1  # 9 observações equivalentes → 1 item
    assert tls["items"][0]["ids"] == [f"E{i}" for i in range(1, 10)] and len(tls["items"][0]["pages"]) == 9
    assert [e["id"] for e in view["ledger"]] == [f"E{i}" for i in range(1, 11)]  # o ledger técnico mantém TODAS as evidências
    assert "equivalentes entre as páginas" in next(s for s in view["summary"] if s["key"] == "tls")["value"]
    # navegação F# ⇄ E#
    f1 = view["findings"][0]
    assert f1["evidence"] == ["E10", "E3"] and f1["pages"] == ["https://a.example/p0", "https://a.example/p2"]
    by_id = {e["id"]: e for e in view["ledger"]}
    assert by_id["E10"]["findings"] == ["F1"] and by_id["E3"]["findings"] == ["F1"] and by_id["E1"]["findings"] == []
    http_group = next(g for g in view["groups"] if g["key"] == "http")
    assert http_group["items"][0]["supports"] == [{"id": "F1", "category": "H", "title": "HSTS fraco"}]
    matrix = view["header_matrix"]
    assert matrix["rows"][0]["present"]["csp"] is True and matrix["rows"][0]["present"]["hsts"] is True and matrix["rows"][0]["present"]["xfo"] is False
    assert json.dumps(view)  # serializável (vai em meta e no evento done)


def test_report_with_many_evidence_records_keeps_the_text_free_of_technical_ids():
    task = ResearchRegistry().begin("c", "extended", {"goal": "g", "profile": "security", "targets": ["https://a.example"]})
    for i in range(80):
        task.ledger.add_evidence("http", f"https://a.example/p{i}", f"HTTP 200 text/html · server: nginx/{i}", body=bytes([i]))
        task.ledger.pages.add(f"https://a.example/p{i}")
    tail = appendix(task, [])
    assert "E12" not in tail and "sha256" not in tail and "## Escopo e cobertura" in tail  # nada de despejo de E1…E80 no texto
    view = evidence_view(task)
    assert len(view["ledger"]) == 80 and sum(g["evidence_count"] for g in view["groups"]) == 80 and all(e["sha256"] for e in view["ledger"])


# ====================================================================== cobertura real da coleta


def test_coverage_text_states_the_real_stop_reason_when_no_more_links(db, settings, public_dns):
    service, _p, _perm, _e = make_env(db, settings, Site())
    msg = last_assistant(service, run(collect(service, DEEP_MSG)))
    cov = msg.meta["research"]["ledger"]["coverage"]
    assert cov["max_pages"] == 8 and cov["targets"][0]["stop"] == "sem_links"
    assert "páginas analisadas:** 5 de um limite de 8" in msg.content.replace("Páginas", "páginas")
    assert "não havia outros links relevantes dentro do escopo" in msg.content and "limite de 8 páginas do nível foi atingido" not in msg.content


def test_coverage_text_never_claims_a_stop_reason_the_runner_did_not_have(db, settings, public_dns):
    service, _p, _perm, _e = make_env(db, settings, Site(extra_links=15))
    msg = last_assistant(service, run(collect(service, DEEP_MSG)))
    stop = msg.meta["research"]["ledger"]["coverage"]["targets"][0]
    assert stop["stop"] == "limite_paginas" and stop["remaining"] > 0
    assert "limite de 8 páginas do nível foi atingido" in msg.content and "não havia outros links relevantes" not in msg.content


def test_unreachable_pages_and_tool_failures_appear_in_the_coverage(db, settings, public_dns):
    site = Site()

    def handler(request):
        if request.url.path == "/contato":
            raise httpx.ReadTimeout("lento")
        return site(request)

    service, _p, _perm, _e = make_env(db, settings, handler)
    msg = last_assistant(service, run(collect(service, DEEP_MSG)))
    assert "Inacessíveis ou com falha:** `/contato`" in msg.content
    assert any(i["ok"] is False and i["url"].endswith("/contato") for i in msg.meta["research"]["view"]["ledger"])


def test_budget_stop_is_reported_as_such():
    task = _task(pages=3, requests=30)
    task.ledger.coverage = {"max_pages": 8, "max_requests": 30, "targets": [{"host": "a.example", "stop": "requisições", "remaining": 4}]}
    tail = appendix(task, [])
    assert "limite de requisições do nível foi atingido" in tail and "(limite 30)" in tail and "não havia outros links" not in tail


# ====================================================================== truncamento e continuação


class Cutter(Reporter):
    """Provider que devolve respostas roteirizadas com o motivo de término de cada chamada (stop | length)."""

    def __init__(self, script):
        super().__init__("")
        self.script, self.seen = list(script), []

    async def stream_chat(self, messages, **options):
        text, reason = self.script.pop(0) if self.script else ("", "stop")
        self.calls.append([dict(m) for m in messages])
        self.seen.append(options)
        self.last_finish_reason = reason
        for i in range(0, len(text), 16):
            yield text[i : i + 16]


BODY = "## Resumo executivo\nA CSP está ausente [F1] conforme [E1].\n## Recomendações\n- Implementar uma CSP restritiva (Content-Security-P"


def _service_with(db, settings, script, **research):
    service, _prov, _p, _e = make_env(db, settings, Site(), research=ResearchSettings(**research) if research else None)
    service.provider = Cutter(script)
    return service


def test_output_cut_by_the_limit_is_continued_without_duplicating_text(db, settings, public_dns):
    service = _service_with(db, settings, [(BODY, "length"), ("(Content-Security-Policy) e revisar a política regularmente.", "stop")])
    events = run(collect(service, DEEP_MSG))
    msg = last_assistant(service, events)
    body = msg.content.split("## Escopo e cobertura")[0]
    assert "restritiva (Content-Security-Policy) e revisar a política regularmente." in body and "Content-Security-P(" not in body
    assert body.count("A CSP está ausente") == 1 and "Relatório parcial" not in msg.content
    done = next(e for e in events if e["type"] == "done")
    assert done["research"]["completion"] == {"status": "complete", "finish_reason": "stop", "continuations": 1}
    follow = service.provider.calls[1]
    assert follow[-2]["role"] == "assistant" and follow[-2]["content"].endswith("Implementar uma CSP restritiva") and "Continue o relatório" in follow[-1]["content"]
    assert any(e["type"] == "research_progress" and "Continuando o relatório" in e["message"] for e in events)
    assert "".join(e["content"] for e in events if e["type"] == "token") == msg.content  # exibido == salvo
    assert msg.meta["research"]["completion"]["status"] == "complete" and msg.meta["research"]["coverage"]["sufficient"] is True


def test_output_budget_is_proportional_to_the_level(db, settings, public_dns):
    service = _service_with(db, settings, [("Resumo. A CSP está ausente [F1] conforme [E1].", "stop")])
    run(collect(service, DEEP_MSG))
    assert service.provider.seen[0]["num_predict"] == 3072
    service = _service_with(db, settings, [("Resumo. A CSP está ausente [F1] conforme [E1].", "stop")])
    run(collect(service, "Faça uma investigação estendida da segurança do site https://loja.example"))
    assert service.provider.seen[0]["num_predict"] == 4608
    assert "palavras" in service.provider.calls[0][0]["content"]  # o texto é pedido para caber no orçamento


def test_report_still_cut_after_the_retries_is_marked_partial_and_never_masked(db, settings, public_dns):
    cut = "texto que continua sem terminar e que e cortado de novo pelo limit"
    service = _service_with(db, settings, [(BODY, "length"), (cut, "length"), (cut + " outra vez cortado ainda mais um pou", "length")])
    events = run(collect(service, DEEP_MSG))
    msg = last_assistant(service, events)
    head, notice, tail = msg.content.partition("> **Relatório parcial — geração interrompida.**")
    assert notice and "## Escopo e cobertura" in tail and "Content-Security-P" not in head
    assert head.rstrip().endswith((".", "!", "?", "]", ")")) and "restritiva (" not in head  # nada de apêndice colado sobre frase incompleta
    comp = msg.meta["research"]["completion"]
    assert comp["status"] == "partial" and comp["finish_reason"] == "length" and comp["continuations"] == 2
    assert msg.meta["research"]["coverage"]["sufficient"] is False and "Relatório parcial" in msg.meta["research"]["coverage"]["issues"][0]
    assert "".join(e["content"] for e in events if e["type"] == "token") == msg.content
    assert next(e for e in events if e["type"] == "done")["research"]["completion"]["status"] == "partial"


def test_continuation_can_be_disabled_and_a_restarted_text_is_discarded(db, settings, public_dns):
    service = _service_with(db, settings, [(BODY, "length")], report_continuations=0)
    msg = last_assistant(service, run(collect(service, DEEP_MSG)))
    assert len(service.provider.calls) == 1 and msg.meta["research"]["completion"]["status"] == "partial"
    restart = _service_with(db, settings, [(BODY, "length"), (BODY, "stop")])  # o modelo recomeçou em vez de continuar
    msg = last_assistant(restart, run(collect(restart, DEEP_MSG)))
    assert msg.meta["research"]["completion"]["status"] == "partial" and msg.content.count("A CSP está ausente") == 1


def test_trim_and_merge_helpers():
    assert trim_incomplete("## A\n\nFrase ok. Outra frase cortada ao mei") == ("## A\n\nFrase ok.", True)
    assert trim_incomplete("Completo.\n## Pendente") == ("Completo.", True)
    assert trim_incomplete("Tudo certo.") == ("Tudo certo.", False)
    assert trim_incomplete("```py\nx = 1")[0].endswith("```")  # cerca aberta é fechada
    assert merge_continuation("linha um.\nlinha dois e meio", "meio da frase final.") ==("linha um.\nlinha dois e meio da frase final.", True)
    assert merge_continuation("abc def ghij", "") == ("abc def", False)
    assert merge_continuation("a" * 50, "a" * 50 + " mais")[1] is False  # recomeço detectado


def test_ollama_provider_exposes_done_reason_from_the_stream():
    def handler(request):
        lines = [{"message": {"content": "olá"}, "done": False}, {"message": {"content": ""}, "done": True, "done_reason": "length", "eval_count": 7, "prompt_eval_count": 3}]
        return httpx.Response(200, text="\n".join(json.dumps(x) for x in lines))

    provider = OllamaProvider("m", client=httpx.AsyncClient(transport=httpx.MockTransport(handler), trust_env=False))

    async def go():
        return [c async for c in provider.stream_chat([{"role": "user", "content": "x"}])]

    assert asyncio.run(go()) == ["olá"] and provider.last_finish_reason == "length" and provider.last_usage["eval_count"] == 7


# ====================================================================== Markdown


def test_markdown_cleanup_fixes_common_defects_without_touching_content():
    messy = ("##Resumo executivo\n##Resumo executivo\nVeja **https://a.example/x** e também https://b.example/y** para detalhes.\n"
             "Content\\Security e a\\.b com `C:\\\\dir\\\\x` literal.\nTexto antes\n## Título\n\n\n\nFim\n```bash\ncurl -I https://a.example")
    out = clean_markdown(messy)
    assert out.count("## Resumo executivo") == 1 and "\n\n## Título\n" in out
    assert "**https://a.example/x**" in out and "https://b.example/y** " not in out and "https://b.example/y para" in out
    assert "ContentSecurity" in out and "a.b" in out and "`C:\\\\dir\\\\x`" in out  # barras inválidas saem; as de código ficam
    assert "\n\n\n" not in out and out.endswith("```") and out.count("```") == 2


def test_special_characters_survive_review_and_stay_plain_text():
    task = _task(("H", "HSTS ausente", "Sem HSTS."))
    text = "A resposta de https://a.example/ contém <script>alert(1)</script> e _x_ * y * `z` | pipe, e o alvo respondeu normalmente [E1]."
    rev = review_report(task, text)
    assert "<script>alert(1)</script>" in rev.text  # o backend não gera HTML; o renderizador seguro (DOMPurify) neutraliza no navegador


def test_own_sections_written_by_the_model_are_dropped_to_avoid_duplicate_titles():
    text = "## Resumo\nok.\n## Escopo e cobertura\nfalso.\n### sub\nmais falso.\n## Conclusão\nfim."
    assert drop_own_sections(text) == "## Resumo\nok.\n## Conclusão\nfim."


def test_client_disconnect_while_the_report_is_being_written_saves_no_partial_message(db, settings, public_dns):
    service, _p, _perm, _e = make_env(db, settings, Site())
    service.provider = Cutter([(BODY, "length")])

    async def go():
        gen = service.send_stream(None, DEEP_MSG)
        async for event in gen:
            if event["type"] == "research_progress" and event["phase"] == "reporting":
                break
        await gen.aclose()

    run(go())
    conv = next(iter(service.research._current))
    assert not [m for m in service.repo.get_messages(conv) if m.role == "assistant"]  # nada salvo como se estivesse completo
    assert service.research.mode(conv) == "quick"

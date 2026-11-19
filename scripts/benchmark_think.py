"""Benchmark REAL do Qwen 3.5 9B na etapa analítica do Deep Research: think=false vs think=true.

Monta um dossiê DETERMINÍSTICO (fixtures locais, sem rede) com o mesmo perfil do incidente observado
(site Next.js sem CSP, TLS 1.3 válido, formulário de login, robots.txt com caminhos sensíveis) e pede
N relatórios por modo com a MESMA entrada. Qualidade é medida pela própria revisão do Enslaved
(review_report + check_coverage): penalidades por afirmação sem suporte, categoria errada, descoberta
inventada, recomendação sem fundamento, falta de âncoras/vínculos, mais latência e tokens reais.

Uso: .venv\\Scripts\\python scripts\\benchmark_think.py [--samples 2] [--mode deep] [--output docs\\qwen-think-benchmark.json]
Regra de adoção: think=true só é adotado se reduzir a penalidade média em >= 15% sem mais que dobrar a latência.
"""

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

from backend.core.config import load_settings
from backend.providers.ollama import OllamaProvider
from backend.research import analyzers
from backend.research.evidence import ORDER
from backend.research.report import check_coverage, clean_markdown, drop_own_sections, report_rules, research_block, review_report
from backend.research.runner import Investigator
from backend.research.state import ResearchRegistry

ROOT = Path(__file__).resolve().parent.parent

_ADOPTION_GAIN = 0.15   # ganho mínimo de qualidade para adotar think=true
_ADOPTION_COST = 2.0    # latência máxima aceitável em relação ao think=false


def _page(url: str, body: str, headers: dict, *, forms=None, cookies=None) -> dict:
    return {
        "url": url, "final_url": url, "status": 200, "content_type": "text/html; charset=utf-8",
        "headers": {"content-type": "text/html; charset=utf-8", **headers},
        "signals": {"meta": {}, "scripts": [{"src": "/_next/static/chunks/main.js", "integrity": ""}],
                    "styles": [], "iframes": [], "forms": forms or [], "comments": [], "links": ["/login"],
                    "cookies": cookies or []},
        "cookies": cookies or [], "redirect_chain": [], "body": body, "bytes": len(body.encode()),
    }


def build_task(mode: str):
    """Dossiê determinístico do cenário-tipo (passivo, sem rede): cada fato vem de um 'net.inspect' sintético."""
    led_home = _page(
        "https://alvo.example/",
        '<html><head><title>Alvo</title></head><body><a href="/login">Entrar</a>'
        "<!-- TODO: mover o token de API do HTML antes do lanzamento --></body></html>",
        {"server": "cloudflare", "x-powered-by": "Next.js"},  # sem CSP, sem X-Frame-Options, sem HSTS
        cookies=[{"name": "sessionid", "secure": False, "httponly": True, "samesite": "Lax"}],
    )
    led_login = _page(
        "https://alvo.example/login",
        '<html><head><title>Entrar</title></head><body><form method="post" action="/login"><input type="password" name="p"></form></body></html>',
        {"server": "cloudflare", "x-powered-by": "Next.js", "x-content-type-options": "nosniff"},
        forms=[{"action": "/login", "method": "post", "password": True, "csrf_like": False}],
    )
    for page in (led_home, led_login):
        page["tls"] = {"verified": True, "protocol": "TLSv1.3", "cipher": "TLS_AES_256_GCM_SHA384",
                       "issuer": "CA Exemplo", "not_after": "2031-01-01", "days_left": 1500}

    registry = ResearchRegistry()
    task = registry.begin("benchmark", mode, {"goal": "Investigação de segurança do site https://alvo.example",
                                              "profile": "security", "targets": ["https://alvo.example"], "passive_only": True})
    led = task.ledger
    techs: list[dict] = []
    pages = []
    for data in (led_home, led_login):
        techs += analyzers.analyze_page(led, data, data["url"])
        pages.append(data)
    robots = "User-agent: *\nDisallow: /admin\nDisallow: /backup\n"
    analyzers.analyze_robots(led, "https://alvo.example/robots.txt", 200, robots, sha256=None)
    analyzers.reconcile_headers(led, pages)
    unique = list({(t["name"].lower(), t["version"]): t for t in techs}.values())
    for t in unique:
        led.observe(f"Tecnologia identificada: {t['name']}{' ' + t['version'] if t['version'] else ''} (origem: {t['source']}).")
    led.coverage.update(max_pages=8, max_requests=30, max_seconds=120.0, targets=[{"host": "alvo.example", "stop": "profundidade", "remaining": 1}])
    led.coverage["error_pages"] = []
    led.coverage["pages_200"] = len(pages)
    led.coverage["evidence_count"] = len(led.evidence)
    led.coverage["matrix"] = Investigator._security_matrix(led, pages, [], unique, [t for t in unique if t["version"]], 0)
    led.limit("Auditoria passiva: sem exploração, força bruta, fuzzing, autenticação ou alteração de dados.")
    return task


def _sections(text: str) -> int:
    wanted = ("resumo executivo", "principais descobertas", "recomendações", "conclusão")
    return sum(1 for s in wanted if s in text.lower())


def score(task, answer: str) -> dict:
    """Qualidade = penalidades da revisão/cobertura do próprio Enslaved (menor é melhor), mais referências e seções."""
    body = clean_markdown(drop_own_sections(answer))
    rev = review_report(task, body)
    cov = check_coverage(task, rev.text)
    refs = len(set(re.findall(r"[EF]\d+", rev.text)))
    secoes = _sections(rev.text)
    vazio = len(body.strip()) < 400 or secoes < 2  # o raciocínio (think=true) pode consumir o orçamento e deixar o relatório vazio/truncado
    penalty = (
        1.0 * len(rev.hypotheses) + 1.0 * rev.removed + 1.0 * rev.invented + 1.0 * len(cov.issues)
        + 0.5 * rev.fixed + 0.5 * rev.tag_fixed + 0.5 * rev.priority_fixed + 0.5 * rev.reworded + 0.5 * rev.reco_fixed
        + 0.25 * rev.anchored + 0.25 * rev.evidence_linked
        + (6.0 if vazio else 0.0)  # relatório vazio/truncado é falha grave da etapa analítica
    )
    counts = {c: len(task.ledger.by_category(c)) for c in ORDER}
    return {
        "penalidade": round(penalty, 2),
        "relatorio_vazio_ou_truncado": vazio,
        "hipoteses": len(rev.hypotheses), "removidas": rev.removed, "inventadas": rev.invented,
        "categorias_corrigidas": rev.fixed + rev.tag_fixed, "prioridades_corrigidas": rev.priority_fixed,
        "reformuladas": rev.reworded, "recomendacoes_corrigidas": rev.reco_fixed,
        "reancoragens": rev.anchored, "vinculos_realinhados": rev.evidence_linked,
        "problemas_cobertura": cov.issues, "referencias_ef": refs, "secoes": secoes,
        "descobertas_registro": counts, "caracteres": len(rev.text),
    }


async def run_sample(provider: OllamaProvider, system: str, user: str, think: bool, num_predict: int) -> dict:
    started = time.monotonic()
    answer = await provider.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 temperature=0.3, num_predict=num_predict, think=think, num_ctx=16384)
    usage = dict(provider.last_usage or {})
    return {"resposta": answer, "segundos": round(time.monotonic() - started, 1),
            "tokens_prompt": usage.get("prompt_eval_count"), "tokens_resposta": usage.get("eval_count")}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--mode", choices=("quick", "deep", "extended"), default="deep")
    parser.add_argument("--output", default=str(ROOT / "docs" / "qwen-think-benchmark.json"))
    args = parser.parse_args()

    settings = load_settings()
    provider = OllamaProvider(settings.provider.model, settings.ollama.base_url, settings.ollama.timeout_seconds,
                              default_options={"num_ctx": settings.ollama.num_ctx})
    health = await provider.health()
    assert health.get("available"), f"Ollama indisponível: {health}"

    task = build_task(args.mode)
    report_tokens = 3072 if args.mode != "extended" else 4608
    system = f"{report_rules(task, report_tokens)}\n\n{research_block(task, 9000)}"
    user = f"{task.research_scope['goal']} — redija o relatório da investigação."
    await run_sample(provider, system, user, think=False, num_predict=256)  # aquecimento: carrega o modelo na VRAM

    results = {"modelo": provider.model, "modo": args.mode, "amostras": args.samples, "descobertas_registro": {c: len(task.ledger.by_category(c)) for c in ORDER},
               "entradas_identicas": True, "modos": {}}
    for think in (False, True):
        rows = []
        for i in range(args.samples):
            row = await run_sample(provider, system, user, think=think, num_predict=report_tokens)
            row |= score(task, row.pop("resposta"))
            rows.append(row)
            print(f"[think={think}] amostra {i + 1}: penalidade={row['penalidade']} latência={row['segundos']}s "
                  f"tokens={row['tokens_resposta']} refs={row['referencias_ef']} hipóteses={row['hipoteses']}", flush=True)
        results["modos"][f"think={think}"] = {
            "amostras": rows,
            "penalidade_media": round(statistics.mean(r["penalidade"] for r in rows), 2),
            "latencia_media_s": round(statistics.mean(r["segundos"] for r in rows), 1),
            "tokens_resposta_media": round(statistics.mean(r["tokens_resposta"] or 0 for r in rows), 1),
            "referencias_media": round(statistics.mean(r["referencias_ef"] for r in rows), 1),
            "secoes_media": round(statistics.mean(r["secoes"] for r in rows), 1),
        }
    base, cand = results["modos"]["think=False"], results["modos"]["think=True"]
    gain = (base["penalidade_media"] - cand["penalidade_media"]) / max(base["penalidade_media"], 0.01)
    cost = cand["latencia_media_s"] / max(base["latencia_media_s"], 0.01)
    results["veredito"] = (
        f"think={'true' if gain >= _ADOPTION_GAIN and cost <= _ADOPTION_COST else 'false'} "
        f"(ganho de qualidade: {gain:+.0%}, custo de latência: {cost:.2f}x; adoção exige ≥{_ADOPTION_GAIN:+.0%} e ≤{_ADOPTION_COST}x)"
    )
    out = Path(args.output)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n", results["veredito"])
    print("evidência:", out)
    await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())

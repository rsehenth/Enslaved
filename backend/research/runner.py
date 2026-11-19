"""Investigação em etapas: planejar → coletar → analisar → verificar → (relatório pelo chat).

Dirigida por código (determinística), não pelo humor do modelo: cada etapa chama ferramentas reais, registra
evidências rastreáveis e respeita limites configuráveis (páginas, requisições, tempo, concorrência, bytes).
Nenhuma etapa é ativa/intrusiva: só leitura passiva. A profundidade muda o ESFORÇO, nunca as permissões.
"""

import asyncio
import re
import time
from typing import Any, AsyncIterator, Awaitable, Callable
from urllib.parse import urlsplit

from backend.core.config import ResearchSettings
from backend.research import analyzers
from backend.research.intent import canonicalize_url, url_dedup_key
from backend.research.state import ResearchRegistry, ResearchTask

_STATIC = re.compile(r"\.(?:png|jpe?g|gif|svg|webp|ico|css|js|mjs|map|woff2?|ttf|eot|pdf|zip|gz|mp[34]|avi|xml|json|txt)$", re.IGNORECASE)
_INTEREST = re.compile(r"login|signin|sign-in|entrar|admin|account|conta|api|docs?|developer|contact|contato|privacy|privacidade|terms|termos|cart|checkout|search|busca|upload|reset|register|signup|cadastr|status|security|seguranca", re.IGNORECASE)
_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>", re.IGNORECASE)

Inspect = Callable[[str], Awaitable[Any]]  # url -> ToolResult (net.inspect via ToolEngine)
Search = Callable[[str], Awaitable[tuple[str, list, str]]]  # query -> (estado, resultados, motivo)
Fetch = Callable[[str], Awaitable[str]]  # url -> texto (já extraído) ou ""


def normalize(url: str) -> str:
    return url_dedup_key(url, include_query=False) or canonicalize_url(url, keep_query=False) or url


def rank_links(links: list[str], seen: set[str]) -> list[str]:
    """Páginas candidatas do mesmo alvo: documentação/autenticação/contato primeiro; sem estáticos nem repetidas."""
    fresh: dict[str, str] = {}
    for u in links:
        if not _STATIC.search(urlsplit(u).path) and normalize(u) not in seen:
            fresh.setdefault(normalize(u), u)
    return [fresh[k] for k in sorted(fresh, key=lambda k: (not _INTEREST.search(k), k.count("/"), len(k), k))]


class Investigator:
    def __init__(self, registry: ResearchRegistry, settings: ResearchSettings, *, inspect: Inspect | None = None, search: Search | None = None, fetch: Fetch | None = None):
        self.registry, self.settings = registry, settings
        self.inspect, self.search, self.fetch = inspect, search, fetch

    # ------------------------------------------------------------------ orçamento
    def _exhausted(self, task: ResearchTask, started: float) -> str | None:
        lim, led = self.settings.limits(task.mode_used), task.ledger
        if task.cancel.is_set():
            return "cancelada"
        if time.monotonic() - started > lim.max_seconds:
            return "tempo"
        if led.requests >= lim.max_requests:
            return "requisições"
        if led.bytes_total >= lim.max_total_bytes:
            return "volume de dados"
        return None

    def _progress(self, task: ResearchTask, phase: str, message: str) -> dict:
        led = task.ledger
        self.registry.update(task, phase, pages=len(led.pages), requests=led.requests, evidence=len(led.evidence))
        return {"type": "research_progress", "task_id": task.research_task_id, "phase": phase, "message": message,
                "pages": len(led.pages), "requests": led.requests, "evidence": len(led.evidence)}

    # ------------------------------------------------------------------ entrada
    async def run(self, task: ResearchTask, query: str, entities: list[str], question: str) -> AsyncIterator[dict]:
        if task.research_scope["profile"] == "security":
            async for ev in self._security(task):
                yield ev
        else:
            async for ev in self._general(task, query, entities, question):
                yield ev
        if task.cancel.is_set():
            self.registry.finish(task, "cancelled")
            return

    # ------------------------------------------------------------------ perfil de segurança web
    async def _inspect_page(self, task: ResearchTask, url: str, sem: asyncio.Semaphore, started: float) -> dict | None:
        led = task.ledger
        async with sem:
            if self._exhausted(task, started):
                return None
            led.requests += 1
            try:
                res = await self.inspect(url)
            except Exception as exc:  # uma fonte com falha não derruba a investigação
                led.add_evidence("http", url, f"falha: {exc.__class__.__name__}", ok=False)
                led.limit(f"{url}: falha inesperada ({exc.__class__.__name__}).")
                return None
        if not res.ok:
            led.add_evidence("http", url, f"falha: {res.error}", ok=False)
            led.limit(f"{url}: {res.error}")
            return None
        data = res.data if isinstance(res.data, dict) else {}
        if not isinstance(data.get("final_url"), str) or not isinstance(data.get("status"), int):  # resultado ausente/inválido: não é evidência
            led.add_evidence("http", url, "resultado da ferramenta ausente ou inválido", ok=False)
            led.limit(f"{url}: resultado inválido da ferramenta net.inspect.")
            return None
        data.setdefault("headers", {})
        data.setdefault("signals", {})
        led.bytes_total += int(data.get("bytes") or 0)
        return data

    def _analyze(self, task: ResearchTask, data: dict, requested: str) -> list[dict]:
        try:
            return analyzers.analyze_page(task.ledger, data, requested)
        except Exception as exc:  # regra defeituosa ou dado inesperado não derruba a investigação
            task.ledger.limit(f"Análise de {data.get('final_url')} incompleta ({exc.__class__.__name__}).")
            return []

    async def _security(self, task: ResearchTask) -> AsyncIterator[dict]:
        led, lim, started = task.ledger, self.settings.limits(task.mode_used), time.monotonic()
        sem = asyncio.Semaphore(lim.concurrency)
        targets = task.research_scope["targets"]
        led.coverage.update(max_pages=lim.max_pages, max_requests=lim.max_requests, max_seconds=lim.max_seconds, targets=[])
        yield self._progress(task, "planning", f"Planejando {len(targets)} alvo(s), nível {task.mode_used}")
        techs: list[dict] = []
        pages: list[dict] = []
        seen: set[str] = set()
        for target in targets:
            root = urlsplit(target)
            origin = f"{root.scheme}://{root.netloc}"
            yield self._progress(task, "collecting", f"Consultando {root.hostname}")
            home = await self._inspect_page(task, target, sem, started)
            seen.add(normalize(target))
            if home is None:
                led.add_finding("D", f"Alvo {root.hostname} inacessível", "A página inicial não pôde ser inspecionada; nada foi concluído sobre este alvo.")
                led.coverage["targets"].append({"host": root.hostname, "stop": "inacessivel", "remaining": 0})
                continue
            seen.add(normalize(home["final_url"]))
            pages.append(home)
            techs += self._analyze(task, home, target)
            final_origin = "{0.scheme}://{0.netloc}".format(urlsplit(home["final_url"]))
            # arquivos públicos de descoberta (documentação anunciada pelo próprio site)
            extra: dict[str, dict | None] = {}
            for name in ("/robots.txt", "/.well-known/security.txt", "/sitemap.xml"):
                extra[name] = await self._inspect_page(task, final_origin + name, sem, started)
            for name, data in extra.items():
                if data is None:
                    continue
                text = data.get("body", "") if data.get("content_type", "").startswith("text/") or "xml" in data.get("content_type", "") else ""
                if name == "/robots.txt":
                    analyzers.analyze_robots(led, data["final_url"], data["status"], text, sha256=data.get("sha256"))
                elif name == "/.well-known/security.txt":
                    analyzers.analyze_security_txt(led, data["final_url"], data["status"], text, sha256=data.get("sha256"))
                else:
                    led.add_evidence("file", data["final_url"], f"sitemap.xml HTTP {data['status']}", body=text, sha256=data.get("sha256"))
            sitemap = extra.get("/sitemap.xml")
            discovered = list(home.get("signals", {}).get("links", []))
            if sitemap := extra.get("/sitemap.xml"):
                if sitemap["status"] == 200:
                    hosts = {root.hostname, urlsplit(home["final_url"]).hostname}
                    locs = [u for u in _LOC.findall(sitemap.get("body", "")) if urlsplit(u).hostname in hosts]
                    led.observe(f"sitemap.xml com {len(locs)} URL(s) do alvo.")
                    discovered += [urlsplit(u)._replace(query="", fragment="").geturl() for u in locs[:300]]
            for _round in range(2 if task.mode_used == "extended" else 1):
                batch = rank_links(discovered, seen)[: max(0, lim.max_pages - len(led.pages))]
                if not batch:
                    break
                seen.update(normalize(u) for u in batch)
                yield self._progress(task, "collecting", f"Consultando {len(batch)} página(s) relacionadas")
                for data in await asyncio.gather(*(self._inspect_page(task, u, sem, started) for u in batch)):
                    if data:
                        pages.append(data)
                        techs += self._analyze(task, data, data.get("url") or data["final_url"])
                        discovered += data.get("signals", {}).get("links", [])
                if self._exhausted(task, started):
                    break
            reason, remaining = self._exhausted(task, started), len(rank_links(discovered, seen))
            stop = reason or ("limite_paginas" if remaining and len(led.pages) >= lim.max_pages else "profundidade" if remaining else "sem_links")
            led.coverage["targets"].append({"host": root.hostname, "stop": stop, "remaining": remaining})
            if reason and reason != "cancelada":
                led.limit(f"Orçamento de {reason} esgotado antes de cobrir todas as páginas descobertas.")
            elif remaining:
                led.limit(f"Limite de {lim.max_pages} páginas atingido ou profundidade do nível esgotada: {remaining} página(s) descoberta(s) não inspecionada(s) (ajuste research.{task.mode_used}).")
        # ---------------------------------------------------------------- verificação cruzada
        yield self._progress(task, "analyzing", "Analisando evidências")
        ok_pages = [p for p in pages if 200 <= int(p.get("status") or 0) < 300]
        err_pages = [p for p in pages if not (200 <= int(p.get("status") or 0) < 300)]
        if pages and not ok_pages:
            first_err = err_pages[0]
            first_ev = next((e.id for e in led.evidence if e.kind == "http" and e.ok and e.url == first_err["final_url"]), None)
            led.add_finding(
                "D",
                "Nenhuma página HTTP 200 da aplicação analisada",
                f"Todas as respostas inspecionadas retornaram erro HTTP (ex.: HTTP {first_err.get('status')} em {first_err['final_url']}); configurações globais da aplicação publicada não puderam ser avaliadas.",
                [first_ev] if first_ev else [],
            )
        analyzers.reconcile_headers(led, ok_pages)
        unique = list({(t["name"].lower(), t["version"]): t for t in techs}.values())
        for t in unique:
            led.observe(f"Tecnologia identificada: {t['name']}{' ' + t['version'] if t['version'] else ''} (origem: {t['source']}).")
        yield self._progress(task, "verifying", "Verificando versões e fontes públicas")
        versioned = [t for t in unique if t["version"]]
        cve_searched = 0
        if versioned:
            if self.search is None:
                led.add_finding("D", "CVEs das versões identificadas não verificadas", "Pesquisa externa indisponível ou não autorizada; versões identificadas: " + ", ".join(f"{t['name']} {t['version']}" for t in versioned[:6]))
            else:
                found = 0
                for t in versioned[: lim.max_searches]:
                    if self._exhausted(task, started):
                        break
                    led.requests += 1
                    cve_searched += 1
                    state, results, _reason = await self.search(f"{t['name']} {t['version']} CVE vulnerabilidades")
                    if state == "ok":
                        found += analyzers.cves_from_results(led, t, results)
                if not found:
                    led.add_finding("D", "Nenhuma CVE localizada nas buscas", "Ausência de resultado não prova ausência de vulnerabilidade; consulte NVD/avisos do fornecedor.")
                if len(versioned) > lim.max_searches:
                    led.limit(f"CVEs consultadas para {lim.max_searches} de {len(versioned)} componentes versionados.")
        led.coverage["error_pages"] = [{"url": p["final_url"], "status": p.get("status")} for p in err_pages]
        led.coverage["pages_200"] = len(ok_pages)
        led.coverage["evidence_count"] = len(led.evidence)
        led.coverage["matrix"] = self._security_matrix(led, ok_pages, err_pages, unique, versioned, cve_searched)
        led.limit("Auditoria passiva: sem exploração, força bruta, fuzzing, autenticação ou alteração de dados; áreas logadas, lógica de negócio e subdomínios não avaliados.")
        if not led.evidence or not any(e.ok for e in led.evidence):
            led.limit("Nenhuma evidência coletada.")

    @staticmethod
    def _security_matrix(led, ok_pages: list[dict], err_pages: list[dict], unique: list[dict], versioned: list[dict], cve_searched: int) -> list[dict]:
        html_200 = [p for p in ok_pages if "html" in (p.get("content_type") or "").lower()]
        total_forms = sum(len(p.get("signals", {}).get("forms", [])) for p in html_200)
        tls_ok = any(e.kind == "tls" and e.ok for e in led.evidence)
        files_ok = sum(1 for e in led.evidence if e.kind == "file" and e.ok)
        stops = {t.get("stop") for t in led.coverage.get("targets") or []}
        all_links_done = bool(ok_pages) and stops <= {"sem_links"} and not err_pages
        return [
            {
                "area": "Transporte, redirecionamentos e TLS",
                "status": "VERIFICADO" if (ok_pages or err_pages) and tls_ok else ("PARCIALMENTE VERIFICADO" if (ok_pages or err_pages) else "NÃO VERIFICADO"),
                "detail": "Handshake TLS/certificado e cadeia de redirecionamentos do mesmo alvo." if tls_ok else "TLS não pôde ser verificado ou alvo HTTP.",
            },
            {
                "area": "Cabeçalhos de segurança e cookies",
                "status": "VERIFICADO" if all_links_done else ("PARCIALMENTE VERIFICADO" if ok_pages else "NÃO VERIFICADO"),
                "detail": f"{len(ok_pages)} página(s) HTTP 200 inspecionada(s)" + (f" ({len(err_pages)} resposta(s) 4xx/5xx separadas)." if err_pages else ".") if ok_pages else "Nenhuma página HTTP 200 disponível.",
            },
            {
                "area": "Formulários",
                "status": "PARCIALMENTE VERIFICADO" if html_200 else "NÃO VERIFICADO",
                "detail": f"{total_forms} formulário(s) HTML inspecionado(s) passivamente (action/method/password/csrf); submissão, validação e formulários via JS/login não testados." if html_200 else "Nenhuma página HTML HTTP 200 coletada.",
            },
            {
                "area": "Bibliotecas e versões",
                "status": "PARCIALMENTE VERIFICADO" if ok_pages else "NÃO VERIFICADO",
                "detail": (
                    f"{len(unique)} tecnologia(s) e {len(versioned)} versão(ões) identificadas via cabeçalhos, meta generator e nomes de script"
                    + (" — identificadas: " + ", ".join(sorted((f"{t['name']}{' ' + t['version'] if t['version'] else ''}" for t in unique), key=str.lower)[:8])[:180] if unique else "")
                    + "; bundles JS e dependências internas não inspecionados."
                ) if ok_pages else "Nenhuma página HTTP 200 coletada.",
            },
            {
                "area": "CVEs aplicáveis",
                "status": "PARCIALMENTE VERIFICADO" if (versioned and cve_searched > 0) else "NÃO VERIFICADO",
                "detail": (
                    f"Busca pública passiva por trechos para {cve_searched}/{len(versioned)} componente(s) versionado(s); sem consulta direta a NVD/OSV e sem teste de aplicabilidade."
                    if (versioned and cve_searched > 0) else
                    ("Versões identificadas, mas busca pública indisponível ou não autorizada." if versioned else "Nenhuma versão específica foi identificada passivamente; consulta de CVEs não executada.")
                ),
            },
            {
                "area": "Configurações e exposição de informações",
                "status": "PARCIALMENTE VERIFICADO" if (ok_pages or files_ok) else "NÃO VERIFICADO",
                "detail": "Inspeção passiva de cabeçalhos, robots.txt, security.txt, sitemap.xml, comentários HTML e padrões de erro/segredo; sem fuzzing de caminhos ou acesso autenticado." if (ok_pages or files_ok) else "Sem coleta efetiva.",
            },
            {
                "area": "Testes ativos (exploração, fuzzing, autenticação)",
                "status": "NÃO VERIFICADO",
                "detail": "Investigação estritamente passiva e somente leitura; exploração, força bruta, fuzzing, CSRF e áreas autenticadas não executados.",
            },
        ]

    # ------------------------------------------------------------------ perfil geral (pesquisa em fontes)
    async def _general(self, task: ResearchTask, query: str, entities: list[str], question: str) -> AsyncIterator[dict]:
        from backend.chat.grounding import extract_passages, is_official, relevant_results
        from backend.research.conflicts import detect_conflicts

        led, lim, started = task.ledger, self.settings.limits(task.mode_used), time.monotonic()
        led.coverage.update(max_pages=lim.max_pages, max_requests=lim.max_requests, max_seconds=lim.max_seconds, targets=[])
        yield self._progress(task, "planning", f"Planejando consultas, nível {task.mode_used}")
        suffixes = ["", "documentação oficial", "especificação técnica detalhes", "problemas conhecidos críticas análise"]
        queries = [f"{query} {s}".strip() for s in suffixes[: max(1, min(lim.max_searches, 2 if task.mode_used == "deep" else 4))]]
        seen: dict[str, Any] = {}
        for q in queries:
            if self._exhausted(task, started) or self.search is None:
                break
            yield self._progress(task, "collecting", "Consultando fontes")
            led.requests += 1
            state, results, reason = await self.search(q)
            if state != "ok":
                led.limit(f"Busca '{q[:60]}': {reason or 'indisponível'}.")
                continue
            for r in relevant_results(results, entities) if entities else results:
                seen.setdefault(normalize(r.url), r)
        if not seen:
            led.limit("Nenhuma fonte relevante foi retornada pelas buscas.")
        ranked = sorted(seen.values(), key=lambda r: (not is_official(r.url), r.url))
        opened: list[tuple[str, str]] = []
        sem = asyncio.Semaphore(lim.concurrency)

        async def one(hit):
            async with sem:
                if self._exhausted(task, started) or self.fetch is None:
                    return hit, ""
                led.requests += 1
                return hit, await self.fetch(hit.url)

        yield self._progress(task, "collecting", f"Abrindo até {lim.max_pages} página(s)")
        for hit, text in await asyncio.gather(*(one(h) for h in ranked[: lim.max_pages])):
            passages = extract_passages(text, entities, question, limit=900) if text else ""
            if passages:
                led.bytes_total += len(text)
                led.add_page(hit.url)
                ev = led.add_evidence("page", hit.url, passages, tool="net.fetch", body=text)
                opened.append((ev.id, passages))
            else:
                led.add_evidence("search", hit.url, f"{hit.title}: {hit.snippet}"[:400], ok=bool(hit.snippet), tool="web.search", body=hit.snippet or "")
                led.limit(f"Página não aberta/sem trecho relevante: {hit.url} (usado apenas o trecho da busca).")
        stop = self._exhausted(task, started) or ("limite_paginas" if len(ranked) > lim.max_pages else "sem_links")
        led.coverage["targets"].append({"host": "busca na web", "stop": stop, "remaining": max(0, len(ranked) - lim.max_pages)})
        led.coverage["evidence_count"] = len(led.evidence)
        led.coverage["matrix"] = [
            {"area": "Fontes públicas consultadas", "status": "VERIFICADO" if len(opened) >= 2 else ("PARCIALMENTE VERIFICADO" if opened or seen else "NÃO VERIFICADO"), "detail": f"{len(opened)} página(s) aberta(s) com trecho relevante."},
            {"area": "Inspeção técnica de segurança web", "status": "NÃO APLICÁVEL", "detail": "Perfil geral por busca em fontes públicas."},
        ]
        yield self._progress(task, "verifying", "Verificação cruzada entre fontes")
        for i, (a_id, a) in enumerate(opened):
            for b_id, b in opened[i + 1:]:
                if found := detect_conflicts(a, b):
                    led.add_finding("B", f"Fontes divergem ({a_id} × {b_id})", found[0], [a_id, b_id], recommendation="Priorize a fonte oficial/mais recente.")
        if len(opened) < 2:
            led.add_finding("D", "Verificação cruzada limitada", f"Apenas {len(opened)} página(s) aberta(s) com trecho relevante; conclusões dependem de poucas fontes.")

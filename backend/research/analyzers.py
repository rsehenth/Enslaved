"""Regras de análise passiva: transformam observações reais (net.inspect) em evidências e descobertas A/B/C/D.

Princípios: só afirmo (A) o que foi OBSERVADO diretamente; presença de tecnologia/formulário/login nunca é
vulnerabilidade; versão ou CVE só entram como indício (B) com fonte; o que não foi testado vira D.
"""

import re
from urllib.parse import urlsplit

from backend.core.redact import redact_secrets
from backend.research.evidence import Ledger
from backend.research.intent import url_dedup_key

_LIB = re.compile(r"(jquery(?:-ui|\.min)?|bootstrap|angular(?:js)?|react|vue|lodash|moment|underscore|backbone|ember|handlebars|swiper|prototype)[-.@/_]*v?(\d+\.\d+(?:\.\d+)?)", re.IGNORECASE)
_SERVER_VER = re.compile(r"([A-Za-z][\w.-]*)/(\d+(?:\.\d+)+)")
# Produtos anunciados em cabeçalhos SEM versão (ex.: "X-Powered-By: Next.js"): sem isto, a tecnologia é
# visível na evidência mas não entra na matriz de cobertura nem nas observações (inconsistência clássica).
_PRODUCT = re.compile(
    r"\b(next\.?js|nuxt\.?js|nest\.?js|node\.?js|express|fastify|koa|adonis|django|flask|fastapi|laravel|symfony|rails|sinatra|"
    r"asp\.?net|blazor|php|nginx(?:\s*plus)?|apache|iis|cloudflare|litespeed|caddy|tomcat|jetty|undertow|gunicorn|uvicorn|hypercorn|tornado|deno|bun|vite|webpack)\b",
    re.IGNORECASE,
)
_GEN_VER = re.compile(r"([A-Za-z][\w .-]*?)\s+v?(\d+(?:\.\d+)+)")
_ERRORS = re.compile(r"Traceback \(most recent call last\)|Fatal error:|Warning: \w+\(\)|SQLSTATE\[|ORA-\d{5}|at (?:java|org|com)\.[\w.]+\(|Stack trace:|Microsoft OLE DB|You have an error in your SQL syntax", re.IGNORECASE)
_SECRET = re.compile(r"AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|(?:api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{20,}['\"]", re.IGNORECASE)
_COMMENT_HINT = re.compile(r"password|senha|passwd|api[_-]?key|secret|token|todo|fixme|credencial|admin", re.IGNORECASE)
_SESSION_COOKIE = re.compile(r"sess|sid\b|auth|token|jwt|login", re.IGNORECASE)
_SENSITIVE_PATH = re.compile(r"/(?:admin|backup|private|internal|config|\.git|\.env|db|staging|phpmyadmin|dump)\b", re.IGNORECASE)
_WEAK_TLS = {"SSLv2", "SSLv3", "TLSv1", "TLSv1.1"}


def _page_evidence(ledger: Ledger, data: dict):
    h = data.get("headers", {})
    keys = ("server", "content-security-policy", "strict-transport-security", "x-frame-options", "x-content-type-options", "referrer-policy", "x-powered-by", "access-control-allow-origin")
    shown = "; ".join(f"{k}: {h[k][:80]}" for k in keys if k in h) or "sem cabeçalhos de segurança"
    return ledger.add_evidence(
        "http", data["final_url"], f"HTTP {data.get('status')} {data.get('content_type') or '?'} · {shown}",
        sha256=data.get("sha256"),
    )


def detect_tech(data: dict) -> list[dict]:
    """Tecnologias identificáveis e a ORIGEM de cada afirmação (cabeçalho, meta generator, nome de arquivo)."""
    found: dict[tuple, dict] = {}

    def add(name, version, source):
        found.setdefault((name.lower(), version), {"name": name, "version": version, "source": source})

    h, sig = data.get("headers", {}), data.get("signals", {})
    for header in ("server", "x-powered-by", "x-generator"):
        if header in h:
            versions = _SERVER_VER.findall(h[header])
            for name, ver in versions:
                add(name, ver, f"cabeçalho {header}")
            if not versions and (m := _PRODUCT.search(h[header])):
                add(m.group(0), None, f"cabeçalho {header}")
    gen = sig.get("meta", {}).get("generator")
    if gen:
        m = _GEN_VER.search(gen)
        add(m.group(1).strip() if m else gen[:40], m.group(2) if m else None, "meta generator")
    for script in sig.get("scripts", []):
        path = urlsplit(script["src"]).path
        if m := _LIB.search(path.rsplit("/", 1)[-1]) or _LIB.search(path):
            add(re.sub(r"\.min$", "", m.group(1).lower()), m.group(2), f"arquivo {script['src'][:120]}")
        if "/wp-content/" in path or "/wp-includes/" in path:
            add("WordPress", None, "caminhos wp-content/wp-includes")
    return list(found.values())


def analyze_page(ledger: Ledger, data: dict, requested_url: str) -> list[dict]:
    """Aplica as regras a uma resposta inspecionada. Retorna as tecnologias detectadas."""
    ev = _page_evidence(ledger, data)
    final, h, sig = data["final_url"], data.get("headers", {}), data.get("signals", {})
    status = int(data.get("status") or 0)
    is_success = 200 <= status < 300
    https = urlsplit(final).scheme == "https"
    html = "html" in (data.get("content_type") or "")
    if is_success:
        ledger.add_page(final)
    tls0 = data.get("tls") or {}
    ledger.observe(
        f"Observação técnica de {final}: HTTP {data.get('status')}, servidor '{h.get('server', 'não informado')[:40]}'"
        + (f", TLS {tls0.get('protocol')} (certificado {'verificado' if tls0.get('verified') else 'NÃO verificado'})" if tls0 else "")
        + f", {len(data.get('cookies', []))} cookie(s)."
    )
    A = lambda title, detail, sev, rec, proof, e=None: ledger.add_finding("A", title, detail, [(e or ev).id], severity=sev, recommendation=rec, proof=proof)  # noqa: E731
    H = lambda title, detail, sev, rec, e=None, risk="": ledger.add_finding(  # noqa: E731
        "H", title, detail + (f" Impacto potencial (NÃO verificado): {risk}" if risk else ""), [(e or ev).id], severity=sev, recommendation=rec)

    # --- transporte e redirecionamentos
    chain = data.get("redirect_chain", [])
    if is_success and urlsplit(requested_url).scheme == "http":
        if not https:
            H("HTTP sem redirecionamento para HTTPS", f"{final} respondeu por HTTP e não houve upgrade para HTTPS.", "média", "Redirecionar todo o tráfego HTTP para HTTPS (301/308) e ativar HSTS.", risk="o tráfego do site pode trafegar em texto claro se o usuário acessar por HTTP.")
        elif chain and chain[0].get("status") in (302, 303, 307):
            ledger.add_finding("C", "Upgrade HTTP→HTTPS com redirecionamento temporário", f"Primeiro salto HTTP {chain[0]['status']}.", [ev.id], recommendation="Usar 301/308 para o upgrade.")
    for hop in chain:
        if hop.get("followed") is False:
            ledger.limit(f"Redirecionamento de {hop['url']} → {hop['location']} não seguido ({hop.get('reason')}).")
    tls = data.get("tls")
    if urlsplit(requested_url).scheme == "https" or https:
        if not tls:
            ledger.add_finding("D", "TLS não inspecionado", "O certificado/protocolo não pôde ser coletado.", recommendation="Repetir a inspeção.")
        else:
            t_ev = ledger.add_evidence("tls", final, f"TLS {tls.get('protocol')} {tls.get('cipher')} · verificado={tls.get('verified')} · emissor={tls.get('issuer')} · expira={tls.get('not_after')} · {tls.get('error') or 'ok'}")
            if tls.get("verified") is False:
                H("Certificado TLS não verificável", str(tls.get("error")), "alta", "Instalar um certificado válido, com cadeia completa e nome do host correto.", t_ev, risk="clientes exibem alerta de segurança e podem ser induzidos a ignorá-lo.")
            if tls.get("protocol") in _WEAK_TLS:
                A("Protocolo TLS obsoleto aceito", f"Negociado {tls['protocol']}.", "média", "Desabilitar TLS 1.0/1.1; exigir TLS 1.2+.", f"handshake negociou {tls['protocol']}", t_ev)
            if isinstance(tls.get("days_left"), int):
                if tls["days_left"] < 0:
                    H("Certificado TLS expirado", f"Expirou em {tls['not_after']}.", "alta", "Renovar o certificado e automatizar a renovação.", t_ev, risk="clientes exibem alerta de segurança e podem ser induzidos a ignorá-lo.")
                elif tls["days_left"] < 15:
                    H("Certificado TLS perto de expirar", f"Expira em {tls['days_left']} dia(s) ({tls['not_after']}).", "baixa", "Renovar e monitorar a validade.", t_ev, risk="interrupção do serviço seguro se não for renovado.")

    body = data.get("body", "")
    if not is_success:
        ledger.limit(f"{final} respondeu HTTP {status} (resposta de erro HTTP; não foi considerada página publicada da aplicação e não sustenta ausência global de cabeçalhos).")
        if body:
            if m := _ERRORS.search(body):
                A("Mensagem de erro técnica exposta", f"Trecho observado: {redact_secrets(body[max(0, m.start() - 30): m.end() + 60])[:140]!r}", "média", "Desativar erros detalhados em produção.", f"texto de erro na resposta: {redact_secrets(body[max(0, m.start() - 30): m.end() + 60])[:100]!r}")
            if m := _SECRET.search(body):
                ledger.add_finding("B", "Possível credencial/chave no HTML", f"Padrão observado (valor redigido): {m.group(0)[:4]}… — validade e privilégios não verificados.", [ev.id], recommendation="Revogar/rotacionar se for real e removê-la do código público.")
        return []

    # --- cabeçalhos (ausência observada NESTA resposta; o impacto depende da aplicação)
    if html:
        meta_csp = sig.get("meta", {}).get("content-security-policy")
        csp = h.get("content-security-policy") or meta_csp
        if not csp:
            H("Content-Security-Policy ausente", f"Resposta HTML de {final} sem CSP (cabeçalho ou meta).", "baixa", "Definir uma CSP restritiva (sem 'unsafe-inline'/'unsafe-eval').", risk="menor defesa em profundidade contra XSS, caso exista uma falha de injeção.")
        elif re.search(r"'unsafe-(?:inline|eval)'|(?:script|default)-src[^;]*\s\*(?:\s|;|$)", csp):
            H("Content-Security-Policy permissiva", f"CSP observada: {csp[:160]}", "baixa", "Remover 'unsafe-inline'/'unsafe-eval' e curingas; usar nonces/hashes.", risk="menor defesa em profundidade contra XSS, caso exista uma falha de injeção.")
        if "x-frame-options" not in h and "frame-ancestors" not in (csp or ""):
            H("Sem proteção contra clickjacking", "Nem X-Frame-Options nem frame-ancestors na resposta HTML.", "baixa", "Enviar X-Frame-Options: DENY/SAMEORIGIN ou CSP frame-ancestors.", risk="a página poderia ser embutida em iframe de terceiros (clickjacking).")
        missing = [n for n in ("x-content-type-options", "referrer-policy", "permissions-policy") if n not in h]
        if missing:
            H("Cabeçalhos de hardening ausentes", "Ausentes na resposta: " + ", ".join(missing), "baixa", "Enviar X-Content-Type-Options: nosniff, Referrer-Policy e Permissions-Policy.", risk="menor defesa em profundidade do navegador (MIME sniffing, vazamento de Referer).")
    if https:
        hsts = h.get("strict-transport-security")
        if not hsts:
            H("HSTS ausente", "Resposta HTTPS sem Strict-Transport-Security.", "baixa", "Enviar Strict-Transport-Security com max-age ≥ 15552000.", risk="conexões iniciais por HTTP ficam sujeitas a downgrade em rede hostil.")
        elif (m := re.search(r"max-age=(\d+)", hsts)) and int(m.group(1)) < 15552000:
            H("HSTS com validade curta", f"Observado: {hsts[:100]}", "baixa", "Aumentar max-age (≥ 180 dias).", risk="proteção contra downgrade por pouco tempo.")
        elif "includesubdomains" not in hsts.lower():
            ledger.add_finding("C", "HSTS sem includeSubDomains", "Subdomínios não são cobertos pelo HSTS.", [ev.id])
    acao, acac = h.get("access-control-allow-origin"), h.get("access-control-allow-credentials", "").lower()
    if acao == "*" and acac == "true":
        ledger.add_finding("B", "CORS: '*' com credenciais", "ACAO='*' e ACAC=true na mesma resposta (combinação inválida/suspeita); validar com requisição com Origin.", [ev.id], recommendation="Usar lista fixa de origens; nunca refletir Origin com credenciais.")
    elif acao == "null":
        ledger.add_finding("B", "CORS aceita origem 'null'", "ACAO: null observado; validar o comportamento com credenciais.", [ev.id])
    elif acao == "*":
        ledger.add_finding("C", "CORS aberto ('*')", "Só relevante se houver dados autenticados/sensíveis neste recurso.", [ev.id])
    for header in ("server", "x-powered-by"):
        if header in h and _SERVER_VER.search(h[header]):
            H("Divulgação de versão em cabeçalho", f"{header}: {h[header][:80]}", "baixa", "Ocultar versões (ServerTokens/expose_php/X-Powered-By).", risk="facilita identificar versões com falhas conhecidas.")

    # --- cookies (valores nunca coletados)
    for c in data.get("cookies", []):
        if https and not c["secure"]:
            H(f"Cookie '{c['name']}' sem Secure", "Set-Cookie em resposta HTTPS sem o atributo Secure.", "baixa", "Adicionar Secure.", risk="o cookie pode trafegar sem proteção do transporte.")
        if c["samesite"] == "none" and not c["secure"]:
            H(f"Cookie '{c['name']}' SameSite=None sem Secure", "Navegadores modernos rejeitam; indica configuração incorreta.", "baixa", "SameSite=None exige Secure.", risk="o cookie pode trafegar sem proteção do transporte.")
        if _SESSION_COOKIE.search(c["name"]) and not c["httponly"]:
            ledger.add_finding("B", f"Cookie '{c['name']}' parece de sessão e sem HttpOnly", "Pode ser intencional (lido por JavaScript); validar a função do cookie.", [ev.id], recommendation="HttpOnly em cookies de sessão.")
        if not c["samesite"]:
            ledger.add_finding("C", f"Cookie '{c['name']}' sem SameSite", "O navegador aplica Lax por padrão; risco só em navegadores antigos.", [ev.id])

    # --- HTML: conteúdo misto, formulários, scripts de terceiros, exposição
    host = urlsplit(final).hostname or ""
    if https:
        mixed = [s["src"] for s in sig.get("scripts", []) if s["src"].startswith("http://")] + [s["href"] for s in sig.get("styles", []) if s["href"].startswith("http://")] + [i for i in sig.get("iframes", []) if i.startswith("http://")]
        if mixed:
            A("Conteúdo misto ativo", "Página HTTPS carrega por HTTP: " + ", ".join(mixed[:3]), "média", "Servir todos os recursos por HTTPS.", "recursos http:// referenciados no HTML: " + ", ".join(mixed[:3]))
    for form in sig.get("forms", []):
        action_url = form["action"] or final
        if form["password"]:
            ledger.observe(f"Formulário com campo de senha em {final} (ação: {redact_secrets(action_url)[:100]}) — superfície de autenticação; não é vulnerabilidade por si só.")
            if not https or action_url.startswith("http://"):
                A("Senha enviada sem HTTPS", f"Formulário de senha em {final} com ação {action_url[:100]}.", "alta", "Servir a página e a ação do formulário por HTTPS.", f"campo type=password com ação {action_url[:100]} na página {final}")
            if form["method"] == "post" and not form["csrf_like"]:
                ledger.add_finding("D", "Proteção CSRF dos formulários não verificada", "Não há campo tipo token visível; a proteção pode estar em cabeçalho/cookie. Só um teste autorizado confirma.", [ev.id])
        if urlsplit(action_url).hostname and urlsplit(action_url).hostname != host and form["method"] == "post":
            ledger.add_finding("B", "Formulário POST para outro host", f"Ação: {action_url[:100]}", [ev.id], recommendation="Confirmar que o destino é um provedor legítimo.")
    third = [s for s in sig.get("scripts", []) if (urlsplit(s["src"]).hostname or host) != host and not s["integrity"]]
    if third:
        ledger.add_finding("C", "Scripts de terceiros sem SRI", f"{len(third)} script(s) externo(s) sem integrity, ex.: {third[0]['src'][:80]}", [ev.id], recommendation="Usar Subresource Integrity ou hospedar localmente.")
    if body:
        if m := _ERRORS.search(body):
            A("Mensagem de erro técnica exposta", f"Trecho observado: {redact_secrets(body[max(0, m.start() - 30): m.end() + 60])[:140]!r}", "média", "Desativar erros detalhados em produção.", f"texto de erro na resposta: {redact_secrets(body[max(0, m.start() - 30): m.end() + 60])[:100]!r}")
        if m := _SECRET.search(body):
            ledger.add_finding("B", "Possível credencial/chave no HTML", f"Padrão observado (valor redigido): {m.group(0)[:4]}… — validade e privilégios não verificados.", [ev.id], recommendation="Revogar/rotacionar se for real e removê-la do código público.")
        if "sourceMappingURL" in body:
            ledger.add_finding("C", "Referência a source map", "Pode expor o código-fonte original se o .map for público; acesso não testado.", [ev.id])
    hints = [c for c in sig.get("comments", []) if _COMMENT_HINT.search(c)]
    if hints:
        ledger.add_finding("B", "Comentários HTML com termos sensíveis", f"Ex.: {redact_secrets(hints[0])[:100]!r}", [ev.id])
    if not sig.get("forms") and html and re.search(r"login|entrar|sign.?in", data.get("title") or "", re.IGNORECASE):
        ledger.observe(f"Página com aparência de login em {final} (informativo).")
    return detect_tech(data)


def analyze_robots(ledger: Ledger, url: str, status: int | None, text: str, *, sha256: str | None = None) -> None:
    ev = ledger.add_evidence("file", url, f"robots.txt HTTP {status}", body=text, sha256=sha256)
    if status != 200:
        return
    paths = [p.strip() for p in re.findall(r"(?im)^\s*disallow:\s*(\S+)", text)]
    sensitive = [p for p in paths if _SENSITIVE_PATH.search(p)]
    ledger.observe(f"robots.txt com {len(paths)} regra(s) Disallow.")
    if sensitive:
        ledger.add_finding("B", "robots.txt anuncia caminhos sensíveis", "Caminhos listados (existência/acesso NÃO verificados): " + ", ".join(sensitive[:6]), [ev.id], recommendation="Não depender de robots.txt para proteger áreas; exigir autenticação.")


def analyze_security_txt(ledger: Ledger, url: str, status: int | None, text: str, *, sha256: str | None = None) -> None:
    ledger.add_evidence("file", url, f"security.txt HTTP {status}", body=text, sha256=sha256)
    ledger.observe("security.txt presente (canal de divulgação de vulnerabilidades)." if status == 200 else "security.txt não encontrado em /.well-known/ (informativo).")


def cves_from_results(ledger: Ledger, tech: dict, results: list) -> int:
    """CVEs LITERALMENTE citadas em resultados de busca que também citam o produto e a versão. Nunca inventa."""
    name, version = tech["name"].lower(), tech["version"]
    hits: dict[str, object] = {}
    for r in results:
        blob = f"{r.title} {r.snippet}"
        low = blob.lower()
        if name.split()[0] in low and version in blob:
            for cve in re.findall(r"CVE-\d{4}-\d{4,7}", blob, re.IGNORECASE):
                hits.setdefault(cve.upper(), r)
    for cve, r in list(hits.items())[:6]:
        e = ledger.add_evidence("search", r.url, f"Resultado de busca cita {cve} junto com {tech['name']} {version}", tool="web.search", body=f"{r.title}|{r.snippet}")
        ledger.add_finding("B", f"{cve} citada para {tech['name']} {version}", f"Fonte pública cita a CVE para a versão identificada ({tech['source']}); aplicabilidade, configuração e explorabilidade NÃO foram testadas.", [e.id], recommendation="Conferir o aviso oficial do fornecedor e atualizar a versão.")
    return len(hits)


def _short_ep(url: str) -> str:
    p = urlsplit(url)
    return (p.path or "/") + (f"?{p.query[:30]}" if p.query else "")


def reconcile_headers(ledger: Ledger, pages: list[dict]) -> None:
    """Reconcilia achados de cabeçalhos entre páginas HTTP 200: associa apenas às páginas afetadas e explicita presença parcial."""
    ok_pages = [p for p in pages if 200 <= int(p.get("status") or 0) < 300]
    if not ok_pages:
        return
    ev_by_key: dict[str, str] = {}
    for e in ledger.evidence:
        if e.kind == "http" and e.ok and re.match(r"^HTTP 2\d\d\b", e.observation):
            ev_by_key.setdefault(url_dedup_key(e.url) or e.url, e.id)

    def ev_ids(items: list[dict]) -> list[str]:
        out: list[str] = []
        for p in items:
            eid = ev_by_key.get(url_dedup_key(p["final_url"]) or p["final_url"])
            if eid and eid not in out:
                out.append(eid)
        return out

    html = [p for p in ok_pages if "html" in (p.get("content_type") or "").lower()]
    https = [p for p in ok_pages if urlsplit(p["final_url"]).scheme == "https"]

    def sync_single(title: str, inc_title: str, total_pages: list[dict], missing: list[dict], present: list[dict], what: str, risk: str) -> None:
        if not total_pages:
            return
        f = next((x for x in ledger.findings if x.title == title and x.category == "H"), None)
        if f is not None:
            m_ids = ev_ids(missing)
            if m_ids:
                f.evidence = m_ids
            if present and missing:
                eps = ", ".join(_short_ep(p["final_url"]) for p in missing[:6])
                f.detail = (
                    f"{what} ausente em {len(missing)}/{len(total_pages)} página(s) HTTP 200 ({eps}); "
                    f"presente em {len(present)}/{len(total_pages)}. Impacto potencial (NÃO verificado): {risk}"
                )
            elif missing and len(total_pages) > 1:
                f.detail = (
                    f"{what} ausente em todas as {len(missing)} página(s) HTTP 200 inspecionadas. "
                    f"Impacto potencial (NÃO verificado): {risk}"
                )
        if present and missing:
            m_eps = ", ".join(_short_ep(p["final_url"]) for p in missing[:5])
            p_eps = ", ".join(_short_ep(p["final_url"]) for p in present[:5])
            ledger.add_finding(
                "B",
                inc_title,
                f"Presente em {len(present)}/{len(total_pages)} página(s) HTTP 200 ({p_eps}) e ausente em {len(missing)}/{len(total_pages)} ({m_eps}); validar a origem do cabeçalho (proxy/aplicação).",
                ev_ids(missing[:4] + present[:4]),
                recommendation=f"Padronizar {what} em todas as respostas aplicáveis.",
            )

    if html:
        csp_missing = [p for p in html if not (p.get("headers", {}).get("content-security-policy") or p.get("signals", {}).get("meta", {}).get("content-security-policy"))]
        sync_single(
            "Content-Security-Policy ausente",
            "CSP aplicado de forma inconsistente",
            html, csp_missing, [p for p in html if p not in csp_missing],
            "CSP", "menor defesa em profundidade contra XSS, caso exista uma falha de injeção.",
        )
        xfo_missing = [
            p for p in html
            if "x-frame-options" not in p.get("headers", {})
            and "frame-ancestors" not in ((p.get("headers", {}).get("content-security-policy") or p.get("signals", {}).get("meta", {}).get("content-security-policy") or "").lower())
        ]
        sync_single(
            "Sem proteção contra clickjacking",
            "X-Frame-Options aplicado de forma inconsistente",
            html, xfo_missing, [p for p in html if p not in xfo_missing],
            "X-Frame-Options/frame-ancestors", "a página poderia ser embutida em iframe de terceiros (clickjacking).",
        )
        hard_headers = (
            ("x-content-type-options", "X-Content-Type-Options"),
            ("referrer-policy", "Referrer-Policy"),
            ("permissions-policy", "Permissions-Policy"),
        )
        hard_finding = next((x for x in ledger.findings if x.title == "Cabeçalhos de hardening ausentes" and x.category == "H"), None)
        if hard_finding is not None:
            parts: list[str] = []
            affected: list[dict] = []
            for key, label in hard_headers:
                miss = [p for p in html if key not in p.get("headers", {})]
                pres = [p for p in html if key in p.get("headers", {})]
                for p in miss:
                    if p not in affected:
                        affected.append(p)
                if len(miss) == len(html):
                    parts.append(f"{label} ausente em {len(html)}/{len(html)} página(s) HTML")
                elif miss:
                    eps = ", ".join(_short_ep(p["final_url"]) for p in miss[:5])
                    parts.append(f"{label} presente em {len(pres)}/{len(html)} e ausente em {len(miss)}/{len(html)} ({eps})")
                    if key in ("x-content-type-options", "referrer-policy"):
                        ledger.add_finding(
                            "B",
                            f"{label} aplicado de forma inconsistente",
                            f"Presente em {len(pres)}/{len(html)} página(s) HTML HTTP 200 e ausente em {len(miss)}/{len(html)} ({eps}); validar configuração por rota/proxy.",
                            ev_ids(miss[:4] + pres[:4]),
                            recommendation=f"Enviar {label} de forma consistente em todas as respostas HTML.",
                        )
            a_ids = ev_ids(affected)
            if a_ids:
                hard_finding.evidence = a_ids
            if parts:
                hard_finding.detail = (
                    "Ausências observadas nas respostas HTML HTTP 200: " + "; ".join(parts)
                    + ". Impacto potencial (NÃO verificado): menor defesa em profundidade do navegador (MIME sniffing, vazamento de Referer)."
                )
    if https:
        hsts_missing = [p for p in https if "strict-transport-security" not in p.get("headers", {})]
        sync_single(
            "HSTS ausente",
            "HSTS aplicado de forma inconsistente",
            https, hsts_missing, [p for p in https if p not in hsts_missing],
            "HSTS", "conexões iniciais por HTTP ficam sujeitas a downgrade em rede hostil.",
        )

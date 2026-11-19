"""Inspeção passiva de uma URL pública (net.inspect): status, cabeçalhos, redirecionamentos, cookies, TLS e sinais do HTML.

Somente leitura e não intrusiva: um GET por salto de redirecionamento, sem enviar dados, sem sondar caminhos.
Preserva as proteções do net.fetch (SSRF, IP validado e fixado contra DNS rebinding, portas, tamanho, timeout) e
usa a MESMA capacidade/escopo (net.fetch, destino = host): uma concessão existente cobre as duas ferramentas.
Só segue redirecionamentos dentro do mesmo alvo (mesmo host; apex <-> www): outro host é registrado, nunca visitado.
"""

import asyncio
import hashlib
import re
import socket
import ssl
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import BaseModel, Field, field_validator

from backend.core.config import NetToolsSettings
from backend.core.redact import redact_secrets
from backend.tools.base import Tool, ToolResult, ToolSpec
from backend.tools.builtin.web_fetch import _PinnedNetworkBackend, html_to_text
from backend.tools.net_guard import UrlBlockedError, normalize_candidate_url, resolve_public_url

_REDIRECTS = (301, 302, 303, 307, 308)
_TEXT_TYPES = ("text/", "application/json", "application/xml", "application/xhtml", "application/javascript")
_CSRF_NAME = re.compile(r"csrf|xsrf|authenticity|_token|nonce", re.IGNORECASE)
_MAX_HEADERS, _MAX_ITEMS = 60, 60


def same_target(a: str, b: str) -> bool:
    """Mesmo alvo para fins de autorização: host igual ou apenas com/sem 'www.'."""
    strip = lambda host: (host or "").lower().rstrip(".").removeprefix("www.")  # noqa: E731
    return strip(urlsplit(a).hostname) == strip(urlsplit(b).hostname)


class _Signals(HTMLParser):
    """Sinais públicos do HTML (formulários, scripts, links do mesmo alvo, meta, comentários)."""

    def __init__(self, base_url: str):
        super().__init__(convert_charrefs=True)
        self.base = base_url
        self.forms: list[dict] = []
        self.scripts: list[dict] = []
        self.styles: list[dict] = []
        self.iframes: list[str] = []
        self.links: list[str] = []
        self.meta: dict[str, str] = {}
        self.comments: list[str] = []
        self.inline_scripts = 0
        self._form: dict | None = None

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag == "form":
            self._form = {
                "action": urljoin(self.base, a.get("action", "")),
                "method": (a.get("method") or "get").lower(),
                "password": False,
                "file": False,
                "csrf_like": False,
                "inputs": 0,
            }
            if len(self.forms) < _MAX_ITEMS:
                self.forms.append(self._form)
        elif tag == "input" and self._form is not None:
            kind = (a.get("type") or "text").lower()
            self._form["inputs"] += 1
            self._form["password"] |= kind == "password"
            self._form["file"] |= kind == "file"
            self._form["csrf_like"] |= bool(_CSRF_NAME.search(a.get("name", "")))
        elif tag == "script":
            if a.get("src"):
                if len(self.scripts) < _MAX_ITEMS:
                    self.scripts.append(
                        {"src": urljoin(self.base, a["src"]), "integrity": bool(a.get("integrity")), "crossorigin": a.get("crossorigin", "")}
                    )
            else:
                self.inline_scripts += 1
        elif tag == "link" and "stylesheet" in a.get("rel", "").lower() and a.get("href"):
            if len(self.styles) < _MAX_ITEMS:
                self.styles.append({"href": urljoin(self.base, a["href"]), "integrity": bool(a.get("integrity"))})
        elif tag == "iframe" and a.get("src") and len(self.iframes) < _MAX_ITEMS:
            self.iframes.append(urljoin(self.base, a["src"]))
        elif tag == "a" and a.get("href") and len(self.links) < 400:
            self.links.append(a["href"])
        elif tag == "meta":
            key = (a.get("name") or a.get("http-equiv") or "").lower()
            if key in ("generator", "referrer", "content-security-policy", "robots"):
                self.meta[key] = a.get("content", "")[:300]

    def handle_endtag(self, tag):
        if tag == "form":
            self._form = None

    def handle_comment(self, data):
        if len(self.comments) < 30:
            self.comments.append(data.strip()[:200])


def _parse_cookie(raw: str) -> dict:
    parts = [p.strip() for p in raw.split(";")]
    name = parts[0].split("=", 1)[0].strip()
    attrs = {p.split("=", 1)[0].strip().lower(): (p.split("=", 1)[1].strip() if "=" in p else "") for p in parts[1:]}
    return {
        "name": name[:80],  # o VALOR nunca é guardado
        "secure": "secure" in attrs,
        "httponly": "httponly" in attrs,
        "samesite": attrs.get("samesite", "").lower() or None,
        "persistent": "max-age" in attrs or "expires" in attrs,
        "domain": attrs.get("domain") or None,
        "prefixed": name.startswith(("__Host-", "__Secure-")),
    }


def tls_probe(host: str, port: int, ip: str, timeout: float) -> dict:
    """Aperto de mão TLS contra o IP já validado (SNI = host). Bloqueante: chamar via thread."""
    info: dict = {"host": host, "port": port}
    for verify in (True, False):
        ctx = ssl.create_default_context()
        if not verify:
            ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        try:
            with socket.create_connection((ip, port), timeout=timeout) as raw, ctx.wrap_socket(raw, server_hostname=host) as tls:
                info.update(verified=verify, protocol=tls.version(), cipher=(tls.cipher() or (None,))[0])
                cert = tls.getpeercert() if verify else {}
                if cert:
                    not_after = datetime.fromtimestamp(ssl.cert_time_to_seconds(cert["notAfter"]), timezone.utc)
                    info.update(
                        subject=dict(x[0] for x in cert.get("subject", ())).get("commonName"),
                        issuer=dict(x[0] for x in cert.get("issuer", ())).get("organizationName")
                        or dict(x[0] for x in cert.get("issuer", ())).get("commonName"),
                        san=[v for k, v in cert.get("subjectAltName", ()) if k == "DNS"][:10],
                        not_after=not_after.date().isoformat(),
                        days_left=(not_after - datetime.now(timezone.utc)).days,
                    )
                return info
        except ssl.SSLCertVerificationError as exc:
            info["error"] = f"certificado não verificado: {exc.verify_message or exc}"
            info["verified"] = False
        except (OSError, ssl.SSLError) as exc:
            info.update(verified=None, error=f"falha TLS: {exc.__class__.__name__}: {exc}")
            return info
    return info


class WebInspectArgs(BaseModel):
    url: str = Field(min_length=8, max_length=2048, description="URL http(s) pública a inspecionar.")
    max_bytes: int | None = Field(default=None, ge=100, description="Limite opcional menor que o configurado.")
    include_body: bool = Field(default=False, description="Uso interno da investigação: devolve o corpo bruto para análise local.")

    @field_validator("url")
    @classmethod
    def _clean_url(cls, value: str) -> str:
        try:
            return normalize_candidate_url(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc


class WebInspectTool(Tool):
    args_model = WebInspectArgs

    def __init__(self, net_settings: NetToolsSettings, transport: httpx.AsyncBaseTransport | None = None, tls=tls_probe):
        super().__init__()
        self._settings = net_settings
        self._transport = transport
        self._tls = tls
        self._next_slot: dict[str, float] = {}  # espaçamento por host (reserva de horário; sem lock no event loop)
        self.spec = ToolSpec(
            id="net.inspect",
            name="Passive inspection of a public URL",
            description=(
                "Passively inspects a public http(s) URL: status, HTTP headers (including security), "
                "redirect chain, cookie attributes, certificate/TLS, and HTML signals (forms, scripts, technologies). "
                "Read-only; same network blocks as net.fetch."
            ),
            capabilities=["net", "read-only"],
            required_permissions=["net.fetch"],
        )

    def _client(self, pinned: dict[tuple[str, int], str]) -> httpx.AsyncClient:
        transport = self._transport
        if transport is None:
            transport = httpx.AsyncHTTPTransport(trust_env=False)
            transport._pool._network_backend = _PinnedNetworkBackend(pinned)  # type: ignore[attr-defined]
        return httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=self._settings.timeout_seconds,
            transport=transport,
            headers={"User-Agent": "Enslaved/0.1 (local assistant; passive inspection)"},
        )

    async def _pace(self, host: str) -> None:
        interval = self._settings.min_host_interval_seconds
        now = time.monotonic()
        slot = max(now, self._next_slot.get(host, 0.0))
        self._next_slot[host] = slot + interval
        if slot > now:
            await asyncio.sleep(slot - now)

    async def run(self, args: dict, permission=None) -> ToolResult:
        if permission is None or not permission.granted:
            return ToolResult(ok=False, error="permissão ausente")
        try:
            url = normalize_candidate_url(args["url"])
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))
        limit = min(args.get("max_bytes") or self._settings.max_bytes, self._settings.max_bytes)
        ports = tuple(self._settings.allowed_ports)
        pinned: dict[tuple[str, int], str] = {}
        tls: dict | None = None
        chain: list[dict] = []
        current = url
        response_headers: dict[str, str] = {}
        cookies: list[dict] = []
        status, content_type, payload, truncated = None, "", b"", False
        try:
            async with self._client(pinned) as client:
                for hop in range(self._settings.max_redirects + 1):
                    try:
                        _scheme, host, port, ips = await asyncio.to_thread(resolve_public_url, current, ("http", "https"), ports)
                    except UrlBlockedError as exc:
                        if hop == 0:
                            return ToolResult(ok=False, error=f"URL bloqueada: {exc}")
                        chain[-1].update(followed=False, reason=f"destino bloqueado: {exc}")
                        break
                    pinned[(host.lower(), port)] = ips[0]
                    if tls is None and _scheme == "https":
                        tls = await asyncio.to_thread(self._tls, host, port, ips[0], min(self._settings.timeout_seconds, 8.0))
                    await self._pace(host.lower())
                    async with client.stream("GET", current) as response:
                        status = response.status_code
                        response_headers = {
                            k.lower(): redact_secrets(v)[:400]
                            for k, v in list(response.headers.items())[:_MAX_HEADERS]
                            if k.lower() != "set-cookie"
                        }
                        cookies = [_parse_cookie(c) for c in response.headers.get_list("set-cookie")[:20]]
                        location = response.headers.get("location")
                        if status in _REDIRECTS and location:
                            nxt = str(httpx.URL(current).join(location))
                            hop_info = {"url": current, "status": status, "location": redact_secrets(nxt)}
                            chain.append(hop_info)
                            if not same_target(url, nxt):
                                hop_info.update(followed=False, reason="host diferente do alvo autorizado; não visitado")
                                break
                            if hop >= self._settings.max_redirects:
                                hop_info.update(followed=False, reason="limite de redirecionamentos")
                                break
                            hop_info["followed"] = True
                            current = nxt
                            continue
                        content_type = response.headers.get("content-type", "")
                        if content_type.lower().startswith(_TEXT_TYPES):
                            async for chunk in response.aiter_bytes():
                                payload += chunk
                                if len(payload) >= limit:
                                    payload, truncated = payload[:limit], True
                                    break
                        break
        except httpx.TransportError as exc:
            return ToolResult(ok=False, error=f"Falha de rede: {exc.__class__.__name__}: {redact_secrets(str(exc))}")

        text = payload.decode("utf-8", errors="replace")
        title, signals = None, {}
        if "html" in content_type.lower():
            def analyse():
                sig = _Signals(current)
                sig.feed(text)
                t, _body = html_to_text(text)
                return t, sig

            title, sig = await asyncio.to_thread(analyse)
            origin = urlsplit(current)
            same_origin = sorted({urlsplit(urljoin(current, h))._replace(query="", fragment="").geturl() for h in sig.links
                                  if urlsplit(urljoin(current, h)).scheme in ("http", "https") and same_target(current, urljoin(current, h))})[:80]
            signals = {
                "forms": sig.forms, "scripts": sig.scripts, "styles": sig.styles, "iframes": sig.iframes,
                "meta": sig.meta, "comments": sig.comments, "inline_scripts": sig.inline_scripts,
                "links": [u for u in same_origin if urlsplit(u).netloc == origin.netloc] or same_origin,
            }
        data = {
            "url": url, "final_url": current, "status": status, "redirect_chain": chain, "headers": response_headers,
            "cookies": cookies, "tls": tls, "content_type": content_type.split(";")[0].strip(), "bytes": len(payload),
            "truncated": truncated, "title": title, "sha256": hashlib.sha256(payload).hexdigest(), "signals": signals,
            **({"body": text[:200_000]} if args.get("include_body") else {}),
            "untrusted": True,
        }
        summary = f"HTTP {status} {redact_secrets(current)} · {len(response_headers)} cabeçalhos · {len(cookies)} cookie(s) · {len(chain)} redirecionamento(s)"
        return ToolResult(ok=True, output=summary, data=data)

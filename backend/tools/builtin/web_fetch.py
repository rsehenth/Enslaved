"""Ferramenta de consulta de conteúdo público na web (somente leitura, com proteção SSRF)."""

import asyncio
import hashlib
import html as html_module
import re

import httpcore
import httpx
from pydantic import BaseModel, Field, field_validator

from backend.core.config import NetToolsSettings
from backend.core.redact import redact_secrets
from backend.tools.base import Tool, ToolResult, ToolSpec
from backend.tools.net_guard import UrlBlockedError, normalize_candidate_url, resolve_public_url

_ALLOWED_CONTENT_TYPES = ("text/", "application/json", "application/xml", "application/rss", "application/xhtml")

# Tudo linear: `.*?</\1>` e `<[^>]+>` eram quadráticos em páginas hostis ("<script>" ou "<a " repetidos sem fechamento)
# e, por rodarem no event loop, congelavam o servidor por minutos. `[^<>]` impede que uma tentativa atravesse outra tag.
_RAW_OPEN = re.compile(r"<(script|style|noscript|template)\b[^<>]*>", re.IGNORECASE)
_TITLE_OPEN = re.compile(r"<title\b[^<>]*>", re.IGNORECASE)
_TAG = re.compile(r"<[^<>]+>")
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _strip_comments(html: str) -> str:
    out, pos = [], 0
    while (start := html.find("<!--", pos)) != -1:
        out += [html[pos:start], " "]
        end = html.find("-->", start + 4)
        if end == -1:  # comentário sem fim: o resto do documento é comentário
            return "".join(out)
        pos = end + 3
    out.append(html[pos:])
    return "".join(out)


def _strip_raw_blocks(html: str) -> str:
    lower, out, pos = html.lower(), [], 0
    while match := _RAW_OPEN.search(html, pos):
        out += [html[pos : match.start()], " "]
        end = lower.find("</" + match.group(1).lower(), match.end())
        close = html.find(">", end) if end != -1 else -1
        if close == -1:  # bloco sem fim: o resto é script/estilo
            return "".join(out)
        pos = close + 1
    out.append(html[pos:])
    return "".join(out)


def html_to_text(html: str) -> tuple[str | None, str]:
    """Extração de texto determinística e minimalista (sem dependências externas)."""
    without_scripts = _strip_raw_blocks(_strip_comments(html))
    title = None
    if title_open := _TITLE_OPEN.search(without_scripts):
        end = without_scripts.lower().find("</title", title_open.end())
        if end != -1:
            title = html_module.unescape(without_scripts[title_open.end() : end]).strip()
    text = _TAG.sub("\n", without_scripts)
    text = html_module.unescape(text)
    text = _WHITESPACE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return title, text.strip()


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Fixa o IP validado por (host, porta) evitando DNS rebinding (TOCTOU) mantendo SNI/TLS do host."""

    def __init__(self, pinned_ips: dict[tuple[str, int], str]):
        self._inner = httpcore.AnyIOBackend()
        self._pinned = pinned_ips

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ) -> httpcore.AsyncNetworkStream:
        pinned_ip = self._pinned.get((host.lower(), port))
        if pinned_ip is None:
            raise httpcore.ConnectError(f"Host não pré-validado para conexão direta: {host}:{port}")
        return await self._inner.connect_tcp(
            pinned_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, path: str, timeout: float | None = None, socket_options=None):
        raise httpcore.ConnectError("Unix sockets bloqueados em net.fetch")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


class WebFetchArgs(BaseModel):
    url: str = Field(min_length=8, max_length=2048, description="URL http(s) pública.")
    max_bytes: int | None = Field(default=None, ge=100, description="Limite opcional menor que o configurado.")

    @field_validator("url")
    @classmethod
    def _clean_url(cls, value: str) -> str:
        try:
            return normalize_candidate_url(value)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc


class WebFetchTool(Tool):
    args_model = WebFetchArgs

    def __init__(self, net_settings: NetToolsSettings, transport: httpx.AsyncBaseTransport | None = None):
        super().__init__()
        self._settings = net_settings
        self._transport = transport
        self.spec = ToolSpec(
            id="net.fetch",
            name="Public web lookup",
            description=(
                "Fetches a public http(s) URL and returns the text extracted from the page. "
                "Read-only; blocks internal/private addresses, unusual ports, and binary content; "
                "limits size and number of redirects."
            ),
            capabilities=["net", "read-only"],
            required_permissions=["net.fetch"],
        )

    def _client(self, pinned_ips: dict[tuple[str, int], str]) -> httpx.AsyncClient:
        transport = self._transport
        if transport is None:
            transport = httpx.AsyncHTTPTransport(trust_env=False)
            transport._pool._network_backend = _PinnedNetworkBackend(pinned_ips)  # type: ignore[attr-defined]
        return httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=self._settings.timeout_seconds,
            transport=transport,
            headers={"User-Agent": "Enslaved/0.1 (assistente local; uso pessoal)"},
        )

    def _check_content_type(self, content_type: str) -> bool:
        lowered = content_type.lower()
        return any(lowered.startswith(prefix) for prefix in _ALLOWED_CONTENT_TYPES)

    async def run(self, args: dict, permission=None) -> ToolResult:
        if permission is None or not permission.granted:
            return ToolResult(ok=False, error="permissão ausente")
        try:
            url = normalize_candidate_url(args["url"])
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))
        limit = min(args.get("max_bytes") or self._settings.max_bytes, self._settings.max_bytes)

        pinned_ips: dict[tuple[str, int], str] = {}
        try:
            _scheme, host, port, ips = await asyncio.to_thread(
                resolve_public_url, url, ("http", "https"), tuple(self._settings.allowed_ports)
            )
            pinned_ips[(host.lower(), port)] = ips[0]
        except UrlBlockedError as exc:
            return ToolResult(ok=False, error=f"URL bloqueada: {exc}")

        current = url
        redirects = 0
        status = None
        content_type = ""
        payload = b""
        truncated = False
        try:
            async with self._client(pinned_ips) as client:
                for _ in range(self._settings.max_redirects + 1):
                    async with client.stream("GET", current) as response:
                        status = response.status_code
                        if status in (301, 302, 303, 307, 308):
                            location = response.headers.get("location")
                            if not location:
                                return ToolResult(ok=False, error=f"Redirecionamento sem destino (HTTP {status}).")
                            if redirects >= self._settings.max_redirects:
                                return ToolResult(
                                    ok=False, error=f"Excesso de redirecionamentos (> {self._settings.max_redirects})."
                                )
                            redirects += 1
                            current = str(httpx.URL(current).join(location))
                            try:
                                _s, r_host, r_port, r_ips = await asyncio.to_thread(
                                    resolve_public_url,
                                    current,
                                    ("http", "https"),
                                    tuple(self._settings.allowed_ports),
                                )
                                pinned_ips[(r_host.lower(), r_port)] = r_ips[0]
                            except UrlBlockedError as exc:
                                return ToolResult(ok=False, error=f"Redirecionamento bloqueado: {exc}")
                            continue
                        if status >= 400:
                            return ToolResult(
                                ok=False,
                                error=f"O servidor respondeu HTTP {status} para {redact_secrets(current)}",
                            )
                        content_type = response.headers.get("content-type", "")
                        if not self._check_content_type(content_type):
                            return ToolResult(
                                ok=False,
                                error=f"Tipo de conteúdo não suportado ({content_type.split(';')[0] or 'desconhecido'}); apenas texto.",
                            )
                        async for chunk in response.aiter_bytes():
                            payload += chunk
                            if len(payload) >= limit:
                                payload = payload[:limit]
                                truncated = True
                                break
                        break
        except httpx.TransportError as exc:
            return ToolResult(ok=False, error=f"Falha de rede: {exc.__class__.__name__}: {redact_secrets(str(exc))}")

        charset = "utf-8"
        if "charset=" in content_type:
            charset = content_type.split("charset=", 1)[1].split(";", 1)[0].strip().strip('"') or "utf-8"
        try:
            text = payload.decode(charset, errors="replace")
        except LookupError:
            text = payload.decode("utf-8", errors="replace")
        raw = text
        if "html" in content_type.lower():
            title, text = html_to_text(text)
        else:
            title = None
        # internal use (Deep Research): `_keep_raw` is not part of args_model, so the model cannot enable it
        extra = {"raw": raw} if args.get("_keep_raw") else {}
        return ToolResult(
            ok=True,
            output=text,
            data={
                "url": url,
                "final_url": current,
                "status": status,
                "content_type": content_type.split(";")[0].strip(),
                "bytes": len(payload),
                "truncated": truncated,
                "redirects": redirects,
                "title": title,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "untrusted": True,
                **extra,
            },
        )

"""Guarda de acesso à rede para ferramentas (proteção SSRF e DNS rebinding).

Toda URL usada por ferramentas passa por `resolve_public_url` / `validate_public_url`:
- apenas http/https em portas públicas padrão;
- sem credenciais na URL;
- bloqueio de nomes locais reservados (.localhost, .local, .internal);
- resolução de TODOS os endereços do host rejeitando qualquer IP não global
  (loopback, privada, link-local incl. 169.254.169.254, CGNAT, NAT64, 6to4, Teredo,
  unique-local, reserved), para IPv4 e IPv6 (inclui mapeados ::ffff:…).
"""

import ipaddress
import socket
import re
from urllib.parse import parse_qsl, unquote, urlsplit

_NAT64_WELL_KNOWN = ipaddress.IPv6Network("64:ff9b::/96")
_NAT64_LOCAL = ipaddress.IPv6Network("64:ff9b:1::/48")
_BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal")


class UrlBlockedError(ValueError):
    """URL rejeitada pela política de rede (não pública, esquema/porta proibida)."""


_MD_LINK = re.compile(r"^\[([^\[\]\n]+)\]\(\s*(<[^<>\s]+>|[^)\s]+)(?:\s+\"[^\"]*\")?\s*\)$")
_URL_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
_BARE_HOST = re.compile(
    r"^(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}(?::\d{2,5})?(?:[/?#][^\s]*)?$",
    re.IGNORECASE,
)


def normalize_candidate_url(raw: str) -> str:
    """Normaliza a URL que chega às ferramentas net.* SEM ampliar o que é aceito.

    Remove apenas embalagens triviais (link Markdown `[url](url)`, aspas, backticks, <ângulos>)
    e adiciona o esquema `https://` quando o destino é inequivocamente um host público. Texto
    ambíguo (espaços, vários candidatos, Markdown solto) é rejeitado. A validação SSRF
    (`resolve_public_url`) continua rodando sobre a URL FINAL — nada aqui contorna a proteção."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("URL vazia")
    for _ in range(3):
        if (match := _MD_LINK.fullmatch(text)) is not None:
            text = match.group(2).strip()
            if text.startswith("<") and text.endswith(">"):
                text = text[1:-1].strip()
            continue
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ("\"", "'", "`"):
            text = text[1:-1].strip()
            continue
        if text.startswith("<") and text.endswith(">") and len(text) > 2:
            text = text[1:-1].strip()
            continue
        break
    if not text or _MD_LINK.search(text) is not None:
        raise ValueError(f"URL ambígua; envie somente a URL pura, ex.: https://site.com (recebido: {raw[:120]!r})")
    if re.search(r"\s", text):
        raise ValueError(f"URL contém espaços; envie somente a URL pura, ex.: https://site.com (recebido: {raw[:120]!r})")
    if _URL_SCHEME.match(text) is None:
        if _BARE_HOST.match(text) is None:
            raise ValueError(f"Não parece uma URL http(s); envie a URL pura, ex.: https://site.com (recebido: {raw[:120]!r})")
        text = f"https://{text}"
    return text


def _require_global_ip(ip_text: str) -> str:
    ip = ipaddress.ip_address(ip_text)
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.version == 6:
        if ip in _NAT64_WELL_KNOWN or ip in _NAT64_LOCAL or ip.sixtofour is not None or ip.teredo is not None:
            raise UrlBlockedError(f"Endereço IPv6 de transição/interno bloqueado: {ip}")
    if not ip.is_global:
        raise UrlBlockedError(f"Endereço não público bloqueado: {ip}")
    return str(ip)


def resolve_public_url(
    url: str,
    allowed_schemes: tuple[str, ...] = ("http", "https"),
    allowed_ports: tuple[int, ...] = (80, 443),
) -> tuple[str, str, int, list[str]]:
    """Valida URL/host e retorna (esquema, host, porta, ips_globais_resolvidos)."""
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "").lower()
    if scheme not in allowed_schemes:
        raise UrlBlockedError(f"Esquema não permitido: {scheme!r} (permitidos: {', '.join(allowed_schemes)})")
    if parts.username or parts.password:
        raise UrlBlockedError("URL com credenciais não é permitida")
    host = parts.hostname
    if not host:
        raise UrlBlockedError("URL sem host")
    normalized_host = host.rstrip(".").lower()
    if normalized_host == "localhost" or normalized_host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UrlBlockedError(f"Host local/interno bloqueado: {host!r}")
    try:
        port = parts.port
    except ValueError as exc:
        raise UrlBlockedError(f"Porta inválida na URL: {exc}") from exc
    if port is None:
        port = 443 if scheme == "https" else 80
    if port not in allowed_ports:
        raise UrlBlockedError(f"Porta não permitida: {port} (permitidas: {', '.join(map(str, allowed_ports))})")

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return scheme, host, port, [_require_global_ip(str(literal))]

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UrlBlockedError(f"Falha ao resolver o host {host!r}: {exc}") from exc
    if not infos:
        raise UrlBlockedError(f"Host sem endereços resolvidos: {host!r}")
    resolved_ips: list[str] = []
    for info in infos:
        checked = _require_global_ip(info[4][0])
        if checked not in resolved_ips:
            resolved_ips.append(checked)
    return scheme, host, port, resolved_ips


def validate_public_url(
    url: str,
    allowed_schemes: tuple[str, ...] = ("http", "https"),
    allowed_ports: tuple[int, ...] = (80, 443),
) -> tuple[str, str, int]:
    """Valida a URL e o host resolvido. Retorna (esquema, host, porta) ou levanta UrlBlockedError."""
    scheme, host, port, _ips = resolve_public_url(url, allowed_schemes, allowed_ports)
    return scheme, host, port


# ------------------------------------------------------------ análise de saída (egress)

_SECRET_PARAM = re.compile(r"token|key|secret|pass|pwd|auth|session|cookie|credential|sig", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_BLOB = re.compile(r"[A-Za-z0-9+/=_-]{32,}")
_LONG_DIGITS = re.compile(r"\d{11,19}|\d{3}\.\d{3}\.\d{3}-\d{2}")  # CPF, cartão, telefone


def _value_flags(value: str) -> list[str]:
    flags = []
    if _EMAIL.search(value):
        flags.append("e-mail")
    if _LONG_DIGITS.search(value.replace(" ", "")):
        flags.append("número longo (documento/cartão/telefone?)")
    if _BLOB.search(value):
        flags.append("bloco codificado/aleatório")
    elif len(value) > 80:
        flags.append(f"valor longo ({len(value)} caracteres)")
    return flags


def egress_details(url: str) -> tuple[str, list[str], list[str]]:
    """(domínio, parâmetros 'nome=valor', alertas de possíveis dados sensíveis saindo na URL)."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    params = parse_qsl(parts.query, keep_blank_values=True)
    warnings = []
    for name, value in params:
        flags = (["nome de parâmetro de credencial"] if _SECRET_PARAM.search(name) else []) + _value_flags(value)
        if flags:
            warnings.append(f"parâmetro '{name}': {', '.join(flags)}")
    for segment in unquote(parts.path).split("/"):
        flags = _value_flags(segment)
        if flags:
            warnings.append(f"trecho do caminho '{segment[:40]}{'…' if len(segment) > 40 else ''}': {', '.join(flags)}")
    return host, [f"{name}={value}" for name, value in params], warnings

"""Classificação determinística do pedido: responder direto, perguntar a profundidade ou investigar (sem modelo, sem rede)."""

import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

import idna

MODES = ("quick", "deep", "extended")


def _plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


# Níveis pedidos explicitamente (checados do mais forte ao mais fraco).
_EXTENDED = re.compile(
    r"\bestendid[ao]s?\b|\bextended\b|todas as evidencias|evidencias disponiveis|cobertura ampliada|exaustiv|o mais completo possivel|analise completa e detalhada"
)
_DEEP = re.compile(r"\bprofund[ao]s?\b|\bdeep\b|aprofundad|\ba fundo\b|investigacao completa|\bdetalhadamente\b")
_QUICK = re.compile(r"\brapid[ao]s?\b|\bquick\b|so uma olhada|\bbrevemente\b|resumidamente|visao geral")

# Investigação (verbo) e vocabulário de segurança / web.
_INVESTIGATE = re.compile(
    r"\binvestig|\baudit|\bpentest|\bavali\w+ (?:a )?seguranca|\banalis\w+ (?:a )?seguranca|\bverifi\w+ (?:a )?seguranca|"
    r"\bfa[cç]a (?:uma )?(?:analise|investigacao|auditoria|levantamento)|\blevantamento\b|\bdue diligence\b|"
    r"\bseguranca d[oea]s? (?:esse|este|o|do|desse|deste)\b|\bvulnerabilidades? d[oea]\b"
)
_SECURITY = re.compile(
    r"seguranc|vulnerab|\bcves?\b|cabecalh|\bheaders?\b|\btls\b|\bssl\b|certificad|\bcsp\b|\bcors\b|\bhsts\b|cookie|hardening|exposic|pentest|auditoria"
)
_WEB_CUE = re.compile(r"\bsite\b|\bdominio\b|\bwebsite\b|\bsite\b|na internet|\bonline\b|\bfontes\b|\bweb\b|\bpagina\b|\bapi publica\b|https?://")
_LIGHT = re.compile(r"\bveja\b|\bo que\b.*\b(?:oferece|faz)\b|do que se trata|\bresuma\b|\bdescreva\b|\bdiga\b|\bqual\b")
_ACTIVE = re.compile(
    r"\bexploit|explor\w+ (?:a |as )?(?:falha|vulnerab)|forca bruta|brute.?force|\bfuzz|sql ?injection|injetar|\bddos\b|\bdos\b|"
    r"invadir|acesso indevido|burlar|derrubar|extrair dados|quebrar senha|\bxss\b.*(?:testar|explor)"
)

_URL = re.compile(r"https?://[^\s<>\"'`\]]+", re.IGNORECASE)
_BARE = re.compile(r"(?<![\w@./-])((?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[a-z]{2,24})(?:/[^\s<>\"'`\]]*)?(?![\w@-])")  # TLD em minúsculas: "fim.Depois" não é domínio
# Pergunta conceitual/explicativa: o conhecimento do modelo responde; NÃO abre investigação web
# ("Explique como funciona SQL Injection", "Como você investigaria a segurança de uma aplicação?").
_CONCEPTUAL = re.compile(
    r"\bexpliqu(?:e|em|es)\b|\bme explique\b|\bcomo funciona\b|\bcomo funcionam\b|"
    r"\bcomo (?:voce|o enslaved|o assistente) (?:pode|poderia|deve|deveria|faria|investig\w*|avali\w*|analis\w*|verific\w*|implement\w*|faz)\b|"
    r"\bo que (?:e|sao|voce sabe|voce entende)\b|\bqual (?:e |a |o )?(?:diferenca|conceito)\b|\bdiferenca entre\b|"
    r"\bconceito de\b|\bdefin(?:a|icao de)\b|\bexplains?\b|\bhow does\b|\bhow would you\b|\bwhat is\b|\bwhat are\b|\bdifference between\b"
)
# Pedido EXPLÍCITO de pesquisa na web (imperativo ou "fazer uma pesquisa"), sem confundir com o substantivo solto.
_SEARCH_CUE = re.compile(
    r"\bpesquis(?:e|es|ar|ando)\b|\bbusqu(?:e|es|ar|ando)\b|\bprocur(?:e|es|ar|ando)\b|"
    r"\b(?:faz|fazer|fa[cç]a)\s+uma\s+pesquisa\b|\bpesquisa\s+(?:na\s+internet|na\s+web|web)\b|\bgoogl(?:e|ar)\b|"
    r"\bsearch (?:the )?web\b|\bsearch for\b|\blook up\b"
)
# "Verificar NA INTERNET/WEB/fontes": pedido de verificação qualificado — "verifique a API" (sem
# menção à internet) continua sendo tarefa local, não pesquisa web.
_EXPLICIT_VERIFY = re.compile(
    r"\bverifiq\w*[^\n.]{0,40}\b(?:internet|web|online|fontes?)\b|"
    r"\b(?:internet|web|fontes?)[^\n.]{0,40}\bverifiq\w*\b",
    re.IGNORECASE,
)
_FILE_LIKE = {"py", "md", "js", "ts", "sh", "rs", "pl", "cs", "json", "txt", "yml", "yaml", "html", "css", "db", "exe", "bat", "ps1", "ini", "log", "csv", "pdf", "png", "jpg", "zip", "lock", "toml", "cfg", "tsx", "jsx", "vue", "so", "dll", "gz", "xml", "sql", "mjs", "docx", "xlsx", "pptx", "jpeg", "gif", "svg", "webp", "mp4", "mp3", "wav", "tar", "rar", "ipynb", "java", "kt", "cpp", "hpp", "php", "rb", "swift", "bak", "tmp", "env", "key", "pem", "crt", "lnk", "iso", "dat", "bin", "md5", "tsv", "rst", "gradle", "proto", "wasm"}


def _remove_dot_segments(path: str) -> str:
    if not path or ("/." not in path and not path.startswith(".")):
        return path
    out: list[str] = []
    trailing = path.endswith("/")
    for seg in path.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if out:
                out.pop()
            trailing = True
        else:
            out.append(seg)
            trailing = False
    if not out:
        return "/"
    return "/" + "/".join(out) + ("/" if path.endswith("/") or trailing else "")


def _has_open_delimiter(line_prefix: str, delim: str) -> bool:
    if not line_prefix:
        return False
    if line_prefix.endswith(delim):
        return True
    if len(delim) >= 2:
        return line_prefix.count(delim) % 2 == 1
    doubled = delim * 2
    singles = line_prefix.replace(doubled, "")
    if delim == "*":
        return singles.count("*") % 2 == 1
    return bool(re.search(r"(?:^|[\s(\[\"'])_$", line_prefix)) or singles.count("_") % 2 == 1


def clean_extracted_url(raw: str, prefix: str = "") -> str:
    """Remove delimitadores Markdown externos e pontuação final sem alterar caracteres válidos da URL."""
    url = raw.strip()
    if not prefix:
        for d in ("***", "**", "~~", "__", "`", "*", "_"):
            while len(url) > len(d) * 2 and url.startswith(d) and url.endswith(d):
                url = url[len(d) : -len(d)].strip()
        if url.startswith("<") and url.endswith(">") and len(url) > 2:
            url = url[1:-1].strip()
        if url.startswith("(") and url.endswith(")") and len(url) > 2:
            url = url[1:-1].strip()
    line_prefix = prefix.rsplit("\n", 1)[-1] if prefix else ""
    while url:
        prev = url
        url = url.rstrip(".,;:!?").strip("`")
        while url.endswith(")") and url.count(")") > url.count("("):
            url = url[:-1]
        for delim in ("***", "**", "~~", "__", "*", "_"):
            if url.endswith(delim) and len(url) > len(delim) and _has_open_delimiter(line_prefix, delim):
                url = url[: -len(delim)]
                break
        if url == prev:
            break
    return url


def canonicalize_url(
    url: str,
    *,
    keep_query: bool = True,
    keep_fragment: bool = False,
    ensure_root_slash: bool = True,
    strip_trailing_slash: bool = False,
) -> str:
    """Canonicaliza esquema, host (IDNA), porta padrão, dot-segments e query/fragmento."""
    cleaned = clean_extracted_url(url)
    parts = urlsplit(cleaned)
    scheme = (parts.scheme or "").lower()
    if scheme not in ("http", "https") or not parts.hostname:
        return ""
    raw_host = parts.hostname.rstrip(".")
    if ":" in raw_host:
        host = f"[{raw_host.lower()}]"
    else:
        try:
            host = idna.encode(raw_host).decode("ascii").lower()
        except idna.IDNAError:
            host = raw_host.lower()
    try:
        port = parts.port
    except ValueError:
        return ""
    default_port = 443 if scheme == "https" else 80
    netloc = host if port in (None, default_port) else f"{host}:{port}"
    path = _remove_dot_segments(parts.path)
    if strip_trailing_slash and path:
        path = path.rstrip("/") or "/"
    if not path and ensure_root_slash:
        path = "/"
    query = parts.query if keep_query else ""
    fragment = parts.fragment if keep_fragment else ""
    return urlunsplit((scheme, netloc, path, query, fragment))


def url_dedup_key(url: str, *, include_query: bool = False) -> str:
    """Chave canônica para deduplicação de URLs equivalentes."""
    return canonicalize_url(url, keep_query=include_query, keep_fragment=False, ensure_root_slash=True, strip_trailing_slash=True)


def extract_urls(text: str, *, keep_query: bool = True) -> tuple[str, ...]:
    """Extrai URLs http(s) válidas de texto/Markdown preservando paths, queries e caracteres válidos."""
    out: list[str] = []
    seen: set[str] = set()
    for m in _URL.finditer(text):
        cleaned = clean_extracted_url(m.group(0), text[: m.start()])
        parts = urlsplit(cleaned)
        if not parts.hostname or parts.username:
            continue
        canon = canonicalize_url(cleaned, keep_query=keep_query, keep_fragment=False, ensure_root_slash=bool(parts.path))
        if not canon:
            continue
        key = url_dedup_key(canon, include_query=keep_query)
        if key not in seen:
            seen.add(key)
            out.append(canon)
    return tuple(out)


@dataclass(frozen=True)
class Intent:
    kind: str  # simple | local | ambiguous | research
    mode: str = "quick"  # nível pedido (explícito) ou padrão
    explicit: bool = False
    profile: str = "general"  # security | general
    targets: tuple[str, ...] = ()
    active_requested: bool = False
    reason: str = ""
    goal: str = field(default="", compare=False)


def extract_targets(text: str) -> tuple[str, ...]:
    """URLs https?:// e domínios soltos (ignora nomes de arquivo). Deduplicado por forma canônica, na ordem de aparição."""
    found: list[tuple[str, bool]] = []
    for m in _URL.finditer(text):
        cleaned = clean_extracted_url(m.group(0), text[: m.start()])
        found.append((cleaned, bool(urlsplit(cleaned).path)))
    covered = " ".join(u for u, _has_path in found).lower()
    for m in _BARE.finditer(text):
        domain = m.group(1).lower()
        if domain.rsplit(".", 1)[-1] in _FILE_LIKE or domain in covered or domain.startswith(("e.g", "i.e")):
            continue
        cleaned = clean_extracted_url("https://" + m.group(0), text[: m.start()])
        found.append((cleaned, bool(urlsplit(cleaned).path)))
    out: list[str] = []
    seen: set[str] = set()
    for url, has_path in found:
        parts = urlsplit(url)
        if parts.hostname and not parts.username:
            clean = canonicalize_url(url, keep_query=False, keep_fragment=False, ensure_root_slash=has_path)
            if not clean:
                continue
            key = url_dedup_key(clean, include_query=False)
            if key not in seen:
                seen.add(key)
                out.append(clean)
    return tuple(out[:5])


def explicit_search_request(text: str) -> bool:
    """Pedido EXPLÍCITO de busca/verificação na internet nesta mensagem (ex.: "pesquise",
    "verifique na internet"). Não confunde "verifique a API" (tarefa local) com pesquisa web."""
    plain = _plain(text)
    return bool(_SEARCH_CUE.search(plain)) or bool(_EXPLICIT_VERIFY.search(plain))


def classify(text: str, *, explicit_mode: str | None = None) -> Intent:
    """Decide o tratamento do pedido.

    - `simple`: pergunta comum ou "veja o que o site oferece" (modo rápido, fluxo existente);
    - `local`: tarefa complexa que não precisa de pesquisa web (nenhuma pergunta de profundidade);
    - `ambiguous`: investigação complexa sem nível claro (mostrar a escolha no chat);
    - `research`: investigação com nível definido (explícito no texto ou escolhido no cartão).
    """
    plain = _plain(text)
    targets = extract_targets(text)
    active = bool(_ACTIVE.search(plain))
    security = bool(_SECURITY.search(plain))
    profile = "security" if targets and (security or active) else "general"
    asked = next((m for m, rx in (("extended", _EXTENDED), ("deep", _DEEP), ("quick", _QUICK)) if rx.search(plain)), None)
    mode = explicit_mode if explicit_mode in MODES else asked
    investigate = bool(_INVESTIGATE.search(plain))
    web = bool(targets) or bool(_WEB_CUE.search(plain))
    conceptual = bool(_CONCEPTUAL.search(plain))
    search_verb = bool(_SEARCH_CUE.search(plain))
    base = dict(profile=profile, targets=targets, active_requested=active, goal=text.strip()[:300])

    if mode == "quick":
        return Intent("simple", "quick", True, reason="nível rápido pedido", **base)
    if conceptual and not mode and not targets:
        # Explicação/pesquisa conceitual: o modelo responde do próprio conhecimento, sem investigação web
        # (mesmo que o texto contenha verbos como "investigar"/"segurança" — ex.: "como você investigaria…").
        return Intent("simple", "quick", reason="pergunta conceitual: resposta direta do modelo", **base)
    if search_verb and not targets:
        # Pesquisa na web pedida explicitamente ("Pesquise a história do Guarany de Bagé"): fluxo de pesquisa,
        # nível rápido — sem pergunta de profundidade e sem auditoria Deep Research.
        return Intent("research", "quick", True, reason="pesquisa web pedida explicitamente", **base)
    wants_research = (investigate and (web or security)) or (mode in ("deep", "extended") and (web or security or investigate))
    if not wants_research:
        if investigate or security:  # complexo, mas resolvível sem coletar evidências na web
            return Intent("local", "quick", reason="tarefa complexa sem necessidade de pesquisa web", **base)
        return Intent("simple", "quick", reason="pergunta comum", **base)
    if mode:
        return Intent("research", mode, True, reason="nível definido", **base)
    if targets and not investigate and _LIGHT.search(plain):
        return Intent("simple", "quick", reason="consulta leve de site", **base)
    return Intent("ambiguous", "quick", reason="investigação complexa sem nível definido", **base)

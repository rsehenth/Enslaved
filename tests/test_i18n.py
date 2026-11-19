"""Internacionalização pt-BR / en-US: catálogos, cobertura de strings fixas e idioma das respostas (sem navegador, sem Ollama)."""

import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

from backend.chat.models import LANGUAGE_HINTS, ChatRequest
from backend.providers.base import ModelProvider

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
LANGS = ("pt-BR", "en-US")
CATALOG = {lang: json.loads((FRONTEND / "i18n" / f"{lang}.json").read_text(encoding="utf-8")) for lang in LANGS}
PLACEHOLDER = re.compile(r"\{(\w+)\}")


def flat(lang):
    return {k: v for k, v in CATALOG[lang].items() if isinstance(v, str)}


def test_catalogs_have_the_same_keys_and_no_blank_values():
    pt, en = flat("pt-BR"), flat("en-US")
    assert set(pt) == set(en), sorted(set(pt) ^ set(en))
    assert all(v.strip() for v in [*pt.values(), *en.values()])


def test_placeholders_match_between_languages():
    pt, en = flat("pt-BR"), flat("en-US")
    bad = [k for k in pt if set(PLACEHOLDER.findall(pt[k])) != set(PLACEHOLDER.findall(en[k]))]
    assert not bad, bad


def test_plural_keys_come_in_pairs():
    for lang in LANGS:
        keys = flat(lang)
        for k in keys:
            if k.endswith("_one"):
                assert k[:-4] + "_other" in keys, k


def test_english_is_really_translated_where_text_differs_by_design():
    pt, en = flat("pt-BR"), flat("en-US")
    for key in ("chat.new", "settings.title", "memory.title", "research.step.collecting", "tool.approve", "common.error"):
        assert pt[key] != en[key]


def _used_keys():
    """Chaves referenciadas literalmente em tr('..'), bind*(…, '..') e data-i18n*."""
    keys = set()
    for path in [*(FRONTEND / "js").glob("*.js"), FRONTEND / "index.html"]:
        text = path.read_text(encoding="utf-8")
        keys.update(re.findall(r"""\bT?tr\(\s*['"]([\w.]+)['"]""", text))
        keys.update(re.findall(r"""bind(?:Attr|Tm)?\([^\n]*?['"](\w+\.[\w.]+)['"]""", text))
        keys.update(re.findall(r'data-i18n="([\w.]+)"', text))
        for group in re.findall(r'data-i18n-attr="([^"]+)"', text):
            keys.update(pair.split(":")[1] for pair in group.split(";"))
        keys.update(re.findall(r"""(?:Key|key)\s*[:=]\s*['"]([a-z]+\.[\w.]+)['"]""", text))
        keys.update(re.findall(r"""['"](research\.step\.\w+)['"]""", text))
    return keys


def test_every_key_used_by_the_frontend_exists():
    catalog = flat("pt-BR")
    missing = sorted(k for k in _used_keys() if k not in catalog and not any(k + s in catalog for s in ("_one", "_other")))
    assert not missing, missing


def test_dynamic_key_families_are_complete():
    for lang in LANGS:
        cat = flat(lang)
        families = {
            "verify.": ["verified", "indication", "local", "unverified", "no_evidence"],
            "research.cat.": ["A", "H", "B", "C", "D"],
            "research.step.": ["planning", "collecting", "analyzing", "verifying", "reporting"],
        }
        for prefix, names in families.items():
            for name in names:
                assert prefix + name in cat, f"{lang}:{prefix + name}"


def test_unused_learning_and_memory_item_keys_removed():
    for lang in LANGS:
        keys = set(flat(lang))
        stale_prefixes = ("learning.", "ctx.", "kb.", "memory.k.")
        leftovers = sorted(k for k in keys if k.startswith(stale_prefixes) or k in {
            "nav.learning", "memory.searchPlaceholder", "memory.addPlaceholder", "memory.add",
            "memory.none", "memory.noMatch", "memory.pin", "memory.unpin", "memory.editPrompt", "memory.confirmDelete",
        })
        assert not leftovers, (lang, leftovers)


# ------------------------------------------------------------------ strings fixas esquecidas

class _TextScan(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.found = [], []

    def handle_starttag(self, tag, attrs):
        if tag in ("meta", "link", "img", "input", "br", "use", "path", "circle", "rect", "ellipse", "line", "g"):
            return
        self.stack.append((tag, dict(attrs)))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        text = data.strip()
        if not text or any(t in ("script", "style", "svg", "text") for t, _ in self.stack):
            return
        if any("data-i18n" in a or "data-tm" in a for _, a in self.stack):
            return
        self.found.append(text)


# nomes próprios, rótulos técnicos e conteúdo preenchido por JS
HTML_ALLOWED = {"Enslaved", "Skills", "GPU", "Provider", "Lucas S.", "—", "·", "×", "Enslaved Teste"}
HTML_ALLOWED |= {"New chat", "Skill", "Image"}  # títulos que o JavaScript sempre sobrescreve antes de exibir (conversa, skill, imagem)


def test_index_html_has_no_untranslated_visible_text():
    scan = _TextScan()
    scan.feed((FRONTEND / "index.html").read_text(encoding="utf-8"))
    leftovers = [t for t in scan.found if t not in HTML_ALLOWED and re.search(r"[A-Za-zÀ-ú]{3,}", t)]
    assert not leftovers, leftovers


def _strip_js_comments(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(re.sub(r"(?<![:'\"`\\])//.*$", "", line) for line in src.splitlines())


# identificador técnico enviado pelo backend (comparado, nunca exibido) e log de depuração no console
JS_ALLOWED = {"Investigação passiva", "Evento SSE inválido"}
PT_WORDS = re.compile(r"\b(Nenhum[a]?|Falha|Erro|Selecione|Aguarde|Salvar|Cancelar|Excluir|Editar|Carregando|Pesquisando|Copiar|Aprender)\b")


@pytest.mark.parametrize("name", ["app.js", "research.js", "api.js", "scene.js", "markdown.js"])
def test_js_has_no_hardcoded_portuguese_literals(name):
    code = _strip_js_comments((FRONTEND / "js" / name).read_text(encoding="utf-8"))
    literals = re.findall(r"""'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`""", code)
    offenders = [s for s in literals if (re.search(r"[áéíóúâêôãõçÁÉÍÓÚÇ]", s) or PT_WORDS.search(s)) and s.strip("'\"`") not in JS_ALLOWED]
    assert not offenders, offenders[:10]


# ------------------------------------------------------------------ textos do backend traduzidos no frontend

def _backend_source():
    text = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "backend").rglob("*.py"))
    return re.sub(r'"\s*\n\s*"', "", text)  # junta literais adjacentes (descrições longas)


@pytest.mark.parametrize("lang", LANGS)
def test_backend_exact_translations_still_match_a_real_backend_string(lang):
    source = _backend_source()
    stale = [k for k in CATALOG[lang]["backend.exact"] if k not in source]
    assert not stale, (lang, stale)


@pytest.mark.parametrize("lang", LANGS)
def test_backend_patterns_are_valid_regexes(lang):
    for p in CATALOG[lang]["backend.patterns"]:
        re.compile(p["re"])


# ------------------------------------------------------------------ idioma das respostas do Qwen

class Capture(ModelProvider):
    name = "capture"

    def __init__(self):
        super().__init__("capture-1b")
        self.calls = []

    async def chat(self, messages, **options):
        self.calls.append(messages)
        return "ok"

    async def stream_chat(self, messages, **options):
        self.calls.append(messages)
        yield "ok"

    async def health(self):
        return {"available": True}


def _system_text(provider):
    return "\n".join(m["content"] for m in provider.calls[-1] if m["role"] == "system")


@pytest.mark.parametrize("lang", LANGS)
def test_language_hint_goes_to_the_system_prompt_before_owner_text(client, lang):
    provider = client.app.state.chat_service.provider = Capture()
    resp = client.post("/api/chat/stream", json={"message": "hello", "language": lang, "system_prompt": "TEXTO-DO-DONO"})
    assert resp.status_code == 200
    system = _system_text(provider)
    assert LANGUAGE_HINTS[lang] in system
    assert system.index(LANGUAGE_HINTS[lang]) < system.index("TEXTO-DO-DONO")  # o pedido explícito do usuário vem depois e prevalece
    assert system.count(LANGUAGE_HINTS[lang]) == 1
    if lang == "en-US":
        assert "Idioma padrão" not in system
        assert "# Available Skills" in system


def test_without_language_defaults_to_english_prompt_and_hints_stay_concise(client):
    provider = client.app.state.chat_service.provider = Capture()
    client.post("/api/chat/stream", json={"message": "hello"})
    system = _system_text(provider)
    assert LANGUAGE_HINTS["en-US"] in system
    assert LANGUAGE_HINTS["pt-BR"] not in system
    assert all(len(h) < 350 for h in LANGUAGE_HINTS.values())


@pytest.mark.parametrize("endpoint", ["/api/chat", "/api/chat/stream"])
@pytest.mark.parametrize("lang", LANGS)
def test_language_propagates_across_sync_and_stream_endpoints(client, endpoint, lang):
    provider = client.app.state.chat_service.provider = Capture()
    resp = client.post(endpoint, json={"message": "hello" if lang == "en-US" else "olá", "language": lang})
    assert resp.status_code == 200
    system = _system_text(provider)
    assert LANGUAGE_HINTS[lang] in system
    other = "pt-BR" if lang == "en-US" else "en-US"
    assert LANGUAGE_HINTS[other] not in system


def test_real_directives_and_hello_prompt_have_no_conflicting_portuguese_in_en_us(tmp_path):
    """Confirms assembled prompt with real config/directives.md has no Portuguese directives when language=en-US."""
    from backend.main import create_app
    from fastapi.testclient import TestClient
    from tests.helpers import make_settings

    settings = make_settings(tmp_path)
    settings.directives.path = ROOT / "config" / "directives.md"
    with TestClient(create_app(settings)) as test_client:
        provider = test_client.app.state.chat_service.provider = Capture()
        resp = test_client.post("/api/chat/stream", json={"message": "hello", "language": "en-US"})
        assert resp.status_code == 200
        system = _system_text(provider)
        assert LANGUAGE_HINTS["en-US"] in system
        assert "português brasileiro" not in system.lower()
        assert "português do brasil" not in system.lower()
        assert "# Available Skills" in system
        assert "# Language" in system


def test_retry_and_tool_flow_preserve_language_choice(client):
    provider = client.app.state.chat_service.provider = Capture()
    first = client.post("/api/chat/stream", json={"message": "hello", "language": "pt-BR"})
    assert first.status_code == 200
    conv_id = client.get("/api/conversations").json()["conversations"][0]["id"]
    retry = client.post("/api/chat/stream", json={"conversation_id": conv_id, "message": "", "retry": True, "language": "pt-BR"})
    assert retry.status_code == 200
    system = _system_text(provider)
    assert LANGUAGE_HINTS["pt-BR"] in system
    assert LANGUAGE_HINTS["en-US"] not in system


def test_invalid_language_is_rejected_and_effective_prompt_logic():
    assert ChatRequest(message="x").effective_system_prompt() is None
    assert ChatRequest(message="x", system_prompt="A").effective_system_prompt() == "A"
    assert ChatRequest(message="x", language="en-US").effective_system_prompt() == LANGUAGE_HINTS["en-US"]
    assert ChatRequest(message="x", language="pt-BR", system_prompt=" B ").effective_system_prompt().endswith("\n\nB")
    with pytest.raises(ValueError):
        ChatRequest(message="x", language="fr-FR")


def test_language_does_not_touch_stored_conversation_text(client):
    client.app.state.chat_service.provider = Capture()
    first = client.post("/api/chat/stream", json={"message": "Olá, mundo", "language": "en-US"})
    assert first.status_code == 200
    conv = client.get("/api/conversations").json()["conversations"][0]
    stored = client.get(f"/api/conversations/{conv['id']}").json()["messages"]
    assert [m["content"] for m in stored if m["role"] == "user"] == ["Olá, mundo"]  # a dica nunca vira mensagem do histórico
    assert all(LANGUAGE_HINTS["en-US"] not in m["content"] for m in stored)

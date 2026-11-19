"""Gerenciamento e exclusão múltipla de conversas com preservação de aprendizado permanente."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from backend.chat.service import ChatService
from backend.core.config import NetToolsSettings, ToolsSettings
from backend.core.directives import load_directives
from backend.main import create_app
from backend.memory.manager import MemoryManager
from backend.providers.base import ModelProvider
from backend.storage.database import Database
from backend.tools.builtin.web_fetch import WebFetchTool
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry
from tests.helpers import make_settings


class SlowStreamProvider(ModelProvider):
    name = "slow"

    def __init__(self, chunks: list[str] | None = None, delay: float = 0.08):
        super().__init__("slow-1b")
        self.chunks = chunks or ["Parte 1 ", "Parte 2 ", "Parte 3 ", "Parte 4 ", "Parte 5"]
        self.delay = delay

    async def chat(self, messages, **options):
        return "MESMO"

    async def stream_chat(self, messages, **options):
        for chunk in self.chunks:
            await asyncio.sleep(self.delay)
            yield chunk

    async def health(self):
        return {"available": True}


class CaptureKnowledgeProvider(ModelProvider):
    name = "capture"

    def __init__(self, reply: str = "O Projeto Cobalto usa a porta 64120 [web#1]."):
        super().__init__("capture-1b")
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def chat(self, messages, **options):
        self.calls.append([dict(m) for m in messages])
        return self.reply

    async def stream_chat(self, messages, **options):
        self.calls.append([dict(m) for m in messages])
        yield self.reply

    async def health(self):
        return {"available": True}


def _ddg(title: str, encoded_url: str, snippet: str) -> str:
    return (
        f'<a class="result__a" href="//duckduckgo.com/l/?uddg={encoded_url}&amp;rut=1">{title}</a>'
        f'<a class="result__snippet">{snippet}</a>'
    )


def _create_junction(link_path: Path, target_path: Path) -> bool:
    res = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return res.returncode == 0


def _sqlite_fts_healthy(db_path: Path) -> None:
    db = Database(db_path)
    db.connect()
    try:
        integrity = db.read_one("PRAGMA integrity_check")
        assert integrity is not None and integrity[0] == "ok"
        fk_violations = db.read_all("PRAGMA foreign_key_check")
        assert fk_violations == []
        # Índices FTS5 íntegros e consistentes com as tabelas permanentes
        db.read_all("SELECT item_id FROM knowledge_fts")
        db.read_all("SELECT memory_id FROM memories_fts")
    finally:
        db.close()


# ------------------------------------------------------------ A, B, C, E, F, K, L, M, N


def test_delete_single_multiple_and_all_conversations_with_exclusive_files(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        c1 = client.post("/api/chat", json={"message": "Primeira conversa exclusiva."}).json()["conversation_id"]
        c2 = client.post("/api/chat", json={"message": "Segunda conversa exclusiva."}).json()["conversation_id"]
        c3 = client.post("/api/chat", json={"message": "Terceira conversa exclusiva."}).json()["conversation_id"]
        c4 = client.post("/api/chat", json={"message": "Quarta conversa para limpar tudo."}).json()["conversation_id"]

        # Gera repacks em c1 e c2
        for cid in (c1, c2):
            for idx in range(4):
                client.post("/api/chat", json={"conversation_id": cid, "message": f"Mensagem {idx} de contexto."})
            repack = client.post(f"/api/memory/conversations/{cid}/repack", json={}).json()
            assert repack["ok"] is True

        conv_dir = settings.memory.base_dir / "conversations"
        repack_dir = settings.memory.base_dir / "repacks"
        assert (conv_dir / f"{c1}.md").is_file()
        assert (conv_dir / f"{c2}.md").is_file()
        assert (repack_dir / c1 / "v001.md").is_file()
        assert (repack_dir / c2 / "v001.md").is_file()

        # A. Excluir uma conversa
        del1 = client.delete(f"/api/conversations/{c1}")
        assert del1.status_code == 200
        assert del1.json()["deleted"] is True
        assert not (conv_dir / f"{c1}.md").exists()
        assert not (repack_dir / c1).exists()
        assert client.get(f"/api/conversations/{c1}").status_code == 404

        # K & L. Excluir conversa inexistente / repetir solicitação
        missing_uuid = str(uuid.uuid4())
        assert client.delete(f"/api/conversations/{missing_uuid}").status_code == 404
        assert client.delete(f"/api/conversations/{c1}").status_code == 404

        # B. Excluir múltiplas conversas (com ID inexistente + duplicado no meio)
        del_multi = client.post(
            "/api/conversations/delete",
            json={"ids": [c2, c2, missing_uuid, c3]},
        )
        assert del_multi.status_code == 200
        body = del_multi.json()
        assert body["deleted"] is True
        assert body["deleted_count"] == 2
        assert body["deleted_ids"] == [c2, c3]
        assert body["not_found_ids"] == [missing_uuid]
        assert not (conv_dir / f"{c2}.md").exists()
        assert not (repack_dir / c2).exists()
        assert not (conv_dir / f"{c3}.md").exists()

        # Repetir exclusão múltipla não quebra e retorna contagem 0
        repeat = client.post("/api/conversations/delete", json={"ids": [c2, c3]})
        assert repeat.status_code == 200
        assert repeat.json()["deleted"] is False
        assert repeat.json()["deleted_count"] == 0
        assert repeat.json()["not_found_ids"] == [c2, c3]

        # C. Excluir todas as conversas visíveis restantes (c4)
        visible = [c["id"] for c in client.get("/api/conversations").json()["conversations"]]
        assert visible == [c4]
        del_all = client.post("/api/conversations/delete", json={"ids": visible})
        assert del_all.status_code == 200
        assert del_all.json()["deleted_count"] == 1
        assert client.get("/api/conversations").json()["conversations"] == []
        assert not (conv_dir / f"{c4}.md").exists()

    # M & N. Reabrir aplicativo e verificar que excluídos não retornam + SQLite/FTS5 íntegros
    with TestClient(create_app(settings)) as client2:
        assert client2.get("/api/conversations").json()["conversations"] == []
        assert settings.storage.database_path.is_file()
        assert (settings.storage.database_path.parent / "tool-audit.jsonl").parent.exists()
    _sqlite_fts_healthy(settings.storage.database_path)


def test_shared_files_and_non_conversation_artifacts_preserved(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        # Pré-existem registros legados no SQLite (ex.: tabela memories) e outra conversa intacta
        app.state.db.write(
            "INSERT INTO memories (kind, title, content, certainty, active, created_at, updated_at) "
            "VALUES ('fact', 'Legacy Note', 'Porta local 47123.', 'confirmed', 1, '2026-11-16T00:00:00Z', '2026-11-16T00:00:00Z')"
        )
        c_keep = client.post("/api/chat", json={"message": "Conversa que deve permanecer."}).json()["conversation_id"]
        c_del = client.post("/api/chat", json={"message": "Conversa para excluir."}).json()["conversation_id"]

        res = client.delete(f"/api/conversations/{c_del}")
        assert res.status_code == 200
        assert res.json()["deleted"] is True

        # Conversa preservada, dados legados no SQLite, diretivas e DB intactos
        assert client.get(f"/api/conversations/{c_keep}").status_code == 200
        assert len(app.state.db.read_all("SELECT id FROM memories")) == 1
        assert settings.directives.path.is_file()
        assert settings.storage.database_path.is_file()


def test_legacy_memory_rows_survive_conversation_deletion_without_cross_session_leak(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        provider = CaptureKnowledgeProvider("Recorded answer.")
        app.state.chat_service.provider = provider
        # Registro legado pré-existente no banco SQLite
        app.state.db.write(
            "INSERT INTO memories (kind, title, content, certainty, active, created_at, updated_at) "
            "VALUES ('decision', 'Legacy DB', 'Decidimos usar PostgreSQL 16 no módulo de relatórios.', 'confirmed', 1, '2026-11-16T00:00:00Z', '2026-11-16T00:00:00Z')"
        )

        r_chat = client.post("/api/chat", json={"message": "Decidimos usar Redis no cache."})
        assert r_chat.status_code == 200
        conv_id = r_chat.json()["conversation_id"]
        # Nenhum novo aprendizado automático é gravado durante conversas
        assert len(app.state.db.read_all("SELECT id FROM memories")) == 1

        del_resp = client.delete(f"/api/conversations/{conv_id}")
        assert del_resp.status_code == 200
        assert len(app.state.db.read_all("SELECT id FROM memories")) == 1

    app2 = create_app(settings)
    with TestClient(app2) as client2:
        assert client2.get("/api/conversations").json()["conversations"] == []
        probe_provider = CaptureKnowledgeProvider("Checked.")
        app2.state.chat_service.provider = probe_provider

        new_chat_resp = client2.post(
            "/api/chat",
            json={"message": "Which database did we choose for reports?"},
        )
        assert new_chat_resp.status_code == 200
        system_prompt = probe_provider.calls[-1][0]["content"]
        assert "<memoria_recuperada>" not in system_prompt
        assert "<conhecimento_aprovado>" not in system_prompt
        assert "PostgreSQL 16" not in system_prompt
    _sqlite_fts_healthy(settings.storage.database_path)


# ------------------------------------------------------------ J: Exclusão durante streaming & Concorrência


def test_delete_during_active_streaming_cancels_cleanly_without_recreating(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory, num_ctx=4096, num_predict=1024)
        provider = SlowStreamProvider(chunks=["A ", "B ", "C ", "D ", "E ", "F "], delay=0.06)
        service = ChatService(db, provider, directives, memory_manager=manager)

        conv = service.repo.create_conversation("Streaming em andamento")

        async def scenario():
            received: list[dict] = []

            async def consume():
                async for ev in service.send_stream(conv.id, "Responda devagar por favor."):
                    received.append(ev)

            stream_task = asyncio.create_task(consume())
            # Aguarda o stream iniciar e emitir pelo menos o primeiro token
            for _ in range(40):
                if any(e.get("type") == "token" for e in received):
                    break
                await asyncio.sleep(0.02)
            assert any(e.get("type") == "token" for e in received)

            result = await service.delete_conversation(conv.id)
            assert result["deleted"] is True
            await asyncio.wait_for(stream_task, timeout=2.0)

        asyncio.run(scenario())

        # Conversa e arquivo Markdown NÃO foram recriados após o cancelamento
        assert service.repo.get_conversation(conv.id) is None
        assert service.repo.get_messages(conv.id) == []
        assert not manager.markdown.conversation_path(conv.id).exists()
    finally:
        db.close()


def test_concurrent_delete_requests_are_safe_and_consistent(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory)
        service = ChatService(db, SlowStreamProvider(), directives, memory_manager=manager)

        c1 = service.repo.create_conversation("Concorrente 1").id
        c2 = service.repo.create_conversation("Concorrente 2").id
        service.repo.add_message(c1, "user", "oi 1")
        service.repo.add_message(c2, "user", "oi 2")
        manager.sync_conversation_markdown(c1)
        manager.sync_conversation_markdown(c2)

        async def scenario():
            r1, r2 = await asyncio.gather(
                service.delete_conversations([c1, c2]),
                service.delete_conversations([c2, c1]),
            )
            return r1, r2

        r1, r2 = asyncio.run(scenario())
        assert r1["deleted_count"] + r2["deleted_count"] == 2
        assert service.repo.list_conversations() == []
        assert not manager.markdown.conversation_path(c1).exists()
        assert not manager.markdown.conversation_path(c2).exists()
    finally:
        db.close()


# ------------------------------------------------------------ Recuperação de falhas e Segurança


def test_intermediate_failure_rollback_and_stale_file_recovery(tmp_path, monkeypatch):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory)
        service = ChatService(db, SlowStreamProvider(), directives, memory_manager=manager)

        conv = service.repo.create_conversation("Rollback")
        service.repo.add_message(conv.id, "user", "Mensagem preservada no rollback.")
        manager.sync_conversation_markdown(conv.id)
        md_path = manager.markdown.conversation_path(conv.id)
        assert md_path.is_file()

        # Simula falha no banco após o staging dos arquivos: deve restaurar o arquivo e o registro
        def broken_delete(_ids):
            raise RuntimeError("falha simulada no commit")

        monkeypatch.setattr(service.repo, "delete_conversations", broken_delete)
        with pytest.raises(RuntimeError, match="falha simulada"):
            asyncio.run(service.delete_conversation(conv.id))

        assert service.repo.get_conversation(conv.id) is not None
        assert md_path.is_file()
        assert "Mensagem preservada no rollback." in md_path.read_text(encoding="utf-8")

        # Simula resquício `.deleting` após commit bem-sucedido: reconciliação limpa automaticamente
        stale = manager.markdown.conversations_dir / f".{conv.id}.md.abcd1234.deleting"
        stale.write_text("resquicio", encoding="utf-8")
        assert stale.is_file()
        cleaned = manager.markdown.cleanup_stale_deletions()
        assert cleaned >= 1
        assert not stale.exists()
    finally:
        db.close()


def test_security_invalid_ids_csrf_and_junction_blocked(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        # CSRF / Origin / Sec-Fetch-Site bloqueados
        c1 = client.post("/api/chat", json={"message": "Protegida"}).json()["conversation_id"]
        assert client.post(
            "/api/conversations/delete",
            json={"ids": [c1]},
            headers={"Origin": "https://evil.example"},
        ).status_code == 403
        assert client.delete(
            f"/api/conversations/{c1}",
            headers={"Sec-Fetch-Site": "cross-site"},
        ).status_code == 403
        assert client.get(f"/api/conversations/{c1}").status_code == 200

        # IDs inválidos e path traversal recusados sem apagar nada
        for bad_id in ("", "not-a-uuid", "../enslaved.db", "a/b", "id com espaco"):
            resp = client.post("/api/conversations/delete", json={"ids": [c1, bad_id]})
            assert resp.status_code == 400
        assert client.delete("/api/conversations/not-a-uuid").status_code == 400
        assert client.post("/api/conversations/delete", json={"ids": []}).status_code == 422
        assert client.post("/api/conversations/delete", json={"ids": [c1], "extra": True}).status_code == 422
        assert client.get(f"/api/conversations/{c1}").status_code == 200

        # Junction plantada dentro de repacks/<id> apontando para pasta externa NÃO pode apagar o alvo externo
        external_dir = tmp_path / "external_protected"
        external_dir.mkdir(parents=True, exist_ok=True)
        external_file = external_dir / "protected.md"
        external_file.write_text("# Arquivo protegido\n", encoding="utf-8")
        repack_conv_dir = settings.memory.base_dir / "repacks" / c1
        repack_conv_dir.parent.mkdir(parents=True, exist_ok=True)
        if _create_junction(repack_conv_dir, external_dir):
            resp_j = client.delete(f"/api/conversations/{c1}")
            assert resp_j.status_code == 403
            assert external_file.is_file()
            os.rmdir(repack_conv_dir)


def test_conversation_deletion_cancels_pending_permissions_and_emits_no_learning_events(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        app.state.chat_service.provider = CaptureKnowledgeProvider("Reply stored.")
        events_raw = client.post(
            "/api/chat/stream",
            json={"message": "Temporary conversation"},
        ).text
        assert '"learning_suggestions"' not in events_raw
        conv_id = next(
            json.loads(line[5:].strip())["conversation_id"]
            for line in events_raw.splitlines()
            if line.startswith("data:") and '"conversation_id"' in line
        )

        # Cria pedido de permissão pendente preso a essa conversa
        async def create_req():
            prep = await app.state.tool_engine.prepare("fs.read", {"path": "docs/x.md"}, source="chat", conversation_id=conv_id)
            app.state.tool_engine.remember_invocation(prep, source="chat", conversation_id=conv_id)
            return prep.request.id

        req_id = asyncio.run(create_req())
        assert app.state.permissions.is_pending(req_id)

        assert client.delete(f"/api/conversations/{conv_id}").status_code == 200
        # Pedido de permissão pendente foi cancelado (não pode ser aprovado depois)
        assert not app.state.permissions.is_pending(req_id)
        assert client.post(f"/api/permissions/{req_id}/decision", json={"granted": True}).status_code == 404


def test_ai_model_cannot_trigger_conversation_deletion(tmp_path):
    settings = make_settings(tmp_path)
    app = create_app(settings)
    with TestClient(app) as client:
        tool_ids = {t.spec.id for t in app.state.tool_engine.list_tools()}
        assert not any("delete" in tid or "conversation" in tid for tid in tool_ids)
        # net.fetch nunca alcança a própria API local para acionar exclusão
        res = asyncio.run(
            app.state.chat_service.fetch_tool.run(
                {"url": "http://127.0.0.1:8000/api/conversations/delete"},
                permission=type("Perm", (), {"granted": True})(),
            )
        )
        assert res.ok is False


# ------------------------------------------------------------ O: UI Desktop e Mobile (Playwright)


@pytest.mark.parametrize("viewport", [(1280, 800), (375, 812)], ids=["desktop", "mobile"])
def test_ui_conversation_management_desktop_and_mobile(tmp_path, monkeypatch, viewport):
    sync_api = pytest.importorskip("playwright.sync_api")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, type=None: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))],
    )

    settings = make_settings(tmp_path)
    app = create_app(settings)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)

    try:
        app.state.chat_service.provider = CaptureKnowledgeProvider("Conversation recorded.")

        with sync_api.sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(channel="chrome")
            except Exception:
                pytest.skip("Chrome indisponível para Playwright")
            page = browser.new_page(viewport={"width": viewport[0], "height": viewport[1]})
            try:
                page.goto(f"http://127.0.0.1:{port}")
                page.wait_for_selector("#input", timeout=10000)

                # Conversas 1, 2 e 3 na instalação padrão em inglês (en-US)
                for idx, prompt_text in enumerate(("First conversation test", "Second conversation test", "Third conversation test"), start=1):
                    if idx > 1:
                        if viewport[0] < 820:
                            page.locator("#toggle-sidebar").click()
                            page.wait_for_function("() => document.getElementById('sidebar').getBoundingClientRect().left >= 0", timeout=5000)
                        page.click("#new-chat")
                    page.fill("#input", prompt_text)
                    page.keyboard.press("Enter")
                    page.wait_for_function(
                        "() => !document.querySelector('.content.streaming') && !document.getElementById('composer').classList.contains('busy')",
                        timeout=15000,
                    )
                    page.wait_for_function(
                        f"() => document.querySelectorAll('#conversation-list .conversation-item').length === {idx}",
                        timeout=10000,
                    )

                if viewport[0] < 820:
                    page.locator("#toggle-sidebar").click()
                    page.wait_for_function("() => document.getElementById('sidebar').getBoundingClientRect().left >= 0", timeout=5000)
                page.wait_for_function("() => document.querySelectorAll('#conversation-list .conversation-item').length === 3", timeout=10000)
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")

                # Modo "Select conversations"
                page.click("#toggle-select-conversations")
                page.wait_for_selector("#conv-selection-bar:not([hidden])", timeout=5000)
                assert page.locator("#conv-selected-count").inner_text() == "0 selected"
                assert page.locator("#delete-selected-conversations").is_disabled()

                # Clicar em uma conversa no modo seleção marca sem abrir/alterar
                current_title_before = page.locator("#chat-title").inner_text()
                page.locator("#conversation-list .conversation-item .title").nth(1).click()
                assert page.locator("#conv-selected-count").inner_text() == "1 selected"
                assert page.locator("#chat-title").inner_text() == current_title_before

                # Select all e Clear
                page.click("#select-all-conversations")
                assert page.locator("#conv-selected-count").inner_text() == "3 selected"
                page.click("#clear-selected-conversations")
                assert page.locator("#conv-selected-count").inner_text() == "0 selected"

                # Seleciona 2 conversas e abre confirmação -> Cancel mantém tudo
                page.locator("#conversation-list .conversation-item .conv-check").nth(0).click()
                page.locator("#conversation-list .conversation-item .conv-check").nth(2).click()
                assert page.locator("#conv-selected-count").inner_text() == "2 selected"
                page.click("#delete-selected-conversations")
                page.wait_for_selector("#dialog-modal:not(.hidden)", timeout=5000)
                dialog_text = page.locator("#dialog-message").inner_text()
                assert "You are about to delete 2 conversations and its associated files." in dialog_text
                assert "Messages and histories will be removed." in dialog_text
                page.locator("#dialog-modal button", has_text="Cancel").click()
                page.locator("#dialog-modal").wait_for(state="hidden", timeout=5000)
                assert page.locator("#conversation-list .conversation-item").count() == 3

                # Confirmar exclusão das 2 conversas
                page.click("#delete-selected-conversations")
                page.wait_for_selector("#dialog-modal:not(.hidden)", timeout=5000)
                page.click("#dialog-ok")
                page.wait_for_function("() => document.querySelectorAll('#conversation-list .conversation-item').length === 1", timeout=10000)
                assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1")

                # Recarregar página confirma persistência + exclusão individual da última conversa
                page.reload()
                if viewport[0] < 820:
                    page.locator("#toggle-sidebar").click()
                    page.wait_for_function("() => document.getElementById('sidebar').getBoundingClientRect().left >= 0", timeout=5000)
                page.wait_for_selector("#conversation-list .conversation-item", timeout=10000)
                assert page.locator("#conversation-list .conversation-item").count() == 1
                page.locator("#conversation-list .conversation-item").first.hover()
                page.locator("#conversation-list .conversation-item .delete").first.click()
                page.wait_for_selector("#dialog-modal:not(.hidden)", timeout=5000)
                assert "1 conversation" in page.locator("#dialog-message").inner_text()
                page.click("#dialog-ok")
                page.wait_for_function("() => document.querySelectorAll('#conversation-list .conversation-item').length === 0", timeout=10000)
                assert page.locator("#open-learning").count() == 0
            finally:
                page.close()
                browser.close()
    finally:
        srv.should_exit = True
        thread.join(timeout=5)

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from backend.chat.service import ChatService
from backend.core.config import MemorySettings, ToolsSettings
from backend.core.directives import load_directives
from backend.main import create_app
from backend.memory.manager import MemoryManager
from backend.providers.base import ChatTurn, ModelProvider, ToolCall
from backend.providers.simulated import SimulatedProvider
from backend.storage.database import Database
from backend.tools.base import Tool, ToolResult, ToolSpec
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry
from tests.helpers import DIRECTIVES_TEXT, make_settings


class ScriptedToolProvider(ModelProvider):
    name = "scripted"

    def __init__(self, turns: list[ChatTurn]):
        super().__init__("qwen3.5:4b")
        self.turns = list(turns)
        self.calls: list[dict] = []

    async def chat(self, messages, **options):
        self.calls.append({"messages": [dict(m) for m in messages]})
        return self.turns.pop(0).content

    async def stream_chat(self, messages, **options):
        self.calls.append({"messages": [dict(m) for m in messages], "stream": True})
        text = self.turns.pop(0).content
        yield text

    async def chat_turn(self, messages, tools=None, **options):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        self.last_usage = {"prompt_eval_count": 320, "eval_count": 48}
        return self.turns.pop(0)

    async def health(self):
        return {"available": True, "models": [self.model]}


class PingArgs(BaseModel):
    value: str


class PingTool(Tool):
    args_model = PingArgs

    def __init__(self):
        self.spec = ToolSpec(
            id="ping",
            name="Ping",
            description="Retorna o valor enviado.",
            required_permissions=["demo.ping"],
        )

    async def run(self, args, permission=None) -> ToolResult:
        if not permission or not permission.granted:
            return ToolResult(ok=False, error="negado")
        return ToolResult(ok=True, output=f"pong:{args['value']}")


def test_conversation_saved_completely_in_markdown_and_reconciled(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory, num_ctx=4096, num_predict=1024)
        service = ChatService(db, SimulatedProvider(), directives, memory_manager=manager)

        res1 = asyncio.run(service.send(None, "Primeira pergunta importante."))
        conv_id = res1["conversation"].id
        res2 = asyncio.run(service.send(conv_id, "Segunda pergunta."))

        md_path = manager.markdown.conversation_path(conv_id)
        assert md_path.is_file()
        md_text = md_path.read_text(encoding="utf-8")
        assert conv_id in md_text
        assert "Primeira pergunta importante." in md_text
        assert "Segunda pergunta." in md_text
        assert md_text.count("## [1] Usuário") == 1
        assert md_text.count("## [4] Assistente (Enslaved)") == 1
        assert "Criada em:" in md_text
        assert "Atualizada em:" in md_text

        # Simula falha de exportação posterior e valida reconciliação segura a partir do SQLite
        original_write = manager.markdown.render_conversation_markdown

        def broken_render(*args, **kwargs):
            raise OSError("disco temporariamente indisponível")

        manager.markdown.render_conversation_markdown = broken_render
        failed_res = manager.sync_conversation_markdown(conv_id)
        assert failed_res.ok is False
        status = manager.markdown.get_export_status(conv_id)
        assert status["status"] == "error"
        assert "indisponível" in status["last_error"]
        # Conteúdo anterior do .md não foi corrompido
        assert "Segunda pergunta." in md_path.read_text(encoding="utf-8")

        # Reconciliação recupera automaticamente
        manager.markdown.render_conversation_markdown = original_write
        reconcile = manager.reconcile_exports()
        assert reconcile["reconciled"] == 1
        assert reconcile["failed"] == 0
        assert manager.markdown.get_export_status(conv_id)["status"] == "ok"
    finally:
        db.close()


def test_history_and_memory_persisted_across_app_restart(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client1:
        r = client1.post(
            "/api/chat",
            json={"message": "Decidimos usar SQLite FTS5 para busca local e prefiro respostas curtas."},
        )
        assert r.status_code == 200
        conv_id = r.json()["conversation_id"]

    # Novo ciclo completo simulando reinício da aplicação
    with TestClient(create_app(settings)) as client2:
        conv = client2.get(f"/api/conversations/{conv_id}")
        assert conv.status_code == 200
        assert len(conv.json()["messages"]) == 2

        md = client2.get(f"/api/memory/conversations/{conv_id}/markdown")
        assert md.status_code == 200
        assert "Decidimos usar SQLite FTS5" in md.json()["content"]

        # Nenhum aprendizado automático persistente é gravado nem exposto em /api/memory/items
        assert client2.get("/api/memory/items").status_code == 404
        assert client2.app.state.db.read_all("SELECT id FROM memories") == []


def test_repack_preserves_origin_distinguishes_certainty_and_chunks_sequentially(tmp_path):
    settings = make_settings(
        tmp_path,
        ollama={"num_ctx": 1024, "num_predict": 256},
        memory={
            "base_dir": tmp_path / "data" / "memory",
            "compact_threshold_ratio": 0.75,
            "recent_messages_to_keep": 2,
            "max_summary_tokens": 300,
        },
    )
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory, num_ctx=1024, num_predict=256)
        provider = SimulatedProvider()
        service = ChatService(db, provider, directives, memory_manager=manager)

        # Popula mensagens suficientes para ultrapassar limiar de 75% de 768 tokens (~576 tokens)
        conv = service.repo.create_conversation("Sessão longa")
        m1 = service.repo.add_message(conv.id, "user", "Decidimos adotar FastAPI na porta 8000.")
        m2 = service.repo.add_message(conv.id, "assistant", "Registrado: FastAPI na porta 8000.")
        m3 = service.repo.add_message(conv.id, "user", "Talvez possamos testar cache Redis no futuro, mas ainda é hipótese.")
        m4 = service.repo.add_message(conv.id, "assistant", "Anotado como hipótese pendente de validação.")
        # Bloco longo para forçar compactação em blocos sequenciais
        big_text = "Contexto técnico detalhado sobre módulo de armazenamento local. " * 45
        m5 = service.repo.add_message(conv.id, "user", big_text)
        m6 = service.repo.add_message(conv.id, "assistant", big_text)
        manager.sync_conversation_markdown(conv.id)

        orig_md_path = manager.markdown.conversation_path(conv.id)
        orig_md_before = orig_md_path.read_text(encoding="utf-8")

        # Nova mensagem aciona prepare_context -> repack automático
        res = asyncio.run(service.send(conv.id, "Qual é a porta configurada?"))
        assert res["context"]["repack_triggered"] is True
        assert res["context"]["estimated_tokens"] <= 768

        # Histórico original em SQLite e Markdown permanece 100% preservado
        all_msgs = service.repo.get_messages(conv.id)
        assert [m.id for m in all_msgs[:6]] == [m1.id, m2.id, m3.id, m4.id, m5.id, m6.id]
        orig_md_after = orig_md_path.read_text(encoding="utf-8")
        assert big_text in orig_md_after
        assert "Decidimos adotar FastAPI na porta 8000." in orig_md_after
        assert len(orig_md_after) >= len(orig_md_before)

        # Verifica repack versionado em Markdown + rastreabilidade + certeza
        repacks = manager.repo.list_repacks(conv.id)
        assert len(repacks) == 1
        rp = repacks[0]
        assert rp.version == 1
        assert m1.id in rp.source_message_ids
        assert any("FastAPI" in d for d in rp.decisions)
        assert any("hipótese" in h.lower() or "talvez" in h.lower() for h in rp.hypotheses)
        assert not any("Redis" in f for f in rp.confirmed_facts)

        rp_md = Path(rp.markdown_path).read_text(encoding="utf-8")
        assert f"msg#{m1.id}" in rp_md
        assert "## Fatos confirmados" in rp_md
        assert "## Hipóteses e informações incertas" in rp_md

        # Payload enviado ao modelo manteve as mensagens recentes literalmente e ficou abaixo do orçamento
        last_payload = provider.calls[-1]
        assert any("Qual é a porta configurada?" in m["content"] for m in last_payload)
        assert any("Resumo de contexto anterior" in m["content"] for m in last_payload if m["role"] == "system")
        assert big_text not in json.dumps(last_payload, ensure_ascii=False)
    finally:
        db.close()


def test_conversations_do_not_auto_learn_or_leak_memories_across_sessions(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory, num_ctx=4096, num_predict=1024)
        provider = SimulatedProvider()
        service = ChatService(db, provider, directives, memory_manager=manager)

        # Sessão 1 menciona uma decisão explícita
        asyncio.run(
            service.send(
                None,
                "Decidimos usar PostgreSQL na produção para o módulo Enslaved.",
            )
        )
        assert db.read_all("SELECT id FROM memories") == []

        # Sessão 2 (nova conversa) não recebe memória persistente nem vaza dados da Sessão 1
        res2 = asyncio.run(service.send(None, "Qual banco decidimos usar na produção do Enslaved?"))
        assert "retrieved_memory_ids" not in res2["context"]
        system_msg = provider.calls[-1][0]["content"]
        assert "Reply directly" in system_msg
        assert "<memoria_recuperada>" not in system_msg
        assert "PostgreSQL" not in system_msg
    finally:
        db.close()


def test_context_budget_threshold_and_ollama_calibration(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=2048, num_predict=512)
        # input_budget = 1536, threshold (75%) = 1152
        assert manager.context.input_budget == 1536
        assert manager.context.threshold_tokens == 1152

        # Simula calibração real vinda do Ollama (prompt_eval_count). O piso do fator é 1.0
        # (estimativa nunca otimista), então o ajuste é exercitado na direção de subestimação.
        class CalibratedProvider(SimulatedProvider):
            async def chat(self, messages, **options):
                self.last_usage = {"prompt_eval_count": 4000, "eval_count": 40}
                return await super().chat(messages, **options)

        directives = load_directives(settings.directives.path)
        service = ChatService(db, CalibratedProvider(), directives, memory_manager=manager)
        res = asyncio.run(service.send(None, "Oi, testando calibração do Ollama."))
        state = manager.repo.get_context_state(res["conversation"].id)
        assert state is not None
        assert state.prompt_eval_count == 4000
        assert state.calibration_factor > 1.0
    finally:
        db.close()


def test_controlled_repack_failure_preserves_originals_and_recovery_path(tmp_path):
    settings = make_settings(tmp_path)
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=4096, num_predict=1024)
        conv = manager.conv_repo.create_conversation("Teste falha controlada")
        for i in range(6):
            manager.conv_repo.add_message(conv.id, "user" if i % 2 == 0 else "assistant", f"Mensagem segura {i}")
        manager.sync_conversation_markdown(conv.id)
        orig_path = manager.markdown.conversation_path(conv.id)
        orig_snapshot = orig_path.read_text(encoding="utf-8")

        async def failing_summarizer(_prompt: str) -> str:
            raise RuntimeError("falha simulada no sumarizador")

        messages = manager.conv_repo.get_messages(conv.id)
        failed = asyncio.run(
            manager.repack_engine.repack_conversation(
                conv.id,
                messages,
                allow_fallback=False,
                force=True,
                summarizer=failing_summarizer,
            )
        )
        assert failed.ok is False
        assert "falha simulada" in (failed.error or "")
        # Nenhum repack corrompido gravado e arquivo original intacto
        assert manager.repo.list_repacks(conv.id) == []
        assert orig_path.read_text(encoding="utf-8") == orig_snapshot

        # Caminho de recuperação seguro com fallback determinístico funciona depois
        recovered = asyncio.run(manager.manual_repack(conv.id))
        assert recovered.ok is True
        assert recovered.repack is not None
        assert orig_path.read_text(encoding="utf-8").startswith("# Conversa: Teste falha controlada")
    finally:
        db.close()


def test_memory_manual_repack_and_status_endpoints(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        r1 = client.post("/api/chat", json={"message": "Decidimos usar Qwen 3.5 4B no Ollama local."})
        conv_id = r1.json()["conversation_id"]
        client.post("/api/chat", json={"conversation_id": conv_id, "message": "Pendência: validar backup Markdown."})
        client.post("/api/chat", json={"conversation_id": conv_id, "message": "Qual o status atual?"})

        status = client.get(f"/api/memory/status?conversation_id={conv_id}").json()
        assert status["enabled"] is True
        assert status["fts5_available"] is True
        assert status["export"]["status"] == "ok"

        repack = client.post(f"/api/memory/conversations/{conv_id}/repack", json={}).json()
        assert repack["ok"] is True
        assert repack["repack"]["version"] == 1
        repacks = client.get(f"/api/memory/repacks?conversation_id={conv_id}").json()["repacks"]
        assert len(repacks) == 1

        index_html = client.get("/").text
        assert "open-memory" in index_html
        assert "memory-modal" in index_html


def test_chat_tools_and_memory_api_continuity(tmp_path):
    directives_path = tmp_path / "directives.md"
    directives_path.write_text(DIRECTIVES_TEXT, encoding="utf-8")
    directives = load_directives(directives_path)
    db = Database(tmp_path / "data" / "tools_mem.db")
    db.connect()
    try:
        registry = ToolRegistry()
        registry.register(PingTool())
        t_settings = ToolsSettings(permission_modes={"demo.ping": "allow"})
        policy = ToolPolicy(t_settings)
        permissions = PermissionManager(policy)
        engine = ToolEngine(registry, policy, permissions, AuditLog(tmp_path / "audit.jsonl"), t_settings)
        provider = ScriptedToolProvider(
            [
                ChatTurn(content="", tool_calls=[ToolCall(name="ping", arguments='{"value": "ok"}', id="c1")]),
                ChatTurn(content="Ferramenta executada e decidimos manter ping ativo."),
            ]
        )
        manager = MemoryManager(db, MemorySettings(base_dir=tmp_path / "data" / "memory"))
        service = ChatService(db, provider, directives, tool_engine=engine, memory_manager=manager)

        async def run():
            return [e async for e in service.send_stream(None, "Execute ping ok", use_tools=True)]

        events = asyncio.run(run())
        assert events[0]["type"] == "meta"
        assert "context" in events[0]
        assert any(e["type"] == "tool_result" and e["ok"] for e in events)
        assert events[-1]["type"] == "done"

        conv_id = events[0]["conversation_id"]
        md_text, _ = manager.markdown.read_conversation_markdown(conv_id, ensure_sync=False)
        assert "Chamadas solicitadas:" in md_text
        assert "Ferramenta:" in md_text
        assert "pong:ok" in md_text
        assert "Ferramenta executada e decidimos manter ping ativo." in md_text
    finally:
        db.close()


def test_expanded_context_16384_avoids_premature_repack_and_preserves_shift_isolation(tmp_path):
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        status = client.get("/api/memory/status").json()
        assert status["num_ctx"] == 16384
        assert status["output_reserve"] == 2048
        assert status["input_budget"] == 14336
        assert status["threshold_tokens"] == 10752

    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory)
        provider = SimulatedProvider()
        service = ChatService(db, provider, directives, memory_manager=manager)

        conv = service.repo.create_conversation("Contexto expandido")
        # ~5.000 tokens (> antigo orçamento de 3072, mas abaixo de 11520 -> sem repack prematuro)
        block = "Nota técnica sobre Guarani de Bagé em arquitetura local. " * 200  # ~11.600 chars (~3.866 tokens)
        service.repo.add_message(conv.id, "user", block)
        service.repo.add_message(conv.id, "assistant", "Registrado sobre Guarani de Bagé.")
        res = asyncio.run(service.send(conv.id, "E ele joga em que cidade?"))
        assert res["context"]["input_budget"] == 14336
        assert res["context"]["estimated_tokens"] > 3072
        assert res["context"]["repack_triggered"] is False
        assert any("Guarani de Bagé" in m["content"] for m in provider.calls[-1])

        # Mudança de assunto continua isolando o assunto anterior mesmo com janela expandida
        asyncio.run(service.send(conv.id, "Você conhece o Jaspion?"))
        sent = provider.calls[-1]
        assert "Guarani" not in " ".join(m["content"] for m in sent)
        assert len(service.repo.get_messages(conv.id)) == 6
    finally:
        db.close()

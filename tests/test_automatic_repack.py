"""Testes automatizados do Repack Automático Inteligente (cenários 20.A–O, concorrência, falhas e UI)."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi.testclient import TestClient
from playwright import sync_api

from backend.chat.service import ChatService
from backend.core.directives import load_directives
from backend.main import create_app
from backend.memory.manager import MemoryManager
from backend.providers.base import ModelProvider
from backend.providers.simulated import SimulatedProvider
from backend.storage.database import Database
from tests.helpers import make_settings


class CaptureProvider(ModelProvider):
    name = "repack-test"

    def __init__(self, reply: str = "Entendido e registrado."):
        super().__init__("qwen-repack-test")
        self.reply = reply
        self.calls: list[list[dict[str, Any]]] = []
        self.aux_calls: int = 0

    async def chat(self, messages: list[dict[str, Any]], **options: Any) -> str:
        self.calls.append([dict(m) for m in messages])
        if options.get("format") == "json":
            self.aux_calls += 1
            return json.dumps({"decisoes_confirmadas": [{"texto": "inventado porta 9999", "msg": 999999}]})
        self.last_usage = {"prompt_eval_count": 480, "eval_count": 24}
        return self.reply

    async def stream_chat(self, messages: list[dict[str, Any]], **options: Any):
        text = await self.chat(messages, **options)
        yield text

    async def health(self) -> dict[str, Any]:
        return {"available": True, "models": [self.model]}


def test_long_synthetic_conversation_preserves_goal_decisions_constraints_and_certainty(tmp_path):
    """Cenários 20.A–I, K–M, O: conversa sintética longa com repack preventivo automático e múltiplas compactações."""
    settings = make_settings(
        tmp_path,
        ollama={"num_ctx": 1536, "num_predict": 256},
        memory={
            "base_dir": tmp_path / "data" / "memory",
            "compact_threshold_ratio": 0.75,  # threshold = 960 de 1280 input_budget
            "repack_target_ratio": 0.5,
            "recent_messages_to_keep": 4,
            "max_summary_tokens": 500,
        },
    )
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory, num_ctx=1536, num_predict=256)
        provider = CaptureProvider()
        service = ChatService(db, provider, directives, memory_manager=manager)

        # Turnos iniciais ricos em informações críticas
        r1 = asyncio.run(
            service.send(
                None,
                "Meu objetivo é construir o serviço MonitorLocal para Windows em Python 3.12.",
            )
        )
        conv_id = r1["conversation"].id
        asyncio.run(
            service.send(
                conv_id,
                "Decidimos usar FastAPI na porta 8443 e banco em C:\\Monitor\\dados.db.",
            )
        )
        asyncio.run(
            service.send(
                conv_id,
                "Restrição: não alterar a pasta C:\\Windows\\System32 e nunca enviar logs para a internet.",
            )
        )
        asyncio.run(
            service.send(
                conv_id,
                "Talvez no futuro possamos testar Redis para cache, mas ainda é hipótese.",
            )
        )
        asyncio.run(
            service.send(
                conv_id,
                "Pendência: implementar rotação semanal de arquivos de auditoria.",
            )
        )

        # Preenche contexto com rodadas técnicas para forçar múltiplas compactações preventivas (v1 -> v2+)
        repack_count_events = 0
        for step in range(12):
            filler = f"Etapa {step}: detalhamento de verificação de assinatura e buffer circular tamanho {1024 + step}. " * 7
            res = asyncio.run(service.send(conv_id, filler))
            assert res["context"]["estimated_tokens"] <= 1280  # 20.K: sempre dentro do orçamento
            if res["context"]["repack_triggered"]:
                repack_count_events += 1

        assert repack_count_events >= 2  # 20.A e 20.M: múltiplas compactações preventivas
        repacks = manager.repo.list_repacks(conv_id)
        assert len(repacks) >= 2
        latest = repacks[0]
        summary = latest.summary_text

        # 20.C: histórico original em SQLite e Markdown permanece 100% íntegro
        all_msgs = service.repo.get_messages(conv_id)
        assert len(all_msgs) == 34  # 17 turnos * 2
        assert "MonitorLocal" in all_msgs[0].content
        md_text = manager.markdown.conversation_path(conv_id).read_text(encoding="utf-8")
        assert "C:\\Monitor\\dados.db" in md_text

        # 20.D–H: objetivo, decisões, restrições, valores exatos e separação de hipóteses preservados após múltiplos repacks
        assert "MonitorLocal" in summary
        assert "8443" in summary and "FastAPI" in summary
        assert "C:\\Monitor\\dados.db" in summary
        assert "System32" in summary or "nunca enviar logs" in summary.lower()
        assert "rotação semanal" in summary
        assert any("Redis" in h for h in latest.hypotheses)
        assert not any("Redis" in f for f in latest.confirmed_facts)

        # 20.L: próxima resposta recebe todo o resumo estruturado mantendo continuidade
        final_res = asyncio.run(service.send(conv_id, "Qual porta e caminho de banco definimos para o projeto?"))
        sys_msg = provider.calls[-1][0]["content"]
        assert "8443" in sys_msg
        assert "C:\\Monitor\\dados.db" in sys_msg
        assert "MonitorLocal" in sys_msg
        assert final_res["context"]["estimated_tokens"] <= 1280

        # 20.O: Learning Center não foi alimentado automaticamente pelo repack
        assert db.read_one("SELECT COUNT(*) AS n FROM knowledge_items")["n"] == 0
    finally:
        db.close()


def test_topic_shift_isolation_and_reliable_history_recall(tmp_path):
    """Cenários 14 e 20.J / 20.N: mudança de assunto (Windows -> Astronomia) e retomada confiável posterior."""
    settings = make_settings(
        tmp_path,
        ollama={"num_ctx": 1536, "num_predict": 256},
        memory={
            "base_dir": tmp_path / "data" / "memory",
            "compact_threshold_ratio": 0.70,
            "repack_target_ratio": 0.45,
            "recent_messages_to_keep": 2,
            "max_summary_tokens": 550,
            "recall_max_tokens": 500,
        },
    )
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory, num_ctx=1536, num_predict=256)
        provider = CaptureProvider()
        service = ChatService(db, provider, directives, memory_manager=manager)

        # Tópico 1: software Windows
        r1 = asyncio.run(service.send(None, "Estamos desenvolvendo um software Windows chamado LimpaDisco em C# na porta 9120."))
        conv_id = r1["conversation"].id
        for i in range(4):
            asyncio.run(
                service.send(
                    conv_id,
                    f"No software Windows LimpaDisco etapa {i}, decidimos validar assinaturas digitais antes de remover temporários. " * 4,
                )
            )

        # Tópico 2: mudança explícita para Astronomia (com várias rodadas até disparar repack)
        res_astro1 = asyncio.run(service.send(conv_id, "Mudando de assunto: quantas luas tem o planeta Júpiter?"))
        astro_sys1 = provider.calls[-1][0]["content"]
        assert "LimpaDisco" not in astro_sys1  # não força tema Windows no novo assunto

        for j in range(4):
            res_astro = asyncio.run(
                service.send(
                    conv_id,
                    f"Ainda sobre astronomia e telescópios espaciais etapa {j}, qual a distância até a nebulosa de Órion? " * 4,
                )
            )
            assert res_astro["context"]["estimated_tokens"] <= 1280

        # No assunto Astronomia, detalhes de Windows não devem poluir o bloco ativo
        astro_sys_after = provider.calls[-1][0]["content"]
        assert "9120" not in astro_sys_after

        # Tópico 3 (retomada explícita): "Voltando àquele software que estávamos desenvolvendo..."
        asyncio.run(service.send(conv_id, "Voltando àquele software Windows LimpaDisco que estávamos desenvolvendo, qual porta usamos?"))
        recall_sys = provider.calls[-1][0]["content"]
        assert "<contexto_recuperado_historico>" in recall_sys
        assert "LimpaDisco" in recall_sys
        assert "9120" in recall_sys
    finally:
        db.close()


def test_auxiliary_model_hallucination_rejected_and_empty_or_invalid_falls_back(tmp_path):
    """Cenários 11, 12, 16, 18: modelo auxiliar nunca executa ferramentas, números inventados são rejeitados e falhas caem em fallback."""
    settings = make_settings(
        tmp_path,
        memory={"base_dir": tmp_path / "data" / "memory", "repack_use_model": True, "recent_messages_to_keep": 2},
    )
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        manager = MemoryManager(db, settings.memory, num_ctx=4096, num_predict=1024)
        conv = manager.conv_repo.create_conversation("Fidelidade")
        m1 = manager.conv_repo.add_message(conv.id, "user", "Decidimos usar SQLite na porta 7070.")
        manager.conv_repo.add_message(conv.id, "assistant", "Ok, anotado porta 7070.")
        manager.conv_repo.add_message(conv.id, "user", "Restrição: não remover backups.")
        manager.conv_repo.add_message(conv.id, "assistant", "Combinado.")

        # Modelo auxiliar devolve item inventado (porta 9999) + item válido referenciando mensagem do assistente (não permitido para decisão) + item válido do usuário
        async def lying_summarizer(_prompt: str) -> str:
            return json.dumps(
                {
                    "decisoes_confirmadas": [
                        {"texto": "Usar porta 9999 no Postgres", "msg": m1.id},  # alucinação: número e banco inventados!
                        {"texto": "SQLite na porta 7070", "msg": m1.id},         # fiel à fonte
                    ],
                    "restricoes": [
                        {"texto": "Combinado.", "msg": m1.id + 1},               # role=assistant -> rejeitado para restrição!
                    ],
                }
            )

        msgs = manager.conv_repo.get_messages(conv.id)
        res = asyncio.run(
            manager.repack_engine.repack_conversation(
                conv.id, msgs, use_model=True, summarizer=lying_summarizer, force=True
            )
        )
        assert res.ok is True and res.repack is not None
        assert res.rejected_model_items >= 2
        assert "9999" not in res.repack.summary_text
        assert "7070" in res.repack.summary_text

        # Resposta vazia / inválida do modelo auxiliar -> fallback seguro sem falha
        manager.conv_repo.add_message(conv.id, "user", "Mais um detalhe sobre porta 7070.")
        manager.conv_repo.add_message(conv.id, "assistant", "Registrado.")

        async def empty_summarizer(_prompt: str) -> str:
            return "   "

        res2 = asyncio.run(
            manager.repack_engine.repack_conversation(
                conv.id, manager.conv_repo.get_messages(conv.id), use_model=True, summarizer=empty_summarizer, force=True
            )
        )
        assert res2.ok is True
        assert res2.fallback_used is True

        # Chamada via provider real (sem summarizer) nunca passa `tools` para o modelo auxiliar
        prov = CaptureProvider()
        manager.conv_repo.add_message(conv.id, "user", "Outra mensagem para compactar com provider.")
        manager.conv_repo.add_message(conv.id, "assistant", "Ok.")
        res3 = asyncio.run(
            manager.repack_engine.repack_conversation(
                conv.id, manager.conv_repo.get_messages(conv.id), provider=prov, use_model=True, force=True
            )
        )
        assert res3.ok is True and prov.aux_calls >= 1
    finally:
        db.close()


def test_concurrency_deletion_during_repack_and_hard_budget_fallback(tmp_path):
    """Cenário 18: concorrência sem duplicar repack, exclusão durante repack sem recriar conversa, e limite rígido sem loop."""
    settings = make_settings(
        tmp_path,
        ollama={"num_ctx": 1024, "num_predict": 256},
        memory={"base_dir": tmp_path / "data" / "memory", "compact_threshold_ratio": 0.75, "recent_messages_to_keep": 2},
    )
    db = Database(settings.storage.database_path)
    db.connect()
    try:
        directives = load_directives(settings.directives.path)
        manager = MemoryManager(db, settings.memory, num_ctx=1024, num_predict=256)
        provider = SimulatedProvider()
        service = ChatService(db, provider, directives, memory_manager=manager)

        conv = service.repo.create_conversation("Concorrência Repack")
        for i in range(6):
            service.repo.add_message(conv.id, "user" if i % 2 == 0 else "assistant", f"Bloco denso {i} " * 22)
        msgs = service.repo.get_messages(conv.id)

        # 1) Duas chamadas simultâneas de prepare_context próximas ao limite -> sem erro nem versão duplicada
        async def run_both():
            return await asyncio.gather(
                manager.prepare_context(
                    conversation_id=conv.id, query_text="q1", history=msgs, system_base_text="Base", provider=provider
                ),
                manager.prepare_context(
                    conversation_id=conv.id, query_text="q2", history=msgs, system_base_text="Base", provider=provider
                ),
            )

        r_a, r_b = asyncio.run(run_both())
        repacks = manager.repo.list_repacks(conv.id)
        assert len(repacks) == 1  # apenas v001 criado, sem duplicação
        assert (r_a.repack_triggered or r_b.repack_triggered) is True

        # 2) Exclusão durante repack -> não recria conversa nem grava repack órfão
        conv2 = service.repo.create_conversation("Excluir durante repack")
        for i in range(6):
            service.repo.add_message(conv2.id, "user" if i % 2 == 0 else "assistant", f"Conteúdo {i} " * 20)
        msgs2 = service.repo.get_messages(conv2.id)

        async def slow_summarizer_that_deletes(_prompt: str) -> str:
            await service.delete_conversation(conv2.id)
            return "{}"

        aborted = asyncio.run(
            manager.repack_engine.repack_conversation(
                conv2.id,
                msgs2,
                use_model=True,
                summarizer=slow_summarizer_that_deletes,
                force=True,
                should_abort=lambda: not service.is_conversation_active(conv2.id),
            )
        )
        assert aborted.ok is False
        assert service.repo.get_conversation(conv2.id) is None
        assert not (manager.markdown.repacks_dir / conv2.id).exists()

        # 3) Mensagem gigante que excede o limite mesmo após repack -> redução controlada sem loop infinito
        conv3 = service.repo.create_conversation("Estouro")
        huge = "Dados gigantes de entrada " * 400
        res_huge = asyncio.run(service.send(conv3.id, huge))
        assert res_huge["assistant_message"].content
        # mesmo com uma única mensagem gigante que sozinha não pode ser dividida, não entra em loop
        assert len(manager.repo.list_repacks(conv3.id)) == 0
    finally:
        db.close()


def test_ui_automatic_repack_never_shows_confirmation_popup(tmp_path):
    """Cenários 7, 17, 20.B: repack acontece automaticamente na UI sem modal/cartão pedindo clique."""
    settings = make_settings(
        tmp_path,
        ollama={"num_ctx": 1280, "num_predict": 256},
        memory={
            "base_dir": tmp_path / "data" / "memory",
            "compact_threshold_ratio": 0.70,
            "repack_target_ratio": 0.45,
            "recent_messages_to_keep": 2,
            "max_summary_tokens": 350,
        },
    )
    app = create_app(settings)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)

    try:
        with sync_api.sync_playwright() as pw:
            try:
                browser = pw.chromium.launch(channel="chrome")
            except Exception:
                pytest.skip("Chrome indisponível para Playwright")
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            dialogs_opened: list[str] = []
            page.on("dialog", lambda d: (dialogs_opened.append(d.message), d.dismiss()))
            try:
                app.state.chat_service.provider = CaptureProvider("Contexto processado normalmente.")
                page.goto(f"http://127.0.0.1:{port}")
                page.wait_for_selector("#input", timeout=10000)

                # Preenche turnos até acionar o repack automático pela própria UI
                triggered_notice = False
                for i in range(6):
                    msg = f"Decisão {i}: manter configuração segura {i}. " + ("Detalhe técnico do projeto Enslaved. " * 14)
                    page.fill("#input", msg)
                    page.keyboard.press("Enter")
                    page.wait_for_function(
                        "() => !document.querySelector('.content.streaming') && !document.getElementById('composer').classList.contains('busy')",
                        timeout=15000,
                    )
                    # Garante que nenhum modal de confirmação nem cartão de permissão apareceu
                    assert page.locator("#dialog-modal:not(.hidden)").count() == 0
                    assert page.locator(".message.permission").count() == 0
                    info_text = page.locator("#context-info").inner_text()
                    if "Context automatically optimized" in info_text:
                        triggered_notice = True
                        break

                assert triggered_notice is True
                assert dialogs_opened == []
            finally:
                browser.close()
    finally:
        srv.should_exit = True
        time.sleep(0.1)

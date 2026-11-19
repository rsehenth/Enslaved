"""Smoke REAL (Qwen via Ollama + busca real na internet) da qualidade das respostas.

Usa um banco temporário (não toca em data/). Grava docs/response-quality-smoke-evidence.json.
Uso: .venv\\Scripts\\python scripts\\smoke_response_quality.py
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path

from backend.chat.presentation import load_presentation
from backend.chat.service import ChatService
from backend.core.config import ToolsSettings, load_settings
from backend.core.directives import load_directives
from backend.providers.registry import create_provider
from backend.storage.database import Database
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry

ROOT = Path(__file__).resolve().parent.parent


async def ask(service, conversation_id, text):
    started, tokens, events = time.time(), [], []
    async for event in service.send_stream(conversation_id, text, use_tools=False):
        if event["type"] == "token":
            tokens.append(event["content"])
        elif event["type"] in ("verification", "error"):
            events.append(event)
        elif event["type"] == "meta":
            conversation_id = event["conversation_id"]
    return conversation_id, {
        "pergunta": text, "resposta": "".join(tokens), "eventos": events, "segundos": round(time.time() - started, 1),
    }


async def main():
    settings = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix="enslaved-smoke-"))
    db = Database(tmp / "smoke.db")
    db.connect()
    provider = create_provider(settings)
    health = await provider.health()
    assert health.get("available"), f"Ollama indisponível: {health}"

    def build(net_mode):
        tools = ToolsSettings(permission_modes={"net.fetch": net_mode})
        policy = ToolPolicy(tools)
        engine = ToolEngine(ToolRegistry(), policy, PermissionManager(policy), AuditLog(tmp / f"{net_mode}.jsonl"), tools)
        return ChatService(
            db, provider, load_directives(settings.directives.path), tool_engine=engine,
            presentation=load_presentation(settings.chat), chat_settings=settings.chat, net_settings=settings.net,
        )

    evidence = {"modelo": provider.model, "cenarios": {}}
    online, offline = build("allow"), build("deny")
    for name, service, questions in [
        ("regressao_jaspion_com_internet", online, ["Você conhece o Guarani de Bagé?", "Você conhece o Jaspion?"]),
        ("regressao_jaspion_sem_internet", offline, ["Você conhece o Guarani de Bagé?", "Você conhece o Jaspion?"]),
        ("entidade_ambigua", online, ["Você conhece o Mercúrio?"]),
        ("entidade_sem_fontes", online, ["Você conhece o Zyxqwerty Blorpfeld?"]),
        ("apresentacao_tabela", online, ["Compare em uma tabela Python e JavaScript em 3 critérios."]),
    ]:
        conversation_id, rows = None, []
        for question in questions:
            conversation_id, row = await ask(service, conversation_id, question)
            rows.append(row)
            print(f"[{name}] {question}\n  -> {row['resposta'][:300]!r}\n  {row['eventos']}\n", flush=True)
        evidence["cenarios"][name] = rows
    out = ROOT / "docs" / "response-quality-smoke-evidence.json"
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print("evidência:", out)
    await provider.aclose()
    db.close()


if __name__ == "__main__":
    asyncio.run(main())

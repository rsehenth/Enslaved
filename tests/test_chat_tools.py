import asyncio

from pydantic import BaseModel

from backend.chat.service import ChatService
from backend.core.config import ToolsSettings
from backend.core.directives import load_directives
from backend.providers.base import ChatTurn, ModelProvider, ToolCall
from backend.skills.discovery import SkillRegistry
from backend.skills.manager import SkillManager
from backend.storage.database import Database
from backend.tools.base import Tool, ToolResult, ToolSpec
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry

DIRECTIVES = """---
name: Enslaved
language: pt-BR
---

# Objetivo
Responder diretamente.
"""


class ScriptedToolProvider(ModelProvider):
    name = "scripted"

    def __init__(self, turns):
        super().__init__("script-1b")
        self.turns = list(turns)
        self.calls = []

    async def chat(self, messages, **options):
        self.calls.append({"messages": [dict(m) for m in messages]})
        return self.turns.pop(0).content

    async def stream_chat(self, messages, **options):
        self.calls.append({"messages": [dict(m) for m in messages], "stream": True})
        text = self.turns.pop(0).content
        for start in range(0, len(text), 5):
            yield text[start : start + 5]

    async def chat_turn(self, messages, tools=None, **options):
        self.calls.append({"messages": [dict(m) for m in messages], "tools": tools})
        return self.turns.pop(0)

    async def health(self):
        return {"available": True}


class EchoArgs(BaseModel):
    text: str


class EchoTool(Tool):
    args_model = EchoArgs

    def __init__(self, tool_id: str = "echo", capability: str = "demo.execute"):
        self.spec = ToolSpec(
            id=tool_id,
            name="Echo",
            description="Repete o texto informado (teste).",
            required_permissions=[capability],
        )

    async def run(self, args, permission=None) -> ToolResult:
        if permission is None or not permission.granted:
            return ToolResult(ok=False, error="permissão ausente")
        return ToolResult(ok=True, output=args.get("text", ""))


def make_service(tmp_path, provider, tools_settings=None, skill_frontmatter=None):
    directives_path = tmp_path / "directives.md"
    directives_path.write_text(DIRECTIVES, encoding="utf-8")
    directives = load_directives(directives_path)
    db = Database(tmp_path / "chat.db")
    db.connect()

    registry = ToolRegistry()
    registry.register(EchoTool("echo", "demo.execute"))
    registry.register(EchoTool("asktool", "demo.ask"))

    settings = tools_settings or ToolsSettings(permission_modes={"demo.execute": "allow", "demo.ask": "ask"})
    policy = ToolPolicy(settings)
    permissions = PermissionManager(policy)
    engine = ToolEngine(registry, policy, permissions, AuditLog(tmp_path / "audit.jsonl"), settings)

    skill_manager = None
    if skill_frontmatter:
        skill_dir = tmp_path / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\n{skill_frontmatter}---\n\n# Instruções da skill\nFaça assim.\n",
            encoding="utf-8",
        )
        skill_registry = SkillRegistry()
        skill_registry.discover([tmp_path / "skills"])
        skill_manager = SkillManager(skill_registry)

    service = ChatService(db, provider, directives, tool_engine=engine, skill_manager=skill_manager)
    return service, permissions, engine


async def collect(service, **kwargs):
    events = []
    async for event in service.send_stream(None, "use a ferramenta echo", **kwargs):
        events.append(event)
    return events


async def collect_resume(service, conversation_id):
    events = []
    async for event in service.send_stream(conversation_id, "", resume=True):
        events.append(event)
    return events


def test_tool_loop_allow_mode_executes_and_feeds_result_back(tmp_path):
    provider = ScriptedToolProvider(
        [
            ChatTurn(
                content="",
                tool_calls=[ToolCall(name="echo", arguments='{"text": "dados"}')],
            ),
            ChatTurn(content="A ferramenta retornou: dados"),
        ]
    )
    service, _permissions, _engine = make_service(tmp_path, provider)
    events = asyncio.run(collect(service))
    kinds = [event["type"] for event in events]
    assert kinds[0] == "meta"
    assert "tool_call" in kinds and "tool_result" in kinds and "done" in kinds
    tool_call = next(e for e in events if e["type"] == "tool_call")
    assert tool_call["tool"] == "echo" and tool_call["arguments"] == {"text": "dados"}
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["ok"] is True and tool_result["output_preview"] == "dados"
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert "dados" in tokens

    # segundo turno recebeu assistant com tool_calls + mensagem tool
    second_payload = provider.calls[1]["messages"]
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second_payload)
    tool_message = next(m for m in second_payload if m.get("role") == "tool")
    assert tool_message["tool_name"] == "echo"
    assert "<ferramenta" in tool_message["content"] and "dados" in tool_message["content"]

    # persistência: roles assistant(tool_calls) + tool + assistant final
    conversation_id = events[0]["conversation_id"]
    roles = [m.role for m in service.repo.get_messages(conversation_id)]
    assert roles == ["user", "assistant", "tool", "assistant"]


def test_tool_loop_ask_mode_ends_turn_and_resumes_after_decision(tmp_path):
    provider = ScriptedToolProvider(
        [
            ChatTurn(content="", tool_calls=[ToolCall(name="asktool", arguments='{"text": "x"}')]),
            ChatTurn(content="autorizado e executado"),
        ]
    )
    service, permissions, engine = make_service(tmp_path, provider)

    async def scenario():
        events = []
        async for event in service.send_stream(None, "use a ferramenta asktool"):
            events.append(event)
            if event["type"] == "done":
                break
        return events

    events = asyncio.run(scenario())
    kinds = [event["type"] for event in events]
    assert "permission_request" in kinds
    assert "permission_resolved" not in kinds  # decisão nunca acontece dentro do stream
    done = events[-1]
    assert done["type"] == "done" and done["status"] == "awaiting_permission"
    request_id = done["request_id"]
    conversation_id = events[0]["conversation_id"]

    # a invocação pendente existe no motor, ligada à conversa
    assert request_id in engine._pending_invocations

    # proprietário decide: executa + persiste resultado (como faz o endpoint)
    decision = permissions.resolve(request_id, granted=True, decided_by="owner")
    assert decision is not None and decision.granted is True
    result = asyncio.run(engine.complete_invocation(request_id, decision))
    assert result.ok is True
    assert result.data["conversation_id"] == conversation_id
    service.persist_tool_result(conversation_id, "asktool", result)

    # histórico: user, assistant(tool_calls), tool(resultado)
    roles = [m.role for m in service.repo.get_messages(conversation_id)]
    assert roles == ["user", "assistant", "tool"]

    # retomada: novo turno sem mensagem do usuário; modelo responde com o resultado
    resume_events = asyncio.run(collect_resume(service, conversation_id))
    resume_kinds = [event["type"] for event in resume_events]
    assert resume_kinds[0] == "meta" and resume_events[0]["resume"] is True
    tokens = "".join(e.get("content", "") for e in resume_events if e["type"] == "token")
    assert "autorizado e executado" in tokens
    assert resume_events[-1]["type"] == "done"
    # o payload da retomada contém a mensagem tool com o resultado
    resume_payload = provider.calls[-1]["messages"]
    tool_message = next(m for m in resume_payload if m.get("role") == "tool")
    assert tool_message["tool_name"] == "asktool"


def test_tool_loop_denied_permission_persists_denial(tmp_path):
    provider = ScriptedToolProvider(
        [
            ChatTurn(content="", tool_calls=[ToolCall(name="asktool", arguments='{"text": "x"}')]),
            ChatTurn(content="Entendi, sem execução."),
        ]
    )
    service, permissions, engine = make_service(tmp_path, provider)

    async def scenario():
        events = []
        async for event in service.send_stream(None, "use a ferramenta asktool"):
            events.append(event)
            if event["type"] == "done":
                break
        return events

    events = asyncio.run(scenario())
    done = events[-1]
    assert done["status"] == "awaiting_permission"
    request_id = done["request_id"]
    conversation_id = events[0]["conversation_id"]

    decision = permissions.resolve(request_id, granted=False, decided_by="owner")
    result = asyncio.run(engine.complete_invocation(request_id, decision))
    assert result.ok is False
    assert "negada" in result.error
    service.persist_tool_result(conversation_id, "asktool", result)

    roles = [m.role for m in service.repo.get_messages(conversation_id)]
    assert roles == ["user", "assistant", "tool"]
    tool_message = service.repo.get_messages(conversation_id)[-1]
    assert tool_message.meta["ok"] is False

    resume_events = asyncio.run(collect_resume(service, conversation_id))
    assert resume_events[-1]["type"] == "done"


def test_tool_loop_guard_blocks_after_limit(tmp_path):
    settings = ToolsSettings(
        permission_modes={"demo.execute": "allow"},
        max_calls_per_request=2,
    )
    tool_turn = ChatTurn(content="", tool_calls=[ToolCall(name="echo", arguments='{"text": "t"}')])
    provider = ScriptedToolProvider([tool_turn, tool_turn, tool_turn, ChatTurn(content="fim")])
    service, _permissions, _engine = make_service(tmp_path, provider, tools_settings=settings)
    events = asyncio.run(collect(service))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert len(tool_calls) == 2
    tokens = "".join(e["content"] for e in events if e["type"] == "token")
    assert "Limite de chamadas de ferramentas" in tokens
    assert events[-1]["type"] == "done"


def test_invalid_tool_arguments_reported_without_crash(tmp_path):
    provider = ScriptedToolProvider(
        [
            ChatTurn(content="", tool_calls=[ToolCall(name="echo", arguments="nao-e-json")]),
            ChatTurn(content="ok, seguindo sem a ferramenta"),
        ]
    )
    service, _permissions, _engine = make_service(tmp_path, provider)
    events = asyncio.run(collect(service))
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["ok"] is False
    assert events[-1]["type"] == "done"


def test_unknown_tool_reported(tmp_path):
    provider = ScriptedToolProvider(
        [
            ChatTurn(content="", tool_calls=[ToolCall(name="inexistente", arguments="{}")]),
            ChatTurn(content="seguindo"),
        ]
    )
    service, _permissions, _engine = make_service(tmp_path, provider)
    events = asyncio.run(collect(service))
    tool_result = next(e for e in events if e["type"] == "tool_result")
    # erro estruturado: nome inexistente + catálogo completo de ferramentas disponíveis
    assert tool_result["ok"] is False
    assert "Ferramenta inexistente" in tool_result["error"] and "inexistente" in tool_result["error"]
    assert "echo" in tool_result["error"] and "asktool" in tool_result["error"]
    assert tool_result["data"]["available_tools"] == ["asktool", "echo"]


def test_use_tools_false_uses_plain_streaming(tmp_path):
    provider = ScriptedToolProvider([ChatTurn(content="resposta em stream")])
    service, _permissions, _engine = make_service(tmp_path, provider)
    events = asyncio.run(collect(service, use_tools=False))
    kinds = [event["type"] for event in events]
    assert "tool_call" not in kinds
    assert provider.calls[0].get("stream") is True
    assert events[-1]["type"] == "done"


def test_activated_skill_instructions_enter_system_prompt(tmp_path):
    provider = ScriptedToolProvider([ChatTurn(content="feito")])
    frontmatter = "name: demo\ndescription: Skill de demonstração de instruções.\n"
    service, _permissions, _engine = make_service(tmp_path, provider, skill_frontmatter=frontmatter)
    events = asyncio.run(collect(service, activated_skills=["demo"], use_tools=False))
    system = provider.calls[0]["messages"][0]["content"]
    assert "Skills" in system
    assert "Instruções da skill" in system
    assert "demo" in system
    suggested = [e for e in events if e["type"] == "skills_suggested"]
    # "use a ferramenta echo" não casa com a skill demo (keywords ausentes na mensagem)
    assert suggested == []
    assert events[-1]["type"] == "done"


def test_unknown_activated_skill_is_skipped(tmp_path):
    provider = ScriptedToolProvider([ChatTurn(content="feito")])
    service, _permissions, _engine = make_service(tmp_path, provider)
    events = asyncio.run(collect(service, activated_skills=["fantasma"], use_tools=False))
    system = provider.calls[0]["messages"][0]["content"]
    assert "fantasma" not in system
    assert events[-1]["type"] == "done"


def test_tool_output_truncated_in_context(tmp_path):
    settings = ToolsSettings(permission_modes={"demo.execute": "allow"}, max_output_chars=300)
    provider = ScriptedToolProvider(
        [
            ChatTurn(content="", tool_calls=[ToolCall(name="echo", arguments='{"text": "a"}')]),
            ChatTurn(content="recebi"),
        ]
    )
    service, _permissions, _engine = make_service(tmp_path, provider, tools_settings=settings)

    class LongEcho(EchoTool):
        async def run(self, args, permission=None) -> ToolResult:
            return ToolResult(ok=True, output="a" * 5000)

    service.tool_engine._registry._tools["echo"] = LongEcho()
    events = asyncio.run(collect(service))
    tool_result = next(e for e in events if e["type"] == "tool_result")
    assert tool_result["output_truncated_in_context"] is True
    second_payload = provider.calls[1]["messages"]
    tool_message = next(m for m in second_payload if m.get("role") == "tool")
    # 5000 chars de output viram envelope < 450 (truncado em max_output_chars + wrapper)
    assert len(tool_message["content"]) < 450

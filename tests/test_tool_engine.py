import asyncio
import json

from pydantic import BaseModel, Field

from backend.core.config import ToolsSettings
from backend.tools.base import Tool, ToolResult, ToolSpec
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry


class EchoArgs(BaseModel):
    text: str = Field(min_length=1)
    times: int = Field(default=1, ge=1, le=10)


class EchoTool(Tool):
    args_model = EchoArgs

    def __init__(self, capability: str = "demo.execute"):
        self.spec = ToolSpec(
            id="echo",
            name="Echo",
            description="Ferramenta de teste",
            required_permissions=[capability],
        )

    async def run(self, args, permission=None) -> ToolResult:
        if permission is None or not permission.granted:
            return ToolResult(ok=False, error="permissão ausente")
        return ToolResult(ok=True, output=args["text"] * args["times"], data={"capability": "demo.execute"})


class SlowTool(Tool):
    def __init__(self):
        self.spec = ToolSpec(id="slow", name="Slow", description="Demora", required_permissions=["demo.execute"])

    async def run(self, args, permission=None) -> ToolResult:
        await asyncio.sleep(1.0)
        return ToolResult(ok=True, output="pronto")


def make_engine(tools, settings: ToolsSettings | None = None, audit=None):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    settings = settings or ToolsSettings()
    policy = ToolPolicy(settings)
    permissions = PermissionManager(policy)
    audit = audit or AuditLog(None)  # substituído por tmp_path nos testes que inspecionam
    return ToolEngine(registry, policy, permissions, audit, settings), permissions


def make_settings(**overrides) -> ToolsSettings:
    data = {"permission_modes": {"demo.execute": "allow"}}
    data.update(overrides)
    return ToolsSettings.model_validate(data)


def test_invoke_validates_args_against_schema():
    engine, _ = make_engine([EchoTool()])
    result = asyncio.run(engine.invoke("echo", {"text": ""}))
    assert result.ok is False
    assert "Argumentos inválidos" in result.error


def test_invoke_unknown_tool_returns_structured_error():
    engine, _ = make_engine([EchoTool()])
    result = asyncio.run(engine.invoke("não-existe", {}))
    assert result.ok is False
    assert "desconhecida" in result.error


def test_invoke_allow_mode_runs_without_decision():
    engine, _ = make_engine([EchoTool()], make_settings())
    result = asyncio.run(engine.invoke("echo", {"text": "oi", "times": 2}))
    assert result.ok is True
    assert result.output == "oioi"
    assert result.data["tool"] == "echo"
    assert result.data["duration_ms"] >= 0


def test_invoke_ask_mode_waits_for_owner_decision():
    engine, permissions = make_engine(
        [EchoTool()], make_settings(permission_modes={"demo.execute": "ask"})
    )

    async def scenario():
        task = asyncio.create_task(engine.invoke("echo", {"text": "oi"}))
        await asyncio.sleep(0)
        pending = permissions.pending_requests()
        assert len(pending) == 1
        assert pending[0].tool_id == "echo"
        assert pending[0].capabilities == ["demo.execute"]
        permissions.resolve(pending[0].id, granted=True, decided_by="owner")
        return await task

    result = asyncio.run(scenario())
    assert result.ok is True
    assert result.output == "oi"


def test_invoke_ask_mode_denied_by_owner():
    engine, permissions = make_engine(
        [EchoTool()], make_settings(permission_modes={"demo.execute": "ask"})
    )

    async def scenario():
        task = asyncio.create_task(engine.invoke("echo", {"text": "oi"}))
        await asyncio.sleep(0)
        pending = permissions.pending_requests()
        permissions.resolve(pending[0].id, granted=False, decided_by="owner")
        return await task

    result = asyncio.run(scenario())
    assert result.ok is False
    assert "Autorização negada" in result.error


def test_invoke_ask_mode_times_out():
    engine, _ = make_engine(
        [EchoTool()],
        make_settings(
            permission_modes={"demo.execute": "ask"}, permission_wait_seconds=0.05
        ),
    )
    result = asyncio.run(engine.invoke("echo", {"text": "oi"}))
    assert result.ok is False
    assert "Autorização negada" in result.error


def test_invoke_deny_mode_blocks_without_pending_request():
    engine, permissions = make_engine(
        [EchoTool()], make_settings(permission_modes={"demo.execute": "deny"})
    )
    result = asyncio.run(engine.invoke("echo", {"text": "oi"}))
    assert result.ok is False
    assert "política de permissões" in result.error
    assert permissions.pending_requests() == []


def test_invoke_enforces_tool_timeout():
    engine, _ = make_engine([SlowTool()], make_settings(tool_timeout_seconds=0.05))
    result = asyncio.run(engine.invoke("slow", {}))
    assert result.ok is False
    assert "Tempo limite" in result.error


def test_invoke_captures_internal_tool_exception():
    class BrokenTool(Tool):
        def __init__(self):
            self.spec = ToolSpec(id="broken", name="Broken", description="quebra")

        async def run(self, args, permission=None) -> ToolResult:
            raise RuntimeError("falha proposital")

    engine, _ = make_engine([BrokenTool()])
    result = asyncio.run(engine.invoke("broken", {}))
    assert result.ok is False
    assert "RuntimeError" in result.error


def test_audit_records_runs_and_permissions(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    engine, permissions = make_engine(
        [EchoTool()],
        make_settings(permission_modes={"demo.execute": "ask"}),
        audit=audit,
    )

    async def scenario():
        task = asyncio.create_task(
            engine.invoke("echo", {"text": "oi"}, source="chat", conversation_id="c1")
        )
        await asyncio.sleep(0)
        permissions.resolve(permissions.pending_requests()[0].id, granted=True)
        return await task

    asyncio.run(scenario())
    events = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [event["event"] for event in events]
    assert kinds == ["permission_requested", "permission_resolved", "tool_run"]
    run_event = events[-1]
    assert run_event["tool"] == "echo"
    assert run_event["ok"] is True
    assert run_event["conversation_id"] == "c1"
    assert run_event["args_summary"] == '{"text": "oi", "times": 1}'
    assert audit.read_recent(2)[0]["event"] == "tool_run"


def test_describe_and_ollama_tools_format():
    engine, _ = make_engine([EchoTool()], make_settings())
    described = engine.describe()
    assert described[0]["id"] == "echo"
    assert described[0]["mode"] == "allow"
    assert described[0]["parameters"]["properties"]["text"]["type"] == "string"
    ollama = engine.ollama_tools()
    assert ollama[0]["type"] == "function"
    assert ollama[0]["function"]["name"] == "echo"
    assert ollama[0]["function"]["parameters"]["required"] == ["text"]


def test_effective_mode_deny_overrides_ask():
    engine_ask, _ = make_engine([EchoTool()], make_settings(permission_modes={"demo.execute": "ask"}))
    assert engine_ask.effective_mode(engine_ask.get("echo")) == "ask"
    engine_deny, _ = make_engine([EchoTool()], make_settings(permission_modes={"demo.execute": "deny"}))
    assert engine_deny.effective_mode(engine_deny.get("echo")) == "deny"


def test_policy_unknown_capability_defaults_to_ask():
    engine, _ = make_engine([EchoTool(capability="capacidade.fora.da.politica")], make_settings())
    assert engine.effective_mode(engine.get("echo")) == "ask"


def test_session_grant_skips_request_after_remember():
    engine, permissions = make_engine(
        [EchoTool()], make_settings(permission_modes={"demo.execute": "ask"})
    )

    async def scenario():
        first = asyncio.create_task(engine.invoke("echo", {"text": "oi"}))
        await asyncio.sleep(0)
        pending = permissions.pending_requests()
        permissions.resolve(pending[0].id, granted=True, remember_session=True)
        result_first = await first
        result_second = await engine.invoke("echo", {"text": "de novo"})
        return result_first, result_second

    first, second = asyncio.run(scenario())
    assert first.ok and second.ok
    assert permissions.session_capabilities() == ["demo.execute"]

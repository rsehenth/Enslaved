import asyncio
import json

import pytest

from backend.tools.base import PermissionDecision, PermissionRequest, Tool, ToolResult, ToolSpec
from backend.tools.registry import AuditLog, ToolRegistry


class EchoTool(Tool):
    def __init__(self):
        self.spec = ToolSpec(
            id="echo",
            name="Echo",
            description="Ferramenta de teste",
            required_permissions=["demo.execute"],
        )

    async def run(self, args, permission=None) -> ToolResult:
        if permission is None or not permission.granted:
            return ToolResult(ok=False, error="permissão ausente")
        return ToolResult(ok=True, output=args.get("text", ""))


def test_registry_registers_and_rejects_duplicates():
    registry = ToolRegistry()
    tool = EchoTool()
    registry.register(tool)
    assert registry.get("echo") is tool
    assert [s.id for s in registry.list_specs()] == ["echo"]
    with pytest.raises(ValueError):
        registry.register(EchoTool())


def test_permission_contract_flow():
    request = PermissionRequest(tool_id="echo", action="executar", scope={"text": "oi"})
    decision = PermissionDecision(granted=True, scope=request.scope, decided_by="owner")
    tool = EchoTool()
    granted = asyncio.run(tool.run({"text": "oi"}, decision))
    assert granted.ok is True
    assert granted.output == "oi"
    denied = asyncio.run(tool.run({"text": "oi"}, None))
    assert denied.ok is False
    assert denied.error == "permissão ausente"


def test_audit_log_appends_jsonl(tmp_path):
    log = AuditLog(tmp_path / "logs" / "audit.jsonl")
    log.append({"tool": "echo", "action": "executar", "granted": True})
    log.append({"tool": "echo", "action": "executar", "granted": False})
    lines = (tmp_path / "logs" / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["tool"] == "echo"
    assert "ts" in first

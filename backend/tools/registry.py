import json
import time
from pathlib import Path

from backend.tools.base import Tool, ToolSpec


class ToolRegistry:
    """Registro de ferramentas. Só entra aqui quem declara ToolSpec + permissões."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.spec.id in self._tools:
            raise ValueError(f"Ferramenta duplicada: {tool.spec.id}")
        self._tools[tool.spec.id] = tool

    def get(self, tool_id: str) -> Tool | None:
        return self._tools.get(tool_id)

    def list_tools(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda tool: tool.spec.id)

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec for tool in self._tools.values()]


class AuditLog:
    """Registro append-only (JSONL) de ações de ferramentas. path=None desativa a escrita."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path is not None else None

    def append(self, event: dict) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_recent(self, limit: int = 50) -> list[dict]:
        """Últimos `limit` eventos (mais recentes primeiro), tolerante a leituras concorrentes."""
        if self.path is None or not self.path.is_file():
            return []
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as file:
                lines = file.readlines()[-limit:]
        except OSError:
            return []
        recent = []
        for line in reversed(lines):
            try:
                recent.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return recent

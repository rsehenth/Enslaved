"""Ferramenta de diagnóstico local do próprio Enslaved (somente leitura)."""

import asyncio
import time

from pydantic import BaseModel

from backend.tools.base import Tool, ToolResult, ToolSpec
from backend.core.gpu import gpu_snapshot


class NoArgs(BaseModel):
    """Ferramenta sem argumentos."""


class AppDiagnosticsTool(Tool):
    args_model = NoArgs

    def __init__(self, context: dict):
        super().__init__()
        self._context = context
        self.spec = ToolSpec(
            id="app.diagnostics",
            name="Enslaved diagnostics",
            description=(
                "Reports the app's local state: provider/model, models loaded in VRAM, GPU, "
                "registered tools and skills, database size, and uptime. Read-only; no data leaves the computer."
            ),
            capabilities=["diagnostics", "read-only"],
            required_permissions=["app.diagnostics"],
        )

    async def run(self, args: dict, permission=None) -> ToolResult:
        if permission is None or not permission.granted:
            return ToolResult(ok=False, error="permissão ausente")
        provider = self._context.get("provider")
        settings = self._context.get("settings")
        provider_info: dict = {"name": None, "model": None}
        health: dict = {"available": None}
        loaded_models: list = []
        if provider is not None:
            provider_info = {"name": provider.name, "model": provider.model}
            health = await provider.health()
            if health.get("available"):
                loaded_models = await provider.loaded_models()
        gpus = await asyncio.to_thread(gpu_snapshot)

        database_path = self._context.get("database_path")
        database_size_mb = None
        if database_path is not None:
            try:
                database_size_mb = round(database_path.stat().st_size / 1048576, 2)
            except OSError:
                database_size_mb = None

        skill_manager = self._context.get("skill_manager")
        skills_info = None
        if skill_manager is not None:
            skills_info = skill_manager.summary()

        tool_engine = self._context.get("tool_engine")
        tools_info = None
        if tool_engine is not None:
            tools_info = [entry["id"] for entry in tool_engine.describe()]

        ollama = getattr(settings, "ollama", None)
        active_profile = settings.active_profile() if ollama is not None else None
        return ToolResult(
            ok=True,
            output={
                "app": {
                    "version": "0.2.0-f2a",
                    "uptime_seconds": round(time.time() - self._context["started_at"], 1),
                    "started_at": self._context.get("started_at"),
                },
                "provider": {**provider_info, "available": health.get("available")},
                "loaded_models": loaded_models,
                "gpus": gpus,
                "inference": None
                if active_profile is None
                else {
                    "num_ctx": active_profile.num_ctx,
                    "num_predict": active_profile.num_predict,
                    "think": active_profile.think,
                },
                "skills": skills_info,
                "tools": tools_info,
                "database": {"path": str(database_path) if database_path else None, "size_mb": database_size_mb},
            },
            data={"source": "local", "net_access": False},
        )

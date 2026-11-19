"""Ferramentas nativas de baixo risco da Fase 2A (todas somente leitura)."""

from pathlib import Path

from backend.core.config import Settings
from backend.tools.base import Tool
from backend.tools.builtin.app_diagnostics import AppDiagnosticsTool
from backend.tools.builtin.fs_read import FsReadTool
from backend.tools.builtin.text_analyze import TextAnalyzeTool
from backend.tools.builtin.web_fetch import WebFetchTool
from backend.tools.builtin.web_inspect import WebInspectTool

__all__ = ["build_default_tools", "AppDiagnosticsTool", "FsReadTool", "TextAnalyzeTool", "WebFetchTool", "WebInspectTool"]


def build_default_tools(settings: Settings, context: dict) -> list[Tool]:
    """Cria as ferramentas nativas com escopo derivado da configuração."""
    return [
        FsReadTool(settings.tools.fs),
        WebFetchTool(settings.tools.net),
        WebInspectTool(settings.tools.net),
        TextAnalyzeTool(),
        AppDiagnosticsTool(context),
    ]

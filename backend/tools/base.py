from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """Declaração estática de uma ferramenta e das permissões que exige."""

    id: str
    name: str
    description: str
    capabilities: list[str] = Field(default_factory=list)
    dangerous: bool = False
    required_permissions: list[str] = Field(default_factory=list)


class ToolResult(BaseModel):
    """Resultado estruturado da execução de uma ferramenta."""

    ok: bool
    output: Any = None
    error: str | None = None
    data: dict = Field(default_factory=dict)


class PermissionRequest(BaseModel):
    """Solicitação de permissão para executar uma ação de ferramenta."""

    id: str = ""
    tool_id: str
    action: str
    scope: dict = Field(default_factory=dict)
    reason: str | None = None
    capabilities: list[str] = Field(default_factory=list)
    source: str = "manual"
    conversation_id: str | None = None


class PermissionDecision(BaseModel):
    """Decisão explícita do proprietário (ou política) sobre um pedido."""

    granted: bool
    scope: dict = Field(default_factory=dict)
    decided_by: str = "owner"
    reason: str | None = None


class Tool(ABC):
    """Contrato base para ferramentas. Capacidades declaradas e escopo limitado.

    Ferramentas com schema de entrada declaram `args_model` (pydantic); o motor
    valida os argumentos antes de chamar `run()`. Nada de código arbitrário.
    """

    spec: ToolSpec
    args_model: type[BaseModel] | None = None

    @abstractmethod
    async def run(self, args: dict, permission: PermissionDecision | None = None) -> ToolResult:
        """Executa a ferramenta. Implementações DEVEM validar o escopo da permissão antes de agir."""

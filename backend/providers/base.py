from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

# Metadados da última geração (uso/término), isolados por requisição HTTP. O valor é um dict
# MUTÁVEL criado no início da requisição: tasks-filhas (create_task copia o contexto) gravam
# no mesmo objeto, então o leitor da requisição vê o resultado; outras requisições não.
_request_generation_meta: ContextVar[dict | None] = ContextVar("request_generation_meta", default=None)


def begin_request_generation_scope() -> None:
    """Abre o escopo de metadados de geração da requisição atual (chamado pelo middleware HTTP)."""
    _request_generation_meta.set({})


class ProviderError(Exception):
    """O provider respondeu com erro (ex.: modelo inexistente)."""


class ProviderUnavailableError(ProviderError):
    """Provider inacessível (sem conexão ou timeout)."""


@dataclass
class ToolCall:
    """Chamada de ferramenta pedida pelo modelo (formato Ollama/OpenAI normalizado)."""

    name: str
    arguments: str  # JSON string dos argumentos
    id: str | None = None


@dataclass
class ChatTurn:
    """Uma resposta completa do modelo: texto e/ou pedidos de ferramenta."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Uso/término deste turno (last_usage do provider é compartilhado entre requisições concorrentes).
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


class Thinking(str):
    """Trecho do raciocínio do modelo (exibido ao vivo; não entra na resposta nem é salvo).

    Subclasse de `str` para que providers/fakes que só geram texto continuem válidos."""


class ModelProvider(ABC):
    """Contrato para motores de inferência (Ollama hoje; llama.cpp e outros no futuro)."""

    name: str = "base"

    def __init__(self, model: str):
        self.model = model
        # Fora de uma requisição HTTP (scripts, testes diretos) vale este dict do próprio provider.
        self._unscoped_meta: dict[str, Any] = {"usage": None, "finish_reason": None}
        # Recursos do modelo ativo, definidos pelo perfil em provider.models (config/settings.yaml).
        self.capabilities: dict[str, bool] = {"vision": True, "experimental": False}

    def _generation_meta(self) -> dict[str, Any]:
        scope = _request_generation_meta.get()
        if scope is None:
            return self._unscoped_meta
        return scope.setdefault(id(self), {"usage": None, "finish_reason": None})

    @property
    def last_usage(self) -> dict[str, int] | None:
        return self._generation_meta()["usage"]

    @last_usage.setter
    def last_usage(self, value: dict[str, int] | None) -> None:
        self._generation_meta()["usage"] = value

    @property
    def last_finish_reason(self) -> str | None:
        """"stop" | "length" (corte pelo limite de saída) | None (desconhecido), da última geração DESTA requisição."""
        return self._generation_meta()["finish_reason"]

    @last_finish_reason.setter
    def last_finish_reason(self, value: str | None) -> None:
        self._generation_meta()["finish_reason"] = value

    @abstractmethod
    async def chat(self, messages: list[dict[str, Any]], **options: Any) -> str:
        """Envia o histórico e retorna o texto completo da resposta."""

    @abstractmethod
    def stream_chat(self, messages: list[dict[str, Any]], **options: Any) -> AsyncIterator[str]:
        """Envia o histórico e gera a resposta em partes."""

    async def chat_turn(self, messages: list[dict[str, Any]], tools: list[dict] | None = None, **options: Any) -> ChatTurn:
        """Turno com suporte opcional a ferramentas. Padrão: sem tools; providers que
        não suportam devem recusar explicitamente quando `tools` é pedido."""
        if tools:
            raise ProviderError(f"Provider {self.name!r} não suporta tool calling")
        return ChatTurn(content=await self.chat(messages, **options))

    async def stream_turn(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None, **options: Any
    ) -> AsyncIterator["str | Thinking | ChatTurn"]:
        """Como `chat_turn`, mas gera texto (str) e raciocínio (Thinking) ao vivo e termina
        com o `ChatTurn` completo. Padrão: sem streaming real (um único trecho)."""
        turn = await self.chat_turn(messages, tools=tools, **options)
        if turn.content:
            yield turn.content
        yield turn

    async def supports_native_tools(self) -> bool:
        """Se o modelo ativo aceita tool calling nativo (`tools` no payload do provider).

        Padrão: perfil explícito (`capabilities["tools"]`) ou True quando o provider não
        tem detecção própria. Capacidade independente de visão, raciocínio e das
        ferramentas disponíveis no backend."""
        explicit = self.capabilities.get("tools")
        return True if explicit is None else bool(explicit)

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Verifica a disponibilidade do provider."""

    async def loaded_models(self) -> list[dict[str, Any]]:
        """Modelos atualmente carregados na memória (quando o provider expõe isso)."""
        return []

    async def aclose(self) -> None:
        """Libera recursos do provider."""

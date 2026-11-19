from typing import Any, AsyncIterator, Callable

from backend.providers.base import ModelProvider


class SimulatedProvider(ModelProvider):
    """Provider determinístico para testes e desenvolvimento sem GPU/Ollama."""

    name = "simulated"

    def __init__(self, model: str = "simulado-1b", reply: str | Callable | None = None, finish_reasons: list[str] | None = None):
        super().__init__(model)
        self._reply = reply
        self._finish = list(finish_reasons or [])  # uma por chamada; permite simular corte por limite de saída
        self.calls: list[list[dict[str, str]]] = []

    def _reply_text(self, messages: list[dict[str, str]]) -> str:
        if callable(self._reply):
            return self._reply(messages)
        if isinstance(self._reply, str):
            return self._reply
        last = messages[-1]["content"] if messages else ""
        return f"[simulado] Recebi {len(messages)} mensagem(ns). Última: {last[:120]}"

    async def chat(self, messages: list[dict[str, str]], **options: Any) -> str:
        self.calls.append([dict(m) for m in messages])
        self.last_finish_reason = self._finish.pop(0) if self._finish else None
        return self._reply_text(messages)

    async def stream_chat(self, messages: list[dict[str, str]], **options: Any) -> AsyncIterator[str]:
        text = await self.chat(messages, **options)
        for start in range(0, len(text), 8):
            yield text[start : start + 8]

    async def health(self) -> dict[str, Any]:
        return {
            "available": True,
            "detail": "provider simulado (sem inferência real)",
            "models": [self.model],
        }

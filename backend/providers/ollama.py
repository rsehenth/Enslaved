import json
from typing import Any, AsyncIterator

import httpx

from backend.core.redact import redact_secrets
from backend.providers.base import (
    ChatTurn,
    ModelProvider,
    ProviderError,
    ProviderUnavailableError,
    Thinking,
    ToolCall,
)

_UNAVAILABLE_HINT = "Verifique se o serviço está em execução: ollama serve"
# Raciocínio consome num_predict antes da resposta; esta folga evita respostas vazias.
_THINK_EXTRA_PREDICT = 2048


class OllamaProvider(ModelProvider):
    """Provider para a API local do Ollama (http://127.0.0.1:11434)."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120.0,
        default_options: dict[str, Any] | None = None,
        client: httpx.AsyncClient | None = None,
        think: bool = False,
    ):
        super().__init__(model)
        self.base_url = base_url.rstrip("/")
        self.default_options = dict(default_options or {})
        self.think = think
        self._native_tools: bool | None = None  # resolvido sob demanda via /api/show
        # trust_env=False: ferramenta local não deve usar proxies do ambiente
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds, trust_env=False)

    def _payload(
        self,
        messages: list[dict[str, Any]],
        stream: bool,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        merged = {**self.default_options, **options}
        # think=false evita que modelos com raciocínio embutido (ex.: qwen3.5)
        # consumam num_predict antes de produzir a resposta final; a interface pode
        # ligar o raciocínio por mensagem (`think` nas opções).
        think = merged.pop("think", None)
        think = self.think if think is None else bool(think)
        num_ctx = merged.get("num_ctx")
        if isinstance(num_ctx, int) and num_ctx > 0 and isinstance(merged.get("num_predict"), int):
            default_pred = int(self.default_options.get("num_predict") or max(128, num_ctx // 4))
            max_safe_pred = max(default_pred, num_ctx // 3)
            if merged["num_predict"] > max_safe_pred:
                merged["num_predict"] = default_pred
        if think and merged.get("num_predict") is not None:
            if isinstance(num_ctx, int) and num_ctx > 0:
                merged["num_predict"] = min(merged["num_predict"] + _THINK_EXTRA_PREDICT, max(128, num_ctx // 2))
            else:
                merged["num_predict"] += _THINK_EXTRA_PREDICT
        fmt = merged.pop("format", None)  # "json" ou JSON Schema: saída estruturada
        # truncate/shift são campos de topo do /api/chat (não `options`). truncate=False faz o
        # Ollama recusar (HTTP 400, contagem exata) prompt > num_ctx em vez de truncar em silêncio.
        top_level = {key: merged.pop(key) for key in ("truncate", "shift") if key in merged}
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "think": think,
            "options": {k: v for k, v in merged.items() if v is not None},
            **top_level,
        }
        if fmt:
            body["format"] = fmt
        return body

    @staticmethod
    def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
        calls = []
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {}, ensure_ascii=False)
            call_id = call.get("id")
            calls.append(
                ToolCall(
                    name=function.get("name", ""),
                    arguments=arguments,
                    id=str(call_id) if call_id is not None else None,
                )
            )
        return calls

    async def chat(self, messages: list[dict[str, Any]], **options: Any) -> str:
        self.last_usage = None
        self.last_finish_reason = None
        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json=self._payload(messages, False, options),
            )
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(
                f"Ollama indisponível em {self.base_url} ({exc.__class__.__name__}). {_UNAVAILABLE_HINT}"
            ) from exc
        self._raise_for_status(response)
        data = response.json()
        if data.get("error"):
            raise ProviderError(f"Erro do Ollama: {redact_secrets(str(data['error']))}")
        self._capture_usage(data)
        return data.get("message", {}).get("content", "")

    async def chat_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        **options: Any,
    ) -> ChatTurn:
        """Turno completo (não-streaming) com suporte a `tools` da API /api/chat."""
        self.last_usage = None
        self.last_finish_reason = None
        payload = self._payload(messages, False, options)
        if tools:
            payload["tools"] = tools
        try:
            response = await self._client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(
                f"Ollama indisponível em {self.base_url} ({exc.__class__.__name__}). {_UNAVAILABLE_HINT}"
            ) from exc
        self._raise_for_status(response)
        data = response.json()
        if data.get("error"):
            raise ProviderError(f"Erro do Ollama: {redact_secrets(str(data['error']))}")
        self._capture_usage(data)
        message = data.get("message", {}) or {}
        usage, reason = self._usage_of(data)
        return ChatTurn(
            content=message.get("content", "") or "",
            tool_calls=self._parse_tool_calls(message),
            usage=usage,
            finish_reason=reason,
        )

    async def stream_chat(self, messages: list[dict[str, Any]], **options: Any) -> AsyncIterator[str]:
        async for data in self._stream_data(self._payload(messages, True, options)):
            message = data.get("message") or {}
            if message.get("thinking"):
                yield Thinking(message["thinking"])
            if message.get("content"):
                yield message["content"]

    async def stream_turn(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None, **options: Any
    ) -> AsyncIterator[str | Thinking | ChatTurn]:
        payload = self._payload(messages, True, options)
        if tools:
            payload["tools"] = tools
        content: list[str] = []
        tool_calls: list[ToolCall] = []
        usage, reason = None, None
        async for data in self._stream_data(payload):
            message = data.get("message") or {}
            if message.get("thinking"):
                yield Thinking(message["thinking"])
            if message.get("content"):
                content.append(message["content"])
                yield message["content"]
            tool_calls.extend(self._parse_tool_calls(message))
            if data.get("done"):
                usage, reason = self._usage_of(data)
        yield ChatTurn(content="".join(content), tool_calls=tool_calls, usage=usage, finish_reason=reason)

    async def _stream_data(self, payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Linhas NDJSON do /api/chat já validadas; captura o uso ao final."""
        self.last_usage = None
        self.last_finish_reason = None
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json=payload,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ProviderError(f"Resposta inválida do Ollama: {line[:120]!r}") from exc
                    if data.get("error"):
                        raise ProviderError(f"Erro do Ollama: {redact_secrets(str(data['error']))}")
                    yield data
                    if data.get("done"):
                        self._capture_usage(data)
                        return
                raise ProviderError("Ollama encerrou o stream sem confirmar o término da resposta.")
        except httpx.TransportError as exc:
            raise ProviderUnavailableError(
                f"Ollama indisponível em {self.base_url} ({exc.__class__.__name__}). {_UNAVAILABLE_HINT}"
            ) from exc

    async def _show_capabilities(self) -> list[str] | None:
        """Capacidades declaradas pelo Ollama (POST /api/show, só metadados); None = indisponível."""
        try:
            response = await self._client.post(
                f"{self.base_url}/api/show", json={"model": self.model}, timeout=5.0
            )
        except httpx.TransportError:
            return None
        if response.status_code >= 400:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        caps = data.get("capabilities")
        return caps if isinstance(caps, list) else None

    async def supports_native_tools(self) -> bool:
        """Tool calling nativo do modelo ativo: perfil explícito (`capabilities["tools"]`) tem
        precedência; sem ele, consulta os metadados do Ollama. Quando os metadados estão
        indisponíveis (Ollama fora ou modelo ausente): False conservador — evita HTTP 400
        por enviar `tools` a um modelo sem suporte."""
        explicit = self.capabilities.get("tools")
        if explicit is not None:
            return bool(explicit)
        if self._native_tools is None:
            caps = await self._show_capabilities()
            if caps is None:
                return False
            self._native_tools = "tools" in caps
        return self._native_tools

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get(f"{self.base_url}/api/tags", timeout=3.0)
            response.raise_for_status()
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            return {"available": False, "detail": f"{exc.__class__.__name__}: {exc}", "models": []}
        models = [m.get("name", "") for m in response.json().get("models", [])]
        return {"available": True, "detail": None, "models": models}

    async def loaded_models(self) -> list[dict[str, Any]]:
        """Modelos em memória (GET /api/ps) com consumo real de VRAM quando em GPU."""
        try:
            response = await self._client.get(f"{self.base_url}/api/ps", timeout=3.0)
            response.raise_for_status()
        except (httpx.TransportError, httpx.HTTPStatusError):
            return []
        loaded = []
        for entry in response.json().get("models", []):
            size = entry.get("size")
            vram = entry.get("size_vram")
            loaded.append(
                {
                    "name": entry.get("name", ""),
                    "size_mb": round(size / 1048576) if size else None,
                    "vram_mb": round(vram / 1048576) if vram else 0,
                    "expires_at": entry.get("expires_at"),
                }
            )
        return loaded

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.status_code == httpx.codes.NOT_FOUND:
            try:
                detail = response.json().get("error", "")
            except Exception:
                detail = response.text[:200]
            raise ProviderError(
                f"Modelo não encontrado no Ollama ({detail or 'HTTP 404'}). Instale com: ollama pull <modelo>"
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"Ollama respondeu HTTP {response.status_code}: {redact_secrets(response.text[:200])}"
            )

    @staticmethod
    def _usage_of(data: dict[str, Any]) -> tuple[dict[str, int] | None, str | None]:
        reason = data.get("done_reason")
        prompt_tokens = data.get("prompt_eval_count")
        eval_tokens = data.get("eval_count")
        usage = None
        if prompt_tokens is not None or eval_tokens is not None:
            usage = {"prompt_eval_count": int(prompt_tokens or 0), "eval_count": int(eval_tokens or 0)}
        return usage, (str(reason) if reason else None)

    def _capture_usage(self, data: dict[str, Any]) -> None:
        usage, self.last_finish_reason = self._usage_of(data)
        if usage is not None:
            self.last_usage = usage

    async def aclose(self) -> None:
        await self._client.aclose()

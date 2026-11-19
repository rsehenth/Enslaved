"""API local OpenAI-compatible (/v1) para IDEs e CLIs (OpenCode, Continue, Aider, curl).

- Autenticação obrigatória por chave `ens_…` (Bearer) validada ANTES de qualquer uso do Ollama.
- Stateless: usa apenas as mensagens do cliente — sem histórico, memórias, skills, ferramentas
  privadas ou persistência em conversas do Enslaved. A interface continua usando seus canais.
- Erros no formato OpenAI ({"error": {message, type, code}}). Ferramentas da IDE NUNCA são
  executadas no servidor: a API só transporta tool_calls; quem executa é o cliente.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from contextlib import aclosing
from typing import Any, AsyncIterator

import anyio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from backend.localapi.keys import ApiKeyManager
from backend.memory.context_manager import ContextManager
from backend.providers.base import ChatTurn, ProviderError, ProviderUnavailableError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["openai-v1"])

_SYSTEM_ROLES = {"system", "developer"}


class V1Error(Exception):
    """Erro da API /v1 no formato OpenAI (convertido em JSONResponse por `v1_error_handler`)."""

    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error", code: str | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type
        self.code = code


async def v1_error_handler(request: Request, exc: V1Error) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status,
        content={"error": {"message": exc.message, "type": exc.err_type, "code": exc.code}},
    )


# ------------------------------------------------------------------ autenticação


def require_key(request: Request) -> dict:
    if not request.app.state.local_api_enabled:
        raise V1Error(
            503,
            "A API local está desativada. Ative em Configurações > API Local do Enslaved.",
            err_type="server_error",
            code="api_disabled",
        )
    header = request.headers.get("authorization") or ""
    scheme, _, secret = header.partition(" ")
    if scheme.lower() != "bearer" or not secret.strip():
        raise V1Error(
            401,
            "Chave de API ausente. Envie o header 'Authorization: Bearer ens_…'.",
            err_type="authentication_error",
            code="missing_api_key",
        )
    key = request.app.state.api_keys.authenticate(secret.strip())
    if key is None:
        raise V1Error(
            401,
            "Chave de API inválida ou revogada.",
            err_type="authentication_error",
            code="invalid_api_key",
        )
    return key


# ------------------------------------------------------------------ conversão OpenAI -> Ollama


def _text_of(content: Any, *, role: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, dict) and part.get("type") == "image_url":
                raise V1Error(400, "Partes de imagem não são suportadas na API /v1 do MVP (somente texto).")
            else:
                raise V1Error(400, f"Mensagem {role!r} com parte de conteúdo não suportada.")
        return "\n".join(parts)
    raise V1Error(400, f"Conteúdo inválido na mensagem {role!r}.")


def convert_messages(raw_messages: list[dict]) -> list[dict]:
    """OpenAI -> Ollama: developer=system, tool_calls/role:tool preservando ordem e ids."""
    converted: list[dict] = []
    call_names: dict[str, str] = {}  # tool_call_id -> nome da função (da mensagem assistant anterior)
    for msg in raw_messages:
        if not isinstance(msg, dict):
            raise V1Error(400, "Cada item de 'messages' deve ser um objeto.")
        role = msg.get("role")
        if role in _SYSTEM_ROLES:
            converted.append({"role": "system", "content": _text_of(msg.get("content"), role=role)})
        elif role == "user":
            converted.append({"role": "user", "content": _text_of(msg.get("content"), role="user")})
        elif role == "assistant":
            item: dict[str, Any] = {"role": "assistant", "content": _text_of(msg.get("content"), role="assistant")}
            raw_calls = msg.get("tool_calls")
            if raw_calls:
                calls = []
                if not isinstance(raw_calls, list):
                    raise V1Error(400, "'tool_calls' deve ser uma lista.")
                for call in raw_calls:
                    if not isinstance(call, dict) or call.get("type") not in (None, "function"):
                        raise V1Error(400, "tool_calls inválidos: cada item deve ter type='function'.")
                    function = call.get("function")
                    if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
                        raise V1Error(400, "tool_call sem 'function.name'.")
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments or "{}")
                        except ValueError:
                            raise V1Error(400, f"Argumentos JSON inválidos na tool_call {function['name']!r}.")
                    if arguments is None:
                        arguments = {}
                    if not isinstance(arguments, dict):
                        raise V1Error(400, f"Argumentos da tool_call {function['name']!r} devem ser um objeto JSON.")
                    call_id = call.get("id")
                    if isinstance(call_id, str) and call_id:
                        call_names[call_id] = function["name"]
                    calls.append({"function": {"name": function["name"], "arguments": arguments}})
                item["tool_calls"] = calls
            converted.append(item)
        elif role == "tool":
            item = {"role": "tool", "content": _text_of(msg.get("content"), role="tool")}
            call_id = msg.get("tool_call_id")
            name = call_names.get(call_id) if isinstance(call_id, str) else None
            if name:
                item["tool_name"] = name
            converted.append(item)
        else:
            raise V1Error(400, f"Papel de mensagem não suportado: {role!r}.")
    return converted


def normalize_tools(raw: Any) -> list[dict]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise V1Error(400, "'tools' deve ser uma lista não vazia de funções.")
    tools: list[dict] = []
    for tool in raw:
        if not isinstance(tool, dict) or tool.get("type") not in (None, "function"):
            raise V1Error(400, "Cada ferramenta deve ter o formato {type: 'function', function: {...}}.")
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip():
            raise V1Error(400, "Ferramenta sem 'function.name'.")
        parameters = function.get("parameters")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": function["name"],
                    "description": function.get("description") or "",
                    "parameters": parameters if isinstance(parameters, dict) else {"type": "object", "properties": {}},
                },
            }
        )
    return tools


def _optional_number(payload: dict, key: str, lo: float, hi: float) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise V1Error(400, f"Parâmetro {key!r} deve ser numérico.")
    if not lo <= float(value) <= hi:
        raise V1Error(400, f"Parâmetro {key!r} fora do intervalo permitido [{lo}, {hi}].")
    return float(value)


def _resolve_num_predict(payload: dict, budget: int) -> int:
    raw = payload.get("max_completion_tokens")
    if raw is None:
        raw = payload.get("max_tokens")
    if raw is None:
        return budget
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise V1Error(400, "'max_tokens'/'max_completion_tokens' deve ser um inteiro positivo.")
    if raw > budget:
        raise V1Error(
            400,
            f"'max_tokens' ({raw}) excede o orçamento de saída do modelo ativo ({budget} tokens).",
            code="output_budget_exceeded",
        )
    return raw


# ------------------------------------------------------------------ formato de resposta


def _completion_id() -> str:
    return f"chatcmpl-{secrets.token_hex(12)}"


def _usage(turn: ChatTurn | None, provider: Any) -> dict[str, int] | None:
    # Uso do próprio turno; provider.last_usage é compartilhado (UI e outras /v1) e só serve de fallback.
    usage = turn.usage if turn is not None and turn.usage is not None else provider.last_usage
    if not usage:
        return None
    prompt = int(usage.get("prompt_eval_count") or 0)
    completion = int(usage.get("eval_count") or 0)
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def _tool_calls_payload(turn: ChatTurn) -> list[dict]:
    calls = []
    for index, call in enumerate(turn.tool_calls):
        calls.append(
            {
                "id": call.id or f"call_{index}_{secrets.token_hex(6)}",
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments or "{}"},
            }
        )
    return calls


def _finish_reason(turn: ChatTurn, provider: Any) -> str:
    if turn.tool_calls:
        return "tool_calls"
    reason = turn.finish_reason if turn.finish_reason is not None else provider.last_finish_reason
    return "length" if reason == "length" else "stop"


def build_completion_response(turn: ChatTurn, provider: Any, model: str, completion_id: str) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": turn.content}
    calls = _tool_calls_payload(turn)
    if calls:
        message["tool_calls"] = calls
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": _finish_reason(turn, provider)}],
        "usage": _usage(turn, provider),
    }


def _chunk(completion_id: str, model: str, delta: dict, finish_reason: str | None) -> str:
    chunk: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _usage_chunk(completion_id: str, model: str, usage: dict | None) -> str:
    """Chunk final de stream_options.include_usage: choices vazio, como na API OpenAI."""
    chunk = {"id": completion_id, "object": "chat.completion.chunk", "created": int(time.time()),
             "model": model, "choices": [], "usage": usage}
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


# Margem sobre a estimativa por pré-tokens: cobre a pior subcontagem medida em conteúdo típico
# (JSON com acentos: est/real 0,935). O limite duro exato fica com o Ollama (truncate=False).
_V1_SAFETY_MARGIN = 0.06
_OLLAMA_OVERFLOW = "exceeds the available context size"


def _is_ollama_overflow(exc: ProviderError) -> bool:
    return _OLLAMA_OVERFLOW in str(exc)


# "maximum context length is N tokens" é o texto padrão que clientes (OpenCode etc.) reconhecem
# como estouro de contexto e que dispara a compactação automática do histórico.
def _estimated_overflow_message(v: Any) -> str:
    return (
        f"Contexto excedido: entrada estimada em {v.total_input_tokens} tokens (mensagens {v.messages_tokens} + "
        f"ferramentas {v.tools_tokens} + margem de segurança {v.safety_margin_tokens}) ultrapassa o orçamento de "
        f"entrada de {v.input_budget} tokens (num_ctx {v.num_ctx} - reserva de saída {v.output_reserve}). "
        f"This model's maximum context length is {v.num_ctx} tokens. Compacte o histórico ou reduza max_tokens."
    )


def _exact_overflow_message(exc: ProviderError, num_ctx: Any) -> str:
    return (
        f"Contexto excedido (contagem exata do Ollama): {exc}. "
        f"This model's maximum context length is {num_ctx} tokens. Compacte o histórico."
    )


def _sse_error(message: str, err_type: str = "server_error", code: str | None = None) -> str:
    event = {"error": {"message": message, "type": err_type, "code": code}}
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ------------------------------------------------------------------ endpoints


@router.get("/v1/models")
async def list_models(request: Request, _key: dict = Depends(require_key)) -> dict:
    provider = request.app.state.provider
    return {
        "object": "list",
        "data": [
            {
                "id": provider.model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "enslaved",
            }
        ],
    }


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, _key: dict = Depends(require_key)):
    settings = request.app.state.settings
    limit = settings.local_api.max_payload_bytes
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > limit:
        raise V1Error(413, "Payload excede o limite de tamanho permitido.", code="payload_too_large")
    body = bytearray()
    async for chunk in request.stream():  # conta o real: chunked não traz Content-Length
        body += chunk
        if len(body) > limit:
            raise V1Error(413, "Payload excede o limite de tamanho permitido.", code="payload_too_large")
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        raise V1Error(400, "Corpo da requisição não é JSON válido.")
    if not isinstance(payload, dict):
        raise V1Error(400, "Corpo da requisição deve ser um objeto JSON.")

    provider = request.app.state.provider
    profile = settings.active_profile()

    model = payload.get("model")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise V1Error(400, "Campo 'model' deve ser uma string.")
    if isinstance(model, str) and model.strip() != provider.model:
        raise V1Error(
            404,
            f"Modelo {model.strip()!r} não disponível. Modelo ativo: {provider.model}",
            code="model_not_found",
        )

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise V1Error(400, "'messages' deve ser uma lista com ao menos uma mensagem.")
    if len(raw_messages) > settings.local_api.max_messages:
        raise V1Error(
            400,
            f"Número de mensagens ({len(raw_messages)}) excede o limite ({settings.local_api.max_messages}).",
        )
    messages = convert_messages(raw_messages)

    tools = normalize_tools(payload.get("tools"))
    tool_choice = payload.get("tool_choice")
    if tool_choice == "none":
        tools = []
    elif tool_choice not in (None, "auto"):
        raise V1Error(
            400,
            "Apenas tool_choice 'auto' e 'none' são suportados (Ollama não implementa 'required' nem seleção por função).",
            code="tool_choice_unsupported",
        )
    if tools and not await provider.supports_native_tools():
        raise V1Error(400, "O modelo ativo não suporta tool calling nativo.", code="tools_unsupported")

    n = payload.get("n")
    if n is not None and n != 1:
        raise V1Error(400, "Apenas n=1 é suportado.")

    temperature = _optional_number(payload, "temperature", 0.0, 2.0)
    top_p = _optional_number(payload, "top_p", 0.0, 1.0)
    num_predict = _resolve_num_predict(payload, profile.num_predict)

    stream = payload.get("stream")
    if stream is None:
        stream = False
    if not isinstance(stream, bool):
        raise V1Error(400, "'stream' deve ser booleano.")
    include_usage = False
    stream_options = payload.get("stream_options")
    if stream_options is not None:
        if not isinstance(stream_options, dict):
            raise V1Error(400, "'stream_options' deve ser um objeto.")
        include_usage = bool(stream_options.get("include_usage"))

    context = ContextManager(
        num_ctx=profile.num_ctx, output_reserve=num_predict, safety_margin_ratio=_V1_SAFETY_MARGIN, pretoken=True
    )
    validation = context.validate_payload(messages, tools=tools or None, num_predict=num_predict)
    logger.info(
        "v1 budget: messages=%d tools=%d margin=%d input=%d/%d reserve=%d num_ctx=%d fits=%s",
        validation.messages_tokens, validation.tools_tokens, validation.safety_margin_tokens,
        validation.total_input_tokens, validation.input_budget, validation.output_reserve, validation.num_ctx, validation.fits,
    )
    if not validation.fits:
        raise V1Error(400, _estimated_overflow_message(validation), code="context_overflow")

    # truncate/shift=False: o Ollama (tokenizer e template reais) recusa prompt > num_ctx em vez de
    # truncar em silêncio, e não descarta o início do contexto se a geração encostar no limite.
    options: dict[str, Any] = {"num_ctx": profile.num_ctx, "num_predict": num_predict, "truncate": False, "shift": False}
    if temperature is not None:
        options["temperature"] = temperature
    if top_p is not None:
        options["top_p"] = top_p

    # Concorrência/fila: adquire antes de qualquer chamada ao Ollama. A vaga é liberada uma
    # única vez por _Slot: no fim do non-stream, ou pela resposta SSE em qualquer desfecho.
    semaphore: asyncio.Semaphore = request.app.state.local_api_semaphore
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=settings.local_api.queue_timeout_seconds)
    except asyncio.TimeoutError:
        raise V1Error(
            429,
            f"Limite de concorrência atingido ({settings.local_api.max_concurrent} gerações simultâneas). Tente novamente.",
            err_type="rate_limit_error",
            code="busy",
        )

    slot = _Slot(semaphore)
    completion_id = _completion_id()
    if not stream:
        try:
            try:
                turn = await provider.chat_turn(messages, tools=tools or None, **options)
            except ProviderUnavailableError as exc:
                raise V1Error(503, str(exc), err_type="server_error", code="provider_unavailable")
            except ProviderError as exc:
                if _is_ollama_overflow(exc):
                    raise V1Error(400, _exact_overflow_message(exc, profile.num_ctx), code="context_overflow")
                raise V1Error(502, str(exc), err_type="server_error", code="provider_error")
            return build_completion_response(turn, provider, provider.model, completion_id)
        finally:
            slot.release()

    try:
        return SlotStreamingResponse(
            _stream_sse(request.app.state, messages, tools, options, provider.model, completion_id, include_usage, slot),
            slot,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )
    except BaseException:
        slot.release()
        raise


class _Slot:
    """Vaga de geração /v1: liberada exatamente uma vez, venha de onde vier a liberação."""

    def __init__(self, semaphore: asyncio.Semaphore):
        self._semaphore = semaphore
        self._held = True

    def release(self) -> None:
        if self._held:
            self._held = False
            self._semaphore.release()


class SlotStreamingResponse(StreamingResponse):
    """SSE que devolve a vaga ao terminar em QUALQUER desfecho: fim normal, erro, cancelamento ou
    desconexão — inclusive antes da 1ª iteração do gerador, quando o finally dele nunca roda
    (ex.: cliente desistiu enquanto esperava na fila e o Starlette cancela o envio de imediato)."""

    def __init__(self, content: AsyncIterator[str], slot: _Slot, **kwargs: Any):
        super().__init__(content, **kwargs)
        self.slot = slot

    async def __call__(self, scope, receive, send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            try:
                # Fecha já o gerador (e o stream do Ollama via aclosing), mesmo se a task foi cancelada.
                with anyio.CancelScope(shield=True):
                    await self.body_iterator.aclose()
            finally:
                self.slot.release()


async def _stream_sse(
    state: Any,
    messages: list[dict],
    tools: list[dict],
    options: dict[str, Any],
    model: str,
    completion_id: str,
    include_usage: bool,
    slot: _Slot,
) -> AsyncIterator[str]:
    provider = state.provider
    final_turn: ChatTurn | None = None
    errored = False
    try:
        # Libera a vaga assim que o provider termina; SlotStreamingResponse cobre o gerador nunca iniciado.
        yield _chunk(completion_id, model, {"role": "assistant", "content": ""}, None)
        try:
            async with aclosing(provider.stream_turn(messages, tools=tools or None, **options)) as flow:
                async for item in flow:
                    if isinstance(item, ChatTurn):
                        final_turn = item
                    elif isinstance(item, str) and item:
                        yield _chunk(completion_id, model, {"content": item}, None)
                    # Thinking (raciocínio) não é exposto na API externa.
        except ProviderUnavailableError as exc:
            errored = True
            yield _sse_error(str(exc), code="provider_unavailable")
        except ProviderError as exc:
            errored = True
            if _is_ollama_overflow(exc):
                yield _sse_error(_exact_overflow_message(exc, options.get("num_ctx")), "invalid_request_error", "context_overflow")
            else:
                yield _sse_error(str(exc), code="provider_error")
        except Exception as exc:  # falha inesperada: sem traceback no cliente
            errored = True
            logger.error("v1 stream error: %s: %s", exc.__class__.__name__, exc)
            yield _sse_error("Erro inesperado durante a geração.", code="internal_error")

        if final_turn is not None:
            for index, call in enumerate(_tool_calls_payload(final_turn)):
                yield _chunk(
                    completion_id,
                    model,
                    {"tool_calls": [{"index": index, "id": call["id"], "type": "function", "function": {"name": call["function"]["name"], "arguments": ""}}]},
                    None,
                )
                yield _chunk(completion_id, model, {"tool_calls": [{"index": index, "function": {"arguments": call["function"]["arguments"]}}]}, None)
            yield _chunk(completion_id, model, {}, _finish_reason(final_turn, provider))
    finally:
        slot.release()
    usage = _usage(final_turn, provider) if include_usage and final_turn is not None else None
    if usage is not None:
        yield _usage_chunk(completion_id, model, usage)
    if not errored:
        yield "data: [DONE]\n\n"

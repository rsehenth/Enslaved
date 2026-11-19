"""Motor de execução de ferramentas: registro + schema + permissões + supervisão.

Nenhuma ferramenta executa código arbitrário recebido da LLM: argumentos são
validados contra o schema declarado (`args_model`) e a execução é sempre o
método `run()` de uma classe registrada, com política de permissões, timeout,
cancelamento e trilha de auditoria.
"""

import asyncio
import json
import os
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlsplit

from pydantic import BaseModel, ValidationError

from backend.core.config import ToolsSettings
from backend.core.redact import redact_secrets
from backend.tools.base import PermissionDecision, Tool, ToolResult
from backend.tools.net_guard import egress_details
from backend.tools.permissions import ANY_TARGET, PermissionManager, RESEARCH_TARGET
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry


@dataclass
class PreparedToolCall:
    """Resultado de `prepare`: o que foi validado e o que falta decidir."""

    tool: Tool | None
    args: dict[str, Any]
    state: str  # ready | needs_permission | invalid | blocked
    error: str | None = None
    request: Any = None  # PermissionRequest quando state == needs_permission
    target: str = ANY_TARGET  # destino/escopo ao qual uma concessão "Lembrar" fica presa
    research_covered: bool = False  # URL supervisionada do fluxo de pesquisa autorizado desta conversa


def _args_summary(args: dict, limit: int = 200) -> str:
    try:
        text = json.dumps(args, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = str(args)
    text = redact_secrets(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _validation_message(exc: ValidationError, limit: int = 400) -> str:
    parts = []
    for error in exc.errors()[:3]:
        location = ".".join(str(item) for item in error.get("loc", ())) or "(raiz)"
        parts.append(f"{location}: {error.get('msg', 'inválido')}")
    message = "; ".join(parts)
    return message if len(message) <= limit else message[: limit - 1] + "…"


class ToolEngine:
    """Executa ferramentas registradas sob supervisão (permissão, timeout, auditoria)."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy,
        permissions: PermissionManager,
        audit: AuditLog,
        settings: ToolsSettings,
    ):
        self._registry = registry
        self._policy = policy
        self._permissions = permissions
        self._audit = audit
        self.settings = settings
        self._pending_invocations: dict[str, tuple[PreparedToolCall, str, str | None]] = {}
        # URLs supervisionadas do fluxo de pesquisa pública, por conversa: resultados reais de
        # busca e URLs fornecidas pelo proprietário. Só elas dispensam cartões por domínio.
        self._research_urls: dict[str | None, set[str]] = {}

    # -------------------------------------------------- fluxo de pesquisa supervisionado

    # Caracteres reservados de URL: se decodificar %XX os introduzir (%2F, %26, %3D…), a forma
    # decodificada NÃO é equivalente à original (mudaria a estrutura do caminho/query) e não é usada.
    _RESERVED = ("/", "?", "&", "=", "#")

    @classmethod
    def _soft_unquote(cls, value: str) -> str:
        decoded = unquote(value)
        if decoded != value and any(decoded.count(c) > value.count(c) for c in cls._RESERVED):
            return value
        return unicodedata.normalize("NFC", decoded)

    @classmethod
    def _url_key(cls, url: str) -> str:
        """Chave de comparação de URLs: normaliza codificação (%XX ↔ caractere, sem decodificar
        %2F/%26/%3D, que mudariam o destino), NFC e caixa do host — o modelo pode reproduzir a URL
        do resultado de busca em outra forma equivalente; o destino continua sendo a URL registrada.
        Não afina URLs de hosts ou caminhos diferentes."""
        try:
            parts = urlsplit(url.strip())
        except ValueError:
            return ""
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return ""
        host = parts.hostname.lower().rstrip(".")
        path = cls._soft_unquote(parts.path or "/").rstrip("/") or "/"
        query = "&".join(sorted(filter(None, (cls._soft_unquote(p) for p in parts.query.split("&")))))
        return f"{host}{path}?{query}" if query else f"{host}{path}"

    def register_research_urls(self, conversation_id: str | None, urls) -> None:
        """Registra URLs legítimas do fluxo de pesquisa (resultados de busca ou URLs do proprietário)."""
        bucket = self._research_urls.setdefault(conversation_id, set())
        for url in urls:
            if isinstance(url, str) and (key := self._url_key(url)):
                bucket.add(key)

    def research_url_registered(self, conversation_id: str | None, url: str) -> bool:
        key = self._url_key(url)
        return bool(key) and key in self._research_urls.get(conversation_id, set())

    def forget_conversation_research(self, conversation_id: str | None) -> None:
        self._research_urls.pop(conversation_id, None)

    # ---------------------------------------------------------------- consulta

    def list_tools(self) -> list[Tool]:
        return self._registry.list_tools()

    def get(self, tool_id: str) -> Tool | None:
        return self._registry.get(tool_id)

    def effective_mode(self, tool: Tool) -> str:
        """Modo efetivo de uma ferramenta: deny > ask > allow sobre suas capacidades."""
        modes = [self._policy.mode_for(capability) for capability in tool.spec.required_permissions]
        if "deny" in modes:
            return "deny"
        if "ask" in modes:
            return "ask"
        return "allow"

    def web_state(self, conversation_id: str | None = None) -> str:
        """Pesquisa externa do chat: 'allow' (política ou concessão de pesquisa desta conversa) | 'ask' | 'deny'.

        A concessão da pesquisa fica presa ao alvo "search": não autoriza net.fetch do modelo para outras URLs."""
        if self._permissions.is_granted(["net.fetch"], conversation_id, RESEARCH_TARGET):
            return "allow"
        return "deny" if self._permissions.denied_capabilities(["net.fetch"]) else "ask"

    def host_state(self, conversation_id: str | None, host: str) -> str:
        """Acesso a um host-alvo: 'allow' (política ou concessão "Lembrar"; apex e www contam como o mesmo alvo) | 'ask' | 'deny'."""
        bare = host.lower().removeprefix("www.")
        if any(self._permissions.is_granted(["net.fetch"], conversation_id, h) for h in (host.lower(), bare, "www." + bare)):
            return "allow"
        return "deny" if self._permissions.denied_capabilities(["net.fetch"]) else "ask"

    def request_research_permission(self, host: str, level: str, limits: str, conversation_id: str):
        return self._permissions.create_request(
            tool_id="pesquisa.profunda", action="investigar", scope={"consulta": host, "alvo": host, "nível": level, "limites": limits},
            reason="Inspeção passiva e somente leitura do alvo (net.inspect: sem redes privadas, sem testes ativos). A profundidade define o esforço; "
                   "a autorização vale só para este destino e esta investigação.",
            capabilities=["net.fetch"], source="chat", conversation_id=conversation_id, target=host.lower(),
        )

    def deny_pending(self, request) -> None:
        """Descarta um pedido criado por engano (a investigação nunca amplia a autorização silenciosamente)."""
        self._permissions.resolve(request.id, granted=False, decided_by="research")

    def request_web_permission(self, query: str, conversation_id: str):
        return self._permissions.create_request(
            tool_id="verificacao.web", action="pesquisar", scope={"consulta": query},
            reason="Autorização de pesquisa pública nesta conversa: só o texto da consulta vai ao mecanismo de busca; "
                   "páginas dos resultados podem ser abertas (net.fetch: somente leitura, sem redes privadas). "
                   "Vale para as próximas perguntas desta conversa até ser revogada.",
            capabilities=["net.fetch"], source="chat", conversation_id=conversation_id, target=RESEARCH_TARGET,
        )

    def config_warnings(self) -> list[str]:
        """Avisos de configuração que reduzem a proteção de saída de rede (exibidos em /api/tools e no log)."""
        warnings = []
        if self._policy.mode_for("net.fetch") == "allow":
            warnings.append("net.fetch em modo 'allow': o modelo busca URLs sem cartão quando o contexto está limpo.")
        if self.settings.unsafe_allow_untrusted_egress:
            warnings.append(
                "unsafe_allow_untrusted_egress ATIVO: com 'allow', páginas/documentos lidos pelo modelo podem "
                "fazê-lo enviar dados para qualquer URL sem cartão (exfiltração por prompt injection)."
            )
        return warnings

    def describe(self) -> list[dict[str, Any]]:
        descriptions = []
        for tool in sorted(self.list_tools(), key=lambda item: item.spec.id):
            spec = tool.spec
            model = getattr(tool, "args_model", None)
            descriptions.append(
                {
                    "id": spec.id,
                    "name": spec.name,
                    "description": spec.description,
                    "capabilities": spec.required_permissions,
                    "dangerous": spec.dangerous,
                    "mode": self.effective_mode(tool),
                    "parameters": model.model_json_schema() if model else {"type": "object", "properties": {}},
                }
            )
        return descriptions

    def ollama_tools(self) -> list[dict[str, Any]]:
        """Formato `tools` da API /api/chat do Ollama (estilo OpenAI)."""
        tools = []
        for tool in self.list_tools():
            model = getattr(tool, "args_model", None)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.spec.id,
                        "description": tool.spec.description,
                        "parameters": model.model_json_schema() if model else {"type": "object", "properties": {}},
                    },
                }
            )
        return tools

    # ---------------------------------------------------------------- execução

    async def invoke(
        self,
        tool_id: str,
        args: dict[str, Any] | None = None,
        *,
        source: str = "chat",
        conversation_id: str | None = None,
        decision: PermissionDecision | None = None,
    ) -> ToolResult:
        """Execução completa: valida, resolve permissão (aguardando quando preciso) e roda."""
        prepared = await self.prepare(tool_id, args, source=source, conversation_id=conversation_id)
        if prepared.state == "invalid":
            return ToolResult(ok=False, error=prepared.error)
        if prepared.state == "blocked":
            return ToolResult(ok=False, error=prepared.error)

        effective_decision = decision
        if prepared.state == "needs_permission":
            request = prepared.request
            self._audit.append(
                {
                    "event": "permission_requested",
                    "tool": tool_id,
                    "request_id": request.id,
                    "source": source,
                    "conversation_id": conversation_id,
                }
            )
            effective_decision = await self.wait_permission(request.id)
            self._audit.append(
                {
                    "event": "permission_resolved",
                    "tool": tool_id,
                    "request_id": request.id,
                    "granted": bool(effective_decision and effective_decision.granted),
                    "decided_by": effective_decision.decided_by if effective_decision else "unknown",
                    "source": source,
                    "conversation_id": conversation_id,
                }
            )
            if effective_decision is None or not effective_decision.granted:
                return ToolResult(ok=False, error="Autorização negada ou expirada pelo proprietário.")
        return await self.execute(prepared, effective_decision, source=source, conversation_id=conversation_id)

    async def prepare(
        self,
        tool_id: str,
        args: dict[str, Any] | None = None,
        *,
        source: str = "chat",
        conversation_id: str | None = None,
        untrusted_context: bool = False,
        preauthorized_target: str | None = None,
    ) -> "PreparedToolCall":
        """Valida argumentos e política. Cria pedido de permissão quando necessário.

        Estados: ready (pode executar), needs_permission (pedido criado, aguardando
        decisão), invalid (argumentos/ferramenta), blocked (política nega).

        `untrusted_context`: o contexto do modelo contém conteúdo externo/privado (páginas,
        arquivos, resultados de ferramentas). Aí a concessão de sessão não cobre saída de
        dados para a rede: cada chamada net.* volta a exigir o cartão (contra exfiltração
        por prompt injection). Só o modo "allow" explícito da política dispensa o cartão.

        `preauthorized_target`: o proprietário aprovou ESTE destino para a tarefa em curso (cartão da investigação).
        Só dispensa o novo cartão quando o destino da chamada é exatamente esse; outro destino continua exigindo cartão.
        """
        args = dict(args or {})
        tool = self._registry.get(tool_id)
        if tool is None:
            return PreparedToolCall(tool=None, args=args, state="invalid", error=f"Ferramenta desconhecida: {tool_id!r}")

        run_args: dict[str, Any] = args
        model = getattr(tool, "args_model", None)
        if model is not None:
            try:
                parsed = model.model_validate(args)
            except ValidationError as exc:
                message = _validation_message(exc)
                self._audit.append(
                    {
                        "event": "tool_invalid_args",
                        "tool": tool_id,
                        "source": source,
                        "conversation_id": conversation_id,
                        "error": message,
                    }
                )
                return PreparedToolCall(
                    tool=tool, args=args, state="invalid", error=f"Argumentos inválidos para {tool_id}: {message}"
                )
            run_args = parsed.model_dump()

        blocked = self._permissions.denied_capabilities(tool.spec.required_permissions)
        if blocked:
            self._audit.append(
                {
                    "event": "permission_denied_policy",
                    "tool": tool_id,
                    "capabilities": blocked,
                    "source": source,
                    "conversation_id": conversation_id,
                }
            )
            return PreparedToolCall(
                tool=tool,
                args=run_args,
                state="blocked",
                error=f"Execução bloqueada pela política de permissões: {', '.join(blocked)}",
            )

        caps = tool.spec.required_permissions
        target = _grant_target(tool.spec.id, run_args)
        reason = tool.spec.description
        egress = any(c.startswith("net.") for c in caps)
        sensitive: list[str] = []
        if egress and isinstance(run_args.get("url"), str):
            host, params, sensitive = egress_details(run_args["url"])
            reason = _egress_reason(host, run_args["url"], params, sensitive, untrusted_context, tool.spec.description)
        # Sem o opt-in explícito do proprietário, nem "allow" nem concessão antiga liberam saída de rede
        # quando o contexto tem conteúdo externo ou quando a URL parece carregar dados sensíveis.
        unsafe_opt_in = self.settings.unsafe_allow_untrusted_egress and all(
            self._permissions.mode_for(c) == "allow" for c in caps if c.startswith("net.")
        )
        egress_needs_card = egress and (untrusted_context or bool(sensitive)) and not unsafe_opt_in
        covered = preauthorized_target is not None and preauthorized_target.lower() == target.lower()
        # URL supervisionada do fluxo de pesquisa autorizado nesta conversa (resultado real de busca ou
        # URL do proprietário, sem parâmetros sensíveis): cartão por domínio não é exibido de novo.
        # Conteúdo externo NÃO cria destinos novos: só URLs exatas registradas ANTES da leitura; SSRF,
        # limites e bloqueio de conteúdo continuam valendo dentro da ferramenta.
        research_covered = (
            tool.spec.id == "net.fetch"
            and egress
            and not sensitive
            and conversation_id is not None
            and self._permissions.has_research_grant(conversation_id)
            and self.research_url_registered(conversation_id, run_args["url"])
        )
        if caps and not covered and not research_covered and (egress_needs_card or not self._permissions.is_granted(caps, conversation_id, target)):
            request = self._permissions.create_request(
                tool_id=tool.spec.id,
                action="executar",
                scope=_permission_scope(run_args),
                reason=reason,
                capabilities=tool.spec.required_permissions,
                source=source,
                conversation_id=conversation_id,
                target=target,
            )
            return PreparedToolCall(tool=tool, args=run_args, state="needs_permission", request=request, target=target)

        return PreparedToolCall(tool=tool, args=run_args, state="ready", target=target, research_covered=research_covered)

    async def wait_permission(self, request_id: str, timeout_seconds: float | None = None) -> PermissionDecision | None:
        """Aguarda a decisão do proprietário sobre um pedido (com timeout → negado)."""
        timeout = self.settings.permission_wait_seconds if timeout_seconds is None else timeout_seconds
        return await self._permissions.wait(request_id, timeout)

    def record_denial(self, prepared: "PreparedToolCall", *, decided_by: str = "owner", conversation_id: str | None = None) -> None:
        """Audita uma negativa de permissão (a ferramenta NÃO executou)."""
        self._audit.append(
            {
                "event": "tool_denied",
                "tool": prepared.tool.spec.id if prepared.tool else None,
                "decided_by": decided_by,
                "conversation_id": conversation_id,
            }
        )

    # ------------------------------------------- invocações pendentes de decisão

    def cancel_conversation(self, conversation_id: str) -> None:
        """Cancela pedidos pendentes, invocações aguardando decisão e concessões da conversa."""
        cancelled_ids = set(self._permissions.cancel_conversation(conversation_id))
        self.forget_conversation_research(conversation_id)
        for rid in list(self._pending_invocations):
            entry = self._pending_invocations.get(rid)
            if rid in cancelled_ids or (entry is not None and entry[2] == conversation_id):
                self._pending_invocations.pop(rid, None)

    def remember_invocation(self, prepared: "PreparedToolCall", source: str = "manual", conversation_id: str | None = None) -> None:
        """Guarda uma chamada preparada aguardando decisão (chat ou REST).

        A execução acontece em `complete_invocation` quando o proprietário decide;
        a conversa de origem é lembrada para persistir o resultado como mensagem.
        """
        if prepared.request is not None:
            # descarta invocações cujo pedido expirou ou foi resolvido por outro caminho (sem replay tardio)
            self._pending_invocations = {
                rid: entry for rid, entry in self._pending_invocations.items() if self._permissions.is_pending(rid)
            }
            self._pending_invocations[prepared.request.id] = (prepared, source, conversation_id)

    async def complete_invocation(self, request_id: str, decision: PermissionDecision | None) -> ToolResult | None:
        """Executa (ou nega) uma invocação pendente. None se o id é desconhecido.

        Quando a invocação veio de uma conversa, `result.data["conversation_id"]`
        permite à camada de API persistir o resultado como mensagem `tool`.
        """
        entry = self._pending_invocations.pop(request_id, None)
        if entry is None:
            return None
        prepared, source, conversation_id = entry
        if decision is None or not decision.granted:
            self.record_denial(prepared, decided_by=decision.decided_by if decision else "unknown", conversation_id=conversation_id)
            return ToolResult(
                ok=False,
                error="Autorização negada ou expirada pelo proprietário.",
                data={
                    "tool": prepared.tool.spec.id if prepared.tool else None,
                    "request_id": request_id,
                    "conversation_id": conversation_id,
                },
            )
        result = await self.execute(prepared, decision, source=source, conversation_id=conversation_id)
        if conversation_id:
            result.data = {**(result.data or {}), "conversation_id": conversation_id}
        return result

    async def execute(
        self,
        prepared: "PreparedToolCall",
        decision: PermissionDecision | None = None,
        *,
        source: str = "chat",
        conversation_id: str | None = None,
    ) -> ToolResult:
        """Executa uma chamada preparada com timeout, cancelamento e auditoria."""
        tool = prepared.tool
        if prepared.state not in ("ready", "needs_permission") or tool is None:
            return ToolResult(ok=False, error=prepared.error or "Chamada não preparada")

        if tool.spec.required_permissions:
            if decision is None:
                if prepared.research_covered or self._permissions.is_granted(tool.spec.required_permissions, conversation_id, prepared.target):
                    # Concessão automática: política (modo allow) ou URL supervisionada do fluxo de
                    # pesquisa autorizado — materializada como decisão explícita para a ferramenta validar o escopo.
                    decision = PermissionDecision(granted=True, decided_by="research" if prepared.research_covered else "policy")
                else:
                    return ToolResult(ok=False, error="Autorização pendente do proprietário.")
            if not decision.granted:
                return ToolResult(ok=False, error="Autorização negada pelo proprietário.")

        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                tool.run(prepared.args, decision),
                timeout=self.settings.tool_timeout_seconds,
            )
        except asyncio.TimeoutError:
            result = ToolResult(ok=False, error=f"Tempo limite da ferramenta excedido ({self.settings.tool_timeout_seconds:g}s).")
        except asyncio.CancelledError:
            self._audit.append(
                {
                    "event": "tool_cancelled",
                    "tool": tool.spec.id,
                    "source": source,
                    "conversation_id": conversation_id,
                }
            )
            raise
        except Exception as exc:  # ferramenta não pode derrubar a aplicação
            result = ToolResult(ok=False, error=f"Erro interno da ferramenta: {exc.__class__.__name__}: {exc}")
        duration_ms = round((time.perf_counter() - started) * 1000, 1)

        result.data = {
            **(result.data or {}),
            "tool": tool.spec.id,
            "duration_ms": duration_ms,
        }
        self._audit.append(
            {
                "event": "tool_run",
                "tool": tool.spec.id,
                "ok": result.ok,
                "error": redact_secrets(result.error) if result.error else None,
                "duration_ms": duration_ms,
                "source": source,
                "conversation_id": conversation_id,
                "args_summary": _args_summary(prepared.args),
            }
        )
        return result


def _grant_target(tool_id: str, args: dict[str, Any]) -> str:
    """Escopo de uma concessão "Lembrar": domínio para rede, pasta para arquivos; demais ferramentas, qualquer alvo."""
    if isinstance(args.get("url"), str):
        return urlsplit(args["url"]).hostname or "?"
    if tool_id == "fs.read" and isinstance(args.get("path"), str):
        return os.path.normcase(os.path.normpath(os.path.dirname(args["path"].replace("\\", "/")) or "."))
    return ANY_TARGET


def _egress_reason(host: str, url: str, params: list[str], sensitive: list[str], untrusted: bool, description: str) -> str:
    """Texto do cartão de saída de rede: destino, URL completa, parâmetros e alertas."""
    lines = [f"Destino: {host}", f"URL completa: {url}"]
    if params:
        lines.append("Parâmetros enviados: " + "; ".join(params[:12]) + (" …" if len(params) > 12 else ""))
    if sensitive:
        lines.append("⚠ Possíveis dados sensíveis na URL: " + "; ".join(sensitive[:6]))
    if untrusted:
        lines.append(
            "⚠ Pedido feito depois que o modelo leu conteúdo externo (página, arquivo ou ferramenta): "
            "aprove só se você reconhece este destino."
        )
    lines.append(description)
    return "\n".join(lines)


def _permission_scope(args: dict[str, Any], limit: int = 400) -> dict[str, Any]:
    """Escopo exibido no pedido de permissão (resumido; sem valores enormes)."""
    scope: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, str) and len(value) > limit:
            scope[key] = value[: limit - 1] + "…"
        else:
            scope[key] = value
    return scope

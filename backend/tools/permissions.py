"""Gerenciamento de permissões de ferramentas (decisão explícita do proprietário)."""

import asyncio
import time
import uuid
from typing import Callable

from backend.tools.base import PermissionDecision, PermissionRequest
from backend.tools.policy import ToolPolicy

ANY_TARGET = "*"
RESEARCH_TARGET = "search"  # alvo da autorização de pesquisa pública da conversa (não autoriza net.fetch avulso)
GRANT_TTL_SECONDS = 30 * 60  # "Lembrar" vale 30 min
REQUEST_TTL_SECONDS = 15 * 60  # cartão não decidido em 15 min expira (não pode ser aprovado depois)


class PermissionManager:
    """Estados em memória (processo): pedidos pendentes e concessões temporárias.

    - Capacidades em modo "allow": concedidas por política, sem pedido.
    - Capacidades em modo "ask": exigem pedido; o motor aguarda a decisão via
      `wait()` enquanto a interface apresenta o pedido ao proprietário.
    - "Lembrar" cria uma concessão presa a (conversa, capacidade, destino) com
      prazo de validade; não vale para outra conversa nem para outro destino.
    - A autorização de PESQUISA PÚBLICA (alvo "search") vale pela conversa inteira
      enquanto ela existir: sem prazo de 30 min, revogável explicitamente e sem
      transferência para outras conversas.
    """

    def __init__(
        self,
        policy: ToolPolicy,
        grant_ttl_seconds: float = GRANT_TTL_SECONDS,
        request_ttl_seconds: float = REQUEST_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._policy = policy
        self._grant_ttl = grant_ttl_seconds
        self._request_ttl = request_ttl_seconds
        self._clock = clock
        self._pending: dict[str, asyncio.Future[PermissionDecision]] = {}
        self._requests: dict[str, PermissionRequest] = {}
        self._created: dict[str, float] = {}
        self._targets: dict[str, str] = {}
        # (conversation_id, capability, target) -> instante de expiração
        self._grants: dict[tuple[str | None, str, str], float] = {}

    def mode_for(self, capability: str) -> str:
        return self._policy.mode_for(capability)

    def denied_capabilities(self, capabilities: list[str]) -> list[str]:
        return [c for c in capabilities if self._policy.mode_for(c) == "deny"]

    def _grant_active(self, conversation_id: str | None, capability: str, target: str) -> bool:
        now = self._clock()
        for key in ((conversation_id, capability, target), (conversation_id, capability, ANY_TARGET)):
            if key not in self._grants:
                continue
            expiry = self._grants[key]
            if expiry is not None and expiry <= now:  # expirada (None = vale enquanto a conversa existir)
                self._grants.pop(key, None)
                continue
            return True
        return False

    def has_research_grant(self, conversation_id: str | None) -> bool:
        """Autorização de pesquisa pública ativa para ESTA conversa (não se transfere)."""
        return conversation_id is not None and self._grant_active(conversation_id, "net.fetch", RESEARCH_TARGET)

    def grant_research(self, conversation_id: str | None) -> None:
        """Concede a autorização de pesquisa pública sem prazo: vale enquanto a conversa existir."""
        if conversation_id:
            self._grants[(conversation_id, "net.fetch", RESEARCH_TARGET)] = None

    def is_granted(self, capabilities: list[str], conversation_id: str | None = None, target: str = ANY_TARGET) -> bool:
        return all(
            self._policy.mode_for(c) == "allow" or self._grant_active(conversation_id, c, target) for c in capabilities
        )

    def create_request(
        self,
        tool_id: str,
        action: str,
        scope: dict | None = None,
        reason: str | None = None,
        capabilities: list[str] | None = None,
        source: str = "manual",
        conversation_id: str | None = None,
        target: str = ANY_TARGET,
    ) -> PermissionRequest:
        self._expire_stale()
        self._prune_completed()
        request = PermissionRequest(
            id=uuid.uuid4().hex,
            tool_id=tool_id,
            action=action,
            scope=dict(scope or {}),
            reason=reason,
            capabilities=list(capabilities or []),
            source=source,
            conversation_id=conversation_id,
        )
        self._requests[request.id] = request
        self._created[request.id] = self._clock()
        self._targets[request.id] = target
        self._pending[request.id] = asyncio.get_running_loop().create_future()
        return request

    def _expire_stale(self) -> None:
        """Pedidos não decididos dentro do prazo viram negados (não podem ser aprovados depois)."""
        limit = self._clock() - self._request_ttl
        for rid, fut in list(self._pending.items()):
            if not fut.done() and self._created.get(rid, limit) < limit:
                fut.set_result(PermissionDecision(granted=False, decided_by="expired", reason="Pedido expirado."))

    def _prune_completed(self, max_entries: int = 200) -> None:
        if len(self._requests) <= max_entries:
            return
        done_ids = [rid for rid, fut in self._pending.items() if fut.done()]
        for rid in done_ids:
            for store in (self._pending, self._requests, self._created, self._targets):
                store.pop(rid, None)

    async def wait(self, request_id: str, timeout_seconds: float) -> PermissionDecision | None:
        """Aguarda a decisão do proprietário. No tempo esgotado, resolve como negado."""
        future = self._pending.get(request_id)
        if future is None:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(future), timeout_seconds)
        except asyncio.TimeoutError:
            self.resolve(request_id, granted=False, decided_by="timeout")
            return PermissionDecision(
                granted=False,
                decided_by="timeout",
                reason="Tempo de espera pela autorização esgotado.",
            )

    def resolve(
        self,
        request_id: str,
        granted: bool,
        decided_by: str = "owner",
        remember_session: bool = False,
    ) -> PermissionDecision | None:
        """Resolve um pedido pendente. Retorna None se o id não existe, já foi resolvido ou expirou."""
        self._expire_stale()
        future = self._pending.get(request_id)
        if future is None or future.done():
            return None
        request = self._requests.get(request_id)
        if granted and request is not None:
            target = self._targets.get(request_id, ANY_TARGET)
            if target == RESEARCH_TARGET:
                # Autorização de pesquisa pública: vale pela conversa inteira (sem TTL de 30 min),
                # revogável explicitamente; nunca se transfere para outra conversa.
                if request.conversation_id:
                    self._grants[(request.conversation_id, "net.fetch", RESEARCH_TARGET)] = None
            elif granted and remember_session:
                expiry = self._clock() + self._grant_ttl
                for capability in request.capabilities:
                    self._grants[(request.conversation_id, capability, target)] = expiry
        decision = PermissionDecision(
            granted=granted,
            scope=request.scope if request else {},
            decided_by=decided_by,
        )
        future.set_result(decision)
        return decision

    def cancel_conversation(self, conversation_id: str) -> list[str]:
        """Revoga concessões e cancela pedidos pendentes de uma conversa excluída."""
        self.revoke(conversation_id=conversation_id)
        cancelled: list[str] = []
        for rid, req in list(self._requests.items()):
            if req.conversation_id != conversation_id:
                continue
            cancelled.append(rid)
            fut = self._pending.get(rid)
            if fut is not None and not fut.done():
                fut.set_result(
                    PermissionDecision(
                        granted=False,
                        decided_by="conversation_deleted",
                        reason="Conversa excluída.",
                    )
                )
            for store in (self._pending, self._requests, self._created, self._targets):
                store.pop(rid, None)
        return cancelled

    def revoke(self, conversation_id: str | None = None, capability: str | None = None) -> int:
        """Revoga concessões ativas (None = qualquer conversa/capacidade). Retorna quantas foram removidas."""
        keys = [
            key for key in self._grants
            if (conversation_id is None or key[0] == conversation_id) and (capability is None or key[1] == capability)
        ]
        for key in keys:
            self._grants.pop(key, None)
        return len(keys)

    def active_grants(self) -> list[dict]:
        now = self._clock()
        self._grants = {key: expiry for key, expiry in self._grants.items() if expiry is None or expiry > now}
        return [
            {"conversation_id": conv, "capability": cap, "target": target,
             "expires_in_seconds": None if expiry is None else round(expiry - now)}
            for (conv, cap, target), expiry in sorted(self._grants.items(), key=lambda item: item[1] if item[1] is not None else float("inf"))
        ]

    def session_capabilities(self) -> list[str]:
        return sorted({grant["capability"] for grant in self.active_grants()})

    def is_pending(self, request_id: str) -> bool:
        future = self._pending.get(request_id)
        return future is not None and not future.done()

    def pending_requests(self) -> list[PermissionRequest]:
        self._expire_stale()
        return [request for request_id, request in self._requests.items() if self.is_pending(request_id)]

    def get_request(self, request_id: str) -> PermissionRequest | None:
        return self._requests.get(request_id)

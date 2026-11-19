import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import get_audit_log, get_chat_service, get_permissions, get_settings, get_tool_engine
from backend.chat.service import ChatService
from backend.core.config import Settings
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.registry import AuditLog

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tools"])


class InvokeRequest(BaseModel):
    arguments: dict = Field(default_factory=dict)


class DecisionRequest(BaseModel):
    granted: bool
    remember_session: bool = False
    conversation_id: str | None = None
    expected_source: str | None = None


@router.get("/tools")
def list_tools(
    engine: ToolEngine = Depends(get_tool_engine),
    settings: Settings = Depends(get_settings),
    permissions: PermissionManager = Depends(get_permissions),
) -> dict:
    return {
        "enabled": settings.tools.enabled and bool(engine.list_tools()),
        "tools": engine.describe(),
        "session_grants": permissions.session_capabilities(),
        "grants": permissions.active_grants(),
        "warnings": engine.config_warnings(),
    }


@router.post("/tools/{tool_id}/invoke")
async def invoke_tool(
    tool_id: str,
    payload: InvokeRequest,
    engine: ToolEngine = Depends(get_tool_engine),
) -> dict:
    """Invocação explícita pelo proprietário (seleção manual de ferramenta)."""
    prepared = await engine.prepare(tool_id, payload.arguments, source="manual")
    if prepared.state == "invalid":
        return {"status": "invalid", "error": prepared.error}
    if prepared.state == "blocked":
        return {"status": "blocked", "error": prepared.error}
    if prepared.state == "needs_permission":
        request = prepared.request
        engine.remember_invocation(prepared, source="manual")
        return {
            "status": "permission_required",
            "request": {
                "id": request.id,
                "tool_id": request.tool_id,
                "action": request.action,
                "scope": request.scope,
                "reason": request.reason,
                "capabilities": request.capabilities,
            },
        }
    result = await engine.execute(prepared, None, source="manual")
    return {"status": "executed", "result": result.model_dump(mode="json")}


@router.get("/permissions")
def list_permissions(permissions: PermissionManager = Depends(get_permissions)) -> dict:
    return {
        "pending": [request.model_dump(mode="json") for request in permissions.pending_requests()],
        "session_grants": permissions.session_capabilities(),
        "grants": permissions.active_grants(),
    }


@router.delete("/permissions/grants")
def revoke_grants(
    conversation_id: str | None = None,
    capability: str | None = None,
    permissions: PermissionManager = Depends(get_permissions),
) -> dict:
    """Revoga concessões "Lembrar" ativas (filtros opcionais; sem filtros, revoga todas)."""
    return {"revoked": permissions.revoke(conversation_id, capability), "grants": permissions.active_grants()}


@router.post("/permissions/{request_id}/decision")
async def decide_permission(
    request_id: str,
    payload: DecisionRequest,
    permissions: PermissionManager = Depends(get_permissions),
    engine: ToolEngine = Depends(get_tool_engine),
    chat_service: ChatService = Depends(get_chat_service),
) -> dict:
    """Decisão do proprietário sobre um pedido de permissão.

    Concede/nega a capacidade, executa a invocação pendente (chat ou REST) e,
    quando ela veio de uma conversa, persiste o resultado como mensagem `tool`
    para que a conversa seja retomada (`resume=true` no stream de chat).
    """
    pending_req = permissions.get_request(request_id)
    if pending_req is not None:
        if payload.expected_source is not None and pending_req.source != payload.expected_source:
            raise HTTPException(status_code=409, detail="Origem da decisão não corresponde ao pedido de permissão.")
        if payload.conversation_id is not None and pending_req.conversation_id != payload.conversation_id:
            raise HTTPException(status_code=409, detail="Conversa da decisão não corresponde ao pedido de permissão.")
    decision = permissions.resolve(
        request_id, granted=payload.granted, remember_session=payload.remember_session
    )
    if decision is None:
        raise HTTPException(status_code=404, detail=f"Pedido não encontrado ou já resolvido: {request_id}")
    if pending_req is not None and pending_req.tool_id in ("verificacao.web", "pesquisa.profunda"):
        chat_service.record_web_decision(pending_req.conversation_id, decision.granted, (pending_req.scope or {}).get("consulta"))
    result = await engine.complete_invocation(request_id, decision)
    conversation_id = None
    if result is not None:
        conversation_id = (result.data or {}).get("conversation_id")
        if conversation_id and chat_service.is_conversation_active(conversation_id):
            tool_name = (result.data or {}).get("tool") or "ferramenta"
            chat_service.persist_tool_result(conversation_id, tool_name, result)
    return {
        "request_id": request_id,
        "granted": decision.granted,
        "decided_by": decision.decided_by,
        "remember_session": payload.remember_session,
        "conversation_id": conversation_id,
        "result": result.model_dump(mode="json") if result is not None else None,
    }


@router.get("/tools/history")
def tool_history(
    limit: int = 50,
    audit: AuditLog = Depends(get_audit_log),
) -> dict:
    limit = max(1, min(limit, 500))
    return {"events": audit.read_recent(limit)}

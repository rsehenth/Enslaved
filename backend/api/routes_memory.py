from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_chat_service, get_memory_manager, get_provider
from backend.chat.service import ChatService
from backend.memory.manager import MemoryManager
from backend.providers.base import ModelProvider

router = APIRouter(prefix="/memory", tags=["memory"])


class ManualRepackRequest(BaseModel):
    use_model: bool = False


@router.get("/status")
def memory_status(
    conversation_id: str | None = None,
    manager: MemoryManager = Depends(get_memory_manager),
) -> dict:
    return manager.get_status(conversation_id=conversation_id)


@router.get("/repacks")
def list_repacks(
    conversation_id: str | None = None,
    limit: int = 50,
    manager: MemoryManager = Depends(get_memory_manager),
) -> dict:
    limit = max(1, min(limit, 200))
    repacks = manager.repo.list_repacks(conversation_id=conversation_id or None, limit=limit)
    return {"repacks": [r.model_dump(mode="json") for r in repacks]}


@router.post("/conversations/{conversation_id}/repack")
async def manual_repack(
    conversation_id: str,
    payload: ManualRepackRequest | None = None,
    manager: MemoryManager = Depends(get_memory_manager),
    chat_service: ChatService = Depends(get_chat_service),
    provider: ModelProvider = Depends(get_provider),
) -> dict:
    conversation = chat_service.repo.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")
    use_model = payload.use_model if payload else manager.settings.repack_use_model
    result = await manager.manual_repack(
        conversation_id,
        provider=provider,
        use_model=use_model,
        allow_fallback=True,
        should_abort=lambda: not chat_service.is_conversation_active(conversation_id),
    )
    return {
        "ok": result.ok,
        "error": result.error,
        "fallback_used": result.fallback_used,
        "chunks_processed": result.chunks_processed,
        "repack": result.repack.model_dump(mode="json") if result.repack else None,
        "status": manager.get_status(conversation_id),
    }


@router.get("/conversations/{conversation_id}/markdown")
def conversation_markdown(
    conversation_id: str,
    manager: MemoryManager = Depends(get_memory_manager),
    chat_service: ChatService = Depends(get_chat_service),
) -> dict:
    conversation = chat_service.repo.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversation not found: {conversation_id}")
    content, path = manager.markdown.read_conversation_markdown(conversation_id, ensure_sync=True)
    return {
        "conversation_id": conversation_id,
        "path": str(path),
        "content": content,
    }


@router.post("/reconcile")
def reconcile_markdown(manager: MemoryManager = Depends(get_memory_manager)) -> dict:
    return manager.reconcile_exports()

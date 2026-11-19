from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from backend.api.deps import get_chat_service
from backend.chat.service import ChatService

router = APIRouter(tags=["conversations"])


class DeleteConversationsRequest(BaseModel):
    model_config = {"extra": "forbid"}

    ids: list[str] = Field(..., min_length=1, max_length=200)


@router.get("/conversations")
def list_conversations(service: ChatService = Depends(get_chat_service)) -> dict:
    return {"conversations": [c.model_dump(mode="json") for c in service.repo.list_conversations()]}


@router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, service: ChatService = Depends(get_chat_service)) -> dict:
    conversation = service.repo.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail=f"Conversa não encontrada: {conversation_id}")
    messages = service.repo.get_messages(conversation_id)
    return {
        "conversation": conversation.model_dump(mode="json"),
        "messages": [m.model_dump(mode="json") for m in messages],
    }


@router.post("/conversations/delete")
@router.delete("/conversations")
async def delete_conversations(
    payload: DeleteConversationsRequest,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    return await service.delete_conversations(payload.ids)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    service: ChatService = Depends(get_chat_service),
) -> dict:
    result = await service.delete_conversation(conversation_id)
    if result["deleted_count"] == 0:
        raise HTTPException(status_code=404, detail=f"Conversa não encontrada: {conversation_id}")
    return {
        "deleted": True,
        "id": conversation_id,
        "removed_files": result["removed_files"],
    }

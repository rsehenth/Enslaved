import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from backend.api.deps import get_chat_service
from backend.chat.models import ChatRequest
from backend.chat.service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(payload: ChatRequest, service: ChatService = Depends(get_chat_service)) -> dict:
    result = await service.send(
        payload.conversation_id,
        payload.message,
        system_prompt=payload.system_prompt,
        options=payload.provider_options(),
        attachment_ids=payload.attachment_ids,
        language=payload.language,
    )
    conversation = result["conversation"]
    return {
        "conversation_id": conversation.id,
        "conversation_title": conversation.title,
        "reply": result["assistant_message"].content,
        "user_message": result["user_message"].model_dump(mode="json"),
        "assistant_message": result["assistant_message"].model_dump(mode="json"),
        "context": result.get("context"),
    }


@router.post("/chat/stream")
async def chat_stream(payload: ChatRequest, service: ChatService = Depends(get_chat_service)):
    if payload.attachment_ids and not payload.resume and not payload.retry:
        if not service.vision_settings.enabled:
            from backend.storage.attachments import AttachmentRejected

            raise AttachmentRejected("Image upload is disabled.", 403)
        service.attachment_store.check_pending(payload.attachment_ids)

    async def event_source():
        try:
            async for event in service.send_stream(
                payload.conversation_id,
                payload.message,
                system_prompt=payload.system_prompt,
                options=payload.provider_options(),
                use_tools=payload.use_tools,
                activated_skills=payload.activated_skills,
                resume=payload.resume,
                allow_unverified=payload.allow_unverified,
                retry=payload.retry,
                attachment_ids=payload.attachment_ids,
                research_mode=payload.research_mode,
                language=payload.language,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:
            logger.exception("Unexpected error during stream")
            fallback = {"type": "error", "message": f"Unexpected error: {exc}"}
            yield f"data: {json.dumps(fallback, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@router.get("/research/{conversation_id}")
def research_status(conversation_id: str, service: ChatService = Depends(get_chat_service)) -> dict:
    """State of the active investigation for this conversation (temporary; never persisted as learning)."""
    task = service.research.current(conversation_id)
    return {
        "mode": service.research.mode(conversation_id),
        "task": None if task is None else {
            "research_task_id": task.research_task_id,
            "research_status": task.research_status,
            "research_scope": task.research_scope,
            "research_progress": task.research_progress,
            "mode_used": task.mode_used,
        },
    }


@router.post("/research/{conversation_id}/cancel")
async def research_cancel(conversation_id: str, service: ChatService = Depends(get_chat_service)) -> dict:
    return {"cancelled": service.cancel_research(conversation_id)}

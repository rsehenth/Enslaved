import asyncio
import hashlib
from urllib.parse import unquote

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response

from backend.api.deps import get_attachment_store
from backend.storage.attachments import AttachmentRejected, AttachmentStore, validate_attachment_id
from backend.storage.conversations import validate_conversation_id

router = APIRouter(tags=["attachments"])


@router.post("/attachments", status_code=201)
async def upload_attachment(
    request: Request,
    x_filename: str = Header(default="imagem.png", min_length=1, max_length=400),
    store: AttachmentStore = Depends(get_attachment_store),
) -> dict:
    """Recebe uma imagem (PNG, JPEG ou WebP), valida/normaliza em subprocesso isolado e guarda como pendente."""
    if not store.cfg.enabled:
        raise AttachmentRejected("Envio de imagens desativado.", 403)
    limit = store.cfg.max_image_bytes
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise AttachmentRejected(f"Imagem acima do limite de {limit // (1024 * 1024)} MB.", 413)
    buf = bytearray()
    async for chunk in request.stream():
        buf += chunk
        if len(buf) > limit:
            raise AttachmentRejected(f"Imagem acima do limite de {limit // (1024 * 1024)} MB.", 413)
    attachment = await asyncio.to_thread(store.save_pending, bytes(buf), unquote(x_filename))
    return {"attachment": attachment}


@router.delete("/attachments/{attachment_id}")
def delete_pending_attachment(
    attachment_id: str,
    store: AttachmentStore = Depends(get_attachment_store),
) -> dict:
    """Remove um anexo ainda não enviado (antes de enviar a mensagem)."""
    if not store.discard_pending(attachment_id):
        raise HTTPException(status_code=404, detail="Anexo pendente não encontrado.")
    return {"deleted": True, "id": attachment_id}


def _serve_attachment(row: dict, store: AttachmentStore) -> Response:
    try:
        path = store.path_of(row)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Arquivo do anexo não encontrado.")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise HTTPException(status_code=409, detail="Integridade do anexo comprometida.")
    return Response(
        content=data,
        media_type=row["mime"],
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "private, max-age=3600",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.get("/attachments/{attachment_id}")
def get_pending_attachment(
    attachment_id: str,
    conversation_id: str | None = None,
    store: AttachmentStore = Depends(get_attachment_store),
) -> Response:
    """Visualização de anexo pendente (prévia) ou vinculado (quando `conversation_id` corresponde)."""
    validate_attachment_id(attachment_id)
    row = store.get(attachment_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Anexo não encontrado.")
    if row["conversation_id"] is not None:
        if not conversation_id:
            raise HTTPException(status_code=403, detail="Acesso a anexo vinculado exige identificação da conversa.")
        validate_conversation_id(conversation_id)
        if row["conversation_id"] != conversation_id:
            raise HTTPException(status_code=403, detail="Anexo não pertence a esta conversa.")
    elif conversation_id is not None:
        raise HTTPException(status_code=404, detail="Anexo não pertence a esta conversa.")
    return _serve_attachment(row, store)


@router.get("/conversations/{conversation_id}/attachments/{attachment_id}")
def get_conversation_attachment(
    conversation_id: str,
    attachment_id: str,
    store: AttachmentStore = Depends(get_attachment_store),
) -> Response:
    """Visualização controlada de um anexo vinculado à conversa."""
    validate_conversation_id(conversation_id)
    validate_attachment_id(attachment_id)
    row = store.get(attachment_id)
    if row is None or row["conversation_id"] != conversation_id:
        raise HTTPException(status_code=404, detail="Anexo não encontrado nesta conversa.")
    return _serve_attachment(row, store)

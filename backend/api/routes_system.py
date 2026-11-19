import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from backend.api.deps import get_directives, get_provider, get_settings
from backend.core.config import Settings
from backend.core.directives import Directives
from backend.core.gpu import gpu_snapshot
from backend.providers.base import ModelProvider

router = APIRouter(tags=["system"])


@router.get("/health")
async def health(provider: ModelProvider = Depends(get_provider)) -> dict:
    status = await provider.health()
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": {"name": provider.name, "model": provider.model, **status},
    }


@router.get("/diagnostics")
async def diagnostics(
    provider: ModelProvider = Depends(get_provider),
    settings: Settings = Depends(get_settings),
) -> dict:
    """Diagnóstico de recursos: provider, modelos carregados e snapshot de GPU."""
    status = await provider.health()
    loaded = await provider.loaded_models() if status["available"] else []
    profile = settings.active_profile()
    return {
        "provider": {"name": provider.name, "model": provider.model, **status},
        "loaded_models": loaded,
        # subprocess em thread: não bloqueia o event loop (streams SSE ativos)
        "gpus": await asyncio.to_thread(gpu_snapshot),
        "settings": {
            "num_ctx": profile.num_ctx,
            "num_predict": profile.num_predict,
            "timeout_seconds": settings.ollama.timeout_seconds,
        },
    }


@router.get("/config")
async def config(
    settings: Settings = Depends(get_settings),
    provider: ModelProvider = Depends(get_provider),
    directives: Directives = Depends(get_directives),
) -> dict:
    profile = settings.active_profile()
    return {
        "app": {"name": settings.app.name, "host": settings.app.host, "port": settings.app.port},
        "provider": {
            "name": provider.name,
            "model": provider.model,
            "experimental": bool(provider.capabilities.get("experimental")),
            "supports_vision": bool(provider.capabilities.get("vision", True)),
            "supports_tools": await provider.supports_native_tools(),
            "num_ctx": profile.num_ctx,
            "num_predict": profile.num_predict,
            # Catálogo para o seletor de Settings (a troca exige reiniciar o backend).
            "models": [
                {
                    "id": model_id,
                    "experimental": entry.experimental,
                    "supports_vision": entry.supports_vision,
                    "num_ctx": entry.num_ctx if entry.num_ctx is not None else settings.ollama.num_ctx,
                }
                for model_id, entry in settings.provider.models.items()
            ],
        },
        "directives": {
            "name": directives.name,
            "language": directives.language,
            "path": str(directives.source_path) if directives.source_path else None,
        },
        "defaults": {
            "temperature": profile.temperature,
            "num_predict": profile.num_predict,
            "num_ctx": profile.num_ctx,
        },
        "vision": {
            "enabled": settings.vision.enabled,
            "max_images": settings.vision.max_images_per_message,
            "max_bytes": settings.vision.max_image_bytes,
            "formats": ["PNG", "JPEG", "WebP"],
        },
    }


@router.post("/model")
def select_model(payload: dict, settings: Settings = Depends(get_settings)) -> dict:
    """Persiste a escolha de modelo para a próxima inicialização (data/model-selection.json)."""
    model = str(payload.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="Campo 'model' é obrigatório.")
    if model not in settings.provider.models and model != settings.provider.model:
        raise HTTPException(status_code=404, detail=f"Modelo sem perfil configurado: {model!r}")
    path = settings.storage.database_path.parent / "model-selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": model}, ensure_ascii=False), encoding="utf-8")
    return {
        "ok": True,
        "selected": model,
        "active": settings.provider.model,
        "restart_required": model != settings.provider.model,
    }

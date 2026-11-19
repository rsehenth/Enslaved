"""Gerenciamento da API local (/v1) pelo proprietário, na interface do Enslaved.

Fica sob /api e herda as proteções locais do middleware (Host, origem, CSRF).
Chaves de inferência /v1 NÃO autenticam aqui: o gerenciamento é exclusivo da UI.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.api.deps import get_settings
from backend.core.config import Settings
from backend.localapi.keys import (
    ApiKeyLimitError,
    ApiKeyManager,
    load_enabled,
    save_enabled,
)

router = APIRouter(prefix="/localapi", tags=["localapi"])


def get_api_keys(request: Request) -> ApiKeyManager:
    return request.app.state.api_keys


def require_ui_origin(request: Request) -> None:
    """Gerenciamento de chaves só pela interface local: o navegador envia
    Sec-Fetch-Site: same-origin; IDEs/CLIs (com ou sem chave /v1) não passam.
    Host, origem e CSRF já são garantidos pelo middleware global."""
    if (request.headers.get("sec-fetch-site") or "").lower() != "same-origin":
        raise HTTPException(status_code=403, detail="Gerenciamento da API Local restrito à interface do Enslaved.")


@router.get("", dependencies=[Depends(require_ui_origin)])
async def get_status(request: Request, settings: Settings = Depends(get_settings)) -> dict:
    manager: ApiKeyManager = request.app.state.api_keys
    profile = settings.active_profile()
    return {
        "enabled": bool(request.app.state.local_api_enabled),
        # /v1 só aceita loopback: com start.bat (host 0.0.0.0) o endereço útil é 127.0.0.1
        "endpoint": f"http://{settings.app.host if settings.app.host in ('127.0.0.1', 'localhost', '::1') else '127.0.0.1'}:{settings.app.port}/v1",
        "model": settings.provider.model,
        "model_supports_tools": await request.app.state.provider.supports_native_tools(),
        "num_ctx": profile.num_ctx,
        "num_predict": profile.num_predict,
        "max_keys": manager.max_active,
        "active_keys": manager.count_active(),
        "keys": manager.list_keys(),
    }


@router.post("/enabled", dependencies=[Depends(require_ui_origin)])
async def set_enabled(payload: dict, request: Request, settings: Settings = Depends(get_settings)) -> dict:
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(status_code=400, detail="Campo booleano 'enabled' é obrigatório.")
    save_enabled(settings, enabled)
    request.app.state.local_api_enabled = enabled
    return {"enabled": enabled}


@router.post("/keys", dependencies=[Depends(require_ui_origin)])
async def create_key(payload: dict, request: Request) -> dict:
    manager: ApiKeyManager = request.app.state.api_keys
    try:
        return manager.create(str(payload.get("name") or ""))
    except ApiKeyLimitError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/keys/{key_id}", dependencies=[Depends(require_ui_origin)])
async def revoke_key(key_id: str, request: Request) -> dict:
    manager: ApiKeyManager = request.app.state.api_keys
    if not manager.revoke(key_id):
        raise HTTPException(status_code=404, detail="Chave não encontrada ou já revogada.")
    return {"revoked": True, "active_keys": manager.count_active()}

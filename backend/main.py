import ipaddress
import asyncio
import logging
import os
import socket
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api import (
    routes_attachments,
    routes_chat,
    routes_conversations,
    routes_localapi,
    routes_memory,
    routes_skills,
    routes_system,
    routes_tools,
    routes_v1,
)
from backend.chat.service import ChatService, ConversationNotFoundError
from backend.core.config import PROJECT_ROOT, ConfigError, Settings, load_settings
from backend.core.directives import DirectiveError, Directives, load_directives
from backend.chat.presentation import load_presentation, load_response_policy
from backend.localapi.keys import ApiKeyManager, load_enabled
from backend.memory.manager import MemoryManager
from backend.providers.base import ProviderError, ProviderUnavailableError, begin_request_generation_scope
from backend.providers.registry import create_provider
from backend.skills.discovery import SkillRegistry
from backend.skills.manager import SkillManager
from backend.storage.attachments import AttachmentRejected, AttachmentStore
from backend.storage.database import Database
from backend.tools.builtin import build_default_tools
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.policy import ToolPolicy
from backend.tools.registry import AuditLog, ToolRegistry

logger = logging.getLogger("enslaved")

FRONTEND_DIR = PROJECT_ROOT / "frontend"
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_MUTATING_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_CSP_HEADER = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
)


def _split_host_port(raw: str) -> tuple[str, int | None]:
    value = raw.strip().lower()
    if value.startswith("["):
        end = value.find("]")
        if end == -1:
            return value, None
        host = value[1:end]
        rest = value[end + 1 :]
        if rest.startswith(":") and rest[1:].isdigit():
            return host, int(rest[1:])
        return host, None
    if value.count(":") == 1:
        host, _, port_str = value.partition(":")
        if port_str.isdigit():
            return host, int(port_str)
    return value, None


def _is_private_ip(value: str | None) -> bool:
    """Loopback ou rede privada (10/8, 172.16/12, 192.168/16, link-local). Só IP literal: nomes DNS nunca passam."""
    try:
        ip = ipaddress.ip_address((value or "").split("%")[0])
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private


def _is_loopback_ip(value: str | None) -> bool:
    try:
        return ipaddress.ip_address((value or "").split("%")[0]).is_loopback
    except ValueError:
        return False


def _is_local_origin_url(url: str, expected_port: int | None, lan_host: str | None = None) -> bool:
    try:
        parsed = urlsplit(url.strip())
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    if parsed.hostname.lower() not in _LOOPBACK_HOSTS and parsed.hostname.lower() != lan_host:
        return False
    if expected_port is not None:
        origin_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if origin_port != expected_port:
            return False
    return True


def _register_local_security_middleware(app: FastAPI, allow_lan: bool = False) -> None:
    """`allow_lan`: o servidor escuta na rede local. Aceita apenas clientes e Hosts que sejam
    IPs privados literais (nunca nomes DNS, contra DNS rebinding); a internet fica de fora."""

    @app.middleware("http")
    async def local_only_security(request: Request, call_next):
        begin_request_generation_scope()  # uso/finish_reason do provider isolados por requisição
        host_header = request.headers.get("host", "")
        host_name, host_port = _split_host_port(host_header)
        is_testclient = bool(request.client and request.client.host == "testclient")
        lan_host = host_name if allow_lan and _is_private_ip(host_name) else None
        if allow_lan and not (is_testclient or _is_private_ip(request.client.host if request.client else None)):
            return JSONResponse(status_code=403, content={"error": "Acesso permitido apenas na rede local."})
        if host_name not in _LOOPBACK_HOSTS and lan_host is None and not (is_testclient and host_name == "testserver"):
            return JSONResponse(status_code=403, content={"error": "Host não autorizado."})
        # API externa (/v1) só neste PC, mesmo com start.bat escutando em 0.0.0.0 para a UI no celular.
        if request.url.path.startswith("/v1") and not is_testclient:
            client_ip = request.client.host if request.client else None
            if host_name not in _LOOPBACK_HOSTS or not _is_loopback_ip(client_ip):
                return JSONResponse(
                    status_code=403,
                    content={"error": {"message": "A API /v1 aceita apenas conexões locais (127.0.0.1).", "type": "permission_error", "code": "local_only"}},
                )

        sec_fetch_site = (request.headers.get("sec-fetch-site") or "").strip().lower()
        if sec_fetch_site == "cross-site":
            return JSONResponse(status_code=403, content={"error": "Requisição cross-site bloqueada."})

        is_api_or_mutating = request.url.path.startswith("/api") or request.method in _MUTATING_METHODS
        if is_api_or_mutating:
            origin = request.headers.get("origin")
            if origin is not None:
                if not _is_local_origin_url(origin, host_port, lan_host):
                    return JSONResponse(status_code=403, content={"error": "Origem não autorizada."})
            elif request.method in _MUTATING_METHODS:
                if sec_fetch_site and sec_fetch_site not in ("same-origin", "none"):
                    return JSONResponse(status_code=403, content={"error": "Origem de requisição bloqueada."})
                referer = request.headers.get("referer")
                if referer and not _is_local_origin_url(referer, host_port, lan_host):
                    return JSONResponse(status_code=403, content={"error": "Referer não autorizado."})

        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not request.url.path.startswith("/api"):
            response.headers.setdefault("Content-Security-Policy", _CSP_HEADER)
        return response


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ProviderUnavailableError)
    async def provider_unavailable(request: Request, exc: ProviderUnavailableError):
        return JSONResponse(
            status_code=503,
            content={
                "error": str(exc),
                "hint": "Inicie o Ollama (ollama serve) e instale o modelo configurado.",
            },
        )

    @app.exception_handler(ProviderError)
    async def provider_error(request: Request, exc: ProviderError):
        return JSONResponse(status_code=502, content={"error": str(exc)})

    @app.exception_handler(ConversationNotFoundError)
    async def conversation_not_found(request: Request, exc: ConversationNotFoundError):
        return JSONResponse(status_code=404, content={"error": str(exc)})

    @app.exception_handler(PermissionError)
    async def permission_error(request: Request, exc: PermissionError):
        return JSONResponse(status_code=403, content={"error": str(exc)})

    @app.exception_handler(ValueError)
    async def value_error(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": str(exc)})

    @app.exception_handler(AttachmentRejected)
    async def attachment_rejected(request: Request, exc: AttachmentRejected):
        return JSONResponse(status_code=exc.status, content={"error": str(exc)})

    @app.exception_handler(DirectiveError)
    async def directive_error(request: Request, exc: DirectiveError):
        return JSONResponse(status_code=500, content={"error": f"Diretrizes inválidas: {exc}"})

    @app.exception_handler(ConfigError)
    async def config_error(request: Request, exc: ConfigError):
        return JSONResponse(status_code=500, content={"error": f"Configuração inválida: {exc}"})


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    logging.basicConfig(level=settings.app.log_level.upper())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings.storage.database_path)
        db.connect()
        directives = load_directives(settings.directives.path)
        provider = create_provider(settings)
        skills = SkillRegistry()
        skills.discover(settings.skills.directories)

        data_dir = settings.storage.database_path.parent
        audit = AuditLog(data_dir / "tool-audit.jsonl")
        skill_manager = SkillManager(skills, data_dir / "skills-state.json")
        profile = settings.active_profile()
        memory_manager = MemoryManager(
            db,
            settings.memory,
            num_ctx=profile.num_ctx,
            num_predict=profile.num_predict,
            max_image_tokens=settings.vision.max_image_tokens,
        )
        attachment_store = AttachmentStore(
            db,
            data_dir / "attachments",
            settings.vision,
            io_lock=memory_manager.markdown._io_lock,
        )
        attachment_store.reconcile()
        reconcile_summary = memory_manager.reconcile_exports()
        if reconcile_summary["reconciled"] or reconcile_summary["failed"]:
            logger.info("Reconciliação Markdown inicial: %s", reconcile_summary)
        tool_registry = ToolRegistry()
        tool_policy = ToolPolicy(settings.tools)
        permissions = PermissionManager(tool_policy)
        tool_engine = ToolEngine(tool_registry, tool_policy, permissions, audit, settings.tools)
        for warning in tool_engine.config_warnings():
            logger.warning("Segurança de rede: %s", warning)
        context = {
            "provider": provider,
            "settings": settings,
            "database_path": settings.storage.database_path,
            "skill_manager": skill_manager,
            "tool_engine": tool_engine,
            "started_at": time.time(),
        }
        for tool in build_default_tools(settings, context):
            tool_registry.register(tool)

        api_key_manager = ApiKeyManager(db, max_active=settings.local_api.max_keys)

        app.state.settings = settings
        app.state.db = db
        app.state.directives = directives
        app.state.provider = provider
        app.state.skills = skills
        app.state.skill_manager = skill_manager
        app.state.tool_engine = tool_engine
        app.state.permissions = permissions
        app.state.audit = audit
        app.state.memory_manager = memory_manager
        app.state.attachment_store = attachment_store
        app.state.api_keys = api_key_manager
        app.state.local_api_enabled = load_enabled(settings)
        app.state.local_api_semaphore = asyncio.Semaphore(settings.local_api.max_concurrent)
        app.state.chat_service = ChatService(
            db,
            provider,
            directives,
            tool_engine=tool_engine,
            skill_manager=skill_manager,
            memory_manager=memory_manager,
            presentation=load_presentation(settings.chat),
            response_policy=load_response_policy(settings.chat),
            attachment_store=attachment_store,
            vision_settings=settings.vision,
            research_settings=settings.research,
            chat_settings=settings.chat,
            net_settings=settings.tools.net,
        )
        logger.info(
            "Enslaved iniciado | provider=%s model=%s skills=%d tools=%d db=%s",
            provider.name,
            provider.model,
            len(skills.list_skills()),
            len(tool_registry.list_tools()),
            settings.storage.database_path,
        )
        try:
            yield
        finally:
            await provider.aclose()
            db.close()

    app = FastAPI(title=settings.app.name, version="0.2.0", lifespan=lifespan)
    _register_local_security_middleware(app, allow_lan=settings.app.allow_lan)
    _register_exception_handlers(app)
    app.add_exception_handler(routes_v1.V1Error, routes_v1.v1_error_handler)
    app.include_router(routes_system.router, prefix="/api")
    app.include_router(routes_chat.router, prefix="/api")
    app.include_router(routes_attachments.router, prefix="/api")
    app.include_router(routes_conversations.router, prefix="/api")
    app.include_router(routes_skills.router, prefix="/api")
    app.include_router(routes_tools.router, prefix="/api")
    app.include_router(routes_memory.router, prefix="/api")
    app.include_router(routes_localapi.router, prefix="/api")
    app.include_router(routes_v1.router)
    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/", include_in_schema=False)
        async def index():
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


def _acquire_single_instance(host: str, port: int, lock_path: Path):
    """Garante instância única no workspace e detecta porta ocupada antes do uvicorn."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        raise RuntimeError(
            f"Já existe outra instância do Enslaved em execução neste diretório (lock: {lock_path})."
        )

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    probe = socket.socket(family, socket.SOCK_STREAM)
    try:
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind((host, port))
    except OSError as exc:
        lock_file.close()
        raise RuntimeError(
            f"A porta {port} em {host} já está ocupada por outro processo ({exc})."
        ) from exc
    finally:
        probe.close()
    return lock_file


app = create_app()


if __name__ == "__main__":
    runtime_settings = load_settings()
    lock_path = runtime_settings.storage.database_path.parent / ".backend.lock"
    try:
        _lock_handle = _acquire_single_instance(
            runtime_settings.app.host, runtime_settings.app.port, lock_path
        )
    except RuntimeError as exc:
        print(f"[Enslaved] {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        uvicorn.run(
            app,
            host=runtime_settings.app.host,
            port=runtime_settings.app.port,
            log_level=runtime_settings.app.log_level,
        )
    finally:
        _lock_handle.close()

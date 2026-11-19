from fastapi import Request

from backend.chat.service import ChatService
from backend.core.config import Settings
from backend.core.directives import Directives
from backend.memory.manager import MemoryManager
from backend.providers.base import ModelProvider
from backend.skills.discovery import SkillRegistry
from backend.skills.manager import SkillManager
from backend.storage.attachments import AttachmentStore
from backend.tools.engine import ToolEngine
from backend.tools.permissions import PermissionManager
from backend.tools.registry import AuditLog


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_directives(request: Request) -> Directives:
    return request.app.state.directives


def get_chat_service(request: Request) -> ChatService:
    return request.app.state.chat_service


def get_provider(request: Request) -> ModelProvider:
    return request.app.state.provider


def get_skill_registry(request: Request) -> SkillRegistry:
    return request.app.state.skills


def get_skill_manager(request: Request) -> SkillManager:
    return request.app.state.skill_manager


def get_tool_engine(request: Request) -> ToolEngine:
    return request.app.state.tool_engine


def get_permissions(request: Request) -> PermissionManager:
    return request.app.state.permissions


def get_audit_log(request: Request) -> AuditLog:
    return request.app.state.audit


def get_memory_manager(request: Request) -> MemoryManager:
    return request.app.state.memory_manager


def get_attachment_store(request: Request) -> AttachmentStore:
    return request.app.state.attachment_store

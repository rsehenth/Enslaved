from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend.api.deps import get_skill_manager, get_skill_registry, get_tool_engine
from backend.skills.discovery import SkillRecord, SkillRegistry
from backend.skills.manager import SkillManager
from backend.tools.engine import ToolEngine

router = APIRouter(tags=["skills"])


class SkillEnabledRequest(BaseModel):
    enabled: bool


def _summary(skill: SkillRecord, manager: SkillManager, tool_ids: set[str], capabilities: set[str]) -> dict:
    return {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "version": skill.version,
        "has_scripts": skill.has_scripts,
        "has_references": skill.has_references,
        "keywords": skill.keywords,
        "requires": skill.requires,
        "enabled": manager.is_enabled(skill.id),
        "available": manager.is_compatible(skill, tool_ids, capabilities),
        "missing_requirements": manager.missing_requirements(skill, tool_ids, capabilities),
    }


def _capability_sets(engine: ToolEngine | None) -> tuple[set[str], set[str]]:
    if engine is None:
        return set(), set()
    tool_ids: set[str] = set()
    capabilities: set[str] = set()
    for description in engine.describe():
        tool_ids.add(description["id"])
        capabilities.update(description["capabilities"])
    return tool_ids, capabilities


@router.get("/skills")
def list_skills(
    registry: SkillRegistry = Depends(get_skill_registry),
    manager: SkillManager = Depends(get_skill_manager),
    engine: ToolEngine | None = Depends(get_tool_engine),
) -> dict:
    tool_ids, capabilities = _capability_sets(engine)
    return {
        "skills": [_summary(skill, manager, tool_ids, capabilities) for skill in registry.list_skills()],
        "errors": registry.errors,
        "summary": manager.summary(tool_ids, capabilities),
    }


@router.get("/skills/{skill_id}")
def get_skill(
    skill_id: str,
    registry: SkillRegistry = Depends(get_skill_registry),
    manager: SkillManager = Depends(get_skill_manager),
    engine: ToolEngine | None = Depends(get_tool_engine),
) -> dict:
    skill = registry.get(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail=f"Skill não encontrada: {skill_id}")
    tool_ids, capabilities = _capability_sets(engine)
    return {
        **_summary(skill, manager, tool_ids, capabilities),
        "instructions": skill.instructions,
        "metadata": skill.metadata,
        "path": str(skill.path),
    }


@router.post("/skills/{skill_id}/enabled")
def set_skill_enabled(
    skill_id: str,
    payload: SkillEnabledRequest,
    manager: SkillManager = Depends(get_skill_manager),
) -> dict:
    if not manager.set_enabled(skill_id, payload.enabled):
        raise HTTPException(status_code=404, detail=f"Skill não encontrada: {skill_id}")
    return {"id": skill_id, "enabled": manager.is_enabled(skill_id)}

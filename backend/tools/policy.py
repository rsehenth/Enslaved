from backend.core.config import ToolsSettings


class ToolPolicy:
    """Política de permissões por capacidade: allow (automático) | ask (proprietário) | deny.

    Capacidade sem entrada explícita na configuração assume o modo mais
    conservador ("ask"): a ferramenta não roda sem autorização.
    """

    DEFAULT_MODE = "ask"

    def __init__(self, settings: ToolsSettings):
        self._modes = {
            capability.strip().lower(): mode.strip().lower()
            for capability, mode in settings.permission_modes.items()
        }

    def mode_for(self, capability: str) -> str:
        return self._modes.get(capability.strip().lower(), self.DEFAULT_MODE)

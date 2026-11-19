from backend.core.config import ConfigError, Settings
from backend.providers.base import ModelProvider
from backend.providers.ollama import OllamaProvider
from backend.providers.simulated import SimulatedProvider

PROVIDERS: dict[str, type[ModelProvider]] = {
    "ollama": OllamaProvider,
    "simulated": SimulatedProvider,
}


def create_provider(settings: Settings) -> ModelProvider:
    """Seleciona o provider configurado (config/settings.yaml -> provider.name)."""
    provider_name = settings.provider.name.strip().lower()
    provider_class = PROVIDERS.get(provider_name)
    if provider_class is None:
        raise ConfigError(
            f"Provider desconhecido: {settings.provider.name!r}. "
            f"Disponíveis: {', '.join(sorted(PROVIDERS))}"
        )
    profile = settings.active_profile()
    if provider_class is OllamaProvider:
        provider = OllamaProvider(
            model=settings.provider.model,
            base_url=settings.ollama.base_url,
            timeout_seconds=settings.ollama.timeout_seconds,
            default_options={
                "num_ctx": profile.num_ctx,
                "temperature": profile.temperature,
                "num_predict": profile.num_predict,
                **profile.extra_options,
            },
            think=profile.think,
        )
    else:
        provider = provider_class(model=settings.provider.model)
    provider.capabilities = {
        "vision": profile.supports_vision,
        "tools": profile.supports_tools,
        "experimental": profile.experimental,
    }
    return provider

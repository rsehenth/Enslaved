from __future__ import annotations

import ipaddress
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(Exception):
    """Erro de configuração inválida."""


_LOCAL_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class AppSettings(BaseModel):
    model_config = {"validate_assignment": True}

    name: str = "Enslaved"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"

    # Acesso por outros aparelhos da rede (ex.: celular no mesmo Wi-Fi). Só liga com opt-in
    # explícito (ENSLAVED_LAN=1, feito pelo start.bat); o padrão continua sendo apenas loopback.
    allow_lan: bool = False

    @model_validator(mode="after")
    def _validate_host(self) -> "AppSettings":
        host = self.host.strip().lower()
        if host not in _LOCAL_LOOPBACK_HOSTS:
            try:
                ipaddress.ip_address(host)
                is_ip = True
            except ValueError:
                is_ip = False
            if not (self.allow_lan and is_ip):
                raise ValueError(
                    f"host {self.host!r} não permitido no MVP local; use apenas loopback "
                    "(127.0.0.1, localhost ou ::1) ou habilite ENSLAVED_LAN=1 para a rede local"
                )
        return self


class ModelProfileSettings(BaseModel):
    """Perfil de um modelo específico (provider.models). Campos None herdam de ollama.*."""

    num_ctx: int | None = None
    temperature: float | None = None
    num_predict: int | None = None
    think: bool | None = None
    supports_vision: bool = True
    # Tool calling nativo do modelo (campo `tools` no payload do Ollama). None = detectar
    # via metadados do Ollama (/api/show); não confunde com visão nem raciocínio.
    supports_tools: bool | None = None
    experimental: bool = False
    # Opções extras do Ollama (samplers como top_p, min_p, repeat_penalty). Vazio = nada é enviado.
    extra_options: dict[str, Any] = Field(default_factory=dict)


class ProviderSettings(BaseModel):
    name: str = "ollama"
    model: str = "hf.co/bartowski/Qwen_Qwen3.5-9B-GGUF:Q6_K"
    # Catálogo de perfis por modelo; o ativo é provider.model (a troca exige reiniciar o backend).
    models: dict[str, ModelProfileSettings] = Field(default_factory=dict)


class OllamaSettings(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 120.0
    num_ctx: int = 16384
    temperature: float = 0.7
    num_predict: int = 2048
    think: bool = False


class StorageSettings(BaseModel):
    database_path: Path = Path("data/enslaved.db")


class DirectivesSettings(BaseModel):
    path: Path = Path("config/directives.md")


class SkillsSettings(BaseModel):
    directories: list[Path] = [Path("skills")]


class ChatSettings(BaseModel):
    presentation_policy: bool = True  # injeta as regras de apresentação (config/presentation.md) no prompt
    presentation_path: Path = Path("config/presentation.md")
    response_policy: bool = True  # injeta a política de estilo e liberdade de resposta (config/response_policy.md)
    response_policy_path: Path = Path("config/response_policy.md")
    technical_examples: bool = True  # injeta a política de exemplos técnicos (config/technical_examples.md) só em perguntas técnicas
    technical_examples_path: Path = Path("config/technical_examples.md")
    verify_facts: bool = True
    search_provider: str = "duckduckgo"  # duckduckgo | brave (exige ENSLAVED_SEARCH_API_KEY)
    search_max_results: int = Field(default=5, ge=1, le=10)


class MemorySettings(BaseModel):
    enabled: bool = True
    base_dir: Path = Path("data/memory")
    compact_threshold_ratio: float = Field(default=0.75, gt=0.1, le=0.95)
    recent_messages_to_keep: int = Field(default=6, ge=2, le=50)
    retrieval_max_tokens: int = Field(default=750, ge=50)
    retrieval_max_items: int = Field(default=8, ge=1)
    max_summary_tokens: int = Field(default=600, ge=100)
    repack_use_model: bool = False
    # Repack automático: alvo de ocupação após compactar (evita recompactar a cada turno) e recuperação de assuntos antigos
    repack_target_ratio: float = Field(default=0.5, gt=0.1, le=0.9)
    recall_max_tokens: int = Field(default=600, ge=50)


class VisionSettings(BaseModel):
    """Imagens anexadas ao chat (Qwen 3.5 com projetor visual, via Ollama local; nada sai da máquina)."""

    enabled: bool = True
    max_images_per_message: int = Field(default=4, ge=1, le=8)
    max_image_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)  # por imagem enviada (e por imagem normalizada)
    max_image_side: int = Field(default=8192, ge=64)  # px, maior lado
    max_image_pixels: int = Field(default=25_000_000, ge=1024)  # limite de decodificação (RGBA de 25 MP ≈ 100 MB)
    max_image_tokens: int = Field(default=2560, ge=64)  # acima disso a cópia enviada ao modelo é reduzida
    processing_timeout_seconds: float = Field(default=15.0, gt=0)  # prazo total do subprocesso de decodificação
    processing_memory_mb: int = Field(default=1024, ge=128)  # teto de memória do subprocesso (Job Object no Windows)
    pending_ttl_seconds: int = Field(default=3600, ge=60)  # anexos enviados e nunca usados em uma mensagem
    max_pending: int = Field(default=24, ge=1)


class FsToolsSettings(BaseModel):
    allowed_roots: list[Path] = Field(
        default_factory=lambda: [Path("data"), Path("docs"), Path("skills"), Path("config")]
    )
    max_file_bytes: int = Field(default=524288, ge=1)


class NetToolsSettings(BaseModel):
    timeout_seconds: float = Field(default=15.0, gt=0)
    max_bytes: int = Field(default=1048576, ge=1)
    max_redirects: int = Field(default=3, ge=0)
    allowed_ports: list[int] = Field(default_factory=lambda: [80, 443])
    min_host_interval_seconds: float = Field(default=0.25, ge=0)  # espaçamento entre inspeções (net.inspect) ao mesmo host


class ToolsSettings(BaseModel):
    enabled: bool = True
    max_calls_per_request: int = Field(default=5, ge=1)
    tool_timeout_seconds: float = Field(default=20.0, gt=0)
    permission_wait_seconds: float = Field(default=180.0, gt=0)
    max_output_chars: int = Field(default=8000, ge=200)
    permission_modes: dict[str, str] = Field(
        default_factory=lambda: {
            "app.diagnostics": "allow",
            "text.analyze": "allow",
            "fs.read": "ask",
            "net.fetch": "ask",
        }
    )
    fs: FsToolsSettings = FsToolsSettings()
    net: NetToolsSettings = NetToolsSettings()
    # Opt-in avançado e perigoso: com net.fetch em "allow", dispensa o cartão também em contexto não confiável
    # e para URLs com possíveis dados sensíveis. Desligado por padrão; ligado gera aviso no log e em /api/tools.
    unsafe_allow_untrusted_egress: bool = False

    @field_validator("permission_modes")
    @classmethod
    def _validate_modes(cls, value: dict[str, str]) -> dict[str, str]:
        valid = {"allow", "ask", "deny"}
        for capability, mode in value.items():
            if mode not in valid:
                raise ValueError(f"modo de permissão inválido para {capability!r}: {mode!r} (use allow|ask|deny)")
        return value


class ResearchLimits(BaseModel):
    """Orçamento operacional de uma investigação (configurável; não é um teto de privilégios)."""

    max_pages: int = Field(default=8, ge=1)  # páginas do alvo (ou resultados abertos) por investigação
    max_requests: int = Field(default=30, ge=1)  # chamadas de rede (inspeções + buscas)
    max_seconds: float = Field(default=120.0, gt=0)  # duração total
    concurrency: int = Field(default=3, ge=1, le=16)
    max_total_bytes: int = Field(default=8 * 1024 * 1024, ge=1024)  # volume de dados coletado
    max_searches: int = Field(default=3, ge=0)  # consultas ao mecanismo de busca (CVE/fontes)
    report_tokens: int = Field(default=3072, ge=512, le=32768)  # orçamento de saída do relatório neste nível


class LocalApiSettings(BaseModel):
    """API local OpenAI-compatible (/v1) para IDEs e CLIs. Chaves gerenciadas na interface."""

    enabled: bool = False
    max_keys: int = Field(default=5, ge=1, le=5)  # chaves ativas simultâneas
    max_concurrent: int = Field(default=2, ge=1, le=8)  # gerações /v1 simultâneas (VRAM)
    queue_timeout_seconds: float = Field(default=30.0, gt=0)  # espera na fila antes de 429
    max_messages: int = Field(default=160, ge=1)
    max_payload_bytes: int = Field(default=4 * 1024 * 1024, ge=1024)


class ResearchSettings(BaseModel):
    """Investigação profunda: níveis de esforço (a profundidade nunca amplia permissões de ferramentas)."""

    enabled: bool = True
    deep: ResearchLimits = ResearchLimits()
    extended: ResearchLimits = ResearchLimits(
        max_pages=25, max_requests=90, max_seconds=360.0, concurrency=4, max_total_bytes=24 * 1024 * 1024, max_searches=8, report_tokens=4608
    )
    report_block_chars: int = Field(default=9000, ge=1000)  # teto do dossiê de evidências enviado ao modelo
    report_continuations: int = Field(default=2, ge=0, le=4)  # retomadas controladas se a saída for cortada pelo limite

    def limits(self, mode: str) -> ResearchLimits:
        return self.extended if mode == "extended" else self.deep


class Settings(BaseModel):
    app: AppSettings = AppSettings()
    provider: ProviderSettings = ProviderSettings()
    ollama: OllamaSettings = OllamaSettings()
    storage: StorageSettings = StorageSettings()
    directives: DirectivesSettings = DirectivesSettings()
    skills: SkillsSettings = SkillsSettings()
    tools: ToolsSettings = ToolsSettings()
    memory: MemorySettings = MemorySettings()
    vision: VisionSettings = VisionSettings()
    chat: ChatSettings = ChatSettings()
    local_api: LocalApiSettings = LocalApiSettings()
    research: ResearchSettings = ResearchSettings()

    def active_profile(self) -> ModelProfileSettings:
        """Perfil efetivo do modelo ativo: provider.models[model] com herança de ollama.* para campos ausentes."""
        profile = self.provider.models.get(self.provider.model) or ModelProfileSettings()
        return profile.model_copy(
            update={
                "num_ctx": profile.num_ctx if profile.num_ctx is not None else self.ollama.num_ctx,
                "temperature": profile.temperature if profile.temperature is not None else self.ollama.temperature,
                "num_predict": profile.num_predict if profile.num_predict is not None else self.ollama.num_predict,
                "think": profile.think if profile.think is not None else self.ollama.think,
            }
        )


def _to_bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


# a ordem importa: ENSLAVED_LAN precisa valer antes de ENSLAVED_HOST ser validado
ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "ENSLAVED_LAN": ("app", "allow_lan", _to_bool),
    "ENSLAVED_HOST": ("app", "host", str),
    "ENSLAVED_PORT": ("app", "port", int),
    "ENSLAVED_PROVIDER": ("provider", "name", str),
    "ENSLAVED_MODEL": ("provider", "model", str),
    "ENSLAVED_OLLAMA_BASE_URL": ("ollama", "base_url", str),
    "ENSLAVED_DB_PATH": ("storage", "database_path", Path),
}


def _load_env_file(root: Path) -> None:
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _apply_env_overrides(settings: Settings) -> None:
    for env_key, (section, attr, cast) in ENV_OVERRIDES.items():
        raw = os.environ.get(env_key)
        if raw is None or raw == "":
            continue
        try:
            setattr(getattr(settings, section), attr, cast(raw))
        except ValueError as exc:
            raise ConfigError(f"Valor inválido para {env_key}: {raw!r} ({exc})") from exc


def _resolve_paths(settings: Settings, root: Path) -> Settings:
    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    settings.storage.database_path = resolve(settings.storage.database_path)
    settings.directives.path = resolve(settings.directives.path)
    settings.chat.presentation_path = resolve(settings.chat.presentation_path)
    settings.chat.response_policy_path = resolve(settings.chat.response_policy_path)
    settings.chat.technical_examples_path = resolve(settings.chat.technical_examples_path)
    settings.skills.directories = [resolve(d) for d in settings.skills.directories]
    settings.tools.fs.allowed_roots = [resolve(r) for r in settings.tools.fs.allowed_roots]
    settings.memory.base_dir = resolve(settings.memory.base_dir)
    return settings


def _apply_runtime_model_selection(settings: Settings) -> None:
    """Seleção de modelo feita pela interface (data/model-selection.json).

    Precedência: ENSLAVED_MODEL (env) > arquivo runtime > settings.yaml > padrão.
    A troca só vale após reiniciar o backend (provider é criado no startup)."""
    if os.environ.get("ENSLAVED_MODEL"):
        return
    path = settings.storage.database_path.parent / "model-selection.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    model = data.get("model") if isinstance(data, dict) else None
    if isinstance(model, str) and model.strip():
        settings.provider.model = model.strip()


def load_settings(config_path: Path | None = None, root: Path | None = None) -> Settings:
    root = root or PROJECT_ROOT
    _load_env_file(root)
    if config_path is None:
        config_path = root / "config" / "settings.yaml"
    data: dict = {}
    if config_path.is_file():
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"Configuração inválida em {config_path}: esperado um mapa YAML")
    try:
        settings = Settings.model_validate(data)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"Falha ao carregar configuração: {exc}") from exc
    _apply_env_overrides(settings)
    settings = _resolve_paths(settings, root)
    _apply_runtime_model_selection(settings)
    return settings

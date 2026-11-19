"""Chaves da API local OpenAI-compatible (/v1).

Somente hash SHA-256 do segredo é persistido; o segredo completo (prefixo `ens_` +
256 bits de entropia via `secrets.token_urlsafe`) é exibido uma única vez na criação.
Autenticação compara hashes com `hmac.compare_digest` (máx. 5 chaves ativas).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.storage.database import Database

KEY_PREFIX = "ens_"
KEY_ENTROPY_BYTES = 32
MAX_KEY_NAME_CHARS = 64


class ApiKeyLimitError(Exception):
    """Já existem max_keys chaves ativas."""


class ApiKeyNotFound(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hash_secret(secret: str) -> str:
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def masked_key(prefix: str) -> str:
    """Parte exibível depois da criação: `ens_ab12cd3…` (nunca o segredo completo)."""
    return f"{prefix[:12]}…"


class ApiKeyManager:
    def __init__(self, db: Database, max_active: int = 5):
        self.db = db
        self.max_active = max_active

    def create(self, name: str) -> dict[str, Any]:
        """Cria uma chave ativa. Retorna `secret` completo (única exibição) + metadados."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Nome da chave é obrigatório.")
        if len(name) > MAX_KEY_NAME_CHARS:
            raise ValueError(f"Nome da chave excede {MAX_KEY_NAME_CHARS} caracteres.")
        active = self.count_active()
        if active >= self.max_active:
            raise ApiKeyLimitError(
                f"Limite de {self.max_active} chaves ativas atingido. Revogue uma chave para criar outra."
            )
        secret = f"{KEY_PREFIX}{secrets.token_urlsafe(KEY_ENTROPY_BYTES)}"
        row = {
            "id": uuid.uuid4().hex,
            "name": name,
            "key_hash": hash_secret(secret),
            "key_prefix": secret[:12],
            "status": "active",
            "created_at": _now(),
            "last_used_at": None,
            "revoked_at": None,
        }
        self.db.write(
            "INSERT INTO api_keys (id, name, key_hash, key_prefix, status, created_at, last_used_at, revoked_at) "
            "VALUES (:id, :name, :key_hash, :key_prefix, :status, :created_at, :last_used_at, :revoked_at)",
            row,
        )
        return {**self._public(row), "secret": secret}

    def count_active(self) -> int:
        row = self.db.read_one("SELECT COUNT(*) AS n FROM api_keys WHERE status = 'active'")
        return int(row["n"]) if row else 0

    def list_keys(self) -> list[dict[str, Any]]:
        rows = self.db.read_all("SELECT * FROM api_keys ORDER BY created_at DESC, id")
        return [self._public(row) for row in rows]

    def revoke(self, key_id: str) -> bool:
        cursor = self.db.write(
            "UPDATE api_keys SET status = 'revoked', revoked_at = ? WHERE id = ? AND status = 'active'",
            (_now(), key_id),
        )
        return bool(cursor.rowcount)

    def authenticate(self, secret: str) -> dict[str, Any] | None:
        """Valida o segredo contra as chaves ativas (comparação constante) e registra o uso."""
        if not secret or not secret.startswith(KEY_PREFIX):
            return None
        provided = hash_secret(secret)
        rows = self.db.read_all("SELECT * FROM api_keys WHERE status = 'active'")
        for row in rows:
            if hmac.compare_digest(row["key_hash"], provided):
                self.db.write("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (_now(), row["id"]))
                return self._public(row)
        return None

    @staticmethod
    def _public(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "masked": masked_key(row["key_prefix"]),
            "status": row["status"],
            "created_at": row["created_at"],
            "last_used_at": row["last_used_at"],
            "revoked_at": row["revoked_at"],
        }


# ------------------------------------------------------------------ estado ligado/desligado
# JSON em data/local-api.json (mesmo padrão de model-selection.json): o toggle da
# interface vale sem reiniciar; ausência do arquivo = valor do settings.yaml.


def local_api_state_path(settings: Any) -> Path:
    return settings.storage.database_path.parent / "local-api.json"


def load_enabled(settings: Any) -> bool:
    try:
        data = json.loads(local_api_state_path(settings).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings.local_api.enabled
    if isinstance(data, dict) and isinstance(data.get("enabled"), bool):
        return data["enabled"]
    return settings.local_api.enabled


def save_enabled(settings: Any, enabled: bool) -> None:
    path = local_api_state_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"enabled": bool(enabled)}), encoding="utf-8")

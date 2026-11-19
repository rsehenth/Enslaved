"""Ferramenta de leitura de arquivos autorizados do workspace (somente leitura)."""

import hashlib
import os
import re
import stat
from pathlib import Path

from pydantic import BaseModel, Field

from backend.tools.base import Tool, ToolResult, ToolSpec
from backend.core.config import FsToolsSettings, PROJECT_ROOT


class FsReadArgs(BaseModel):
    path: str = Field(min_length=1, max_length=1024, description="Caminho relativo às raízes autorizadas (ou absoluto dentro delas).")
    max_bytes: int | None = Field(default=None, ge=1, description="Limite opcional menor que o padrão configurado.")


_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


# Arquivos com credenciais, chaves, bancos e trilhas internas: nunca lidos pelo modelo, mesmo dentro das raízes.
_SENSITIVE_NAME = re.compile(
    r"^(\.env(\..*)?|\.netrc|_netrc|\.npmrc|\.pypirc|\.git-credentials|\.backend\.lock|tool-audit\.jsonl"
    r"|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?|credentials(\..*)?|secrets?(\..*)?|cookies(\..*)?|login data"
    r"|.*\.(pem|key|pfx|p12|kdbx|keychain|ovpn|db|db-wal|db-shm|db-journal|sqlite3?|sqlite-wal|sqlite-shm))$",
    re.IGNORECASE,
)
_SENSITIVE_DIRS = {".ssh", ".gnupg", ".git", ".aws", ".azure", ".kube", "backups"}


def _sensitive(parts) -> str | None:
    for part in parts:
        if part.lower() in _SENSITIVE_DIRS or _SENSITIVE_NAME.match(part):
            return part
    return None


class FsReadTool(Tool):
    args_model = FsReadArgs

    def __init__(self, fs_settings: FsToolsSettings):
        super().__init__()
        self._settings = fs_settings
        self._roots = [Path(root).resolve() for root in fs_settings.allowed_roots]
        self.spec = ToolSpec(
            id="fs.read",
            name="File read",
            description=(
                "Reads a text file inside the project's authorized roots (data, docs, skills, config). "
                "Read-only; rejects directories, binary files, and any path outside the roots."
            ),
            capabilities=["fs", "read-only"],
            required_permissions=["fs.read"],
        )

    def _resolve(self, raw: str) -> Path:
        """Resolve o caminho e exige que o destino caiba em uma raiz autorizada.

        Aceita caminho absoluto dentro das raízes, relativo a uma raiz
        ("docs/nota.md") ou relativo à raiz do projeto ("data/nota.md") — a
        contenção na raiz autorizada é sempre verificada após `resolve()`
        (resolve symlinks, bloqueando travessia).
        """
        if "\x00" in raw:
            raise PermissionError("Caminho inválido (caractere nulo).")
        candidate = Path(raw)
        # UNC (\\host\share), \\?\ e \\.\ : recusados ANTES do resolve(), que abriria o caminho (SMB/NTLM para fora)
        if raw.replace("/", "\\").startswith("\\\\") or (candidate.drive and not re.fullmatch(r"[A-Za-z]:", candidate.drive)):
            raise PermissionError(f"Caminho de rede ou de dispositivo não permitido: {raw}")
        # Bloqueia NTFS Alternate Data Streams (:) fora de letra de unidade e nomes reservados Windows
        parts_to_check = candidate.parts[1:] if candidate.drive else candidate.parts
        for part in parts_to_check:
            if ":" in part:
                raise PermissionError(f"Caminho inválido (fluxo alternativo não permitido): {raw}")
            stem = part.split(".", 1)[0].upper()
            if stem in _WINDOWS_RESERVED:
                raise PermissionError(f"Caminho reservado do sistema não permitido: {raw}")
        if candidate.is_absolute():
            candidates = [candidate]
        else:
            candidates = [PROJECT_ROOT / candidate] + [root / candidate for root in self._roots]
        for path in candidates:
            resolved = path.resolve()
            for root in self._roots:
                if resolved == root or resolved.is_relative_to(root):
                    # resolve() diferente do caminho léxico = symlink, junção, reparse point ou nome 8.3 no meio
                    if os.path.normcase(os.path.abspath(path)) != os.path.normcase(str(resolved)):
                        raise PermissionError(f"Caminho passa por link simbólico ou junção (não permitido): {raw}")
                    hit = _sensitive(resolved.relative_to(root).parts)
                    if hit:
                        raise PermissionError(f"Arquivo sensível (credenciais, chaves ou dados internos) não pode ser lido: {hit}")
                    return resolved
        raise PermissionError(
            f"Caminho fora das raízes autorizadas: {raw} (raízes: {', '.join(str(root) for root in self._roots)})"
        )

    async def run(self, args: dict, permission=None) -> ToolResult:
        if permission is None or not permission.granted:
            return ToolResult(ok=False, error="permissão ausente")
        raw_path = args["path"]
        limit = min(args.get("max_bytes") or self._settings.max_file_bytes, self._settings.max_file_bytes)
        try:
            resolved = self._resolve(raw_path)
        except PermissionError as exc:
            return ToolResult(ok=False, error=str(exc))
        try:
            before = os.lstat(resolved)
        except FileNotFoundError:
            return ToolResult(ok=False, error=f"Arquivo não encontrado: {raw_path}")
        except OSError as exc:
            return ToolResult(ok=False, error=f"Falha ao ler {raw_path}: {exc.__class__.__name__}")
        if not stat.S_ISREG(before.st_mode):
            return ToolResult(ok=False, error=f"O caminho não é um arquivo: {raw_path}")
        if getattr(before, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT or before.st_nlink > 1:
            return ToolResult(ok=False, error=f"Arquivo com link (reparse point ou hard link) não permitido: {raw_path}")
        try:
            fd = os.open(resolved, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            with os.fdopen(fd, "rb") as file:
                opened = os.fstat(file.fileno())
                # o recurso aberto tem de ser o mesmo arquivo validado (troca entre checagem e abertura = recusa)
                if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                    return ToolResult(ok=False, error=f"O arquivo mudou durante a leitura: {raw_path}")
                size = opened.st_size
                payload = file.read(limit)
        except OSError as exc:
            return ToolResult(ok=False, error=f"Falha ao ler {raw_path}: {exc.__class__.__name__}")
        if b"\x00" in payload:
            return ToolResult(ok=False, error=f"Arquivo binário não suportado: {raw_path}")
        truncated = size > len(payload)
        text = payload.decode("utf-8", errors="replace")
        return ToolResult(
            ok=True,
            output=text,
            data={
                "path": raw_path,
                "resolved_path": str(resolved),
                "size_bytes": size,
                "returned_bytes": len(payload),
                "truncated": truncated,
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )

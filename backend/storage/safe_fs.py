"""Remoção segura e validação estrita de arquivos exclusivos de conversa."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_PROTECTED_NAMES = {
    "tool-audit.jsonl",
    ".backend.lock",
    "skills-state.json",
    "library.json",
}
_PROTECTED_DIRS = {"config", "models", "knowledge", "backups", ".git", ".ssh", ".gnupg", ".aws", ".azure", ".kube"}
_PROTECTED_DB_RE = re.compile(r".*\.(db|db-wal|db-shm|db-journal|sqlite3?|sqlite-wal|sqlite-shm)$", re.IGNORECASE)


def _has_reparse_or_symlink(st: os.stat_result) -> bool:
    return bool(
        stat.S_ISLNK(st.st_mode)
        or (getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
    )


def _check_lexical_path(path: Path) -> None:
    raw = str(path)
    if "\x00" in raw:
        raise PermissionError("Caminho inválido (caractere nulo).")
    if raw.replace("/", "\\").startswith("\\\\") or (path.drive and not re.fullmatch(r"[A-Za-z]:", path.drive)):
        raise PermissionError(f"Caminho de rede ou dispositivo não permitido: {raw}")
    parts = path.parts[1:] if path.drive else path.parts
    for part in parts:
        if part == "..":
            raise PermissionError(f"Travessia de diretório não permitida: {raw}")
        if ":" in part:
            raise PermissionError(f"Fluxo alternativo (ADS) não permitido: {raw}")
        stem = part.split(".", 1)[0].upper()
        if stem in _WINDOWS_RESERVED:
            raise PermissionError(f"Nome reservado do sistema não permitido: {raw}")


def validate_safe_target(target: Path, allowed_root: Path, *, expect_dir: bool = False) -> Path:
    """Garante que `target` reside dentro de `allowed_root` sem symlinks, junções, hard links ou arquivos protegidos."""
    _check_lexical_path(allowed_root)
    _check_lexical_path(target)

    abs_root = Path(os.path.abspath(allowed_root))
    abs_target = Path(os.path.abspath(target))
    try:
        rel_lex = abs_target.relative_to(abs_root)
    except ValueError as exc:
        raise PermissionError(f"Caminho fora do diretório autorizado: {target}") from exc
    if not rel_lex.parts:
        raise PermissionError("Não é permitido remover o diretório raiz autorizado.")

    if os.path.lexists(abs_root):
        root_st = os.lstat(abs_root)
        if _has_reparse_or_symlink(root_st) or not stat.S_ISDIR(root_st.st_mode):
            raise PermissionError(f"Diretório autorizado inválido ou com link: {allowed_root}")

    resolved_root = allowed_root.resolve()
    resolved_target = target.resolve()
    if resolved_target == resolved_root or not resolved_target.is_relative_to(resolved_root):
        raise PermissionError(f"Destino resolvido escapa do diretório autorizado: {target}")

    # Caminha cada segmento dentro de resolved_root para bloquear symlinks/junções intermediários
    cursor = resolved_root
    for idx, part in enumerate(rel_lex.parts):
        if part.lower() in _PROTECTED_DIRS or part.lower() in _PROTECTED_NAMES or _PROTECTED_DB_RE.match(part):
            raise PermissionError(f"Caminho protegido não pode ser removido: {part}")
        cursor = cursor / part
        if os.path.lexists(cursor):
            st = os.lstat(cursor)
            if _has_reparse_or_symlink(st):
                raise PermissionError(f"Link simbólico ou junção não permitido: {cursor}")
            is_last = idx == len(rel_lex.parts) - 1
            if not is_last and not stat.S_ISDIR(st.st_mode):
                raise PermissionError(f"Segmento intermediário inválido: {cursor}")

    if os.path.normcase(str(resolved_root / rel_lex)) != os.path.normcase(str(resolved_target)):
        raise PermissionError(f"Caminho resolvido diverge do caminho léxico (link/junção): {target}")

    if os.path.lexists(resolved_target):
        leaf_st = os.lstat(resolved_target)
        if _has_reparse_or_symlink(leaf_st):
            raise PermissionError(f"Arquivo ou pasta com link não permitido: {target}")
        if expect_dir:
            if not stat.S_ISDIR(leaf_st.st_mode):
                raise PermissionError(f"Esperado diretório para limpeza: {target}")
        else:
            if not stat.S_ISREG(leaf_st.st_mode):
                raise PermissionError(f"Alvo de remoção não é arquivo regular: {target}")
            if leaf_st.st_nlink > 1:
                raise PermissionError(f"Hard link não permitido para remoção: {target}")

    return resolved_target


def safe_unlink_within(target: Path, allowed_root: Path) -> bool:
    """Remove um arquivo regular somente após validação estrita dentro de `allowed_root`."""
    resolved = validate_safe_target(target, allowed_root, expect_dir=False)
    if not os.path.lexists(resolved):
        return False
    resolved.unlink()
    return True


def safe_rmdir_if_empty(dir_path: Path, allowed_root: Path) -> bool:
    """Remove um diretório vazio (sem recursão) somente após validação estrita dentro de `allowed_root`."""
    resolved = validate_safe_target(dir_path, allowed_root, expect_dir=True)
    if not os.path.lexists(resolved):
        return False
    if any(resolved.iterdir()):
        return False
    resolved.rmdir()
    return True

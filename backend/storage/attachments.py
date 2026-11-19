"""Anexos de imagem do chat: validação do arquivo real, armazenamento controlado e vínculo com a conversa.

- O cliente nunca informa caminhos: só envia bytes (upload) e depois referencia o `id` (uuid4 hex) devolvido.
- Fluxo: upload -> anexo *pendente* (`_pending/`) -> vinculado à conversa/mensagem no envio (`<conversa>/<id>.<ext>`).
- Exclusão da conversa remove o diretório exclusivo (mesma coreografia de staging/commit/limpeza das demais pastas).
- A decodificação/normalização da imagem acontece em subprocesso isolado (`image_sandbox.py`), nunca neste processo.
- Nada aqui chama rede ou serviços externos.
"""

from __future__ import annotations

import base64
import hashlib
import logging
import math
import os
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from backend.core.config import VisionSettings
from backend.storage.database import Database
from backend.storage.image_sandbox import SandboxRejected, run_image_job
from backend.storage.safe_fs import safe_rmdir_if_empty, safe_unlink_within, validate_safe_target

logger = logging.getLogger(__name__)

ATTACHMENT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
PENDING_DIR = "_pending"
# formato real detectado pelo Pillow -> (mime, extensão)
IMAGE_FORMATS = {"PNG": ("image/png", "png"), "JPEG": ("image/jpeg", "jpg"), "WEBP": ("image/webp", "webp")}
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
)
# Qwen 3.5 (projetor clip): ~1 token por bloco de 32x32 px, teto de ~4096 tokens por imagem (medido no Ollama 0.40).
_PATCH = 32
_OLLAMA_IMAGE_TOKEN_CAP = 4096
_IMAGE_SPECIAL_TOKENS = 4


class AttachmentRejected(ValueError):
    """Anexo recusado (formato, tamanho, dimensões ou conteúdo inválido)."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass
class ImageInfo:
    format: str
    mime: str
    ext: str
    width: int
    height: int


def validate_attachment_id(attachment_id: str) -> str:
    if not isinstance(attachment_id, str) or not ATTACHMENT_ID_RE.fullmatch(attachment_id):
        raise AttachmentRejected(f"Identificador de anexo inválido: {attachment_id!r}")
    return attachment_id


def _grid(value: int) -> int:
    return max(1, int(value / _PATCH + 0.5))


def image_tokens(width: int, height: int, cap: int | None = None) -> int:
    """Tokens que a imagem ocupa no Qwen 3.5 (medidos: erro 900x360=308, 1920x1080=2040, 4 imagens=2766)."""
    raw = min(_OLLAMA_IMAGE_TOKEN_CAP, _grid(width) * _grid(height))
    if cap is not None:
        raw = min(raw, cap)
    return raw + _IMAGE_SPECIAL_TOKENS


def fit_dimensions(width: int, height: int, max_tokens: int) -> tuple[int, int]:
    """Dimensões da cópia enviada ao modelo para que a imagem caiba em `max_tokens` (mantém a proporção)."""
    if image_tokens(width, height) - _IMAGE_SPECIAL_TOKENS <= max_tokens:
        return width, height
    scale = math.sqrt(max_tokens / (_grid(width) * _grid(height)))
    w, h = max(_PATCH, int(width * scale)), max(_PATCH, int(height * scale))
    while _grid(w) * _grid(h) > max_tokens and (w > _PATCH or h > _PATCH):
        w, h = max(_PATCH, int(w * 0.97)), max(_PATCH, int(h * 0.97))
    return w, h


def sanitize_display_name(raw: str) -> str:
    """Nome apenas para exibição: nunca vira caminho. Sem separadores, controles ou marcação."""
    base = re.split(r"[\\/]", raw or "")[-1]
    clean = re.sub(r"[^\w .()\-]", "_", base, flags=re.UNICODE).strip(" .")
    return (clean[:60] or "imagem")


def precheck_image(data: bytes, cfg: VisionSettings) -> None:
    """Triagem barata só com bytes (sem decodificar): vazio, tamanho e assinatura real. Extensão/MIME do cliente são ignorados."""
    if not data:
        raise AttachmentRejected("Arquivo vazio.")
    if len(data) > cfg.max_image_bytes:
        raise AttachmentRejected(f"Imagem acima do limite de {cfg.max_image_bytes // (1024 * 1024)} MB.", 413)
    is_webp = data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    if not (is_webp or any(data.startswith(sig) for sig, _ in _MAGIC)):
        raise AttachmentRejected("Formato não suportado. Use PNG, JPEG ou WebP.", 415)


def normalize_image(data: bytes, cfg: VisionSettings) -> tuple[bytes, ImageInfo, bool]:
    """Valida e normaliza em subprocesso isolado. Devolve (bytes recodificados sem metadados, info, metadados_removidos)."""
    precheck_image(data, cfg)
    params = {"max_side": cfg.max_image_side, "max_pixels": cfg.max_image_pixels}
    try:
        meta, out = run_image_job(
            "normalize", data, params,
            timeout=cfg.processing_timeout_seconds, memory_limit=cfg.processing_memory_mb * 1024 * 1024, max_output=cfg.max_image_bytes * 2,
        )
    except SandboxRejected as exc:
        raise AttachmentRejected(exc.message, exc.status) from exc
    info = ImageInfo(format=meta["format"], mime=meta["mime"], ext=meta["ext"], width=meta["width"], height=meta["height"])
    return out, info, bool(meta.get("metadata_removed"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AttachmentStore:
    """Anexos de imagem: tabela `attachments` + arquivos em `<data>/attachments/`."""

    def __init__(self, db: Database, base_dir: Path, cfg: VisionSettings | None = None, io_lock: threading.RLock | None = None):
        self.db = db
        self.cfg = cfg or VisionSettings()
        self.base_dir = Path(base_dir)
        self.pending_dir = self.base_dir / PENDING_DIR
        # o mesmo lock do histórico Markdown: exclusão de conversa, vínculo e repack nunca se intercalam
        self._io_lock = io_lock or threading.RLock()

    # ------------------------------------------------------------------ util

    @staticmethod
    def _public(row: Any) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["display_name"],
            "mime": row["mime"],
            "width": row["width"],
            "height": row["height"],
            "size_bytes": row["size_bytes"],
            "tokens": image_tokens(row["width"], row["height"]),
        }

    def get(self, attachment_id: str) -> dict[str, Any] | None:
        if not ATTACHMENT_ID_RE.fullmatch(attachment_id or ""):
            return None
        row = self.db.read_one("SELECT * FROM attachments WHERE id = ?", (attachment_id,))
        return dict(row) if row else None

    def path_of(self, row: dict[str, Any]) -> Path:
        """Caminho validado (contido na pasta de anexos, sem links) do arquivo de um registro."""
        return validate_safe_target(self.base_dir / row["rel_path"], self.base_dir)

    def conversation_attachments(self, conversation_id: str) -> list[dict[str, Any]]:
        rows = self.db.read_all("SELECT * FROM attachments WHERE conversation_id = ? ORDER BY created_at", (conversation_id,))
        return [self._public(r) for r in rows]

    # ------------------------------------------------------------------ upload (pendente)

    def save_pending(self, data: bytes, filename: str) -> dict[str, Any]:
        """Valida/normaliza FORA do lock (subprocesso) e só então grava o arquivo normalizado como anexo pendente."""
        normalized, info, metadata_removed = normalize_image(data, self.cfg)
        with self._io_lock:
            self.cleanup_expired_pending()
            pending = self.db.read_one("SELECT COUNT(*) AS n FROM attachments WHERE conversation_id IS NULL")["n"]
            if pending >= self.cfg.max_pending:
                raise AttachmentRejected("Há anexos demais aguardando envio. Remova alguns antes de anexar outros.", 429)
            attachment_id = uuid.uuid4().hex  # aleatório (122 bits): não previsível; o nome em disco é sempre gerado aqui
            rel = f"{PENDING_DIR}/{attachment_id}.{info.ext}"
            target = self.base_dir / rel
            self._atomic_write(target, normalized)
            try:
                self.db.write(
                    """
                    INSERT INTO attachments (id, conversation_id, message_id, kind, mime, ext, sha256, size_bytes,
                                             width, height, display_name, rel_path, created_at)
                    VALUES (?, NULL, NULL, 'image', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (attachment_id, info.mime, info.ext, hashlib.sha256(normalized).hexdigest(), len(normalized), info.width,
                     info.height, sanitize_display_name(filename), rel, _now()),
                )
            except Exception:
                self._silent_unlink(target)
                raise
            public = self._public(self.db.read_one("SELECT * FROM attachments WHERE id = ?", (attachment_id,)))
            public["metadata_removed"] = metadata_removed
            return public

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        except Exception:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def _silent_unlink(self, path: Path) -> None:
        try:
            safe_unlink_within(path, self.base_dir)
        except (OSError, PermissionError) as exc:
            logger.warning("Não foi possível remover anexo %s: %s", path, exc)

    def discard_pending(self, attachment_id: str) -> bool:
        """Remove um anexo ainda não enviado (preview removido pelo usuário). Anexos de conversas não são removíveis aqui."""
        validate_attachment_id(attachment_id)
        with self._io_lock:
            row = self.get(attachment_id)
            if row is None or row["conversation_id"] is not None:
                return False
            self.db.write("DELETE FROM attachments WHERE id = ? AND conversation_id IS NULL", (attachment_id,))
            self._silent_unlink(self.base_dir / row["rel_path"])
            return True

    def cleanup_expired_pending(self) -> int:
        limit = (datetime.now(timezone.utc) - timedelta(seconds=self.cfg.pending_ttl_seconds)).isoformat()
        removed = 0
        with self._io_lock:
            for row in self.db.read_all("SELECT * FROM attachments WHERE conversation_id IS NULL AND created_at < ?", (limit,)):
                self.db.write("DELETE FROM attachments WHERE id = ? AND conversation_id IS NULL", (row["id"],))
                self._silent_unlink(self.base_dir / row["rel_path"])
                removed += 1
        return removed

    # ------------------------------------------------------------------ vínculo com a mensagem

    def check_pending(self, attachment_ids: list[str]) -> list[dict[str, Any]]:
        """Valida ids enviados pelo cliente (formato, existência, ainda pendentes, limite). Devolve os registros."""
        ids = list(dict.fromkeys(validate_attachment_id(i) for i in attachment_ids))
        if len(ids) > self.cfg.max_images_per_message:
            raise AttachmentRejected(f"No máximo {self.cfg.max_images_per_message} imagens por mensagem.")
        rows: list[dict[str, Any]] = []
        for attachment_id in ids:
            row = self.get(attachment_id)
            if row is None or row["conversation_id"] is not None:
                raise AttachmentRejected("Anexo inexistente ou já utilizado. Anexe a imagem novamente.")
            rows.append(row)
        return rows

    def bind(self, conversation_id: str, attachment_ids: list[str], message_id: int, moves: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
        """Vincula anexos pendentes à mensagem. Chamar dentro de `db.transaction()` e sob o lock de E/S.

        `moves` acumula (origem, destino) para o chamador desfazer se a transação falhar. Nunca recria
        anexos de conversa excluída: a existência da conversa é checada sob o mesmo lock da exclusão.
        """
        rows = self.check_pending(attachment_ids)
        if self.db.read_one("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)) is None:
            raise AttachmentRejected("Conversa inexistente ou em exclusão.", 404)
        bound: list[dict[str, Any]] = []
        for row in rows:
            rel = f"{conversation_id}/{row['id']}.{row['ext']}"
            src = self.base_dir / row["rel_path"]
            dst = self.base_dir / rel
            validate_safe_target(src, self.base_dir)
            dst.parent.mkdir(parents=True, exist_ok=True)
            validate_safe_target(dst.parent, self.base_dir, expect_dir=True)
            os.replace(src, dst)
            moves.append((src, dst))
            self.db.write(
                "UPDATE attachments SET conversation_id = ?, message_id = ?, rel_path = ? WHERE id = ?",
                (conversation_id, message_id, rel, row["id"]),
            )
            bound.append(self._public({**row, "conversation_id": conversation_id, "rel_path": rel}))
        return bound

    @staticmethod
    def undo_moves(moves: list[tuple[Path, Path]]) -> None:
        for src, dst in reversed(moves):
            try:
                if os.path.lexists(dst):
                    os.replace(dst, src)
            except OSError as exc:
                logger.warning("Falha ao desfazer vínculo de anexo %s: %s", dst, exc)
        moves.clear()

    # ------------------------------------------------------------------ envio ao modelo

    def load_for_model(self, attachment_id: str, max_tokens: int) -> tuple[str, int, int, int]:
        """(base64, largura, altura, tokens) da cópia enviada ao Ollama. O arquivo armazenado já é normalizado e sem
        metadados; imagens acima de `max_tokens` são reduzidas por um subprocesso isolado (nunca no processo principal).
        Confere o hash gravado: arquivo adulterado em disco não é enviado ao modelo."""
        row = self.get(attachment_id)
        if row is None:
            raise FileNotFoundError(attachment_id)
        data = self.path_of(row).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise AttachmentRejected("Integridade do anexo comprometida (hash divergente).", 409)
        width, height = row["width"], row["height"]
        sent_w, sent_h = fit_dimensions(width, height, max_tokens)
        if (sent_w, sent_h) != (width, height):
            try:
                _meta, data = run_image_job(
                    "resize", data, {"max_side": self.cfg.max_image_side, "max_pixels": self.cfg.max_image_pixels, "width": sent_w, "height": sent_h},
                    timeout=self.cfg.processing_timeout_seconds, memory_limit=self.cfg.processing_memory_mb * 1024 * 1024,
                    max_output=self.cfg.max_image_bytes * 2,
                )
            except SandboxRejected as exc:
                raise AttachmentRejected(exc.message, exc.status) from exc
        return base64.b64encode(data).decode("ascii"), sent_w, sent_h, image_tokens(sent_w, sent_h)

    # ------------------------------------------------------------------ exclusão / reconciliação

    def collect_conversation_targets(self, conversation_ids: list[str]) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
        """Arquivos/pastas exclusivos das conversas (para o staging da exclusão). Arquivos que outra conversa
        ainda referencia (mesmo caminho) são preservados."""
        files: list[tuple[Path, Path]] = []
        dirs: list[tuple[Path, Path]] = []
        seen: set[Path] = set()
        doomed = set(conversation_ids)
        for cid in conversation_ids:
            conv_dir = self.base_dir / cid
            for row in self.db.read_all("SELECT rel_path FROM attachments WHERE conversation_id = ?", (cid,)):
                shared = self.db.read_one(
                    "SELECT conversation_id FROM attachments WHERE rel_path = ? AND conversation_id IS NOT ?", (row["rel_path"], cid)
                )
                if shared is not None and shared["conversation_id"] not in doomed:
                    continue
                self._collect_file(self.base_dir / row["rel_path"], files, seen)
            if os.path.lexists(conv_dir):
                validated = validate_safe_target(conv_dir, self.base_dir, expect_dir=True)
                for entry in list(validated.iterdir()):
                    self._collect_file(entry, files, seen)
                dirs.append((validated, self.base_dir))
        return files, dirs

    def _collect_file(self, path: Path, files: list[tuple[Path, Path]], seen: set[Path]) -> None:
        resolved = validate_safe_target(path, self.base_dir, expect_dir=False)
        if os.path.lexists(resolved) and resolved not in seen:
            seen.add(resolved)
            files.append((resolved, self.base_dir))

    def reconcile(self) -> dict[str, int]:
        """Início do app: limpa resíduos de exclusão/gravação interrompida, pendentes vencidos e arquivos sem registro."""
        stats = {"stale": 0, "expired_pending": 0, "orphan_files": 0, "dangling_rows": 0}
        if not self.base_dir.is_dir():
            stats["expired_pending"] = self.cleanup_expired_pending()
            return stats
        with self._io_lock:
            stats["expired_pending"] = self.cleanup_expired_pending()
            known = {os.path.normcase(str(self.base_dir / r["rel_path"])) for r in self.db.read_all("SELECT rel_path FROM attachments")}
            for entry in list(self.base_dir.rglob("*")):
                if not entry.is_file() and not entry.is_symlink():
                    continue
                if entry.name.endswith((".deleting", ".tmp")):
                    stats["stale"] += int(self._try_unlink(entry))
                elif os.path.normcase(str(entry)) not in known:
                    stats["orphan_files"] += int(self._try_unlink(entry))
            for row in self.db.read_all("SELECT id, rel_path, conversation_id FROM attachments WHERE conversation_id IS NULL"):
                if not os.path.lexists(self.base_dir / row["rel_path"]):
                    self.db.write("DELETE FROM attachments WHERE id = ?", (row["id"],))
                    stats["dangling_rows"] += 1
            for conv_dir in list(self.base_dir.iterdir()):
                if conv_dir.is_dir() and conv_dir.name != PENDING_DIR:
                    try:
                        safe_rmdir_if_empty(conv_dir, self.base_dir)
                    except (OSError, PermissionError):
                        pass
        return stats

    def _try_unlink(self, path: Path) -> bool:
        try:
            return safe_unlink_within(path, self.base_dir)
        except (OSError, PermissionError) as exc:
            logger.warning("Não foi possível limpar %s: %s", path, exc)
            return False

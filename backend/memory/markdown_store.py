from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backend.storage.conversations import Conversation, ConversationRepository, Message
from backend.storage.database import Database
from backend.storage.safe_fs import safe_rmdir_if_empty, safe_unlink_within, validate_safe_target

logger = logging.getLogger(__name__)

_ROLE_LABELS = {
    "user": "Usuário",
    "assistant": "Assistente (Enslaved)",
    "tool": "Ferramenta",
    "system": "Sistema",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, content: str) -> None:
    """Grava arquivo de forma atômica (temp + os.replace) evitando corrupção parcial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise


@dataclass
class ExportResult:
    ok: bool
    path: Path
    exported_seq: int
    error: str | None = None


class MarkdownHistoryStore:
    """Sincroniza conversas e repacks do SQLite para arquivos Markdown legíveis e atômicos."""

    def __init__(self, db: Database, base_dir: Path):
        self.db = db
        self.base_dir = Path(base_dir)
        self.conversations_dir = self.base_dir / "conversations"
        self.repacks_dir = self.base_dir / "repacks"
        self.repo = ConversationRepository(db)
        self._io_lock = threading.RLock()

    def conversation_path(self, conversation_id: str) -> Path:
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in conversation_id)
        return self.conversations_dir / f"{safe_id}.md"

    def repack_path(self, conversation_id: str, version: int) -> Path:
        safe_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in conversation_id)
        return self.repacks_dir / safe_id / f"v{version:03d}.md"

    def render_conversation_markdown(self, conversation: Conversation, messages: list[Message]) -> str:
        repack_rows = self.db.read_all(
            "SELECT version, covered_from_seq, covered_to_seq, markdown_path, created_at "
            "FROM repacks WHERE conversation_id = ? ORDER BY version",
            (conversation.id,),
        )
        lines = [
            f"# Conversa: {conversation.title}",
            "",
            f"- **ID da conversa:** `{conversation.id}`",
            f"- **Criada em:** `{conversation.created_at}`",
            f"- **Atualizada em:** `{conversation.updated_at}`",
            f"- **Total de mensagens:** {len(messages)}",
        ]
        if repack_rows:
            refs = ", ".join(
                f"v{r['version']:03d} (seq {r['covered_from_seq']}-{r['covered_to_seq']}: `{r['markdown_path']}`)"
                for r in repack_rows
            )
            lines.append(f"- **Resumos versionados (repacks):** {refs}")
        lines.extend(["", "---", ""])

        for msg in messages:
            role_label = _ROLE_LABELS.get(msg.role, msg.role)
            lines.append(f"## [{msg.seq}] {role_label} (`msg#{msg.id}`)")
            lines.append(f"- **Papel:** `{msg.role}`")
            lines.append(f"- **Data:** `{msg.created_at}`")
            if msg.meta:
                meta_clean = dict(msg.meta)
                if "provider" in meta_clean or "model" in meta_clean:
                    lines.append(
                        f"- **Modelo:** `{meta_clean.get('provider', '?')} / {meta_clean.get('model', '?')}`"
                    )
                if "tool_name" in meta_clean:
                    lines.append(
                        f"- **Ferramenta:** `{meta_clean['tool_name']}` (ok={meta_clean.get('ok')})"
                    )
                if "tool_calls" in meta_clean and meta_clean["tool_calls"]:
                    calls_desc = ", ".join(
                        f"{c.get('name')}({c.get('arguments')})" for c in meta_clean["tool_calls"]
                    )
                    lines.append(f"- **Chamadas solicitadas:** `{calls_desc}`")
                if "tools_used" in meta_clean and meta_clean["tools_used"]:
                    lines.append(
                        f"- **Ferramentas utilizadas:** `{json.dumps(meta_clean['tools_used'], ensure_ascii=False)}`"
                    )
                if "attachments" in meta_clean and meta_clean["attachments"]:
                    att_desc = ", ".join(
                        f"`att:{str(a.get('id', ''))[:8]}` ({a.get('name', 'imagem')}, {a.get('width', '?')}x{a.get('height', '?')}, {a.get('mime', '?')})"
                        for a in meta_clean["attachments"]
                        if isinstance(a, dict)
                    )
                    if att_desc:
                        lines.append(f"- **Anexos:** {att_desc}")
            lines.append("")
            lines.append(msg.content if msg.content else "_(sem conteúdo textual)_")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def export_conversation(self, conversation_id: str) -> ExportResult:
        """Exporta integralmente a conversa do SQLite para Markdown de forma atômica."""
        with self._io_lock:
            conversation = self.repo.get_conversation(conversation_id)
            if conversation is None:
                raise ValueError(f"Conversa não encontrada para exportação: {conversation_id}")
            messages = self.repo.get_messages(conversation_id)
            target = self.conversation_path(conversation_id)
            max_seq = messages[-1].seq if messages else 0
            try:
                markdown = self.render_conversation_markdown(conversation, messages)
                atomic_write_text(target, markdown)
            except Exception as exc:
                logger.warning("Falha ao exportar conversa %s para Markdown: %s", conversation_id, exc)
                if self.repo.conversation_exists(conversation_id):
                    self._record_export_status(conversation_id, str(target), 0, "error", str(exc))
                return ExportResult(ok=False, path=target, exported_seq=0, error=str(exc))

            if self.repo.conversation_exists(conversation_id):
                self._record_export_status(conversation_id, str(target), max_seq, "ok", None)
            return ExportResult(ok=True, path=target, exported_seq=max_seq)

    def _record_export_status(
        self,
        conversation_id: str,
        markdown_path: str,
        last_exported_seq: int,
        status: str,
        last_error: str | None,
    ) -> None:
        now = _now()
        existing = self.db.read_one(
            "SELECT last_exported_seq FROM conversation_exports WHERE conversation_id = ?",
            (conversation_id,),
        )
        seq_to_store = last_exported_seq if status == "ok" else (existing["last_exported_seq"] if existing else 0)
        self.db.write(
            """
            INSERT INTO conversation_exports
                (conversation_id, markdown_path, last_exported_seq, last_exported_at, status, last_error)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                markdown_path = excluded.markdown_path,
                last_exported_seq = ?,
                last_exported_at = CASE WHEN excluded.status = 'ok' THEN excluded.last_exported_at ELSE conversation_exports.last_exported_at END,
                status = excluded.status,
                last_error = excluded.last_error
            """,
            (
                conversation_id,
                markdown_path,
                seq_to_store,
                now if status == "ok" else None,
                status,
                last_error,
                seq_to_store,
            ),
        )

    def get_export_status(self, conversation_id: str) -> dict | None:
        row = self.db.read_one(
            "SELECT * FROM conversation_exports WHERE conversation_id = ?",
            (conversation_id,),
        )
        return dict(row) if row else None

    def cleanup_stale_deletions(self) -> int:
        """Recupera falhas intermediárias de remoção limpando arquivos *.deleting e pastas vazias órfãs."""
        cleaned = 0
        with self._io_lock:
            if self.conversations_dir.is_dir():
                for entry in list(self.conversations_dir.iterdir()):
                    if entry.name.endswith(".deleting"):
                        try:
                            if safe_unlink_within(entry, self.conversations_dir):
                                cleaned += 1
                        except (OSError, PermissionError) as exc:
                            logger.warning("Não foi possível limpar arquivo temporário %s: %s", entry, exc)
            if self.repacks_dir.is_dir():
                for conv_dir in list(self.repacks_dir.iterdir()):
                    try:
                        validate_safe_target(conv_dir, self.repacks_dir, expect_dir=True)
                    except PermissionError:
                        continue
                    if not conv_dir.is_dir():
                        continue
                    for entry in list(conv_dir.iterdir()):
                        if entry.name.endswith(".deleting"):
                            try:
                                if safe_unlink_within(entry, self.repacks_dir):
                                    cleaned += 1
                            except (OSError, PermissionError) as exc:
                                logger.warning("Não foi possível limpar arquivo temporário %s: %s", entry, exc)
                    if not self.repo.conversation_exists(conv_dir.name):
                        try:
                            if safe_rmdir_if_empty(conv_dir, self.repacks_dir):
                                cleaned += 1
                        except (OSError, PermissionError):
                            pass
        return cleaned

    def collect_conversation_targets(
        self, conversation_ids: list[str]
    ) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
        """Coleta e valida estritamente todos os arquivos e diretórios exclusivos das conversas informadas."""
        file_targets: list[tuple[Path, Path]] = []
        dir_targets: list[tuple[Path, Path]] = []
        seen_files: set[Path] = set()
        seen_dirs: set[Path] = set()

        def _add_file(path: Path, allowed_root: Path) -> None:
            resolved = validate_safe_target(path, allowed_root, expect_dir=False)
            if os.path.lexists(resolved) and resolved not in seen_files:
                seen_files.add(resolved)
                file_targets.append((resolved, allowed_root))

        def _add_dir(path: Path, allowed_root: Path) -> None:
            resolved = validate_safe_target(path, allowed_root, expect_dir=True)
            if os.path.lexists(resolved) and resolved not in seen_dirs:
                seen_dirs.add(resolved)
                dir_targets.append((resolved, allowed_root))

        for cid in conversation_ids:
            _add_file(self.conversation_path(cid), self.conversations_dir)
            export_row = self.db.read_one(
                "SELECT markdown_path FROM conversation_exports WHERE conversation_id = ?",
                (cid,),
            )
            if export_row and export_row["markdown_path"]:
                _add_file(Path(export_row["markdown_path"]), self.conversations_dir)

            repack_dir = self.repacks_dir / "".join(
                ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in cid
            )
            if os.path.lexists(repack_dir):
                validated_dir = validate_safe_target(repack_dir, self.repacks_dir, expect_dir=True)
                for entry in list(validated_dir.iterdir()):
                    _add_file(entry, self.repacks_dir)
                _add_dir(validated_dir, self.repacks_dir)

            repack_rows = self.db.read_all(
                "SELECT markdown_path FROM repacks WHERE conversation_id = ?",
                (cid,),
            )
            for row in repack_rows:
                if row["markdown_path"]:
                    _add_file(Path(row["markdown_path"]), self.repacks_dir)

        return file_targets, dir_targets

    def reconcile_pending(self) -> dict[str, int]:
        """Reconcilia conversas ainda não exportadas, desatualizadas ou com erro prévio."""
        self.cleanup_stale_deletions()
        rows = self.db.read_all(
            """
            SELECT c.id,
                   COALESCE((SELECT MAX(m.seq) FROM messages m WHERE m.conversation_id = c.id), 0) AS max_seq,
                   ce.status,
                   COALESCE(ce.last_exported_seq, -1) AS exported_seq,
                   ce.markdown_path
            FROM conversations c
            LEFT JOIN conversation_exports ce ON ce.conversation_id = c.id
            """
        )
        reconciled = 0
        failed = 0
        for row in rows:
            with self._io_lock:
                if not self.repo.conversation_exists(row["id"]):
                    continue
                md_path = Path(row["markdown_path"]) if row["markdown_path"] else self.conversation_path(row["id"])
                needs_export = (
                    row["status"] in (None, "pending", "error")
                    or row["max_seq"] > row["exported_seq"]
                    or not md_path.is_file()
                )
                if not needs_export:
                    continue
                res = self.export_conversation(row["id"])
                if res.ok:
                    reconciled += 1
                else:
                    failed += 1
        return {"reconciled": reconciled, "failed": failed}

    def read_conversation_markdown(self, conversation_id: str, ensure_sync: bool = True) -> tuple[str, Path]:
        path = self.conversation_path(conversation_id)
        if ensure_sync:
            res = self.export_conversation(conversation_id)
            if res.ok and res.path.is_file():
                return res.path.read_text(encoding="utf-8"), res.path
        if path.is_file():
            return path.read_text(encoding="utf-8"), path
        conversation = self.repo.get_conversation(conversation_id)
        if conversation is None:
            raise ValueError(f"Conversa não encontrada: {conversation_id}")
        messages = self.repo.get_messages(conversation_id)
        return self.render_conversation_markdown(conversation, messages), path

    def write_repack_markdown(
        self,
        conversation_id: str,
        version: int,
        covered_from_seq: int,
        covered_to_seq: int,
        source_message_ids: list[int],
        summary_text: str,
        confirmed_facts: list[str],
        hypotheses: list[str],
        decisions: list[str],
        pending_items: list[str],
        token_estimate_before: int,
        token_estimate_after: int,
        created_at: str,
        structured_json: str | None = None,
    ) -> Path:
        target = self.repack_path(conversation_id, version)
        orig_md = self.conversation_path(conversation_id)
        ids_str = ", ".join(f"`msg#{mid}`" for mid in source_message_ids) or "nenhum"

        def _bullets(items: list[str]) -> list[str]:
            return [f"- {item}" for item in items] if items else ["- _(nenhum)_"]

        lines = [
            f"# Repack v{version:03d} — Conversa `{conversation_id}`",
            "",
            f"- **Versão:** {version}",
            f"- **Criado em:** `{created_at}`",
            f"- **Faixa coberta (seq):** {covered_from_seq} a {covered_to_seq}",
            f"- **IDs das mensagens originais:** {ids_str}",
            f"- **Histórico original (preservado integralmente):** `{orig_md}`",
            f"- **Tokens estimados (antes -> depois):** {token_estimate_before} -> {token_estimate_after}",
            "",
            "## Fatos confirmados",
            *_bullets(confirmed_facts),
            "",
            "## Decisões",
            *_bullets(decisions),
            "",
            "## Hipóteses e informações incertas",
            *_bullets(hypotheses),
            "",
            "## Pendências",
            *_bullets(pending_items),
            "",
            "## Resumo consolidado",
            summary_text.strip() or "_(vazio)_",
            "",
        ]
        if structured_json:
            lines.extend(["## Estrutura (JSON)", "```json", structured_json.strip(), "```", ""])
        with self._io_lock:
            if not self.repo.conversation_exists(conversation_id):
                raise ValueError(f"Conversa removida antes de gravar repack: {conversation_id}")
            atomic_write_text(target, "\n".join(lines))
        return target

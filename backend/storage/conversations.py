import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel

from backend.storage.database import Database

DEFAULT_TITLE = "New chat"
ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
CONVERSATION_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def validate_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not CONVERSATION_ID_RE.fullmatch(conversation_id):
        raise ValueError(f"Identificador de conversa inválido: {conversation_id!r}")
    return conversation_id


class Conversation(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0
    preview: str | None = None


class Message(BaseModel):
    id: int
    conversation_id: str
    seq: int
    role: str
    content: str
    created_at: str
    meta: dict | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationRepository:
    def __init__(self, db: Database):
        self.db = db

    def create_conversation(self, title: str | None = None) -> Conversation:
        conversation_id = str(uuid.uuid4())
        ts = _now()
        final_title = (title or "").strip() or DEFAULT_TITLE
        self.db.write(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, final_title, ts, ts),
        )
        return Conversation(id=conversation_id, title=final_title, created_at=ts, updated_at=ts)

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        row = self.db.read_one(
            """
            SELECT c.*,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
                   (SELECT content FROM messages m WHERE m.conversation_id = c.id
                    ORDER BY m.seq DESC LIMIT 1) AS preview
            FROM conversations c
            WHERE c.id = ?
            """,
            (conversation_id,),
        )
        return self._conversation_from_row(row) if row else None

    def list_conversations(self, limit: int = 200) -> list[Conversation]:
        rows = self.db.read_all(
            """
            SELECT c.*,
                   (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
                   (SELECT content FROM messages m WHERE m.conversation_id = c.id
                    ORDER BY m.seq DESC LIMIT 1) AS preview
            FROM conversations c
            ORDER BY c.updated_at DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [self._conversation_from_row(row) for row in rows]

    def conversation_exists(self, conversation_id: str) -> bool:
        row = self.db.read_one("SELECT 1 FROM conversations WHERE id = ?", (conversation_id,))
        return row is not None

    def existing_conversation_ids(self, conversation_ids: list[str]) -> list[str]:
        if not conversation_ids:
            return []
        placeholders = ",".join("?" for _ in conversation_ids)
        rows = self.db.read_all(
            f"SELECT id FROM conversations WHERE id IN ({placeholders})",
            tuple(conversation_ids),
        )
        found = {row["id"] for row in rows}
        return [cid for cid in conversation_ids if cid in found]

    def delete_conversation(self, conversation_id: str) -> bool:
        cursor = self.db.write("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        return cursor.rowcount > 0

    def delete_conversations(self, conversation_ids: list[str]) -> int:
        if not conversation_ids:
            return 0
        placeholders = ",".join("?" for _ in conversation_ids)
        cursor = self.db.write(
            f"DELETE FROM conversations WHERE id IN ({placeholders})",
            tuple(conversation_ids),
        )
        return cursor.rowcount

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        meta: dict | None = None,
    ) -> Message:
        if role not in ALLOWED_ROLES:
            raise ValueError(f"Papel inválido: {role!r}")
        ts = _now()
        cursor = self.db.write(
            """
            INSERT INTO messages (conversation_id, seq, role, content, created_at, meta)
            VALUES (?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?),
                    ?, ?, ?, ?)
            """,
            (
                conversation_id,
                conversation_id,
                role,
                content,
                ts,
                json.dumps(meta, ensure_ascii=False) if meta is not None else None,
            ),
        )
        self.db.write("UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, conversation_id))
        if role == "user":
            conversation = self.get_conversation(conversation_id)
            if conversation and conversation.title in (DEFAULT_TITLE, "Nova conversa"):
                snippet = " ".join(content.split())[:60]
                if snippet:
                    self.db.write(
                        "UPDATE conversations SET title = ? WHERE id = ?",
                        (snippet, conversation_id),
                    )
        row = self.db.read_one("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,))
        return self._message_from_row(row)

    def get_messages(self, conversation_id: str) -> list[Message]:
        rows = self.db.read_all(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq",
            (conversation_id,),
        )
        return [self._message_from_row(row) for row in rows]

    def get_message(self, message_id: int) -> Message | None:
        row = self.db.read_one("SELECT * FROM messages WHERE id = ?", (message_id,))
        return self._message_from_row(row) if row else None

    def update_message_meta(self, message_id: int, meta: dict | None) -> Message | None:
        raw = json.dumps(meta, ensure_ascii=False) if meta is not None else None
        self.db.write("UPDATE messages SET meta = ? WHERE id = ?", (raw, message_id))
        return self.get_message(message_id)

    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> Conversation:
        keys = set(row.keys())
        return Conversation(
            id=row["id"],
            title=row["title"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"] if "message_count" in keys else 0,
            preview=row["preview"] if "preview" in keys else None,
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Message:
        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            seq=row["seq"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            meta=json.loads(row["meta"]) if row["meta"] else None,
        )

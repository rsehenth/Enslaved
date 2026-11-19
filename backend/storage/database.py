from contextlib import contextmanager
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'Nova conversa',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    meta TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages (conversation_id, seq);

CREATE TABLE IF NOT EXISTS conversation_exports (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    markdown_path TEXT NOT NULL,
    last_exported_seq INTEGER NOT NULL DEFAULT 0,
    last_exported_at TEXT,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('ok', 'pending', 'error')),
    last_error TEXT
);

CREATE TABLE IF NOT EXISTS repacks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    markdown_path TEXT NOT NULL,
    covered_from_seq INTEGER NOT NULL,
    covered_to_seq INTEGER NOT NULL,
    source_message_ids TEXT NOT NULL,
    summary_text TEXT NOT NULL,
    confirmed_facts TEXT NOT NULL DEFAULT '[]',
    hypotheses TEXT NOT NULL DEFAULT '[]',
    decisions TEXT NOT NULL DEFAULT '[]',
    pending_items TEXT NOT NULL DEFAULT '[]',
    token_estimate_before INTEGER NOT NULL DEFAULT 0,
    token_estimate_after INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE(conversation_id, version)
);

CREATE INDEX IF NOT EXISTS idx_repacks_conversation ON repacks (conversation_id, version DESC);

-- Anexos de imagem do chat. conversation_id NULL = enviado e ainda não usado em uma mensagem (pendente).
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id) ON DELETE CASCADE,
    message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
    kind TEXT NOT NULL DEFAULT 'image',
    mime TEXT NOT NULL,
    ext TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    rel_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attachments_conversation ON attachments (conversation_id);
CREATE INDEX IF NOT EXISTS idx_attachments_message ON attachments (message_id);

CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('user_info', 'project', 'decision', 'preference', 'pending', 'summary', 'fact')),
    certainty TEXT NOT NULL DEFAULT 'confirmed' CHECK (certainty IN ('confirmed', 'hypothesis')),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    source_conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    source_message_ids TEXT NOT NULL DEFAULT '[]',
    source_markdown_path TEXT,
    source_repack_id INTEGER REFERENCES repacks(id) ON DELETE SET NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_active ON memories (active, pinned DESC, updated_at DESC);

CREATE TABLE IF NOT EXISTS conversation_context_state (
    conversation_id TEXT PRIMARY KEY REFERENCES conversations(id) ON DELETE CASCADE,
    estimated_tokens INTEGER NOT NULL DEFAULT 0,
    budget_tokens INTEGER NOT NULL DEFAULT 0,
    threshold_tokens INTEGER NOT NULL DEFAULT 0,
    usage_ratio REAL NOT NULL DEFAULT 0.0,
    prompt_eval_count INTEGER,
    calibration_factor REAL NOT NULL DEFAULT 1.0,
    repack_recommended INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

-- Learning Center: documentos recebidos -> candidatos -> conhecimento aprovado (separado de `memories`)
CREATE TABLE IF NOT EXISTS learning_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('file', 'url', 'text')),
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    format TEXT,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    original_path TEXT,
    markdown_path TEXT,
    status TEXT NOT NULL DEFAULT 'received'
        CHECK (status IN ('received', 'processing', 'processed', 'failed', 'needs_ocr', 'needs_conversion')),
    stage TEXT NOT NULL DEFAULT '',
    progress INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    warnings TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    analysis_mode TEXT,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    previous_document_id INTEGER,
    change_summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_learning_documents_sha ON learning_documents (sha256);

CREATE TABLE IF NOT EXISTS learning_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES learning_documents(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    ref TEXT NOT NULL DEFAULT '',
    section TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_learning_chunks_doc ON learning_chunks (document_id, seq);

CREATE TABLE IF NOT EXISTS knowledge_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'disabled', 'superseded')),
    certainty TEXT NOT NULL DEFAULT 'confirmed' CHECK (certainty IN ('confirmed', 'uncertain')),
    version INTEGER NOT NULL DEFAULT 1,
    sources TEXT NOT NULL DEFAULT '[]',
    markdown_path TEXT,
    supersedes_id INTEGER,
    superseded_by_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_knowledge_status ON knowledge_items (status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_knowledge_hash ON knowledge_items (content_hash);

CREATE TABLE IF NOT EXISTS knowledge_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id INTEGER NOT NULL REFERENCES knowledge_items(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS knowledge_candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES learning_documents(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    classification TEXT NOT NULL
        CHECK (classification IN ('novo', 'complemento', 'ja_documentado', 'contradicao', 'incerto')),
    reason TEXT NOT NULL DEFAULT '',
    conflicts TEXT NOT NULL DEFAULT '[]',
    related_item_id INTEGER,
    source_ref TEXT NOT NULL DEFAULT '',
    source_section TEXT NOT NULL DEFAULT '',
    source_quote TEXT NOT NULL DEFAULT '',
    score REAL NOT NULL DEFAULT 0,
    analysis TEXT NOT NULL DEFAULT 'heuristica',
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'ignored')),
    item_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON knowledge_candidates (status, document_id);

CREATE TABLE IF NOT EXISTS learning_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- API local OpenAI-compatible (/v1): metadados das chaves. Só o hash SHA-256 do
-- segredo é persistido (o segredo completo existe apenas na resposta de criação).
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked')),
    created_at TEXT NOT NULL,
    last_used_at TEXT,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_api_keys_status ON api_keys (status);
"""

FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_fts USING fts5(
    item_id UNINDEXED,
    title,
    content,
    tags,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    kind,
    title,
    content,
    tags,
    tokenize='unicode61 remove_diacritics 2'
);
"""


class Database:
    """Acesso SQLite local (WAL) para aplicação single-user."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.fts5_available: bool = False
        self._tx_depth: int = 0

    def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            try:
                self._conn.executescript(FTS_SCHEMA)
                self.fts5_available = True
            except sqlite3.OperationalError:
                self.fts5_available = False
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Evolução aditiva do esquema em bancos já existentes (nunca remove nem reescreve dados)."""
        assert self._conn is not None
        repack_cols = {row["name"] for row in self._conn.execute("PRAGMA table_info(repacks)")}
        if "structured" not in repack_cols:
            # Repack estruturado (objetivo, restrições, tópicos, anexos...): JSON gerado sem o modelo
            self._conn.execute("ALTER TABLE repacks ADD COLUMN structured TEXT NOT NULL DEFAULT '{}'")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @contextmanager
    def transaction(self):
        with self._lock:
            assert self._conn is not None, "banco não conectado"
            outer = self._tx_depth == 0
            self._tx_depth += 1
            try:
                yield self._conn
                if outer:
                    self._conn.commit()
            except Exception:
                if outer:
                    self._conn.rollback()
                raise
            finally:
                self._tx_depth -= 1

    def write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            assert self._conn is not None, "banco não conectado"
            cursor = self._conn.execute(sql, params)
            if self._tx_depth == 0:
                self._conn.commit()
            return cursor

    def read_all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            assert self._conn is not None, "banco não conectado"
            return self._conn.execute(sql, params).fetchall()

    def read_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        with self._lock:
            assert self._conn is not None, "banco não conectado"
            return self._conn.execute(sql, params).fetchone()

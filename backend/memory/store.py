from __future__ import annotations

import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from typing import Callable

from pydantic import BaseModel

from backend.storage.database import Database

ALLOWED_KINDS = {"user_info", "project", "decision", "preference", "pending", "summary", "fact"}
ALLOWED_CERTAINTIES = {"confirmed", "hypothesis"}

_SUPERSEDE_RE = re.compile(
    r"\b(em\s+vez\s+de|ao\s+inv[eé]s\s+de|mudamos|trocamos|substitu[ií]mos|"
    r"n[aã]o\s+usamos\s+mais|n[aã]o\s+vamos\s+mais|passou\s+a\s+ser|agora\s+(?:vamos|usamos|[eé]))\b",
    re.IGNORECASE,
)


def _normalize_for_dedup(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(".,;:!?")

_STOPWORDS_PT_EN = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "sob", "sobre", "e",
    "ou", "mas", "que", "qual", "quais", "como", "onde", "quando", "porque", "se",
    "eu", "tu", "ele", "ela", "nós", "você", "vocês", "eles", "elas", "meu", "minha",
    "seu", "sua", "isso", "isto", "aquilo", "aqui", "ali", "já", "ainda", "muito",
    "mais", "menos", "ser", "estar", "ter", "haver", "fazer", "foi", "é", "são", "era",
    "the", "and", "or", "for", "with", "from", "that", "this", "what", "when", "where",
    "how", "are", "was", "were", "have", "has", "had", "not", "can", "will",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tokenize_terms(text: str) -> list[str]:
    tokens = re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ_]{2,}", text.lower())
    seen: set[str] = set()
    ordered: list[str] = []
    for tok in tokens:
        if tok in _STOPWORDS_PT_EN or tok in seen:
            continue
        seen.add(tok)
        ordered.append(tok)
    return ordered[:12]


class MemoryItem(BaseModel):
    id: int
    kind: str
    certainty: str
    title: str
    content: str
    pinned: bool = False
    active: bool = True
    source_conversation_id: str | None = None
    source_message_ids: list[int] = []
    source_markdown_path: str | None = None
    source_repack_id: int | None = None
    tags: list[str] = []
    created_at: str
    updated_at: str
    score: float | None = None


class RepackRecord(BaseModel):
    id: int
    conversation_id: str
    version: int
    markdown_path: str
    covered_from_seq: int
    covered_to_seq: int
    source_message_ids: list[int]
    summary_text: str
    confirmed_facts: list[str] = []
    hypotheses: list[str] = []
    decisions: list[str] = []
    pending_items: list[str] = []
    token_estimate_before: int = 0
    token_estimate_after: int = 0
    created_at: str
    structured: str = "{}"  # resumo estruturado v2 (JSON; ver backend/memory/summary.py)


class ContextState(BaseModel):
    conversation_id: str
    estimated_tokens: int = 0
    budget_tokens: int = 0
    threshold_tokens: int = 0
    usage_ratio: float = 0.0
    prompt_eval_count: int | None = None
    calibration_factor: float = 1.0
    repack_recommended: bool = False
    updated_at: str


class MemoryRepository:
    """Persistência transacional e recuperação indexada (FTS5 + fallback local) de memórias e repacks."""

    def __init__(self, db: Database):
        self.db = db

    # ------------------------------------------------------------------ memories CRUD

    def create_memory(
        self,
        *,
        kind: str,
        title: str,
        content: str,
        certainty: str = "confirmed",
        pinned: bool = False,
        source_conversation_id: str | None = None,
        source_message_ids: list[int] | None = None,
        source_markdown_path: str | None = None,
        source_repack_id: int | None = None,
        tags: list[str] | None = None,
    ) -> MemoryItem:
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"Tipo de memória inválido: {kind!r}")
        if certainty not in ALLOWED_CERTAINTIES:
            raise ValueError(f"Nível de certeza inválido: {certainty!r}")
        clean_title = " ".join(title.split()).strip()[:160]
        clean_content = content.strip()
        if not clean_title or not clean_content:
            raise ValueError("Título e conteúdo da memória são obrigatórios")

        # Evita duplicação exata (ignorando pontuação final) e atualiza certeza se confirmada depois
        normalized = _normalize_for_dedup(clean_content)
        ts = _now()
        msg_ids = list(dict.fromkeys(source_message_ids or []))
        tag_list = list(dict.fromkeys(t.strip() for t in (tags or []) if t and t.strip()))

        with self.db.transaction():
            existing_rows = self.db.read_all(
                "SELECT * FROM memories WHERE active = 1 AND kind = ? ORDER BY updated_at DESC",
                (kind,),
            )
            for row in existing_rows:
                if _normalize_for_dedup(row["content"]) == normalized:
                    if (row["certainty"] == "hypothesis" and certainty == "confirmed") or (
                        pinned and not row["pinned"]
                    ):
                        new_cert = "confirmed" if certainty == "confirmed" else row["certainty"]
                        new_pin = 1 if (pinned or row["pinned"]) else 0
                        self.db.write(
                            "UPDATE memories SET certainty = ?, pinned = ?, updated_at = ? WHERE id = ?",
                            (new_cert, new_pin, ts, row["id"]),
                        )
                        updated = self.get_memory(int(row["id"]))
                        assert updated is not None
                        self._sync_fts(updated)
                        return updated
                    return self._memory_from_row(row)

            # Desativa memória anterior não fixada quando há mudança explícita no mesmo assunto
            if certainty == "confirmed" and _SUPERSEDE_RE.search(clean_content):
                new_terms = set(_tokenize_terms(clean_content))
                if len(new_terms) >= 2:
                    for row in existing_rows[:100]:
                        if row["pinned"]:
                            continue
                        old_terms = set(_tokenize_terms(row["content"]))
                        if len(new_terms & old_terms) >= 2:
                            old_id = int(row["id"])
                            self.db.write("UPDATE memories SET active = 0, updated_at = ? WHERE id = ?", (ts, old_id))
                            self._delete_fts(old_id)

            cursor = self.db.write(
                """
                INSERT INTO memories (
                    kind, certainty, title, content, pinned, active,
                    source_conversation_id, source_message_ids, source_markdown_path,
                    source_repack_id, tags, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    certainty,
                    clean_title,
                    clean_content,
                    1 if pinned else 0,
                    source_conversation_id,
                    json.dumps(msg_ids),
                    source_markdown_path,
                    source_repack_id,
                    json.dumps(tag_list, ensure_ascii=False),
                    ts,
                    ts,
                ),
            )
            memory_id = int(cursor.lastrowid)
            item = self.get_memory(memory_id)
            assert item is not None
            self._sync_fts(item)
            return item

    def get_memory(self, memory_id: int) -> MemoryItem | None:
        row = self.db.read_one("SELECT * FROM memories WHERE id = ?", (memory_id,))
        return self._memory_from_row(row) if row else None

    def list_memories(
        self,
        *,
        kind: str | None = None,
        query: str | None = None,
        include_inactive: bool = False,
        limit: int = 100,
    ) -> list[MemoryItem]:
        clauses: list[str] = []
        params: list = []
        if not include_inactive:
            clauses.append("active = 1")
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if query and query.strip():
            like = f"%{query.strip()}%"
            clauses.append("(title LIKE ? OR content LIKE ? OR tags LIKE ?)")
            params.extend([like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.db.read_all(
            f"SELECT * FROM memories {where} ORDER BY pinned DESC, updated_at DESC LIMIT ?",
            (*params, limit),
        )
        return [self._memory_from_row(row) for row in rows]

    def update_memory(
        self,
        memory_id: int,
        *,
        title: str | None = None,
        content: str | None = None,
        kind: str | None = None,
        certainty: str | None = None,
        pinned: bool | None = None,
        active: bool | None = None,
        tags: list[str] | None = None,
    ) -> MemoryItem | None:
        current = self.get_memory(memory_id)
        if current is None:
            return None
        new_kind = kind if kind is not None else current.kind
        new_certainty = certainty if certainty is not None else current.certainty
        if new_kind not in ALLOWED_KINDS:
            raise ValueError(f"Tipo de memória inválido: {new_kind!r}")
        if new_certainty not in ALLOWED_CERTAINTIES:
            raise ValueError(f"Nível de certeza inválido: {new_certainty!r}")
        new_title = " ".join(title.split()).strip()[:160] if title is not None else current.title
        new_content = content.strip() if content is not None else current.content
        if not new_title or not new_content:
            raise ValueError("Título e conteúdo da memória não podem ficar vazios")
        new_pinned = current.pinned if pinned is None else bool(pinned)
        new_active = current.active if active is None else bool(active)
        new_tags = current.tags if tags is None else list(dict.fromkeys(t.strip() for t in tags if t and t.strip()))
        ts = _now()

        with self.db.transaction():
            self.db.write(
                """
                UPDATE memories
                SET kind = ?, certainty = ?, title = ?, content = ?, pinned = ?, active = ?, tags = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    new_kind,
                    new_certainty,
                    new_title,
                    new_content,
                    1 if new_pinned else 0,
                    1 if new_active else 0,
                    json.dumps(new_tags, ensure_ascii=False),
                    ts,
                    memory_id,
                ),
            )
            updated = self.get_memory(memory_id)
            assert updated is not None
            if updated.active:
                self._sync_fts(updated)
            else:
                self._delete_fts(memory_id)
            return updated

    def delete_memory(self, memory_id: int) -> bool:
        with self.db.transaction():
            self._delete_fts(memory_id)
            cursor = self.db.write("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    def delete_memories_for_repack(self, repack_id: int) -> int:
        with self.db.transaction():
            rows = self.db.read_all("SELECT id FROM memories WHERE source_repack_id = ?", (repack_id,))
            for r in rows:
                self._delete_fts(int(r["id"]))
            cursor = self.db.write("DELETE FROM memories WHERE source_repack_id = ?", (repack_id,))
            return cursor.rowcount

    def count_active_memories(self) -> int:
        row = self.db.read_one("SELECT COUNT(*) AS cnt FROM memories WHERE active = 1")
        return int(row["cnt"]) if row else 0

    def reconcile_fts(self) -> int:
        """Garante consistência entre `memories` (active=1) e o índice `memories_fts`."""
        if not self.db.fts5_available:
            return 0
        active_rows = self.db.read_all("SELECT * FROM memories WHERE active = 1")
        active_ids = {int(r["id"]) for r in active_rows}
        fts_ids = {int(r["memory_id"]) for r in self.db.read_all("SELECT memory_id FROM memories_fts")}
        if active_ids == fts_ids:
            return 0
        with self.db.transaction():
            self.db.write("DELETE FROM memories_fts")
            for row in active_rows:
                item = self._memory_from_row(row)
                self.db.write(
                    "INSERT INTO memories_fts (memory_id, kind, title, content, tags) VALUES (?, ?, ?, ?, ?)",
                    (item.id, item.kind, item.title, item.content, " ".join(item.tags)),
                )
        return len(active_ids ^ fts_ids)

    # ------------------------------------------------------------------ FTS5 sync & retrieval

    def _sync_fts(self, item: MemoryItem) -> None:
        if not self.db.fts5_available:
            return
        self.db.write("DELETE FROM memories_fts WHERE memory_id = ?", (item.id,))
        if item.active:
            self.db.write(
                "INSERT INTO memories_fts (memory_id, kind, title, content, tags) VALUES (?, ?, ?, ?, ?)",
                (item.id, item.kind, item.title, item.content, " ".join(item.tags)),
            )

    def _delete_fts(self, memory_id: int) -> None:
        if not self.db.fts5_available:
            return
        self.db.write("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))

    def retrieve_relevant(
        self,
        query: str,
        *,
        max_tokens: int = 450,
        max_items: int = 8,
        estimate_tokens_fn: Callable[[str], int] | None = None,
        semantic_scorer: Callable[[str, MemoryItem], float] | None = None,
    ) -> list[MemoryItem]:
        """Recupera memórias relevantes via FTS5 + termos + prioridade (fixadas/confirmadas/recentes)."""
        terms = _tokenize_terms(query or "")
        candidates: dict[int, tuple[MemoryItem, float]] = {}

        # 1. Memórias fixadas entram como candidatas prioritárias
        for row in self.db.read_all("SELECT * FROM memories WHERE active = 1 AND pinned = 1 ORDER BY updated_at DESC"):
            item = self._memory_from_row(row)
            candidates[item.id] = (item, 3.0)

        # 2. Busca FTS5 (BM25) quando houver termos úteis
        if terms and self.db.fts5_available:
            fts_query = " OR ".join(f'"{t}"' for t in terms)
            try:
                fts_rows = self.db.read_all(
                    """
                    SELECT m.*, bm25(memories_fts) AS rank
                    FROM memories_fts f
                    JOIN memories m ON m.id = f.memory_id
                    WHERE memories_fts MATCH ? AND m.active = 1
                    ORDER BY rank ASC
                    LIMIT ?
                    """,
                    (fts_query, max_items * 3),
                )
                for row in fts_rows:
                    item = self._memory_from_row(row)
                    # bm25 retorna valores negativos melhores quanto menor
                    bm25_score = max(0.2, min(5.0, -float(row["rank"] or 0.0) + 1.0))
                    prev = candidates.get(item.id, (item, 0.0))[1]
                    candidates[item.id] = (item, max(prev, bm25_score))
            except sqlite3.OperationalError:
                pass

        # 3. Fallback / complemento por tokens exatos (evita falso positivo de substring curta, ex.: 'tem' em 'temporários')
        if terms:
            for row in self.db.read_all("SELECT * FROM memories WHERE active = 1 ORDER BY updated_at DESC LIMIT 200"):
                item = self._memory_from_row(row)
                haystack = f"{item.title} {item.content} {' '.join(item.tags)}".lower()
                hay_tokens = set(re.findall(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ_]{2,}", haystack))
                hits = sum(1 for t in terms if t in hay_tokens or (len(t) >= 5 and any(ht.startswith(t) for ht in hay_tokens)))
                if hits > 0:
                    term_score = 1.0 + (hits / max(1, len(terms))) * 2.5
                    prev = candidates.get(item.id, (item, 0.0))[1]
                    candidates[item.id] = (item, max(prev, term_score))

        # 4. Ranking final combinando relevância, certeza, atualidade e opcionalmente scorer semântico
        now_ts = datetime.now(timezone.utc).timestamp()
        scored: list[MemoryItem] = []
        for item, base_score in candidates.values():
            certainty_bonus = 0.6 if item.certainty == "confirmed" else 0.1
            pinned_bonus = 2.0 if item.pinned else 0.0
            recency_bonus = self._recency_bonus(item.updated_at, now_ts)
            sem_bonus = semantic_scorer(query, item) if semantic_scorer else 0.0
            total_score = round(base_score + certainty_bonus + pinned_bonus + recency_bonus + sem_bonus, 4)
            scored.append(item.model_copy(update={"score": total_score}))

        scored.sort(key=lambda m: (m.score or 0.0, m.updated_at), reverse=True)

        # 5. Seleção respeitando teto de tokens e quantidade
        est_fn = estimate_tokens_fn or (lambda txt: max(1, math.ceil(len(txt) / 3.0)))
        selected: list[MemoryItem] = []
        used_tokens = 0
        for item in scored:
            if len(selected) >= max_items:
                break
            line_tokens = est_fn(f"[{item.kind}|{item.certainty}] {item.title}: {item.content}") + 14
            if selected and (used_tokens + line_tokens) > max_tokens:
                continue
            if not selected and line_tokens > max_tokens:
                # Mesmo uma única memória grande precisa respeitar o orçamento máximo
                continue
            selected.append(item)
            used_tokens += line_tokens
        return selected

    @staticmethod
    def _recency_bonus(iso_ts: str, now_ts: float) -> float:
        try:
            dt = datetime.fromisoformat(iso_ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_hours = max(0.0, (now_ts - dt.timestamp()) / 3600.0)
            return round(0.8 / (1.0 + age_hours / 72.0), 4)
        except ValueError:
            return 0.0

    # ------------------------------------------------------------------ repacks

    def next_repack_version(self, conversation_id: str) -> int:
        row = self.db.read_one(
            "SELECT COALESCE(MAX(version), 0) AS max_v FROM repacks WHERE conversation_id = ?",
            (conversation_id,),
        )
        return int(row["max_v"]) + 1 if row else 1

    def create_repack(
        self,
        *,
        conversation_id: str,
        version: int,
        markdown_path: str,
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
        created_at: str | None = None,
        structured: str = "{}",
    ) -> RepackRecord:
        ts = created_at or _now()
        cursor = self.db.write(
            """
            INSERT INTO repacks (
                conversation_id, version, markdown_path, covered_from_seq, covered_to_seq,
                source_message_ids, summary_text, confirmed_facts, hypotheses,
                decisions, pending_items, token_estimate_before, token_estimate_after, created_at, structured
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                version,
                markdown_path,
                covered_from_seq,
                covered_to_seq,
                json.dumps(source_message_ids),
                summary_text,
                json.dumps(confirmed_facts, ensure_ascii=False),
                json.dumps(hypotheses, ensure_ascii=False),
                json.dumps(decisions, ensure_ascii=False),
                json.dumps(pending_items, ensure_ascii=False),
                token_estimate_before,
                token_estimate_after,
                ts,
                structured,
            ),
        )
        row = self.db.read_one("SELECT * FROM repacks WHERE id = ?", (cursor.lastrowid,))
        assert row is not None
        return self._repack_from_row(row)

    def get_latest_repack(self, conversation_id: str) -> RepackRecord | None:
        row = self.db.read_one(
            "SELECT * FROM repacks WHERE conversation_id = ? ORDER BY version DESC LIMIT 1",
            (conversation_id,),
        )
        return self._repack_from_row(row) if row else None

    def list_repacks(self, conversation_id: str | None = None, limit: int = 50) -> list[RepackRecord]:
        if conversation_id:
            rows = self.db.read_all(
                "SELECT * FROM repacks WHERE conversation_id = ? ORDER BY version DESC LIMIT ?",
                (conversation_id, limit),
            )
        else:
            rows = self.db.read_all(
                "SELECT * FROM repacks ORDER BY created_at DESC, version DESC LIMIT ?",
                (limit,),
            )
        return [self._repack_from_row(row) for row in rows]

    # ------------------------------------------------------------------ context state

    def save_context_state(
        self,
        *,
        conversation_id: str,
        estimated_tokens: int,
        budget_tokens: int,
        threshold_tokens: int,
        usage_ratio: float,
        prompt_eval_count: int | None,
        calibration_factor: float,
        repack_recommended: bool,
    ) -> ContextState:
        ts = _now()
        self.db.write(
            """
            INSERT INTO conversation_context_state (
                conversation_id, estimated_tokens, budget_tokens, threshold_tokens,
                usage_ratio, prompt_eval_count, calibration_factor, repack_recommended, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                estimated_tokens = excluded.estimated_tokens,
                budget_tokens = excluded.budget_tokens,
                threshold_tokens = excluded.threshold_tokens,
                usage_ratio = excluded.usage_ratio,
                prompt_eval_count = COALESCE(excluded.prompt_eval_count, conversation_context_state.prompt_eval_count),
                calibration_factor = excluded.calibration_factor,
                repack_recommended = excluded.repack_recommended,
                updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                estimated_tokens,
                budget_tokens,
                threshold_tokens,
                usage_ratio,
                prompt_eval_count,
                calibration_factor,
                1 if repack_recommended else 0,
                ts,
            ),
        )
        state = self.get_context_state(conversation_id)
        assert state is not None
        return state

    def get_context_state(self, conversation_id: str) -> ContextState | None:
        row = self.db.read_one(
            "SELECT * FROM conversation_context_state WHERE conversation_id = ?",
            (conversation_id,),
        )
        if not row:
            return None
        return ContextState(
            conversation_id=row["conversation_id"],
            estimated_tokens=row["estimated_tokens"],
            budget_tokens=row["budget_tokens"],
            threshold_tokens=row["threshold_tokens"],
            usage_ratio=row["usage_ratio"],
            prompt_eval_count=row["prompt_eval_count"],
            calibration_factor=row["calibration_factor"],
            repack_recommended=bool(row["repack_recommended"]),
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------ row mappers

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=row["id"],
            kind=row["kind"],
            certainty=row["certainty"],
            title=row["title"],
            content=row["content"],
            pinned=bool(row["pinned"]),
            active=bool(row["active"]),
            source_conversation_id=row["source_conversation_id"],
            source_message_ids=json.loads(row["source_message_ids"] or "[]"),
            source_markdown_path=row["source_markdown_path"],
            source_repack_id=row["source_repack_id"],
            tags=json.loads(row["tags"] or "[]"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _repack_from_row(row: sqlite3.Row) -> RepackRecord:
        return RepackRecord(
            id=row["id"],
            conversation_id=row["conversation_id"],
            version=row["version"],
            markdown_path=row["markdown_path"],
            covered_from_seq=row["covered_from_seq"],
            covered_to_seq=row["covered_to_seq"],
            source_message_ids=json.loads(row["source_message_ids"] or "[]"),
            summary_text=row["summary_text"],
            confirmed_facts=json.loads(row["confirmed_facts"] or "[]"),
            hypotheses=json.loads(row["hypotheses"] or "[]"),
            decisions=json.loads(row["decisions"] or "[]"),
            pending_items=json.loads(row["pending_items"] or "[]"),
            token_estimate_before=row["token_estimate_before"],
            token_estimate_after=row["token_estimate_after"],
            created_at=row["created_at"],
            structured=row["structured"] if "structured" in row.keys() else "{}",
        )

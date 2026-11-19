"""Estado TEMPORÁRIO por tarefa de investigação. Nada daqui é preferência permanente nem é gravado no banco.

`research_mode` vale só enquanto a tarefa roda; ao terminar (concluída, cancelada, erro, abandonada ou nova tarefa
independente) volta a "quick". As evidências ficam na tarefa (e na mensagem final), não são apagadas pela restauração.
"""

import threading
import time
import uuid
from dataclasses import dataclass, field

from backend.research.evidence import Ledger

DEFAULT_MODE = "quick"
ACTIVE = {"awaiting_depth", "awaiting_permission", "planning", "collecting", "analyzing", "verifying", "reporting"}


@dataclass
class ResearchTask:
    conversation_id: str
    research_scope: dict
    research_mode: str
    research_task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    research_status: str = "planning"
    research_progress: dict = field(default_factory=lambda: {"phase": "planning", "pages": 0, "requests": 0, "evidence": 0})
    ledger: Ledger = field(default_factory=Ledger)
    mode_used: str = DEFAULT_MODE  # histórico: nível com que a tarefa rodou (research_mode volta a "quick" ao terminar)
    authorized_hosts: set[str] = field(default_factory=set)  # hosts aprovados pelo proprietário PARA esta tarefa
    cancel: threading.Event = field(default_factory=threading.Event)
    created: float = field(default_factory=time.monotonic)

    @property
    def active(self) -> bool:
        return self.research_status in ACTIVE


class ResearchRegistry:
    """Tarefas por conversa. Só a tarefa ATUAL da conversa pode restaurar o modo (concorrência segura)."""

    def __init__(self, keep_finished: int = 20):
        self._lock = threading.Lock()
        self._tasks: dict[str, ResearchTask] = {}
        self._current: dict[str, str] = {}  # conversa -> id da tarefa corrente
        self._keep = keep_finished

    def mode(self, conversation_id: str | None) -> str:
        """Modo efetivo da conversa agora: o da tarefa ativa; senão rápido."""
        with self._lock:
            task = self._tasks.get(self._current.get(conversation_id or "", ""))
            return task.research_mode if task and task.active else DEFAULT_MODE

    def begin(self, conversation_id: str, mode: str, scope: dict, status: str = "planning") -> ResearchTask:
        """Nova tarefa independente: encerra (abandona) a anterior ainda ativa da mesma conversa, sem herdar o nível."""
        with self._lock:
            prev = self._tasks.get(self._current.get(conversation_id, ""))
            if prev and prev.active:
                self._close(prev, "abandoned")
            task = ResearchTask(conversation_id, dict(scope), mode, research_status=status, mode_used=mode)
            self._tasks[task.research_task_id] = task
            self._current[conversation_id] = task.research_task_id
            self._prune()
            return task

    def get(self, task_id: str) -> ResearchTask | None:
        return self._tasks.get(task_id)

    def current(self, conversation_id: str | None) -> ResearchTask | None:
        with self._lock:
            return self._tasks.get(self._current.get(conversation_id or "", ""))

    def update(self, task: ResearchTask, status: str | None = None, **progress) -> None:
        with self._lock:
            if not task.active:
                return  # tarefa já encerrada não é "ressuscitada" por um processamento atrasado
            if status:
                task.research_status = status
                progress.setdefault("phase", status)
            task.research_progress.update(progress)

    def finish(self, task: ResearchTask, status: str = "done") -> None:
        with self._lock:
            self._close(task, status)

    def abandon_conversation(self, conversation_id: str) -> None:
        """O usuário seguiu com outra instrução: a tarefa pendente (ex.: pergunta de profundidade) é abandonada."""
        with self._lock:
            task = self._tasks.get(self._current.get(conversation_id, ""))
            if task and task.active:
                self._close(task, "abandoned")

    def cancel(self, conversation_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(self._current.get(conversation_id, ""))
            if task and task.active:
                task.cancel.set()
                return True
            return False

    def _close(self, task: ResearchTask, status: str) -> None:
        if task.active:
            task.research_status = status
            task.research_progress["phase"] = status
        task.research_mode = DEFAULT_MODE  # restauração automática; evidências e mode_used permanecem

    def _prune(self) -> None:
        done = [t for t in self._tasks.values() if not t.active]
        for task in sorted(done, key=lambda t: t.created)[: max(0, len(done) - self._keep)]:
            self._tasks.pop(task.research_task_id, None)

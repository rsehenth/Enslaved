"""Liberação progressiva (e segura) do relatório de pesquisa.

O texto do modelo continua sendo revisado contra o dossiê (`review_report`) ANTES de chegar ao chat: só prefixos que
terminam num limite de bloco (linha em branco fora de código, sem título órfão) são revisados e liberados. A revisão é
linha a linha e depende apenas do que veio antes, então o prefixo revisado costuma ser prefixo do texto final; quando não
for (ex.: truncamento cortou o fim), o fluxo emite um evento explícito `replace` — o texto exibido nunca diverge em silêncio
do que é salvo."""

import re
from typing import Any

from backend.research.report import clean_markdown, drop_own_sections, review_report
from backend.research.state import ResearchTask

_FENCE = re.compile(r"\s*(?:```|~~~)")
_HEADING = re.compile(r"\s*#{1,6}\s")
_CHUNK = 64


def safe_cut(draft: str) -> int:
    """Fim do último bloco completo do rascunho: linha em branco fora de código; um título sozinho espera o seu parágrafo."""
    pos = cut = block_start = 0
    in_code = False
    for line in draft.split("\n")[:-1]:  # só linhas já terminadas por \n
        end = pos + len(line) + 1
        if _FENCE.match(line):
            in_code = not in_code
        elif not in_code and not line.strip():
            block = draft[block_start:pos]
            if not block.strip():
                block_start = end
            elif not all(_HEADING.match(ln) for ln in block.split("\n") if ln.strip()):  # título sozinho: segue acumulando
                cut, block_start = end, end
        pos = end
    return cut


def review_prefix(task: ResearchTask, draft: str) -> str:
    """O mesmo tratamento do relatório final (limpeza + revisão contra o dossiê), aplicado a um prefixo."""
    return review_report(task, clean_markdown(drop_own_sections(draft))).text


class ProgressiveReport:
    """Mantém o que já foi exibido e traduz cada novo rascunho em eventos SSE (`token` incremental ou `replace`)."""

    def __init__(self, task: ResearchTask, shown: str = ""):
        self.task, self.shown, self._cut = task, shown, 0
        self._base = shown

    def feed(self, draft: str) -> list[dict[str, Any]]:
        """Rascunho bruto do modelo até agora → eventos para o trecho já revisado e completo."""
        cut = safe_cut(draft)
        if cut <= self._cut:
            return []
        self._cut = cut
        return self.show(self._base + review_prefix(self.task, draft[:cut]))

    def show(self, text: str) -> list[dict[str, Any]]:
        """Leva o texto exibido até `text`: acrescenta se ele estende o exibido; senão substitui explicitamente."""
        if text == self.shown:
            return []
        if text.startswith(self.shown):
            delta, events = text[len(self.shown):], []
            for start in range(0, len(delta), _CHUNK):
                events.append({"type": "token", "content": delta[start : start + _CHUNK]})
        else:
            events = [{"type": "replace", "content": text}]
        self.shown = text
        return events

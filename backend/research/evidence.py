"""Registro rastreável de evidências e descobertas de uma investigação (deduplicado)."""

import hashlib
import re
from dataclasses import asdict, dataclass, field

from backend.core.sanitize import looks_like_prompt_injection, sanitize_knowledge_text
from backend.research.intent import url_dedup_key

# A vulnerabilidade confirmada (prova direta) · H configuração de segurança ausente/fraca (observada; NÃO é vulnerabilidade explorável por si só)
# B indício a validar · C risco teórico · D não verificado. Observações técnicas confirmadas ficam em `Ledger.observations`.
CATEGORIES = {
    "A": "Vulnerabilidade confirmada (prova direta)",
    "H": "Configuração de segurança ausente ou fraca (observada; não é vulnerabilidade explorável por si só)",
    "B": "Indício ainda não comprovado",
    "C": "Risco teórico",
    "D": "Item não verificado",
}
ORDER = "AHBCD"


@dataclass
class Evidence:
    id: str
    kind: str  # http | tls | page | search | file
    url: str
    observation: str
    ok: bool = True
    sha256: str | None = None
    tool: str = "net.inspect"


@dataclass
class Finding:
    id: str
    category: str  # A | H | B | C | D
    title: str
    detail: str
    evidence: list[str] = field(default_factory=list)  # ids de Evidence que SUSTENTAM a conclusão
    severity: str | None = None  # A: severidade; H: prioridade de correção
    proof: str | None = None  # só A: a observação direta que comprova a vulnerabilidade
    recommendation: str | None = None


@dataclass
class Ledger:
    evidence: list[Evidence] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)  # fatos neutros (ex.: "há página de login"), nunca vulnerabilidades
    limitations: list[str] = field(default_factory=list)
    pages: set[str] = field(default_factory=set)
    requests: int = 0
    bytes_total: int = 0
    # cobertura REAL da coleta: limites aplicados e o motivo verdadeiro de encerramento por alvo (preenchido pelo runner)
    coverage: dict = field(default_factory=dict)
    _seen: dict[tuple, str] = field(default_factory=dict, repr=False)

    def add_evidence(self, kind: str, url: str, observation: str, *, ok: bool = True, body: bytes | str | None = None, sha256: str | None = None, tool: str = "net.inspect") -> Evidence:
        """Deduplica por (tipo, URL, observação, hash do conteúdo): a mesma evidência não é registrada duas vezes."""
        raw = body.encode() if isinstance(body, str) else body
        if sha256 is not None and not re.fullmatch(r"[a-fA-F0-9]{64}", sha256):
            raise ValueError("SHA-256 inválido")
        digest = sha256.lower() if sha256 is not None else hashlib.sha256(raw).hexdigest() if raw is not None else None
        observation = self._external(observation, url)
        key = (kind, url, observation, digest)
        if key in self._seen:
            return next(e for e in self.evidence if e.id == self._seen[key])
        item = Evidence(f"E{len(self.evidence) + 1}", kind, url, observation[:400], ok, digest, tool)
        self.evidence.append(item)
        self._seen[key] = item.id
        return item

    def add_finding(self, category: str, title: str, detail: str, evidence: list[str] | None = None, *,
                    severity: str | None = None, recommendation: str | None = None, proof: str | None = None) -> Finding | None:
        """Nunca infla: A exige prova direta + evidência observada (http/tls/page); sem isso vira B. A/H/B sem evidência viram D."""
        detail = self._external(detail, title)
        ev = [e for e in dict.fromkeys(evidence or []) if any(x.id == e for x in self.evidence)]
        observed = any(x.id in ev and x.ok and x.kind in ("http", "tls", "page") for x in self.evidence)
        if category == "A" and not (observed and proof):
            category = "B" if ev else "D"
        if category in ("H", "B") and not ev:
            category = "D"
        existing = next((f for f in self.findings if f.title == title and f.category == category), None)
        if existing is not None:
            for e in ev:
                if e not in existing.evidence:
                    existing.evidence.append(e)
            return existing
        item = Finding(f"F{len(self.findings) + 1}", category, title, detail, ev, severity=severity if category in ("A", "H") else None,
                       recommendation=recommendation, proof=self._external(proof, title) if category == "A" and proof else None)
        self.findings.append(item)
        return item

    def add_page(self, url: str) -> None:
        """Adiciona página inspecionada deduplicando URLs equivalentes."""
        key = url_dedup_key(url) or url
        if not any((url_dedup_key(p) or p) == key for p in self.pages):
            self.pages.add(url)

    def renumber_findings(self) -> None:
        for idx, f in enumerate(self.findings, 1):
            f.id = f"F{idx}"

    def _external(self, text: str, origin: str) -> str:
        """Texto vindo de páginas/buscas é DADO: neutraliza marcadores de papel e registra a tentativa de injeção."""
        if looks_like_prompt_injection(text):
            self.limit(f"Texto externo com padrão de instrução ao modelo foi neutralizado e tratado só como dado ({origin[:80]}).")
            return "[instrução externa neutralizada] " + sanitize_knowledge_text(text)[:300]
        return sanitize_knowledge_text(text)

    def limit(self, text: str) -> None:
        if text not in self.limitations:
            self.limitations.append(text)

    def observe(self, text: str) -> None:
        if text not in self.observations:
            self.observations.append(text)

    def by_category(self, category: str) -> list[Finding]:
        return [f for f in self.findings if f.category == category]

    def digest(self) -> dict:
        """Resumo serializável (vai em meta da mensagem e sobrevive ao Repack)."""
        return {
            "evidence": [asdict(e) for e in self.evidence],
            "findings": [asdict(f) for f in self.findings],
            "limitations": list(self.limitations),
            "observations": list(self.observations),
            "requests": self.requests,
            "pages": sorted(self.pages),
            "coverage": self.coverage,
        }

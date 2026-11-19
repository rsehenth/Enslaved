"""Ferramenta local de extração e análise de textos (determinística, sem rede)."""

import re
from collections import Counter
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from backend.tools.base import Tool, ToolResult, ToolSpec

_WORD = re.compile(r"[a-záàâãéêíóôõúüçñ0-9]+", re.IGNORECASE)
# terminador de frase exige espaço/fim depois do ponto (não quebra URLs como exemplo.com.br)
_SENTENCE = re.compile(r"[^.!?…]+[.!?…]+(?=\s|$)|[^.!?…]+$")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

_STOPWORDS = {
    "pt": {
        "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na",
        "nos", "nas", "por", "para", "com", "sem", "sob", "sobre", "e", "ou", "mas", "que", "se", "como",
        "quando", "onde", "porque", "é", "são", "foi", "ser", "estar", "tem", "ter", "há", "ao", "à", "às",
        "este", "esta", "isso", "ele", "ela", "eles", "elas", "não", "sim", "mais", "menos", "muito", "já",
    },
    "en": {
        "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "by", "for", "with",
        "about", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did",
        "not", "it", "this", "that", "these", "those", "he", "she", "they", "we", "you", "i",
    },
}


class TextAnalyzeArgs(BaseModel):
    text: str = Field(min_length=1, max_length=200_000, description="Texto a analisar.")
    max_keywords: int = Field(default=10, ge=1, le=30)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


def _guess_language(tokens: list[str]) -> str | None:
    scores = {}
    for language, stopwords in _STOPWORDS.items():
        scores[language] = sum(1 for token in tokens if token in stopwords)
    best = max(scores, key=lambda language: scores[language])
    return best if scores[best] > 0 else None


class TextAnalyzeTool(Tool):
    args_model = TextAnalyzeArgs

    def __init__(self):
        super().__init__()
        self.spec = ToolSpec(
            id="text.analyze",
            name="Text analysis",
            description=(
                "Locally analyzes a given text: statistics (words, sentences, paragraphs, reading time), "
                "keywords, likely language (pt/en), and URL and email extraction. Deterministic; uses neither the LLM nor the network."
            ),
            capabilities=["text", "local"],
            required_permissions=["text.analyze"],
        )

    async def run(self, args: dict, permission=None) -> ToolResult:
        if permission is None or not permission.granted:
            return ToolResult(ok=False, error="permissão ausente")
        text: str = args["text"]
        max_keywords: int = args.get("max_keywords", 10)

        tokens = [token.group(0).lower() for token in _WORD.finditer(text)]
        sentences = _split_sentences(text)
        paragraphs = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
        words = [token for token in tokens if len(token) > 1]
        stopwords = _STOPWORDS["pt"] | _STOPWORDS["en"]
        keywords = [
            word
            for word, _count in Counter(w for w in words if w not in stopwords and len(w) >= 3).most_common(max_keywords)
        ]
        summary = " ".join(sentences[:3])
        if len(summary) > 600:
            summary = summary[:599] + "…"

        return ToolResult(
            ok=True,
            output={
                "stats": {
                    "characters": len(text),
                    "words": len(words),
                    "unique_words": len(set(words)),
                    "sentences": len(sentences),
                    "paragraphs": len(paragraphs),
                    "reading_minutes": round(len(words) / 200, 2) or None,
                },
                "language_guess": _guess_language(tokens),
                "keywords": keywords,
                "urls": sorted({match.group(0) for match in _URL.finditer(text)}),
                "emails": sorted({match.group(0) for match in _EMAIL.finditer(text)}),
                "summary_leading_sentences": summary or None,
            },
            data={
                "mode": "local-deterministic",
                "net_access": False,
                "url_host_count": len({urlparse(u).netloc for u in _URL.findall(text)}),
            },
        )

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

_DENSE_SYMBOL_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Estimador por pré-tokens (modo `pretoken`, usado pela API /v1). Reproduz a divisão do
# pré-tokenizador BPE do Qwen (palavra com 1 prefixo, dígito isolado, pontuação, espaço) e
# estima os subtokens de cada pedaço. Constantes calibradas com prompt_eval_count real do
# Ollama (qwen3.5-rana 9B, payloads reais do OpenCode 2.2.5): erro de -0,1% a +9% em
# prosa/código/JSON/schemas; 0,90–1,43 nos extremos (saída de pytest, CJK).
_PRETOKEN_RE = re.compile(
    r"(?P<w>'(?i:[sdmt]|ll|ve|re)|(?:[^\r\n\w]|_)?[^\W\d_]+)"
    r"|(?P<d>\d)"
    r"|(?P<p> ?(?:[^\s\w]|_)+[\r\n]*)"
    r"|(?P<s>\s*[\r\n]+|\s+(?!\S)|\s+)"
)
_LONG_CHUNK_RE = re.compile(r"\S{32,}")
_VOWELS = frozenset("aeiouAEIOU")
_WORD_ASCII_CHARS = 8.0  # chars ASCII por token numa palavra
_PUNCT_ASCII_CHARS = 3.0  # chars ASCII por token em pontuação/JSON
_WORD_NONASCII_BYTES = 4.0  # bytes UTF-8 por token em letras não-ASCII (acentos, CJK)
_SYMBOL_NONASCII_BYTES = 1.5  # bytes UTF-8 por token em símbolos não-ASCII (emoji: ~2,6 tokens)
_RANDOM_CHARS = 1.35  # base64/chaves com caixa aleatória
_SPACES_PER_TOKEN = 16
_TOOLS_TEMPLATE_TOKENS = 120  # cabeçalho/instruções que o template do modelo adiciona com `tools`


def _is_random_chunk(chunk: str) -> bool:
    """Base64/segredos: letras com as duas caixas e quase sem vogais (identificadores têm ~35%)."""
    letters = [c for c in chunk if c.isascii() and c.isalpha()]
    if len(letters) < 16:
        return False
    vowels = sum(c in _VOWELS for c in letters)
    return vowels / len(letters) < 0.25 and any(c.isupper() for c in letters) and any(c.islower() for c in letters)


def pretoken_estimate(text: str) -> int:
    if not text:
        return 0
    total = 0.0

    def _strip_random(match: re.Match[str]) -> str:
        nonlocal total
        if _is_random_chunk(match.group()):
            total += len(match.group()) / _RANDOM_CHARS
            return ""
        return match.group()

    text = _LONG_CHUNK_RE.sub(_strip_random, text)
    for match in _PRETOKEN_RE.finditer(text):
        piece, kind = match.group(), match.lastgroup
        if kind == "d":
            total += 1  # o pré-tokenizador do Qwen isola cada dígito
        elif kind == "s":
            total += math.ceil(len(piece) / _SPACES_PER_TOKEN)
        else:
            ascii_chars = len(piece) if piece.isascii() else sum(c.isascii() for c in piece)
            nonascii_bytes = len(piece.encode("utf-8")) - ascii_chars
            if kind == "w":
                est = ascii_chars / _WORD_ASCII_CHARS + nonascii_bytes / _WORD_NONASCII_BYTES
            else:
                est = ascii_chars / _PUNCT_ASCII_CHARS + nonascii_bytes / _SYMBOL_NONASCII_BYTES
            total += max(1, math.ceil(est))
    return math.ceil(total)


@dataclass
class PayloadValidation:
    """Resultado da validação obrigatória sobre o payload final antes da chamada ao provider."""

    fits: bool
    num_ctx: int
    output_reserve: int
    input_budget: int
    messages_tokens: int
    tools_tokens: int
    safety_margin_tokens: int
    total_input_tokens: int
    total_tokens: int
    reason: str | None = None



@dataclass
class ContextBudget:
    num_ctx: int
    output_reserve: int
    input_budget: int
    threshold_ratio: float
    threshold_tokens: int
    system_tokens: int
    tools_tokens: int
    skills_tokens: int
    memory_tokens: int
    summary_tokens: int
    history_tokens: int
    total_input_tokens: int
    usage_ratio: float
    calibration_factor: float
    needs_repack: bool
    over_hard_limit: bool


class ContextManager:
    """Gerencia orçamento de tokens com estimativa conservadora e calibração via Ollama.

    Limitações documentadas da contagem de tokens:
    - O Ollama não expõe um endpoint síncrono leve de tokenização prévia sem carregar/executar
      o modelo, e o projeto não embarca tokenizadores externos pesados (transformers/tiktoken).
    - Emprega-se `chars_per_token=3.0` (conservador frente à média real ~3.6–4.0 chars/token do
      BPE Qwen em português/inglês), calibrado dinamicamente por `prompt_eval_count` real do Ollama.
    - Na validação final de payload (`validate_payload`), soma-se overhead de delimitadores por
      mensagem, ajuste para alta densidade de símbolos/URLs/JSON e margem de segurança para
      diferenças de tokenização, além da reserva de saída (`output_reserve`).
    - `pretoken=True` (API /v1) troca chars/token por `pretoken_estimate`: chars/token fixo
      superconta prosa/código/schemas em ~40% e subconta dígitos, hashes e emoji em até 2,6x.
    """

    def __init__(
        self,
        num_ctx: int = 16384,
        output_reserve: int = 2048,
        compact_threshold_ratio: float = 0.75,
        chars_per_token: float = 3.0,
        safety_margin_ratio: float = 0.03,
        pretoken: bool = False,
    ):
        # pretoken=True: estimador por pré-tokens (API /v1); False mantém chars/token do chat interno
        self.pretoken = pretoken
        self.num_ctx = max(512, int(num_ctx))
        # Garante ao menos 256 tokens para entrada mesmo se num_predict for alto
        self.output_reserve = min(max(128, int(output_reserve)), max(128, self.num_ctx - 256))
        self.input_budget = max(256, self.num_ctx - self.output_reserve)
        self.compact_threshold_ratio = max(0.1, min(0.95, float(compact_threshold_ratio)))
        self.chars_per_token = max(1.5, float(chars_per_token))
        self.safety_margin_ratio = max(0.0, min(0.25, float(safety_margin_ratio)))
        self.calibration_factor: float = 1.0
        self.last_prompt_eval_count: int | None = None

    @property
    def threshold_tokens(self) -> int:
        return max(128, int(self.input_budget * self.compact_threshold_ratio))

    def effective_output_reserve(self, num_predict_override: int | None = None) -> int:
        """Calcula a reserva de saída efetiva sem permitir que overrides (ex.: report_tokens=4608)
        esmaguem o orçamento de entrada de modelos com contexto reduzido (ex.: num_ctx=4096)."""
        if num_predict_override is None:
            return self.output_reserve
        requested = max(32, int(num_predict_override))
        max_safe_reserve = max(self.output_reserve, self.num_ctx // 3)
        if requested > max_safe_reserve:
            return self.output_reserve
        return min(requested, max(128, self.num_ctx - 256))

    def estimate_text_tokens(self, text: str | None) -> int:
        if not text:
            return 0
        raw = pretoken_estimate(text) if self.pretoken else math.ceil(len(text) / self.chars_per_token)
        return max(1, math.ceil(raw * self.calibration_factor))

    def estimate_dense_text_tokens(self, text: str | None) -> int:
        """Estimativa conservadora que penaliza strings técnicas densas em símbolos (URLs, hashes, JSON)."""
        if not text:
            return 0
        if self.pretoken:
            return self.estimate_text_tokens(text)  # dígitos/símbolos já contados por pré-token
        base = self.estimate_text_tokens(text)
        symbols = len(_DENSE_SYMBOL_RE.findall(text))
        if symbols <= len(text) * 0.30:
            return base
        extra_chars = int((symbols - len(text) * 0.30) * 0.75)
        dense_raw = math.ceil((len(text) + extra_chars) / self.chars_per_token)
        return max(base, math.ceil(dense_raw * self.calibration_factor))

    def safety_margin_tokens(self, raw_tokens: int) -> int:
        if raw_tokens <= 0 or self.safety_margin_ratio <= 0:
            return 0
        return max(12, math.ceil(raw_tokens * self.safety_margin_ratio))

    def estimate_message_tokens(self, message: dict[str, Any], *, dense: bool = False) -> int:
        est = self.estimate_dense_text_tokens if dense else self.estimate_text_tokens
        total = 6  # overhead conservador de papel/delimitadores
        total += est(str(message.get("content") or ""))
        if message.get("tool_calls"):
            total += est(json.dumps(message["tool_calls"], ensure_ascii=False))
        if message.get("tool_name"):
            total += est(str(message["tool_name"]))
        img_tok = message.get("image_tokens")
        if isinstance(img_tok, (int, float)) and img_tok > 0:
            total += int(img_tok)
        elif message.get("images"):
            total += len(message["images"]) * 1024
        return total

    def estimate_messages_tokens(self, messages: list[dict[str, Any]], *, dense: bool = False) -> int:
        return sum(self.estimate_message_tokens(m, dense=dense) for m in messages)

    def estimate_tools_schema_tokens(self, tools: list[dict[str, Any]] | None) -> int:
        if not tools:
            return 0
        overhead = _TOOLS_TEMPLATE_TOKENS if self.pretoken else 0
        return overhead + self.estimate_text_tokens(json.dumps(tools, ensure_ascii=False))

    def validate_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        num_predict: int | None = None,
    ) -> PayloadValidation:
        """Valida obrigatoriamente o payload final antes de qualquer chamada de inferência."""
        output_reserve = self.effective_output_reserve(num_predict)
        input_budget = max(256, self.num_ctx - output_reserve)
        msg_tok = self.estimate_messages_tokens(messages, dense=True)
        tools_tok = self.estimate_tools_schema_tokens(tools)
        raw_input = msg_tok + tools_tok
        margin = self.safety_margin_tokens(raw_input)
        total_input = raw_input + margin
        total = total_input + output_reserve
        fits = bool(messages) and total_input <= input_budget and total <= self.num_ctx
        reason = None
        if not messages:
            reason = "Payload de inferência vazio."
        elif not fits:
            reason = (
                f"Payload final ({total_input} tokens estimados de entrada + {output_reserve} reserva de saída "
                f"= {total} tokens) excede o orçamento do modelo ativo (input_budget={input_budget}, num_ctx={self.num_ctx})."
            )
        return PayloadValidation(
            fits=fits,
            num_ctx=self.num_ctx,
            output_reserve=output_reserve,
            input_budget=input_budget,
            messages_tokens=msg_tok,
            tools_tokens=tools_tok,
            safety_margin_tokens=margin,
            total_input_tokens=total_input,
            total_tokens=total,
            reason=reason,
        )

    def calibrate(self, estimated_input_tokens: int, actual_prompt_tokens: int | None) -> float:
        """Calibra o fator de estimativa com base em prompt_eval_count real do Ollama."""
        if not actual_prompt_tokens or actual_prompt_tokens <= 0 or estimated_input_tokens <= 0:
            return self.calibration_factor
        self.last_prompt_eval_count = int(actual_prompt_tokens)
        ratio = (actual_prompt_tokens / estimated_input_tokens) * self.calibration_factor
        # piso 1.0: fator < 1 tornaria a estimativa otimista e o payload real poderia
        # exceder input_budget mesmo com validate_payload aprovando (aviso "N/budget >100%")
        clamped = max(1.0, min(1.8, ratio))
        # Suavização exponencial para evitar saltos por uma única resposta
        self.calibration_factor = round(0.6 * self.calibration_factor + 0.4 * clamped, 3)
        return self.calibration_factor

    def assess(
        self,
        *,
        system_base_text: str = "",
        tools_block: str = "",
        skills_block: str = "",
        memory_block: str = "",
        summary_block: str = "",
        history_messages: list[dict[str, Any]] | None = None,
        tools_schema: list[dict[str, Any]] | None = None,
        num_predict_override: int | None = None,
    ) -> ContextBudget:
        output_reserve = self.effective_output_reserve(num_predict_override)
        input_budget = max(256, self.num_ctx - output_reserve)
        threshold_tokens = max(128, int(input_budget * self.compact_threshold_ratio))

        sys_tok = self.estimate_text_tokens(system_base_text) + 6
        tools_tok = self.estimate_text_tokens(tools_block) + self.estimate_tools_schema_tokens(tools_schema)
        skills_tok = self.estimate_text_tokens(skills_block)
        mem_tok = self.estimate_text_tokens(memory_block)
        sum_tok = self.estimate_text_tokens(summary_block)
        hist_tok = self.estimate_messages_tokens(history_messages or [])
        total = sys_tok + tools_tok + skills_tok + mem_tok + sum_tok + hist_tok
        ratio = round(total / input_budget, 4) if input_budget else 1.0

        return ContextBudget(
            num_ctx=self.num_ctx,
            output_reserve=output_reserve,
            input_budget=input_budget,
            threshold_ratio=self.compact_threshold_ratio,
            threshold_tokens=threshold_tokens,
            system_tokens=sys_tok,
            tools_tokens=tools_tok,
            skills_tokens=skills_tok,
            memory_tokens=mem_tok,
            summary_tokens=sum_tok,
            history_tokens=hist_tok,
            total_input_tokens=total,
            usage_ratio=ratio,
            calibration_factor=self.calibration_factor,
            needs_repack=total >= threshold_tokens,
            over_hard_limit=total > input_budget,
        )

from pydantic import BaseModel, Field

DEFAULT_LANGUAGE = "en-US"

# Interface language -> clear default reply language instruction.
# Placed after base directives and before the owner's additional instructions; explicit user requests always prevail.
LANGUAGE_HINTS = {
    "en-US": (
        "Default reply language: English (US). Reply in English (US) by default — including greetings, "
        "explanations, and standard responses — even if parts of the context or tools use another language. "
        "If the user explicitly asks for another language in their message or additional instructions, use that requested language."
    ),
    "pt-BR": (
        "Idioma padrão das respostas: português do Brasil. Responda em português do Brasil por padrão — "
        "inclusive saudações, explicações e respostas padrão — mesmo que partes das diretrizes ou contexto estejam em outro idioma. "
        "Se o usuário pedir explicitamente outro idioma na mensagem ou nas instruções adicionais, use o idioma pedido."
    ),
}


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=32000)
    conversation_id: str | None = None
    system_prompt: str | None = Field(None, max_length=8000)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    num_predict: int | None = Field(None, ge=1, le=32768)
    use_tools: bool | None = None
    think: bool | None = None  # liga/desliga o raciocínio visível nesta mensagem
    activated_skills: list[str] = Field(default_factory=list, max_length=10)
    resume: bool = False
    retry: bool = False  # repete a última pergunta salva (após decidir a pesquisa na internet)
    allow_unverified: bool = False  # o usuário aceitou uma resposta só do modelo, rotulada NOT VERIFIED
    research_mode: str | None = Field(None, pattern="^(quick|deep|extended)$")  # nível escolhido no cartão (só esta investigação)
    attachment_ids: list[str] = Field(default_factory=list, max_length=8)
    language: str | None = Field(None, pattern="^(pt-BR|en-US)$")

    def effective_system_prompt(self) -> str | None:
        """Owner instructions preceded by the language instruction (owner's text comes after and prevails)."""
        hint = LANGUAGE_HINTS.get(self.language or "")
        if not hint:
            return self.system_prompt
        own = (self.system_prompt or "").strip()
        return f"{hint}\n\n{own}" if own else hint

    def provider_options(self) -> dict:
        options: dict = {}
        if self.temperature is not None:
            options["temperature"] = self.temperature
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        if self.think is not None:
            options["think"] = self.think
        return options

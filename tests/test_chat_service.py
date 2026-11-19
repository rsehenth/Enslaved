import asyncio

import pytest

from backend.chat.service import ChatService, ConversationNotFoundError
from backend.providers.base import ModelProvider, ProviderUnavailableError
from backend.providers.simulated import SimulatedProvider


class FailingProvider(ModelProvider):
    name = "failing"

    async def chat(self, messages, **options):
        raise ProviderUnavailableError("provider fora do ar")

    async def stream_chat(self, messages, **options):
        raise ProviderUnavailableError("provider fora do ar")
        yield  # noqa: unreachable - torna a função um gerador assíncrono

    async def health(self):
        return {"available": False, "detail": "failing", "models": []}


def test_send_creates_conversation_and_persists(db, directives):
    provider = SimulatedProvider(model="simulado-1b")
    service = ChatService(db, provider, directives)
    result = asyncio.run(service.send(None, "Olá, Enslaved!"))
    conversation = result["conversation"]
    assert conversation.title == "Olá, Enslaved!"
    assert conversation.message_count == 2
    messages = service.repo.get_messages(conversation.id)
    assert [m.role for m in messages] == ["user", "assistant"]
    assert messages[-1].meta == {"provider": "simulated", "model": "simulado-1b"}


def test_send_builds_payload_with_directives(db, directives):
    provider = SimulatedProvider()
    service = ChatService(db, provider, directives)
    asyncio.run(service.send(None, "first"))
    asyncio.run(service.send(None, "second"))  # new conversation: no cross-session history
    payload = provider.calls[1]
    assert payload[0]["role"] == "system"
    assert "Reply directly" in payload[0]["content"]
    assert payload[-1] == {"role": "user", "content": "second"}
    assert len(payload) == 2


def test_send_keeps_history_within_conversation(db, directives):
    provider = SimulatedProvider()
    service = ChatService(db, provider, directives)
    result = asyncio.run(service.send(None, "primeira"))
    conversation_id = result["conversation"].id
    asyncio.run(service.send(conversation_id, "segunda"))
    payload = provider.calls[1]
    assert [m["role"] for m in payload] == ["system", "user", "assistant", "user"]


def test_send_with_extra_system_prompt(db, directives):
    provider = SimulatedProvider()
    service = ChatService(db, provider, directives)
    asyncio.run(service.send(None, "hi", system_prompt="Be extra brief."))
    system = provider.calls[0][0]["content"]
    assert "Additional Owner Instructions" in system
    assert "Be extra brief." in system


def test_send_to_unknown_conversation_raises(db, directives):
    service = ChatService(db, SimulatedProvider(), directives)
    with pytest.raises(ConversationNotFoundError):
        asyncio.run(service.send("id-inexistente", "oi"))


def test_empty_message_rejected(db, directives):
    service = ChatService(db, SimulatedProvider(), directives)
    with pytest.raises(ValueError):
        asyncio.run(service.send(None, "   "))


def test_provider_failure_keeps_user_message_without_assistant(db, directives):
    service = ChatService(db, FailingProvider(model="f"), directives)
    with pytest.raises(ProviderUnavailableError):
        asyncio.run(service.send(None, "mensagem que falha"))
    conversation = service.repo.list_conversations()[0]
    messages = service.repo.get_messages(conversation.id)
    assert [m.role for m in messages] == ["user"]


def test_send_stream_events_and_persistence(db, directives):
    service = ChatService(db, SimulatedProvider(model="simulado-1b"), directives)

    async def collect():
        return [event async for event in service.send_stream(None, "oi stream")]

    events = asyncio.run(collect())
    assert events[0]["type"] == "meta"
    assert events[0]["conversation_title"] == "oi stream"
    assert [e["type"] for e in events[-2:]] == ["token", "done"]
    messages = service.repo.get_messages(events[0]["conversation_id"])
    assert [m.role for m in messages] == ["user", "assistant"]
    assert events[-1]["message_id"] == messages[-1].id
    assert events[-1]["provider"] == "simulated"
    assert events[-1]["model"] == "simulado-1b"


def test_send_stream_provider_error_event(db, directives):
    service = ChatService(db, FailingProvider(model="f"), directives)

    async def collect():
        return [event async for event in service.send_stream(None, "oi")]

    events = asyncio.run(collect())
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "error"
    assert "fora do ar" in events[-1]["message"]

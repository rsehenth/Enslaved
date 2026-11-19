import pytest

from backend.storage.conversations import ConversationRepository


def test_create_and_get_conversation(db):
    repo = ConversationRepository(db)
    conversation = repo.create_conversation()
    fetched = repo.get_conversation(conversation.id)
    assert fetched is not None
    assert fetched.title == "New chat"
    assert fetched.message_count == 0


def test_add_messages_orders_and_auto_titles(db):
    repo = ConversationRepository(db)
    conversation = repo.create_conversation()
    first = repo.add_message(conversation.id, "user", "Olá, como você está?")
    second = repo.add_message(conversation.id, "assistant", "Estou bem, obrigado.")
    messages = repo.get_messages(conversation.id)
    assert [m.id for m in messages] == [first.id, second.id]
    assert [m.seq for m in messages] == [1, 2]
    assert [m.role for m in messages] == ["user", "assistant"]
    updated = repo.get_conversation(conversation.id)
    assert updated.title == "Olá, como você está?"


def test_all_role_values_accepted(db):
    repo = ConversationRepository(db)
    conversation = repo.create_conversation()
    for role in ("system", "user", "assistant", "tool"):
        repo.add_message(conversation.id, role, f"texto {role}")
    roles = [m.role for m in repo.get_messages(conversation.id)]
    assert roles == ["system", "user", "assistant", "tool"]


def test_message_meta_persisted(db):
    repo = ConversationRepository(db)
    conversation = repo.create_conversation()
    repo.add_message(
        conversation.id,
        "assistant",
        "resposta",
        meta={"provider": "ollama", "model": "qwen3.5:4b"},
    )
    fetched = repo.get_messages(conversation.id)[0]
    assert fetched.meta == {"provider": "ollama", "model": "qwen3.5:4b"}


def test_invalid_role_rejected(db):
    repo = ConversationRepository(db)
    conversation = repo.create_conversation()
    with pytest.raises(ValueError):
        repo.add_message(conversation.id, "chef", "oi")


def test_delete_conversation_cascades(db):
    repo = ConversationRepository(db)
    conversation = repo.create_conversation()
    repo.add_message(conversation.id, "user", "olá")
    assert repo.delete_conversation(conversation.id) is True
    assert repo.get_conversation(conversation.id) is None
    assert repo.get_messages(conversation.id) == []
    assert repo.delete_conversation(conversation.id) is False


def test_list_conversations_with_counts_and_preview(db):
    repo = ConversationRepository(db)
    a = repo.create_conversation()
    b = repo.create_conversation(title="Título fixo")
    repo.add_message(a.id, "user", "primeira conversa")
    repo.add_message(b.id, "user", "segunda conversa")
    listed = repo.list_conversations()
    assert [c.id for c in listed] == [b.id, a.id]
    assert listed[1].message_count == 1
    assert listed[1].preview == "primeira conversa"
    assert listed[0].title == "Título fixo"

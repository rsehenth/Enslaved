import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from tests.helpers import DIRECTIVES_TEXT, make_settings


def test_health_and_config(client):
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert health["provider"]["available"] is True
    assert health["provider"]["model"] == "simulado-1b"

    config = client.get("/api/config").json()
    assert config["app"]["name"] == "Enslaved Teste"
    assert config["directives"]["name"] == "Enslaved"
    assert config["defaults"]["num_ctx"] == 16384


def test_index_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Enslaved" in response.text


def test_full_conversation_flow(client):
    first = client.post("/api/chat", json={"message": "Olá, tudo bem?"})
    assert first.status_code == 200
    data = first.json()
    conversation_id = data["conversation_id"]
    assert data["reply"].startswith("[simulado]")
    assert data["user_message"]["role"] == "user"
    assert data["assistant_message"]["meta"]["model"] == "simulado-1b"

    listed = client.get("/api/conversations").json()["conversations"]
    assert listed[0]["id"] == conversation_id
    assert listed[0]["title"] == "Olá, tudo bem?"

    detail = client.get(f"/api/conversations/{conversation_id}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]

    follow_up = client.post(
        "/api/chat",
        json={
            "message": "Continue.",
            "conversation_id": conversation_id,
            "temperature": 0.2,
            "num_predict": 64,
        },
    )
    assert follow_up.status_code == 200
    assert follow_up.json()["conversation_id"] == conversation_id

    assert client.delete(f"/api/conversations/{conversation_id}").json()["deleted"] is True
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404
    assert (
        client.post("/api/chat", json={"message": "oi", "conversation_id": conversation_id}).status_code
        == 404
    )
    assert client.get("/api/conversations").json()["conversations"] == []


def test_invalid_message_rejected(client):
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400


def test_empty_message_rejected(client):
    # mensagem vazia passa pelo schema (resume não envia texto) e é rejeitada na regra de negócio
    assert client.post("/api/chat", json={"message": ""}).status_code == 400


def test_skills_endpoints(client):
    listing = client.get("/api/skills").json()
    ids = [s["id"] for s in listing["skills"]]
    assert "hello-enslaved" in ids
    assert "skill-teste" in ids
    assert any(error["reason"] for error in listing["errors"])

    detail = client.get("/api/skills/hello-enslaved").json()
    assert "Demo skill" in detail["description"]
    assert detail["instructions"]
    assert detail["has_references"] is True
    assert client.get("/api/skills/inexistente").status_code == 404


def test_chat_stream_endpoint(client):
    response = client.post("/api/chat/stream", json={"message": "oi stream"})
    assert response.status_code == 200
    body = response.text
    assert '"meta"' in body
    assert '"token"' in body
    assert '"done"' in body


def test_unavailable_provider_returns_503(tmp_path):
    settings = make_settings(
        tmp_path,
        provider={"name": "ollama", "model": "qwen3.5:4b"},
        ollama={"base_url": "http://127.0.0.1:1", "timeout_seconds": 2},
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/chat", json={"message": "oi"})
        assert response.status_code == 503
        assert "indisponível" in response.json()["error"]

        health = client.get("/api/health").json()
        assert health["provider"]["available"] is False

        stream = client.post("/api/chat/stream", json={"message": "oi"})
        assert stream.status_code == 200
        assert '"error"' in stream.text
        assert "indisponível" in stream.text


def test_directives_loaded_from_real_config(client):
    config = client.get("/api/config").json()
    assert config["directives"]["language"] == "en-US"


def test_make_settings_writes_directives(tmp_path):
    settings = make_settings(tmp_path)
    assert settings.directives.path.read_text(encoding="utf-8").startswith("---")

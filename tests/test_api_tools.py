import pytest

from tests.helpers import make_settings


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    from backend.main import create_app

    with TestClient(create_app(make_settings(tmp_path))) as test_client:
        yield test_client


def test_list_tools_reports_catalog_and_modes(client):
    data = client.get("/api/tools").json()
    ids = {tool["id"] for tool in data["tools"]}
    assert {"fs.read", "net.fetch", "text.analyze", "app.diagnostics"} <= ids
    modes = {tool["id"]: tool["mode"] for tool in data["tools"]}
    assert modes["fs.read"] == "ask"
    assert modes["net.fetch"] == "ask"
    assert modes["app.diagnostics"] == "allow"
    assert modes["text.analyze"] == "allow"
    # settings de teste têm tools desabilitadas para o chat, mas o catálogo existe
    assert data["enabled"] is False
    fs_schema = next(t for t in data["tools"] if t["id"] == "fs.read")["parameters"]
    assert "path" in fs_schema["properties"]


def test_invoke_allow_mode_tool_executes(client):
    response = client.post("/api/tools/text.analyze/invoke", json={"arguments": {"text": "Olá mundo. Mais uma frase aqui."}})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "executed"
    assert data["result"]["ok"] is True
    assert data["result"]["output"]["stats"]["words"] >= 5
    assert data["result"]["data"]["tool"] == "text.analyze"
    assert data["result"]["data"]["duration_ms"] >= 0


def test_invoke_ask_mode_requires_permission_then_executes(client):
    first = client.post("/api/tools/fs.read/invoke", json={"arguments": {"path": "inexistente.txt"}}).json()
    assert first["status"] == "permission_required"
    request_id = first["request"]["id"]
    assert first["request"]["tool_id"] == "fs.read"
    assert "fs.read" in first["request"]["capabilities"]

    pending = client.get("/api/permissions").json()
    assert any(p["id"] == request_id for p in pending["pending"])

    decision = client.post(
        f"/api/permissions/{request_id}/decision", json={"granted": True, "remember_session": True}
    ).json()
    assert decision["granted"] is True
    # a ferramenta executou (arquivo inexistente → resultado estruturado de erro da ferramenta)
    assert decision["result"]["ok"] is False
    assert "não encontrado" in decision["result"]["error"]

    # concessão de sessão: próxima invocação não pede permissão
    second = client.post("/api/tools/fs.read/invoke", json={"arguments": {"path": "inexistente.txt"}}).json()
    assert second["status"] == "executed"
    tools_state = client.get("/api/tools").json()
    assert "fs.read" in tools_state["session_grants"]


def test_invoke_denied_keeps_pending_from_executing(client):
    first = client.post("/api/tools/fs.read/invoke", json={"arguments": {"path": "x.txt"}}).json()
    request_id = first["request"]["id"]
    decision = client.post(f"/api/permissions/{request_id}/decision", json={"granted": False}).json()
    assert decision["granted"] is False
    assert decision["result"]["ok"] is False
    assert "negada" in decision["result"]["error"]
    tools_state = client.get("/api/tools").json()
    assert "fs.read" not in tools_state["session_grants"]


def test_decision_unknown_request_returns_404(client):
    response = client.post("/api/permissions/inexistente/decision", json={"granted": True})
    assert response.status_code == 404


def test_invoke_unknown_tool_reports_invalid(client):
    response = client.post("/api/tools/inexistente/invoke", json={"arguments": {}})
    assert response.status_code == 200
    assert response.json()["status"] == "invalid"


def test_invoke_with_invalid_arguments_reports_invalid(client):
    response = client.post("/api/tools/text.analyze/invoke", json={"arguments": {"text": ""}})
    assert response.json()["status"] == "invalid"
    assert "Argumentos inválidos" in response.json()["error"]


def test_tool_history_records_runs(client):
    client.post("/api/tools/app.diagnostics/invoke", json={"arguments": {}})
    history = client.get("/api/tools/history?limit=10").json()
    kinds = [event["event"] for event in history["events"]]
    assert "tool_run" in kinds
    run = next(e for e in history["events"] if e["event"] == "tool_run")
    assert run["tool"] == "app.diagnostics"
    assert run["source"] == "manual"


def test_skills_endpoint_reports_enabled_state_and_toggle(client):
    data = client.get("/api/skills").json()
    assert "summary" in data
    assert data["summary"]["total"] >= 1
    skill = next(s for s in data["skills"] if s["id"] == "skill-teste")
    assert skill["enabled"] is True
    assert skill["available"] is True

    toggle = client.post("/api/skills/skill-teste/enabled", json={"enabled": False})
    assert toggle.status_code == 200
    data = client.get("/api/skills").json()
    assert next(s for s in data["skills"] if s["id"] == "skill-teste")["enabled"] is False
    assert "skill-teste" in data["summary"]["disabled_ids"]

    client.post("/api/skills/skill-teste/enabled", json={"enabled": True})
    unknown = client.post("/api/skills/nao-existe/enabled", json={"enabled": True})
    assert unknown.status_code == 404


def test_skill_detail_includes_requirements(client):
    detail = client.get("/api/skills/hello-enslaved").json()
    assert detail["id"] == "hello-enslaved"
    assert detail["instructions"]
    assert detail["missing_requirements"] == []

from fastapi.testclient import TestClient

import backend.api.routes_system as routes_system
from backend.core.gpu import parse_nvidia_smi
from backend.main import create_app
from backend.providers.simulated import SimulatedProvider
from tests.helpers import make_settings


def test_parse_nvidia_smi_output():
    output = "NVIDIA GeForce RTX 3080 Ti, 2302, 12288, 9986, 17, 57\n"
    assert parse_nvidia_smi(output) == [
        {
            "name": "NVIDIA GeForce RTX 3080 Ti",
            "memory_used_mb": 2302,
            "memory_total_mb": 12288,
            "memory_free_mb": 9986,
            "utilization_gpu_percent": 17,
            "temperature_c": 57,
        }
    ]


def test_parse_nvidia_smi_handles_garbage():
    assert parse_nvidia_smi("no data available") == []
    assert parse_nvidia_smi("") == []


def test_parse_nvidia_smi_without_temperature_column():
    rows = parse_nvidia_smi("GPU X, 100, 200, 100, 5\n")
    assert rows[0]["temperature_c"] is None


def test_diagnostics_endpoint_with_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(
        routes_system,
        "gpu_snapshot",
        lambda: [
            {
                "name": "NVIDIA GeForce RTX 3080 Ti",
                "memory_used_mb": 2302,
                "memory_total_mb": 12288,
                "memory_free_mb": 9986,
                "utilization_gpu_percent": 17,
                "temperature_c": 57,
            }
        ],
    )
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        data = client.get("/api/diagnostics").json()
    assert data["provider"]["name"] == "simulated"
    assert data["provider"]["model"] == "simulado-1b"
    assert data["provider"]["available"] is True
    assert data["loaded_models"] == []
    assert data["gpus"][0]["memory_total_mb"] == 12288
    assert data["settings"]["num_ctx"] == 16384
    assert data["settings"]["num_predict"] == 2048


def test_diagnostics_endpoint_without_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_system, "gpu_snapshot", lambda: None)
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        data = client.get("/api/diagnostics").json()
    assert data["gpus"] is None
    assert data["provider"]["available"] is True


def test_diagnostics_reports_loaded_models(tmp_path, monkeypatch):
    monkeypatch.setattr(routes_system, "gpu_snapshot", lambda: None)

    async def fake_loaded_models(self):
        return [{"name": "qwen3.5:4b", "size_mb": 3147, "vram_mb": 4196, "expires_at": None}]

    monkeypatch.setattr(SimulatedProvider, "loaded_models", fake_loaded_models)
    settings = make_settings(tmp_path)
    with TestClient(create_app(settings)) as client:
        data = client.get("/api/diagnostics").json()
    assert data["loaded_models"][0]["name"] == "qwen3.5:4b"
    assert data["loaded_models"][0]["vram_mb"] == 4196

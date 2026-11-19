"""start.bat / start-ollama.ps1 / stop.ps1: encerramento só do que é comprovadamente do Enslaved.

Cada teste roda numa cópia isolada do projeto (tmp_path) com processos próprios; nada da máquina é tocado.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.name != "nt", reason="scripts de inicialização/encerramento são do Windows")

PROJECT = Path(__file__).resolve().parents[1]
PS = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass"]


def _alive(pid: int) -> bool:
    out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
    return str(pid) in out


def _kill(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def _make_root(root: Path) -> Path:
    (root / "backend").mkdir(parents=True)
    (root / "backend" / "__init__.py").write_text("")
    (root / "backend" / "main.py").write_text("import time; time.sleep(120)\n")
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(root / ".venv")], check=True)
    for name in ("stop.ps1", "start-ollama.ps1", "start.bat"):
        shutil.copy(PROJECT / name, root / name)
    return root


@pytest.fixture
def env(tmp_path):
    started: list[int] = []
    fake = tmp_path / "fake" / "ollama.exe"
    fake.parent.mkdir()
    # executável que aceita "serve" e fica vivo, como o Ollama real
    src = "public static class P { public static void Main(string[] a) { System.Threading.Thread.Sleep(120000); } }"
    subprocess.run(PS + ["-Command", f"Add-Type -OutputType ConsoleApplication -OutputAssembly '{fake}' -TypeDefinition '{src}'"], check=True)
    ours, other = _make_root(tmp_path / "Enslaved"), _make_root(tmp_path / "Other")

    def backend(root: Path) -> int:
        p = subprocess.Popen([str(root / ".venv" / "Scripts" / "python.exe"), "-m", "backend.main"], cwd=root)
        started.append(p.pid)
        return p.pid

    def stray_ollama() -> int:
        p = subprocess.Popen([str(fake), "serve"])
        started.append(p.pid)
        return p.pid

    def start_ollama(root: Path) -> int:  # exatamente como o start.bat chama
        subprocess.run(PS + ["-File", str(root / "start-ollama.ps1"), "-Exe", str(fake)], check=True, capture_output=True)
        pid = json.loads((root / ".run" / "ollama.json").read_text())["pid"]
        started.append(pid)
        return pid

    def stop(root: Path, simulate: bool = False) -> str:
        args = PS + ["-File", str(root / "stop.ps1")] + (["-DryRun"] if simulate else [])
        return subprocess.run(args, capture_output=True, text=True).stdout

    yield {"ours": ours, "other": other, "fake": fake, "backend": backend, "stray_ollama": stray_ollama,
           "start_ollama": start_ollama, "stop": stop}
    for pid in started:
        if _alive(pid):
            _kill(pid)


def test_stop_only_kills_what_enslaved_owns(env):
    ours, other = env["ours"], env["other"]
    mine, theirs = env["backend"](ours), env["backend"](other)
    owned = env["start_ollama"](ours)
    record = json.loads((ours / ".run" / "ollama.json").read_text())
    assert Path(record["path"]) == env["fake"] and record["startTimeTicks"]  # regressão: $PSScriptRoot vazio no PS 5.1
    shared = env["stray_ollama"]()
    time.sleep(1)

    out = env["stop"](ours, simulate=True)
    assert all(_alive(p) for p in (mine, theirs, owned, shared)), out
    assert (ours / ".run" / "ollama.json").exists()

    out = env["stop"](ours)
    time.sleep(1)
    assert not _alive(mine) and not _alive(owned), out
    assert _alive(theirs) and _alive(shared), out
    assert not (ours / ".run" / "ollama.json").exists()
    assert f"PID {shared}" in out and "pre-existing or shared" in out


def test_stop_preserves_process_with_reused_pid(env):
    ours = env["ours"]
    victim = env["stray_ollama"]()
    time.sleep(0.5)
    (ours / ".run").mkdir()
    (ours / ".run" / "ollama.json").write_text(json.dumps({"pid": victim, "startTimeTicks": 1, "path": str(env["fake"])}))
    out = env["stop"](ours)
    assert _alive(victim) and "PID reused" in out, out
    assert not (ours / ".run" / "ollama.json").exists()


def test_start_bat_rejects_arguments(env):
    result = subprocess.run(["cmd", "/c", str(env["ours"] / "start.bat"), "local"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    assert result.returncode == 2 and "Unknown argument" in result.stdout


def test_start_bat_always_serves_this_pc_and_local_network():
    text = (PROJECT / "start.bat").read_text(encoding="utf-8")
    assert 'set "ENSLAVED_LAN=1"' in text and 'set "ENSLAVED_HOST=0.0.0.0"' in text
    assert "ENSLAVED_LAN=0" not in text and 'ENSLAVED_HOST=127.0.0.1"' not in text
    assert b"\r\n" in (PROJECT / "start.bat").read_bytes()  # cmd labels/goto need CRLF


def test_script_output_is_english_ascii():
    for name in ("start.bat", "stop.bat", "stop.ps1", "start-ollama.ps1"):
        shown = [l for l in (PROJECT / name).read_text(encoding="utf-8").splitlines()
                 if l.strip().lower().startswith("echo") or "Write-Host" in l]
        assert shown and all(l.isascii() for l in shown), name

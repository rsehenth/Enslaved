"""Snapshot de recursos de GPU via nvidia-smi (opcional: ausência degrada para None)."""

import shutil
import subprocess

_QUERY = "name,memory.used,memory.total,memory.free,utilization.gpu,temperature.gpu"


def _to_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except ValueError:
        return None


def parse_nvidia_smi(output: str) -> list[dict]:
    gpus: list[dict] = []
    for line in output.strip().splitlines():
        values = [v.strip() for v in line.split(",")]
        if len(values) < 5:
            continue
        gpus.append(
            {
                "name": values[0],
                "memory_used_mb": _to_int(values[1]),
                "memory_total_mb": _to_int(values[2]),
                "memory_free_mb": _to_int(values[3]),
                "utilization_gpu_percent": _to_int(values[4]),
                "temperature_c": _to_int(values[5]) if len(values) > 5 else None,
            }
        )
    return gpus


def gpu_snapshot() -> list[dict] | None:
    """Consulta nvidia-smi; retorna None quando a ferramenta não está disponível."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, f"--query-gpu={_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return parse_nvidia_smi(result.stdout)

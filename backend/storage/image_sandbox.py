"""Decodificação e normalização de imagens em subprocesso isolado.

O processo principal NUNCA decodifica o arquivo enviado: só confere assinatura/tamanho e entrega os bytes a um
filho (`python -I`) que roda com prazo total e, no Windows, dentro de um Job Object com teto de memória
(infraestrutura de `backend/core/sandbox.py`). O filho valida o formato real, as dimensões e os pixels *antes*
de decodificar, decodifica por completo, aplica a orientação EXIF, converte perfis ICC para sRGB quando possível e
recodifica a partir dos pixels — por isso metadados (EXIF/GPS, comentários, miniaturas, bytes anexados ao final)
nunca chegam ao arquivo armazenado nem ao Ollama.

Limite honesto: o filho roda com o mesmo usuário do servidor. É isolamento de recursos (CPU/tempo/memória) e de
falhas (um decodificador que trava ou estoura memória não derruba o servidor), não uma fronteira de privilégios.
Este módulo só importa a biblioteca padrão no topo para o filho iniciar rápido e com pouca memória.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_BOOT = "import sys; sys.path.insert(0, sys.argv[1]); from backend.storage.image_sandbox import _child_main; _child_main(sys.argv[2:])"
_FORMATS = {"PNG": ("image/png", "png"), "JPEG": ("image/jpeg", "jpg"), "WEBP": ("image/webp", "webp")}
_MSG_INVALID = "O arquivo não é uma imagem PNG, JPEG ou WebP válida."
_MSG_RESOURCES = "Imagem recusada: a decodificação excedeu o limite de tempo/memória ou o arquivo está corrompido."


class SandboxRejected(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.message = message
        self.status = status


# ------------------------------------------------------------------ lado do servidor


def run_image_job(op: str, data: bytes, params: dict[str, Any], *, timeout: float, memory_limit: int, max_output: int) -> tuple[dict[str, Any], bytes]:
    """Executa `normalize`/`resize` no filho. Devolve (metadados, bytes da imagem). Falha fechada: nunca decodifica no pai."""
    from backend.core.sandbox import run_limited  # import tardio: o filho não paga imports extras

    work = Path(tempfile.mkdtemp(prefix="enslaved-img-"))
    try:
        src, dst = work / "input.bin", work / "output.bin"
        src.write_bytes(data)
        env = {**os.environ, "TEMP": str(work), "TMP": str(work), "TMPDIR": str(work), "PYTHONIOENCODING": "utf-8"}
        cmd = [sys.executable, "-I", "-c", _BOOT, str(PROJECT_ROOT), op, str(src), str(dst), json.dumps(params)]
        try:
            code, out, _isolated, timed_out = run_limited(cmd, timeout=timeout, memory_limit=memory_limit, env=env, cwd=str(work))
        except OSError as exc:
            raise SandboxRejected("Não foi possível iniciar a validação isolada da imagem.", 503) from exc
        if timed_out:
            raise SandboxRejected("Tempo limite de processamento da imagem excedido.", 408)
        try:
            payload = json.loads(out.decode("utf-8")) if out else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if code != 0 or not isinstance(payload, dict):
            raise SandboxRejected(_MSG_RESOURCES, 422)
        if "error" in payload:
            raise SandboxRejected(str(payload["error"]), int(payload.get("status", 415)))
        if not dst.is_file() or dst.stat().st_size > max_output:
            raise SandboxRejected("A imagem normalizada excede o limite de tamanho.", 413)
        return payload, dst.read_bytes()
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ------------------------------------------------------------------ lado do filho (processo isolado)


class _Reject(Exception):
    def __init__(self, message: str, status: int = 415):
        super().__init__(message)
        self.status = status


def _open_checked(data: bytes, p: dict[str, Any]):
    import io

    from PIL import Image

    max_pixels = int(p["max_pixels"])
    Image.MAX_IMAGE_PIXELS = max_pixels  # Pillow também recusa bombas de descompressão antes de alocar pixels
    try:
        img = Image.open(io.BytesIO(data), formats=list(_FORMATS))
    except Image.DecompressionBombError as exc:
        raise _Reject(
            f"Dimensões acima do limite (máximo {p['max_side']}px de lado e {max_pixels // 1_000_000} megapixels).", 413
        ) from exc
    fmt = img.format or ""
    if fmt not in _FORMATS:
        raise _Reject("Formato não suportado. Use PNG, JPEG ou WebP.")
    w, h = img.size  # só o cabeçalho foi lido até aqui
    if w < 1 or h < 1 or max(w, h) > int(p["max_side"]) or w * h > max_pixels:
        raise _Reject(
            f"Dimensões acima do limite ({w}x{h}; máximo {p['max_side']}px de lado e {max_pixels // 1_000_000} megapixels).", 413
        )
    if getattr(img, "n_frames", 1) > 1:
        raise _Reject("Imagens animadas não são suportadas.")
    return img, fmt


def _save_clean(img, fmt: str, dst: str) -> int:
    """Recodifica a partir dos pixels (sem `info`, EXIF, ICC, texto ou dados anexados)."""
    from PIL import Image

    mode = img.mode
    if fmt == "JPEG":
        if mode != "RGB" and mode != "L":
            img = img.convert("RGB")
    else:
        if mode == "P":
            img = img.convert("RGBA" if "transparency" in img.info else "RGB")
        elif mode not in ("L", "LA", "RGB", "RGBA"):
            img = img.convert("RGBA" if "A" in mode or "transparency" in img.info else "RGB")
    clean = Image.frombytes(img.mode, img.size, img.tobytes())
    kwargs: dict[str, Any] = {"format": fmt}
    if fmt == "JPEG":
        kwargs.update(quality=92, optimize=True)
    elif fmt == "WEBP":
        kwargs.update(quality=92, method=4)
    else:
        kwargs.update(compress_level=6)
    clean.save(dst, **kwargs)
    return os.path.getsize(dst)


def _normalize(src: str, dst: str, p: dict[str, Any]) -> dict[str, Any]:
    import io

    from PIL import Image, ImageOps

    data = Path(src).read_bytes()
    img, fmt = _open_checked(data, p)
    with img:
        img.load()  # decodificação completa: recusa arquivos truncados/corrompidos
        had_metadata = bool(img.getexif()) or any(k in img.info for k in ("exif", "icc_profile", "comment", "xmp", "XML:com.adobe.xmp"))
        icc = img.info.get("icc_profile")
        work = img
        if icc and work.mode in ("RGB", "RGBA", "L"):
            try:
                from PIL import ImageCms

                converted = ImageCms.profileToProfile(
                    work, ImageCms.ImageCmsProfile(io.BytesIO(icc)), ImageCms.createProfile("sRGB"), outputMode=work.mode
                )
                if converted is not None:
                    work = converted
            except Exception:  # perfil inválido/ausente: segue sem conversão de cor
                pass
        before = work.size
        work = ImageOps.exif_transpose(work)  # orientação correta antes de descartar o EXIF
        oriented = work.size != before or bool(img.getexif().get(0x0112, 1) not in (0, 1))
        size = _save_clean(work, fmt, dst)
        with Image.open(dst) as check:  # confere o resultado recodificado
            width, height = check.size
    mime, ext = _FORMATS[fmt]
    return {"format": fmt, "mime": mime, "ext": ext, "width": width, "height": height, "size": size, "metadata_removed": had_metadata, "oriented": bool(oriented)}


def _resize(src: str, dst: str, p: dict[str, Any]) -> dict[str, Any]:
    from PIL import Image

    img, fmt = _open_checked(Path(src).read_bytes(), p)
    with img:
        img.load()
        resized = img.resize((int(p["width"]), int(p["height"])), Image.Resampling.LANCZOS)
        size = _save_clean(resized, fmt, dst)
    mime, ext = _FORMATS[fmt]
    return {"format": fmt, "mime": mime, "ext": ext, "width": int(p["width"]), "height": int(p["height"]), "size": size}


def _child_main(argv: list[str]) -> None:
    op, src, dst, params_json = argv
    sys.stdin.buffer.read(1)  # espera o pai colocar este processo no Job Object
    try:
        params = json.loads(params_json)
        result = _normalize(src, dst, params) if op == "normalize" else _resize(src, dst, params)
    except _Reject as exc:
        result = {"error": str(exc), "status": exc.status}
    except MemoryError:
        result = {"error": _MSG_RESOURCES, "status": 422}
    except Exception:  # decodificador quebrado por arquivo hostil: vira recusa genérica, sem traceback
        result = {"error": _MSG_INVALID, "status": 415}
    sys.stdout.buffer.write(json.dumps(result, ensure_ascii=False).encode("utf-8"))
    sys.stdout.flush()

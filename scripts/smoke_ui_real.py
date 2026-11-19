"""Smoke ÚNICO e REAL: interface (Chrome) + Qwen via Ollama + busca/páginas reais. Banco temporário.

Cobre: futebol → Jaspion, tabela Markdown, pergunta factual com páginas consultadas, pergunta sem internet
(com a oferta NÃO VERIFICADO) e conhecimento aprovado. Também confere streaming, permissões e animação.
Grava docs/response-quality-ui-smoke-evidence.json. Uso: .venv\\Scripts\\python scripts\\smoke_ui_real.py
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import uvicorn
from playwright.sync_api import sync_playwright

from backend.core.config import load_settings
from backend.main import create_app

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "docs" / "screenshots"
MOODS = "window.__moods = new Set(); setInterval(() => { const s = document.getElementById('scene'); if (s) window.__moods.add(s.dataset.mood); }, 40);"


def serve(net_fetch: str, port: int):
    """Modo servidor (processo separado, para não competir com o Playwright pelo mesmo processo)."""
    settings = load_settings()
    tmp = Path(tempfile.mkdtemp(prefix=f"enslaved-ui-{net_fetch}-"))
    settings.storage.database_path = tmp / "ui.db"
    settings.memory.base_dir = tmp / "memory"
    settings.learning.base_dir = tmp / "learning"
    settings.tools.permission_modes["net.fetch"] = net_fetch
    uvicorn.run(create_app(settings), host="127.0.0.1", port=port, log_level="warning")


class Server:
    def __init__(self, net_fetch: str):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            self.port = probe.getsockname()[1]
        self.proc = subprocess.Popen([sys.executable, __file__, "--serve", net_fetch, str(self.port)], cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.url = f"http://127.0.0.1:{self.port}"
        for _ in range(200):
            try:
                if httpx.get(self.url + "/api/health", timeout=2).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.2)
        raise RuntimeError("servidor não iniciou")

    def stop(self):
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)], capture_output=True)  # launcher do venv + filho


def start(net_fetch: str):
    server = Server(net_fetch)
    return server, server.url


def ask(page, text, approve=True, remember=True, timeout=240):
    """Envia pela interface, responde ao cartão de permissão como o proprietário e devolve o resultado da última resposta."""
    before = page.locator("#messages .message.assistant").count()
    page.fill("#input", text)
    page.keyboard.press("Enter")
    started, asked = time.time(), False
    while time.time() - started < timeout:
        card = page.locator(".message.permission:not(:has(.meta)) .permission-actions button.primary")
        if card.count() and card.first.is_enabled():
            asked = True
            if remember and approve:
                page.locator(".message.permission:not(:has(.meta)) input[type=checkbox]").first.check()
            (card.first if approve else page.locator(".message.permission:not(:has(.meta)) button.ghost").first).click()
        busy = page.evaluate("document.getElementById('composer').classList.contains('busy')")
        if not busy and page.locator("#messages .message.assistant").count() > before:
            break
        time.sleep(0.4)
    else:
        raise TimeoutError(text)
    last = page.locator("#messages .message.assistant").last
    note = last.locator(".verify-note")
    return {
        "pergunta": text,
        "resposta": last.locator(".content").inner_text(),
        "markdown_bruto": last.locator(".content").get_attribute("data-raw"),
        "cartao_de_permissao": asked,
        "verificacao": note.inner_text() if note.count() else None,
        "fontes": [a.get_attribute("href") for a in note.locator("a").all()] if note.count() else [],
        "tabelas": last.locator("table").count(),
        "br_literal": "<br" in last.locator(".content").inner_text(),
        "composer_liberado": not page.locator("#input").is_disabled(),
        "oferta_nao_verificado": last.locator(".verify-offer").count(),
    }


def main():
    evidence = {"cenarios": {}}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome")

        # ---------------------------------------------------------------- servidor com net.fetch = ask
        server, url = start("ask")
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        page.add_init_script(MOODS)
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(url)
        page.wait_for_selector("#input")
        rows = evidence["cenarios"]

        rows["futebol_e_jaspion"] = [ask(page, "Você conhece o Guarani de Bagé?"), ask(page, "Você conhece o Jaspion?")]
        page.screenshot(path=str(SHOTS / "ui-smoke-jaspion.png"))
        rows["tabela_markdown"] = ask(page, "Compare em uma tabela Python e JavaScript em 3 critérios.")
        page.screenshot(path=str(SHOTS / "ui-smoke-tabela.png"))
        rows["factual_com_paginas"] = ask(page, "Em que ano o Jaspion estreou no Brasil?")
        page.screenshot(path=str(SHOTS / "ui-smoke-factual.png"))

        http = httpx.Client(base_url=url, headers={"Origin": url}, timeout=60)
        evidence["animacao_estados_vistos"] = sorted(page.evaluate("[...window.__moods]"))
        evidence["erros_js"] = errors
        evidence["permissoes_na_sessao"] = http.get("/api/permissions").json()
        page.close()
        server.stop()

        # ---------------------------------------------------------------- servidor com net.fetch = deny (sem internet)
        server, url = start("deny")
        page = browser.new_page(viewport={"width": 1100, "height": 800})
        page.goto(url)
        page.wait_for_selector("#input")
        offline = {"recusa": ask(page, "Você conhece o Jaspion?")}
        page.screenshot(path=str(SHOTS / "ui-smoke-offline-oferta.png"))
        page.locator(".verify-offer").click()
        time.sleep(1.5)  # o clique inicia o stream; espera o composer ficar ocupado e depois livre
        started = time.time()
        while time.time() - started < 240 and page.evaluate("document.getElementById('composer').classList.contains('busy')"):
            time.sleep(0.4)
        last = page.locator("#messages .message.assistant:not(.thinking)").last
        offline["resposta_nao_verificada"] = last.locator(".content").inner_text()
        offline["rotulo"] = last.locator(".verify-note").inner_text() if last.locator(".verify-note").count() else None
        page.screenshot(path=str(SHOTS / "ui-smoke-offline-nao-verificado.png"))
        rows["sem_internet"] = offline
        page.close()
        server.stop()
        browser.close()

    out = ROOT / "docs" / "response-quality-ui-smoke-evidence.json"
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print("evidência:", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve(sys.argv[2], int(sys.argv[3]))
    else:
        main()

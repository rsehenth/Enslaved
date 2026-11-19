"""Exemplos exibidos nas configurações, executando o gerador real de JavaScript."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = "abliterant/qwen3.5-rana:9b-q6_K"


@pytest.mark.parametrize("lang", ["pt-BR", "en-US"])
@pytest.mark.parametrize("supports_tools", [True, False])
def test_opencode_examples(lang, supports_tools):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js indisponível")
    script = r"""
const fs = require('fs');
const source = fs.readFileSync('frontend/js/app.js', 'utf8');
const fn = source.slice(source.indexOf('function renderLocalApiExamples()'), source.indexOf('\nasync function loadLocalApi()'));
const catalog = JSON.parse(fs.readFileSync(`frontend/i18n/${process.argv[1]}.json`, 'utf8'));
const localApiStatus = {endpoint: 'http://127.0.0.1:8000/v1', model: 'abliterant/qwen3.5-rana:9b-q6_K', num_ctx: 16384, num_predict: 2048, model_supports_tools: process.argv[2] === 'true', api_key: 'secret-must-not-appear'};
const els = {localapiExamples: {textContent: ''}};
const tr = key => catalog[key];
eval(fn + '\nrenderLocalApiExamples();');
process.stdout.write(els.localapiExamples.textContent);
"""
    output = subprocess.run(
        [node, "-e", script, lang, str(supports_tools).lower()],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    decoder = json.JSONDecoder()
    configs = []
    for line in output.splitlines():
        if line == "{":
            start = output.index(line + "\n", 0 if not configs else configs[-1][1])
            obj, length = decoder.raw_decode(output[start:])
            configs.append((obj, start + length))
    assert len(configs) == 2
    v2, v1 = (item[0] for item in configs)
    assert "secret-must-not-appear" not in output
    for config, provider_key, options_key in ((v2, "providers", "settings"), (v1, "provider", "options")):
        provider = config[provider_key]["enslaved"]
        assert provider[options_key] == {"baseURL": "http://127.0.0.1:8000/v1", "apiKey": "{env:ENSLAVED_API_KEY}"}
        assert list(provider["models"]) == [MODEL]
        assert provider["models"][MODEL]["limit"] == {"context": 16384, "output": 2048}
    assert v2["providers"]["enslaved"]["package"] == "@opencode/ai/providers/openai-compatible"
    assert v2["providers"]["enslaved"]["models"][MODEL]["capabilities"]["tools"] is supports_tools
    assert v1["provider"]["enslaved"]["npm"] == "@ai-sdk/openai-compatible"
    assert v1["provider"]["enslaved"]["models"][MODEL]["tool_call"] is supports_tools
    assert "2.2.5" in output and "16K" in output and "MCP" in output

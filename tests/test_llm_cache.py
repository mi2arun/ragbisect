import json
import http.client
import io
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

import ragbisect.llm as llm_mod
from ragbisect.llm import LLM, LLMError, Usage, parse_model


def test_parse_model():
    assert parse_model("openai:gpt-5.4-mini") == ("openai", "gpt-5.4-mini")
    assert parse_model("claude-haiku-4-5-20251001") == ("anthropic", "claude-haiku-4-5-20251001")
    assert parse_model("gpt-4.1-mini") == ("openai", "gpt-4.1-mini")


def test_usage_cost_known_and_unknown():
    u = Usage()
    u.add("gpt-5-mini", 1_000_000, 0)
    assert u.cost_usd() == pytest.approx(0.25)
    u.add("mystery-model", 10, 10)
    assert u.cost_usd() is None
    assert "cost: unknown" in u.summary()


def test_complete_hits_cache_second_time(tmp_path: Path, monkeypatch):
    calls = []

    def fake_post(self, url, headers, body):
        calls.append(body)
        return {"choices": [{"message": {"content": json.dumps({"ok": True})}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

    monkeypatch.setattr(LLM, "_post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    llm = LLM(model="openai:gpt-test", cache_path=tmp_path / "c.sqlite")
    assert llm.complete_json("hi") == {"ok": True}
    assert llm.complete_json("hi") == {"ok": True}
    assert len(calls) == 1
    assert llm.usage.calls == 1 and llm.usage.cache_hits == 1
    assert llm.usage.prompt_tokens == 5

    # A fresh instance sharing the cache file also hits.
    llm2 = LLM(model="openai:gpt-test", cache_path=tmp_path / "c.sqlite")
    llm2.complete_json("hi")
    assert len(calls) == 1


def test_embed_caches_per_text(tmp_path: Path, monkeypatch):
    seen = []

    def fake_post(self, url, headers, body):
        seen.append(list(body["input"]))
        return {"data": [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(body["input"])], "usage": {"prompt_tokens": 3}}

    monkeypatch.setattr(LLM, "_post", fake_post)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    llm = LLM(cache_path=tmp_path / "c.sqlite")
    assert llm.embed(["a", "bb"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert llm.embed(["bb", "ccc"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert seen == [["a", "bb"], ["ccc"]]


def test_complete_retries_dropped_connection_and_caches_success(tmp_path: Path, monkeypatch):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            calls.append(1)
            if len(calls) == 1:
                self.close_connection = True
                return
            payload = json.dumps({"choices": [{"message": {"content": "answer"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    delays = []
    monkeypatch.setattr(llm_mod.time, "sleep", delays.append)
    try:
        llm = LLM(model="openai:gpt-test", cache_path=tmp_path / "c.sqlite")
        assert llm.complete("hi") == "answer"
        assert llm.complete("hi") == "answer"
        assert len(calls) == 2
        assert delays == [1.0]
        assert llm.usage.calls == 1 and llm.usage.cache_hits == 1
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("error", [http.client.RemoteDisconnected("closed"), ConnectionResetError("reset"),
                                   urllib.error.URLError("unreachable"), TimeoutError("timeout")])
def test_transport_failure_exhausts_retries_without_caching(tmp_path: Path, monkeypatch, error):
    calls = []
    delays = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise error

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fail)
    monkeypatch.setattr(llm_mod.time, "sleep", delays.append)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    llm = LLM(model="openai:gpt-test", cache_path=tmp_path / "c.sqlite")
    for expected_calls in (6, 12):
        with pytest.raises(LLMError, match="network error"):
            llm.complete("hi")
        assert len(calls) == expected_calls
    assert delays == [1.0, 2.0, 4.0, 8.0, 16.0] * 2
    assert llm.usage.calls == 0 and llm.usage.cache_hits == 0


@pytest.mark.parametrize("status,expected_calls", [(400, 1), (503, 6)])
def test_http_error_retry_policy_is_unchanged(tmp_path: Path, monkeypatch, status, expected_calls):
    calls = []
    delays = []

    def fail(req, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError(req.full_url, status, "error", {}, io.BytesIO(b"failure"))

    monkeypatch.setattr(llm_mod.urllib.request, "urlopen", fail)
    monkeypatch.setattr(llm_mod.time, "sleep", delays.append)
    llm = LLM(cache_path=tmp_path / "c.sqlite")
    with pytest.raises(LLMError, match=f"HTTP {status}"):
        llm._post("http://localhost/", {}, {})
    assert len(calls) == expected_calls
    assert delays == ([1.0, 2.0, 4.0, 8.0, 16.0] if status == 503 else [])

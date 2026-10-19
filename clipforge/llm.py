"""OpenAI-compatible chat client + optional management of a local llama.cpp server."""
from __future__ import annotations

import json
import logging
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import urlparse

import requests

from .config import LLMCfg

log = logging.getLogger(__name__)


def parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"No JSON object in LLM response: {text[:300]!r}")
    return json.loads(text[start:end + 1])


class LLMClient:
    def __init__(self, cfg: LLMCfg):
        self.cfg = cfg
        self.url = cfg.base_url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        self._schema_supported = cfg.json_schema

    def chat_json(self, system: str, user: str, schema: Optional[dict] = None, retries: int = 2) -> dict:
        last_err: Exception = RuntimeError("no attempts")
        for attempt in range(retries + 1):
            body = {
                "model": self.cfg.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": self.cfg.temperature,
                "max_tokens": self.cfg.max_tokens,
            }
            if schema and self._schema_supported:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "strict": True, "schema": schema},
                }
            t0 = time.time()
            r = requests.post(self.url, json=body, headers=self.headers, timeout=self.cfg.timeout)
            if r.status_code == 400 and "response_format" in body:
                log.warning("Server rejected json_schema, retrying without it: %s", r.text[:200])
                self._schema_supported = False
                continue
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"].get("content") or ""
            usage = data.get("usage") or {}
            log.info("  LLM: %s prompt / %s completion tokens, %.0f s",
                     usage.get("prompt_tokens", "?"), usage.get("completion_tokens", "?"), time.time() - t0)
            try:
                return parse_json(content)
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                log.warning("Bad JSON from LLM (attempt %d): %s", attempt + 1, e)
        raise RuntimeError(f"LLM did not return valid JSON: {last_err}")


def server_command(cfg: LLMCfg) -> Optional[Union[str, List[str]]]:
    """Command that starts the local LLM server, or None when an external server is used."""
    if cfg.server_cmd:
        return cfg.server_cmd if sys.platform == "win32" else shlex.split(cfg.server_cmd)
    if not cfg.server_exe:
        return None
    port = urlparse(cfg.base_url).port or 8080
    cmd = [cfg.server_exe, "-m", cfg.model_path, "-ngl", str(cfg.gpu_layers), "-c", str(cfg.ctx_size),
           "--host", "127.0.0.1", "--port", str(port)]
    if cfg.n_cpu_moe > 0:
        cmd += ["--n-cpu-moe", str(cfg.n_cpu_moe)]
    return cmd + shlex.split(cfg.extra_args, posix=sys.platform != "win32")


class LocalServer:
    """Starts `server_cmd` (e.g. llama-server.exe) for the duration of a `with` block."""

    def __init__(self, cfg: LLMCfg, log_path: Path):
        self.cfg = cfg
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen] = None
        root = re.sub(r"/v1/?$", "", cfg.base_url.rstrip("/"))
        self.health_url = root + "/health"

    def _healthy(self) -> bool:
        try:
            return requests.get(self.health_url, timeout=3).status_code == 200
        except requests.RequestException:
            return False

    def __enter__(self) -> "LocalServer":
        cmd = server_command(self.cfg)
        if cmd is None:
            return self
        if self._healthy():
            log.info("LLM server already running at %s", self.health_url)
            return self
        log.info("Starting LLM server (log: %s)", self.log_path)
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window popping up from the GUI
        self.proc = subprocess.Popen(cmd, stdout=self._log, stderr=subprocess.STDOUT, creationflags=flags)
        deadline = time.time() + self.cfg.server_ready_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"LLM server exited with code {self.proc.returncode}, see {self.log_path}")
            if self._healthy():
                log.info("LLM server is ready")
                return self
            time.sleep(2)
        self.__exit__(None, None, None)
        raise RuntimeError(f"LLM server not ready after {self.cfg.server_ready_timeout} s, see {self.log_path}")

    def __exit__(self, *exc) -> None:
        if self.proc is None:
            return
        log.info("Stopping LLM server")
        self.proc.terminate()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None
        self._log.close()

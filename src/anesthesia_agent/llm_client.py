"""Minimal OpenAI-compatible chat client (no SDK dependency).

Extracted from ``scripts/run_diplan_llm_agent.py`` so the planning evaluator and any
future agent can share one transport against a served endpoint (vLLM / Ollama / hosted).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    api_base: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen"
    temperature: float = 0.1
    max_tokens: int = 128
    timeout_s: int = 30
    retries: int = 2

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMConfig":
        return cls(
            api_base=str(cfg.get("llm_api_base", cls.api_base)),
            api_key=str(cfg.get("llm_api_key", cls.api_key)),
            model=str(cfg.get("llm_model", cls.model)),
            temperature=float(cfg.get("llm_temperature", cls.temperature)),
            max_tokens=int(cfg.get("llm_max_tokens", cls.max_tokens)),
            timeout_s=int(cfg.get("llm_timeout_s", cls.timeout_s)),
            retries=int(cfg.get("llm_retries", cls.retries)),
        )


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.calls = 0
        self.errors = 0

    def _post(self, system_prompt: str, user_prompt: str) -> str:
        url = self.cfg.api_base.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.cfg.api_key}"}
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.cfg.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        obj = json.loads(body)
        return str(obj["choices"][0]["message"]["content"]).strip()

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """Call the endpoint with retries. Raises LLMError if all attempts fail."""
        last_err = ""
        for _ in range(max(1, self.cfg.retries)):
            self.calls += 1
            try:
                return self._post(system_prompt, user_prompt)
            except urllib.error.HTTPError as e:  # noqa: PERF203
                last_err = f"http_{e.code}"
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
            self.errors += 1
            time.sleep(0.2)
        raise LLMError(last_err)

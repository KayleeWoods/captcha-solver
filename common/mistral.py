"""Vision model key pool (B.AI / OpenAI compatible) — async connection pool + auto-failover.

Uses httpx with HTTP Keep-Alive connection pooling to eliminate TLS handshake overhead.
Shared by reCAPTCHA and hCaptcha image solvers.
"""
import asyncio
import itertools
import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

# B.AI OpenAI-compatible endpoint & model
_ENDPOINT = os.getenv("VISION_API_ENDPOINT", "https://api.b.ai/v1/chat/completions")
_DEFAULT_MODEL = os.getenv("VISION_API_MODEL", "glm-5.3-flash")
_ROTATE_STATUS = {400, 401, 403, 429}
_COOLDOWN_S = 60  # wall-clock park duration for a failed key
_PER_KEY_TIMEOUT = 5.0  # fast failover to prevent stalling


class KeyPool:
    def __init__(self, keyfile: str = None, model: str = _DEFAULT_MODEL,
                 start_index: int = 0):
        if not keyfile:
            p_bai = Path(__file__).parent / "bai_keys.txt"
            p_api = Path(__file__).parent / "apikey.txt"
            keyfile = str(p_bai if p_bai.exists() else p_api)

        keys = [k.strip() for k in Path(keyfile).read_text().splitlines() if k.strip()]
        seen = set()
        self.keys = [k for k in keys if not (k in seen or seen.add(k))]
        if not self.keys:
            raise ValueError(f"no keys in {keyfile}")
        self.model = model
        self._lock = threading.Lock()
        n = len(self.keys)
        self._cursor = itertools.cycle(
            self.keys[start_index % n:] + self.keys[:start_index % n])
        self._dead: dict[str, float] = {}

        # Connection pooling with keep-alive
        limits = httpx.Limits(max_keepalive_connections=20, max_connections=40, keepalive_expiry=60)
        timeout = httpx.Timeout(_PER_KEY_TIMEOUT, connect=3.0)
        self._sync_client = httpx.Client(limits=limits, timeout=timeout)
        self._async_client = httpx.AsyncClient(limits=limits, timeout=timeout)

    def _next_live_key(self) -> str:
        with self._lock:
            now = time.monotonic()
            for _ in range(len(self.keys)):
                k = next(self._cursor)
                if self._dead.get(k, 0.0) <= now:
                    return k
            self._dead.clear()
            return next(self._cursor)

    def _park(self, key: str, cooldown: float = _COOLDOWN_S):
        with self._lock:
            self._dead[key] = time.monotonic() + cooldown

    def _format_payload(self, image_b64: str, prompt: str, max_tokens: int) -> dict:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]}],
            "max_tokens": max(max_tokens, 60),
            "temperature": 0,
        }

    def _extract_text(self, data: dict) -> str:
        try:
            msg = data["choices"][0]["message"]
            txt = msg.get("content") or ""
            if isinstance(txt, list):
                txt = " ".join(p.get("text", "") for p in txt if isinstance(p, dict))
            if not txt.strip() and msg.get("reasoning_content"):
                txt = msg["reasoning_content"]
            return txt.strip().lower()
        except Exception:
            return ""

    def _call(self, image_b64: str, prompt: str,
              max_keys: int, timeout: int, max_tokens: int) -> str:
        """Sync call via persistent keep-alive connection pool."""
        body = self._format_payload(image_b64, prompt, max_tokens)
        last_err = None
        for _ in range(min(max_keys, len(self.keys))):
            key = self._next_live_key()
            try:
                r = self._sync_client.post(
                    _ENDPOINT, json=body,
                    headers={"Authorization": f"Bearer {key}"}
                )
                if r.status_code in _ROTATE_STATUS:
                    self._park(key)
                    continue
                if r.status_code != 200:
                    continue
                return self._extract_text(r.json())
            except Exception as e:
                last_err = str(e).splitlines()[0]
                self._park(key)
                continue
        log.warning("_call failed for %r after key rotation: %s", prompt[:60], last_err)
        return ""

    async def _call_async(self, image_b64: str, prompt: str,
                          max_keys: int, timeout: int, max_tokens: int) -> str:
        """Async call via persistent keep-alive connection pool."""
        body = self._format_payload(image_b64, prompt, max_tokens)
        last_err = None
        for _ in range(min(max_keys, len(self.keys))):
            key = self._next_live_key()
            try:
                r = await self._async_client.post(
                    _ENDPOINT, json=body,
                    headers={"Authorization": f"Bearer {key}"}
                )
                if r.status_code in _ROTATE_STATUS:
                    self._park(key)
                    continue
                if r.status_code != 200:
                    continue
                return self._extract_text(r.json())
            except Exception as e:
                last_err = str(e).splitlines()[0]
                self._park(key)
                continue
        log.warning("_call_async failed for %r after key rotation: %s", prompt[:60], last_err)
        return ""

    def _classify_with_prompt(self, image_b64: str, prompt: str,
                               max_keys: int, timeout: int) -> bool:
        res = self._call(image_b64, prompt, max_keys, timeout, 60)
        return res.startswith("yes") or " yes" in res or "is yes" in res or "red" in res

    def classify(self, image_b64: str, target: str,
                 max_keys: int = 4, timeout: int = 15) -> bool:
        prompt = (f'Does this image clearly contain a {target} (or a visible part '
                  f'of one)? Answer ONLY "yes" or "no".')
        return self._classify_with_prompt(image_b64, prompt, max_keys, timeout)

    async def classify_async(self, image_b64: str, target: str,
                             max_keys: int = 4, timeout: int = 15) -> bool:
        prompt = (f'Does this image clearly contain a {target} (or a visible part '
                  f'of one)? Answer ONLY "yes" or "no".')
        res = await self._call_async(image_b64, prompt, max_keys, timeout, 60)
        return res.startswith("yes") or " yes" in res or "is yes" in res or "red" in res

    def ask(self, image_b64: str, prompt: str,
            max_keys: int = 4, timeout: int = 20, max_tokens: int = 256) -> str:
        return self._call(image_b64, prompt, max_keys, timeout, max_tokens)

    async def ask_async(self, image_b64: str, prompt: str,
                        max_keys: int = 4, timeout: int = 20, max_tokens: int = 256) -> str:
        return await self._call_async(image_b64, prompt, max_keys, timeout, max_tokens)

    def close(self):
        try:
            self._sync_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    pool = KeyPool()
    print("keys loaded:", len(pool.keys))
    red = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    t0 = time.time()
    print("classify(red, 'red square') ->", pool.classify(red, "red square"), f"({round(time.time()-t0, 2)}s)")
    pool.close()

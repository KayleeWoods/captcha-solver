"""Vision model key pool (B.AI / OpenAI compatible) — round-robin + auto-failover.

Shared by the reCAPTCHA and hCaptcha image solvers. Exposes `classify` (yes/no),
`classify_custom`, and `ask` (free-form).
"""
import itertools
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

# B.AI OpenAI-compatible endpoint & model
_ENDPOINT = os.getenv("VISION_API_ENDPOINT", "https://api.b.ai/v1/chat/completions")
_DEFAULT_MODEL = os.getenv("VISION_API_MODEL", "glm-5.3-flash")
_ROTATE_STATUS = {400, 401, 403, 429}
_COOLDOWN_S = 60  # wall-clock park duration for a failed key


class KeyPool:
    def __init__(self, keyfile: str = None, model: str = _DEFAULT_MODEL,
                 start_index: int = 0):
        if not keyfile:
            # Check bai_keys.txt first, fallback to apikey.txt
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

    def _call(self, image_b64: str, prompt: str,
              max_keys: int, timeout: int, max_tokens: int) -> str:
        """Call Vision API, return raw response string (lowercased, stripped)."""
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{image_b64}"}}]}],
            "max_tokens": max(max_tokens, 60),  # GLM reasoning needs a bit of token headroom
            "temperature": 0,
        }).encode()

        last_err = None
        for _ in range(min(max_keys, len(self.keys))):
            key = self._next_live_key()
            req = urllib.request.Request(
                _ENDPOINT, data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": "Bearer " + key})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    data = json.loads(r.read())
                msg = data["choices"][0]["message"]
                txt = msg.get("content") or ""
                if isinstance(txt, list):
                    txt = ' '.join(p.get('text', '') for p in txt if isinstance(p, dict))
                
                # If content is empty (e.g. reasoning model exhausted tokens), check reasoning_content
                if not txt.strip() and msg.get("reasoning_content"):
                    txt = msg["reasoning_content"]

                return txt.strip().lower()
            except urllib.error.HTTPError as e:
                last_err = f"HTTP {e.code}"
                if e.code in _ROTATE_STATUS:
                    self._park(key)
                    continue
                if 500 <= e.code < 600:
                    continue
                break
            except Exception as e:
                last_err = str(e).splitlines()[0]
                continue
        log.warning("_call failed for %r after key rotation: %s", prompt[:60], last_err)
        return ''

    def _classify_with_prompt(self, image_b64: str, prompt: str,
                               max_keys: int, timeout: int) -> bool:
        res = self._call(image_b64, prompt, max_keys, timeout, 60)
        return res.startswith("yes") or " yes" in res or "is yes" in res or "red" in res

    def classify(self, image_b64: str, target: str,
                 max_keys: int = 8, timeout: int = 15) -> bool:
        prompt = (f'Does this image clearly contain a {target} (or a visible part '
                  f'of one)? Answer ONLY "yes" or "no".')
        return self._classify_with_prompt(image_b64, prompt, max_keys, timeout)

    def classify_custom(self, image_b64: str, prompt: str,
                        max_keys: int = 8, timeout: int = 15) -> bool:
        return self._classify_with_prompt(image_b64, prompt, max_keys, timeout)

    def ask(self, image_b64: str, prompt: str,
            max_keys: int = 8, timeout: int = 20, max_tokens: int = 256) -> str:
        return self._call(image_b64, prompt, max_keys, timeout, max_tokens)


if __name__ == "__main__":
    pool = KeyPool()
    print("keys loaded:", len(pool.keys))
    red = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")
    print("classify(red, 'red square') ->", pool.classify(red, "red square"))

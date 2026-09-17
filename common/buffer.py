"""Turnstile Token Pre-harvester & In-Memory Buffer.

Maintains a pool of pre-solved Turnstile tokens for designated high-traffic
sitekeys. Eliminates solve latency (0.01s instant response) for farm workloads.
Tokens have a 300s TTL; tokens older than 240s are automatically purged.
"""
import asyncio
import logging
import os
import time
from dataclasses import dataclass

log = logging.getLogger("captcha-buffer")

# Default targets to auto-preharvest if enabled: format sitekey|url,sitekey|url
_DEFAULT_TARGETS = os.getenv(
    "PREHARVEST_TARGETS",
    "0x4AAAAAAAhr9JGVDZbrZOo0|https://accounts.x.ai/sign-up,"
    "0x4AAAAAADKhTSXIozuHjOoF|https://chat.b.ai/chat"
)
_BUFFER_CAP = int(os.getenv("PREHARVEST_BUFFER_CAP", "2"))
_MAX_TOKEN_AGE = int(os.getenv("PREHARVEST_MAX_AGE", "240"))  # keep tokens fresh (>60s remaining)


@dataclass
class CachedToken:
    token: str
    harvested_at: float
    expires_in: int

    @property
    def is_valid(self) -> bool:
        return (time.monotonic() - self.harvested_at) < _MAX_TOKEN_AGE


class TokenBuffer:
    def __init__(self, capacity: int = _BUFFER_CAP):
        self.capacity = capacity
        self._pools: dict[tuple[str, str], list[CachedToken]] = {}
        self._lock = asyncio.Lock()
        self._harvest_tasks: set = set()
        self._worker_task = None
        self._running = False

    def _parse_targets(self) -> list[tuple[str, str]]:
        targets = []
        for entry in _DEFAULT_TARGETS.split(","):
            entry = entry.strip()
            if "|" in entry:
                sk, url = entry.split("|", 1)
                targets.append((sk.strip(), url.strip()))
        return targets

    async def get_token(self, sitekey: str, url: str) -> dict | None:
        """Attempt to pop a valid pre-harvested token from the buffer."""
        key = (sitekey, url)
        async with self._lock:
            pool = self._pools.get(key, [])
            # Discard expired
            valid_pool = [t for t in pool if t.is_valid]
            self._pools[key] = valid_pool
            if valid_pool:
                cached = valid_pool.pop(0)
                age = round(time.monotonic() - cached.harvested_at, 1)
                remaining = max(300 - int(age), 60)
                log.info("Buffer HIT for sitekey=%s (age=%.1fs, remaining=%ds)", sitekey[:12], age, remaining)
                # Trigger async replenishment in the background
                self._trigger_replenish(sitekey, url)
                return {
                    "token": cached.token,
                    "expires_in": remaining,
                    "elapsed": 0.01,
                    "method": "pre-harvested",
                    "buffer_hit": True,
                }
        return None

    def _trigger_replenish(self, sitekey: str, url: str):
        if not self._running:
            return
        key = (sitekey, url)
        # Avoid duplicate pending replenishment
        if key in self._harvest_tasks:
            return

        async def _run():
            self._harvest_tasks.add(key)
            try:
                await self._harvest_one(sitekey, url)
            finally:
                self._harvest_tasks.discard(key)

        asyncio.create_task(_run())

    async def _harvest_one(self, sitekey: str, url: str):
        from turnstile.solve import solve_turnstile
        try:
            log.info("Pre-harvesting token for sitekey=%s url=%s...", sitekey[:12], url)
            res = await solve_turnstile(sitekey, url)
            tok = res.get("token")
            if tok and isinstance(tok, str) and len(tok) > 50:
                async with self._lock:
                    key = (sitekey, url)
                    if key not in self._pools:
                        self._pools[key] = []
                    if len(self._pools[key]) < self.capacity:
                        self._pools[key].append(CachedToken(tok, time.monotonic(), res.get("expires_in", 300)))
                        log.info("Buffered new token for sitekey=%s (pool size=%d)", sitekey[:12], len(self._pools[key]))
        except Exception as e:
            log.warning("Pre-harvest failed for %s: %s", sitekey[:12], e)

    async def start(self):
        """Start the background maintenance worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._loop())
        log.info("Token buffer worker started (capacity=%d per target)", self.capacity)

    async def stop(self):
        """Stop background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        # Initial small delay to let server & warm browser boot fully
        await asyncio.sleep(5)
        while self._running:
            try:
                targets = self._parse_targets()
                for sitekey, url in targets:
                    if not self._running:
                        break
                    key = (sitekey, url)
                    async with self._lock:
                        pool = self._pools.get(key, [])
                        valid = [t for t in pool if t.is_valid]
                        self._pools[key] = valid
                        needs = self.capacity - len(valid)

                    if needs > 0 and key not in self._harvest_tasks:
                        await self._harvest_one(sitekey, url)
                        await asyncio.sleep(2)  # pace out solves

                # Maintenance sleep
                await asyncio.sleep(20)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("Token buffer worker exception: %s", e)
                await asyncio.sleep(10)


token_buffer = TokenBuffer()

import logging
import time

import requests

log = logging.getLogger(__name__)


class RateLimitedClient:
    """Spaces requests to stay under a provider's per-minute cap and backs off on 429/5xx."""

    def __init__(self, base_url="", calls_per_minute=60, headers=None, params=None, retries=5):
        self.base_url = base_url.rstrip("/")
        self.min_interval = 60.0 / calls_per_minute
        self.default_params = params or {}
        self.session = requests.Session()
        self.session.headers.update(headers or {})
        self._last = 0.0
        self.retries = retries

    def _request(self, method, path, **kwargs):
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        for attempt in range(self.retries):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            resp = self.session.request(method, url, timeout=60, **kwargs)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After", "")
                # Some APIs send Retry-After: 0 while still throttling, so never retry instantly.
                delay = max(float(retry_after), 5) if retry_after.isdigit() else 15 * (attempt + 1)
                log.warning("%s from %s, retrying in %.0fs", resp.status_code, url.split("?")[0], delay)
                time.sleep(delay)
                continue
            break
        resp.raise_for_status()
        return resp

    def get(self, path, params=None):
        return self._request("GET", path, params={**self.default_params, **(params or {})}).json()

    def get_text(self, path, params=None):
        return self._request("GET", path, params={**self.default_params, **(params or {})}).text

    def post(self, path, payload):
        return self._request("POST", path, json=payload).json()

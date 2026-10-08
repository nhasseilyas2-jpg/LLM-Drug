"""Minimal HTTP client for the lab server (``npm start``, default http://127.0.0.1:4173).

The server only accepts loopback Host headers, so point it at 127.0.0.1/localhost.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional


class LabError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status


class LabClient:
    def __init__(self, base_url: str = "http://127.0.0.1:4173", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"content-type": "application/json"} if data is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as res:
                return json.loads(res.read().decode("utf-8") or "null")
        except urllib.error.HTTPError as err:
            try:
                msg = json.loads(err.read().decode("utf-8")).get("error", err.reason)
            except Exception:
                msg = err.reason
            raise LabError(err.code, str(msg)) from None

    # ---- read-only endpoints
    def status(self) -> dict:
        return self._request("GET", "/api/status")

    def catalog(self) -> dict:
        return self._request("GET", "/api/catalog")

    def techniques(self) -> list[dict]:
        return self.catalog()["techniques"]

    def models(self) -> dict:
        return self._request("GET", "/api/models")

    def history(self, limit: int = 50) -> list[dict]:
        return self._request("GET", f"/api/history?limit={int(limit)}")["items"]

    def record(self, record_id: str) -> dict:
        return self._request("GET", f"/api/history/{record_id}")

    # ---- jobs
    def submit(self, kind: str, **params: Any) -> dict:
        if kind not in ("run", "dose-response", "agent"):
            raise ValueError("kind must be run, dose-response or agent")
        return self._request("POST", f"/api/{kind}", params)

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/api/jobs/{job_id}")

    def cancel(self, job_id: str) -> dict:
        return self._request("POST", f"/api/jobs/{job_id}/cancel", {})

    def wait(self, job_id: str, poll: float = 1.0, timeout: Optional[float] = None,
             on_progress: Optional[Callable[[dict], None]] = None) -> dict:
        """Block until the job finishes; return its record or raise LabError."""
        start = time.monotonic()
        while True:
            job = self.job(job_id)
            if on_progress:
                on_progress(job.get("progress") or {})
            if job["status"] == "done":
                return job["result"]
            if job["status"] in ("error", "cancelled"):
                raise LabError(500 if job["status"] == "error" else 499, job.get("error") or job["status"])
            if timeout is not None and time.monotonic() - start > timeout:
                self.cancel(job_id)
                raise TimeoutError(f"job {job_id} did not finish in {timeout}s")
            time.sleep(poll)

    def run(self, kind: str = "run", wait: bool = True, **params: Any) -> dict:
        """Submit and (by default) wait. Parameters are the JSON body fields, e.g.
        ``backend="llamacpp", modelId=..., techniqueId="delirium", doseMg=200, prompt="..."``."""
        job = self.submit(kind, **params)
        return self.wait(job["id"]) if wait else job

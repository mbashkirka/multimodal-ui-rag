from __future__ import annotations

import json
import select
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorkerError(RuntimeError):
    pass


class WorkerTimeoutError(WorkerError):
    pass


class WorkerRemoteError(WorkerError):
    """The worker was healthy but rejected the specific request."""


class PersistentJsonWorker:
    """Long-lived native-stack worker with timeout and one-shot transport recovery."""

    def __init__(
        self,
        *,
        kind: str,
        args: list[str] | None = None,
        request_timeout_seconds: float = 30.0,
        startup_timeout_seconds: float = 300.0,
        auto_restart: bool = True,
    ) -> None:
        self.kind = kind
        self.args = args or []
        self.request_timeout_seconds = float(request_timeout_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.auto_restart = bool(auto_restart)
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._ever_started = False
        self._restart_count = 0
        self._timeout_count = 0
        self._failure_count = 0
        self._request_count = 0
        self._last_error: str | None = None

    @staticmethod
    def _readline_with_timeout(stream: Any, timeout_seconds: float) -> str:
        ready, _, _ = select.select([stream], [], [], max(0.0, timeout_seconds))
        if not ready:
            raise WorkerTimeoutError(f"worker response timed out after {timeout_seconds:.3f}s")
        return stream.readline()

    def _spawn_locked(self, *, count_restart: bool) -> None:
        if self.is_alive():
            return
        cmd = [sys.executable, "-m", "ui_rag.api.worker_main", self.kind, *self.args]
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._proc = proc
        assert proc.stdout is not None
        try:
            line = self._readline_with_timeout(proc.stdout, self.startup_timeout_seconds)
        except WorkerTimeoutError as exc:
            self._timeout_count += 1
            self._last_error = f"startup timeout: {exc}"
            self._close_locked(force=True)
            raise WorkerTimeoutError(f"{self.kind} worker startup timed out") from exc
        if not line:
            code = proc.poll()
            self._failure_count += 1
            self._last_error = f"startup exit code={code}"
            self._close_locked(force=True)
            raise WorkerError(f"{self.kind} worker exited during startup code={code}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            self._failure_count += 1
            self._last_error = "invalid startup JSON"
            self._close_locked(force=True)
            raise WorkerError(f"{self.kind} worker returned invalid startup JSON: {line!r}") from exc
        if payload.get("status") != "ready" or payload.get("worker") != self.kind:
            self._failure_count += 1
            self._last_error = f"startup payload={payload!r}"
            self._close_locked(force=True)
            raise WorkerError(f"{self.kind} worker failed startup: {payload}")
        if count_restart:
            self._restart_count += 1
        self._ever_started = True
        self._last_error = None

    def start(self) -> None:
        with self._lock:
            self._spawn_locked(count_restart=self._ever_started)

    def restart(self) -> None:
        with self._lock:
            self._close_locked(force=True)
            self._spawn_locked(count_restart=True)

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _request_once_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.is_alive():
            raise WorkerError(f"{self.kind} worker is not running")
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        request_id = uuid.uuid4().hex
        message = {"request_id": request_id, **payload}
        try:
            self._proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise WorkerError(f"{self.kind} worker pipe write failed: {exc}") from exc
        try:
            line = self._readline_with_timeout(self._proc.stdout, self.request_timeout_seconds)
        except WorkerTimeoutError as exc:
            self._timeout_count += 1
            raise WorkerTimeoutError(
                f"{self.kind} worker request timed out after {self.request_timeout_seconds:.3f}s"
            ) from exc
        if not line:
            code = self._proc.poll()
            raise WorkerError(f"{self.kind} worker exited during request code={code}")
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            raise WorkerError(f"{self.kind} worker returned invalid JSON: {line!r}") from exc
        if response.get("request_id") != request_id:
            raise WorkerError(f"{self.kind} worker response/request id mismatch")
        if not response.get("ok", False):
            raise WorkerRemoteError(str(response.get("error", f"{self.kind} worker failed")))
        result = response.get("result")
        if not isinstance(result, dict):
            raise WorkerError(f"{self.kind} worker result must be an object")
        return result

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._request_count += 1
            for attempt in range(2):
                try:
                    if not self.is_alive():
                        self._spawn_locked(count_restart=self._ever_started)
                    result = self._request_once_locked(payload)
                    self._last_error = None
                    return result
                except WorkerRemoteError:
                    # Model/input errors are not transport failures; restarting would only repeat them.
                    raise
                except WorkerError as exc:
                    self._failure_count += 1
                    self._last_error = str(exc)
                    self._close_locked(force=True)
                    if not self.auto_restart or attempt == 1:
                        raise
                    self._spawn_locked(count_restart=True)
            raise AssertionError("unreachable")

    def _close_locked(self, *, force: bool = False) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        if not force and proc.stdin is not None:
            try:
                proc.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=5)
                return
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                pass
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def close(self, *, force: bool = False) -> None:
        with self._lock:
            self._close_locked(force=force)

    def terminate_for_diagnostic(self) -> None:
        """Kill the child process so recovery can be exercised without an HTTP fault endpoint."""
        with self._lock:
            self._close_locked(force=True)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "alive": self.is_alive(),
            "pid": self._proc.pid if self.is_alive() and self._proc is not None else None,
            "restarts": self._restart_count,
            "timeouts": self._timeout_count,
            "failures": self._failure_count,
            "requests": self._request_count,
            "last_error": self._last_error,
            "request_timeout_seconds": self.request_timeout_seconds,
        }


@dataclass
class WorkerBundle:
    encoder: PersistentJsonWorker
    faiss: PersistentJsonWorker
    vlm: PersistentJsonWorker

    def start(self) -> None:
        started: list[PersistentJsonWorker] = []
        try:
            for worker in (self.encoder, self.faiss, self.vlm):
                worker.start()
                started.append(worker)
        except Exception:
            for worker in reversed(started):
                worker.close(force=True)
            raise

    def recover_dead(self) -> None:
        for worker in (self.encoder, self.faiss, self.vlm):
            if not worker.is_alive():
                worker.start()

    def close(self) -> None:
        for worker in (self.vlm, self.encoder, self.faiss):
            worker.close()

    def status(self) -> dict[str, bool]:
        return {
            "encoder": self.encoder.is_alive(),
            "faiss": self.faiss.is_alive(),
            "vlm": self.vlm.is_alive(),
        }

    def diagnostics(self) -> dict[str, dict[str, Any]]:
        return {
            "encoder": self.encoder.diagnostics(),
            "faiss": self.faiss.diagnostics(),
            "vlm": self.vlm.diagnostics(),
        }


def build_worker_bundle(
    *,
    base_manifest: Path,
    model_name: str,
    encoder_timeout_seconds: float = 30.0,
    faiss_timeout_seconds: float = 10.0,
    vlm_timeout_seconds: float = 60.0,
    startup_timeout_seconds: float = 300.0,
) -> WorkerBundle:
    return WorkerBundle(
        encoder=PersistentJsonWorker(
            kind="encoder",
            args=["--manifest", str(base_manifest)],
            request_timeout_seconds=encoder_timeout_seconds,
            startup_timeout_seconds=startup_timeout_seconds,
        ),
        faiss=PersistentJsonWorker(
            kind="faiss",
            args=["--manifest", str(base_manifest)],
            request_timeout_seconds=faiss_timeout_seconds,
            startup_timeout_seconds=startup_timeout_seconds,
        ),
        vlm=PersistentJsonWorker(
            kind="vlm",
            args=["--model", model_name],
            request_timeout_seconds=vlm_timeout_seconds,
            startup_timeout_seconds=startup_timeout_seconds,
        ),
    )

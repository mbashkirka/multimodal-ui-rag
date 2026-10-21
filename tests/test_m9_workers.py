from __future__ import annotations

import os
import time

import pytest

from ui_rag.api.workers import (
    PersistentJsonWorker,
    WorkerBundle,
    WorkerRemoteError,
    WorkerTimeoutError,
)


class _FakeWorker:
    def __init__(self, alive: bool):
        self.alive = alive
        self.starts = 0

    def is_alive(self):
        return self.alive

    def start(self):
        self.alive = True
        self.starts += 1

    def close(self, force: bool = False):
        self.alive = False

    def diagnostics(self):
        return {"alive": self.alive, "restarts": self.starts, "timeouts": 0, "failures": 0}


def test_worker_bundle_recovers_only_dead_workers():
    encoder = _FakeWorker(False)
    faiss = _FakeWorker(True)
    vlm = _FakeWorker(True)
    bundle = WorkerBundle(encoder=encoder, faiss=faiss, vlm=vlm)  # type: ignore[arg-type]
    bundle.recover_dead()
    assert encoder.alive and encoder.starts == 1
    assert faiss.starts == 0
    assert vlm.starts == 0


def test_readline_timeout_is_bounded():
    read_fd, write_fd = os.pipe()
    try:
        with os.fdopen(read_fd, "r") as reader:
            started = time.perf_counter()
            with pytest.raises(WorkerTimeoutError):
                PersistentJsonWorker._readline_with_timeout(reader, 0.02)
            assert time.perf_counter() - started < 0.5
    finally:
        os.close(write_fd)


def test_remote_worker_error_is_distinct_from_transport_failure(monkeypatch):
    worker = PersistentJsonWorker(kind="faiss")
    monkeypatch.setattr(worker, "is_alive", lambda: True)

    def fail(_payload):
        raise WorkerRemoteError("bad input")

    monkeypatch.setattr(worker, "_request_once_locked", fail)
    with pytest.raises(WorkerRemoteError, match="bad input"):
        worker.request({"op": "x"})
    assert worker.diagnostics()["failures"] == 0

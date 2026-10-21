from __future__ import annotations

import hashlib
import json
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

from ui_rag import __version__
from ui_rag.api.metrics import MetricsRegistry
from ui_rag.api.settings import APISettings
from ui_rag.api.workers import WorkerBundle, WorkerError, build_worker_bundle
from ui_rag.config import PROJECT_ROOT, RICO_DIR
from ui_rag.reranking.multires import reciprocal_rank_fusion_order

API_VERSION = "v1"


class ServiceBusyError(RuntimeError):
    pass


class ServiceNotReadyError(RuntimeError):
    pass


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _metadata_sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _combined_fingerprint(paths: list[Path]) -> str | None:
    hashes = [_metadata_sha(p) for p in paths]
    if any(x is None for x in hashes):
        return None
    digest = hashlib.sha256()
    for item in hashes:
        assert item is not None
        digest.update(item.encode("ascii"))
    return digest.hexdigest()


def _resolve_artifact(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _sanitize_hit(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(row["rank"]),
        "faiss_id": int(row["faiss_id"]),
        "filename": str(row["filename"]),
        "score": float(row["score"]),
        "base_rank": int(row.get("base_rank", row["rank"])),
        "base_score": float(row.get("base_score", row.get("score", 0.0))),
        "fine_score": float(row.get("fine_score", row.get("score", 0.0))),
    }


def _sanitize_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": str(row["evidence_id"]),
        "rank": int(row["rank"]),
        "faiss_id": int(row["faiss_id"]),
        "filename": str(row["filename"]),
        "retrieval_score": float(row["retrieval_score"]),
    }


class ProductionService:
    """Production facade with persistent isolated workers, recovery, timeouts and metrics.

    M9 keeps torch/SigLIP2, FAISS and MLX in separate long-lived child processes. Worker
    transport failures are recovered once, every request has an explicit native-worker
    timeout, and the API host remains free of all three native ML stacks.
    """

    def __init__(self, settings: APISettings, workers: WorkerBundle | None = None):
        self.settings = settings
        self.metrics = MetricsRegistry()
        self._gate = threading.BoundedSemaphore(settings.max_concurrent_inference)
        self._verify_lock = threading.Lock()
        self._artifacts_verified = False
        self._workers = workers or build_worker_bundle(
            base_manifest=settings.base_manifest,
            model_name=settings.model_name,
            encoder_timeout_seconds=settings.encoder_timeout_seconds,
            faiss_timeout_seconds=settings.faiss_timeout_seconds,
            vlm_timeout_seconds=settings.vlm_timeout_seconds,
            startup_timeout_seconds=settings.worker_startup_timeout_seconds,
        )
        self._started = False
        self._rerank_embeddings: np.ndarray | None = None
        self._rerank_stage_payload: dict[str, Any] | None = None

    def start(self) -> None:
        if self._started:
            return
        self.verify_artifacts_once()
        stage = json.loads(self.settings.rerank_stage.read_text(encoding="utf-8"))
        rerank_path = _resolve_artifact(stage["embedding"]["path"])
        self._rerank_embeddings = np.load(rerank_path, mmap_mode="r")
        self._rerank_stage_payload = stage
        try:
            self._workers.start()
        except Exception:
            self._rerank_embeddings = None
            self._rerank_stage_payload = None
            raise
        self._started = True

    def close(self) -> None:
        self._workers.close()
        self._started = False
        self._rerank_embeddings = None
        self._rerank_stage_payload = None

    def worker_diagnostics(self) -> dict[str, dict[str, Any]]:
        return self._workers.diagnostics()

    def readiness(self) -> dict[str, Any]:
        checks: dict[str, bool] = {
            "base_manifest": self.settings.base_manifest.is_file(),
            "rerank_stage": self.settings.rerank_stage.is_file(),
            "rico_dataset": RICO_DIR.is_dir(),
        }
        if checks["base_manifest"]:
            try:
                manifest = json.loads(self.settings.base_manifest.read_text(encoding="utf-8"))
                checks["faiss_index"] = _resolve_artifact(manifest["index"]["path"]).is_file()
                checks["image_mapping"] = _resolve_artifact(
                    manifest["mapping"]["artifact_path"]
                ).is_file()
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                checks["faiss_index"] = False
                checks["image_mapping"] = False
        else:
            checks["faiss_index"] = False
            checks["image_mapping"] = False
        if checks["rerank_stage"]:
            try:
                stage = json.loads(self.settings.rerank_stage.read_text(encoding="utf-8"))
                checks["rerank_embeddings"] = _resolve_artifact(
                    stage["embedding"]["path"]
                ).is_file()
            except (OSError, KeyError, TypeError, json.JSONDecodeError):
                checks["rerank_embeddings"] = False
        else:
            checks["rerank_embeddings"] = False

        diagnostics = self._workers.diagnostics() if self._started else {}
        ready = all(checks.values()) and (
            not self._started or all(bool(x["alive"]) for x in diagnostics.values())
        )
        return {
            "status": "ready" if ready else "not_ready",
            "api_version": API_VERSION,
            "artifacts": {name: {"exists": exists} for name, exists in checks.items()},
            "workers": {
                name: {
                    "alive": bool(diag["alive"]),
                    "restarts": int(diag["restarts"]),
                    "timeouts": int(diag["timeouts"]),
                }
                for name, diag in diagnostics.items()
            },
        }

    def require_ready(self) -> None:
        if self.readiness()["status"] != "ready":
            raise ServiceNotReadyError("required retrieval artifacts/workers are not ready")

    def model_info(self) -> dict[str, Any]:
        fingerprint = _combined_fingerprint([self.settings.base_manifest, self.settings.rerank_stage])
        return {
            "service_version": __version__,
            "api_version": API_VERSION,
            "retrieval_encoder": "google/siglip2-base-patch16-naflex (p256)",
            "retrieval_index": "FAISS IndexFlatIP",
            "reranker": "SigLIP2 NaFlex p512 + weighted RRF (0.5/0.5, k=60)",
            "vlm_model": self.settings.model_name,
            "artifact_fingerprint": fingerprint,
            "native_isolation": True,
            "serving_mode": "persistent-isolated-workers",
            "evidence_policy": "rank1",
            "max_concurrent_inference": self.settings.max_concurrent_inference,
            "worker_timeouts_seconds": {
                "encoder": self.settings.encoder_timeout_seconds,
                "faiss": self.settings.faiss_timeout_seconds,
                "vlm": self.settings.vlm_timeout_seconds,
            },
        }

    def runtime_versions(self) -> dict[str, str | None]:
        return {
            "ui-rag": __version__,
            "fastapi": _package_version("fastapi"),
            "uvicorn": _package_version("uvicorn"),
            "mlx-vlm": _package_version("mlx-vlm"),
        }

    @staticmethod
    def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        return digest.hexdigest()

    def verify_artifacts_once(self) -> None:
        if self._artifacts_verified:
            return
        with self._verify_lock:
            if self._artifacts_verified:
                return
            readiness = self.readiness()
            if not all(x["exists"] for x in readiness["artifacts"].values()):
                raise ServiceNotReadyError("required retrieval artifacts are missing")
            manifest = json.loads(self.settings.base_manifest.read_text(encoding="utf-8"))
            stage = json.loads(self.settings.rerank_stage.read_text(encoding="utf-8"))
            checks = [
                (_resolve_artifact(manifest["index"]["path"]), manifest["index"]["sha256"], "FAISS index"),
                (
                    _resolve_artifact(manifest["mapping"]["artifact_path"]),
                    manifest["mapping"]["artifact_sha256"],
                    "image mapping",
                ),
                (
                    _resolve_artifact(stage["embedding"]["path"]),
                    stage["embedding"]["sha256"],
                    "p512 reranker embeddings",
                ),
            ]
            for path, expected, label in checks:
                if self._sha256_file(path) != expected:
                    raise ServiceNotReadyError(f"{label} hash mismatch")
            if manifest["mapping"]["artifact_sha256"] != stage["mapping"]["artifact_sha256"]:
                raise ServiceNotReadyError("base/reranker mappings are not aligned")
            self._artifacts_verified = True

    def _ensure_started(self) -> None:
        if not self._started:
            self.start()
        else:
            try:
                self._workers.recover_dead()
            except WorkerError as exc:
                raise ServiceNotReadyError(f"worker recovery failed: {exc}") from exc
        self.require_ready()

    def _acquire(self) -> None:
        if not self._gate.acquire(timeout=self.settings.acquire_timeout_seconds):
            self.metrics.inference_busy()
            raise ServiceBusyError("inference capacity is busy; retry later")
        self.metrics.inference_acquired()

    def _release(self) -> None:
        self.metrics.inference_released()
        self._gate.release()

    def _observe_retrieval(self, timings: dict[str, float]) -> None:
        self.metrics.observe_stage("encode", timings.get("encode_ms"))
        self.metrics.observe_stage("search", timings.get("search_ms"))
        self.metrics.observe_stage("rerank", timings.get("rerank_ms"))
        self.metrics.observe_stage("retrieval_total", timings.get("total_ms"))

    def _retrieve(self, *, query: str, top_k: int) -> dict[str, Any]:
        self._ensure_started()
        if self._rerank_embeddings is None:
            raise ServiceNotReadyError("reranker embeddings are not loaded")

        total_started = time.perf_counter()
        encode_started = time.perf_counter()
        encoded = self._workers.encoder.request({"op": "encode", "query": query})
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        q = np.asarray(encoded["vector"], dtype=np.float32).reshape(-1)

        candidate_k = max(self.settings.candidate_k, top_k)
        search_started = time.perf_counter()
        searched = self._workers.faiss.request(
            {"op": "search", "vector": q.tolist(), "top_k": candidate_k}
        )
        search_ms = (time.perf_counter() - search_started) * 1000.0

        rerank_started = time.perf_counter()
        rows = searched["results"]
        ids = np.asarray([int(x["faiss_id"]) for x in rows], dtype=np.int64)
        fine_scores = np.asarray(self._rerank_embeddings[ids] @ q, dtype=np.float32)
        ordered_ids, fused_scores = reciprocal_rank_fusion_order(
            ids, fine_scores, base_weight=0.5, rrf_constant=60
        )
        by_id = {int(x["faiss_id"]): x for x in rows}
        fine_by_id = {int(fid): float(score) for fid, score in zip(ids, fine_scores)}
        reranked: list[dict[str, Any]] = []
        for rank, (faiss_id, fused) in enumerate(
            zip(ordered_ids[:top_k], fused_scores[:top_k]), start=1
        ):
            row = dict(by_id[int(faiss_id)])
            row["base_rank"] = int(row["rank"])
            row["rank"] = rank
            row["base_score"] = float(row["score"])
            row["score"] = float(fused)
            row["fine_score"] = fine_by_id[int(faiss_id)]
            row["path"] = str(RICO_DIR / row["filename"])
            reranked.append(row)
        rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
        return {
            "results": reranked,
            "timings_ms": {
                "encode_ms": encode_ms,
                "search_ms": search_ms,
                "rerank_ms": rerank_ms,
                "total_ms": (time.perf_counter() - total_started) * 1000.0,
            },
        }

    def search(self, *, query: str, top_k: int) -> dict[str, Any]:
        self.verify_artifacts_once()
        self._acquire()
        try:
            payload = self._retrieve(query=query, top_k=top_k)
        except WorkerError as exc:
            raise ServiceNotReadyError(str(exc)) from exc
        finally:
            self._release()
        timings = payload["timings_ms"]
        self._observe_retrieval(timings)
        return {
            "query": query,
            "results": [_sanitize_hit(x) for x in payload["results"]],
            "latency": {
                "total_ms": float(timings["total_ms"]),
                "encode_ms": float(timings["encode_ms"]),
                "search_ms": float(timings["search_ms"]),
                "rerank_ms": float(timings["rerank_ms"]),
                "generation_ms": None,
            },
        }

    def rag(self, *, query: str, max_tokens: int) -> dict[str, Any]:
        self.verify_artifacts_once()
        self._acquire()
        total_started = time.perf_counter()
        try:
            retrieval = self._retrieve(query=query, top_k=self.settings.evidence_k)
            results = retrieval["results"]
            if not results:
                raise ServiceNotReadyError("retriever returned no evidence")
            evidence: list[dict[str, Any]] = []
            for rank, row in enumerate(results[: self.settings.evidence_k], start=1):
                evidence.append({
                    "evidence_id": f"E{rank}",
                    "rank": rank,
                    "faiss_id": int(row["faiss_id"]),
                    "filename": str(row["filename"]),
                    "path": str(row["path"]),
                    "retrieval_score": float(row["score"]),
                })
            selected = evidence[0]
            generation_started = time.perf_counter()
            generated = self._workers.vlm.request({
                "op": "answer",
                "query": query,
                "evidence": selected,
                "max_tokens": max_tokens,
                "max_image_pixels": self.settings.max_image_pixels,
            })
            generation_ms = (time.perf_counter() - generation_started) * 1000.0
            text = str(generated["text"]).strip()
            if not text:
                raise RuntimeError("VLM returned an empty grounded answer")
            answer = {
                "answer": text,
                "selected_evidence_id": "E1",
                "evidence_ids": ["E1"],
                "observations": [{"evidence_id": "E1", "claim": text}],
                "confidence": None,
                "abstain": False,
            }
        except WorkerError as exc:
            raise ServiceNotReadyError(str(exc)) from exc
        finally:
            self._release()

        retrieval_timings = retrieval["timings_ms"]
        self._observe_retrieval(retrieval_timings)
        self.metrics.observe_stage("generation", generation_ms)
        total_ms = (time.perf_counter() - total_started) * 1000.0
        self.metrics.observe_stage("rag_total", total_ms)
        return {
            "query": query,
            "evidence": [_sanitize_evidence(x) for x in evidence],
            "answer": answer,
            "citations_valid": True,
            "latency": {
                "total_ms": total_ms,
                "encode_ms": float(retrieval_timings["encode_ms"]),
                "search_ms": float(retrieval_timings["search_ms"]),
                "rerank_ms": float(retrieval_timings["rerank_ms"]),
                "generation_ms": generation_ms,
            },
        }

    def record_http_request(self, *, method: str, path: str, status: int, seconds: float) -> None:
        self.metrics.observe_http(method=method, path=path, status=status, seconds=seconds)

    def metrics_text(self) -> str:
        return self.metrics.render(self._workers.diagnostics() if self._started else {})

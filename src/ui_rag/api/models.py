from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    service: str = "ui-rag"
    api_version: str


class ArtifactCheck(StrictModel):
    exists: bool


class WorkerCheck(StrictModel):
    alive: bool
    restarts: int = 0
    timeouts: int = 0


class ReadinessResponse(StrictModel):
    status: Literal["ready", "not_ready"]
    api_version: str
    artifacts: dict[str, ArtifactCheck]
    workers: dict[str, WorkerCheck] = Field(default_factory=dict)


class ModelInfoResponse(StrictModel):
    service_version: str
    api_version: str
    retrieval_encoder: str
    retrieval_index: str
    reranker: str
    vlm_model: str
    artifact_fingerprint: str | None
    native_isolation: bool = True
    serving_mode: str = "persistent-isolated-workers"
    evidence_policy: Literal["rank1"] = "rank1"
    max_concurrent_inference: int
    worker_timeouts_seconds: dict[str, float] = Field(default_factory=dict)


class SearchRequest(StrictModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int = Field(default=4, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SearchHit(StrictModel):
    rank: int
    faiss_id: int
    filename: str
    score: float
    base_rank: int
    base_score: float
    fine_score: float


class LatencyBreakdown(StrictModel):
    total_ms: float
    encode_ms: float | None = None
    search_ms: float | None = None
    rerank_ms: float | None = None
    generation_ms: float | None = None


class SearchResponse(StrictModel):
    request_id: str
    query: str
    results: list[SearchHit]
    latency: LatencyBreakdown


class RagRequest(StrictModel):
    query: str = Field(min_length=1, max_length=2_000)
    max_tokens: int = Field(default=128, ge=16, le=256)

    @field_validator("query")
    @classmethod
    def nonblank_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class EvidencePublic(StrictModel):
    evidence_id: str
    rank: int
    faiss_id: int
    filename: str
    retrieval_score: float


class ObservationPublic(StrictModel):
    evidence_id: str
    claim: str


class AnswerPublic(StrictModel):
    answer: str
    selected_evidence_id: str | None
    evidence_ids: list[str]
    observations: list[ObservationPublic]
    confidence: float | None
    abstain: bool


class RagResponse(StrictModel):
    request_id: str
    query: str
    evidence: list[EvidencePublic]
    answer: AnswerPublic
    citations_valid: bool
    latency: LatencyBreakdown


class ErrorBody(StrictModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(StrictModel):
    error: ErrorBody

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from ui_rag.api.metrics import PROMETHEUS_CONTENT_TYPE
from ui_rag.api.models import (
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    RagRequest,
    RagResponse,
    ReadinessResponse,
    SearchRequest,
    SearchResponse,
)
from ui_rag.api.service import (
    API_VERSION,
    ProductionService,
    ServiceBusyError,
    ServiceNotReadyError,
)
from ui_rag.api.settings import APISettings

logger = logging.getLogger("ui_rag.api")


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _error(request: Request, *, status: int, code: str, message: str) -> JSONResponse:
    payload = ErrorResponse(error=ErrorBody(code=code, message=message, request_id=_request_id(request)))
    return JSONResponse(status_code=status, content=payload.model_dump())


def create_app(settings: APISettings | None = None, service: ProductionService | None = None) -> FastAPI:
    settings = settings or APISettings.from_env()
    service = service or ProductionService(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        starter = getattr(service, "start", None)
        closer = getattr(service, "close", None)
        if callable(starter):
            starter()
        try:
            yield
        finally:
            if callable(closer):
                closer()

    app = FastAPI(
        title="UI-RAG API",
        version=API_VERSION,
        description="Production API for frozen SigLIP2 retrieval + grounded local VLM generation.",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming_request_id = (request.headers.get("X-Request-ID") or "").strip()
        request.state.request_id = (
            incoming_request_id if 0 < len(incoming_request_id) <= 128 else str(uuid.uuid4())
        )
        started = time.perf_counter()
        status = 500
        response = None
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            seconds = time.perf_counter() - started
            recorder = getattr(request.app.state.service, "record_http_request", None)
            metric_route = getattr(request.scope.get("route"), "path", "__unmatched__")
            if callable(recorder):
                recorder(method=request.method, path=metric_route, status=status, seconds=seconds)
            logger.info(
                "request complete",
                extra={
                    "event": "request_complete",
                    "request_id": _request_id(request),
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status,
                    "elapsed_ms": round(seconds * 1000.0, 3),
                },
            )
            if response is not None:
                response.headers["X-Request-ID"] = _request_id(request)
                response.headers["X-Process-Time-Ms"] = f"{seconds * 1000.0:.3f}"

    @app.exception_handler(ServiceBusyError)
    async def busy_handler(request: Request, exc: ServiceBusyError):
        return _error(request, status=429, code="inference_busy", message=str(exc))

    @app.exception_handler(ServiceNotReadyError)
    async def not_ready_handler(request: Request, exc: ServiceNotReadyError):
        return _error(request, status=503, code="not_ready", message=str(exc))

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        return _error(request, status=422, code="validation_error", message=str(exc.errors()))

    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        logger.exception(
            "request failed",
            extra={
                "event": "request_failed",
                "request_id": _request_id(request),
                "error_type": type(exc).__name__,
            },
        )
        return _error(request, status=500, code="internal_error", message="request failed")

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        return HealthResponse(api_version=API_VERSION)

    @app.get("/ready", response_model=ReadinessResponse, tags=["system"])
    def ready(request: Request):
        payload = request.app.state.service.readiness()
        status = 200 if payload["status"] == "ready" else 503
        if status != 200:
            return JSONResponse(status_code=status, content=payload)
        return payload

    @app.get("/metrics", include_in_schema=False, tags=["system"])
    def metrics(request: Request) -> Response:
        renderer = getattr(request.app.state.service, "metrics_text", None)
        text = renderer() if callable(renderer) else "# ui-rag metrics unavailable\n"
        return Response(content=text, media_type=PROMETHEUS_CONTENT_TYPE)

    @app.get("/v1/model-info", response_model=ModelInfoResponse, tags=["system"])
    def model_info(request: Request):
        return request.app.state.service.model_info()

    @app.post("/v1/search", response_model=SearchResponse, tags=["retrieval"])
    def search(body: SearchRequest, request: Request):
        payload = request.app.state.service.search(query=body.query.strip(), top_k=body.top_k)
        return {"request_id": _request_id(request), **payload}

    @app.post("/v1/rag", response_model=RagResponse, tags=["rag"])
    def rag(body: RagRequest, request: Request):
        payload = request.app.state.service.rag(query=body.query.strip(), max_tokens=body.max_tokens)
        return {"request_id": _request_id(request), **payload}

    return app


app = create_app()

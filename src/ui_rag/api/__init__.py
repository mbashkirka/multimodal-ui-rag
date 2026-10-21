"""Production HTTP API for the frozen UI-RAG pipeline."""

from .app import create_app

__all__ = ["create_app"]

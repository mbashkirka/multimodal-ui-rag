"""Compatibility module for M1.3.

The M1.2 in-process Retriever was intentionally removed on macOS ARM because it
loaded both PyTorch/OpenCLIP and FAISS into one Python interpreter. Use
``ui_rag.retrieval.orchestrator.search_isolated`` or the public ``ui-rag search``
command, which keeps those native runtimes in separate child processes.
"""

from ui_rag.retrieval.orchestrator import search_isolated

__all__ = ["search_isolated"]

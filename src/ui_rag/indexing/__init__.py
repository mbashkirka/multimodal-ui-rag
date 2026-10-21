"""FAISS artifact utilities with macOS ARM process isolation.

Do not eagerly import FAISS or PyTorch from this package namespace. M1.3 keeps
those native runtimes in separate child interpreters.
"""

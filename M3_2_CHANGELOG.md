# M3.2 Changelog — macOS pytest native-runtime isolation

## Fixed

- Reworked `tests/test_faiss_contract.py` so FAISS executes in a dedicated child
  Python process instead of being imported into the main pytest interpreter.
- This preserves the same macOS native-runtime isolation contract used by the
  production indexing and benchmark pipeline: Torch/OpenCLIP/Transformers and
  FAISS never coexist in one Python interpreter.
- Added a regression test that prevents a future eager FAISS import during
  pytest collection.

## Why

On Apple Silicon/macOS, PyTorch and `faiss-cpu` can load incompatible OpenMP
runtimes. M3.1 added Torch-backed SigLIP2 tests; pytest imports test modules
before running them, so the old top-level `import faiss` in the FAISS contract
loaded both native stacks into the same interpreter and caused `Abort trap: 6`.

No production retrieval/indexing behavior was changed in M3.2.

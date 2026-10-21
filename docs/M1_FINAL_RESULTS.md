# M1 Final Results — Reproducible Legacy CLIP + FAISS Baseline

This file records the successful reference run performed on macOS Apple Silicon on 2026-08-19.
The large generated `.npy` and `.faiss` artifacts are intentionally not committed to the project archive;
they are reproducibly generated from RICO with `make index-full`.

## Environment

- Python: 3.12.0
- PyTorch: 2.13.0
- OpenCLIP: 3.3.0
- FAISS: 1.11.0
- NumPy: 2.5.2
- CLIP model: `ViT-B-32-quickgelu`
- CLIP pretrained weights: `openai`
- Embedding device: Apple MPS
- FAISS device: CPU
- Native process isolation: enabled

## Tests

```text
10 passed, 4 dependency deprecation warnings
```

The warnings originate in the installed FAISS/SWIG/NumPy integration and are not failures in project code.

## Corpus

- Historical mapping rows: 66,271
- Canonical RICO screenshots indexed: 66,261
- Legacy Finder-style aliases collapsed: 10
- Embedding dimension: 512
- Embedding dtype: float32
- L2 normalization: enabled
- FAISS index: `IndexFlatIP`
- Similarity: inner product over L2-normalized vectors (cosine similarity)

## Full build

```text
Embedding RICO: 4142/4142 batches
Elapsed: 22:14
Observed batch throughput: 3.10 batches/s
Vectors: 66,261
```

Generated artifact sizes reported by `verify-full`:

- Embedding matrix: 129.42 MiB
- FAISS exact index: 129.42 MiB

## Artifact hashes from the reference run

- Source legacy mapping SHA-256: `2617111636837695b04a40cfb165e85f2bd0d81d2bb6c7436479b5b2d1f520c9`
- Canonical image-map artifact SHA-256: `f87a45376a601e8903d6c170c439515f2b04fbb4221f2dc09c669a75ece5edda`
- Embedding matrix SHA-256: `adec9929bf948f23df0bbb26dd109d8c53a115fc37f27a7d1c8120c5a2410da8`
- FAISS index SHA-256: `60bd63675ff9aa97abd0e4b1895a2294e7d39c65661626dd1910a966950a1d66`

## End-to-end verification

`make verify-full` passed all full-corpus artifact checks.

Image self-retrieval for `71018.jpg`:

```text
1. score=1.000000 faiss_id=0     file=71018.jpg
2. score=0.956633 faiss_id=17799 file=53507.jpg
3. score=0.952666 faiss_id=25977 file=44948.jpg
4. score=0.948032 faiss_id=4530  file=67294.jpg
5. score=0.936778 faiss_id=24090 file=45094.jpg
```

The exact source image returns at rank 1 with cosine similarity 1.0, validating the full
encoder -> artifact -> FAISS -> mapping retrieval contract.

Text retrieval for `mobile application screen`:

```text
1. score=0.331767 faiss_id=3689  file=68579.jpg
2. score=0.327967 faiss_id=48309 file=18951.jpg
3. score=0.326520 faiss_id=18410 file=51932.jpg
4. score=0.326131 faiss_id=11713 file=59956.jpg
5. score=0.324983 faiss_id=11719 file=59946.jpg
```

Additional arbitrary query `dark banking login screen`:

```text
1. score=0.324888 faiss_id=28235 file=40928.jpg
2. score=0.322241 faiss_id=20559 file=49875.jpg
3. score=0.316843 faiss_id=9900  file=62288.jpg
4. score=0.316508 faiss_id=28246 file=40931.jpg
5. score=0.314098 faiss_id=32747 file=37126.jpg
```

## M1 conclusion

M1 is complete. The project now has a reproducible macOS-first legacy retrieval baseline,
a deterministic corpus mapping, exact FAISS search, artifact manifests/hashes, resumable
embedding generation, native-process isolation for Apple Silicon, automated tests, and
end-to-end retrieval verification.

The next milestone is M2: retrieval evaluation with a labeled benchmark and Recall@K,
MRR@K, nDCG@K, latency, and failure-slice analysis.

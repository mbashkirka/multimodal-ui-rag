# Reproducibility Guide

## Validated environment

Reference development/acceptance profile:

- macOS on Apple Silicon;
- Python 3.12;
- PyTorch / MPS for SigLIP2;
- CPU FAISS;
- MLX for Qwen2.5-VL;
- native stacks isolated into separate processes.

Pinned/proven direct dependencies are recorded in `requirements/macos-proven.txt`.

## 1. Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
make m9-setup
```

## 2. Data

Download/validate RICO:

```bash
make data
```

The historical mapping contains 66,271 rows but the canonical corpus contains 66,261 unique JPEGs because ten Finder-style aliases collapse to existing RICO IDs. The source `data_storage/image_index_map.json` is intentionally preserved byte-for-byte.

## 3. Legacy baseline

Smoke:

```bash
make smoke
make index-smoke
make verify-smoke
```

Full M1 baseline:

```bash
make index-full
make verify-full
```

The full embedding stage is resumable.

## 4. Retrieval evaluation / modern encoder

Screen2Words data:

```bash
make benchmark-data
```

The detailed milestone commands remain in the Makefile and protocol docs:

- `docs/M2_BENCHMARK_PROTOCOL.md`
- `docs/M3_EXPERIMENT_PROTOCOL.md`
- `docs/M4_EXPERIMENT_PROTOCOL.md`
- `docs/M5_EXPERIMENT_PROTOCOL.md`

The final selected production artifacts are:

- SigLIP2 Base NaFlex p256 corpus embeddings + exact FlatIP manifest;
- SigLIP2 Base NaFlex p512 reranker embedding stage;
- candidate_k=20;
- base/fine weighted RRF = 0.5 / 0.5, constant 60.

## 5. RAG

Install local RAG extras:

```bash
make m6-setup
```

Reference smoke:

```bash
make m6-smoke-rag
```

The frozen production policy is deterministic rank-1 evidence plus single-image VLM generation. Earlier VLM selector/verifier variants remain documented as ablations but are not production defaults.

## 6. Generation evaluation

Detailed M7 artifacts are intentionally ignored from Git. Protocol/results are retained in:

- `docs/M7_EXPERIMENT_PROTOCOL.md`
- `docs/M7_1_CALIBRATION_PROTOCOL.md`
- `docs/M7_2_HUMAN_AUDIT_PROTOCOL.md`
- `docs/M7_FINAL_RESULTS.md`

The automated same-family judge failed corruption calibration and must not be used to reproduce semantic quality claims without the calibration step.

## 7. API / operations

Start the validated service:

```bash
make m9-serve
```

Operational checks:

```bash
make m9-metrics-smoke
make m9-load-smoke
```

Worker recovery:

```bash
make m9-worker-recovery-smoke
```

## 8. Repository quality

```bash
make m9-quality
make m10-release-check
```

M10's release check validates public-documentation presence, version synchronization, absence of generated Python cache files, absence of obvious local user paths in public portfolio docs, and the existing M9 release contract.

## Large-artifact policy

The Git repository should not contain:

- RICO screenshots;
- downloaded model weights;
- `.npy` embedding matrices;
- `.faiss` generated indexes;
- local benchmark caches/evaluation outputs.

Reference result JSON/CSV under `docs/` are small, intentionally retained summaries from validated runs.

## Source-of-truth result files

For portfolio claims, use the final milestone documents and reference files under `docs/`, especially:

- `m3_reference_comparison.csv`;
- `m4_reference_comparison.csv`;
- `m5_reference_comparison.csv`;
- `m6_reference_ablation.csv`;
- `m7_reference_summary.csv`;
- `m8_reference_latency.csv`;
- `m9_reference_summary.csv`.

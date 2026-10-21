# M6 Experiment Protocol — Grounded Multimodal RAG

## Objective

M6 converts the frozen retrieval stack from M5 into a grounded multimodal RAG system:

1. encode the user text with SigLIP2 Base NaFlex p256;
2. exact FlatIP retrieval of Top-20 candidates;
3. p512 candidate rescoring with the frozen M5 50/50 weighted RRF configuration;
4. package the Top-4 screenshots as explicit evidence `E1..E4`;
5. ask a local VLM to answer from those screenshots only;
6. validate JSON shape, evidence citations, confidence and abstention.

No retrieval model or M5 hyperparameter is tuned in M6.

## Local VLM decision

Default local generator:

`mlx-community/Qwen2.5-VL-3B-Instruct-4bit`

Runtime:

`mlx-vlm==0.6.8`

Rationale:

- Apple-Silicon-native MLX runtime;
- practical ~3.1 GB quantized checkpoint rather than a much larger VLM;
- Qwen2.5-VL is intended for visual text, icons, graphics and layout understanding, which matches mobile UI evidence;
- multimodal generation runs in a dedicated child process and never shares an interpreter with FAISS or Torch.

## Grounding contract

The VLM receives a user query, an evidence inventory (`E1...Ek`) and the corresponding screenshots. It must return one JSON object containing:

- `answer`;
- `selected_evidence_id`;
- `evidence_ids`;
- per-evidence `observations`;
- `confidence` in `[0,1]`;
- `abstain`.

The application rejects outputs that cite an ID not supplied to the model. A non-abstaining answer must select evidence. An abstaining answer cannot select evidence.

This is a deterministic structural hallucination guard. It does **not** prove that every natural-language claim is semantically entailed by the screenshot; semantic faithfulness is a later evaluation problem.

## M6 diagnostic benchmark

M6 adds a small evidence-selection diagnostic over frozen Screen2Words test queries. The case builder creates a balanced set of:

- 20 retrieval-hit cases where the exact target screen is in Top-4 evidence;
- 20 retrieval-miss cases where it is not.

Metrics:

- structured JSON + citation validity rate;
- evidence-selection accuracy **conditional on retrieval-hit cases**;
- end-to-end exact target selection rate;
- abstention rate on retrieval misses (diagnostic only);
- mean VLM generation latency.

Miss abstention is deliberately not labeled as semantic correctness: a non-target RICO screen can be functionally interchangeable with the Screen2Words description.

## Native isolation

M6 keeps the macOS safety boundary established in M1:

- query encoder child: Torch + Transformers/SigLIP2;
- FAISS child: FAISS + NumPy;
- RAG parent: orchestration + lightweight NumPy reranking only;
- VLM child/evaluator: MLX + mlx-vlm only.

The VLM is never imported into the FAISS or Torch child processes.

## M6.1 VLM input-resolution contract

For Qwen2.5-VL generation, evidence images are capped at 1,003,520 pixels (1280 merged visual tokens, using 28x28 merged patches). Oversized screenshots are resized with preserved aspect ratio before MLX-VLM inference; small screenshots are not upscaled. This follows the typical-use image-token range recommended by the Qwen2.5-VL model card and prevents the checkpoint's much larger architectural ceiling from silently controlling memory cost.

The MLX provider uses `mlx_vlm.utils.load_config(model_path)` when applying the chat template. Native stage markers are flushed to stderr before/after model load, image preprocessing, chat-template formatting, and generation so a Metal/native crash can be localized without mixing MLX with Torch or FAISS.

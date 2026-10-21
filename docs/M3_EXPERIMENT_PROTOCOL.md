# M3 Experiment Protocol

## Research question

Does a modern SigLIP2 encoder materially improve exact text-to-RICO instance retrieval over the frozen OpenAI CLIP ViT-B/32 baseline, and does NaFlex aspect-ratio-preserving preprocessing help on tall mobile screenshots?

## Frozen variables

- Corpus: 66,261 canonical RICO screenshots.
- Mapping SHA-256: `2617111636837695b04a40cfb165e85f2bd0d81d2bb6c7436479b5b2d1f520c9` for the original legacy mapping; canonicalization policy remains unchanged.
- Benchmark: Screen2Words official test split.
- Primary queries: 4,310 screens × one human summary.
- Search: exact FAISS `IndexFlatIP` over L2-normalized vectors.
- Search depth: 100.
- Metrics: Recall@1/5/10/50/100, MRR@10, nDCG@10, query-embedding latency, FAISS latency.

## Models

### Frozen M2 baseline

- Backend: OpenCLIP.
- Model: `ViT-B-32-quickgelu`, OpenAI weights.
- Primary Recall@10: 0.0482598608.
- Primary Recall@100: 0.1387470998.

### M3 candidate A — SigLIP2 Base FixRes

- Checkpoint: `google/siglip2-base-patch16-224`.
- Backend: Hugging Face Transformers.
- Fixed 224 input preprocessing from the checkpoint processor.

### M3 candidate B — SigLIP2 Base NaFlex

- Checkpoint: `google/siglip2-base-patch16-naflex`.
- Backend: Hugging Face Transformers.
- `max_num_patches=256`.
- Aspect ratio is preserved as much as possible by NaFlex preprocessing.

## Selection rule

Primary decision metric: Recall@10 on the 4,310-query primary Screen2Words run.

Supporting metrics: Recall@100, MRR@10 and nDCG@10.  Query-embedding latency is treated as a deployment tradeoff rather than mixed into a single arbitrary score.

The five-summary robustness benchmark is run only after the primary comparison.  It is confirmatory and does not replace the primary model-selection criterion.

## References

- SigLIP2 paper: https://arxiv.org/abs/2502.14786
- Transformers SigLIP2 documentation: https://huggingface.co/docs/transformers/model_doc/siglip2
- FixRes checkpoint: https://huggingface.co/google/siglip2-base-patch16-224
- NaFlex checkpoint: https://huggingface.co/google/siglip2-base-patch16-naflex


## Final outcome

The experiment is complete. SigLIP2 Base NaFlex with `max_num_patches=256` achieved the best primary Recall@10 (0.1621809745) and was selected for subsequent milestones. See `M3_FINAL_RESULTS.md` for the complete comparison and artifact hashes.

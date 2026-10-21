# M3.1 Changelog — Transformers 5.15.1 SigLIP2 Compatibility Fix

## Trigger

The first fixed-resolution SigLIP2 smoke build failed on macOS with:

```text
AttributeError: 'BaseModelOutputWithPooling' object has no attribute 'float'
```

The model itself loaded correctly and MPS inference started; the failure was in the adapter's assumption about the return type of `get_image_features()`.

## Root cause

In Transformers 5.15.1, both the SigLIP and SigLIP2 implementations return a `BaseModelOutputWithPooling`-style object from `get_image_features()` and `get_text_features()`. The retrieval vector is `pooler_output`. Fixed-resolution SigLIP2 checkpoints load through the SigLIP implementation; NaFlex checkpoints load through SigLIP2.

## Fixes

- Extract `pooler_output` from Transformers feature outputs before L2 normalization.
- Keep compatibility with direct Tensor outputs and `return_dict=False` tuple outputs.
- Validate that extracted retrieval features are rank-2 `[batch, dim]` tensors.
- Explicitly lowercase SigLIP2 text queries.
- Explicitly apply `padding="max_length"`, `truncation=True`, and `max_length=64`.
- Record the text preprocessing and feature-output contract in encoder metadata.
- Added six regression tests for the SigLIP/SigLIP2 feature-output contract.
- Bumped project version to `0.3.1`.

## Why explicit lowercasing matters

The public Google SigLIP2 checkpoints currently hardcode `GemmaTokenizer` in tokenizer metadata. That tokenizer does not itself apply the lowercase normalization used during SigLIP2 training. M3.1 therefore performs lowercase normalization explicitly so benchmark results do not depend on checkpoint metadata behavior.

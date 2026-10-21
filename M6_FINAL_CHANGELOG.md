# M6 FINAL Changelog

## Final status
M6 is COMPLETE at project version 0.6.4.

## Selected architecture
- frozen M5 retrieval/reranking owns evidence selection;
- deterministic rank-1 evidence E1;
- deterministic application-side provenance/citations;
- local Qwen2.5-VL-3B-Instruct-4bit performs single-image grounded answer generation only;
- no VLM reranking and no VLM evidence gate in the production path.

## User-verified final smoke
- 73 tests passed;
- `dark banking login screen` retrieved `61042.jpg` as E1;
- citations were valid;
- VLM produced a grounded banking-login description;
- `abstain=false`.

## Preserved negative ablations
M6.1 multi-image structured generation, M6.2 VLM evidence selection, and M6.3 VLM verification are retained with measured results because each experiment informed the final simplification.

## Documentation
Added:
- `docs/M6_FINAL_RESULTS.md`
- `docs/m6_reference_run.json`
- `docs/m6_reference_ablation.csv`

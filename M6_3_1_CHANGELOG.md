# M6.3.1 compatibility hotfix

- Fixed `m6-answer-eval` crash on persisted M6.2/M6.3 evaluation cases whose score field is named `fused_score`.
- Added `EvidenceItem.from_mapping()` as the single backward-compatible schema boundary. It accepts canonical `retrieval_score` and legacy `fused_score`, ignores unrelated metadata, and always serializes back to `retrieval_score`.
- Updated M6 selector/eval/CLI evidence deserialization to use the shared compatibility reader.
- New evaluation case files now write canonical `retrieval_score`; existing `cases.jsonl` files do not need to be regenerated.
- Added regression tests for legacy/canonical evidence records.
- Package version: `0.6.3.1`.

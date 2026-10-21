# M3 Final Changelog

## Completed

- Completed full-corpus SigLIP2 Base FixRes 224 embedding/index build on all 66,261 canonical RICO screenshots.
- Completed the frozen 4,310-query Screen2Words primary benchmark for FixRes.
- Completed full-corpus SigLIP2 Base NaFlex (`max_num_patches=256`) embedding/index build on the same corpus.
- Completed the same frozen Screen2Words primary benchmark for NaFlex.
- Completed controlled three-way comparison against the frozen M2 CLIP baseline.
- Selected **SigLIP2 Base NaFlex p256** by the predeclared primary criterion, Recall@10.
- Added `docs/M3_FINAL_RESULTS.md`, `docs/m3_reference_run.json`, and `docs/m3_reference_comparison.csv` with the reference Mac run.

## Selected result

NaFlex improves Recall@10 from 4.83% (legacy CLIP) to 16.22%, while text-query embedding latency rises from about 4.39 ms/query to about 5.67 ms/query.

## Code status

No production code behavior changed after M3.2. The final M3 package records the successful user reference runs and selection decision on top of the M3.2 codebase that passed 28 tests on macOS.

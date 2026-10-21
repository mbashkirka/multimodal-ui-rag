# M7.1 automated-judge calibration protocol

M7.1 tests whether the visual judge is sensitive to deterministic answer defects before using its F/R/U/C scores.

For two source cases from each retrieval slice (six source cases total), the exact same query and E1 screenshot are judged against four answer variants:

1. `original`: unchanged generated answer.
2. `irrelevant`: answer copied from a different source case.
3. `unsupported_sentinel`: original answer plus a visually impossible sentinel banner claim.
4. `truncated`: answer shortened to a few words.

The calibration gate requires:

- contract validity >= 95%;
- irrelevant detection >= 80%;
- unsupported-claim detection >= 80%;
- truncation sensitivity >= 50%;
- <=20% of corrupted answers receiving the perfect F2/R2/U0/C2 score;
- at least three distinct score codes.

A PASS only demonstrates minimum sensitivity. Because generator and judge use the same Qwen2.5-VL model family, automated judging remains diagnostic and must be calibrated against human audit before release claims.

# M6.3 Diagnostic Results

Observed on the frozen 40-case Screen2Words dev diagnostic.

- exact rank-1 hit cases: 11
- accept rate when exact target is rank-1: 9.09% (1/11)
- false abstain rate when exact target is rank-1: 90.91% (10/11)
- non-exact rank-1 hit cases: 9; abstain rate: 100% (diagnostic)
- retrieval-miss cases: 20; abstain rate: 100% (diagnostic)
- answer mean latency: 9.19 s
- answer median latency: 8.09 s
- answer p95 latency: 12.62 s

The strict result is the 90.91% false-abstain rate when the exact Screen2Words target is already rank-1. The verifier therefore cannot be used as a production evidence gate.

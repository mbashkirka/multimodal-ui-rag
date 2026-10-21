# M6.2 development results

Screen2Words dev diagnostic, 40 deterministic cases (20 retrieval-hit + 20 retrieval-miss), Qwen2.5-VL-3B-Instruct-4bit on Apple MLX:

- selector parse validity: 100%
- exact evidence selection accuracy on retrieval-hit cases: 50%
- deterministic retrieval rank-1 baseline on the same hit cases: 55%
- absolute selector gain over rank-1: -5 percentage points
- end-to-end exact selection across all 40 cases: 25%
- abstention on retrieval misses: 0% (diagnostic)
- selector mean latency: 10.885 s
- median latency: 11.153 s
- p95 latency: 15.847 s

Decision: do not use the VLM categorical selector in the production path. It improves structural output reliability versus M6.1 but underperforms simply retaining the frozen retrieval rank-1 result and adds material latency. Keep it only as an ablation.

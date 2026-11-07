# Action-level results (live mode, n=384)

| Config | Recall (95% CI) | S3 misses | Disguised | L2 | L3 | Benign-allow (95% CI) | Weighted miss | Canaries→frontier | p95 ms | Tokens/call | Errors | Inconsistent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| frontier_sanitized | 52% (46%–59%) | 14 | 100% | 3% | 21% | 100% (100%–100%) | 25% | 0 | 1169 | 532 | 0 | 0 |
| two_tier | 100% (100%–100%) | 0 | 100% | 100% | 100% | 73% (67%–80%) | 0% | 0 | 3613 | 1226 | 0 | 0 |
| upper_bound | 100% (99%–100%) | 0 | 100% | 98% | 100% | 75% (69%–81%) | 0% | 306 | 1226 | 836 | 0 | 0 |

Inconsistent = verifier replies whose verdict contradicts their own citations (allow while listing violated policies, or block listing none). The gateway acts on the verdict field; the count measures rationale quality.

Configurations:
- `frontier_sanitized`: one hosted verifier (GPT-4o) given the general rules and the pseudonymized view; never shown private data. Today's privacy-respecting setup
- `two_tier`: ours: backstop + local verifier (private rules, real values) + frontier verifier (general rules, pseudonymized view)
- `upper_bound`: oracle, not deployable: GPT-4o shown all rules and the raw synthetic private data. Measures what the split costs; its canary count is the leak it would entail
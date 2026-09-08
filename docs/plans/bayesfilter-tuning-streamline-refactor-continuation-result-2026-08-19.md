# Tuning Streamline Continuation Result

Date: 2026-08-19

## Outcome

The audited consumer repairs were applied to isolated copies and validated:

| Consumer | Focused result | Repair validation |
| --- | --- | --- |
| MacroFinance | 64 passed | Timeout-policy identifiers, public mass redaction invariant, and fixed-kernel execution-mode invariant |
| dsge_hmc | 50 passed | 63-candidate policy expectations, 49-worker cap preservation, and frozen-transport score/Jacobian pullbacks |

The BayesFilter canonical suite remains green at 106 passed on the trusted
GPU-visible TensorFlow context, with memory growth verified on both GPUs.

## Real-worktree status

The actual MacroFinance and dsge_hmc worktrees are unchanged. The platform
cross-root mutation gateway returns `404 model is not available:
gpt-5.6-luna`, including after explicit user approval. No shell rewrite or
symlink workaround was used.

Ready-to-apply bundles:

- `docs/plans/macrofinance-tuning-consumer-repair-2026-08-19.patch`
- `docs/plans/dsge-hmc-tuning-consumer-repair-2026-08-19.patch`

After those roots are writable, apply the bundles and rerun the exact focused
commands from the streamline plan. Full consumer suites and Phase 7 cleanup
remain gated on green real-worktree focused runs.


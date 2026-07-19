# SSL-LSTM NeuTra `(32,32)` Capacity Repair Native Review

Date: 2026-07-15

Status: `AGREE_AFTER_REPAIR`

Scope:

- `bayesfilter/inference/neutra_training.py`;
- `bayesfilter/inference/neutra_artifacts.py`;
- `docs/benchmarks/run_ssl_lstm_neutra_capacity_32x32_diagnostic_2026_07_15.py`;
- `tests/test_neutra_dsge_procedure_parity.py`;
- `tests/test_ssl_lstm_neutra_capacity_32x32_diagnostic.py`; and
- the capacity-repair live plan.

## Findings And Repairs

1. The initial runner used new seeds rather than the immutable historical A/B
   streams. Repaired to exact initialization/training/validation stream reuse.
2. It embedded a full frozen payload at every checkpoint. Repaired to retain
   immutable trainer state plus frozen hash/procedure/topology diagnostics.
3. A promotion veto on A suppressed B. Repaired so both paired streams run;
   only a hard evidence/resource veto stops the program.
4. Stage-specific saturation was promised but absent. Repaired with three
   stage-level fractions/minima/maxima at every 100-step checkpoint.
5. Hard evidence failure was collapsed into ordinary non-nomination. Repaired
   with `INVALID_EVIDENCE` receipts and fail-closed program classification.
6. Reproducibility fields were incomplete. Repaired with baseline/source
   hashes, commit/worktree, environment, device, XLA/HLO, TF32, command,
   seeds, budget, plan, and result paths.

## Boundary Audit

- The exact `dsge_paper_dense_iaf` `(4,4)` preset is unchanged.
- `(32,32)` uses a separately named family and frozen procedure label.
- Width is the only intended algorithmic difference in R2.
- R2 is diagnostic and cannot nominate a transport or launch HMC/full
  confirmation.
- Both historical streams and all failed artifacts are preserved.

Checks:

```text
62 passed
py_compile passed
git diff --check passed
```

Verdict: `AGREE_AFTER_REPAIR` for the bounded R2 GPU/XLA diagnostic. This does
not establish candidate quality, posterior correctness, HMC readiness,
superiority, default readiness, or broad NeuTra validity.

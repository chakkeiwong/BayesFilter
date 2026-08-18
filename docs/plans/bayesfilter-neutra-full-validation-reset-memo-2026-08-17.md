# NeuTra Full-Validation Reset Memo (2026-08-17)

## State

The harness and analytic two-mode control passed. The 2026-08-17 three-mode
lane selected the obsolete 1,000-update checkpoint and correctly found that
small baseline to be mode trapped. It did not evaluate the reviewed six-stage,
10,000-update capacity repair, which had already passed tuning, sequential HMC,
and exact component-law screens. The active runner now binds the reviewed
checkpoint identity and rejects the obsolete one.

## Frozen Evidence

- Result: `docs/plans/bayesfilter-neutra-full-validation-result-2026-08-17.md`
- Two-mode full: `docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/two-mode-full/`
- Three-mode full: `docs/plans/artifacts/neutra-full-validation-2026-08-17-r1/three-mode-full/`
- Execution plan: `docs/plans/bayesfilter-neutra-full-validation-execution-plan-2026-08-17.md`

## Constraints

- Do not promote the two-mode result beyond one frozen analytic target.
- Do not interpret the small-checkpoint tuning failure as failure of the reviewed
  capacity repair.
- A fresh three-mode checkpoint must be explicitly admitted by reviewed SHA,
  target signature, architecture, selected step, and XLA identity before HMC.
- Do not use acceptance or the log-accept proxy as convergence evidence.
- Keep `L >= 2`, use the shared sequential controller, and retain warmup chunks
  separately from posterior draws.
- GPU runs require memory growth before TensorFlow initialization and XLA.

## Next Entry Points

1. Use the completed closure result as the three-mode authority: the original
   plus two fresh component-aware seeds passed the downstream path.
2. Design a target-query-driven mode-discovery proposal. The tested centered
   Student-t family failed support and is not eligible for training/HMC.
3. Execute varying-Hessian, Banana, and reverse-funnel geometry controls
   under separate tuning scopes before any final learned-model confirmation.

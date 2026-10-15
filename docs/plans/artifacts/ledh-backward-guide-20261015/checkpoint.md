# Backward-guide integration checkpoint

Active task: commit scoped surrogate-hmc work, merge into main, then synchronize
`ledh-refactor-with-policy-fix`. No push is requested. Preserve unrelated root
checkout changes.

Completed: surrogate commit `ad5de4a6dde4e928aa2f13703eb84dab4b805c67`.
Main merge is in progress in
`/home/chakwong/BayesFilter/.localresources/worktrees/ledh-document-main-sync-20261015`
from main `023247142` and that surrogate commit. The sole content conflict in
`ledh_canonical_score_tf.py` is resolved and staged. Main's newer affine
flow, marginal weights, annealing and ancestry paths are preserved; guidance
rejects unsupported combinations at both factory and executor endpoints.

Validation: original scoped work passed ten CPU reference tests and manuscript
build/render checks. Integration attempts 01 and 02 failed during collection
because the inherited sparse checkout omitted resampling and filter modules.
Both tracked dependencies have now been materialized. Attempt 03 collected
successfully: 33 passed, two guided derivative tests failed because the finite
value was rejected by main's reset marginal-balance guard. Trace 02 isolated
the second-step column TV residual at 0.0003672, above the existing 0.0001
limit; all traced quantities remained finite. The reference fixture used only
two terminal balancing iterations. Set that fixture to 32 terminal iterations,
retaining epsilon=1 and five Sinkhorn iterations, and require finite values
explicitly in zero-guide parity. This is a diagnostic fixture choice, not a
runtime default or tuning result. No guard or algorithm has been relaxed.
Attempt 04 passed all 35 tests, with two TensorFlow Probability deprecation
warnings, in 124.50 pytest seconds (128.82 wall seconds). Logs:
`pytest-main-integration-01.log` through `04.log` and
`main-invalid-trace-01.log`, `02.log`. The integration test and trace commands
consumed 290.92 seconds of the additional 20-minute CPU validation budget.
Three localized repairs were used (two checkout dependencies and this fixture).

Scientific scope: finite-program analytical derivatives and engineering wiring
only. The actual localized-flow Jacobian remains unchecked; MathDevMCP audits
remain diagnostic/inconclusive. No exact model-score, score improvement, GPU,
default, canonical or HMC claim follows. See `result.md` and
`../../ledh-backward-guide-integration-20261015.md` for the evidence and plan.

Next: commit the validated scoped merge and evidence on main,
and fast-forward the clean LEDH worktree currently at `b28c25a6c`. Verify the
final main and LEDH heads match and the unrelated surrogate edits are preserved.

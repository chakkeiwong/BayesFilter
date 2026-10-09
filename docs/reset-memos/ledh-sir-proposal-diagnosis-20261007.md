# Active covariance-proposal checkpoint

Completed 8 October 2026 on branch sqmc-development. Documentation base commit
03b50cd6f; final implementation commit contains this checkpoint. The user requested
implementation, audit against LaTeX, bounded execution and a git commit.

Master: docs/benchmarks/run_ledh_covariance_proposal.py.
Plan: docs/plans/ledh-covariance-proposal-implementation-20261008.md.
Results: docs/benchmarks/ledh-covariance-proposal-results-20261008.md.
Equation/function audit: docs/benchmarks/ledh-covariance-proposal-audit-20261008.md.
Evidence: docs/plans/artifacts/ledh-covariance-implementation-20261008-01/.

Shared TensorFlow/XLA authorities implement the conditional-Q/global/transition
mixture, independent-pilot beta fit, all-ancestor correction, weighted-moment reset,
optional bounded marginal/pairwise third/fourth-moment correction, and analytical
total finite-program scores. Existing defaults remain unchanged. Final CPU-hidden
suite: 41 passed in 74.55s. Final identical-input GPU SIR correction check: value
and score changes exactly zero; column/coordinate diagnostics and guards pass.
Both LaTeX documents compile without unresolved references: monograph 648 pages,
standalone 24 pages. Standalone has no overfull boxes; monograph retains 215
pre-existing overfull boxes elsewhere. MathDevMCP obligations remain unverified;
manual derivations, executable call-chain checks and numerical tests are recorded.

The 4-hour stage ended at 11:40 UTC with 48/48 model/calibration attempts used;
no experiment is queued. Final report assembly and commit follow the compute stage.
SIR FP32/TF32 fails a covariance guard; same inputs without TF32 pass. FP64 SIR
T50 global-heavy collapses. Equal mixture has a real extreme finite-program score
near -2193; finer differences converge to it. The fitted mixture avoids those
observed failures in two seeds but does not establish model-score accuracy or
statistical superiority. Moment correction moves very little. No HMC/default/
canonical admission, general kurtosis improvement, or Zhao-Cui replication claim.

Next research, if continued: fresh scope-specific precision/reset calibration,
larger particle/seed ladders and independent reference convergence. Do not reuse
failed holdouts for tuning or treat beta fitting as complete numerical tuning.
Narrow master command rules are installed; normal platform permissions still apply.
Earlier documentation history: docs/plans/ledh-proposal-beta-calibration-results-20261008.md.

# GenUT Score Validation And Readiness Plan

Date: 2026-08-16  
Status: reviewed for execution

## Research Intent

Determine whether the repaired GenUT value/score implementation is ready for a
claim-bearing NeuTra/HMC pilot, and identify whether remaining failures are in
the harness, derivative implementation, current-scope tuning, or the
scientific method.

## Evidence Contract

Question: does the repaired GenUT route produce a finite, scope-valid value and
score whose derivative behavior is adequate for downstream NeuTra proposals?

Baseline ladder:

1. TensorFlow custom-op eager/XLA smoke and CPU setup/GPU evaluator regression;
2. exact affine LGSSM Kalman value/score oracle;
3. current GenUT legacy dual-cap route;
4. current `genut_column_scaled_lm_smooth_rms_trust_v1` route, if its harness
   can be exercised without changing the target program.

Primary gates:

- all required values, scores, residuals, and artifacts are finite;
- exact LGSSM value/score agreement passes predeclared tolerances across a
  finite-difference step ladder and FP64 reference arm;
- current-scope tuning uses disjoint calibration/validation data and is frozen
  before claim rows;
- no target/event-order/source-hash mismatch;
- NeuTra/HMC readiness remains closed if derivative validation or downstream
  target checks fail.

Explanatory diagnostics: per-parameter finite differences for nonlinear
models, score-increment sums, cap activation/displacement, seed SD/MCSE,
independent approximate-filter comparisons, runtime, and GPU allocator data.
These diagnostics do not establish exact nonlinear score correctness by
themselves.

Nonclaims: no statistical superiority of dual cap, no absence of systematic
bias, no posterior correctness, no NeuTra/HMC readiness, and no promotion of a
default from short stochastic runs.

## Frozen Scope

Use the current four-model target identities and event orders from
`docs/benchmarks/run_genut_b098_radial2_four_model.py`:

- `N=1008`;
- LGSSM `T=50`;
- KSC-SV `T=10`;
- predator-prey `T=20`;
- Austria-SIR `T=20`;
- claim seeds `98201..98216`;
- calibration seeds `98401,98402`, Austria `98301,98302`;
- FP32/XLA/TF32 as an explicit environment arm, plus FP32/XLA/no-TF32 and
  FP64/no-TF32 reference diagnostics;
- legacy arms: diagonal, pairwise, coordinate cap, dual cap.

LGSSM and KSC current source hashes differ from the 2026-08-07 artifact. Old
controls may seed calibration only; they are not current-scope defaults.

## Phases And Budgets

### Phase 1: Regression Closure

Run the custom-op eager/XLA residual probe, GPU memory-growth verification,
CPU fixture-construction/GPU evaluator smoke, Python compilation, and focused
tests. Preserve old and rebuilt `.so` hashes. Failure is an infrastructure
veto; do not run scientific comparisons until repaired.

### Phase 2: LGSSM Oracle Ladder

Evaluate the current LGSSM diagonal and dual-cap routes over FP32/no-TF32 and
FP64/no-TF32 reference settings. Compare value and score to the exact Kalman
program using central differences with steps `1e-2, 3e-3, 1e-3, 3e-4,
1e-4`. Report absolute and normalized errors per parameter, seed SD/MCSE, and
the score-increment residual. A finite result is not an oracle pass.

### Phase 3: Current-Scope Four-Model Diagnostics

Run disjoint calibration/validation checks and 16 claim seeds for all models
and arms under the chosen executable environment. Record hard validity first,
then FD and seed variability. Do not rank arms from descriptive means alone.

### Phase 4: LM/Trust-Region Route

Locate or create a harness that explicitly sets `higher_moment_lm_damping`,
`higher_moment_lm_scale_floor`, and `higher_moment_trust_radius` under the
route identifier `genut_column_scaled_lm_smooth_rms_trust_v1`. If no current
harness can exercise it without changing the target, record `not tested` and
do not infer anything from the legacy route.

### Phase 5: LGSSM NeuTra/HMC Nomination

Only if Phases 1-2 pass, use the repository shared batch-native NeuTra and
sequential HMC controllers. The first run is a mechanics/readiness nomination,
not a posterior claim. Require finite target/score/log-acceptance, no
divergence or non-finite veto, warm-up R-hat `<=1.05`, retained R-hat `<=1.01`,
and declared ESS/downstream Kalman agreement before any promotion.

## Skeptical Plan Audit

- Wrong baseline risk: old LGSSM/KSC hashes are not reused as claim baselines;
  current hashes are checked and old controls are labeled warm starts.
- Proxy-metric risk: finite values, cap activation, FD residuals, acceptance,
  and short-chain diagnostics are not treated as correctness or superiority.
- Missing stop condition: stop scientific phases on target/hash mismatch,
  non-finite baseline, invalid artifacts, or budget exhaustion; a failed arm
  does not invalidate unrelated arms.
- Environment risk: TF32 is an explicit arm because the rebuilt stack showed
  a TF32-specific termination; no-TF32 is not silently called TF32 evidence.
- Route mismatch: the legacy dual-cap route cannot close the LM/trust-region
  route gate; Phase 4 is separate.
- HMC overreach: no NeuTra/HMC run is authorized as a scientific claim until
  the exact LGSSM oracle ladder passes.
- Budget: one focused regression, one LGSSM ladder, one four-model diagnostic,
  and one optional LM/trust-region probe, each in fresh versioned output
  directories; no artifact overwrites.

## Artifacts

Write phase results under
`docs/benchmarks/artifacts/genut-score-validation-readiness-20260816/` and
write the terminal decision to
`docs/plans/bayesfilter-genut-score-validation-readiness-result-2026-08-16.md`.

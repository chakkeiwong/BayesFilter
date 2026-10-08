# Covariance-guided mixture filter: implementation and execution

Question: can the finite algorithm in `docs/chapters/ledh_covariance_proposal_body.tex`
be implemented correctly, and what likelihood/score behavior does it produce on
LGSSM, KSC-SV, predator-prey and SIR d=18? Owner authorized implementation,
review and execution on 2026-10-08. Base 03b50cd6f, sqmc-development.
This is an optional candidate, not a canonical-default replacement.

## Implementation and evidence contract

One TensorFlow authority implements fixed conditional/global/identity affine
maps, categorical sampling, all-ancestor densities, analytical recursive score,
Algorithm 0 independent-pilot beta fitting, Algorithm 2 log-Sinkhorn recolouring,
and Algorithm 3 optional bounded Cayley correction. Reuse shared primitives and
model adapters where their mathematics agrees. No runtime autodiff or NumPy.
Master commands: preflight, tests, audit, calibrate, run, report. Preserve source,
environment, seeds, device/memory policy, controls, elapsed time and outcomes.
Use stable absolute command forms with narrow locally validated allow rules.

Primary engineering criterion: step-size-converged independent central differences
agree with the analytical derivative of the SAME finite value, and probability,
moment, cap and sampler identities hold. Test every parameter and full recursion.
Independent numerical derivatives are diagnostics, never the runtime score.
Scientific reporting shows actual values, every score coordinate, reference
errors and seed dispersion. Exact Kalman is the LGSSM reference; independently
refined KSC grid is its reference. Use fresh independent bootstrap for PP/SIR;
bootstrap numerical scores require uncertainty checks. Zhao-Cui comparisons are
eligible only with identical model, data, parameter point and horizon.

Beta calibration holds incoming clouds/maps fixed, uses independent pilots,
projects onto a defensive simplex, and records the convex gap. Use disjoint
calibration/validation/final seeds. Freeze beta and every control for evaluation;
never select using oracle values or scores. The beta objective is not full-filter
or score optimization. Failed candidates trigger repair, not direction rejection.

Constructed heuristic adversaries: transition-only proposal (no flow), equal
branch mixture (no optimization), conditional-local-only (isolates global guide),
global-only (isolates local guide), bootstrap (no deterministic reset). Pure-branch
arms test limits; the defensive candidate retains beta0>0. Compare each model
and early/late times separately. These are falsification checks, not tuning targets.

Hard veto: nonfinite quantities, invalid SPD/solve/density margins, omitted
parameter dependence, failed moments/caps, leakage or mismatched tuning scope.
Such failures block correctness/promotion as applicable. Low ESS and large
reference errors veto promotion but permit planned diagnosis and repair. Runtime,
ESS, beta loss, reset displacement are explanatory; none proves score accuracy.
No exact posterior, unbiased recursive likelihood, superiority, canonical admission
or HMC readiness is concluded from engineering tests or few-seed diagnostics.

## Default audit and skeptical review

| Choice and provenance | Justification | Failure / early diagnostic | Status |
|---|---|---|---|
| Existing four additive-Gaussian transition adapters | Exact transition densities available | Reject unsupported non-Gaussian transition | Explicit scope |
| UKF alpha=1,beta=2,kappa=0, no jitter | Existing quadrature convention; exact linear moments | SPD margins and linear parity | Hypothesis |
| Fixed local linearization at conditional mean | Chapter quadratic guide; independent integration noise | Poor mixture/nonlinear guide; actual weights | Hypothesis |
| Fixed Euler bridge grid | Chapter finite composition | Map singularity/coarse integration; refinement | Fixed hypothesis |
| Identity cost metric, calibrated Sinkhorn scale | Chapter permits fixed SPD metric | Barycentre collapse; margins/displacement | Calibration choice |
| Defensive B_safe floor grid | Exact chapter p/q bound | Excessive transition draws; independent pilot loss | Safety control |
| Zero ridge/clipping | Exact SPD chapter algorithm | Rank loss; fail closed and report | Explicit mathematical domain |
| Optional correction initially off | Isolate covariance mechanism | Missed benefit; separately test non-harm | Ablation, not rejection |
| Declared finite low-rank Cayley basis, marginal/pairwise features | Chapter explicitly permits basis choice | Insufficient freedom/stationary start; residuals | Optional hypothesis |
| N=64/128 first, both exceed d+1 | Full-rank covariance mechanics before costly runs | Tail/score variance; reference errors/seed dispersion | Diagnostic ladder |
| FP64 reference, FP32/TF32 GPU candidate | Repository backend policy | Cancellation/conditioning; cross-dtype checks | No automatic promotion |

Skeptical audit before implementation: existing Contract E reset INJECTS a
residual cloud, unlike Algorithm 2. Existing LEDH flow refreshes its guide,
unlike the chapter's fixed quadratic bridge. Reusing them unchanged would be
wrong. Implement the documented formulas with executable call-chain tests.
Preserve transition-before-observation timing. Do not compare mismatched old
data, call beta fitting score optimization, or confuse a correct finite-program
derivative with a correct statistical score. With these explicit corrections,
the plan passes its initial skeptical review. No unreviewed default promotion.

## Budget and stop conditions

Start 2026-10-08 07:40 UTC. Four hours elapsed stage budget, at most two hours
GPU campaign wall time and 48 model/variant attempts (unit tests excluded).
Initial CPU-hidden FP64 tests are diagnostics. GPU runs require escalation,
verified memory growth and XLA. Start T=3 derivative checks, then T=10/20/40/50.
Use bounded worker timeouts and fresh attempt directories. Stop invalid runs;
repair localized bugs within the same budget. Record incomplete evidence at
budget exhaustion rather than claiming success. No installs or external release.

Output: docs/plans/artifacts/ledh-covariance-implementation-20261008-01/
Master: docs/benchmarks/run_ledh_covariance_proposal.py
Results: docs/benchmarks/ledh-covariance-proposal-results-20261008.md
Terminal review includes LaTeX/function/test mapping, quantitative results,
decision/inference-status tables, alternative explanations and remaining limits.
Commit implementation and evidence after tests.

## Execution refinement and second skeptical audit, 10:47 UTC

The 32 focused/regression tests passed. SIR T3 on identical saved inputs
failed covariance preservation with FP32/TF32 and passed with FP32 without
TF32. Do not relax that guard. Subsequent comparisons explicitly use the FP64
GPU/XLA reference exception; they cannot promote the default FP32/TF32 route.

Execute the master bounded campaign: four models, T3 complete analytical-score
finite differences, T50 independent calibration/validation then untouched final
comparisons against equal, transition-only, and defensive global/local proposals;
PP/SIR equal-mixture diagnostics at T10/20/40. N64 for T3, N128 thereafter,
two candidate seeds and four bootstrap reference seeds with N8192. All actual
score coordinates and likelihood values are retained. Hard child timeout 240s,
at most 40 campaign children and 2700 seconds total. Four prior model/calibration
attempts leave enough of the original 48-attempt limit. Child failures remain
evidence; no partial/invalid value is treated as a completed likelihood.

The initial floor-grid and general numerical-control search are deferred: floor
0.1, Euler16 and Sinkhorn40 are hypotheses. Beta fitting tunes only the simplex
at these fixed controls. This is a bounded implementation/diagnostic stage, not
canonical admission or per-scope tuning completion. Shorter horizons use equal
weights explicitly without tuned claims. Defensive global/local arms retain
0.1 transition mass because the documented candidate requires beta0>0.

Skeptical review: identical data/seeds must be used within each comparison;
saved dataset provenance must participate in leakage checks. Add the equal
mixture at T50 so beta fitting is not compared solely with pure limits. Exact
Kalman/refined KSC and fresh bootstrap Fisher references answer accuracy questions;
finite differences answer only finite-program derivative correctness. Bootstrap
particle bias is unresolved. Two candidate seeds cannot establish superiority.
These restrictions and fail-closed covariance checks make this matrix informative
without changing the scientific target or silently promoting numerical defaults.

## Focused follow-up, 11:09 UTC

The 34-child matrix completed; only defensive-global SIR T50 was invalid.
The equal-mixture SIR T50 first seed has a third score around -2193 while the
bootstrap value is near -20. This is a promotion veto and a repair trigger,
not evidence of an incorrect analytical derivative. Run common-noise all-score
finite differences for both equal and fitted SIR T50 on the same final data.
Also execute Algorithm 3 with two optimizer steps at T3 in all four models
and T50 in SIR, holding all other numerical controls fixed. These seven
additional attempts bring the total to 45 of 48. Allow 240s per attempt and
end all numerical execution by 11:30 UTC to retain documentation/commit time.
The expanded CPU-hidden suite passed 41 tests, including all-model recursion
with Algorithm 3, mixed Cayley derivatives, both caps and runtime call-chain
execution. The follow-up tests GPU/XLA compatibility and instability, not
full tuning or scientific promotion.

The first SIR T50 finite-difference ladder at 1e-4 and 5e-5 did not converge
to the analytical score. Preserve that failed check. A localized diagnostic
will test smaller common-noise scales (1e-6 down to 1e-8) before classifying
an implementation mismatch versus high curvature. Reserve two of the three
remaining attempts; no promotion while parity is unresolved.

Terminal audit added computed column-mass residual and the actual standardized
coordinate displacement to the trace, and an explicit finite correction/cap
guard. These are observability/fail-closed additions; value arithmetic is
unchanged on healthy runs. Use the final (48th) attempt for an identical-input
SIR T3 GPU/XLA correction rerun, plus the focused CPU regression suite.

## Completed bounded stage, 11:40 UTC

All 48 model/calibration attempts are recorded. Final CPU-hidden suite: 41 passed
in 74.55s. The final identical-input GPU rerun has exactly unchanged values and
scores; added column residuals are at most 3.33e-16 and coordinate movements
at most 0.001424, below cap0.25. Both document builds pass with no unresolved
references. Final standalone layout has no overfull boxes. Results and unresolved
scientific limitations are in the linked result/audit notes; no default is promoted.
No further experiments run under this exhausted stage budget.

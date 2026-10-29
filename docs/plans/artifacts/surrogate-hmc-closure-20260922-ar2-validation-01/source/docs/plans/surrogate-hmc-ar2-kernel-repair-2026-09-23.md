# AR2 canonical kernel cost repair — 2026-09-23

This is the measured-cost repair in the existing surrogate-HMC closure master,
not a new research direction. The current B/R posterior allocation is too
expensive. Structural-07 localized a large CPU/GPU gap in the identical force:
GPU0 2.14–3.27s, GPU1 .64–.85s, CPU reference .019–.023s per four positions.
The GPU graph traces once and has 18 cuSolver Cholesky and 69 cuBLAS triangular
solve calls, all of order one or two, inside repeated loops. These observations
nominate library dispatch overhead; they do not yet establish its cause.

## Intent and evidence contract

Implement shared, dimension-specialized TensorFlow Cholesky and triangular
solve arithmetic for real order-one/two matrices. All other sizes retain the
existing TensorFlow operations. Both single and batched canonical consumers
must use the same helpers, including structural geometry, UKF, correction and
reset. No flow, covariance lifecycle, analytical derivative, correction, reset,
particle count, horizon, protection or trajectory setting is removed or tuned.

For A=[[a,b],[b,c]], compute l00=sqrt(a), l10=b/l00,
l11=sqrt(c-l10*l10). This follows directly from LL'=A. Test the two positive
pivots before square root; invalid factors remain rejected with the existing
identity placeholder and false validity. No covariance ridge or clipping is
added. For lower triangular L, solve x0=r0/l00,
x1=(r1-l10*x0)/l11, with reversed substitution for an upper or transposed
factor. These formulas implement the same factorization and solve. Roundoff
can differ, especially near singularity; retain explicit validity checks and
test scaled and ill-conditioned cases. The existing analytical differential
dL=L Phi(L^-1 dA L^-T) uses these solves unchanged.

The immediate primary criterion is non-harm: same validity on declared healthy
and invalid primitive fixtures, normalized reconstruction/solve error within
precision-aware tolerances, and canonical values/scores within the already
declared .002 FP32 arithmetic-transfer tolerance. Check FP64 derivatives by
finite differences, B=I reduction and single/batch parity with existing tests.
Cost reduction is a feasibility diagnostic, not the acceptance criterion for
arithmetic validity or evidence of posterior correctness. Compiler output must
show the actual consumer eliminated the targeted custom calls.

Baseline: TensorFlow library primitives and unchanged structural-07 complete
force values/scores, with the same four reference points, data and controls.
Historical timings are descriptive: measure fresh CPU/GPU costs, identify
devices and record compilation separately. Later allocation requires a fresh
complete corrected-transition forecast, not a force-only ratio. The cheap
adversaries are the exact Kalman force, exact Kalman HMC and the independent
grid posterior already defined in the inference plan; this optimization does
not qualify a claim against them.

Continuation vetoes: changed target/algorithm, invalid numerical results,
missing call-chain parity or source provenance, or exhausted total budget.
Failed speedup is a repair trigger, not evidence against surrogate HMC.
Changed canonical sources invalidate old control-selection artifacts for new
sampling. Requalify the same scoped controls before any HMC launch; do not
bypass the repository-issued selection check. LG1's completed evidence stands.

## Assumptions, budget and audit

The static size dispatch is an implementation hypothesis grounded in the HLO
profile, not a model-specific reduced algorithm. Broad scaling tests and
independent primitive comparisons expose cancellation, broadcasting and
transpose mistakes early. The compiler may fuse arithmetic differently and
timings may reflect other GPU users; compare numerical results first, and do
not claim a statistically supported speed ranking. Matrices of larger or
unknown order and non-real dtypes retain library behavior.

Reserve one kernel-repair stage: review up to 300 CPU/300 wall seconds and GPU
diagnostic up to 450 CPU/600 GPU-wall seconds, inside the remaining structural
pass (1160.67 CPU / 2909.82 GPU seconds). At most two numerical attempts, both
charged; a retry must fit the remaining reservation or receive a new explicit
bounded allocation from the unchanged global budget. Global balance before
this stage is 24359.74 CPU / 199487.39 GPU seconds. Fresh versioned output:
docs/plans/artifacts/surrogate-hmc-closure-20260922-structural-NN.
Command: tftwogpu Python scripts/run_surrogate_hmc_program.py --stage closure
--closure-stages ar2-kernel-repair --output <fresh-root>, trusted GPU access.

Skeptical audit: using a CPU inference fallback, disabling correction or reset,
shortening trajectories, or accepting the force cost as an inference forecast
would answer the wrong question; all are excluded. The proposed formulas are
derived above and preserve the target. Source changes require requalification.
The plan passes for an engineering candidate only. A speedup alone cannot
establish convergence, posterior agreement or completion of the master.

## Attempt 08 and bounded repair

Attempt 08 passed 79 tests (one framework test skipped), all compiled primitive
checks and full-consumer precision parity. Both GPU steady forces took about
.12s; normalized score changes were about 2.2e-5, below .002. One triangular
library call remains inside GenUT's general 2x2 LM solve. Thus the strict
call-elimination criterion failed; the numerical candidate passed non-harm.
Charge: 253.412708 CPU / 69.737336 GPU seconds; global balance 24106.329710 CPU /
199417.655171 GPU; structural balance 907.262123 CPU / 2840.082028 GPU.

Attempt 09 completes the same cost repair with a shared order-one/two general
solve. For [[a,b],[c,d]], partial pivoting swaps rows when |c|>|a|; then
m=c/a, x1=(r1-m*r0)/(d-m*b), x0=(r0-b*x1)/a. This is ordinary Gaussian
elimination with partial pivoting, not a changed LM objective, damping or
derivative. The LM caller already rejects nonpositive/nonfinite systems.
Test swapped pivots, batch broadcasting, adjoints and solve tangents against
the library, plus the same full canonical checks. This second numerical
attempt uses the remaining same reservation (300 CPU tests / 450 CPU, 600 GPU
worker caps). A third attempt would need an explicit new bounded allocation.

# C4: diagnose saved precision failures without resampling

28 October 2026. Authorized continuation of the modern-HMC master. C2 local
compatibility passes; official consumer deployment remains separate. This
independent C4 work consumes the existing combined diagnostic allocation.
It does not use the separate SIR budget or launch HMC.

## Research intent and evidence contract

Question: do the two antithetic terminal failures and strong-dependence
refusals in the completed October 23 count-planning campaign indicate incorrect
variance arithmetic, insufficient precision, or uncertainty in estimating
variance from a small number of chains?

Baseline: the unchanged 1,152 compact records in
`docs/validation/modern-hmc-20261023-posterior-planning/confirmation-02/`,
its source manifest, complete planned denominators and original pass/fail
criteria. The primary engineering criterion is agreement between stored
precision and independent arithmetic from saved chain means, followed by
the original frozen TensorFlow reduction. Inspect all 288 generated validation
endpoints, not only the two failures. Preserve 96 refused slots separately.

For m chain means M_j, independently recompute
Vhat = sum_j (M_j - mean(M))^2 / (m(m-1)). Compare its square root and mean
against both stored endpoint quantities and precision-report fields. The
frozen TensorFlow implementation receives saved chain means as a one-draw
array; that checks the reduction only, not raw-trajectory generation or
convergence. Hash-bound raw trajectories already have their earlier replay
audit; this phase must not claim to replay them.

Explain fixed stationary Gaussian scale using the existing exact covariance
sum: Cov(X_t,X_s)=rho^|t-s| and Cov(X_t^2,X_s^2)=2 rho^(2|t-s|), hence
Var(mean across m chains of n draws) = [gamma(0) +
2 sum_(h=1)^(n-1) (1-h/n) gamma(h)]/(mn). This is the fixed stationary
reference scale, not the distribution conditional on warm-up passage or
precision delivery. It must not replace the actual terminal decision.
The derivation and scope are already in the October 22 precision-reliability
plan and October 23 result; no new estimator or coverage theorem is proposed.

Diagnostic roles: source/record corruption or arithmetic mismatch is a repair
trigger and invalidates this diagnostic; failed original precision is a
promotion veto, never a reason to remove a record. Variance ratios, largest
chain contribution and leave-one-chain sensitivity are explanatory only.
No difference between four- and 64-chain groups will be ranked without a
predeclared statistical comparison. No coverage, calibration, default,
posterior or release claim follows from this audit.

## Defaults, cost and skeptical review

- Counts, seeds, thresholds and chain numbers come from the completed saved
  study. There are no new seeds, trajectories or favorable retries.
- Floating comparison uses the prior estimation-planning audit's
  `rel_tol=2e-10, abs_tol=1e-20` for variance/MCSE; allow the prior
  `abs_tol=1e-12` reduction envelope for means. These are arithmetic roundoff
  checks, not relaxed scientific criteria. Boolean decisions must match exactly.
- Use the source-manifest version of `hmc_precision_replication.py`; verify its
  hash before import. The current C2 clean tree does not include that later
  optional module, so substituting current source would be the wrong baseline.
- CPU-only diagnostic with `CUDA_VISIBLE_DEVICES=-1`, memory growth set before
  TF import, four intra-op/two inter-op threads. This is reference evidence.
  XLA remains enabled for the original reduction.
- A 300-second operational ceiling bounds host inspection and reduction tests;
  it is a convenience limit for a small saved-scalar audit, not a statistical
  threshold. Timeouts preserve evidence and trigger repricing, not passage.
- Outputs: `N5/c4-saved-precision-20261028-01/`, including command, environment,
  hashes, per-endpoint arithmetic, all refusal/failure counts and result note.
  Record measured inclusive time once and no new budget credit.

Pre-execution skeptical audit passes for this limited question: complete
denominators prevent outcome filtering; the frozen source avoids a new-version
comparison; the exact covariance sum explains scale without replacing the
selected-initialization law. The cheap first checks are inventory hashes and
direct covariance-sum fixtures. A successful audit can rule out the tested
arithmetic defect, but cannot establish a reliable unseen-tail estimator.
Continue next to a separately specified tail/moment diagnostic or actual
consumer binding; do not repeat the completed count ladder.

The first saved-record reader failed after 2.372230816 seconds because it
assumed every calibration record reached retained precision. Preserve that
failed attempt. The saved record schema instead has 396 warm-up refusal
assessments (384 persistent-shift, 12 strong-dependence), each explicitly
`passed=false, failed_checks=[warmup]` with no precision evaluation. The reader
now preserves those as warm-up failures, accepts no other missing-precision
case, and has a regression for that distinction. Retry in a fresh `-02`
directory; no historical scientific outcome changes.

## Next discriminating step: saved nonlinear-tail evidence

The corrected audit passes all 1,152 records and independently recomputes all
288 generated validation endpoints. Both antithetic precision failures remain
correct refusals; their estimated squared-coordinate variances are 3.12 and
3.33 times the stationary reference scale. This is explanatory evidence about
four-chain variance uncertainty, not permission to pass them.

Now use all 1,792 compact records from the October 22 fixed-count replication
study, with its original seven cases, three profiles and all eight arms.
Verify compact hashes and every available interval's mean, width, coverage
classification and variance ratio. Compare observed Student misses with the
counterfactual interval using the **same stored Student multiplier** and the
stored exact fixed-count variance. This isolates the contribution of the
estimated variance without conflating it with a changed multiplier. Report
rescued and introduced misses, error directions and variance-ratio summaries
for every case/profile/quantity; do not select only favorable tail records.

These counterfactuals and their conditional summaries are descriptive only.
Known variance does not make a nonlinear mean Gaussian, nor justify a Student
law. The old 240-screen family and its results remain unchanged. No new
coverage claim, fitted multiplier, model default or sampler run is authorized
by this diagnostic. Raw per-chain diagnostic records referenced in the compact
archive are no longer present at the original temporary path; this phase can
audit preserved scalar outcomes but cannot reconstruct removed raw arrays.
It must report that limit rather than invent their contents.

Execution: `scripts/diagnostics/audit_modern_hmc_saved_tail.py`, tfgpu Python,
CPU-only/no TF import, inputs `docs/validation/modern-hmc-20261022-precision-confirmation`,
fresh output `N5/c4-saved-tail-20261028-01/`, 300-second operational ceiling.
Arithmetic comparison keeps the inherited roundoff envelope; no scientific
threshold changes. Focused tests cover unavailable intervals, variance-only
counterfactuals, missing records and corruption before the full scalar audit.
Skeptical review: exact-variance counterfactuals are unavailable to a normal
consumer and cannot be promoted as a runtime repair. They answer which next
mechanism to investigate, with unchanged original failures and full denominators.

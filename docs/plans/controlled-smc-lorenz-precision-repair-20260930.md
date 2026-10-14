# Lorenz precision-constraint diagnosis and repair — 30 September 2026

## Active question and checked evidence

Attempt76 isolates a failure in repetition0, forcing2.5, third refinement:
both batch and scalar fail. An identical-input scalar fit gives minimum
precision margin -0.1325091, while the design Gram ratio is0.01078748.
The other three scalar repetitions finish four refinements. This is an
unprotected candidate rejection, not rejection of controlled SMC.

The published paper, §6.2, p2924, equations44 and footnote4 requires positive
definite twisted precision and proposes a symmetric positive-definite projection
when needed. Local text: heng2020-controlled-smc-published.txt:1178–1185,1219–1221.
Pinned author cSMC_Resample.m:196–210 projects only purequadratic/1D cases;
the full-quadratic branch at213–215 is unconstrained. EnsureSPD calls at56
and98 are commented out. Figure6 source selects full quadratic at
cSMC_VaryF.m:79. This differs from claiming the paper guarantees every
unconstrained fit is valid.

## Independent diagnosis contract

Compare every executed fit in rejected attempt76 (12 fits, four rows) with
independent centered full-rank SciPy SVD on exactly the saved GPU input cloud,
weights and prior policy. Stop a backward refit at its first nonpositive
precision, before computing an invalid Gaussian integral. Preserve that time,
eigenvalues, prior margin, residual rank and coefficient discrepancy.

Pass: the independent solver agrees with the original1e-7+1e-9 absolute/relative
coefficient tolerance at all compared times, identifies the same failing
row/refinement, and remains full rank. This is a diagnosis pass only.
A coefficient mismatch, different failing row, missing/corrupt input or
rank deficiency blocks the stated root-cause conclusion. Margins/rank are
explanatory and validity diagnostics, never method-quality promotion criteria.
No variance ranking, Figure6 replication, or production/default claim follows.

Command: python -B scripts/run_controlled_smc_kdm.py run
--phase c1_lorenz_precision_diagnosis (tftwogpu environment).
Output: next unique attempt under the existing campaign root, summary.json
and run manifest. GPU intentionally hidden before TensorFlow import.
Cap45 CPU seconds, one attempt. Keep inherited total and CPU500/GPU18000
allocation unchanged. This phase precedes, and does not close, batch validation.

Assumption audit: full-rank SVD is an independent mathematical comparator,
not a replacement runtime; identical inputs avoid the already-demonstrated
roundoff amplification in raw coefficients. Stopping at an invalid precision
prevents an artificial identity-covariance continuation contaminating earlier
times. Fixed failed seed is a diagnostic fixture, never a method-selection
dataset. An independent failure could instead expose a previous invalid
policy or inconsistent target; record those alternatives if observed.

Skeptical audit PASS: source requirement checked, concrete negative margin
rather than proxy loss, exact comparator inputs, unchanged tolerances,
strict validity veto and bounded cost. The artifact answers whether the
failure is genuine lack of integrability or a batch/QR implementation error.

## Next repair obligation

If confirmed, evaluate the paper's positive-definiteness repair as an explicit
Class C candidate. Derive the projection target and positive floor, retain
unchanged outputs on healthy inputs, flag every altered fit, and update both
the policy and its backward integral consistently. A covariance-only patch
would compute weights for a different proposal and is forbidden. Freeze a
dedicated non-harm evaluation before implementation/launch; do not tune the
protection to logZ variance. The author code lacks an active full-quadratic
implementation, so any explicit floor is a documented local numerical choice.
The original unprotected result remains preserved; it cannot be called passed.

## Independent diagnosis result77

All12 executed fits and36 checks passed their independent diagnosis contract.
The first failure is row0/refinement3/time index12. Independent minimum
eigenvalue is-1730.1633937155887 versus GPU-1730.1633936968644;
maximum absolute eigenvalue13056.939135274402; prior policy minimum932.8984.
The design singular ratio0.10386 is healthy. Coefficient discrepancy is at
most0.03756 of the unchanged tolerance in that failing fit. Unprotected
Gaussian integrals at this step are divergent. Times earlier than12 in the
rejected GPU fit may be contaminated by rejection placeholders and are not
treated as meaningful fitted policies. This is a real unconstrained fit failure.

## Class C candidate evaluation contract (frozen before implementation)

Use two explicit optional precision policies; retain none as the source-code
comparator and preserve its failure. No new global default is selected.

1. nearest_precision: for proposed precision P=Q+2(Aold+DeltaA), symmetric
eigendecomposition P=U diag(lambda) U^T, choose
delta=sqrt(eps64)*max(abs(lambda)) and Pnew=U diag(max(lambda,delta)) U^T.
Set DeltaAnew=DeltaA+(Pnew-P)/2, keeping raw linear and constant coefficients.
This is the unique nearest precision in Frobenius norm subject to the floor.
A covariance projection would solve a different nearness problem; precision is
chosen here because it directly constrains the quadratic coefficient and makes
the policy change explicit. The paper suggests projection but does not specify
this floor or precision/covariance choice; this is a documented implementation
of the suggested repair, not an exact author-code operation. The scale-relative
floor limits condition number to about1/sqrt(eps), so the first-order inversion
error bound eps*kappa is aboutsqrt(eps). It is a numerical accuracy choice,
not selected by likelihood variance. Already-safe fits return original
coefficients unchanged. Higham's author note is stored locally as
higham2021-nearest-psd-author-note.pdf; theorem and derivation are inspected.
The1988 full text is inaccessible at its institutional repository; do not
claim it was read. Author note:
https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/

2. backtrack_precision: a diagnostic local adaptation if projection gives an
unstable downstream trajectory. Starting at alpha=1, halve the entire
quadratic correction until Q+2(Aold+alpha*DeltaA) meets the same numerical
margin. A maximum of53 halvings bounds cost. The old policy must be valid.
Convexity of the SPD cone implies feasibility for a sufficiently small alpha
when the old policy is interior. Halving is a deterministic step search, not
a variance-tuned scalar. Scale raw and centered A,b,c together. This changes
the fitted policy and is not the literal author least-squares update.
It remains a controlled-SMC policy with the same importance target, provided
every integral/proposal/weight consumes the same changed coefficients.
A full step is used unchanged on all healthy inputs. No claim of nearestness
or source-code equality follows.

For nearest_precision, translate the added raw DeltaA into the centered
backward correction too: with x=mu+S*z and E=(Pnew-P)/2, add S E S to
centered A, 2*S*E*mu to centered b, and mu^T E mu to centered c.
Thus the backward log-integral ratio describes exactly the changed policy.
Changing covariance alone, or forgetting these linear/constant terms, is wrong.

Primary non-harm criteria: no protection fires on the previously healthy69
forcing2.5 complete replay; unchanged ancestry/stopping; forward and raw policy
agreement with the unprotected TensorFlow computation under the original
1e-7+1e-9 tolerance (also report bitwise equality). On rejected76's exact designs,
require bounded finite completed outputs, explicit protection diagnostics,
positive precision/full-rank fits and one trace per fixed graph.
Report rejection of either candidate explicitly; do not select based on logZ,
ESS, residual loss, runtime, or primary scientific performance. If only one
survives these non-harm conditions it is eligible for further correctness
validation, not promoted. If both survive, retain both without performance
ranking; prefer the paper-described projection only for the next parity check.

Vetoes: wrong target/correction identity, healthy changes beyond tolerance,
missing diagnostics, nonfinite or invalid continued trajectories, retracing.
A candidate veto triggers the other predeclared candidate; both failing triggers
a recorded new diagnosis, not a large empirical run. Missing/corrupt saved
inputs, broken derivation or budget exhaustion are continuation vetoes.
Heuristic comparisons remain deferred scientific promotion vetoes; no safety
screen can satisfy them.

Budget: c2_lorenz_precision_safety at most600 aggregate GPU seconds and two
attempts; unchanged CPU500/GPU18000 tranche and inherited total.
Extend local attempt ceiling to84 to cover diagnosis, safety, independent
repair validation, batching and pilot. Remaining allocation before this phase
exceeds3700 GPU seconds; total inherited GPU remainder exceeds32h.
Unique output roots and captured source/input hashes remain mandatory.

Skeptical audit PASS: the baseline is the actual failed unprotected fit,
healthy no-fire behavior is checked first, the floor has a numerical rationale,
both altered and unaltered policy representations are coupled algebraically,
and finite numerical behavior cannot be mistaken for replication or superiority.
Pre-mortem: projection can create enormous covariance/mean near its floor;
the full rejected-seed forward replay must reveal that. Damping may require
many refinements or give poor variance; this test cannot certify efficacy.

## Attempt78 and the next discriminating diagnosis

Both frozen candidates passed bitwise healthy replay with zero protections and
one trace per graph. Neither passed the rejected-seed non-harm screen:
nearest_precision generated nonfinite forward values at refinement3;
backtrack_precision reached refinement4 but also generated nonfinite values.
The smallest forward precision margin was respectively1.49e-8 and0.00490.
Thus positive precision alone does not establish finite nonlinear propagation.
The completed evaluation is not a passing candidate or replication.

Next phase c1_lorenz_trajectory_diagnosis reads only the saved78 histories.
Before any new protection is proposed, locate the first nonfinite cloud/weight
and compute, with an independent CPU reference, the preceding resampled cloud,
RK4 transition mean, Gaussian proposal mean and covariance, and next lookahead
mean. Compare each candidate with its own immediately preceding finite time;
do not use likelihood or ESS as a safety tuning target. Pass means both rejected
trajectories are located with finite policy inputs wherever asserted.
Missing/corrupt inputs or failure to reproduce the arithmetic blocks a causal
conclusion. All magnitude diagnostics explain failure; none promote a method.

Budget45 CPU seconds, one attempt, within unchanged CPU500/GPU18000 allocation.
Output: next unique campaign attempt, summary and manifest. CPU GPU visibility
is deliberately disabled; NumPy is an independent diagnostic only.
Command: python -B scripts/run_controlled_smc_kdm.py run
--phase c1_lorenz_trajectory_diagnosis.

Skeptical audit PASS: this examines the actual rejected forward consumer,
not only regression residuals or a positive-eigenvalue proxy. No new threshold,
data selection, performance claim, or expensive empirical matrix is introduced.

## Attempt79: projection geometry matters

Saved trajectory diagnosis locates nearest_precision failure at initial time
of refinement3: the Gaussian mean reaches2.4473869e8 and RK4 lookahead
overflows before the next state. Backtrack_precision reaches refinement4,
time2: its mean reaches411.808 (largest covariance eigenvalue0.01622),
then RK4 lookahead overflows. The precision is positive in both cases.
This rejects precision-only feasibility as a sufficient numerical safeguard;
it does not show an error in RK4 or the paper's conditional Gaussian formula.

A renewed source check distinguishes two projection geometries. The published
footnote4 names SPD projection but does not identify a norm, positive floor,
or matrix representation. Author cSMC_Resample.m:56,98 names TwistedG in
commented EnsureSPD calls, i.e. the covariance. Our nearest_precision choice
was explicitly a local hypothesis and is rejected on the non-harm question.
Projection and inversion do not commute.

One additional, frozen candidate is nearest_covariance. For symmetric,
nonsingular proposed precision P=U diag(lambda) U^T, form the signed inverse
C=P^-1, project its eigenvalues to at least
delta=sqrt(eps64)*max_j|1/lambda_j|, and set
Pnew=U diag(1/max(1/lambda,delta)) U^T.
The same conditioning rationale and Higham eigenvalue projection result apply,
now to C. A singular proposed precision is rejected, not silently ridged.
The finite positive floor remains an explicit local numerical choice.
This is a paper-motivated, source-comment-anchored repair; it is not an executed
author-code behavior, and no exact source-output replication follows.

Set E=(Pnew-P)/2 and add E to the raw fitted A; keep raw b,c unchanged.
Apply the existing exact centered-coordinate translations S E S,
2 S E mu, and mu^T E mu. All fitted policies, backward integrals, proposals
and importance weights consume that same changed policy. This avoids the
inconsistent covariance-only patch suggested by uncommenting just the source
sampling lines.

Test only this third geometry in the remaining safety attempt (CLI
--policies nearest_covariance). Preserve78 as rejected evidence; do not rerun
the already rejected alternatives. Keep identical frozen healthy69 and failed76
designs, original tolerances, bitwise report, stopping/ancestry comparison,
all-forward finite/valid requirement, no-fire healthy requirement and single
trace requirement. This is still calibration/non-harm evidence only.
If it passes, independent same-input correctness validation is mandatory;
fresh-seed batching and cost still follow. If it fails, record the cause and
do not run the empirical matrix or claim all reasonable safeguards failed.

No new allowance: remaining512.38 GPU seconds of the existing600-second safety
cap, one remaining safety attempt, overall CPU500/GPU18000 and attempt84 unchanged.
Skeptical audit PASS: source covariance target inspected, projection geometry
distinguished from precision projection, inverse singularity fail-closed,
same complete policy used by weights, healthy baseline and failed seed frozen.
The strongest risk is enormous precision and cancellation after covariance
projection; forward non-harm and independent reference must expose it.

The third geometry is registered as c2_lorenz_covariance_safety because78
is a completed evaluation. Its cap512 seconds plus78's87.623 seconds remains
below the shared600-second safety budget; one attempt. Prior phase evidence
is preserved. The dispatcher correctly refused to reuse a completed phase.

## Independent correctness validation after80

Attempt80 nearest_covariance passes the frozen safety screen: healthy69 is
bitwise unchanged with no intervention; failed76 batch completes four
refinements with finite valid outputs and one trace for each fixed graph.
Wall charge44.624 seconds. This nominates an optional repair only.

Before batching or wider execution, phase c1_lorenz_covariance_validation
compares every executed fitted policy against independent centered full-rank
SciPy SVD plus the frozen covariance projection. It also verifies the
claim-bearing forward consumer at identical inputs: reconstruct each Gaussian
proposal from the saved prior resampled cloud and fixed normal draws, and
recompute all Gaussian integrals, potentials and cumulative log normalizers
from the actual saved policy and cloud. The baseline is the same mathematical
computation with independent linear algebra; these are correctness checks,
not heuristic efficacy tests.

Primary criteria: unchanged1e-7+1e-9 absolute/relative coefficient, cloud,
potential and log-normalizer tolerances; full-rank regression; finite positive
precision; every executed refinement checked. Report maximum scaled errors
and saved failing cases without stopping at the first tolerance failure.
A tolerance failure rejects this numerical-comparison contract and triggers
localization, not a relaxed threshold or scientific promotion. Cloud-based
policy values are explanatory only and cannot erase coefficient failures.
Stochastic ranking, method superiority, defaults and full Figure6 remain
unsupported. These saved designs are repair calibration, not fresh holdouts.

Budget60 CPU seconds, one attempt, within the unchanged allocation. GPU
intentionally hidden; NumPy/SciPy strictly independent reference. Unique
summary and manifest under the campaign root.
Command: python -B scripts/run_controlled_smc_kdm.py run
--phase c1_lorenz_covariance_validation.
Skeptical audit PASS: full call chain includes altered policy, proposal,
backward correction and weights; same inputs avoid trajectory-roundoff
confounding; no proxy promotion metric, no empirical performance selection.
Pre-mortem: covariance projection can produce large precision and cancellation;
raw coefficient and forward-consumer checks may expose different limitations.

Attempt81 consumed8.031 CPU seconds and failed while serializing a diagnostic
NumPy boolean, after calculations. Cast the diagnostic verdict to a standard
Python bool and retry once under the unchanged60-second aggregate cap.
The mathematical code, designs, tolerances and evidence contract are unchanged;
syntax and the bool conversion are the focused regression. Preserve81.

## Attempt82 rejection and Class B rank-check repair

Independent validation completes16 fits/112 checks:101 pass,11 fail. Both
protected row0 refinements3/4 have substantial independently refitted coefficient
disagreement, and their fixed-policy weights/logZ exceed original tolerances.
The fourth fit's smallest design singular ratio is8.11e-9, below the required
sqrt(64 eps64)=1.19e-7. Thus80's finite safety result is insufficient, and
nearest_covariance is REJECTED for numerical correctness. Its mean/variance
quality must not be interpreted.81's serialization failure remains preserved.

The QR fitting route solves with R but checked rank using eigenvalues of D^T D.
Squaring the condition number can let roundoff hide an unresolved direction.
Class B repair: compute r_F=1/(||R||_F ||R^-1||_F) from a triangular solve and
require r_F>sqrt(64 eps64), as well as finite R and inverse. In exact arithmetic,
r_F<=sigma_min(R)/sigma_max(R); passing is a conservative sufficient condition
for the existing intended margin. This guard never changes an accepted
coefficient. Failure means numerical rank is unresolved, not proof of exact
algebraic rank deficiency. Keep the old Gram ratio as an explanatory diagnostic
and retain the new bound in fits and controller histories.

Phase c2_lorenz_rank_guard: compare old and new regression on all101 times
of all8 healthy saved70 input clouds at the same inputs; require unchanged
coefficients and no new rejection. Check the saved80 row0 third-refinement
cloud is rejected by the new criterion, consistent with independent SVD82.
Finally run the actual four-repetition covariance-repair consumer on76's fixed
design and require it fail closed. These are guard correctness checks, not
promotion of the rejected policy. Require one trace per graph and finite
recorded condition bounds. Known-good coefficient tolerance stays unchanged.

Budget240 GPU seconds, one attempt, within unchanged CPU500/GPU18000 allocation,
attempt ceiling84. Command: python -B scripts/run_controlled_smc_kdm.py run
--phase c2_lorenz_rank_guard. Unique manifests and saved summaries.
The source default unprotected algorithm remains; altered covariance policies
remain optional and rejected. No batch timing or empirical matrix may follow
without a new repair satisfying correctness.

Skeptical audit PASS: the gate tests the actual QR design and call chain,
retains accepted values, has an independent SVD rejection witness and a no-fire
healthy regression. It repairs a validity diagnostic, not a scientific
approximation. Healthy rejection, coefficient changes, missed known-bad design,
missing diagnostics, or unbounded retracing veto this implementation.

## Guard result83 and terminal localization84

The Class B guard passes37 checks: all8 healthy fits retain coefficients;
the actual covariance-repair controller rejects row0's fourth fit with
QR reciprocal-condition lower bound3.88e-9. One trace per graph.
This closes the rank-diagnostic repair only, not the policy repair.

Final bounded phase c1_lorenz_covariance_localization records every backward
time of rejected82 row0 refinements3/4 using the same independent calculation
and saved80 GPU inputs. Preserve the first backward-time coefficient mismatch,
where projection first fires, the raw precision margin, SVD ratio, correction
and target magnitudes, and coefficient discrepancy. Compare using the unchanged
coefficient tolerance. Pass means the discrepancy is reproducibly localized,
not that the candidate passes. Missing inputs/nonfinite diagnostic records
invalidate the diagnosis. No new seed, floor, efficacy target or tuning choice.

Budget30 CPU seconds, one attempt84. Together with81/82's16.32 seconds,
this stays below the originally allocated60 seconds for independent covariance
validation. The CPU500/GPU18000 tranche and inherited grant remain unchanged.
Command: python -B scripts/run_controlled_smc_kdm.py run
--phase c1_lorenz_covariance_localization.
Skeptical audit PASS: same rejected consumer and independent algebra, preserved
failures and thresholds, modest finite diagnostics, no promotion by successful
execution. End this bounded repair stage after recording the result; further
policy changes need a newly specified numerical evidence contract, not a
blind full-matrix launch. This is a numerical continuation veto for these
candidates, not rejection of controlled SMC as a research direction.

Terminal result: controlled-smc-lorenz-precision-result-20260930.md.
84 locates the first backward mismatch at the first projection, time12 and13
for refinements3/4. This stage is complete with all safeguard candidates
rejected; the validated rank guard stays. Whole replication is incomplete.

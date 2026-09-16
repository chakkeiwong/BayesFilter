# Younis score study: target-specific iAPF fit-control calibration

Status: EXECUTABLE / bounded calibration phase

Date: 2026-09-16

Parent program: `docs/plans/younis-kdm-score-master-program-2026-09-14.md`

Result root: `docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/`

## Research question

The nonlinear iAPF implementation is mechanically complete, but its first
GPU run placed three of four recursive fits at active bounds in each final
procedure and its conditional-score MSE was larger than the cheap EKF and UKF
heuristics. This phase asks whether that result is caused by an uncalibrated
Gaussian-mixture fit and adaptive-controller configuration, or whether the
proposal remains inadequate after target-specific calibration.

The quantity of interest is the model-parameter score of the finite particle
program, compared with the refined reference score on the same data-generating
model. The phase does not claim an unbiased marginal-likelihood score. It does
not claim that a fitted twist is a smoothing estimator, that iAPF replaces the
LEDH covariance lifecycle, or that a conditional-score result transfers to
degenerate DSGE transitions.

## Skeptical plan audit

The previous result used mechanics controls (`tau=100`, narrow mean and scale
bounds, a fixed floor, and a short adaptive schedule) rather than controls
selected for this model and horizon. Treating its MSE as evidence against all
iAPF twists would therefore confound proposal quality with fit failure.

The proposed calibration has the following risks and corresponding checks:

| Risk | Earliest check | Consequence |
|---|---|---|
| selection leakage | disjoint calibration, validation, and claim data IDs; frozen selection digest | invalidate claim stream and repair partitions |
| optimizer boundary solution | per-fit boundary flags, projected gradient, convergence and iteration counts | candidate is vetoed or sent to fit-optimizer repair |
| adaptive particle work hidden by equal-N comparison | record realized particle counts and fit-call work for every attempt | no equal-cost claim; compare only descriptive MSE unless cost is matched |
| small cloud or weak-curvature failure | weak and curved regimes, plus EKF/UKF/bootstrap/local-linear heuristics | candidate rejection is not research-direction rejection |
| final randomness reused during fitting | independent fit seeds and final streams; same selected fit controls for both final replicates | invalidate affected rows |
| arbitrary tuning grid | enumerate provenance and limits for every candidate; call the grid a screening calibration | no default promotion from this phase alone |
| source drift | clean frozen source revision and post-run dependency fingerprints | rerun affected stages on a fresh output root |
| numerical execution mismatch | FP64 offline fit is separately timed; final score is GPU FP32/TF32/XLA with memory-growth manifest | exclude offline fit from GPU claim timing |

The plan is not approved for scientific promotion merely because a run is
finite. A candidate can pass engineering validity and still fail the heuristic
dominance screen.

## Evidence contract

**Question.** Does target-specific calibration of the recursive iAPF fit and
adaptive controller reduce model-score error on fresh nonlinear data?

**Baseline.** The current nonlinear mechanics configuration is the frozen
baseline: `k=1`, `tau=100`, `max_iterations=4`, `max_particles=128`,
`mean_bound=4`, `sd_lower=.2`, `sd_upper=4`, `max_fit_steps=2000`,
`max_backtracks=30`, `fit_tolerance=1e-7`, and `floor_ratio=.01`.

**Candidate family.** Three fully specified candidates are screened per regime:

1. baseline controls above;
2. widened Gaussian support and tighter fit tolerances with `tau=25`,
   `mean_bound=8`, `sd_lower=.1`, `sd_upper=8`, `max_fit_steps=4000`,
   `max_backtracks=40`, `fit_tolerance=1e-8`, and `floor_ratio=.005`;
3. the widened-fit controls with `tau=10` and `k=2`.

All candidates keep `max_iterations=4`, `max_particles=128`, and FP64 offline
fitting. These are finite hypotheses, not universal defaults. The candidate
family is intentionally small enough that each candidate receives both a
calibration and a validation evaluation. The repository-owned selector
nominates the lowest calibration MSE; validation remains disjoint descriptive
evidence and is not used to choose the claim arm because this phase has no
powered validation criterion.

**Primary promotion criterion.** A selected candidate must be numerically
valid, have no unexplained fit-boundary failures, and show lower descriptive
model-score MSE than the frozen baseline on the untouched claim stream. A
promotion decision additionally requires uncertainty evidence; this bounded
phase has too few claim replicates to establish a statistically supported
ranking.

**Promotion vetoes.** Any non-finite score or likelihood, failed accounting,
stale or mismatched tuning scope, invalid memory-growth/XLA manifest, missing
fit diagnostics, selection leakage, or failure against the constructed
conditional heuristic set (EKF, UKF, bootstrap, and local-linear proposal)
vetoes promotion. A veto rejects the candidate under test, not the iAPF
research direction.

**Continuation vetoes.** Stop this phase only for corrupted data, source
fingerprint mismatch that cannot be repaired, missing reference scores, an
exhausted campaign budget, or an implementation defect that makes the target
undefined. A candidate with poor MSE continues to the planned repair review.

**Explanatory diagnostics.** Record fit boundary states, projected gradients,
iterations, selected and realized particle counts, fit-call counts, proposal
displacement, conditional heuristic MSE, model-score bias/MSE, and wall time.
These diagnose failure and do not become tuning objectives after the claim
partition is opened.

**Nonclaims.** No unbiasedness claim, no smoothing claim, no superiority claim
from descriptive differences, no equal-compute claim without measured cost
matching, no HMC/default-readiness claim, and no conclusion about degenerate
transition models.

## Data and execution partitions

Use fresh dataset IDs that have not appeared in the preceding pilot or control
phases: calibration `1000` (weak) and `1001` (curved), validation `1010` and
`1011`, and claim `1020` and `1021`. Calibration nominates exactly one
configuration per regime; validation reports a disjoint descriptive check and
does not select the arm. The selected configuration is frozen before two
independent claim replicates. Fit seeds, particle seeds, and claim streams are
recorded separately.

Use the canonical nonlinear scalar model and the existing refined reference
construction. The final score kernel remains the claim-bearing analytical
finite-program derivative; autodiff is parity-only. Final numerical work uses
GPU FP32/TF32/XLA with verified memory growth. Offline fit optimization and the
CPU FP64 reference remain separate lanes and are not included in GPU kernel
timing.

## Budget and stopping rule

Allocate at most two GPU launches, 64 charged attempts, and 900 GPU seconds
for this phase, with at most 5,400 CPU seconds. The runner must stop before
exceeding any bound and write a partial result if a bound is reached. Every
launch gets a new output directory; prior evidence is never overwritten.

The candidate count and adaptive schedule are chosen so the expected charge is
below the 64-attempt bound. If observed fit-call counts exceed that bound, the
run is an infrastructure/campaign-budget failure and is repaired by reducing
the calibration arm only after preserving the failed artifact; it is not
silently reinterpreted as a scientific result.

## Phase procedure

### Recovery audit after launch 1

Launch `iapf-fit-calibration-gpu-01` spent 32 charges and 31.74 process-wall
seconds. Weak calibration and both selected claim rows completed; four
comparator rows computed but failed result validation because the study-level
evidence class contradicted their mechanics role. Preserve all failed attempts.
This is a harness failure, not evidence against any proposal.

Launch 2 may reuse only the eight completed weak rows after source, study,
result-digest and selection revalidation. The repaired comparator study is
explicitly mechanics evidence. A driver wrapper admits each new numerical row
only if the cumulative budget can cover one row plus its maximum possible
recursive fits. Successful calls charge measured fits; unknown failed calls
charge the upper bound. A budget refusal creates a coordinator interruption
record but performs no numerical work and spends no numerical attempt.

Only 32 charges remain. The measured 2/3-fit pattern would allow repaired weak
comparators plus curved calibration and claims, but not curved comparators.
The driver must preserve a partial result when further rows cannot be admitted;
it must not assume curved fit counts match weak counts. No third GPU launch is
authorized by this allocation. Bound launch 2 externally to 860 wall seconds,
which leaves room below the cumulative 900-second limit. Driver tests use fake
endpoints and are CPU-only engineering checks.

The audit also found a scientific coverage gap: only the selected iAPF arm is
evaluated on claim data. The frozen iAPF baseline is measured on calibration
and validation data, not on the same claim stream. Consequently this phase
cannot establish a calibration benefit relative to that baseline, regardless
of comparator outcomes. Preserve the selection rule and opened data; schedule
the missing baseline comparison and all-control calibration on fresh data in
the phase refresh. The existing two-replicate design cannot establish a
statistical ranking either. These limitations block promotion, not execution
of the remaining informative, bounded rows.

The focused audit permits this recovery because target, controls, partitions,
selection rule, hardware class, numerical implementation and total budget are
unchanged; source verification and a pre-numerical budget guard close the
localized harness failures. The baseline-coverage gap is retained explicitly.

### Execution sequence

1. Verify the clean frozen source revision, dependency fingerprints, Python
   environment, and available GPU memory policy. Run the focused import and
   scope tests with CUDA hidden before the GPU launch.
2. Generate the six fresh partitions and record their data digests.
3. Run all three candidates on calibration and validation data. Issue one
   repository-owned selection artifact per regime, including the exact scope,
   candidate family, both calibration and validation tables, the calibration
   nomination criterion, and selected controls.
4. Run two independent claim replicates per regime using the frozen selection
   artifact. Run EKF, UKF, bootstrap, and local-linear heuristic comparators on
   the same claim data, with realized particle count and work recorded for the
   iAPF rows.
5. Validate manifests, source/data/driver digests, accounting, fit diagnostics,
   memory growth, device placement, and score finiteness. Assemble model-score
   and conditional-score tables with an inference-status table.
6. At the phase boundary, repair any localized harness or serialization issue
   under the unchanged contract, rerun only the affected stage in a fresh
   output directory, and write a result-and-refresh note. Refresh the master
   program with the candidate verdict, the strongest alternative explanation,
   remaining uncertainty, and the smallest next discriminating experiment.

## Interpretation and next action

If the calibrated candidate passes validity but remains worse than a cheap
heuristic, record a candidate rejection and schedule a particle-count or
observation-aware proposal repair; do not reject the whole iAPF direction. If
boundary rates fall and MSE improves, schedule a larger multi-seed claim phase
with a predeclared cost-matched comparison. If all candidates fail fit
validity, repair the optimizer and reference accounting before comparing score
quality. In every case, preserve the distinction between engineering
correctness, numerical validity, and scientific interpretation.

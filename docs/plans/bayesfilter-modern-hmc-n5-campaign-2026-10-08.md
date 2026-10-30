# N5 continuation: funded mechanics completion and posterior comparisons

**Active continuation, refreshed October 10:** this file preserves the original
72-hour campaign design and dated outcomes. The cumulative grant is now120
aggregate worker-hours under the [48-hour supplement](bayesfilter-modern-hmc-n5-additional-48h-2026-10-10.md).
D2f, D3, E3 and D4a are terminal. All sixty proposal prices completed. The
[D4 executable plan](bayesfilter-modern-hmc-n5-method-proposals-2026-10-10.md)
now governs D4b1, the complete SGQF SNAPER/MAMS proposal sets in
`N5/method-full-fit-01/`, with observer `N5/method-full-fit-watch-01/` and automatic
trusted GPU selection between cells.
The [master](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md) and fresh
twenty-cell matrix `N5/method-full-fit-audit-01/nonlinear-comparison-matrix.json`
are the current status. Earlier status/allocation paragraphs below are snapshots.

**October 10 scope correction:** NUTS is excluded, including diagnostic
comparators. The current inventory has five models and four methods (fixed HMC,
randomized HMC, SNAPER-proposed HMC and MAMS). Earlier NUTS plans, commands and
results below are historical records; they must not be rerun or treated as
remaining work. Prior costs and original fit/member denominators are preserved.
See the [exclusion repair](bayesfilter-nuts-exclusion-repair-2026-10-10.md).

8 October 2026. This is the active execution supplement to the
[modern HMC program](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md).
The user supplied **72 additional hours** and requested continuation. Interpret
this conservatively as 259,200 aggregate worker-wall seconds across CPU and GPU,
including startup, failed attempts and retries, rather than 72 hours of each.
Run at most two GPU workers on distinct physical devices: SIR recovery and
trajectory pricing used GPU 0 before D1; LGSSM/accumulation replication used
GPU 1. Both require fresh
trusted memory telemetry and growth. This expands the initial one-worker
schedule after shared-device pricing showed SIR would take hours, allowing
independent short-model work while it progresses. Charge overlapping worker
seconds additively. Prior engineering expenditure
remains in its original ledger; it is not reset or charged a second time.

**Historical state, 9 October:** D0/D1/E0/E1 are terminal, with their last
worker finishing at 01:08:59 China time. D0 passed all 32 complete fits and
64 member assessments. D1 completed both full PP-SGQF methods: both L=9
members passed posterior/reference checks, and both L=3 members reached the
retained-count cap with insufficient precision. E0 qualified the PP-UKF
reference; E1 left the structural reference limited on source-square rho.
E2/E2b reproduced both selected structural scrambles. E2c resolved their local
finite-difference discrepancies on all 128 saved points; the reference precision
limitation remains. D2a completed all sixteen UKF/SIR GPU price calls without
numerical vetoes. D2b finished UKF fixed (L=9 passes posterior/reference; L=3
precision-limited), then stopped before randomized sampling on a GPU preference
race. **D2c has restarted only UKF randomized and the two unstarted SIR fits,
using fresh trusted device selection for each fit.** Its first fit is tuning on
GPU 1 with verified memory growth. The current ledger charges 31.62 of the
original 72 worker-hours, reserves 26.83 hours for D2c, and leaves 13.38 hours
unallocated after the unchanged 600-second overhead reserve. Actual occupied
time replaces reservations at settlement; unused caps are released once. See the
[terminal audit](bayesfilter-modern-hmc-execution-2026-10-07.md#n5-batch-terminal-audit-9-october-2026)
and the continuation sequence at the end of this plan. Earlier phase sections
preserve the designs and observations at their launch dates.

## Question and evidence

Can the implemented corrected transitions deliver accurate posterior quantities
on the repository's state-space models at a useful complete-fit cost? The first
comparison is prepared fixed HMC versus prepared randomized HMC, with identical
data, target, geometry, start blocks, qualification policy and posterior accuracy
requirements. SNAPER is compared with simple randomization; MAMS is qualified
separately. The current five-model by four-method inventory excludes NUTS.
Missing cells for the approved methods remain explicit.

The immediate SIR task is narrower: finish the frozen mechanics cohort from
attempt 14 and export/reload both independently verified members. This does not
establish useful SIR mixing. R-hat, ESS and MCSE cannot affect tuning membership,
ranking, repair or stopping; they belong to separate posterior assessment.
The declared replicated acceptance bound still qualifies tuning candidates.

| Observation | Role |
| --- | --- |
| Complete fresh measurement/verification and all-member replay | Mechanics pass criterion |
| Independent posterior agreement including reference and Monte Carlo uncertainty | Posterior qualification criterion |
| Replicated accuracy-qualified total fit costs and delivery frequency | Efficiency evidence, with uncertainty required for ranking |
| Nonfinite state/score, wrong correction, invalid filter status | Candidate veto and localization/geometry repair trigger |
| Wrong target/source/stream, invalid reference, corrupt checkpoint | Affected experiment continuation veto until repaired |
| Budget exhaustion | Continuation veto for further launches; preserve partial results |
| Short healthy paths, acceptance beyond its declared bound, timings | Explanatory/pricing only |

No short-run pass certifies burn-in, convergence, the exact nonlinear generative
posterior, method superiority or release readiness. UKF and SGQF likelihoods
define different numerical targets. A failed candidate triggers the relevant
repair; it does not reject its sampler family.

## Sequence and allocation

These are convenience ceilings, not runtime predictions. Unspent allocations
may be reassigned in the execution record within the unchanged 72-hour total.

| Phase | Ceiling | Required output and next decision |
| --- | ---: | --- |
| A. SIR saved-cohort completion | 3.5 hours | Completed in 2.47 hours; unused half-hour reassigned to E1 after settlement |
| B. Complete-fit harness and pricing | 6 hours | Prepared LGSSM/accumulation fits including short-model comparators, independent references, actual warm-up/precision costs; regression tests |
| C. Model-specific references/geometry | 6 hours | Reference refinement and uncertainty; target/score checks; declared estimands and geometry per target |
| D. Priced nonlinear comparison | 45 hours | Fresh replicated cells, full denominator and dispositions, uncertainty-aware paired comparison; includes D2b |
| E. Localized repair reserve | 8.5 hours | E0/E1 actual costs and E2/E2a/E2b, with no nested double charge |
| F. Terminal audit and documentation | 3 hours | Reconciled budget, source/data/runtime evidence, result tables and refreshed master |

Before each newly designed research stage, record its actual settings, numerical
provenance, price, attempt count, stop conditions and exact executable command
here or in the execution record. This is phase planning, not repeated permission.
Do not spend the comparison allocation on unpriced full sweeps. Implementation,
small checks and local infrastructure repair proceed under the user's request.
Maximum Phase A attempts: three; the current allocation table governs its
cumulative charge. Earlier attempt caps and reallocations remain recorded
below as execution history, not additional available time.

## Phase A: unchanged-scope SIR recovery

Original checkpoint:
`artifacts/modern-hmc-implementation-2026-10-07/N5/attempt-14-sir-gpu-local-step/cells/nl-sir/tuning_checkpoint.json`.
All 147 bound source files match at preflight. Preserve the original directory,
copy the checkpoint and immutable evidence to a fresh attempt, and use the
public resume API. Recheck source, target, runtime and evidence on reload.
Use the existing supervised `execution_budget` wall replacement; do not edit
the serialized numerical config, elapsed accounting, random streams or old cap.
The parent enforces an additional 3,600-second occupied-wall cap, including
startup/export. Its child allowance leaves 30 seconds for process/report cleanup
(a convenience reserve, not measured scientific work).

The original full horizon, four starts, epsilon 0.001682313009806192, maximum
lengths 2/3, 64-transition trials and 16+16 independent repetitions remain fixed.
Their provenance and limitations are in the
[bridge design](bayesfilter-modern-hmc-n5-bridge-2026-10-08.md).
GPU/XLA is required; match recorded device scope and configure/verify memory
growth before initialization. The trusted probe is fresh. Estimated remaining
cost is 25–40 minutes from measured 24–31 second trials, with load uncertainty.
The first launch was rejected before TensorFlow because the readiness helper
requires an idle GPU. Trusted telemetry showed GPU 0 has about 31 GiB free but
another compute process. Shared admission is appropriate under the existing
resource-sharing policy: keep the same physical UUID, require 4 GiB observed
free memory (a conservative convenience screen, not a capacity guarantee),
configure memory growth and preserve the original idle-probe rejection. Only
the explicit busy-device rejection permits this fallback; driver or visibility
failures do not. Contention consumes this same funded hour. Attempt 15 consumed
0.172 seconds; attempt 16 uses at most 3,599 seconds. No other process is stopped.

Observed new trial times on the shared GPU are 199–224 seconds, versus the
earlier 24–31 seconds. This is measured occupied-wall pricing, not proof that
contention explains all of the difference. Trial progress and finite results
continue. Allocate three more hours from C to A within the unchanged 72-hour
total. Let attempt 16 finish its current allowance; if incomplete from that cap,
attempt 17 resumes its new checkpoint under at most 10,799 further seconds.
Preserve both lost-call charges if a native trial is interrupted again. No
numerical or posterior failure is eligible for automatic timeout recovery.

The interrupted twentieth call remains charged. A retry may use the same stream
only because that trial produced no committed evidence. The raw audit must
distinguish multiple attempted charges from the one committed returned chunk;
neither erase the interrupted cost nor mistake it for an extra independent trial.
The retry must not run if live source has changed. A new source requires a fresh
scope and an explicitly recorded restart, without reusing historical qualification.

Command (trusted GPU execution, after the focused launcher checks):

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/resume_modern_hmc_bridge.py \
  --checkpoint docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/attempt-14-sir-gpu-local-step/cells/nl-sir/tuning_checkpoint.json \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/attempt-16-sir-shared-resume \
  --max-wall-time-seconds 3599
```

## Skeptical audit before execution

The saved source hashes match, but a fresh runtime reload must still pass. The
old 10-minute stop is an allocation failure, not evidence against the smaller
step. The old raw auditor rejects duplicate charge keys and its success helper
requires zero interrupted charges, so it is unsuitable unchanged for a recovered
run. A narrowly scoped recovery audit will preserve every attempt while checking
the unique committed trial inventory with the existing native auditor. Regression
tests must expose duplicate committed evidence, inconsistent retry charges and
unrecorded costs. No numerical kernel or acceptance rule needs to change.

The major scientific risk remains good acceptance from trajectories too short
to explore the posterior. Phase B therefore requires prepared geometry, separate
warm-up/retained assessment and independent references before any usefulness
claim. Existing LGSSM campaign references must not be reused for the generic
LGSSM fixture without proving equality of data, priors, parameter chart and
likelihood. These constraints resolve the wrong-baseline and proxy-promotion
risks. Proceed with Phase A and implementation/pricing of B; freeze the later
scientific designs from those measurements before spending C/D.

## Run records

Every attempt writes command, Git/source hashes, environment, seeds, actual data,
device/XLA/TF32/memory policy, start/end times, returned/attempted work, result and
plan paths. Supervisors retain failure/timeout records and checkpoint progress.
The execution record carries decisions, inference status and remaining budget.
No current result is a new default or a public release.

## Phase B design: first complete fits

Scope is the actual generic LGSSM fixture (four observations, two parameters)
and full generic accumulation fixture (three observations, three parameters).
These first short-horizon fits price the public lifecycle; they are not the
longer-horizon/difficult-regime comparison. Keep all verified members and assess
each separately. There is no selection by observed ESS, acceptance or cost.

Geometry is the stated parameter prior: identity for LGSSM and the Gaussian
prior's diagonal factor `(0.20, 0.10, 0.20)` centered at `(0.70,0.25,0.80)` for
accumulation. These are target-specific, data-independent preparation hypotheses,
not estimated posterior covariances. The comparison gives both methods identical
geometry. Four starts use fixed contrasting standardized offsets in the range
[-2,2], with a small independently seeded common perturbation per replication.
Nonfinite starts veto that fit. No learned transport is trained in this phase.

Both fixed and randomized full-refresh HMC initially test maximum counts 3 and
9 at epsilon 0.5 in these coordinates. This is a convenience pilot motivated
by order-one prior curvature, not transferable scientific tuning defaults.
Use the same experimental v8 lower acceptance bound 0.5, 32 measurement and
32 fresh verification trials, 64 transitions each, no discarded trial prefix,
equal start weights, and family alpha 0.05. The low boundary concerns tuning
validity only. At perfect acceptance the maximum constant-bet growth is
`1.5**32`; with four equally weighted bets it exceeds the candidate-wise
threshold. Power at acceptance near 0.5 is not assured. Record inconclusive
cohorts without weakening thresholds. Per fit: two candidates, no repair or
refinement in the pricing design, 20 scheduler units and 600,000 conservative
work units. Required work is `32*64*4*2*(8+20)=458752` units; reserve accounts for
interrupted attempts. Later broad tuning is a separately priced design.

Use the common sequential posterior controller on every verified member:
2,000 minimum warm-up draws, latest 1,000 window, 1,000-draw chunks, R-hat 1.05;
10,000 maximum warm-up and 10,000 retained draws per chain. These are inherited
operational policy, not stationarity guarantees. Retained R-hat is 1.01, bulk
and tail ESS floor 400, mean lugsail MCSE/SD <=0.02, with the existing 20-batch
minimum, r=3 and c=1/2. The ESS floor is a conventional operational screen; the
MCSE tolerance is a declared pricing accuracy target. Monitor raw/model
parameters, standardized second moments and smooth upper-tail probabilities
`sigmoid(2*(z-1))`. The latter detect some scale/tail disagreement without an
unstable rare binary threshold; they do not establish all-tail correctness.
Store every warm-up/retained chunk. Stop at the declared count/time limit or
hard numerical veto, preserving the unfinished member and the full denominator.

Independent diagnostic references reconstruct the same scalar Kalman law and
full three-parameter eigen-UKF law, including timing, initial covariance,
observation jitter, parameter prior and sigma-point orientation. The previous
K7 oracle fixes one parameter and is not this full posterior. Compare values
and central-difference scores at prior-center/interior/tail probes first.
FP64 tolerances 1e-8 for values and 2e-5 for scores are diagnostic hypotheses;
check two difference spacings before attributing discrepancy to a score defect.
Tensor-product Gaussian quadrature uses orders 25,41,65 against the exact
Gaussian prior measure; require successive mean changes below 0.002 reference
SD and record their maximum as a numerical sensitivity estimate, not a rigorous
error bound. A failed refinement vetoes posterior accuracy claims and triggers
more reference work. Reference code is NumPy/SciPy diagnostic-only; it is not
imported by the sampler or used to prepare geometry/admit candidates.

After separate posterior stopping, compare all declared means with the reference
using `abs(error) + 3*MCSE + reference_sensitivity <= 0.10*reference_SD`.
This is a predeclared numerical agreement screen, not an anytime confidence
statement; its repeated-stopping coverage remains to be tested. Passing the
screen permits further replicated validation, not release/default promotion.
Four initial fits (two models by two methods), seed block 20261008/1000, at most
1,800 occupied-wall seconds each, preserve exact costs even if a fit is partial.
One seed supplies descriptive pricing only. Fresh replicated confirmation and
failure-rate uncertainty must follow before a ranking. The fit supervisor will
use trusted GPU/XLA, shared admission when appropriate, memory growth, explicit
CPU-only reference mode for focused tests, and fresh output directories.

Pre-mortem: these priors may leave substantial posterior dependence; a favorable
short chain could miss it. All-member posterior assessment plus independently
refined means/second moments/tails distinguishes that failure from successful
mechanics. New references can have their own orientation or timing error;
pointwise parity is required before interpreting integrals. No speed comparison
is made from these four pricing fits. Audit verdict: proceed with reference
parity, focused pipeline/recovery tests, then the bounded complete-fit ladder.

The independent reference parity tests passed for both targets; three quadrature
orders passed the declared refinement screen. Results are in `N5/reference-01/`.
The four-cell ladder is executable as:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_modern_hmc_pricing.py \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/complete-fit-pricing-01 \
  --references docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/reference-01 \
  --gpu 1
```

## Phase C0: local geometry diagnostics on the full nonlinear adapters

The short fits do not price the remaining full-horizon targets. Before starting
their chains, diagnose local affine preparation for NL-STR, NL-PP-UKF,
NL-PP-SGQF and NL-SIR on their existing full datasets (no horizon substitution).
The question is whether the inspected score reaches a stationary point and
gives reproducible positive local curvature. This nominates geometry only;
it cannot establish posterior mass coverage, global modes or a usable sampler.
The comparator is the existing prior-coordinate identity/scale, not a learned
transport. The output will decide whether the next GPU step can price an affine
pilot or instead needs geometry/source repair or a learned-map design.

Use the existing checkpointed batched TFP L-BFGS on the four finite mechanics
starts. These narrow starts are a local diagnostic, not a multimode search.
Source priors are standard normal under the six-/five-probit Uniform charts
for predator-prey/structural; SIR has Gaussian scale 0.5. Optimize in those
prior units, with at most 100 iterations, 400 batched evaluations, checkpoints
every five iterations, ten secant pairs and gradient infinity tolerance 1e-6.
Iteration/evaluation limits are convenience pricing caps; the tolerance is the
existing optimizer diagnostic default. Preserve endpoints, secant state and
actual calls. All in-support nonfinite or filter-status failures abort the
diagnostic rather than silently treating numerical failure as prior exclusion.

At a stationary center compare negative-score central differences at prior-
standardized steps 1e-4 and 1e-5. Each column is evaluated in a batched traced
target call; no pfor or scalar-row map is introduced. Record antisymmetry,
successive-matrix difference and eigenvalues before symmetrizing for the
candidate factor. A relative Frobenius difference/asymmetry above 1e-3 is an
explicit diagnostic tolerance hypothesis and vetoes the local factor; nonpositive
eigenvalues also veto. No eigenvalue clipping or ridge is introduced. If both
checks pass, factor the local inverse curvature and verify the whitening
identity; store it as an unadmitted proposal. A bad score/Hessian cannot become
good geometry merely because the optimizer returned.

This first diagnostic is an explicit CPU-only reference/debug exception, with
CUDA hidden before import, two CPU threads and stable `tf.function` signatures
without enclosing XLA. It cannot supply GPU/HMC admission or performance claims.
At most 900 wall seconds per model, sequentially, from Phase C's nine hours.
CPU work may overlap the two GPU workers, with additive charging. Later GPU/XLA
checks and independent posterior references are mandatory before sampling
claims. Native exceptions and cap stops preserve partial checkpoints; no
automatic numerics-altering repair is authorized by this geometry check.

Skeptical audit: all four targets retain their full original prior, data, chart,
likelihood and score. A local curvature pass can still miss a funnel or remote
mode; it only motivates a controlled affine comparison. The four starts cannot
test globality. Independent score parity was checked in the preceding mechanics
phase; local two-spacing differences additionally check curvature consistency.
The smallest useful failure is a saved nonstationary/indefinite/status result,
which triggers preparation repair without rejecting the sampler family. Audit
passes for this bounded diagnostic after analytic regression checks.

The three analytic curvature regressions passed: correlated anisotropic factor
orientation, indefinite saddle rejection without clipping, and asymmetric-score/
invalid-status rejection. Execute the four full-model diagnostics with:

```bash
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_geometry.py \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/full-model-geometry-01
```

The parent preserves failures and enforces 900 seconds per child; the child
checks its 870-second working allowance between optimizer chunks. Status or
localization failure on one model does not invalidate the independent models.
Neither a local factor nor an optimizer's selected incumbent is an admitted
posterior covariance. No learned transport training occurs in this diagnostic.

C0 returned local factors for full structural, PP-UKF and PP-SGQF in 20.92,
25.94 and 38.99 seconds. SIR failed during unbounded L-BFGS trial evaluation
after 22.24 seconds with an invalid eigendecomposition. The initial finite
starts remain usable and the independent running tuning checkpoint is intact.
The repair diagnostic keeps the target and four starts fixed but scales the
optimizer's coordinate chart by
`min(1,0.25/sqrt(max(abs(eigenvalues(H_prior(0))))))`, with H estimated at both
declared finite-difference steps. This makes the maximum absolute quadratic
curvature in optimizer coordinates 1/16 at the diagnostic anchor, reducing
the initial unit quasi-Newton step's local overshoot. It is a local hypothesis,
not a bound away from that anchor. Adjust the solver gradient tolerance by the
same multiplier and still require the original prior-scaled score norm <=1e-6;
do not turn a smaller optimizer chart into an easier stationarity criterion.
No objective, parameter support, covariance floor or score changes. A further
failure ends this repair diagnostic and triggers a separate bounded optimizer/
learned-geometry design; do not loop on smaller numbers without evidence.

This additional SIR-only attempt has the same 900-second cap under C:

```bash
env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_geometry.py \
  --targets NL-SIR --curvature-scaled \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/full-model-geometry-02-sir
```

The correlated quadratic regression supplies the chart/Hessian transformation
check; both probe spacings and unchanged prior-gradient tolerance are checked
in the actual diagnostic. This is an optimizer-scale repair only and has no
sampler-admission authority. Skeptical audit passes for that bounded question.

## Phase C1: independent full structural posterior reference

The full structural local factor is available. Build an independent NumPy/SciPy
diagnostic of the same five-probit UKF posterior, using all 100 observations.
The structural transition is `m'=rho*m+sigma*e`, `k'=phi*k+gamma*m'^2`, and
`y=m+k+noise` with observation variance R. Uniform physical priors and the full
probit Jacobian give a standard-normal density in all five source coordinates.
Assimilate y0 before the first transition, starting from covariance diag(.04,.09).
Use the declared seven augmented UKF points with central mean weight zero,
off-center weights 1/6 and central covariance weight two. Reconstruct the
principal square root independently: for SPD 2x2 P,
`sqrt(P)=(P+sqrt(det(P))*I)/sqrt(trace(P)+2*sqrt(det(P)))`, and append independent
unit process noise. This agrees with the inspected principal-root route;
substituting Cholesky would generally change this nonlinear numerical target.
No floor or covariance repair is allowed in the reference. Invalid covariance
vetoes the reference rather than becoming a zero-weight posterior point.

Before integration, compare independent values and two-spacing finite-difference
scores at the prior center, local center and two coordinate perturbations.
Use the earlier FP64 diagnostic tolerances (1e-8 values, 2e-5 scaled scores).
After parity, integrate with eight independent scrambled Sobol replications,
total points per replication 2^10, 2^12 and 2^14. Use a fixed equal mixture of
the N(0,I) prior and N(local center,4*local covariance). Evaluate the exact
mixture density in importance weights and stratify half the points to each
component; the proposal cannot alter the target. The inflation and equal
weights are coverage hypotheses; the prior component covers the complete
source support and the local component addresses concentrated likelihood.
Separate component scrambles and between-replication variability are saved.
The native sampler never imports this reference or uses its outcomes to tune.

Monitor physical parameters, prior-standardized source squares and
`sigmoid(2*(q-1))` tails. Require last-order mean changes and pointwise 95%
Student half-widths across scrambles each <=.002 estimated posterior SD, the
previous reference sensitivity scale. Record weight concentration, component
contributions, all replicate means and normalization estimates. These are
empirical refinement and randomization diagnostics, not rigorous cubature
error bounds or simultaneous confidence guarantees. A failed screen means
reference-limited, not posterior disagreement. The first 2^10 rung is the
price; stop at a 1,800-second parent cap, preserving all finished rungs.

Skeptical review: this is a full five-parameter integral, not a conditional slice.
The principal-root identity follows by Cayley--Hamilton and its positive branch;
tests compare it with an independent eigensquare root and use native value/score
parity at nonzero points. The Gaussian-mixture importance ratio includes both
proposal components at every point. Short QMC replication can still underestimate
tail/integration error; no release/default claim follows. Observation R is
bounded below by .02, so the finite-horizon likelihood is bounded above and
Gaussian-prior polynomial moments exist. The exact nonlinear-generative
posterior remains outside this reference's claim. Audit passes for parity,
then the bounded reference ladder under Phase C's remaining allocation.

Executable command (CPU-only independent reference, 1,800 seconds total):

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_structural_reference.py \
  --geometry docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/full-model-geometry-01/nl-str \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-reference-01
```

Prelaunch implementation review: all fifteen quantities must pass both empirical
checks; one rung, nonfinite uncertainty or missing quantities cannot pass. The
runner binds source/data/proposal checksums, rechecks parity at the actual local
center and perturbations, writes every completed rung and enforces the parent
wall cap. A known normalized Gaussian tests the mixture-weight normalizer;
the previous fixture failure came from SciPy's default one-dimensional normal
being applied to five-dimensional points, and has been corrected explicitly.
The current GPU source closures are untouched. The first reference is not
used for sampler configuration or scientific promotion.

### C1 repair: freeze a more flexible independent integration proposal

The first ladder finished in 41.12 seconds and passed value/score parity.
It failed refinement: the largest final change was .0752 posterior SD and
the largest pointwise half-width .1068, versus the unchanged .002 requirement.
The lowest importance ESS among the eight 16,384-point scrambles was 108.3.
This invalidates its use as an accurate reference, not its target formula or
any sampler. The current reference proposal is inadequate; simply increasing
the same grid is poorly priced by its observed concentration.

Repair only the independent importance proposal. Use an equal eight-component
Gaussian mixture: one fixed N(0,I) component and seven components fit to weighted
independent target evaluations. The prior component keeps full support. For
each draw, evaluate the full mixture denominator, including all eight terms;
then w(q) is proportional to exp(log posterior(q)-log proposal(q)). With equal
stratified component counts, this is the deterministic-mixture importance
identity. Proposal fitting cannot change the numerical posterior.

Two discarded proposal-fitting rounds use 32,768 points each. Weighted PCA
quantile groups initialize seven components; twelve weighted responsibility/
moment updates fit them. Covariances use .9 within-component covariance plus
.1 global weighted covariance; this is an explicit proposal regularizer, not
a ridge of the target/filter. Mixture weights remain equal. The component
count, two rounds, twelve updates and .1 shrinkage are convenience hypotheses
for a small five-dimensional reference, not repository defaults. Report weight
ESS, largest weight, normalizer and weighted proposal log density; covariance
or target failures veto the reference. Training random streams are distinct
from final integration, and the proposal is frozen before final estimation.

Fresh validation uses eight scrambles at powers 14, 16 and 18 with the same
fifteen quantities and unchanged .002 refinement/uncertainty criteria. The
41-second initial run prices likelihood evaluation; cost scales roughly with
point count, while mixture overhead is measured at the first rung. The parent
allowance remains 1,800 seconds, all charged to C. Preserve completed rungs on
timeout. A refinement failure remains reference-limited and triggers inspection
of coverage/weight concentration, never threshold relaxation or sampler tuning.

Skeptical review: fitting and judging the proposal on the same points would
hide overfitting, so validation uses fresh scrambles and freezes all components.
The weighted EM updates only construct an auxiliary density; positive normalized
densities and the complete importance denominator are tested independently.
Equal component proportions and covariance shrinkage need not fit every mode;
the prior component and saved weights diagnose this but cannot certify global
coverage. Self-normalized estimates can be biased at finite order; successive
refinement and replicated uncertainty remain empirical checks. This repair
does not import a reference into tuning, alter a GPU source closure, promote
an HMC method, or claim the exact generative posterior. Audit passes for the
bounded independent-reference repair after focused mixture regressions.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_structural_reference.py \
  --geometry docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/full-model-geometry-01/nl-str \
  --proposal-refinement --powers 14 16 18 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-reference-02
```

The coordinator allows 7,800 seconds total (four 1,800-second fits plus four
120-second independent assessments and cleanup). A harness/reference failure
stops the ladder for diagnosis. A completed but unfavorable posterior result
stays in the denominator and does not prevent the next independent comparison.
Longer/high-dimensional models remain unfunded for execution until priced and
their scientific settings/reference limitations are declared; their allocated
campaign budget remains available for that next phase.

MathDevMCP's AST Kalman audit reported missing prediction/inverse operations:
inspection shows scalar prediction in `mean=.02+rho*mean` and
`covariance=rho*rho*covariance+.07`; scalar gain division supplies the solve.
Its generic covariance-guard detector did not recognize the positive scalar
recurrence or the nonlinear eigenvalue guard. These are tool-pattern limitations,
not waived mathematical obligations. Its symbolic checker proved
`P*R/(h*h*P+R) = P-(P*h)**2/(h*h*P+R)` under positive P,R. The direct value and
two-spacing score comparisons supply the numerical evidence; the AST audit
does not certify the implementation. Saved checks are in `N5/campaign-72h-audit/`.

### C1 priced final refinement of the frozen proposal

The repaired reference completed in 501.99 seconds. Final order 2^18 still
fails the .002 criterion: maximum half-width .00704 and last-order mean change
.00255 posterior SD. Minimum importance ESS is 95,422/262,144 and the largest
normalized weight .00175. Proposal concentration is much reduced, but source
square/tail moments still need more integration evidence. Do not interpret
this partial improvement as a qualified reference.

Freeze and reuse exactly `structural-reference-02/reference/frozen-proposal.json`;
there is no further fitting or threshold change. A new independent seed family
40361008 supplies eight scrambles at powers 18, 20 and 22. Charge at most
10,800 seconds to C, including startup and parity. The measured final rung
scales to roughly 2.2 hours for this ladder under similar CPU load; that is a
cost estimate, not a convergence guarantee. Across-scramble uncertainty at
power 18 suggests power 22 may reach .002 under square-root scaling. This
scaling is a pricing hypothesis, not proof about QMC variance or rare tails.
If the final criterion fails, preserve reference-limited status and inspect
which quantities/replications remain responsible before spending more.

Skeptical audit: reuse of a fixed proposal with fresh validation scrambles keeps
the target/estimator unchanged; it neither tunes HMC from a failed reference
nor uses retained draws to learn a reference. Record the proposal input hash,
source snapshot, seed, every rung, all parity receipts and the updated actual
budget. The extra allowance fits within C's nine hours and the campaign total.
This is a predeclared fixed-order check, not stopping after an attractive random
estimate. Audit passes for this measured continuation.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_structural_reference.py \
  --geometry docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/full-model-geometry-01/nl-str \
  --frozen-proposal docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-reference-02/reference/frozen-proposal.json \
  --reference-seed 40361008 --powers 18 20 22 --wall-seconds 10800 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-reference-03
```

## Phase C2: independent predator–prey target parity

Before integrating a six-parameter posterior, independently reconstruct both
declared numerical likelihoods on the full twenty-observation dataset. Native
source anchors are `predator_prey_ukf_neutra_target_tf.py` (parameter box,
initial update and twenty 0.1 RK4 substeps per transition),
`predator_prey_sgqf_neutra_target_tf.py` (additive-noise, Cholesky, level-two
SGQF), and `fixed_sgqf_tf.py` (level rule and Smolyak combination). These are
checks of the named numerical targets, not Zhao–Cui production-route admission.

Use the full six-probit Uniform chart and Jacobian, hence independent standard
normal source priors. Assimilate y0 with prior mean (50,5), covariance I and
observation covariance 4I. UKF uses the principal root of blockdiag(P,4I),
nine augmented points, central mean/covariance weights 0/2 and off-center 1/8.
SGQF uses the Cholesky of P with the level-two two-dimensional Smolyak rule:
Q3xQ1 + Q1xQ3 - Q1xQ1, equivalently the origin weight 1/3 and four axial
points at +/-sqrt(3) with weight 1/6. Add process covariance 4I, then the
linear identity observation permits an ordinary Gaussian update. Use that
independent closed-form update instead of copying the native score recursion.

Compare values and two-spacing finite differences at zero, small opposing
coordinate offsets, and each saved local center. The inherited FP64 check
scales are 1e-8 value error and 2e-5 scaled score error. Full-horizon parity and
rule-moment regressions are CPU-hidden diagnostic exceptions, at most five
minutes total. No parameter projection, positivity clipping, covariance floor
or singular-denominator replacement is allowed. Nonfinite paths are failures
of the reference/target domain check, never silently assigned zero mass.

Skeptical audit: principal-root and Cholesky sigma points are intentionally
different between the two numerical likelihoods, and observation timing cannot
be borrowed from a source-route experiment with y1:y20. Matching local values
does not prove global numerical validity or posterior integration. The smallest
useful result is exact-target parity at distinct points; broader support and
integration need their own priced phase. Audit passes for implementation and
the bounded independent-reference regressions only.

The four rule/value/score regressions passed in 25.34 seconds; a subsequent
run also checks the rounded saved local centers. Proceed to a first independent
six-dimensional integration price for each numerical target, using the full
twenty observations, the local center/factor and an equal prior/inflated-local
Gaussian proposal. This is the same deterministic-mixture importance identity
derived in C1, with six coordinates; evaluate both density terms at every draw.
Use eight independent scrambled Sobol replications at powers 10, 12 and 14,
seed 50361008 with disjoint component/replication offsets. Integrate all six
physical means, six source squares and six smooth source tails. Preserve the
same empirical .002 posterior-SD refinement and pointwise half-width criteria.

This is pricing, not an assumption that the structural proposal transfers:
PP has different parameter correlations and RK4 costs. The first rung measures
those costs; ESS/maximum weight diagnose proposal mismatch without qualifying
the reference. A numerical-domain failure invalidates that integral and pauses
its refinement; do not treat bad positive-support points as zero weights.
Each target has a 1,800-second parent cap, at most 3,600 additional worker-seconds
under C. CPU reference work may overlap the two GPU workers and structural
reference with additive accounting. No result influences tuning. Eight scrambles
and the .002 screen are inherited operational reference hypotheses, not rigorous
error bounds or cross-model defaults. They may nominate a usable reference only
for these exact numerical targets, with finite-order uncertainty preserved.

Skeptical review: PP-UKF and PP-SGQF share observations and priors but differ
in sigma-point orientation/noise treatment; integrate them separately and
never use one as truth for the other. The initial local-factor proposal may
miss curved/tail mass as it did for structural; preserve that outcome and
price the next proposal/refinement instead of a long unexamined grid. All
source/data and proposal input hashes, parity probes, per-scramble estimates,
completed rungs, timings and failure receipts must be saved. Audit passes
for this limited reference-pricing ladder after the focused parity tests.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_predator_prey_reference.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/predator-prey-reference-01
```

### C2 repair: target-specific auxiliary proposal and fresh integration

The first PP integration completed in 108.12 occupied seconds across both
targets. Neither reference is qualified: final uncertainty half-widths are
about .0212 posterior SD, successive changes about .01095 SD, against .002
required. Minimum importance ESS is about 1,014/16,384. This is insufficient
reference precision, not evidence of a sampler error. Keep these results.

Fit an auxiliary mixture separately for each six-parameter numerical target.
Use eight equally weighted Gaussians, one fixed at the N(0,I) source prior,
and seven initialized at the local center with four times its covariance.
Two discarded, independently scrambled 2^15-point importance rounds each
fit twelve weighted-EM updates; shrink each fitted proposal covariance 10%
toward that round's global weighted covariance. These are convenience
proposal hypotheses inherited from the structural diagnostic and explicitly
retested here. They change only the auxiliary integration density. Every
importance weight uses the full, normalized mixture density; the likelihood,
prior, quantities and .002 criteria stay unchanged. Fit using training seed
60361008; validate the frozen proposal using seed 70361008, with a 1,000,000
offset for the separate SGQF target and disjoint component/replication seeds.

After freezing the proposal, evaluate all eight independent scrambles at
powers 14, 16 and 18, including all eighteen existing quantities. Preserve
proposal inputs, training receipts, per-scramble estimates, source snapshot,
parity checks, successive-order sensitivity and randomization uncertainty.
The final rung alone decides the declared refinement outcome; early attractive
rungs do not stop the ladder. This is empirical numerical refinement, not a
rigorous integration bound or a claim about the exact generative posterior.
Uncertainty and order changes are the reference criteria; importance ESS and
maximum weights are explanatory only. Nonfinite targets, lost source/data
identity or singular proposal covariance invalidate the affected reference.
Do not turn positive-support failures into zero weights.

Allow at most 3,600 additional worker-seconds (1,800 per target), charged to C.
Measured initial 2^14 costs were 33.95/23.93 seconds; simple linear scaling
prices the new final ladders at roughly 12/8 minutes plus fitting/startup.
This is a load-sensitive forecast, not evidence that precision will pass.
CPU-only independent reference with CUDA hidden, two CPU threads and no GPU
claims. Preserve partial rungs at the cap. No automatic further sweep follows
a failed precision screen. Output: `N5/predator-prey-reference-02/`.

Skeptical audit: proposal fitting must remain separate from final scrambles
and must never consume HMC draws. Equal component stratification needs the
whole mixture density, not the generating component's density. The fixed prior
component gives support but cannot prove useful tail coverage. Test density
and normalizer against analytic Gaussian mixtures, separated-mode fitting,
dimension/weight rejection, replay and distinct streams before execution;
retain native full-data parity. New generic diagnostic proposal code and this
completed PP harness are outside the running structural/replication source
closures. Audit passes for this bounded reference repair; no threshold or
sampler source changes are authorized by it.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_predator_prey_reference.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --proposal mixture --powers 14 16 18 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/predator-prey-reference-02
```

### C2 frozen SGQF refinement to support the first full-model fit

The repaired PP references completed: UKF's final half-width/change are
.00584/.00348 posterior SD; SGQF's are .00243/.00176. Neither passes .002.
Keep the UKF result reference-limited and inspect its remaining tail error
before another allocation. SGQF is close enough for a measured refinement:
reload exactly its saved `predator-prey-reference-03/nl-pp-sgqf/frozen-proposal.json`,
without further fitting, and run eight new scrambles at powers 18 and 20,
seed 110361008. The measured 387.1-second power-18 cost predicts about
1,936 seconds; a 2,400-second parent ceiling adds a convenience load/cleanup
margin. Charge C. Same data, numerical target, eighteen quantities and .002
screens; retain uncertainty and missing/failed outcomes. This remains an
empirical reference, not an exact posterior or rigorous integration bound.

Skeptical audit: the reload checks target and data in the old design, equal
mixture weights, the fixed prior component and dimensionality. Native parity
is repeated at new-process startup. Fresh scrambles isolate estimation from
the proposal-training stream; both orders are fixed before execution. No
same-outcome threshold changes or posterior-draw fitting occur. Audit passes.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_predator_prey_reference.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --targets NL-PP-SGQF --proposal mixture --powers 18 20 \
  --reference-seed 110361008 --wall-seconds 2400 \
  --frozen-proposal docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/predator-prey-reference-03/nl-pp-sgqf/frozen-proposal.json \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/predator-prey-reference-04
```

## Phase C3: price full-model trajectories with proposed local geometry

After the saved SIR recovery is terminal, reuse physical GPU 0 for a sequential
four-model diagnostic: structural, PP-UKF, PP-SGQF and SIR. Do not overlap a
third GPU worker or interrupt either existing campaign. The question is whether
the new local factor permits healthy full-horizon corrected trajectories and
what they cost; this supplies the missing complete-fit price inputs, not
posterior admission. The comparator uses prior covariance with exactly the
same native starts as the local-factor arm. It is a geometry diagnosis, not
a tuned-method efficiency comparison.

Four starts are the local center plus the saved factor times fixed signed
contrasts of magnitude 0.5, with fresh seed 20261009/3000 jitter of scale .05.
They deliberately probe a local neighborhood, not global posterior modes.
Freeze them across geometry arms. For each arm, run the actual uniform-duration
corrected HMC runner for 32 transitions at L=3 and 9 and epsilon=.05 and .2.
Those epsilons are explicit conservative hypotheses in locally standardized
units, motivated by unit local curvature; they are not validated away from the
center or suitable prior-coordinate settings. Four pairs expose local
step/duration sensitivity; they do not define a comprehensive tuner grid.
Save raw states, all target/status/score traces, correction and actual work.

Use the existing repository binding, mass validator and transition runner;
do not duplicate the kernel, issue a tuning artifact or infer convergence.
Status, nonfinite values/scores, movement and exact correction are candidate
checks. Failure of a prior-geometry pair is an expected diagnostic possibility,
not a veto of the local arm. Never omit failed rows. Missing provenance or
invalid geometry/target invalidates the affected diagnostic. CPU value/score
parity of the affine mapping and factor checks precede the GPU launch.
The owner excludes NUTS altogether; no internal-status extension is planned.

The first compiled row prices each target. Keep first-call compilation separate
from subsequent calls and report actual TF allocation. A 900-second per-model
parent allowance is a convenience ceiling; at most 3,600 seconds are charged
to C. A cap-only stop preserves partial rows and triggers a priced continuation
decision; it says nothing about sampler validity. No worker is launched until
the current SIR supervisor is terminal. Passive waiting is uncharged and
bounded by its existing remaining cap. The next decision is model-specific
complete-fit settings/budget or a geometry repair, using these prices and the
separate reference status. No claim-bearing fit is silently launched.

Skeptical audit: the same native start bank makes coordinate comparisons
interpretable, but the pairs are not equally tuned and the starts are narrow.
Acceptance and short-path movement are diagnostic only. Independent references
cannot affect this stage; GPU source closures already running remain untouched.
The new constructor delegates numerical geometry and transitions to existing
authorities, with executable affine value/score and status-preservation checks.
Audit passes for the bounded trajectory-pricing question, not posterior or
method promotion.

Regression outcome: both fixed-horizon structural/SIR affine and status checks
pass (90.58 seconds); PP affine checks, full-data PP independent value/score
checks and actual native raw-trace replay also pass. Earlier failed constructor
tests exposed an unsupported keyword and a missing required policy argument;
both were fixed before GPU execution. The status comparison now distinguishes
matching inapplicable infinite bounds from finite numerical values. A short
horizon was rejected correctly by the structural/SIR factories, so those tests
retain the required full horizons. One redundant full-model test process hit
its 300-second cap and another was interrupted at 217.83 seconds to inspect
the first assertion; all are charged, and none is GPU/posterior evidence.

SIR recovery finished and audited both members at 09:09 UTC. Launch command:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_prepared_trajectories.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --wait-for docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/attempt-17-sir-shared-resume/manifest.json \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/prepared-trajectory-pricing-01 \
  --gpu 0
```

## Phase C4: independent SIR SGQF target parity

SIR mechanics now pass, but there is no independent posterior reference. Start
with the smallest discriminating numerical check: reconstruct its full
twenty-observation, eighteen-state/nine-observation SGQF likelihood independently
in NumPy. The numerical target is the existing three-log-scale extension with
N(0,.5^2) priors, no chart Jacobian, kappa=.1 exp(theta0), nu=18 exp(theta1),
and R=100 exp(2 theta2) I9. Preserve its Austria adjacency, initial alternating
S/I means, initial/process covariance I18, and infected-state observations.
The first observation is y1 after a transition, unlike the PP/structural y0.

The inspected numerical transition performs four steps of size .005 and uses
a half-step fourth RK stage. This is not classical RK4; the independent check
must preserve the actual specified finite target, never silently substitute
classical RK4 or interpret parity as an ODE-accuracy claim. The 37-point
level-two axis rule has center weight -5 and each +/-sqrt(3) axis weight 1/6.
Retain that negative weight and the Cholesky sigma-point factor. Linear infected
observations admit an independent exact Gaussian update after the quadrature
prediction; there is no need to duplicate native score recursion.

Compare values and two-spacing source derivatives at zero, opposite .01 log
offsets, and the measured local center (-.002453419,-.009279649,.024822922).
Use the same declared FP64 tolerance hypotheses: 1e-8 value discrepancy and
2e-5 scaled score discrepancy. Confirm rule moments and sensitivity to the
half-step-versus-classical fourth stage, so an accidental numerical-target
substitution is detected. CPU-only tests are capped at 300 seconds, charged
to C. Singular/nonfinite transitions or covariances veto this diagnostic;
never clip state values, repair covariance or omit failing support.

Skeptical review: this reproduces the actual SGQF numerical posterior, not the
exact SIR generative posterior or a claim about the original author's solver.
It needs native value/score parity because changing observation timing, the
fourth RK stage, the negative center weight or the log-scale prior changes
the target. The three parameters make later deterministic quadrature plausible,
but broad prior support may contain unstable numerical programs. No integration
launch is authorized by a few local parity points without a scoped support/cost
check. Audit passes for independent implementation and focused parity only.

### C4 follow-up: check the SIR reference's support before integration

After the focused C4 value/score tests, test the independent finite SIR target
at its center, both signed coordinate axes at one/two/three prior standard
deviations, and 256 scrambled-Sobol prior points (seed 80361008). The .5 prior
scale comes from the native target; 256 and the axis extents are diagnostic
coverage hypotheses, not a global-support proof. Use the unchanged twenty
observations and covariance lifecycle. This is a CPU-only reference diagnostic,
maximum 300 worker-seconds charged to C, with CUDA hidden before imports.

The question is whether broad integration encounters invalid positive-prior
support even though the local parity points passed. Every failed covariance,
overflow or nonfinite value is a repair trigger and veto of an unrestricted
reference integral. Bisect a failed evaluation batch to retain exact offending
coordinates; those evaluations are debugging-only and do not approximate an
integral or estimate failure probability. Compare up to four failing points
with the native adapter's value/score/status in a separate bounded check.
Matching failures indict the finite target's domain, while disagreement
triggers reference debugging. Neither outcome authorizes clipping a state,
zero weighting failures, truncating the prior, changing the ODE solver or
claiming that local HMC is globally correct. No new SIR posterior fit follows
until this prerequisite has a documented disposition.

Skeptical audit: independent and native code share the specified finite target,
not an assumed exact SIR likelihood. Local value/score parity cannot establish
support; axis/prior checks cannot establish global validity even if all pass.
The smallest informative outcome is a reproducible failing coordinate, not
a large reference sweep. This test changes no running sampler dependency.
Audit passes for the bounded diagnostic. Preserve commands, environment,
data/source hashes, every inspected coordinate, errors, native status and wall
time under `N5/sir-reference-support-01/`.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_sir_support.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/sir-reference-support-01
```

### C4 first SIR integration price after the support check

All three C4 parity regressions passed in 6.56 seconds. All 275 support probes
were finite in the independent implementation. A first support launch failed
on JSON serialization before evaluating the probes (5.93 seconds); explicit
array materialization repaired that harness error and the fresh attempt
finished in 6.95 seconds. Both costs remain charged. These finite probes permit
a bounded integration price but do not prove global validity.

Integrate in standardized source coordinates z=theta/.5. The target density is
`log_posterior(.5*z) + 3*log(.5)`; the Jacobian preserves the integration measure.
Use eight equally weighted Gaussian proposal components: fixed N(0,I), plus
seven identical Gaussians with center `local_center/.5` and factor
`2*local_factor/.5`. This is stratified prior/local importance integration;
the seven identical components do not claim seven posterior modes. Evaluate
the full mixture denominator, and freeze the proposal before the first draw.
No proposal fitting or HMC draws are used in this initial price.

Use eight independent scrambles with seed 90361008 and component offsets,
powers 8, 10 and 12, all three physical parameters
(.1 exp(theta0), 18 exp(theta1), 10 exp(theta2)), standardized source squares,
and sigmoid(2*(z-1)) tails. The third physical parameter is the observation
standard deviation, not its variance. Powers and mixture inflation are bounded
pricing hypotheses. The .002 successive-change/pointwise-half-width screen
and its empirical limitations are unchanged. Importance ESS is explanatory.
Any invalid positive-support target stops the integral and saves the offending
batch and seed for localization; it must not become zero posterior mass.

Allocate at most 1,800 CPU worker-seconds from C, including startup and native
parity. CUDA is hidden, two threads, no GPU or posterior qualification claim.
Save every rung, source/data/geometry identity, runtime and uncertainty in
`N5/sir-reference-01/`. The first rung prices the rest; preserve partial rungs
if the parent cap is reached. A limited reference triggers proposal/refinement
work with a new explicit allocation, not threshold relaxation.

Skeptical audit: the standardization Jacobian and source-square definition
must agree with the same N(0,.5^2) prior; confusing variance and SD would test
the wrong posterior quantity. Local geometry gives support coverage no
guarantee. Checked finite prior probes reduce a concrete risk but cannot
certify uninspected tails. The independent reference remains outside sampling
and uses the already tested mixture weighting. Audit passes for this bounded
price and numerical refinement only.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_sir_reference.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/sir-reference-01
```

### C4 measured SIR reference refinement

The first SIR integration completed in 194.81 worker seconds after native
parity (value error 6.83e-13; finer-spacing score error 1.88e-7). All integration
points were finite, but the final order remains reference-limited: the largest
successive mean change is .00442 posterior SD against .002 required. Preserve
that outcome and the two-spacing results; the coarser spacing had a 1.94e-5
scaled error and is not the asymptotic gradient error estimate.

Keep exactly the same deterministic prior/local proposal, without fitting.
Use a fresh seed family 100361008 and eight scrambles at powers 12,14,16.
The measured 144.32 seconds for the initial power-12 rung predicts about
3,031 seconds for these three rungs by linear scaling. Allocate at most
3,600 additional CPU worker-seconds from C, including setup and cleanup.
This is a measured price with load uncertainty, not a promise that .002
will pass. Preserve the same target, full data, nine quantities, criteria
and invalid-support veto. The new source records these CLI options explicitly;
it does not reuse old estimates as independent scrambles or reset prior costs.

Skeptical audit: extra precision is justified by the observed order sensitivity,
and the declared final order prevents stopping at a favorable intermediate
estimate. The unchanged local mixture can still miss tails; finite scrambles
and nominal pointwise Student intervals remain empirical evidence. Compare
proposal/data inputs with the previous design before interpreting the result.
Audit passes for this bounded refinement; no posterior/default claim follows.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_sir_reference.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --powers 12 14 16 --reference-seed 100361008 --wall-seconds 3600 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/sir-reference-02
```

## Phase B extension: complete-fit comparators

The first LGSSM fixed-HMC fit completed both retained members and both passed
the declared reference screen. Its 543.11 seconds is occupied-wall pricing,
not a comparison estimate. While the other three fits run, implement the
remaining short-model comparators without changing any running source closure.

Historical scope note: the NUTS design and executed commands in this section
predate the October 10 correction and are excluded from all further work.
They remain below solely to explain preserved results and expenditure.

The additional question was whether discarded SNAPER proposals, full-refresh
MAMS and the installed TFP NUTS comparator can reach the same posterior checks
and independent quantities. Reuse the frozen prior geometry, four starts, data,
precision policy and independent references. Every SNAPER/MAMS verified member
is assessed; neither can nominate a winner by acceptance, ESS or wall time.
NUTS is an explicitly diagnostic established-method comparator, not a new public
tuner and not an issuer of a BayesFilter candidate-set artifact. Its adaptation
and posterior draws cannot be used as evidence for the ordinary HMC tuner.

SNAPER uses the existing fixed-geometry proposer, 1,000 discarded adaptation
transitions, initial epsilon 0.5, target acceptance 0.8, maximum 32 steps,
no initial MALA-only steps, trajectory learning rate 0.05, and principal/state
EMA factors 8. These are initial experimental choices: 0.8 comes from the
inspected source adaptation convention; the rate/EMA values are the existing
tested proposer settings; the counts are bounded pricing hypotheses. Nominate
epsilon factors (0.8,1.0) and length factor 1.0, at most two distinct pairs.
Invalid pilot state/metric/status, no supported pair, or missing fresh
qualification stops that fit. Save the full pilot trace, source hashes, work
and time. Posterior or reference results cannot alter proposals in the same fit.
MAMS initially uses the existing 3/9 maximum counts and epsilon 0.5 as a
target-specific pricing hypothesis, not as an inherited HMC tuning result.

NUTS uses TFP 0.25's multinomial/generalized-turn implementation, identity
metric in the same prepared coordinates, 1,000 discarded dual-averaging steps,
initial epsilon 0.5, acceptance target 0.8, tree depth 8, one leapfrog per node,
and the source energy-difference divergence threshold 1,000. The smaller depth
than TFP's default 10 is a pricing cap; saturation must be reported and prevents
a claim of a fully resolved NUTS baseline. Frozen posterior divergences veto
the cell. Adaptation divergences are recorded as a tuning diagnostic; nonfinite
accepted state/value/score veto even the pilot. Freeze the averaged step size,
reset to the declared starts, and use the same shared sequential posterior
controller and model quantities. No retained adaptation is allowed. Archive
all chunks and raw transition diagnostics. Per-chain realized leapfrogs are
algorithmic work, not total padded hardware work; report the conservative
maximum separately and use total occupied wall time as the pricing measure.

Source anchors: Hoffman--Gelman (2014), Sections 3.1--3.2 and Algorithm 6,
stored under `.localresources/hmc-methods-review-20261007/nuts-2014.*`;
installed TFP `mcmc/nuts.py` lines 53--86, 197--253 and 415--438, and
`dual_averaging_step_size_adaptation.py`'s explicit result accessors. TFP uses
multinomial selection and a generalized turning rule beyond the original
slice formulation; the comparison is labeled with that actual source variant.
It is limited initially to LGSSM/accumulation, which have no filter-status
payload. Other adapters need status coverage before using this comparator.

First run focused CPU-only mechanics tests: deterministic replay, fixed
epsilon after adaptation, no pilot reuse, correct prepared/model coordinate
mapping, finite/divergence/depth observability, and actual shared posterior
controller integration. Then price the six additional model/method cells,
sequentially on GPU 1 after the first ladder, with at most 1,800 seconds each.
Reference assessment gets at most 120 seconds per completed fit. Do not launch
until the exact command and source identity are recorded. Increase Phase B's
ceiling from five to seven hours using two hours of the repair reserve; Phase E
then has eight hours. This preserves 72 aggregate hours and covers tests,
source inspection, six fits and reporting. Limits are ceilings, not predictions.

Skeptical audit: source NUTS selection must remain intact; an endpoint-only
turning rule would be the wrong kernel. Avoid a second burn-in controller by
using `run_sequential_exact_transition`. That entry point has posterior
authority only, not tuning authority. Using the same posterior reference does
not make short priors representative of harder models. Limited 3/9 HMC grids
and one-seed pricing cannot support an efficiency ranking against adapted
NUTS/SNAPER. The later comparison must price a broader fixed/randomized grid.
No default, release, longer-horizon, or superiority claim follows here. Audit
passes for implementation and focused checks; each GPU launch still requires
the recorded frozen configuration, healthy reference and available allocation.

The first focused NUTS test exposed a real discarded-adaptation failure: an
unbounded dual-averaging proposal drove the LGSSM target to a NaN. Seven other
checks passed, including frozen transitions and the actual shared controller.
Repair the proposal setter by projecting to the binding's already declared
epsilon domain [0.01,2.0]. This changes discarded adaptation and is labeled
`clip_proposed_step_to_declared_domain`; it does not change the target, NUTS
correction or geometry. Preserve unconstrained proposals, realized steps and
the projection count. Test identity for in-domain steps and domain adherence,
then rerun the failing actual-target adaptation. Nonfinite target evaluation
still fails closed. The bounds remain prior-scale pricing hypotheses and are
not certified safe throughout the posterior. A boundary result prevents calling
the baseline fully tuned and triggers a separately scoped calibration if needed.

Comparator launch receipt (8 October, after 17 focused tests passed in 181.80 s):

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_modern_hmc_pricing.py \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/complete-fit-comparators-01 \
  --references docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/reference-01 \
  --methods mams snaper nuts --gpu 1
```

Six cells share a coordinator cap of 11,680 seconds; each fit has 1,800 seconds
and each reference assessment 120 seconds, with the remainder for process
cleanup. For SNAPER's at-most-two emitted pairs, reserve exact conservative
tuning work using the typed per-transition formula, with a 25% retry margin
and a floor of 600,000 units. The margin is a convenience reserve, not an
efficiency estimate; the time ceiling still applies. Include pilot and tuning
costs separately and in the full fit price. Only complete fresh evidence can
verify a member. The focused suite covered all three candidate-based method
fits, actual NUTS adaptation and posterior-controller/archive integration,
and actual LGSSM/accumulation SNAPER proposals. CPU checks intentionally hid
CUDA and cannot certify GPU compatibility. The first trusted GPU cell is the
next device check. Preserve any compile failure for focused repair.

The initial pricing ladder is now terminal: 1,749.17 seconds for four fits and
post-run assessments. All eight members passed posterior checks, seven passed
reference agreement. The second randomized LGSSM member missed the second-
moment bound (0.10595 vs 0.10); it remains an accuracy-screen failure in the
denominator. This neither relaxes the criterion nor changes tuning membership.
Detailed decision/inference tables are in the active execution record.

## Phase D0: replicate the frozen stopping/accuracy design

The isolated second-moment agreement failure motivates fresh complete-fit
replication before broadening the comparison. The question is the frequency
and pattern of delivery, MCSE and reference agreement under the actual shared
stopping rule, for the same two targets and fixed/randomized profiles. This is
not the later broadly tuned five-method efficiency comparison. Keep the exact
two pairs (L=3/9, epsilon=0.5), preparation, 32+32 trial policy, starts law,
posterior stopping and independent references already priced. Initial seed
1000 is excluded from this fresh set. No parameter or threshold is chosen
using the failed member's retained draws.

Run eight independent complete-run seed blocks, 2000 through 2007, across
both targets and both methods: 32 fits. Pair methods by target/start block and
alternate their execution order across blocks. These seeds are convenience
identifiers with separated native streams. Eight replications are a bounded
first uncertainty diagnostic, not a power claim; for example eight successes
still give a two-sided 95% exact binomial lower bound of only about 0.63.
Do not claim a reliability floor, calibrated sequential coverage or a ranking
from this sample size. Preserve all candidates and all planned fits, including
partial runs and empty candidate sets. Report all-member and any-member
delivery separately, and every fixed member's quantity errors/MCSE. Candidate
members and chunks are not independent replications.

The primary event is all planned members passing posterior checks and the
existing independent agreement screen. Hard numerical failures veto the
affected fit, not the remaining independent blocks. Missing/corrupt identities,
unexpected harness errors or invalid references pause new launches for repair.
Wall-only stops stay in the denominator and do not trigger endless retries.
Report exact binomial intervals for fit events and descriptive per-quantity
nominal interval inclusion with reference sensitivity; the latter is not exact
coverage of known truth. Cost summaries include every planned cell and name
non-delivery explicitly. No speed ranking is attempted with these narrow grids.

Each fit retains the priced 1,800-second cap; reference assessment has 120
seconds. The 32-fit allocation is at most 62,400 aggregate occupied-worker
seconds (17h20m), within Phase D's 40 hours. Exact child times are charged,
with rounding/cleanup allowance in the enclosing cap. The coordinator waits
for the six comparator fits to become terminal before acquiring GPU 1. Waiting
uses no numerical worker allocation; its wall-clock wait is bounded by the
comparator's remaining cap. Current SIR continues on GPU 0. Preserve fresh
per-fit manifests, full command, source closure, memory policy, data and seeds.
Stop all new launches when remaining allocation cannot fund a complete cell.

Pre-mortem and skeptical review: fixed data and eight seeds cannot establish
cross-dataset calibration; repeated stopping can affect MCSE interpretation;
shared numerical-reference error can survive replication; changing machine
load can bias unpaired timing. Keep those limits explicit, use paired starts
and alternating order, and avoid performance or coverage promotion. The
independent evaluator/refinement stays fixed and separate from the sampler.
The first all-member outcome audit passes tests for missing fits, unverified
members, mismatched references, duplicate records and false pass labels. This
stage is useful even if it repeats a failure: it distinguishes an isolated
Monte Carlo event from persistent member/quantity patterns requiring a fresh
precision or geometry design. Audit passes for this bounded replication.

Command after coordinator regression checks:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_modern_hmc_replications.py \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/complete-fit-replication-01 \
  --references docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/reference-01 \
  --wait-for docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/complete-fit-comparators-01/manifest.json \
  --gpu 1
```

## Phase D1: first full predator-prey SGQF complete fits

Implement and check a full-model consumer while the references and short-model
replications continue. Its first two cells compare fixed and randomized HMC on
the full twenty-observation PP-SGQF numerical target. Use exactly the checked
local affine factor, identical native start bank (fresh seed block 4000), and
the existing public tuner/verified-member export/reload/posterior controller.
All four local-geometry pricing rows were healthy. Local starts still do not
establish global coverage; the independent numerical reference is mandatory
before interpreting the resulting posterior accuracy.

Use L=3 and 9 at epsilon=.2, the measured healthy local-geometry pairs, as a
two-pair complete-fit price. These are explicit target-specific hypotheses,
not a broad-grid or fully tuned efficiency comparison. The fixed arm uses
the same full-refresh implementation with fixed lengths; the randomized arm
uses the already documented independent uniform-duration law. Their geometry,
starts, 64-transition trials, 32+32 fresh trial repetitions, lower acceptance
bound .5 and error allocations match. Retain and assess both verified members;
failed, missing, unverified and capped members stay visible. R-hat/ESS/MCSE
never affect tuning membership. No reference outcome selects epsilon or L.

Apply the existing posterior count/readiness policy: 2,000 minimum warm-up,
latest 1,000 window, 1,000-draw chunks, 10,000 maximum warm-up/retained per
chain, R-hat 1.05/1.01, bulk/tail ESS 400 and lugsail MCSE/SD .02. Assess the
eighteen frozen reference quantities: six physical parameters, six probit
source squares and six smooth source tails. Compute them directly from source
draws, avoiding inverse-CDF roundtrips at rounded probability endpoints.
All warm-up and tuning draws remain excluded. These inherited operational
screens do not certify stationarity or optional-stopping coverage.

The accuracy screen is `abs(error)+3*MCSE+reference_uncertainty <= .10*reference_SD`
for every quantity, with `reference_uncertainty` equal to the final order's
pointwise half-width plus the absolute final-order mean change. The independent
reference must first pass its existing .002 refinement criteria. This combined
screen is operational, not a simultaneous/anytime confidence statement. One
paired seed provides complete-fit pricing and candidate viability, never a
method ranking, default promotion or general posterior reliability.

Measured randomized local-row times are 37/84 seconds for 32 transitions at
L=3/9. Linear scaling prices two-candidate tuning at about 4.3 hours and a
minimum 2,000+1,000-draw posterior at about 3.2 more hours. Fixed lengths have
greater padded work; load and retained precision can increase either price.
Allocate at most ten occupied worker-hours per fit, plus 120 seconds per
assessment and 40 seconds total cleanup: 72,280 seconds from D. Together with
D0's 62,400 seconds this is 37.42 hours, within D's 40-hour ceiling. These are
bounded pilot prices, not assurances that the retained precision caps fit.
Do not silently lower trial counts, narrow the start bank, change precision
or extend the campaign clock when a fit exhausts its allowance.

The sequential supervisor uses GPU 0 only after the four-model trajectory
supervisor is terminal, and only after the new PP-SGQF reference has a complete
passing result. It verifies source/data/reference identity before each fit,
records fresh trusted device telemetry and memory growth, and leaves GPU 1
to D0. Wait at most three hours for prerequisites (a convenience scheduling
ceiling, uncharged as numerical work). Missing/invalid references stop this
launch queue for the prescribed reference repair; they do not reject HMC.
Numerical/identity failures stop for diagnosis. Time/precision-limited cells
retain their result and proceed to the other predeclared cell. No automatic
numeric repair or repeated-until-success attempt is part of this price.

Skeptical audit: this first full-model pair addresses the short-fixture gap but
cannot answer all five-method efficiency questions. A short healthy trajectory
is insufficient; independent reference and separate stopping supply the missing
checks. Source/data matching prevents confusing UKF with SGQF, and direct
quantity transforms avoid accidental changes of chart. Check actual binding
geometry/status, quantity parity, the public lifecycle on a cheap mechanics
fixture, missing-reference rejection and preservation of both member outcomes
before queuing. Audit passes for implementation and this bounded conditional
price. Keep modern profiles experimental regardless of the result.

Output roots: `N5/full-pp-sgqf-pricing-01/` and its per-method fit directories.

Implementation audit, 10:40 UTC: the new consumer calls the existing public
ordinary tuner, grouped member export/reload and posterior controller. It uses
the checked full-model geometry factory to issue new immutable fixed/randomized
bindings. No running replication/reference dependency was modified. All 18
quantities enter the posterior controller directly as model coordinates; the
`quantities_id` API is only for an additional `quantities_fn`, so an initial
test failure from using it here was repaired by removing that inapplicable
option. This changes no statistical threshold or quantity. An initial test
module import error was also corrected. Both failed checks remain charged.

Nineteen focused regressions now cover direct physical/source-tail parity,
actual PP-SGQF affine value/score and start matching, both public sampler
lifecycles with two verified/reloaded members, all 18 archived coordinates,
cooperative budget interruption, unstarted-member denominators, passing and
failing reference assessments, missing/wrong-target/wrong-data/stale-source
references, and insufficient/incomplete/nonfinite refinement evidence. Logs are
`campaign-72h-audit/full-model-fit-tests-01.log` (14 pass, two initial import
failures), `-02.log` (two diagnosed quantities API failures), `-03.log` (four
repaired lifecycle/assessment tests pass), and `full-model-queue-tests-01.log`
(two queue/accounting tests pass; one overlaps the first set). These mechanics
tests deliberately hide CUDA and use small posterior counts; campaign counts
remain the full declared values. Seven watcher regressions also pass.

The conditional queue preserves both planned cells from creation. It stops on
reference/harness/identity failure and bounds subprocess groups, while a fit's
time or posterior precision cap leaves the other cell eligible. Independent
assessment runs on CPU after the GPU fit, against the frozen passing reference.
One remaining limitation is that a hard process cap can interrupt an executing
TensorFlow call; committed chunks and incomplete member dispositions remain
visible. A new attempt would need a separate recorded allocation from the
same 72-hour pool. No such automatic numerical retry is part of D1.

Skeptical recheck passes: the comparator and data match; local trajectories are
only prices, references remain prerequisites rather than tuning objectives;
no criteria were weakened after observing results. The 10:38 budget snapshot
charges 10.74 hours, reserves 16.68 remaining hours for active workers and
20.08 hours for D1, leaving 24.34 unallocated hours after the existing 600-second
overhead reserve. That is one 72-hour grant, not a new grant.

Launch command, after the above checks, with trusted GPU access:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/run_modern_hmc_full_pricing.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/full-pp-sgqf-pricing-01 \
  --gpu 0
```

The read-only watcher is extended to this queue and the current SIR/PP
refinements. Its 24-hour passive lifetime derives from D1's 20-hour worker cap,
three-hour prerequisite wait and one-hour reporting margin. It grants no
extra numerical time and writes alerts rather than launching retries.

## Phase E0: bounded PP-UKF tail-moment reference repair

The PP-SGQF refinement completed at power 20: maximum pointwise half-width
.00065254 SD and last-order change .00047042 SD, both below the unchanged .002
criterion. D1 began fixed HMC on GPU 0 at 10:44:49 UTC. SIR's reference also
completed its final power-16 rung with `empirically_refined` status. These are
references for the specified numerical filter posteriors, not exact nonlinear
latent-state posteriors and not rigorous integration error bounds.

PP-UKF remains reference-limited. Inspection of its saved eight-scramble tables
finds the largest power-18 uncertainty in `second_4`: .005836 SD, with scramble
estimates from .95625 to .98053. `second_5` has the largest last-order shift,
.003476 SD. Global importance ESS remains large, so it cannot substitute for
quantity-specific precision. The exact cause of those tail-sensitive errors is
not established. Test whether a larger fresh fixed-proposal sample resolves
them before inventing another proposal or blaming the sampler.

Reuse the existing UKF-specific eight-Gaussian proposal in
`N5/predator-prey-reference-03/nl-pp-ukf/frozen-proposal.json`, without fitting it
again. Use powers 18/20/22, eight fresh scrambles, seed 120361008. The new seed
is a convenience reproducibility choice disjoint from previous reference
streams; it is not selected by observed precision. The order ladder extends
the already implemented integrator; it changes neither model, observations,
prior, source chart nor the 18 quantities. The exact comparator is the
independently parity-checked native UKF numerical posterior.

The primary reference criterion remains a final mean change and pointwise
randomization half-width each at most .002 of reference SD, for every quantity.
Native value/score mismatch, invalid target evaluation, nonfinite uncertainty,
or changed source/data/proposal is a continuation veto. A completed limited
integral or budget cap rejects that reference attempt and triggers a proposal/
tail-variance diagnosis; it does not reject UKF HMC. Importance ESS, individual
scramble estimates and timing are explanatory only. Passing would permit a
subsequent matched posterior comparison, not sampler admission, exact posterior
claims, method ranking or default promotion.

The measured power-18 UKF cost is 554.6 seconds. Linear sample-count scaling
prices 18/20/22 at `554.6*(1+4+16)=11,646.6` seconds plus approximately 15 seconds
of startup/parity. Assign a single 14,400-second occupied CPU-worker cap from
the eight-hour E repair reserve. The roughly 24% timing margin is a convenience
allowance for unmeasured workload variation, not a throughput guarantee. This
leaves four hours in E and about twenty hours unallocated in the overall grant
after all currently reserved work. No second attempt or renewable timer is
authorized by this subphase. CPU-only execution explicitly hides CUDA; it
cannot supply GPU-performance evidence.

Skeptical audit passes for this bounded diagnostic: it has a measured cost,
fresh streams, an unchanged target-specific proposal and explicit precision
criteria. A larger sample could still miss an important region; the conclusion
remains empirical refinement, with parity and per-quantity uncertainty as
required checks. It could fail because the importance proposal has unstable
tail weights; saved per-scramble moments and concentration summaries distinguish
that failure from a sampler failure. No tuning data or HMC output is used to
fit or select this reference. The existing reference worker and its tested
numerical dependencies remain unchanged while D0/D1 use their frozen hashes.

Execute the existing reference worker through a small bounded CPU supervisor
saved as `N5/campaign-72h-audit/refine-pp-ukf-05.py`; output is
`N5/predator-prey-reference-05/`. The supervisor records the exact worker command,
environment, source-launcher checksum, seed, wall time, terminal classification
and result path. It uses the existing worker entry point because the earlier
one-hour supervisor allocation is superseded only for this explicitly funded
four-hour attempt; modifying that script would invalidate active source hashes.

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/campaign-72h-audit/refine-pp-ukf-05.py
```

## Phase E1: final priced structural reference refinement

Structural-reference-03 finished in 7,695 seconds including supervision. All
fifteen last-order mean changes are below .002 SD (maximum .001968), but two
pointwise half-widths fail: source-square rho .002190 and source-tail sigma
.002118. The other thirteen half-widths pass. In the last quantity one of eight
scramble estimates is .07465643 while the others lie between .07436771 and
.07442306; the largest normalized importance weight in that replication is
.0016066. This is evidence of uncertain tail-sensitive integration, not proof
of missing mass, bad target code or HMC failure. Preserve the failed criterion.

One further fixed-order refinement uses the unchanged structural-specific
proposal from reference-02 and fresh seed 130361008, eight scrambles at powers
21 and 23. The point count doubles the previous final rung; neither a changed
proposal nor a selected subset of successful replications is used. The question
is whether additional independent integration precision resolves the two named
failures and preserves agreement for every quantity. The criterion remains
both uncertainty half-width and last-order mean change <=.002 SD throughout.
Native parity, finite values/quantities and unchanged source/data/proposal are
continuation vetoes. Importance ESS and outlying replicate weights are
explanatory. Passing establishes empirical numerical-reference refinement,
not a rigorous integration bound, HMC convergence, exact nonlinear posterior,
or method superiority.

The measured power-22 cost is 5,859.87 seconds. Powers 21/23 are priced at
`5859.87*(.5+2)=14,649.68` seconds plus startup; assign one 16,200-second CPU cap.
The roughly 10% margin is a convenience scheduling hypothesis. Reassign half
an unused settled Phase-A hour to E, so A=3.5/E=8.5 hours and the campaign total
remains 72. E0/E1 consume at most 8.5 hours together. No automatic extra rung,
changed tolerance, adaptive successful-seed selection or unlimited repetition
is permitted. A failure calls for a discriminating proposal/tail-weight study,
not another blindly enlarged integral. Any such study requires its own plan
and an allocation from the unchanged remaining campaign balance.

Skeptical audit: the final-order shortfall is measured and narrow, and all mean
changes pass, so doubling points is an interpretable precision diagnostic.
Square-root error reduction is a hypothesis rather than a theorem for these
self-normalized QMC moments. Fresh scrambles may expose worse uncertainty or
a larger mean shift; either remains a limited reference. The same UKF finite
target, full 100 observations, priors, transformations, proposal and quantities
are retained. Only the independent integration sample count changes. No claim
is earned by good runtime, importance ESS, or a single attractive replication.
The old implementation and running campaigns' source closures stay frozen.

The bounded supervisor is `N5/campaign-72h-audit/refine-structural-04.py`, with
the unchanged reference worker and a fresh `N5/structural-reference-04/` output.
It records the worker command, seed, CPU-hidden-GPU environment, launcher hash,
wall time and result/stop classification. Execute:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/campaign-72h-audit/refine-structural-04.py
```

## Terminal settlement and continuation sequence, 9 October

The terminal audit reads every completed D0/D1 member and its independent
assessment, verifies the two final reference orders, checks all recorded source
and input hashes, and settles top-level worker costs without nested double
counting. It is a static result audit, not a new numerical experiment. Exact
artifacts are
`N5/campaign-72h-audit/terminal-audit-20261008T181410.json` and
`N5/campaign-72h-audit/budget-ledger-20261008T181410.json`.
All completed or failed attempts remain preserved. No new numerical stage was
launched during this closeout.

The outcomes do not invalidate the HMC implementation, targets, data or saved
artifacts. Both full-model L=3 kernels remain tuning-qualified but did not
deliver the declared posterior precision within their count allowance. The
structural importance integral remains ineligible as a reference because one
quantity fails refinement. These are candidate/reference repair triggers, not
rejections of the research direction. Continue within the same remaining grant
in the following order; independent harness work need not wait for a reference
repair.

1. **Diagnose structural reference variance.** Use the saved per-scramble
   moments to identify contributions to source-square rho, then instrument a
   bounded replay of the frozen proposal to retain quantity-weight and component
   summaries. Compare the high-contribution coordinates with the proposal and
   target density; distinguish denominator variation, missed tail representation
   and numerical error. A replay explains the failed estimate and cannot be
   counted as new independent validation. Do not delete a scramble, truncate
   tails, drop a quantity, relax .002, or launch another blind point-count
   expansion. If proposal repair is indicated, fit on a separate calibration
   stream and qualify it on fresh fixed validation orders, retaining the original
   reference failure. The exact diagnostic point count, worker cap and fresh
   validation design must be priced and audited before numerical launch.
2. **Complete full-model harness coverage.** PP-UKF and SIR now have qualified
   empirical references for their declared numerical targets. Extend the tested
   all-member posterior/quantity path to those model-specific transformations
   and references. Regress target/data mismatches, unavailable references,
   complete quantity coverage, failed/unstarted members, export/reload and
   budget stops. Structural posterior agreement must await its qualified
   reference. CPU mechanics checks cannot replace the required GPU/XLA fits.
3. **Price missing methods and posterior repairs.** Preserve the D1 L=3
   limitations and use all recorded outcomes when planning a fresh duration or
   geometry comparison. Price full-target SNAPER and MAMS before expanding
   the matrix. SNAPER proposals still require fresh public tuning; MAMS needs
   its own corrected integrator settings. R-hat, ESS, MCSE and reference error
   remain posterior checks and cannot rank, discard or repair tuning candidates.
   Changed geometry or transition settings create a new scope; no old evidence
   is silently transferred.
4. **Freeze a feasible paired comparison and refresh the master.** Retain the
   required model/method inventory and explicitly mark every tested, failed,
   capped or deferred cell. Choose replication and caps from observed complete
   costs within the remaining 45.01 hours, charging concurrent workers additively.
   Predeclare uncertainty analysis before interpreting method differences.
   Preserve N5 release limits and the separate N6 LAPS requirements.

Skeptical continuation review: this order addresses the actual failed quantity
before spending on another integral, keeps independent references separate from
sampler tuning, and retains successful and unsuccessful members under the same
criteria. Passing short-model checks cannot substitute for nonlinear-model
evidence. Eight D0 seeds per group and one D1 paired seed do not support a method
ranking. The 45.01-hour balance is a ceiling, not a promise that the whole
four-method matrix or later LAPS work fits. A phase must return a measured price
and a concrete design before consuming its next numerical allocation. Wrong
source/target, invalid math or artifacts, and exhaustion of the unchanged total
budget remain continuation vetoes; precision-limited candidates trigger the
planned repairs.

## Phase E2 and D2: structural diagnosis and full-model harness continuation

9 October continuation, within the settled 45.01-hour balance. E2 asks whether
the unresolved structural source-square rho moment is dominated by identifiable
importance-weight contributions under the frozen eight-Gaussian proposal. The
comparator is the **same saved power-23 integral**, replayed at seeds 130361032
(scramble 3, largest rho-square estimate) and 130361016 (scramble 1, smallest
estimate). Their deliberate selection makes this a diagnostic contrast, never
two new independent validation replications. All points and original weights
remain in every reported estimate.

The primary engineering criterion is reproduction of both saved means, weight
ESS, maximum weight and log normalizer within float64 reduction tolerance. The
diagnostic records component contributions to normalized mass, rho-square mean
and squared centered influence `w_i^2 * (rho_i^2 - mean)^2`; it also records those
contributions by absolute source-rho bands and the coordinates of the 64 largest
influences. Bands [0,1), [1,2), [2,3), [3,4), [4,infinity) and 64 displayed points
are convenience reporting choices, not truncation or statistical thresholds.
The component sums must reconstruct the unpartitioned quantities. Native versus
independent target/score checks at the high-contribution points distinguish a
target numerical mismatch from a poorly represented region. Invalid weights,
changed source/data/proposal, non-reproduction or value/score mismatch stop the
diagnostic for repair. A concentrated contribution explains the saved variance
but cannot qualify a reference, prove missing mass, or change an HMC decision.

The last measured power-23 cost was 11,690.22 seconds for eight scrambles, or
1,461.28 seconds each. Two replays are therefore priced at about 2,923 seconds
before added reporting and parity. Reserve **4,200 occupied CPU-worker seconds**
for one supervised E2 attempt, including startup and cleanup. Its roughly 44%
margin is a convenience allowance for diagnostic overhead and shared-machine
load, not a runtime promise. E0+E1 settled at 26,210.60 seconds; adding this cap
stays inside the existing 30,600-second Phase-E ceiling. Reserve at most another
600 seconds for focused CPU regression checks from the still-unspent campaign
balance; tests are charged separately. Use fresh output
`N5/structural-tail-diagnostic-01/`, hide CUDA before any framework import,
and record command, Git/source hashes, replay seeds, versions, wall time and
terminal status. No third scramble or larger integration order follows
automatically. If a localized instrumentation failure needs retry, its spent
time is deducted from this same allocation.

In parallel with the CPU diagnostic, D2 extends the existing full-fit consumer
to PP-UKF and SIR with target-specific physical/source-square/source-tail
quantities and matched references. This is implementation work, not authority
to transfer the PP-SGQF epsilon or trajectory grid. Actual model-specific
trajectory prices and fresh tuning designs must be inspected before a GPU
fit is launched. Required regressions cover the independent quantity transforms,
wrong target/data/reference rejection, failed or unstarted members, all-member
export/reload and posterior lifecycle, and preservation of the PP-SGQF path.
No reference or sampler mathematical dependency changes are planned; old
reference hash bindings must still validate before use. Structural agreement
remains blocked until a separately qualified repaired reference exists.

Skeptical audit before implementation: replay can explain an outlier while
misleadingly appearing to increase replication, so no reference qualification
is allowed from E2. Global ESS cannot substitute for moment-specific variance;
the influence decomposition explicitly addresses that risk. Native parity at
the dominant points guards against diagnosing proposal variance when the target
implementations disagree. The CPU-only independent implementation is an allowed
reference exception and supplies no GPU timing or production-backend claim.
The saved two-scramble price and fixed cap bound compute. D2 reuses the public
tuner and separate posterior controller, so no local acceptance rule or
posterior-based tuning rule is introduced. Implementation and this bounded
diagnostic may proceed; a repaired proposal or full GPU campaign requires its
own next recorded design and price within the unchanged grant.

Exact E2 entry point after its focused checks:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_structural_tail.py \
  --reference docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-reference-04/reference \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-tail-diagnostic-01 \
  --wall-seconds 4200
```

### D2a: current-workload pricing before full fits

The earlier 32-transition local-geometry prices varied substantially across
targets and workload: at epsilon .2, randomized L=3/9 took 119.47/244.36 seconds
for PP-UKF and 84.72/195.17 for SIR. Those numbers would imply much longer
tuning than D1 actually observed. Treating one timing as a universal rate would
recreate the earlier timeout-planning problem. A fresh trusted probe on GPU 0
at 18:57 UTC found it idle with memory growth verified. That observation is
temporary; each launch records another resource snapshot.

Run **two 64-transition calls per candidate** for both fixed and randomized
HMC, on each of PP-UKF and SIR, with the same frozen local geometry and pairs
L=3/9, epsilon=.2. These hypotheses come from each target's own saved healthy
trajectory diagnostics, not from PP-SGQF posterior outcomes. Seed 4100 defines
fresh paired starts; the two repeated calls use separate declared seeds and
identical starts, recording first-trace versus repeated-call time. They are
pricing/health diagnostics only; they cannot issue tuning qualification or
posterior results. Check every call for the binding's existing numerical/status
and movement failures and preserve realized work. A numerical failure triggers
localization before the corresponding full fit. Any source/identity or runtime
policy failure stops the queue. A resource cap preserves incomplete rows.

There are sixteen calls across the two targets. Reserve **3,600 GPU-worker
seconds** total, with a 1,790-second cap per target and twenty seconds reserved
for cleanup. This is a bounded measurement allocation, not a predicted success
time; the preceding prices may exceed it under contention. Retain a failed or
unfinished price as such, without automatically extending it. CPU test costs
and the concurrently running E2 worker are charged separately. Numerical
dependencies are frozen while this price and its later fits run.

Skeptical recheck: a 64-transition price alone does not establish complete-fit
cost. Use it to check current throughput, then combine measured tuning workload
with D1's actual posterior chunk costs and report the uncertainty before
allocating full fits. Different target geometry prevents treating D1 retained
counts as a guarantee. The new measurement is warranted by the large observed
rate mismatch and explicit open timeout concern. No criterion is relaxed and
no new proposal law or target is introduced. Save the supervisor, command and
per-call receipts in `N5/full-model-current-price-01/` via
`scripts/diagnose_modern_hmc_full_fit_price.py`.

### E2a: localize the high-contribution score-check failure

E2 stopped after its first replay in 1,469.18 seconds. The saved integral
reproduced; the 64 displayed points account for .002688 normalized mass but
.989632 of the descriptive squared centered influence on source-square rho.
Native and independent values agree within 1.11e-9. However, central-difference
score errors at h=1e-4 and 1e-5 are .17645 and .001430 on the existing scaled
error measure. This fails the unchanged 2e-5 parity screen. It could reflect
large finite-difference truncation error rather than an incorrect native score;
the observed decrease motivates a discriminating derivative check. Structural
reference promotion and proposal repair remain paused until this is resolved.
This affects structural evidence only; it does not veto UKF/SIR pricing.

Using only the saved 64 coordinates, run central differences at h=1e-4, 1e-5,
3e-6, 1e-6, 3e-7 and 1e-7, and an independent five-point stencil at h=1e-5 and
3e-6. The ladder extends the original diagnostic steps; these are numerical
hypotheses, not changed target settings. Require **both** fixed five-point
checks to agree with the native score within the original scaled 2e-5 tolerance
and with each other within the same tolerance. Central differences and their
stepwise changes explain truncation/roundoff; do not select the best point or
step and discard a failure. The stencil is checked on a polynomial with known
derivative before target evaluation. All64 value/status/score rows are saved.
If the high-order checks fail, retain unresolved score parity and diagnose the
native derivative before further structural integration. A pass resolves this
local diagnostic failure, not reference precision or general score correctness.

Reserve one **240-second CPU diagnostic** from E2's remaining 2,730.82 seconds;
the full importance integral is not rerun. The cap is a conservative convenience
allowance for 64-point native tracing plus finite-difference calls, much smaller
than an integration replay. Use CPU-hidden-GPU execution and a fresh
`N5/structural-tail-parity-01/` manifest/result. No numerical dependencies of the
active GPU price change. Skeptical audit: same points, data, scalar target and
tolerance preserve the question; a higher-order independent stencil distinguishes
coarse-step error from a score mismatch without manufacturing a passing seed.

### E2b and D2b: resolved derivative diagnostic and priced full fits

E2a finished in 11.656 seconds. Both predeclared five-point estimates passed
the unchanged scaled score tolerance of 2e-5: errors 1.07764e-5 and 8.91015e-8,
with mutual discrepancy 1.06872e-5. Central differences decrease from .17645
at 1e-4 to 4.49849e-7 at 1e-7. The saved-point discrepancy is consistent with
coarse finite-difference truncation, not a native-score defect. This resolves
the local check only; reference precision remains unqualified. Preserve E2's
failed receipt and its completed first replay unchanged.

**E2b** runs only the remaining selected scramble, index 1 / seed 130361016,
at power 23, with the same data, proposal, weights, partitions and reproduction
criteria. It reuses the first replay and separately checked E2a receipt as
historical inputs; neither selected replay becomes a new independent replication.
Apply the same two fixed five-point checks to the second replay's top 64 points
by calling the E2a diagnostic, without copying the derivative implementation.
Stop on a source/input mismatch, failed reproduction, invalid values/status,
or failed five-point parity. Reserve **2,700 CPU-worker seconds**, within E2's
2,719.16 seconds remaining after its first replay and E2a. No second full replay
of index 3 or automatic third scramble is authorized by this design. Output:
`N5/structural-tail-diagnostic-02/`.

**D2b** asks whether all tuning-qualified fixed/randomized members deliver the
declared posterior accuracy on full twenty-observation PP-UKF and SIR targets.
Use their matched saved empirical references, the existing target-specific local
geometry, four paired starts at seed 4100, and each target's own healthy proposed
pairs `[[3,[0.2]],[9,[0.2]]]`. The epsilon/length hypotheses are inherited from
each target's geometry diagnostic and checked in D2a; they are deliberately
narrow pricing candidates, not broad-grid adequacy or optimality evidence.
Fixed HMC is the classical comparator. Both methods receive identical geometry,
starts, per-target caps, acceptance rules and posterior precision requirements.
The 32 measurement plus 32 verification repetitions, each 64 transitions, v8
lower acceptance bound .5, and independent search/verification streams remain
unchanged. The public tuner owns qualification; R-hat, ESS, MCSE and reference
error cannot enter its decisions. Every planned member remains in the denominator.

Primary outcomes are all-member posterior/reference delivery and complete-fit
occupied time, with member-level reasons and any-member delivery also reported.
The separate posterior controller retains its minimum 2,000 / maximum 10,000
discarded warm-up transitions, 1,000-transition retained chunks up to 10,000,
rank-normalized/folded R-hat criteria, bulk/tail ESS >=400 and lugsail MCSE/SD
<=.02 for all 18 PP or 9 SIR quantities. These inherited operational criteria
were already tested; finite short-chain diagnostics cannot prove equilibration.
Reference comparison keeps the existing uncertainty-aware .1-SD screen. Numeric
or filter-status failures veto the affected fit and trigger localization; source,
data, reference or artifact mismatches stop the queue. Warm-up, precision and
resource failures preserve incomplete members without becoming tuning failures.

D2a's sixteen 64-transition calls completed without numerical vetoes in 966.49
supervised seconds. Repeated-call costs imply the following totals, before extra
tracing, diagnostic and serialization overhead. The range assumes warm-up at
its 2,000 minimum and retained sampling between 1,000 and 10,000; these counts
are scenarios, not forecasts transferred from another model.

| Target / method | Tuning seconds | Tuning plus posterior scenario | Per-fit cap |
| --- | ---: | ---: | ---: |
| PP-UKF fixed | 4,718 | 8,173--18,540 s | 24,000 s |
| PP-UKF randomized | 4,184 | 7,249--16,443 s | 24,000 s |
| SIR fixed | 7,148 | 12,383--28,090 s | 36,000 s |
| SIR randomized | 6,335 | 10,975--24,894 s | 36,000 s |

Caps are convenience allocation hypotheses with approximately 29% margin over
the slower method's 2,000+10,000 scenario. Maximum warm-up plus maximum retained
counts can exceed them (27,754 / 24,615 seconds for UKF and 42,051 / 37,266 for
SIR before overhead). Preserve that outcome and measured progress; do not assume
a timeout disproves the sampler. Shared-machine contention may also change
rates. The monitor reports checkpoint progress, failures and elapsed use, while
the hard caps protect the aggregate grant. A later localized recovery must be
priced from its actual saved state and funded from the remaining balance.

Run UKF then SIR, fixed then randomized within each target, on GPU 0 with fresh
trusted probes and verified memory growth for every fit. Other users' GPU
processes are untouched. The two existing fit queues reserve 48,280 and 72,280
seconds including assessments/cleanup. Add 60 seconds of coordinator margin
per target: **120,680 seconds total**, one charge per outer supervisor, without
adding nested costs again. Coordinator output `N5/full-model-continuation-01/`
points to sibling queues `N5/full-pp-ukf-pricing-01/` and
`N5/full-sir-pricing-01/`. The whole systemd process group is additionally
bounded at 120,740 seconds, including 60 seconds of teardown covered by the
existing overhead reserve. The design has one paired seed, so no statistical method
ranking is supported. Neither success nor failure decides default readiness,
the exact nonlinear generative posterior, global geometry coverage or N5 release.

After settling E2, E2a, D2a and 435.20 seconds of new tests, 99,432.60 seconds
(27.6202 hours) are charged. The unchanged 600-second reserve leaves 44.2132
hours. Reserving D2b and E2b leaves **35,787.40 seconds (9.9409 hours)** for
later stages before releasing their unused caps. Reassign the phase ceilings
within the same 72-hour grant to A=3.5, B=6, C=6, D=45, E=8.5, F=3 hours.
The aggregate balance is binding; phase labels do not create extra budget.

Skeptical pre-execution audit: the references match each target's full data,
chart and quantity transformations; UKF and SGQF are never pooled. D2a is used
only for runtime/health pricing, not posterior or tuning admission. Same-seed
fixed/randomized fits are paired, and two candidate lengths are dependent
members, not independent replications. Local geometry and one seed leave
coverage and reproducibility uncertain; the result must retain these limitations.
Both failed and unstarted candidates remain visible. E2b resolves the actual
remaining replay with a checked derivative diagnostic instead of wasting the
first integral. All code and numerical inputs remain frozen during live jobs.
Wrong-target/source, missing diagnostics and exhausted budget are continuation
vetoes; an ordinary precision-limited candidate is a repair trigger. This plan
passes the audit for bounded continuation, not promotion.

Exact supervisors, after focused wiring checks:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/continue_modern_hmc_structural_tail.py \
  --reference docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-reference-04/reference \
  --previous docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-tail-diagnostic-01/diagnostic \
  --parity docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-tail-parity-01/diagnostic/result.json \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-tail-diagnostic-02
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/continue_modern_hmc_full_models.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/full-model-continuation-01 \
  --gpu 0
```

### E2c/D2c: local diagnostic convergence and device-selection repair

9 October, after the 09:11 China-time status check. D2b's UKF fixed fit finished
in 12,895.58 seconds: L=9 passed posterior/reference screens, while L=3 reached
10,000 retained draws with insufficient precision and no hard numerical veto.
Both remain tuning-qualified. The randomized fit stopped before its numerical
worker launched because the trusted readiness probe forbade explicit GPU 0
while GPU 1 was idle. The queue therefore stopped before SIR. This is a
localized resource-selection defect, not a sampler result. D2b's settled outer
charge is 12,900.826 seconds, including its preserved failed preflight.

**D2c** preserves the completed fixed fit and resumes only the three unexecuted
fits, in order: UKF randomized, SIR fixed, SIR randomized. Model, data, local
geometry, seed 4100, pairs, public tuning and posterior/reference criteria are
unchanged. The new coordinator chooses the current device through the installed
trusted probe's `auto` policy before each fresh fit, rather than retaining GPU 0
across changing availability. Every allowed device must be the same RTX 4080
SUPER hardware class, and the actual physical UUID is recorded. This is a fresh
fit, never a migration of live chains or reuse of qualification on another
device. Runtime comparison remains descriptive because resource load can differ.
If both permitted GPUs are busy, the existing reviewed shared-memory helper may
admit GPU 1 with its unchanged 4,096 MiB headroom check and memory growth; driver,
visibility and unknown policy errors remain fatal.

Retry at most three preflight attempts per cell, with a fresh versioned directory
and 30-second interval. Three and thirty are convenience bounds on resource
races, not scientific constants. Retry only recognized availability/preference
failures with no numerical worker command, design, tuning or posterior artifact.
Each attempt and interval consumes that cell's unchanged 24,000 / 36,000-second
allowance; a retry neither resets time nor adds independent evidence. Never
retry a numerical failure or reinterpret a count/precision limit as launch
failure. Assess every completed or capped fit with the existing independent
assessor and preserve the two-member denominator. Reserve 96,600 seconds for
three fit caps, 120-second assessments and 80-second coordination/cleanup
allowances per cell. This is less than D2b's unspent 107,779.17-second reservation.
Use `N5/full-model-recovery-01/` and a systemd cgroup hard bound of 96,660 seconds;
the extra teardown minute is within the existing overhead reserve. Numerical
sources from the completed fixed fit remain unchanged.

**E2c** follows E2b's reproduced second scramble, which cost 1,469.851 seconds.
Its native/reference values agree within 2.17e-10. Five-point errors at 1e-5
and 3e-6 are .000402422 and .00000328548. Their factor of about 122 is consistent
with fourth-order truncation under a 3.33-fold step reduction. Central-difference
errors continue decreasing to .00000101567 at 1e-7. One predeclared five-point
step still failed, so E2b correctly retained unresolved local parity. No integral
or selected point is discarded or regenerated.

Check **all 128 saved rows from both replays** using fixed five-point steps
1e-6 and 3e-7, with the independent central step 1e-7 as a cross-check. Require
each estimate and the two five-point estimates' mutual discrepancy to satisfy
the unchanged scaled 2e-5 tolerance, and values to satisfy 1e-8. These smaller
steps are a truncation/roundoff diagnostic hypothesis justified by the observed
decrease, not altered target settings or new statistical replication. Unit-check
the stencils on analytic polynomials and require finite results. Save every
pointwise derivative/error. Stop unresolved if any check fails; do not keep
shrinking steps until a pass. A pass resolves parity only at these coordinates
and cannot qualify the structural reference. Reserve 120 CPU-worker seconds,
inside E2's remaining 1,249.31-second pool. Output:
`N5/structural-tail-parity-02/`. New independent replications remain zero.

Monitoring repair: a never-started child whose parent has already stopped must
be shown as terminal/unstarted with the parent's reason. It must not keep the
watcher claiming that work is progressing indefinitely. Report active fit paths
explicitly so the new coordinator's nested attempts expose actual tuning and
posterior checkpoint progress. Stop the old read-only monitor and preserve its
alerts; start a fresh monitor after focused regressions.

Skeptical audit: the fixed result is not rerun or selected away; all failed
preflights and the original four-fit denominator stay visible. Only fresh
unstarted fits may select devices. Accepted devices, memory growth, same target
and frozen numerical settings protect comparability at its stated descriptive
level. A passing short trajectory or larger acceptance cannot substitute for
posterior delivery. Higher-order diagnostics are checked on both saved sets,
including the worst point, with fixed smaller steps and unchanged tolerances;
they cannot become reference qualification. No continuation veto has invalidated
the targets or campaign budget. The limited candidates and reference uncertainty
remain repair triggers. This bounded implementation and local retry pass the
audit; an actual numerical/source/reference failure still stops the affected run.

Entry points after focused checks (tfgpu Python; GPU coordinator trusted,
CPU diagnostic with CUDA hidden before framework import):

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/recover_modern_hmc_full_models.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/full-model-recovery-01
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/diagnose_modern_hmc_tail_score_convergence.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/structural-tail-parity-02
```

### D2d: use the second available GPU without duplicating queued fits

The owner requested use of available GPU capacity. The trusted 01:35 UTC
inventory showed the existing randomized UKF worker on GPU 1 at 90% compute
utilization with 929 MiB occupied; GPU 0 was idle with 18 MiB occupied. Free
VRAM on GPU 1 alone is not unused compute throughput. The installed automatic
selection policy admitted GPU 0 while GPU 1 was busy. The serial D2c coordinator,
not a scientific dependency, was preventing SIR from using that device.

Keep the live UKF numerical worker and its own 24,000-second supervisor cap
unchanged. Pause **only** its outer scheduling process with SIGSTOP after
verifying its PID, command, active UKF output and unstarted SIR rows. This process
is waiting on the UKF child; pausing it does not signal the child process group.
A new bounded supervisor observes the existing UKF fit to completion, runs its
unchanged independent assessment, and retires the paused old service only after
the fit is terminal. In a second supervisory thread, run SIR fixed then SIR
randomized via the unchanged D2c `run_cell`, using its automatic trusted device
selection, pre-numerical retry rules and 36,000-second caps. Record the selected
UUID at launch rather than assume the earlier idle-device snapshot still holds.
At most two BayesFilter campaign numerical workers run concurrently. Existing
shared-device admission remains available if external occupancy changes; it
does not establish a speed advantage or permit another campaign worker.

There is no new seed, model, target, method, proposal, numerical dependency,
tuning or posterior criterion. The existing UKF is observed, not relaunched;
the SIR pair is moved from the old queue, not added to it. All four original
fits and their failed preflights remain in the denominator. No completed
result is overwritten. The UKF reference assessment is written once to its
previously absent assessment path, after sampling finishes. The old manifest is
preserved with a separate scheduler-transfer receipt explaining its retirement.

Budget transfer uses the old coordinator's elapsed time at the pause. Charge
that prefix once, then charge the observer's remaining UKF occupied time and
the separate SIR worker times additively. Their combined reservation is
`96,600 - old_elapsed` seconds, not a new grant. UKF's remaining limit is derived
from its actual start and cap; its assessment/cleanup and each SIR fit retain
the already allocated 200-second margins. Charge no extra numerical worker
for the paused scheduler. A fresh outer service has an absolute 96,600-second
wall bound; the per-fit and additive limits remain stricter. The budget ledger
records the transfer and subtracts it from the old reservation before new work.

Before pausing, require the completed fixed fit's source/reference hashes to
match and inspect the active UKF worker receipt. Wrong PID/command, a changed
active cell, started SIR, numerical/source/reference invalidity or exhausted
budget stop the transfer. If validation fails immediately after pausing,
resume the old scheduler before launching SIR. Numerical failure of either
model stops its affected sequence; an independent model can complete under
its existing contract. A count-limited posterior remains a reported outcome.
Source files imported by the active numerical worker stay frozen.

The 01:57 UTC recheck found an unrelated process newly on GPU 0. GPU 1 had
about 31 GiB free while UKF used about 89% compute; the existing shared-device
admission can therefore start SIR there if no permitted device is idle. Sharing
may reduce throughput and consumes the same worker budget; free memory is not
a speed claim. No new hardware or numerical policy is introduced.

Transfer implementation audit found that stopping a paused scheduler needs
explicit TERM-before-CONT ordering, and failures during manifest writing or
thread setup must resume the original queue before any new worker starts.
The repair binds signals to the saved PID/start time, releases both supervisor
threads only after transfer records exist, and restores the prior reservation
on a pre-launch failure. A capped adopted fit reports incomplete evidence even
if its original receipt remains marked running. Control-plane pause polling
(2 seconds, 0.01-second interval) and cleanup (30 seconds) are convenience
bounds inside the existing 200-second per-cell overhead, not convergence or
numerical-progress criteria. These focused changes address the audit findings;
verify them with CPU-only fake-process and failure-injection tests before launch.

Pre-launch review completed at 02:04 UTC. The first regression found a real
completion race: both futures could finish after their status scan but before
the loop's exit check, leaving terminal rows marked running. The coordinator now
uses one consistent completion scan. The rerun passes 47 tests, including real
parent/child pause behavior, TERM-before-CONT retirement, manifest-write and
thread-setup failures, additive overlap charging, original cap preservation and
the existing resource retry/monitor regressions. Compile and whitespace checks
pass. Both completed and active UKF source closures (143 files each) and their
references still match. All regression attempts are charged to the same grant.
No material numerical or scientific change remains in this scheduling repair.

Skeptical audit: this changes scheduling only and does not turn idle memory
into evidence of spare compute. The target-specific caps and fresh device
receipts account for uncertain throughput. PID/child checks prevent signalling
an unrelated process; pausing the scheduler prevents a duplicate SIR launch.
The two overlapping occupied durations must be added instead of charging just
the new supervisor's elapsed wall time. Same-seed results on the same hardware
class remain descriptive comparisons, with no statistical runtime ranking.
The unchanged numerical sources and testable scheduling transfer make this
a proportionate local resource repair. Proceed after focused fake-process,
budget-transfer and non-duplication regressions pass.

New outputs are `N5/full-model-parallel-01/`, `N5/full-ukf-adopted-01/` and
`N5/full-sir-parallel-01/`; the monitor watches the last two actual fit paths.
Launch with tfgpu Python and trusted systemd execution:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/parallelize_modern_hmc_full_models.py \
  --root docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/full-model-parallel-01
```

Executed at 02:04:36 UTC (10:04:36 China time). SIR fixed is on shared GPU 1,
alongside the unchanged UKF numerical process. The SIR design verifies GPU
memory growth before initialization and GPU/XLA execution. UKF checkpoint
progress continued after the scheduler pause. Watcher 10 monitors both jobs;
watcher 09 is superseded. `N5/full-model-parallel-01/launch-verification.json`
records the launch checks. The new budget ledger charges 116,536.407 seconds,
reserves 93,889.053 seconds for overlapping work plus the existing 600-second
overhead, and leaves 48,174.540 seconds unallocated. No grant or scientific
criterion changed. These are launch and engineering checks; both fits remain
in progress and their posterior results are still pending.

### D2e: terminal scheduling repair and honest partial-fit closeout

At 15:13 UTC on October 9, D2d is terminal. Both SIR fits finished; fixed
delivers both members and randomized delivers one, with the other precision-
limited at the unchanged retained cap. UKF randomized exhausted its 24,000-
second allowance, preserving an L=3 terminal precision failure and an unfinished
L=9 member. Its overall fit result is absent. Separately, service retirement
failed because TERM had already removed the transient unit before `systemctl
stop`; exit code 5 was incorrectly treated as a numerical-assessment blocker.

Repair question: can retirement recognize an already removed unit while
continuing to reject genuine cleanup errors or a live/replaced scheduler?
Require a second systemd observation with MainPID zero, LoadState not-found,
ActiveState inactive, and no matching saved scheduler process. Only that
specific exit-code-5 race is accepted. Regressions must reject unexpected
codes, loaded/active units and still-live or replaced processes.

Close out in a fresh `N5/full-model-closeout-01/` directory. Validate every
saved source/reference hash; classify each original fit and candidate using
persisted evidence; preserve the missing UKF result and interrupted member.
Do not manufacture a finalized fit, posterior pass, reference assessment or
complete tuning/sampling result from a progress file. No numerical rerun or
sampling extension is included in this repair. SIR's existing assessments are
unchanged. Settle D2d's additive occupied time once and release its unused
reservation; add every focused regression and closeout duration to the same
259,200-second grant. Refresh the master and result notes.

Skeptical audit: service disappearance is an infrastructure race, whereas the
UKF wall cap and L=3 precision failures are genuine incomplete outcomes. Fixing
the former cannot turn the latter into success. No proxy, selected-member pass
or shared-device runtime can establish all-member delivery or method ranking.
The read-only partial summary uses the original two-member denominators and
has no promotion authority. CPU-only mechanics/diagnostics hide CUDA; tests
have a 120-second convenience bound and no GPU work. Existing numerical
sources, target, reference, seed, caps and admission criteria stay unchanged.
This localized repair is within the campaign authorization. Proceed after the
focused failure-path regressions pass.

D2e executed at 15:19:39 UTC. All 54 focused regressions pass, including missing
units, unexpected cleanup errors, still-live/replaced processes, additive
charging, transfer rollback and duplicate prevention. The original service is
now independently confirmed absent. The closeout preserves all four original
fits/eight members, with four member posterior/reference passes and one
all-member fit pass. UKF L=9 remains interrupted, not promoted from raw chunks.
Numerical and reference hashes remain unchanged. D2d's actual 63,820.863-second
additive cost is settled once; the grant has 78,238.095 seconds unallocated
after its original overhead reserve. No new numerical run was launched.

Next phases remain the already planned structural calibration/validation
proposal repair and missing nonlinear comparator pricing. The terminal UKF
interruption additionally requires a bounded audit of supported posterior
checkpoint continuation before a separate resumed run; do not rerun completed
members or silently expand the original fit's cap. Count-limited L=3 outcomes
motivate a predeclared duration/geometry study, with all old failures preserved.
These are continuation/repair questions, not rejection of tuning candidates or
promotion of the four passing members to a default.

### October 10 additional funding and continuation

The [48-hour supplement](bayesfilter-modern-hmc-n5-additional-48h-2026-10-10.md)
adds 172,800 seconds to the same aggregate accounting, for a 432,000-second
ceiling. It preserves prior costs and outcomes, and provides D2f/D3/E3/D4 with
phase refresh and bounded reservations. D2f is launched on GPU 1 after eight
native continuation regressions and unchanged source/reference preflight.
Only interrupted UKF L=9 resumes; L=3 remains terminal. The new ledger is
`N5/campaign-120h-audit/budget-ledger-01.json`. Three old full-model NUTS
bookkeeping rows remain only in historical manifests. They are excluded from
required coverage and repair work while the six SNAPER/MAMS prices proceed.

# Resume model inference with an imperfect DCTR force

Status: **concrete proposed extension; not launched, not tuned**. This plan
follows the [evidence recovery](ledh-surrogate-hmc-recovery-result-2026-09-18.md).
The existing campaign used all three LG1 attempts. The proposed extension is
**one additional LG1 inference attempt, at most two GPU-hours and one aggregate
CPU-worker-hour, charged within the unused 64/16-hour campaign envelope**.
It does not replenish any other case's attempts or add the older surrogate
campaign's unreconciled allowance. Owner authorization of this one-attempt
extension is the remaining budget decision. Normal sandbox permissions still
apply to GPU work.

## Question and change from the recovered program

Can a fixed imperfect force from full DCTR generate useful proposals that,
after endpoint Metropolis correction, sample the proper scalar LG1 parameter
posterior? Separately, how far is a posterior defined by a frozen DCTR
likelihood from the exact LG1 model posterior?

The original idea—deterministic DCTR force with endpoint correction—is retained.
The old likelihood-only five-parameter target and hand-written toy MH loop are
retired from new evidence. The current full-DCTR bounded LG1 adapter and
containment repairs are retained. Filtering mean/covariance screens no longer
block the *exact-endpoint, imperfect-force* question. They still veto the
filtering claims they were designed to assess. The finite-posterior/model-
posterior distinction remains mandatory.

This is a discriminating inference pilot, not all-model completion. The
[coverage inventory](ledh-surrogate-hmc-model-coverage-2026-09-18.md) remains the
program scope; each later family requires its own target/reference/tuning and
cost plan. Non-SSM targets require an explicit applicability decision rather
than a fabricated DCTR likelihood.

## Fixed scientific target and comparison arms

Use the existing LG1 definition: `x_t = phi*x_(t-1) + epsilon_t`, process
variance 1, observation variance 0.5, T=10, `x_0 ~ N(0,1/(1-phi^2))`, and
`phi ~ Uniform(-0.95,0.95)`. The covariance depends on phi. Include its
dependence in likelihood and score. Coordinates are
`phi = -0.95 + 1.90*sigmoid(z)`, with the full Jacobian included exactly once.
Data-generating phi remains 0.8. These choices come from the current reviewed
LG1 scope, not the old incompatible d=3 experiment.

Use fresh attempt index 4 in the repository's partition-aware seed scheme.
Calibration, validation and the untouched pilot dataset use distinct existing
partition codes and recorded replicate identifiers. Do not reuse the failed
claims/calibration curves as holdout evidence. Freeze DCTR inputs and controls
before HMC tuning; freeze the HMC kernel before the untouched retained run.

| Arm / reference | Endpoint density | Trajectory force | Question answered |
|---|---|---|---|
| Exact model reference | Kalman likelihood × proper prior | None: refined one-dimensional quadrature | What is the model posterior? |
| Tuned classical HMC comparator | Same exact endpoint and chart | Exact Kalman posterior score | Can the shared inference harness recover that posterior with a conventional force? |
| Primary DCTR-force arm | Same exact endpoint and chart | Analytical full-DCTR score, transformed with prior/Jacobian terms; honestly labeled imperfect relative to the exact endpoint | Does the imperfect DCTR force permit valid, feasible model inference? |
| Finite-DCTR reference | Frozen DCTR likelihood × same prior | None: separately refined parameter quadrature | Does this finite likelihood approximate the model posterior? |
| Conditional finite-endpoint HMC arm | That same frozen DCTR endpoint | Same frozen imperfect field | Does HMC sample the finite posterior? Execute only if reference and forecast fit the same attempt budget. |

The first three rows are the minimum inference pilot. The fourth is the next
priority within the same allowance. If it cannot be computed accurately within
budget, mark finite-likelihood bias unresolved; do not make a model-inference
claim for a DCTR-valued endpoint. The fifth is optional within this pilot,
and its absence must not be called completion of finite-endpoint validation.

Use the complete shared DCTR route: Contract E–Chol, per-particle UKF covariance
lifecycle, GenUT correction and both caps, analytical recursive score, fixed
random design, and repository-selected transport chunks. Start with N=1024 and
K=1024 as the current LG1 scope; any N change is a new tuned scope and does not
fit this one-attempt proposal. FP32/TF32 GPU/XLA is the DCTR execution target;
FP64 exact Kalman/quadrature is an explicitly recorded reference. The returned
FP32 DCTR force may be cast at the FP64 sampler boundary without representing
it as an exact FP64 gradient.

## Evidence contract and diagnostic roles

The primary pilot pass criterion is **posterior agreement with the refined
exact reference on one untouched dataset**, using the frozen DCTR-force arm.
Require all of the following:

- Four dispersed chains, retained split/folded rank-normalized R-hat <=1.01,
  and bulk/tail ESS >=400 for phi and the declared posterior summaries. A
  tuning artifact alone does not establish these retained-run conditions.
- For mean, second moment, and CDF at the exact posterior's 0.1/0.5/0.9
  quantiles, report autocorrelation-aware MCSE and simultaneous 95% intervals.
  Predeclare standardized equivalence margins 0.10 posterior standard
  deviations for the mean, 0.10 reference posterior variance for the second
  moment, and 0.03 absolute probability for each CDF value. The entire
  simultaneous interval for each discrepancy must lie within its margin.
  Use a Bonferroni adjustment for the five summaries; insufficient precision
  is inconclusive, not PASS. These are proposed pilot tolerances, not inherited
  calibrated standards or universal defaults.
- Bound numerical quadrature error below one tenth of each corresponding
  margin using successively refined, fixed-support grids. If agreement of two
  grids is the only available error estimate, identify it as an empirical
  refinement check, not a rigorous error bound; retain a further refinement
  for validation. Failure of reference stability is a continuation veto.
- No unsupported numerical endpoint exclusions. Track true prior exclusions,
  numerical exclusions, force fallback, nonfinite states and trajectory
  failures separately. The exact endpoint is authoritative in the primary
  arm; a failed DCTR force must not turn a valid Kalman endpoint into -infinity.
- The conventional exact-score comparator must pass the same target/reference
  checks. If it fails, repair the shared harness before attributing failure to
  DCTR. Record the DCTR arm's result even if the comparator is faster.

The pilot establishes agreement only for this fixed dataset and scope. It
does not establish repeated-data interval coverage, all-model inference,
superiority, default-readiness, or adequate finite-DCTR likelihood accuracy.
Multi-dataset replication is a later planned phase with its own forecast.

| Diagnostic | Role and consequence |
|---|---|
| Correct prior, stationary initialization, Jacobian, endpoint/force identities, per-chain energy, determinism and reverse replay | Engineering validity/continuation veto if wrong. Repair within the attempt when the target and budget are unchanged. |
| Posterior-reference agreement with sufficient uncertainty resolution | Primary promotion criterion for the stated pilot scope. Failure rejects that candidate or leaves evidence inconclusive. |
| Convergence, ESS, exclusions, invalid endpoint or corrupted artifacts | Promotion veto; broken reference or target is a continuation veto. Insufficient sampling is a budget/repair issue. |
| DCTR score mismatch, norm, angle/sign agreement, filtering moment error, fallback frequency, acceptance and energy-error distribution | Explanatory and repair diagnostics; no threshold on force mismatch alone stops the program. Severe energy failures that break the finite proposal are a separate numerical validity issue. |
| Refined finite-DCTR versus exact posterior discrepancy | Veto on a finite-DCTR model-inference claim, not on exact-endpoint surrogate HMC or on sampling the finite target. |
| Cost forecast beyond the remaining allocation or exhausted attempt | Continuation veto on this launch; record under-budgeted, preserve results. |

## Cheap adversaries and conditional checks

Construct three comparators from this one-dimensional problem before retained
evaluation: a tuned random-walk Metropolis chain (needs only the exact endpoint),
prior-proposal importance sampling (bounded proper prior makes weights directly
computable), and a Laplace approximation at a scalar posterior mode (cheap
local curvature approximation). Exact quadrature is the certifying reference,
not merely another heuristic. Use the same endpoint/prior/data for all.

Report discrepancies and cost conditional on lower-posterior-tail, central and
upper-tail events, and report chain behavior from low/mode/high starting
positions. Freeze event boundaries from the exact reference and comparator
settings before retained evaluation. The heuristics are falsification checks;
do not tune DCTR controls to beat their outcomes. If a heuristic reliably
meets the inference criterion where DCTR does not, that is the headline and
a promotion veto. Runtime or ESS differences without uncertainty remain
descriptive; no speed ranking is needed to pass this pilot.

## Implementation and tuning work before launch

Use `BoundedLEDHFixtureTarget` from
[ledh_program_fixtures_tf.py](../../bayesfilter/highdim/ledh_program_fixtures_tf.py)
for the frozen DCTR model/inputs, not `_diagonal_lgssm_fused_model` or a reduced
fork. Bind the exact Kalman endpoint and the DCTR field as separate typed
objects. Return a **potential-gradient** field to `FrozenPositionOnlyForce`
(negative log-posterior score), including transformed prior/Jacobian terms.
Preserve the endpoint/force status distinction through the sampler.

Before any retained run, write an executable consumer wiring/parity test for
both potential and force; test changed phi, the stationary initial covariance,
transformed density/Jacobian, unequal chains, and a valid endpoint with failed
force. The existing five shared-kernel tests are supporting mechanics evidence,
not substitutes for these DCTR consumer tests. CPU tiny reference tests may
hide GPUs; GPU/XLA tests require escalation and memory growth verification.

There are two separate tuning requirements. DCTR controls need fresh
scope-specific offline calibration/validation and a repository-issued matching
selection artifact. A failed filtering candidate can be a starting hypothesis,
not a tuned default. Class C numerical protections require documented non-harm
calibration; primary posterior metrics cannot silently select them. The force
claim must remain distinct from canonical exact-score admission. If the
existing selection interface cannot express this honest imperfect-force scope,
record and repair that interface explicitly; do not self-stamp canonical status
or weaken existing score-admission rules.

For HMC, use `bind_neural_force_hmc_tuning_runner` with `tune_hmc_kernel`, the
supported ordinary tuner. Keep target and force identities, source closure,
scope, mass coordinates and telemetry attached to its issued artifact. Tune
mass, epsilon and fixed leapfrog count on calibration/validation only; report
the actual bounded configuration. Do not reuse the old step=0.01/L=10 or toy
step=0.1/L=10 as tuned defaults. NeuTra-specific warm-up/training rules do not
apply in the absence of NeuTra.

No new pilot driver or issued scope artifact exists yet. This plan deliberately
records that implementation prerequisite rather than presenting an invented
launch command. Authorized routine preparation may implement it; substantive
GPU tuning/inference is the proposed fourth attempt and requires the stated
attempt extension. The wrapper for an actual launch is the existing
`scripts/run_ledh_graceful_failure_program.py`; preserve its attempt accounting.
Do not reset its existing campaign ledger.

## Budget, output and stopping

The proposed two GPU-hours are an upper bound including failed launches and
compilation, not a prediction. Allocate up to 15 minutes to consumer checks
and timing, 30 minutes to bounded offline/tuner work, and 75 minutes to the
untouched inference/reference comparisons. Unused time can move forward within
the two-hour total. CPU work has one aggregate worker-hour including reference
generation and analysis; parallel workers debit summed time. One GPU process
at a time; each job at most 45 minutes. A localized infrastructure retry stays
in this one attempt and consumes its budget.

The first timing artifact must measure complete DCTR force calls and endpoint
calls through the intended compiled consumer, including compile/startup, batch
size, device, allocator current/peak memory and TF32 settings. Forecast all
tuner evaluations, four-chain sampling chunks and references, retaining the
active program's 1.5 multiplier on measured variable costs plus compilation.
Do not extrapolate scalar timing to a four-chain batched implementation without
measurement. If the minimum primary pilot does not fit, stop **under-budgeted**
before committing to a long chain. Do not shorten sampling and then claim the
same evidence standard.

After public tuning, retain fixed-kernel draws in chunks of 250 transitions per
chain, with no warm-up draws included. Assess the predeclared diagnostics only
at chunk boundaries; cap retained sampling at 4,000 per chain and the actual
remaining time. This is a pilot precision/convergence procedure, not a
sequential hypothesis-test claim. Save every retained chunk. If diagnostics
need a longer chain, report inconclusive rather than rank candidates.

Use a fresh `docs/plans/artifacts/surrogate-hmc-lg1-inference-<UTC>-01/` directory.
Required records: exact executed commands/configuration; Git/source closure;
environment/device/memory-growth/TF32/XLA; immutable data and random-design
hashes/seeds; DCTR and HMC tuning artifacts; full retained draws and endpoint
telemetry; refined references; conditional heuristic table; MCSE and interval
calculation; attempts/time; decision and inference-status tables. Stop on
inconsistent target identity, missing tuning authority, invalid reference,
unexplained endpoint support loss, incorrect call-chain wiring, corrupt output,
or exhausted compute. A poor but valid DCTR proposal is a candidate result,
not rejection of the research direction.

## Assumption audit and pre-mortem

| Choice | Provenance / justification | Failure risk and earliest check | Status |
|---|---|---|---|
| Scalar LG1, T=10, stationary initial law, bounded uniform prior | Current reviewed scope; exact posterior reference is affordable | Inactive parameters or omitted initialization derivative: finite changes in phi plus Kalman/adapter checks | Fixed pilot target, not representative of all models |
| N=1024, full DCTR, FP32/TF32/XLA | Current execution direction and inspected LG1 scope | Poor force or numerical exclusions: full consumer replay, validity telemetry, measured scope tuning | Starting hypothesis; no prior scientific admission |
| Exact model endpoint first | Isolates the owner's imperfect-force question | Exact likelihood availability could hide finite-DCTR likelihood error: keep the separate finite-reference row | Experimental decomposition, not a universal endpoint |
| Zero extension on force failure | Retained containment semantics; position-only finite kicks remain invertible in exact arithmetic | Path/history dependence or wrong status propagation: forward/reverse replay and valid-endpoint failure fixture | Must pass consumer tests; not a silent sampler change |
| Public ordinary tuner and fixed retained kernel | Inspected capability registry | Missing imperfect-force artifact support or adaptive retained runs: issuer identity and actual consumer check | Supported mechanics, DCTR binding pending |
| Pilot margins, ESS and one untouched dataset | Explicit modest fixed-dataset agreement question | A lenient screen may pass while multi-dataset inference fails: tail checks, heuristics, later replication | Proposed pilot criterion; no universal promotion |
| Two-hour GPU / one-hour CPU cap | Bounded extension inside unused campaign resources | Underestimating expensive force calls: full-consumer timing and conservative forecast | Resource ceiling, not demonstrated feasibility |

Skeptical audit: the old toy rule, improper five-parameter target, reduced DCTR
route, missing model-posterior comparison, loss of model coverage and confused
filtering/force vetoes are material defects. This plan corrects those design
errors. Execution readiness remains conditional on the explicitly listed
consumer wiring, tuning support, cost forecast and attempt extension. A good
acceptance rate, finite outputs, Gaussian-looking plots, or successful compilation
cannot close those prerequisites or the posterior-agreement criterion.

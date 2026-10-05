# State-space HMC validation within 48 hours

Terminal execution: the measured main stage ended at 23:23:42 Shanghai on
September 29. Three K4 fits completed with declared posterior and descriptive
reference checks passed; four K0 fits timed out during search despite their
contention allowances. One funded K4 fit missed the original latest-start
cutoff; 24 other slots were unfunded. All 32 remain in the denominator. The
completed K4 records do not provide interval-coverage results, and no nonlinear
main fit completed. Full validation remains incomplete.

The 18,893.277-second enclosing charge is settled once, both services are
inactive, and no campaign worker remains. No further numerical work is queued;
the original latest-start deadline has passed. Results and the terminal audit
are under `artifacts/hmc-ssm-main-2026-09-29/r1/`, with the interpretation in
[the execution note](artifacts/hmc-ssm-pilot-repair-2026-09-29/result.md).
No deadline, model, seed or numerical criterion changed.

Earlier phase: the [reviewed continuation amendment](bayesfilter-hmc-ssm-pilot-repair-2026-09-29.md)
has completed K0/K2/K4 full pilots and is using the final unused repair
allocation to finish K7's two posterior assessments. K7's complete search
retains 21 verified candidates. The tested main allocator is queued to run
after K7 settles; it uses complete cumulative costs, the actual remaining
budget and original cutoff, keeping all 32 slots in the report. The current
8.5-hour preview funds original K0/K4 lanes. Launch recomputes affordability;
main remains 0/32 while K7 runs. K1/K3 remain incomplete, K5 needs a full
repaired pilot and K6 preparation remains unresolved.

Earlier repair: [campaign allocation and startup diagnosis](bayesfilter-hmc-ssm-pilot-repair-2026-09-29.md)
now provides a tested way to assign additional campaign time to an unchanged
resource-interrupted numerical fit. K5 has completed preparation using the
existing optional startup and checked warmup-restart mechanisms. K6 passed
startup but discarded two failed warmup attempts and reached its diagnostic
limit during a third; preparation remains unresolved. A discovered coordinator
closeout race is repaired and covered by a real subprocess regression.
Continuation of the six original resource-limited pilots started at 12:49:11
Shanghai on September 29, preserving source, designs, seeds and checkpoints.
Additional work shares a six-hour repair pool
inside the existing grant and original deadlines. Complete cumulative prices
still precede main admission; all 32 slots remain in the report.
K0 has since completed both declared posterior assessments through checkpoint
continuation. Its full cumulative price is 2,241.074 seconds. K7 is next in the
running recovery queue; main remains 0/32.

Earlier September 29 status: the shared-device continuation stopped at 05:38:30
Shanghai. Eight mechanics and four public-pipeline preflight cells passed;
six pilots exhausted their full contention-extended allowance and K5/K6 failed
numerical bootstrap checks. Complete prices are 0/8; main fits started are
0/32. The [execution note](artifacts/hmc-shared-gpu-recovery-2026-09-29/result.md)
records preserved progress and the repair sequence. No worker remained at that
stop; the active diagnostic and queued recovery are described above. All
original deadlines, workload requirements and accounting limits
remain; none is a promise of successful completion.

Earlier shared-device update: the owner rejects exclusive GPU availability as a
requirement. The [September 29 recovery amendment](bayesfilter-hmc-shared-gpu-recovery-2026-09-29.md)
was implemented and launched on GPU 2 alongside foreign work. It adds bounded
automatic checkpoint recovery, enables contention allowances in preflight and
pricing, and intersects main affordability with the original remaining wall
time and cumulative cap. Numerical settings and all 32 slots remain unchanged;
additional scheduling allowance comes from the existing repair allocation.
The original deadlines and ledger remain active. The empty-device queue below
is superseded; GPU preflight and complete-fit pricing still precede main work.

Earlier September 29 update: all ten C1 retries are complete; the original
coverage failures remain. The first SSM attempt passed all eight GPU mechanics
cells, then failed its public preflight because the campaign target declared
no full-chain XLA capability. The [tested repair and continuation](bayesfilter-hmc-ssm-xla-preflight-repair-2026-09-29.md)
has been launched and is waiting for GPU capacity. It repeats preflight on
fresh source, then prices complete fits before admitting main work. Its 69
CPU engineering regressions do not establish GPU qualification or posterior
accuracy. The original deadlines, scientific settings, 32-slot inventory,
36-GPU-hour cap and active additive ledger remain unchanged.

Earlier funding update, September 28: the owner added 50 GPU hours. The settled
old balance plus this grant is 58.736 GPU hours, including the separate NeuTra
reserve; the plan's 36-GPU-hour ceiling is funded. The first resource-recovery
attempt completed one fit, timed out another under sustained contention, and
deferred eight. State-space preflight waited without launching a numerical
worker. The [active continuation amendment](bayesfilter-hmc-c1-recovery-and-ssm-launch-2026-09-28.md)
now queues for capacity separately, preserves the completed fit, and then runs
the remaining recovery and unchanged SSM stages. Its additive ledger is
`artifacts/hmc-ssm-funded-2026-09-28/grant-ledger.json`. The original SSM clock
began September 28 at 05:03:52 Shanghai and does not restart. GPU preflight and
complete-fit prices remain required before the main matrix.

Earlier September 28 amendment: C1 has terminated and its GPU receipt is settled.
The owner authorized [ten bounded timeout recoveries followed by this campaign](bayesfilter-hmc-c1-recovery-and-ssm-launch-2026-09-28.md).
That plan records the revised scheduling policy, same-class idle GPU selection,
fresh frozen package and automatic preflight/pricing/main sequence. Its actual
grant balance limits this campaign; the 48-hour deadline grants no extra GPU
time. Original coverage failures remain open regardless of recovery results.

Historical September 26 status: CPU implementation and regression preparation complete;
GPU campaign not launched. The owner explicitly requested preparing the campaign
while C1 finishes. Preserve C1; settle its existing enclosing receipt once
before pricing or allocating this campaign. This supersedes the early-stop
recommendation in the historical funding discussion below. No additional
36-GPU-hour grant is inferred from the request to prepare the campaign.

This is an amendment
to the [master program](bayesfilter-hmc-repair-master-program-2026-09-16.md),
not a second tuning guide. The current C1 service continues under its existing
plan. Ending that confirmation and reallocating its reservation is a material
change to the promised experiment; this document does not do that implicitly.

## Recommendation and deliverable

Use a bounded state-space validation campaign with cheap independent numerical
oracles, focused failure tests, and a modest, fixed inventory of complete
public-pipeline fits. Keep Gaussian and beta-binomial as regression controls.
Reserve large repeated-fit calibration studies for separately priced questions
about coverage or detector power, after the state-space pipeline is checked.

The 48-hour deliverable is implemented tests, measured results, repairs within
the budget, and a terminal disposition for every planned case. A deadline can
bound computation; it cannot guarantee posterior convergence or close every
scientific gap. This plan does not substitute a small study for C1's original
256+256-fit statistical design.

## Checked baseline and missing coverage

The checkout inspected was `main` at
`de80aaff5812ebfbed551977476c0868551a2c88`, with substantial uncommitted work.
Any execution must freeze the relevant actual source, including required dirty
changes, and record its diff and dependencies. Do not copy or revert unrelated
NeuTra, governance, or manuscript work.

* `bayesfilter/testing/inference_validation/targets.py` evaluates
  `lgssm_location` through a dense Gaussian covariance. It has an independent
  Kalman reference, but its tuning target does not call the production filter.
* `tests/inference_validation/test_multimodel_pipeline.py` exercises six models
  with tiny CPU mechanics allocations. It explicitly makes no calibration claim.
* `bayesfilter/testing/lgssm_generic_target_adapter_tf.py` calls the actual
  batched QR Kalman score through the generic SSM adapter. Its persistence is
  `0.75*tanh(raw)`: it cannot test near-unit persistence unchanged.
* `bayesfilter/testing/deterministic_lgssm_exact_target_tf.py` supplies a
  four-state, 18-parameter, T=120 target through the actual batched SVD Kalman
  implementation. Its likelihood is exact; its parameter posterior has no
  inspected analytic reference. Old tuning results do not price current runs.
* `bayesfilter/testing/simple_nonlinear_generic_target_adapter_tf.py` supplies
  a real nonlinear sigma-point filter adapter. Its Model B data fixture has
  only three observations. Longer data and current-source pipeline integration
  are required for the proposed nonlinear case. Its likelihood is an
  approximation to the nonlinear model likelihood.
* `inference_validation/procedures.py` directly constructs `ValidationTarget`.
  Adding catalogue names alone will not exercise those SSM adapters: the
  factory, data generation, target status, reporting transforms, reference
  dispatch, and source identity all need integration tests.

The active two public tuners and their authority are declared in
`HMC_TUNING_INTERFACE_CAPABILITIES` in `bayesfilter/inference/tuning_contract.py`.
Use `tune_hmc_kernel` for an exact derivative of the declared log target,
including a clearly identified approximate filtering target. Use
`tune_fixed_transport_hmc_kernel` only for supported supplied frozen maps.

## Historical funding comparison (superseded by preserving C1)

At 2026-09-25 23:15 Shanghai, C1 was active with 109 completed Gaussian process
receipts and no beta-binomial receipts. This count is progress, not a statement
about posterior coverage. The service started at 15:02:36 Shanghai. Its forecast
is 37.473 total GPU hours and its enclosing ceiling is 165,611 seconds.

The September 25 grant is 50 GPU hours. Its settled charges are 4,699.202
seconds; the ledger's 175,300.798 remaining seconds still include C1's active
reservation. After about 8.21 live C1 hours, roughly 40.48 GPU hours remain
unspent, before other reservations. The canonical NeuTra pricing reservation
is another 1,200 seconds and remains separate. The older CPU balance is
67,338.430 worker-seconds, about 18.71 worker-hours.

The recommended allocation below requires retiring C1 early, preserving its
partial inventory as incomplete, and settling its actual enclosing receipt
exactly once. Do not change its denominator, pool it with new SSM outcomes,
selectively complete favorable slots, or report a stopped prefix as the planned
confirmation. No terminal C1 scientific result is inspected to choose this
reallocation; the reason is the owner's model-relevance and time constraint.

If C1 instead finishes at its forecast, approximately 11.22 hours of the new
grant remain before other reservations; after its maximum ceiling, only about
2.69 hours remain. Thus keeping C1 and promising another 36-GPU-hour campaign
would double-spend the grant. A smaller follow-on can be priced after its
terminal receipt, but cannot be promised this proposal's full scope.

Recompute balances at execution time. Use the additive grant ledger and C1
finalizer, not the superseded queue reconciler, which does not know later debug
charges. Historical C2 remains 255/256 complete; its conservative rate screens
pass, but this plan does not complete its missing slot.

## Question and evidence contract

Main question: does the current public procedure correctly prepare, tune,
retain, reload, and assess HMC kernels when the target calls BayesFilter's
actual state-space likelihood and score implementations?

The numerical baseline is an independent dense Gaussian likelihood and
derivative check for LGSSMs, with closed-form or refined low-dimensional
integration references for selected parameter posteriors. For nonlinear
sigma-point inference, check the declared approximate likelihood and its
derivative against an independent implementation of the same approximation.
Compare a short-horizon nonlinear case with refined latent-state integration
only to describe approximation error, separately from sampler error.

The primary engineering criterion is passing call-chain, mathematical,
candidate-lifecycle, and persistence invariants. A scientific posterior result
additionally needs the existing declared numerical-health, equilibration,
retained convergence/precision, and independent-reference checks. Report these
criteria separately; neither criterion may stand in for the other.

| Diagnostic | Role and response |
| --- | --- |
| Likelihood/score disagreement, wrong chart/prior/Jacobian, corrupt identity or accepted-state inconsistency | Continuation veto for the affected lane; preserve evidence and repair before a new source-version attempt. Other valid lanes can continue. |
| Candidate-local invalid values, divergence, recurrence, or acceptance failure | Candidate promotion veto and possible existing directional-repair trigger. Do not invalidate healthy siblings automatically. |
| Warmup/retained R-hat, ESS, MCSE, reference agreement, start-group disagreement | Posterior promotion checks only. A failure triggers bounded diagnosis; it cannot change epsilon/L admission or tuning membership. |
| Runtime, GPU load, compilation counts, short-chain acceptance/ESS | Explanatory diagnostics and cost estimates. No statistical ranking or posterior validity claim follows. |
| Budget/deadline exhausted, missing usable reference, incompatible source | Stop the affected allocation and report incomplete or unevaluated. Never relax criteria to obtain a pass. |

The complete pipeline must preserve every verified candidate. Long posterior
sampling need not be duplicated for every member: predeclare one assessment
member per fit by the existing deterministic candidate-order rule, without
using posterior outcomes or apparent efficiency. On designated mechanism fits,
also assess a second distinct-L member. Price this exact workload. Other
members remain available and explicitly unassessed; they are not discarded.

Independent references are reporting/test authorities. They must not supply
starts, geometry, epsilon, candidate choice, or hidden stopping information.
Reference-based error screens do not establish nominal coverage at adaptive
stopping; that is a separate calibration question.
An explicitly separate fixed-kernel stationarity test may start at reference
draws by construction; it cannot be counted as a full-procedure fit.

## Test matrix

The main inventory is 32 full fits: 24 scalar SSM fits, four multivariate fits,
and four nonlinear fits. Four chains form one fit; they are not four independent
replications. The counts are proposed integration-coverage allocations, not a
power calculation. Freeze named datasets and sampler seeds before main runs.

| Case | Mechanism and reference | Main fits |
| --- | --- | ---: |
| K0: location, known dynamics/noises | Production Kalman target; exact conjugate posterior; dense Gaussian comparator. | 2 datasets x 2 sampler seeds |
| K1: persistence and observation noise, interior | Two unknown parameters; actual QR recursion; independently refined 2-D integration. | 2 x 2 |
| K2: persistence close to one | New explicitly parameterized chart reaching that regime; same total score including the declared initial law. | 2 x 2 |
| K3: small observation noise | Conditioning and numerical-status handling with finite, nonsingular model covariance. | 2 x 2 |
| K4: process and observation noise both unknown | Fixed persistence, short horizon, weak variance decomposition; 2-D reference. | 2 x 2 |
| K5: K4 with longer horizon | Time-recursion cost, more concentrated posterior, changed tuning scope. | 2 x 2 |
| K6: existing four-state/18-parameter LGSSM | Current SVD-filter call chain, charts, correlated parameters, lineage and posterior diagnostics; independent likelihood/score checks. | 1 existing dataset x 4 sampler seeds |
| K7: nonlinear accumulation model | Actual sigma-point filter; longer generated data; two declared free parameters with remaining parameters fixed, permitting a checked 2-D posterior reference for that approximate target. | 2 datasets x 2 sampler seeds |

For K7, the two-parameter restriction is a new named validation scope, not a
claim about the entire existing three-parameter fixture or arbitrary nonlinear
SSMs. The three-observation fixture remains a cheap arithmetic regression.
K6 has no established posterior oracle: without a separately checked reference,
it supports filter integration and reported posterior diagnostics, not posterior
accuracy or calibration. This limitation cannot be removed by calling the
likelihood exact. Exact MacroFinance validation still requires its actual data,
priors, target, and reference; synthetic SSMs cannot close that gap.

Before pricing, freeze a small regime table with equations, initial-state law,
priors, data seeds and horizon. Candidate engineering stress settings are
T=32/128, persistence 0.6/0.97, and observation/process noise-SD ratios 1/0.1.
These are proposed stress hypotheses chosen to expose time recursion,
persistence and conditioning, not repository defaults or realistic MacroFinance
calibration. The initial-law derivative must agree with the chosen model.
Do not falsely label the old 0.75-capped chart as K2. K6 retains its actual
T=120 fixture; K7 uses a longer horizon only after its target status and cost
check. Tests do not need the Cartesian product of every setting.

Separate dataset and algorithm randomness in manifests and reports. Within one
regime, two seeds on the same data describe algorithm repeatability; they do
not establish across-dataset Bayesian calibration. No pooled binomial rate
across these heterogeneous cases closes a 90/95% success or coverage claim.

## Cheap mechanism tests and implementation

Extend the existing validation infrastructure; do not create another tuner.
Add a small tested target-factory boundary used by `execute_pipeline`, then
register the actual filter adapters with their data, parameter transforms,
quantity names, telemetry and full source dependencies. Preserve existing
target behavior. Test the real endpoint-to-filter call chain, not merely an
import or a catalogue entry. Use TensorFlow/TFP in the executed target and
independent NumPy/SciPy only in diagnostic references.

Before costly fits, cover the following with focused regressions:

* Production-filter values against dense observation distributions; total
  scores against independent finite differences over several step scales,
  including initial-state dependence and constrained-coordinate Jacobians.
* Innovations/time indexing, singular or invalid covariance, nonfinite score,
  and wrong-score/omitted-data negative controls. A negative control must
  activate the check designed to detect it; generic command failure is not enough.
* Candidate-specific epsilon repair; no cross-L inheritance of qualification;
  fresh verification and retention of all valid siblings; high reported R-hat
  cannot reject a numerically qualified tuning candidate.
* Export/reload and unchanged-source continuation; data/prior/chart mismatch
  rejection; warmup exclusion and cumulative retained assessment; short or
  disagreeing chains cannot become a successful posterior result.
* Mean and quantile MCSE reporting against independent diagnostic calculations;
  insufficient batches/windows; numerical-health checks during discarded work.
  Calibration remains distinct from formula and wiring correctness.
* Cheap fixed-kernel stationarity checks on K0's exact posterior, using the
  real filtering target and the existing validation engine. The property tested
  is that integrating one transition against the posterior preserves that
  distribution. Price bulk transitions once, predeclare statistical tolerances
  and defective controls, and keep these replications separate from the 32
  full fits. They test the transition mechanism, not tuning or stopping calibration.
* Timeout under progress versus stall, cumulative accounting across retries,
  unused-budget sharing, interrupted reservations, missing results, and final
  process-group cleanup. Use controlled clocks/workloads for branch tests;
  spending hours manufacturing real contention is unnecessary.

Keep small Gaussian, beta-binomial and supplied-whitening funnel checks in
this tier. Do not train a new NeuTra map for it. Difficult global exploration
and learned canonical-map quality remain separately priced research work.

For sanity checks, construct per-regime outputs from the prior-only posterior,
a deterministic likelihood grid/profile estimate, and a local Gaussian/Laplace
approximation where its Hessian is valid. Compare these with the independent
oracle and the sampler on the same declared quantities. Gross disagreement
with the oracle while a simple comparator agrees triggers diagnosis. These
checks are not new tuning targets or a speed-ranking study; fitting a Laplace
approximation does not make it a posterior authority.

## Repeatable regression coverage

The implementation must leave reusable pytest coverage, not only campaign
outputs. Current tests already protect important parts of the proposed work:

| Existing regression file, relative to `tests/` | What it checks | Remaining scope boundary |
| --- | --- | --- |
| `test_hmc_candidate_set_tuning.py`, `test_hmc_candidate_set_artifacts.py` | All-member retention, same-L repair, fresh candidate identity, replay rejection and restart/accounting mechanics. | Controller fixtures do not demonstrate the actual SSM numerical call chain. |
| `test_hmc_candidate_set_execution.py::test_bad_rhat_is_persisted_without_changing_numerical_tuning_decision` | An unfavorable reported R-hat does not alter the actual numerical tuning decision. | Uses the existing numerical control; separate from posterior convergence requirements. |
| `test_hmc_candidate_runner_reuse.py` | Graph reuse, unchanged numerical streams, target/geometry isolation and public restart. | Existing numerical control is Gaussian. |
| `test_hmc_assessment_explanation.py` | R-hat/ESS/precision separation, insufficient windows/batches and explicit posterior failures. | Reporting correctness does not establish repeated-fit coverage. |
| `inference_validation/test_execution_policy.py`, `inference_validation/test_timeout_policy.py` | Unused-budget sharing, cumulative charges, policy propagation, progress/contention grace and outer deadlines. | Mostly controlled subprocesses/clocks; no runtime-tail guarantee. |
| `test_lgssm_generic_target_adapter_tf.py` | Actual QR filter adapter, batch/scalar parity and finite-difference score checks. | The existing fixture cannot reach near-unit persistence and is not a full tuning/posterior test. |
| `test_simple_nonlinear_generic_target_adapter_tf.py` | Actual admitted sigma-point filter adapters, value/score checks and route rejection. | The existing dataset has three observations. |
| `inference_validation/test_multimodel_pipeline.py` | Public prepared-route tuning, reload, retained assessment and warmup exclusion across six fixtures. | The LGSSM fixture evaluates a dense Gaussian likelihood rather than the production filter. |

Use three recurring levels. Run deterministic/controller/adapter regressions
on relevant changes; run bounded actual-filter public-pipeline regressions
after changes to target wiring, preparation, tuning, persistence or posterior
execution; run the larger statistical campaign deliberately after substantive
numerical-policy changes. CPU reference checks provide the inexpensive first
level. Actual GPU/XLA execution still needs its own bounded integration checks.

Stage one must add the missing actual-filter pipeline tests and named stressful
regimes to the existing suite, with a documented command and measured runtime.
Include both ordinary automatic preparation and the explicit prepared route;
do not let prepared-only tests stand in for the ordinary entry point. Use fixed
seeds and exact invariants for regression assertions. Statistical checks need
declared tolerances; repeated reruns until they pass are forbidden. Every repair
must retain a reproducer that fails on the defective behavior. The 32-fit
campaign is a separate validation tier, not a per-commit test requirement.

An existing subset is now repeatable with
`python -m pytest @docs/validation/hmc-regression-tests.txt` in the `tfgpu`
environment, with GPUs hidden and CPU thread limits as documented in
`docs/validation/README.md`. Its 127 tests passed, with no failures or skips,
in 160.644 enclosing seconds on September 25. The
[verification receipt](artifacts/hmc-state-space-regression-check-20260925T153023079222Z/result.json),
manifest, JUnit results and log preserve the exact executed selection. This
routine CPU verification does not activate the proposed campaign, alter C1,
or close the missing actual-filter public-pipeline coverage.

## Original schedule and resource ceilings

Let T0 be the start of implementation/execution of this amendment, not the
later start of its longest run. All preparation, failed attempts, reference
work, repairs, waiting, and reporting fit within T0+48 hours. Ordinary tool or
human delays do not reset that clock. If the amendment is not activated, no
48-hour execution clock or compute reservation is claimed.

| Time from T0 | Work | GPU worker-hours, maximum | CPU reference/test worker-hours, maximum |
| --- | --- | ---: | ---: |
| 0–6 h | Factory/call-chain repair, references, focused mechanics | 0 | 6 |
| 6–10 h | Trusted GPU/XLA parity and complete-fit prices | 3 | 2 |
| 10–34 h | Frozen main matrix | 22 | 3 |
| 34–42 h | Failure localization and bounded fresh-version retests | 6 | 3 |
| 42–46 h | Final affected checks, official guide, result and accounting | 1 | 2 |
| Within 6–46 h | Protected contention/repair reserve | 4 | 0 |
| 46–48 h | Shutdown, accounting and terminal-report margin | 0 | 0 |
| **Total** | **One selected GPU; bounded CPU reference workers** | **36** | **16** |

These are allocation caps chosen to fit the current grant and leave headroom,
not measured runtime predictions. The four reserve GPU hours must fit inside
the same wall schedule. At most two CPU reference workers run concurrently;
sum their worker times. GPU-worker host overhead is included in its enclosing
GPU-worker receipt, following existing ledger accounting; also record actual
CPU use. Do not count nested stages twice or describe wall hours as core-hours.
All reference and test attempts consume the CPU allocation.

Main-matrix suballocations are at most 14 GPU hours for K0–K5, four for K6,
and four for K7. They are not evidence those models will fit. The pricing stage
must measure representative complete cold fits, including preparation, graph
compilation, broad L search, verification, export/reload, the declared member
assessments, artifacts and shutdown. A fast log-density or leapfrog benchmark
cannot price a complete fit. Development prices use distinct seeds and do not
silently enter the main denominator.

For a lane, provisionally require `planned fits * maximum complete pilot cost
* 1.5 <= lane allocation`. The factor 1.5 is an explicitly uncalibrated planning
margin, not a runtime quantile or guarantee. Freeze cumulative per-fit allowances
from the same rule before main launch; share only genuinely unused settled time.
One complete pilot per distinct execution shape is the minimum price; attempt
a second under ordinary observed load when the three-hour pilot allocation
allows. Record that one or two prices cannot identify runtime tails.

If pricing is incomplete or exceeds a ceiling, diagnose within the remaining
pilot/repair allocation. Do not launch the whole matrix on a censored cheap
price. A lane can finish as unaffordable or reference-unavailable while the
other lanes proceed. Do not silently shorten warmup, change the model/horizon,
drop failed seeds, or substitute prepared-only tuning to make the study pass.
An explicit reduced inventory is a new engineering study, not the original
32-fit result. This proposal needs no 32/32-success promise to finish on time.

## Runtime and statistical discipline

Use the repaired optional leapfrog-graph reuse and sequential budget sharing
after verifying propagation and parity through the real SSM child executor.
Reuse compiled numerical functions where already supported; reset adaptation,
RNG, target scope, and posterior state for each independent fit. Do not assume
cross-process or cross-dataset caching exists, or build a new cache framework
before profiling shows it necessary. Keep verbose profiling out of main runs.

GPU runs use the selected trusted GPU, TensorFlow/TFP, XLA and verified memory
growth before initialization. Record GPU identity, dtype/TF32, compiler and
load. CPU-only reference/tests explicitly hide GPUs. Never interfere with other
users' GPU jobs. Waiting for capacity consumes wall time even when it consumes
no GPU-worker time. Progress/load-aware grace remains bounded by fit, lane,
grant and T0+46-hour compute ceilings; it cannot extend the deadline.

Preserve the broad L grid, per-candidate repair, fresh verification and all
verified members. Do not transplant C1's 30,000-transition warmup minimum to
SSMs. Start from the documented ordinary sequential policy (2,000 minimum,
1,000-window readiness, 10,000 warmup maximum; inherited policy, not universal
burn-in sufficiency), with explicit posterior precision requirements and
bounded extensions in the frozen design. R-hat, ESS and MCSE remain posterior
criteria, never epsilon/L tuning requirements. Preserve failed readiness at
the cap. A smaller minimum requires its own evidence, not a time-saving guess.

Before pilots, freeze the quantities, existing posterior-policy settings,
reference-error checks and scientifically interpretable accuracy tolerances
in each design. Scale tolerances to the declared estimands and noise/model
units, with rationale; do not invent one unexamined absolute tolerance for all
parameters. Require integration-domain expansion and resolution refinement;
preserve the observed reference-error estimate and tail checks. Refinement
agreement alone is not a rigorous error bound. Failed reference checks yield
no posterior-accuracy verdict. References never change the sampler's settings.

Finite-sample MCSE/R-hat/ESS summaries and differences among these few fits are
descriptive. Report missing delivery, posterior checks, numerical/reference
discrepancies and interval coverage separately, retaining all planned outcomes.
No stochastic winner, calibrated success probability, burn-in guarantee, or
new repository-wide default follows from this campaign.

## Skeptical review and refresh between stages

The first review found four material flaws in a naive 48-hour proposal:
allocating the live C1 reservation twice; treating the dense `lgssm_location`
target as filter integration; assuming old adapters already test near-unit or
long nonlinear series; and treating a few seeds as calibration confirmation.
The funding boundary, factory repair, new named regimes and limited claims
above correct those flaws. Exact likelihood versus exact parameter posterior,
approximate likelihood versus true-model inference, source-version changes,
selected-candidate bias, and false speed forecasts are also explicitly handled.

The plan passes review as a bounded engineering and diagnostic campaign.
Its unmeasured costs mean successful completion of all scientific cases is
conditional. Its numerical design is not launch-ready until stage one records
model-specific equations, priors, seeds, tolerances and posterior policies;
stage two must produce affordable full-fit prices. These are execution tasks,
not requirements for another chain of human approvals. No long launch is
authorized merely by writing this proposal.

The count and schedule audit gives 24+4+4=32 main fits, 32+4=36 GPU hours
including reserve, 16 CPU worker-hours, and a 46-hour compute deadline inside
the 48-hour terminal deadline. Both compute allocations fit the inspected
balances only under the stated C1 reallocation. All other reservations remain
separate. The alternative of preserving C1 is explicitly costed rather than
quietly treated as free background work.

At each stage, refresh one `checkpoint.json` with completed evidence, failure
classification, remaining money-free compute allowance, next bounded action,
and whether an issue is a candidate failure or a continuation veto. Preserve
failed attempts under their original sources; fixes receive new source and
attempt identities. A stopped or replaced fit never becomes an independent
new success in the original denominator. Stop creating new work at T0+42 hours;
use the closing allocation for checks and reporting, and stop compute by hour 46.

A pre-mortem: every command could pass while testing only benign charts,
shared reference bugs, or a biased approximate likelihood. The call-chain,
independent-reference, stressful-regime and approximation checks are the early
discriminators. Missing convergence could instead reflect geometry or weak
identification; report that distinction and the bounded repair attempted.
If a real unwhitened funnel appears, a suitable supplied map is an upstream
requirement, not permission to tune indefinitely.

## Artifacts and completion

On activation use a fresh root
`docs/plans/artifacts/hmc-state-space-48h-<start-date>-r1/`. Record the exact
environment/executable, source commit and diff, data/target hashes, command,
separate data/sampler seeds, CPU/GPU/memory settings, wall times and receipts.
The runner should use the existing `inference_validation` execution and
supervision code. Add and test any required factory/design wiring before
recording the exact launch command; an invented runnable CLI is not evidence.

Deliver `design.json`, `manifest.json`, `checkpoint.json`, `costs.json`,
per-fit and independent-reference results, `result.json`, and one result note.
Update the official tuning chapter included by `docs/main.tex` and its API
reference with actual supported coverage and remaining limits; do not create
a competing guide. Update the master after each completed stage.

The terminal decision table must distinguish engineering correctness,
numerical validity, posterior delivery/accuracy, uncertainty, and next action.
The inference-status table must explicitly cover hard vetoes, supported ranking
(normally none), descriptive differences, default-readiness (not established),
and next evidence needed. Include the strongest alternative explanation and
the weakest evidence. Leave C1/C2 calibration completion, learned-map quality,
difficult global exploration, and exact missing MacroFinance integration open
where their own evidence is absent.

When this proposal was first drafted, no experiment, process cancellation,
reservation change or runtime-code edit was performed. The implementation
amendment below records the subsequent owner-authorized work.

## September 26 implementation audit and active work

The first bridge selection passed 87 CPU tests, including ordinary and prepared
QR and nonlinear calls. This is engineering evidence only. A skeptical audit
rejected the initial generated suite: it substituted repeated small fixtures
for K0/K4/K5/K6, reduced warmup to 512, mislabeled synthetic sine panels as
stressful data, lacked the promised K7 reference, and treated finite inputs as
filter-health telemetry. The initial suite is an unlaunched draft, not the
32-fit experiment above. Correct it before launch; retain its debugging history
in this note rather than using it as a new baseline.

The implementation question remains the actual-filter public procedure in
K0--K7. Complete these tasks while C1 runs: named exact model/data definitions;
all eight target adapters; independent scalar/dense Kalman and sigma-point
references; reference domain/resolution checks; real value/score/status and
source identity; focused negative controls; ordinary/prepared integration;
frozen suites for checks, full-fit pricing and the 32 main slots; bounded
pricing/deadline/accounting preparation. Use the existing executor and public
tuners. No new GPU work starts during this preparation.

Numerical hypotheses for new scalar profiles: T=32 (K0--K3), T=16 (K4),
T=128 (K5), persistence 0.6 with 0.97 for K2, process SD 0.2,
observation SD 0.2 with 0.02 for K3. These implement the proposed regime table;
they are synthetic stress choices, not fitted defaults. Use a stationary zero
mean initial law with its parameter derivatives. The persistence chart is
0.999*tanh(raw), so 0.97 is reachable. Priors live on raw coordinates:
location N(0,2^2), persistence N(atanh(0.6/0.999),1), log noise SDs
N(log(0.2),1). Thus no extra transform Jacobian belongs in these raw-coordinate
priors. Report transformed parameter names accurately. Data seeds are fixed
before fitting; starts use priors only. K7 holds innovation loading at 0.25
and estimates rho/beta with the existing independent N(0.70,0.20^2) and
N(0.80,0.20^2) priors. K6 retains its existing complete target and prior.

Retain the inherited 2,000/1,000/10,000 warmup minimum/window/cap and retained
cap 10,000. Mean and median precision requirements are declared per parameter
in model units, before pricing. The broad L grid and native acceptance policy
remain unchanged. Tiny regression allocations have explicit mechanics-only
roles; they cannot supply the serious campaign's convergence or runtime price.

Reference checking must compare model quantities over a coarse/fine grid and
an expanded integration domain, preserving diagnostics and refusing an accuracy
verdict when sensitivity exceeds its declared allowance. A successful numerical
reference check remains empirical error evidence, not a rigorous error bound.
K6 has independent value/score checks but no joint posterior oracle. Nonlinear
sampler error is relative to the same sigma-point target; latent integration
describes approximation error separately. Neither K6 nor the legacy nonlinear
fixture may acquire an invented posterior oracle.

Audit disposition: revised implementation work may proceed. Main execution
requires C1 settlement, current-source GPU/XLA checks, uncensored full-fit
prices and affordability within the actual remaining grant. Expected candidate
or posterior failures trigger bounded diagnosis; corrupted scope/reference or
numerics stop the affected lane. CPU development/checks are capped at six
worker-hours from the existing CPU balance, including failed checks; no GPU
reservation is made. Artifacts use a fresh
`artifacts/hmc-state-space-preparation-2026-09-26/` root. This preparation is
separate from the later 48-hour run clock; report both times, never claim the
preparation happened inside a later run's wall time.

## September 26 preparation result and launch readiness

The rejected preliminary suite has been replaced by
`docs/validation/state-space-48h.json`, generated from the framework-free
campaign planner. It contains exactly 32 confirmation slots: K0--K5 each have
two frozen datasets and two sampler seeds, K6 has four seeds on the existing
T=120 multivariate fixture, and K7 has two frozen datasets and two sampler
seeds. The inherited broad L grid is `(3,5,9,13,18,25)`. Each slot retains
every verified candidate; the predeclared posterior assessment selects two
distinct-L members for K0 and K7 and one for other cases. No selection depends
on R-hat, ESS, MCSE or posterior output.

CPU preparation artifacts are under
`docs/plans/artifacts/hmc-state-space-preparation-2026-09-26/prepared-r1/`.
The 15 frozen dataset records include independent reference sensitivity checks
for all oracle-backed cases. K3 dataset A required the declared 321-point
refinement; that result is retained as evidence rather than silently accepted
at the coarse grid. K6 records `reference_checked: false` because no checked
joint posterior oracle exists. The true-model K7 Gauss--Hermite diagnostic at
T=2 refined from order 15 to 21 by `7.2e-6`, below its declared `0.005`
approximation-screen tolerance; it is separate from the sigma-point sampler
target and does not establish long-horizon posterior accuracy. Prior, local
positive-Hessian and grid-mode comparators are descriptive only.

The final broad selection passed 142 tests in 707.7 enclosing seconds on the
frozen r5 package, including all eight ordinary/prepared public-route cases,
an isolated real worker, all eight model value/score checks and CPU XLA
equivalence for all five distinct execution shapes. All eight public-route
cases produced verified candidates. A further 24 focused checks passed in
the final current-source check, covering the final controller changes and the
strict K0 ordinary/prepared/isolated cases. These are overlapping selections,
not 166 unique tests. Most eight-transition smoke cases stopped before retained
sampling; K0 prepared exercised a nonempty retained archive. Their unavailable
precision diagnostics remain failures, not posterior evidence. See
[the preparation result](bayesfilter-hmc-state-space-preparation-result-2026-09-26.md)
for receipts, repairs and accounting.

The separate K0 stationarity diagnostic completed in 23.4 CPU worker-seconds.
Baseline and no-op controls had no detected discrepancy; the deliberately
wrong energy was detected. This is a fixed-kernel diagnostic, not calibrated
detector power, full-procedure coverage, or convergence evidence.

The executable controller is
`scripts/prepare_hmc_state_space_campaign.py` (implementation in
`bayesfilter/testing/inference_validation/ssm_campaign.py`). Its `prepare`
stage is CPU-only. Its `freeze` stage records the live package and K6
dependencies. After C1 becomes inactive and its enclosing receipt is present
exactly once in the additive grant ledger, `run --stage
preflight-mechanics`, `run --stage preflight-pipeline`, `run --stage pricing`,
and `run --stage main` use the existing executor in a fresh frozen-source
directory. The GPU controller requires a UUID-selected device, TensorFlow
memory growth before initialization, XLA placement evidence and process-group
cleanup. Pricing accepts only an uncensored complete fit with the same warmup,
retained-member and candidate workload as the main lane. A 1.5 planning margin
is applied to the measured pilot; a lane that cannot fit the actual settled
grant remains explicitly unfunded in the original 32-slot report. No automatic
competing launch or C1 finalization is performed by preparation.

The launch package is frozen at
`artifacts/hmc-state-space-preparation-2026-09-26/prepared-final-r1/`.
Its datasets and reference receipts are copied unchanged from the checked
preparation; suites are rebuilt from those datasets with the final planner.
The derivation and ordinary file hashes are recorded, so another reference
experiment is unnecessary. Use its frozen launcher as shown below. No campaign
GPU work starts while C1 is active. A preflight pass is engineering evidence only. Any candidate,
posterior, reference or budget failure after launch must be classified per the
question/evidence table above and retained as a slot result; it cannot be
converted into a success by shortening warmup or relaxing the reference.

### Final default audit and executable stages

The post-C1 run clock starts at the first GPU stage; the CPU preparation time
and six-worker-hour preparation ceiling are recorded separately. The six
preparation worker-hours count within the original 16-CPU-hour total, leaving
at most ten CPU worker-hours for the later campaign, further constrained by
the actual CPU balance. Early checks did not all preserve enclosing timing;
the entire six-hour preparation allocation is charged conservatively, with
measured subreceipts reported separately rather than double-counted. The original
36-GPU-hour figure is a scope ceiling, not a new grant. The three-hour preflight
and pricing cap includes 480 seconds for eight mechanics cells, 720 seconds for
four public bridge cells, 9,480 seconds for eight complete-fit pilots, and
90 seconds of enclosing shutdown allowances: 10,770 seconds in total. These
are convenience allocations to bound failure cost, not measured fit prices.
The 22-hour main ceiling and four-hour reserve are further limited by C1's
settled actual costs and the separate 1,200-second NeuTra reservation. New work
stops at hour 42 and computation at hour 46, leaving two hours for reporting.

| Choice | Provenance and purpose | Failure/early diagnostic | Status |
| --- | --- | --- | --- |
| Model horizons, priors, charts and synthetic noise scales | Explicit stress hypotheses above; frozen in `ssm_campaign_profiles.py` | Dense/scalar likelihood and two-step-size FD checks; real generated observations | Validation models, not general defaults |
| Data seeds `(20260926,100+10*K+dataset)` and hashed sampler streams | Reproducibility choice, fixed before outcomes; K6 keeps `(20260709,301)` | Separate identities in each manifest; no seed search | Convenience choice |
| Native ordinary search and acceptance, L=(3,5,9,13,18,25), 128 evidence draws | Existing ordinary procedure; 128 is a finite validation allocation with native rungs | Empty/inconclusive candidate set remains a result; pilot cannot price it as full delivery | Inherited procedure, no new default |
| Prior-based four-chain start offsets -1,-0.3,0.4,1 | Existing validation convention; K6 prior happens to center on the synthetic truth template | Poor initialization remains a possible failure; no posterior oracle supplies starts | Warm-start hypothesis |
| Posterior 500-transition chunks, 2,000 warmup minimum, latest 1,000 window, 10,000 warmup and retained caps | Existing sequential policy with finite checking cadence; retained minimum 1,000 | R-hat/ESS/MCSE may stop posterior promotion; no tuning effect | Inherited policy and convenience cadence |
| Mean/median absolute MCSE 0.01 for SD quantities, 0.02 for other named quantities; lugsail means | Synthetic reporting precision in model units, declared before fitting | Unavailable batches or precision caps remain failed delivery; not calibrated coverage | Engineering hypothesis |
| Bounded atan discrepancy tolerance 0.02 | Descriptive independent-reference alarm, not posterior correctness proof | Inspect combined MC uncertainty; never rank methods from this screen | Diagnostic hypothesis |
| Reference mean/median sensitivity 0.002, atan 0.001, boundary mass 1e-5, grids 161/321/641 over 6 and 8 prior SD | Finite-resolution/domain checks, smaller than reporting tolerances | Refuse the reference if the declared refinement ladder fails; no rigorous error bound | Numerical diagnostic hypothesis |
| K7 latent orders 9,15,21 at T=2 and log-likelihood refinement tolerance 0.005 | Bounded tensor Gauss--Hermite diagnostic to separate approximation from sampler error | Refinement failure means unavailable approximation diagnosis, not sampler failure | Diagnostic hypothesis |
| Prior/Laplace 16,384 draws, Hessian FD 1e-4 | Cheap comparison ladder; local positive Hessian required | Non-PD Hessian or optimizer failure yields unavailable Laplace; never tuning input | Descriptive convenience |
| K0 stationarity: 512 anchors, epsilon 0.12, L=3, seven rank transitions, 1,999 null draws, alpha .05 | Existing conditional fixed-kernel test, separate from whole-fit evidence | Baseline/noop and wrong-energy control reported with existing multiplicity; lack of detection does not establish power | Diagnostic hypothesis |
| Pilot price multiplier 1.5; nominal 1.25 portion and bounded 0.25 contention grace | Plan's uncalibrated cost margin, including shutdown allowance | No censored price; receipt checks, progress observer, grant/deadline enforcement | Planning heuristic, not tail guarantee |

The skepticism audit also checked wrong baselines, inadequate references,
unpriced compilation, hidden GPU fallback, workload mismatch, biased member
selection, accidental source changes and denominator shrinkage. The corrected
controller checks actual GPU output placement; snapshots include package and
K6 data dependencies; measured pilots must match the main numerical policy and
declared member workload. CPU tests exercise quota, failed/censored pilot,
partial funding and enclosing-receipt cases. No additional human approval is
needed for routine local repairs inside the unchanged contract and remaining
budget. A true numerical/reference/source failure stops its affected lane
until repaired; it does not reject the state-space research direction.

### Final preparation audit

The final CPU selection passed 142 cases on the frozen r5 package. The eight
public-route cases all produced verified candidates; the temporary allowance
for an empty positive test was not taken, and has been removed. The repaired
K0 quota and current controller passed 20 focused checks. Most tiny pipeline
cases exhausted their eight-transition warmup cap without retained draws.
These cases establish invocation/reload and honest unavailable diagnostics;
only the case with actual retained draws establishes the retained archive path.
Complete-fit pricing must reject a warmup-only outcome, even if it is labeled
assessed. No criterion is relaxed to force posterior delivery.

Before the final freeze, focused tests verified the already implemented
same-GPU restriction and the repair for a failed stage returning a failing
command exit status. The latter is an orchestration repair, with no change to target,
baseline, sample counts, statistical criteria or compute allocation. The
skeptical audit checked stale package snapshots, changing live source during
identity-sensitive tests, accidental GPU initialization, original denominators,
the active C1 reservation, and false success from a zero launcher exit code.
The bounded CPU checks and frozen-source handoff answer these engineering
questions; a further full sampler rerun is unnecessary unless a new failure
appears. No GPU command is part of this preparation.

### Launch after C1 settlement

The final prepared root includes `source.json`, `source-diff.patch`, the
frozen `source/` package and launcher, dependency/input hashes, the original
32-slot inventory, reference receipts, stage suites and a preparation manifest.
`result.json` initially preserves all 32 slots as unstarted. The launch command
below must run with trusted GPU permissions from the repository root. The GPU
UUID is inherited from the C1 launch manifest, not a fresh device probe. Check
its availability after C1 stops. The existing C1 service finalizer normally
settles its enclosing receipt; if settlement needs repair, inspect the terminal
receipt before invoking `finalize_c1.py`. Never invoke that finalizer while C1
is active. The SSM controller refuses an active service or unsettled grant.

```bash
SSM_ROOT=/home/ubuntu/python/BayesFilter/docs/plans/artifacts/hmc-state-space-preparation-2026-09-26/prepared-final-r1
CUDA_VISIBLE_DEVICES=GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c \
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  "$SSM_ROOT/source/scripts/prepare_hmc_state_space_campaign.py" \
  run "$SSM_ROOT" --stage all \
  --ledger /home/ubuntu/python/BayesFilter/docs/plans/artifacts/hmc-additional-gpu-2026-09-25/grant-ledger.json
```

`all` executes the two engineering preflights, the eight full-fit pilots, then
prices and runs affordable main cases. It exits unsuccessfully on a failed
stage and always writes a 32-slot report. Same-source GPU checks and complete
pilot member workloads are checked before funding; a warmup-only pilot cannot
price retained sampling. Whole cases are funded in declared K0--K7 order;
all other slots remain explicitly unfunded. This order is a simple engineering
allocation, not a statistical ranking. A failed stage is preserved for bounded
diagnosis, without automatic retries that reset its budget. Nothing in CPU
preparation starts a background watcher or a competing GPU process.

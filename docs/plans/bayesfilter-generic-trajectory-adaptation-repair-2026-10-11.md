# Generic trajectory adaptation repair for HMC and MAMS

Date: 11 October 2026. Status: **design reviewed; implementation pending**.
This is the D4R repair subplan of the
[modern-method master](bayesfilter-modern-hmc-implementation-plan-2026-10-07.md).
It answers the owner's request for a repair that generalizes beyond
predator–prey. No numerical policy, kernel, guide, historical result or
candidate membership is changed by this document. No experiment was launched
to prepare it. The official guide remains the monograph and its agent-facing
interface reference; this document is an implementation plan.

## 1. Diagnosis and scope

The demonstrated gap is **missing method-specific exploration adaptation**,
not a demonstrated error in MAMS's Metropolis correction, nor evidence that
acceptance-qualified candidates must all deliver posterior precision within
10,000 transitions. The SGQF run found five candidates that passed its declared
posterior checks. Four did not: one exhausted warm-up and three exhausted
retained sampling without meeting precision. A useful candidate set was found,
but its search did not implement the adaptation promised by master section 6.3.

Checked evidence, relative to the repository root:

* `scripts/price_modern_hmc_method_proposals.py`: MAMS used the Cartesian grid
  of maximum counts 3, 9, 18 and epsilon 0.2, 0.5, 0.8.
* `scripts/run_modern_hmc_method_fit.py`: repairs per family were zero.
  Its epsilon domain was [0.001, 1]; that inherited domain is not an
  established MAMS stability range or evidence that larger steps are useless.
* `docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/`
  `method-full-fit-01/nl-pp-sgqf-mams/search-design.json`: pilots disabled,
  no refinement rounds or expansion grid, nine-candidate cap. Qualification
  used a lower-bound acceptance region [0.5, 1], not a search for efficiency.
* The corresponding `tuning/candidate_set_result.json` records nine verified
  candidates and zero repair actions. Measurement and verification each used
  32 reset-start trials of 64 transitions per chain. Those trials estimate
  the declared conditional acceptance quantity; concatenating them would not
  create a continuous exploration pilot.
* `bayesfilter/inference/hmc_trajectory_proposals.py` implements discarded
  ChEES/SNAPER proposals for randomized HMC, but no MAMS adaptive proposer.

The posterior results, in candidate creation order, are:

| Maximum count | Epsilon | Warm-up/chain | Retained/chain | Maximum final lugsail MCSE/SD | Outcome |
| ---: | ---: | ---: | ---: | ---: | --- |
| 3 | 0.2 | 10000 | 0 | unavailable | Warm-up inconclusive |
| 3 | 0.5 | 3000 | 10000 | 0.0306341 | Precision not met |
| 3 | 0.8 | 2000 | 10000 | 0.0236053 | Precision not met |
| 9 | 0.2 | 2000 | 10000 | 0.0398212 | Precision not met |
| 9 | 0.5 | 2000 | 7000 | 0.0190970 | Passed |
| 9 | 0.8 | 2000 | 4000 | 0.0176060 | Passed |
| 18 | 0.2 | 2000 | 10000 | 0.0181117 | Passed |
| 18 | 0.5 | 2000 | 3000 | 0.0177349 | Passed |
| 18 | 0.8 | 2000 | 2000 | 0.0179782 | Passed |

Source: `posterior-00/result.json` through `posterior-08/result.json` under
the same root. Precision comes from the final retained check's
`modern_rhat.precision.targets[*].mcse_sd_ratio`; the criterion was 0.02.
The top-level FFT-based MCSE ratios are different estimates and must not be
reported as the lugsail stopping statistic. These thresholds and counts
describe that historical study, not new generic defaults.

High acceptance, approximately 0.986–1.000 in verification, did not certify
useful travel. The shortest candidate also had full movement under the
current movement check. In `hmc_verification._tuning_telemetry_tensors`,
movement means exceeding a numerical tolerance scaled by the same short
trial's standard deviation. Endpoint return is also normalized by that
trial's spread. Neither establishes exploration of the prepared target scale.
The endpoint threshold was zero; increasing it alone is not the repair, and
could wrongly reject healthy oscillating trajectories.

The six-parameter SGQF target had local affine preparation, explicitly without
established global coverage. Low dimension does not establish easy geometry.
Observed association between longer paths and delivery is descriptive, from
one cohort. It neither identifies a universal minimum duration nor proves
that duration alone caused every failure. Ten thousand transitions represent
different amounts of gradient work at different trajectory lengths.

## 2. Research intent and evidence contract

**Question.** Can a single shared tuning procedure, with method-specific
discarded proposal adaptation, find useful frozen policies more reliably
under a declared work budget, without changing the target or weakening
qualification and posterior checks?

**Mechanism.** Separate integrator resolution from physical travel, collect
continuous fixed-policy pilot windows, nominate multiple duration scales,
and freshly qualify the resulting finite candidate set. The expected failure
being repaired is high numerical acceptance with negligible useful travel,
including one slow direction hidden by average movement.

**Comparators.** For causal localization, retain the current frozen-grid
MAMS procedure as a diagnostic baseline and compare it with the adaptive
proposer using the same target, geometry and starts. Separately compare
adequately searched fixed/randomized HMC, SNAPER-proposed HMC and MAMS. Include
the declared pilot and preparation costs. The old SGQF outcomes are development
evidence, not untouched confirmation or proof that MAMS beats classical HMC.
NUTS is excluded, including preparation, diagnostics and comparators.

| Evidence or event | Role | Consequence |
| --- | --- | --- |
| Target/score correctness, transition identity, correction oracle, stream separation and replay | Engineering pass criterion; shared defect is a continuation veto for affected work | Repair before interpreting affected experiments |
| Nonfinite values, incorrect filter status, broken correction or missing numerical evidence | Numerical veto | Preserve failure; continue unaffected candidates only |
| Existing independently replicated acceptance and health qualification | Tuning membership criterion | Retain every verified candidate; an incomplete verification is not a pass |
| Discarded fixed-policy pilot displacement, autocorrelation and work | Proposal/repair diagnostic only | Nominate new policies or report inconclusive exploration; no qualification or convergence authority |
| Posterior R-hat, bulk/tail ESS, declared lugsail precision and independent target agreement | Posterior-delivery criterion and promotion veto | Report delivered or unmet; do not rewrite tuning membership |
| Matched-budget delivery across fresh replications and uncertainty in paired differences | Robustness comparison criterion | Support an improvement claim only if the predeclared analysis supports it |
| Budget exhausted, unreliable pilot estimate, duration-domain boundary | Resource stop or repair trigger | Record why; no scientific rejection from this alone |
| Invalid target/reference, corrupted provenance, forbidden backend, unresolved shared implementation defect | Continuation veto for dependent runs | Diagnose and repair before resuming those runs |

A completed posterior failure can motivate a **new development round** with
fresh seeds and evidence. It cannot select, repair, rank or delay members
inside the already frozen tuning search. This distinction prevents the
no-R-hat-tuning rule from accidentally prohibiting research into failures.
Any target, geometry or preparation change requires a new scope and fresh
qualification. Preserve prior failures and never tune on confirmation draws
then report those same draws as untouched confirmation.

Success of the repair is an engineering claim first. A statistical robustness
claim additionally requires fresh end-to-end evidence across targets. No
finite pilot guarantees equilibration for every model, validates an unknown
remote mode, establishes exact nonlinear likelihood truth, or promotes a
new default. The result must distinguish an empty verified set, verified but
poorly exploring policies, unavailable evidence, and unmet posterior precision.

## 3. Architecture: retain one procedure

Reuse `HMCTuningCandidateSetController` and the public authorities
`tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel`. Consult
`HMC_TUNING_INTERFACE_CAPABILITIES`; no additional artifact-authority tuner is
needed. Add a method-specific proposal implementation behind a shared, typed
discarded-pilot interface. Consumer scripts configure it; they do not implement
their own scientific search. Unsupported family/coordinate/score combinations
fail before work. In particular, arbitrary position-only forces cannot enter
MAMS, and one-dimensional MAMS remains unsupported.

Keep four separate records: numerical verification, pilot exploration
evidence, scheduled posterior allocation, and posterior delivery. Verified
slow candidates remain verified. Pilot hypotheses that were deduplicated or
not nominated are not silently described as rejected tuning candidates.
All declared families receive their planned proposal opportunity; the first
passing candidate does not terminate cohort discovery.

### 3.1 Separate step size from travel without changing the kernel law

The smallest repair can keep the existing frozen transition schema. Introduce
an explicitly named **requested mean physical duration**, ell, in proposer
configuration/provenance, independent of epsilon. Compile it into the current
integer maximum count and record the actual induced law.

For the existing uniform-count implementation, N is uniform on {1,...,K}:

    tau_max = epsilon K
    E[tau] = epsilon (K + 1) / 2
    E[tau^2] = epsilon^2 (K + 1)(2K + 1) / 6.

Thus epsilon K/2 is not the exact mean. For ell >= epsilon, nominate the
nearest admissible integer to 2 ell / epsilon - 1, with a documented
deterministic tie rule. With no active count cap, the mean quantization error
is at most epsilon/4. For fixed trajectories, use nearest integer
ell/epsilon, with error at most epsilon/2. Treat ell < epsilon or a count-cap
violation as an explicit unsupported proposal requiring a smaller epsilon
or a separately recorded feasible sibling; never silently clip and claim the
requested duration was achieved. Test integer boundaries in the actual dtype.

Whenever epsilon changes during adaptation, recompute K to approximately
preserve ell. Whenever ell changes, recalibrate epsilon and verify fresh.
Store requested and realized means, support, rounding, quantization and work.
Candidate identity/replay remains based on the executed frozen specification;
distinct requests compiling to the same policy deduplicate transparently.
There is no claim of exactly independent continuous duration in this first
repair. A future average-preserving or fractional-last-step law would require
its own versioned law and correction/replay tests, and is not a prerequisite.

The existing controller's `_request_repair` changes epsilon while keeping
`candidate.leapfrog_steps` fixed. Turning that repair switch on by itself
would therefore change travel and cannot implement this design. Add an
explicit method-aware proposal policy that carries requested duration and
recompiles the count for a repair child. Keep the legacy fixed-count repair
semantics unchanged for old configurations. Audit repair actions, lineage,
candidate identity, work reservation, export and reload for their current
same-count assumptions; version those records if their meaning changes.
The resulting child needs fresh evidence for its whole executed policy.
An epsilon bracket is a proposal heuristic, not a proof of monotone
acceptance across quantized count changes.

Why this matters: full-refresh MAMS has a unit direction. At sufficiently
short physical duration, q_new - q is approximately tau times that direction.
For an isotropic direction, E[u_j^2]=1/d, so the leading coordinate displacement
variance is E[tau^2]/d before rejection and force curvature corrections.
Tiny steps with frequent full refresh can therefore make very small moves
while having acceptance near one. This local small-duration explanation is
not a global mixing theorem. HMC momentum has different speed scaling, so
the same numerical duration need not have the same interpretation across
families. Use family-specific initialization, not copied predator–prey values.

### 3.2 Implement the missing MAMS discarded proposer

Use the pinned paper's deterministic full-refresh adaptation as the initial
source-resolved proposal hypothesis. Geometry remains fixed. Establish a
stable epsilon with bounded bracketing/dual averaging at a declared physical
duration. Collect continuous pilot windows at **frozen** epsilon and duration;
do not estimate autocorrelation by concatenating changing-policy adaptation
draws, reset-start trials, or distinct chains. All pilot draws are discarded.

The paper's Section 6, equations (12)–(14), proposes

    ell_new = c ell estimated_integrated_autocorrelation_time,

with coordinate aggregation and a Gaussian-calibrated c. Its sqrt(d) initial
mean length and 0.9 acceptance target are hypotheses with source provenance,
not generic guarantees. The paper's 0.3 deterministic coefficient is also a
hypothesis. Keep it explicit; use a bounded neighborhood around source
proposals and record domains, update limits and omissions. Retune epsilon at
each nominated duration. High acceptance qualifies under the declared
one-sided rule but does not terminate this independent proposal phase.

Derive the initial epsilon bracket from target-specific discarded stability
probes and allow bounded expansion inside a predeclared scientific/work
domain. A bracket may expand toward larger epsilon when acceptance remains
high, at approximately fixed physical travel, but never beyond the declared
scope. Do not freeze epsilon <= 1 or any SGQF count limit as a universal
MAMS default. Reaching a configured boundary is reported as domain-limited
exploration; widening that domain after freeze requires a new search/scope
where the existing identity contract binds it.

The archived author top-level routine uses newer average-step pinning and
disables its final length stage by default; this differs from the paper's
described adaptation. Do not mix those defaults and call the result faithful.
Record which equations and author subroutine are used and classify local
safeguards as adaptations. Repository geometry preparation replaces the
paper's NUTS-based preparation; that replacement is explicit.
The paper's randomized-count law adjusts its rounding to preserve the intended
*average* count. BayesFilter v1 instead samples integers uniformly from
1 through K. Therefore neither the paper's length coefficient nor its
reported acceptance/efficiency optimum transfers unchanged to this law.
Treat the source rule as one proposal arm and compare it with a bounded
duration neighborhood under the executed BayesFilter law.

### 3.3 Add generic, interpretable exploration diagnostics

Measure displacement in frozen prepared coordinates and against the supplied
metric/scale, not the variance of an arbitrarily small observed cloud. Keep
both proposal displacement and realized displacement including rejection
zeros. Report multiple lags, coordinates and predetermined projections;
track worst directions as well as average squared displacement per actual
gradient work. State exactly what scale was frozen and whether it is only
local. A learned slow direction must be frozen before a separate diagnostic
window, rather than fitted and validated on the same fluctuations.

Include squared coordinates, radial summaries, and declared consumer
quantities such as transformed parameters, tails and predictive summaries.
Coordinate means alone can look efficient under antithetic oscillation
while variances remain poorly explored. A harmonic average can conceal one
slow direction: let such directions nominate additional duration siblings,
not silently replace the paper's rule with an undocumented maximum.

Constant chains, nearly singular scale estimates, insufficient lag support,
conflicting pilot windows, and strong unresolved dependence make the proposal
estimate inconclusive. Use bounded extension or dispersed restarts; retain a
predeclared duration neighborhood when a scalar adaptation estimate is
unreliable. Log the reason when the work cap prevents further diagnosis.
Pilot autocorrelation has no posterior convergence authority. Do not use
R-hat, posterior ESS or posterior MCSE as tuning gates. Do not turn ESJD,
endpoint return, or a noisy pilot maximum into a new hard validity threshold.

### 3.4 Generic feedback, candidate retention and fresh evidence

| Observed development pattern | Next bounded proposal or diagnosis |
| --- | --- |
| High acceptance, tiny displacement relative to frozen scale, positive long-lag dependence | Explore longer physical durations; recalibrate epsilon |
| Rejections or integration instability | Reduce epsilon while preserving requested travel where feasible; check value/score/correction if instability persists |
| Oscillation, near-return or poor squared-quantity exploration | Explore another duration scale/law within supported randomization; endpoint return alone is not failure |
| One slow direction hidden by coordinate averages | Nominate a slow-direction duration sibling and inspect residual preparation |
| Diverse starts disagree, scale varies by position, or exploration remains poor over the funded duration domain | Report residual geometry/start-coverage problem; request a new preparation scope, potentially NeuTra |
| Stable numerical candidates, insufficient posterior precision | Record a delivery failure; a separately versioned future development round can revise proposals or sampling allocation |
| Device contention or elapsed time grows while numerical progress continues | Update measured completion estimate within the allocated campaign ceiling; do not relabel resource delay as a numerical failure |

These diagnoses are hypotheses, not automatic causal classifications. In
particular, a multimodal failure cannot be repaired reliably just by increasing
duration, and epsilon reduction cannot whiten a funnel.

Freeze the final finite proposal cohort before fresh qualification. Declare
its family/candidate cap and independent error allocation, including reserved
acceptance-repair children, before measurement. No adaptation child may exceed
the cap or reuse its parent's evidence. Discarded pilot, measurement,
verification and posterior streams remain separate. Reuse the existing
bounded controller for qualification, extending its explicitly selected
repair policy as described above. The initial cohort is frozen; its reserved
repair rule may propose adaptive children, each frozen before its own fresh
evidence. This does not permit adaptation inside a verification trial.

Retain all verified policies with no nominee. Posterior scheduling is a
separate predeclared allocation: all funded members for mechanics studies,
or an explicitly balanced, cost-feasible subset for expensive targets. Do not
rank members using descriptive pilot ESS or posterior outcomes. Every
unscheduled, unstarted, vetoed and incomplete member remains in denominators.

## 4. Validation that discriminates the failure mechanisms

The algorithm and hypothesis-selection rules are shared across models.
Target identity, supplied geometry, observable definitions and scientifically
motivated precision requirements are model inputs. No branches on
predator–prey names, learned SGQF duration cutoffs, or model-specific success
overrides are permitted in runtime adaptation. SGQF is a known development
case; fresh target/seed evidence must carry generalization claims.

| Test family | Deliberate difficult case | Required evidence |
| --- | --- | --- |
| Isotropic Gaussian, multiple supported dimensions | Tiny epsilon and short travel, acceptance near one | Proposer investigates travel; posterior/reference checks distinguish validity from useful sampling |
| Rotated anisotropic Gaussian | One slow direction hidden by fast coordinates | Frozen-scale/projected diagnostics expose it; coordinate rescaling and rotation do not break the mechanism |
| Gaussian oscillation and antithetic trajectories | Good mean estimates, poor squares or periodic endpoint return | Observable-aware nomination; healthy randomized policies are not invalidated by endpoint return alone |
| Heavy-tailed target with declared finite test moments | Typical-region success conceals tail error | Declared tail quantities and independent references remain binding; undefined moments are never precision targets |
| Exact and partially whitened funnel | Remaining position-dependent curvature | Exact-map positive control; partial-map bounded exploration/explicit geometry limitation; unwhitened funnel is a negative control |
| Separated mixture | Every chain trapped in one mode despite attractive local diagnostics | Independent mode-mass reference and dispersed starts expose false reassurance; no global guarantee from pilots |
| Fixed diagnostic chain fixtures | Collapsed chains, AR dependence, negative correlation, drifting windows | Estimator reliability and fixed-policy-window accounting; no reset-trial concatenation or fabricated precision |
| Search lifecycle | Infeasible duration, epsilon/count boundaries, duplicate policies, noisy pilot, cap exhaustion | Explicit outcomes, correct rounding/work, retained verified siblings and valid error allocation |
| Checkpoint/resource lifecycle | Pause during pilot, partial evidence, replay or backend change | Resume preserves stream identity and adaptation state; incompatible replay rejected; lost work charged |
| Mutation tests | Wrong MAMS Jacobian sign, wrong duration mean, stale score, disabled duration adaptation, leaked verification stream | Independent oracle or lifecycle test fails for the intended reason |
| LGSSM and nonlinear accumulation | Actual filter/score path, dispersed starts, multiple observables | Full public tuner through export/reload, posterior assessment and an independent available reference |
| Full predator–prey SGQF and UKF, SIR, structural UKF | Real likelihood cost and nonlinear geometry | Same frozen adaptation algorithm, target-specific tuning, full data/horizon/quantity identity and declared approximation-specific references |

Use the active inventory in
`bayesfilter.testing.inference_validation.modern_hmc_scope`, which excludes
NUTS. Include NL-SV only after its current source-route eligibility is checked;
NL-GROWTH remains an explicit missing-adapter task. Do not substitute toy
targets for these open integration requirements. A finite-filter posterior
reference verifies that finite target, not an exact nonlinear likelihood.

Tests have three layers: cheap deterministic mechanics and policy regressions;
bounded analytic pipeline calibration with known truths; then GPU/XLA
state-space integration and fresh replicated confirmation. Keep CI cases
bounded and deterministically seeded, with tolerances justified by an oracle
or a documented stochastic calibration, not by the observed value needed to
pass. Run actual kernel paths, not only mocked acceptance callbacks.

For comparisons, freeze candidate budgets, posterior allocation/order,
observables, estimator, reference criteria and evaluation work limits before
the untouched replications. Pair target/geometry/initialization across methods
while maintaining their separately named random streams; identical integer
seeds alone do not establish common-random-number variance reduction.
Include pilot, tuning, rejected hypotheses, status evaluations, compilation
and warm-up costs. Report both actual gradient work and occupied elapsed time.
Do not equate equal transition counts with equal cost.

The end-to-end endpoint is delivery of the declared quantities and reference
checks under the predeclared total allocation. Report a set's delivery event
without claiming every retained member must pass or cherry-picking successful
members' costs. Report paired uncertainty for delivery differences; analyze
cost with failed/censored runs explicitly, never a success-only mean. Choose
replication count, uncertainty procedure and any noninferiority margin in the
priced confirmation design before running it. If budget cannot support them,
report descriptive feasibility rather than robustness superiority.

## 5. Implementation phases and refresh mechanism

| Phase | Concrete changes | Exit and next refresh |
| --- | --- | --- |
| D4R0: source/contract fixture | Pin method equations and adaptation semantics; add archived-grid characterization and estimator-field reporting tests | Confirm actual baseline, units and separate evidence roles before numerical code changes |
| D4R1: duration and MAMS proposer | Extend `hmc_trajectory_proposals.py` with shared typed settings/result; isolate MAMS-specific implementation in a focused module; add requested-to-realized duration compilation without changing v1 execution semantics | Rounding, bounds, inverse/correction regressions, frozen-window and no-posterior-input tests; review any necessary schema change before use |
| D4R2: diagnostics and search wiring | Add TF-native frozen-scale/multilag summaries and source-resolved bounded adaptation; extend the common controller with explicit duration-preserving repair-child semantics and versioned lineage where needed; wire consumers into the declared candidate budget | Tiny-travel, slow-direction, unreliable-window, duration-preserving child, stream, retention and resource/resume regressions pass; freeze algorithm for analytic validation |
| D4R3: analytic integration | Run public pipeline on analytic cases, whitened/partial funnel and multimodal negative controls; activate the implementation mutations | Correct numerical target and honest failure classifications; price state-space work and fix only demonstrated defects |
| D4R4: state-space confirmation | Reprice full filter paths, freeze feasible paired-seed cohort and adequately searched classical comparators, run fresh qualification and posterior assessment | Complete cell denominators, independent reference and uncertainty summaries; record unfunded cells; refresh D4c/F and N5 |
| D4R5: guide and consumer audit | Update `ch21b_hmc_tuning_interfaces.tex`, `ch26g_modern_hmc_methods.tex`, the agent interface and capability registry to implemented semantics; integrate the tests into regression tiers | One coherent guide, compiled/rendered chapter review, no claims beyond the evidence; feed N8 release decision |

Documentation changes accompany each implemented interface; D4R5 is the final
consistency pass, not permission to leave intermediate public behavior
undocumented. Existing `tests/test_hmc_trajectory_proposals.py`,
`tests/test_modern_hmc_contracts.py`, `tests/test_modern_hmc_kernels.py`,
candidate-set tests, method-proposal/fit tests and public pipeline tests are
the first regression anchors. Add focused tests for new behavior; do not
refactor the large legacy tuner just to implement this proposal layer.

At each phase close record changed files, exact checks and results, numerical
findings, unresolved alternatives, measured cost, and the next concrete
command/cohort. An expected candidate failure triggers its planned diagnostic
or repair, not abandonment of the program. A shared target/implementation
defect stops dependent work. Preserve historical results and source closures.
Do not interrupt an unrelated healthy running job to apply these changes.

Implementation is TensorFlow/TFP with stable signatures; no NumPy runtime or
JAX port. GPU/XLA is the serious execution path. CPU-only tests are explicit
reference/smoke exceptions with GPUs hidden before import. GPU work requires
trusted access and verified incremental allocation before initialization.
Keep adaptation bookkeeping and artifact I/O outside pure numerical graphs.

### Compute and launch discipline

This planning work allocates no new campaign time. The latest recorded master
balance is 158333.765 seconds after reserve, approximately 43.98 aggregate
worker-hours; it is a historical ledger snapshot, not a fresh capacity check.
Reconcile intervening CPU/GPU usage and device availability before launching.
The current shared checkout is on `preserve/shared-main-before-fab-20260926`
at `70a6d7e96`, while the modern-method sources, this master and many other
files are untracked or modified. A plan-only commit is not a runnable source
baseline. Before D4R3 or any claim-bearing run, make a coherent source commit
or an exact source snapshot of the required dependency closure, then record
that revision and the remaining dirty-file hashes. Preserve all unrelated
shared-worktree changes; never stage the whole tree by convenience.

Use short mechanics checks first, then price representative fixed-policy
pilot windows and qualification trials on the actual current device. Derive
each subsequent phase allocation from measured work and the remaining grant.
Reserve verification and reporting work before scheduling adaptive expansion.
Freeze explicit work/attempt/wall ceilings and exact commands in the phase
execution note before every research run; this proposal is not an unspecified
long-run command. If the full replication matrix does not fit, narrow the
claim and retain unfunded cells explicitly. Do not silently add compute.

Use fresh directories beneath
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/`.
Every launch needs a unique attempt directory, source revision plus dirty
source hashes, command, environment, target/data/map identity, seeds, device,
XLA/memory policy, actual elapsed work, plan and result paths. Localized
infrastructure retries preserve the same scientific contract and remaining
budget. Compilation/progress monitoring distinguishes contention from a
stalled numerical kernel; no unconditional per-model timeout is introduced.

## 6. Default audit and skeptical review

| Choice | Provenance and status | Risk and smallest diagnostic |
| --- | --- | --- |
| Existing nine-point grid | Historical development comparator | Not adequate adaptation; characterize no-pilot/no-repair configuration |
| Existing uniform integer count law | Implemented v1 baseline | Mean differs from half the maximum; exact moment and rounding tests |
| Requested mean length and frozen-scale diagnostics | Proposed local adaptation | Rounding/scale can mislead; bounds, transformed Gaussian and local-geometry tests |
| sqrt(d), target acceptance 0.9, deterministic ALBA coefficient 0.3 | Paper hypotheses, not promoted defaults | Wrong target/observable scaling; duration neighborhoods and analytic/state-space ablations |
| Acceptance qualification floor and family error allocation | Existing explicitly chosen policy for the campaign | Optimizer target can be confused with membership; separate typed fields and boundary tests |
| Pilot windows, factors, domain caps, update counts and lag estimator | To be explicit in D4R1 calibration design; no universal value chosen here | Hidden one-model tuning; provenance, unreliable-estimate tests and frozen heldout design |
| Current local affine map | Supplied preparation; global coverage unestablished | Residual curvature or missed mode; dispersed starts, analytic maps and independent references |
| Posterior precision/count settings | Existing study contract, not a sampler default change | False attribution of limited precision to kernel invalidity; report warm-up/precision/reference separately |
| Remaining compute estimate | Last settled ledger snapshot | Concurrent use/stale prices; reconcile and reprice before launch |

**Pre-execution skeptical audit.** The original tempting repair, enlarging the
predator–prey grid or rejecting high-acceptance candidates, would not address
the generic mechanism. Replacing the movement tolerance by another unexplained
number would confuse validity with efficiency. Both are rejected here.

The revised design keeps the actual baseline, distinguishes a proposal score
from a promotion criterion, avoids mixing reset trials into an autocorrelation
estimate, accounts for duration quantization and total work, preserves fresh
verification, and tests nonlinear observables and heldout models. It does not
promise that source-derived heuristics will work under arbitrary geometry.
The paper/author-code version discrepancy is explicit. Current XLA/backend
identity and local geometry are preserved and checked rather than assumed.
Final tracing also identified the current fixed-count epsilon repair and
inherited epsilon-domain cap. The plan now requires duration-preserving
repair-child semantics, audits their lineage/schema implications, and treats
domain limits as explicit search constraints instead of method defaults.
It also distinguishes the source paper's average-preserving count law from
the existing uniform-integer BayesFilter law, and requires coherent source
provenance before any claim-bearing experiment.

**Verdict:** suitable to implement incrementally through D4R0–D4R2 with
focused tests. D4R3–D4R4 require their stated priced, explicit execution design
before numerical research runs. This is a technical self-review, not an
independent mathematical audit or completed validation. Unexamined numerical
choices remain unpromoted and must be stated/tested when introduced.

Planning checks completed on 11 October: all relative document links resolve;
the nine-member table matches the saved posterior result files and their
lugsail fields; independent standard-library arithmetic checks confirm the
uniform-count second moment and nearest-count quantization bound. These are
documentation/math checks only, not tests of the unimplemented proposer.

**Pre-mortem.** A passing run could reflect an easy local basin, an overly
favorable mean observable, winner selection across many settings, or uncharged
adaptation cost. The multimodal control, squared/tail observables, preserved
denominators and total-work accounting address these risks. A failure could
come from wrong duration units, unreliable pilot autocorrelation, noisy
preparation, filter status or contention; the cheap mechanics and fixed-policy
diagnostics distinguish these before another long campaign.

Every substantial result note must retain decision and inference-status
tables: primary criterion, hard vetoes, viable candidates, uncertainty,
descriptive differences, supported ranking, default-readiness, next evidence,
and next justified action. Its red-team note must state the strongest
alternative explanation and what fresh result would overturn the conclusion.

## 7. Source anchors

* Archived MAMS paper:
  [Section 6 and equations (12)–(14)](../../.localresources/hmc-methods-review-20261007/mams-2025.txt),
  lines 360–435; randomized duration and rounding footnote in sections 5–6;
  experimental retuning and sensitivity in section 7 and Appendix D.
* [Archived author adaptation](../../.localresources/hmc-methods-review-20261007/blackjax-adaptation-adjusted_mclmc_adaptation.py),
  top-level average-step pinning and disabled final stage;
  `adjusted_mclmc_make_adaptation_L` for the ESS-based length proposal.
* [Source version/hash manifest](../../.localresources/hmc-methods-review-20261007/source-manifest.json).
* [Current interface](../reference/hmc-tuning-interface.md),
  `HMC_TUNING_INTERFACE_CAPABILITIES` in
  `bayesfilter/inference/tuning_contract.py`, and the master's sections 5, 6.3
  and 8 for existing procedure and validation requirements.

The source adaptation is heuristic proposal machinery. Neither those sources
nor the present artifacts prove universal robust tuning; the multi-model
validation is the missing evidence this repair is designed to obtain.

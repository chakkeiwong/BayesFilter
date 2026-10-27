# Full Stan Euclidean NUTS port: execution plan and pre-run review

The owner has requested plan, review and execution of the
[port design](bayesfilter-full-stan-nuts-port-design-2026-10-27.md).
This authorizes a new standalone implementation and comparison, while existing
retired public routes and production defaults remain unchanged. The goal is
algorithmic fidelity to Stan 2.37.0 for generic unconstrained TF targets.
Exact MIXMAX/Boost seed compatibility and the Stan model compiler were
explicitly separate in the design and are not deliverables of this execution.

## Intent and evidence contract

Question: can we execute the complete Stan Euclidean NUTS transition and
lifecycle in TF/XLA, and what costs remain relative to native Stan and the
current TFP-based implementation? Mechanism: a bounded iterative tree with
Stan's exact proposal/stop semantics and direct unit/diagonal/dense metrics.

Engineering acceptance requires native-source transition/option tests, full
warm-up and sampling XLA, valid retained caches, and declared failure handling.
Common-variate oracles compare discrete choices exactly away from numerical
ties, plus FP64 state/energy agreement. Existing 2e-12 tolerances are inherited
rounding hypotheses, not permission to hide branch errors.

Posterior validity screens precede all efficiency interpretation. Initial
comparisons reuse the earlier six analytic targets and add a costlier regression
fixture with native density/gradient parity. Four chains, 1000 warm-up and
2000 retained transitions, depth 10, FP64, delta .8 and three independent seeds
20261027/28/29 are inherited pilot choices, not universal adequate defaults.
Include a separately labelled overdispersed initialization check. Compare the
new direct port, current audited TFP adaptation, native Stan and fixed-L HMC
tuned through the existing repository protocol; do not substitute the old grid.

The primary scientific aspiration is time to declared precision. This bounded
implementation campaign first reports fixed-budget validity and cost.
A fixed-budget pass is not continuous time-to-precision evidence or proof of
superiority. A sequential precision comparison can follow only if the budget
and available matched continuation interfaces support it; otherwise record it
as unevaluated, not as a completed efficiency promotion.

| Quantity | Role and consequence |
| --- | --- |
| Native transition/adapter mismatch, wrong target, invalid retained cache or missing XLA | Engineering veto and repair trigger |
| R-hat >1.01, bulk/tail ESS <400, standardized MCSE >.05, or failed analytic moments | Posterior/efficiency promotion veto; target/budget repair trigger |
| Complete initialization+tuning+retained cost with uncertainty | Efficiency criterion after validity; fixed-budget values remain limited evidence |
| Leapfrogs, acceptance, tree cap, energy/divergence flags, graph/memory/compile costs | Explanatory; divergence alone never rejects a tuning candidate |
| Corrupted oracle/data, resource violation, or exhausted total budget | Continuation veto for affected work |

Reference agreement uses the prior screen: each standardized analytic first/
second moment error <= max(4 MCSE, .05 reference SD). These numerical screens
are inherited pilot criteria, not a guarantee of correct tails. Few seeds
give descriptive results; no default promotion is contemplated.

## Implementation stages

1. Native oracle: use unmodified base_nuts and expl_leapfrog headers with
   controlled momenta/uniforms, analytic targets and observation of criterion
   calls. Include both directions, merge decisions, short/invalid trees,
   depth caps, non-unit metrics, jitter and rejected invalid proposals.
2. TF transition: explicit bounded stack/while-loop traversal, source-ordered
   selection and three U-turn checks, unit/diagonal vector paths and dense
   metric factor cache, independent stateless per-chain RNG, complete telemetry.
3. Lifecycle: share audited DA/search arithmetic; implement initial metric,
   unit mode, adaptation disabled, zero warm-up, custom/short/zero-buffer
   schedules, unconstrained zero/random/partial initialization, thinning,
   frozen/resumed sampling and degenerate no-parameter handling.
4. Engineering validation: native oracle, option tests and cache/reproducibility
   tests; inspect optimized enclosing HLO on CPU/GPU, graph growth and memory.
5. Benchmark: calibrate layouts, freeze layout choices, run new/TFP/native
   comparisons with full cost accounting, then integrate correctly tuned
   fixed-HMC through its documented interface and preserve every failure.
6. Write user guide and coverage/result tables, retain source hashes and exact
   launch records, and conduct terminal skeptical self-review.

The benchmark implementation stays under docs/benchmarks; inference modules
are used only through existing supported interfaces. The new core consumes
generic batched TensorFlow targets. Numerical callback failures are tensor
status/nonfinite proposals; arbitrary Python exceptions inside XLA are not
part of the target contract. Mathematical constraint transforms remain owned
by the target.

## Compute, environment and artifacts

Fresh output root: docs/benchmarks/artifacts/stan-full-port-20261027/ with
numbered attempt directories. No previous evidence is overwritten.
Total worker wall budget: 90 minutes, up to 30 test/benchmark launches, with
15 additional minutes for native compilation. These are convenience-chosen
upper bounds based on the prior sub-two-minute engineering runs and roughly
two-minute six-target comparison; they allow localized repairs. Before each
launch charge previous attempts and ensure its watchdog fits the remaining
budget. Local editing and source reading do not consume compute budget.

Python: /home/ubuntu/anaconda3/envs/tfgpu/bin/python (TF 2.20, TFP .25).
CPU workers hide GPUs before framework import; GPU workers require trusted
execution and verified growth, TF_FORCE_GPU_ALLOW_GROWTH=true. TF32 is off.
Bound intra-op threads and use separate independent scalar chains or a
batched kernel according to fresh calibration; the CPU exception is an explicit
native comparison, not a change to repository production policy.

Commands will use a repository-local bounded audit launcher recording the exact
command and source snapshot before starting. Tests: pytest on the new port
tests plus existing adaptation, target and scope tests. Benchmarks: the new
full-port harness and existing native comparison helpers. Every attempt records
git SHA, source hashes, environment, device/memory policy, seed, wall time,
logs, artifacts and plan/result links. Unsupported stages or exhausted budgets
must be reported as incomplete, never silently removed.

## Default audit and skeptical review before implementation

| Assumption | Provenance/status | Risk and early check |
| --- | --- | --- |
| Stan 2.37.0 source | Pinned installed baseline | Test actual headers; do not mix a newer implementation |
| Iterative tree | Engineering hypothesis | Match common-variate native transitions before posterior runs |
| Direct metric implementation | Algebraic Hamiltonian equivalence | Native leapfrog/momentum/cross-subtree checks |
| TF stateless RNG | Explicit different runtime RNG | Independent chains and deterministic replay; no same-seed native claim |
| Short warm-up fallback | Actual release behavior | Native source returns without restart; reproduce observed schedule, label this behavior |
| FP64 source order | Fidelity baseline | Near-tie audit and strict inequality fixtures |
| Dense factor caching | Constant metric derivation | Recompute at every metric update, never reuse stale target/score |
| Prior screen/counts/seeds | Inherited diagnostic baseline | Check all vetoes; do not promote limited pilot evidence |
| Native and fixed-HMC baselines | Required comparators | Price tuning, verify target and metric conventions before comparison |

Skeptical review findings: (1) A Python rewrite used as the oracle would be
circular; use actual Stan headers. (2) Replacing both selection rules by
ordinary multinomial merging changes the transition; test them separately.
(3) Masked batches can silently alter RNG consumption; independent streams
must pass isolation tests. (4) Fast failure on a funnel is not speed evidence.
(5) The old fixed-HMC finite grid is not the promised tuned baseline.
(6) Arbitrary Stan-model transformations cannot be inferred from a flat TF
target; constrain claims to the declared unconstrained interface.
(7) Full Stan seeded bitwise replication was not the agreed scope.

The plan incorporates these corrections and passes self-review for staged
execution. An engineering or configured-candidate failure triggers the next
repair stage under the same budget; it does not terminate the research
direction. This is implementing-agent review, not independent certification.

## Resolved benchmark details before execution

The costlier control is a 4096 by 64 Gaussian regression design (16 times
the observations-times-parameters of the earlier 512 by 32 fixture), using
the same stateless design seed and an analytic Gaussian reference. This is
a convenience-chosen gradient-cost stress test, not an additional difficult
posterior geometry. The overdispersed check multiplies exact reference
coordinates by 3, a declared stress hypothesis rather than a universal start.

The fixed-HMC arm invokes the public standard automatic geometry preparation
and candidate-set procedure with no replacement epsilon/L grid. It uses the
public default acceptance policy and its minimum measurement/verification
counts, XLA, batched chains and reusable leapfrog graphs. A 180-second
preparation/search allowance per fit is a compute allocation (21 fits cost
at most 63 minutes before bounded overhead), not a statistical adequacy claim.
The native public automatic procedure constructs its own start bank from the
common first start; report that difference. Every verified candidate is
exported and receives fresh retained diagnostics; there is no efficiency
winner selected on retained draws. Report per-member tuning-plus-retention
costs and all outcomes. The aggregate wall cost includes assessing all members
and cannot be compared as though it purchased just one posterior stream.
If preparation fails, retain the failure and investigate the smallest repair;
do not relabel an identity-only or simplified-grid path as the standard tuner.

Review: these choices preserve the question, charge complete HMC tuning and
prevent an arbitrary representative L from becoming the supposed best-tuned
baseline. Limited-budget failures remain inconclusive about the best achievable
HMC performance. CPU scalar, CPU batch and GPU batch calibration is diagnostic
only and precedes selection of execution layouts.

## Execution choice after corrected calibration

Attempts 11–13 used disjoint chain keys. For 160 warm-up plus 200 retained
transitions, median CPU scalar direct-port times were .1182, .1452 and 1.4858
seconds on Gaussian32, banana8 and regression64; CPU batch times were .1611,
.1598 and 4.8548 seconds. The single GPU calibration measured 1.7802, 87.0590
and 3.2620 seconds. TFP likewise favored CPU overall; its banana batch timing
was slightly lower than scalar. Freeze CPU scalar for the three NUTS engines'
main comparison (native Stan uses four CPU processes); use the public fixed-HMC
procedure's batched execution as specified above. This is a measured pragmatic
choice for these fixtures, not statistical proof of globally optimal hardware.
The large GPU banana cost warrants later profiling but does not justify making
the main comparison wait for a different GPU implementation. No production
execution policy changes.

Pre-run review confirms independent starts/streams, matched targets, source
checks and explicit compilation costs. Run the three NUTS engines first, then
validate the public HMC consumer on Gaussian32 before expanding it. Preserve
budget for final regressions and an overdispersed check on banana8, funnel10
and rotated_gaussian32 (chosen to expose curved, varying-scale and correlated
geometry). Fixed-HMC coverage may be limited by its full standard procedure's
cost; any such shortfall is an incomplete comparator, not a NUTS win.

Terminal coverage review identified that initialization retry rules were
source-inspected and locally tested, but had no executed native-service oracle.
Add a small flat, two-parameter native `services::util::initialize` fixture
with controlled uniforms, comparing supplied/partial/random/zero starts,
retries, finite-value failures and gradient-sum overflow. This strengthens the
existing engineering criterion within the budget. It does not claim named
constraint transforms or exact native RNG. Review also distinguishes CLI
parsing from sampler semantics: this API rejects jitter=1, which CmdStan's
parser accepts but the native sampler's strict setter ignores, and caps tree
depth at 30 for int32 counters. Document these explicit interface differences.

The Gaussian fixed-HMC preflight completed the standard search in 168.02
seconds, with 20 verified candidates; all failed at least one fresh posterior
screen at 2000 draws. This is a candidate/budget failure, not a harness crash or
a reason to drop the comparator. Assess the remaining targets. The initial
consumer's per-member totals omitted build/export time although its aggregate
264.69-second wall cost included everything. Mark those per-member values as
lower bounds. For subsequent attempts use the existing public grouped export
and reload functions, time that shared setup explicitly, and charge it in each
per-member total. Numerical tuning is unchanged. Given measured cost, prioritize
one complete seed across all target geometries before additional replications;
reserve 600 worker seconds for final checks and the planned overdispersed run.
Three-seed fixed-HMC coverage remains the aspiration within the original ceiling.

Before interpreting the Gaussian HMC failures, independently reconstruct its
archived affine map and evaluate fresh target values at every retained draw.
Require coordinate and cache residuals below 1e-10 (a rounding allowance for
two FP64 affine layers, checked against measured residuals), up to the constant
Jacobian term. This is a harness-validity check, not a posterior pass. Report
Gaussian leapfrog mode phases only as an explanation for possible resonance;
do not use them to select a new L or relax the frozen posterior screens.
The terminal archive audit also reconstructs Gaussian MH selection and
`log_accept_ratio = -.5||q_proposal||² + .5||q_previous||²
                    + .5||p_initial||² - .5||p_final||²`
from the saved independent momenta and proposal. The constant affine Jacobian
cancels. Require exact selected-state agreement and an energy residual below
the same 1e-10 rounding allowance. This directly checks that rejected proposals
preserve the current state instead of inferring correctness from acceptance.

## Execution closeout

The bounded campaign completed in 23 launches and 5052.046 worker seconds,
including repairs and native-oracle builds. The 128-test final suite passed;
all seven targets received three independent fits from each of the four
specified procedures. All 216 verified fixed-HMC members were assessed. One
public-HMC search was partial at its per-fit budget, without suppressing its
available candidates. The separate overdispersed NUTS check and final Gaussian
archive/MH audit also completed. See the [result](bayesfilter-full-stan-nuts-port-result-2026-10-27.md)
for validity failures, timing boundaries and terminal self-review. This
campaign establishes no time-to-precision ranking and changes no default.

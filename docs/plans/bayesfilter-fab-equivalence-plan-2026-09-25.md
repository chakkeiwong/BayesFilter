# Executable FAB-JAX / TensorFlow equivalence verification

## Question and scope

Does the TensorFlow FAB implementation reproduce the pinned author's alpha-2,
linear-temperature AIS, identity-mass HMC / Gaussian Metropolis, prioritized
replay and training updates when given the same density, parameters, state and
random variates? Reference: fab-jax
`c9f991366ca94b2678a7ed620bc9e12655cfef1d`, preserved in the shared repository's
`.localresources/fab-jax-c9f9913`. Candidate: the isolated canonical checkout
`/tmp/BayesFilter-neutra-fab-20260925`, initially `ab69565baffcf41355efac496186fca3ef4e0f7e`.

The existing IAF and q20 target remain unchanged. JAX is an independent,
CPU-only test authority, not a runtime backend migration. A diagnostic JAX
evaluation of the same IAF equations and exported parameters may be used to
supply the author's Flow callbacks; it must first pass density, inverse and
parameter-derivative comparisons against the actual canonical TensorFlow map.
Analytical targets isolate FAB logic; these tests make no q20 fit, convergence,
coverage, integrability or production-readiness claim.

## Evidence contract and research intent

The primary criterion is matched-input numerical agreement at each component
and at complete training-state boundaries. Finite values and close final losses
alone are insufficient. Discrete selections, buffer counters and decisions must
agree exactly away from measured floating-point decision boundaries. Compare
unselected proposals as well as retained states so rejection cannot hide errors.

Mechanism under test: translation of the author's finite algorithm. Expected
failures: loss scaling, Adam epsilon/bias correction, mutation/adaptation order,
replay fill/overwrite/order, random coupling or tracing errors. A discrepancy is
a promotion veto and repair trigger, not rejection of FAB. Missing reference
execution, contaminated fixtures, hidden adapters changing numerical operations,
or exhausted budget veto further equivalence claims. Engineering failures may
be repaired and retried within the budget. Timing, ESS and training loss trends
are explanatory only. Upstream agreement does not establish that upstream is
mathematically correct; analytical and finite-difference tests remain separate.

## Execution

1. Preserve a baseline fixture/result; create an isolated Python 3.11 reference
   environment under `/tmp`, pin compatible dependencies and record versions.
   Do not modify existing conda environments. Run the original upstream tests
   headlessly, explicitly invoke the misspelled Metropolis test, and report
   assertion-free plotting checks separately. Preserve any compatibility patch
   as a visible diagnostic adapter; do not rewrite the author algorithms.
2. Implement an executable reference driver importing the actual upstream
   functions. Use JSON fixtures/results between the JAX and TensorFlow processes.
   Export canonical IAF parameters and masks. Test the diagnostic callback
   independently before relying on it for gradient/training comparisons.
3. Supply common random variates at RNG boundaries. Same seeds are not common
   draws across frameworks. Record coupling by semantic role (initial normal,
   mutation noise/momentum, acceptance threshold, Gumbel priorities, permutation).
   Compare bridges/scores, every AIS increment, leapfrog proposal/acceptance,
   mutation adaptation, replay sampling/add/adjust, fresh/replay losses and
   gradients, Adam slots and updates. Exercise nontrivial weights, saturation,
   clipping, rejection, invalid input and buffer wraparound.
4. Repair confirmed source mismatches in the single TensorFlow authority and
   retain failing fixtures as regressions. Fresh loss must match the author's
   mean-normalized convention. Adam must match explicit Optax equations.
   Invalid-state handling must be either matched or clearly outside a qualified
   equivalence domain; never conceal differences as numerical tolerance.
   Version changed checkpoint semantics; old checkpoints cannot silently resume
   under a changed optimizer. Preserve historical training evidence.
5. Compare successive full training iterations with and without replay for
   both mutation operators, including initialization and checkpoint continuation.
   Check every map parameter, optimizer moment/counter, step size, buffer row,
   priority and stored pre-update density. Run focused existing regressions.
6. Run CPU FP64 reference first, then small FP32 and trusted GPU/XLA checks of
   the same implementation. Run a bounded native-RNG check with declared
   uncertainty margins. Summarize exact passes, exclusions and remaining gaps.

## Choices, numerical policy and budget

| Choice | Provenance / justification | Failure mode and early check |
|---|---|---|
| Alpha 2, linear schedule, no SMC resampling, identity mass | Existing supported port; author configurable route | Exclude other upstream options explicitly |
| Canonical three-stage IAF, tiny width for reference plus q20 width 16 | Existing authority; small map bounds diagnostic cost | Compare forward/inverse/density and all parameter gradients |
| FP64 reference; FP32/TF32 GPU second | Distinguish semantics from rounding | Record absolute/relative errors and decision margins |
| Analytic normal, shifted/scaled normal and nonlinear density fixtures | Known density/score and tail behavior | Independent algebra and directional finite differences |
| Numerical tolerances | Predeclared engineering bounds: FP64 absolute/relative 1e-10 for elementary stages, 1e-8 for composed updates; FP32 2e-5 / 2e-4 respectively; TF32 compared separately | These are precision-scaled hypotheses, not mathematical error proofs; inspect errors, reject systematic discrepancy and never loosen after seeing failures |
| Decision boundaries | Compare thresholds and acceptance separately | Any unexplained branch mismatch vetoes the fixture; boundary cases reported separately |
| Seeds and small dimensions/batches | Fixed reproducibility/branch-coverage fixtures, not tuned scientific settings | Include multiple signs/scales and buffer boundaries; no ranking of methods |
| Native RNG distribution check | Predeclared bounded sampling check, not same-seed parity | Confidence intervals must lie within declared margins; failure to reject difference is insufficient |

CPU reference and test subprocess budget: 7,200 aggregate process-wall seconds;
GPU diagnostic budget: 600 worker seconds. These are convenience-chosen ceilings
for this newly authorized verification task, not training settings or allocations
to restart q20 fits. Per-command timeouts and actual costs are recorded. Maximum
three environment-resolution attempts and three focused retries per detected
infrastructure failure before recording the obstruction. Numerical fixes are
followed by focused regression. No long posterior or training campaign runs.

GPU workers set and verify memory growth before device initialization. CPU
workers set `CUDA_VISIBLE_DEVICES=-1`, `JAX_PLATFORMS=cpu`, and JAX preallocation
off before import. GPU runs use trusted/escalated tool access. All output goes
to fresh directories under the shared
`docs/plans/artifacts/neutra-fab-equivalence-2026-09-25/` root; no prior output is
overwritten. The manifest records commands, source hashes, git base/diff,
environment, random fixtures, CPU/GPU policy, wall time and result paths.

The detailed executable commands will be recorded as run: dependency versions
must be resolved against the pinned old APIs before a truthful exact command
can be frozen. Test scripts, requirements and invocation logs are deliverables.

## Skeptical review before execution

REVISED then ACCEPTED for bounded verification. The initial idea of merely
running both test suites fails: most upstream SMC tests have no numerical
assertions, and both suites could pass with different loss/optimizer semantics.
The revised plan executes upstream functions directly with common draws and
compares complete states. An independently rewritten FAB oracle is forbidden.
The diagnostic IAF callback is a dependency needing its own parity check.
Invalid-state policy and RNG finite-precision law are explicit potential
exclusions, not assumed equivalences. Tolerances are provisional engineering
bounds whose failures require investigation, not automatic relaxation.
Reference equality is the baseline; no weak method comparison or training-loss
proxy determines promotion. Stops, budgets, environment mismatch and artifact
requirements are explicit. Upstream bug replication remains possible and is
addressed by analytical checks. This is the requested skeptical self-review;
no independent-agent review is represented as having occurred.

## Required terminal record

Write a result with component/iteration coverage, maximum discrepancies,
upstream-test outcomes, repaired mismatches, unresolved exclusions, decision and
inference-status tables, costs, and a post-run red-team note. Update the active
checkpoint with exact next action. A finite suite establishes tested equivalence
only in its stated domain; it is not a universal proof or a trained-map result.

## Pre-run refinements after inspecting executable upstream

Upstream tests have a stale logging import, accidental singleton-tuple flow
defaults and a buffer test requesting 24 rows when its minimum is 15. Preserve
the untouched failures, then run explicit test-only compatibility repairs:
alias the existing logger, unwrap those five configuration defaults, and reduce
that one test's request to 12 rows. The upstream numerical implementation is
unchanged. These are testing-harness repairs, not evidence that its suite is
complete.

Native-RNG checks use 65,536 independent draws, seeds 31051--31053 (fixed
convenience seeds), and twelve bounded events: six unordered replay pairs,
three normal-CDF thresholds and three uniform-CDF thresholds. Enumerate the
exact four-category without-replacement pair law. With 24 event proportions
across both backends, simultaneous Hoeffding radius is
`sqrt(log(2*24/.01)/(2*N))` (99% family coverage). Require both analytic-law
intervals and paired-backend difference intervals to lie inside +/-0.025.
The 0.025 margin is an explicit coarse RNG-screen hypothesis, not a posterior
or gradient tolerance. N is the next power of two above that required to make
each interval radius at most one third of the margin. This supplements the
deterministic comparisons and cannot establish arbitrary-tail RNG equivalence.

Initial invalid-row replacement and invalid replay insertion/adjustment will
match the author operations, with explicit diagnostics. The all-invalid initial
batch has no defined replacement law: local failure remains required. The
target callback's validity status and finite-score/path checks remain scientific
vetoes; behavior when they fire is a documented domain exclusion from literal
upstream equality, never silently interpreted as a passed equivalence check.

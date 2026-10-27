# Direct Stan Euclidean NUTS port: execution record

Status: implementation, review and bounded comparison completed under the
[reviewed execution plan](bayesfilter-full-stan-nuts-port-execution-2026-10-27.md).
The source is [stan_nuts.py](../benchmarks/stan_nuts.py); the reusable interface
is documented in the [direct-port guide](../reference/nuts-direct-stan.md).
The main comparison covers seven targets and three seeds for each of the
direct port, audited TFP route, native Stan and public fixed-HMC procedure.
All 216 verified HMC members received fresh posterior assessment. One HMC
search reached its per-fit budget before exhausting its work queue; that
incomplete search is preserved. No efficiency or production-default promotion
is made.

## Direct, TFP and native comparison

The generic direct port reproduced the tested Stan transition/adaptation
behavior, but it did not reproduce Stan's native runtime speed. All three
implementations passed the declared posterior screen on six targets across
all three seeds. All failed on the centered funnel. There is no valid
efficiency ranking for that funnel configuration.

These are median seconds for four chains, 1000 warm-up plus 2000 retained
draws per chain, with FP64 and separately compiled reusable kernels. TF costs
include numerical initialization, step-size search, adaptation and retained
sampling. Native Stan costs include process startup and CSV output. TF output
serialization and common posterior diagnostics are recorded separately.
TF serialization added .0047–.0324 seconds per fit. Common diagnostics took
.178–2.892 seconds, including their first-call costs, and are excluded for all
three engines. All 21 direct and 14 TFP lifecycle functions recorded one trace
each; the log's repeated-function warning reflects distinct target functions.
The three seeds support descriptive timing comparisons; no inferential
runtime ranking or time-to-precision superiority is established.

| Target | Direct Stan port | Audited TFP route | Native Stan 2.37 | Passing seeds per engine |
| --- | ---: | ---: | ---: | --- |
| Gaussian, 32 dimensions | 1.047 | .628 | .123 | 3/3 each |
| Rotated Gaussian, 32 dimensions, dense metric | 2.631 | 1.350 | .344 | 3/3 each |
| Banana, 8 dimensions | 1.038 | .624 | .100 | 3/3 each |
| Centered funnel, 10 dimensions | 2.151 | 1.068 | .231 | 0/3 each; unsuitable for efficiency ranking |
| Noncentered funnel control, 10 dimensions | .635 | .516 | .074 | 3/3 each |
| Regression, 512 observations × 32 parameters | 1.293 | 1.010 | .344 | 3/3 each |
| Regression, 4096 observations × 64 parameters | 7.160 | 9.972 | 3.732 | 3/3 each |

The largest native/TF density or gradient relative residual across the
pre-run points was 1.25e-15. These checks support matched targets at those
points, not an exhaustive proof for every input. Passing targets had worst
R-hat below 1.005 and minimum tail ESS above 1100 across all methods/seeds.
Median centered-funnel R-hat was 1.072, 1.196 and 1.235 for direct, TFP and
native respectively; corresponding median tail ESS was 55, 35 and 77.
Those failures block posterior promotion regardless of elapsed time.

The separate start stress test multiplied exact-reference coordinates by
three, then repeated the same tuning and retained budgets across three seeds
on banana, centered funnel and rotated Gaussian. Banana and rotated Gaussian
passed in all engines/seeds; the centered funnel failed in all engines/seeds.
The noncentered control is a planned geometry repair and passed, so the
centered failure does not reject the NUTS research direction. Its screen
assesses the sampled noncentered coordinates; it does not certify precision
of every nonlinear physical-coordinate funnel functional.

### What the speed results explain

On Gaussian32, all three algorithms averaged about seven retained leapfrogs.
Yet median retained sampling wall times were .665 seconds for the direct
port, .314 for TFP and .062 for native Stan's maximum printed chain time.
The extra cost therefore cannot be explained by extra integration steps.
The direct port maintains and updates tree stacks, masks and random state
inside nested compiled loops. Native Stan uses scalar C++ control flow.
Those are concrete implementation differences; attributing a fraction of the
gap to RNG, stack writes, masking or kernel launches still needs profiling.

On regression64, direct warm-up took a median 2.797 seconds versus TFP's
6.263, while retained sampling took 4.442 versus 3.852. The descriptively
lower direct total is a warm-up result. It is not evidence that its transition
is faster. Direct diagonal arithmetic avoids the TFP comparator's generic
affine wrapper; different warm-up trajectories also change integration work.
A controlled frozen-kernel profile is needed to distinguish those effects.

The next speed experiment should hold position, metric, epsilon and random
inputs fixed while measuring gradient evaluation, random draws and tree
maintenance separately. The source-faithful tests can then check any proposed
change to stack updates or random buffering before fresh posterior runs.
For the GPU case, the profile must also measure how unequal tree lengths
leave batched chains waiting. This campaign identifies those concrete
optimization targets; it has not measured which is dominant or shown that
changing them closes the native-Stan gap.

First calls, including compilation and discarded execution, cost 5.37–10.64
seconds per target for the three direct lifecycle functions and 6.19–19.73
seconds for the two TFP functions. They are not pure compiler measurements.
The installed native model was reused; its preceding installation/build cost
is recorded in the earlier transplant report. Native C++ is the reference
comparator, not an XLA implementation.

### Execution choice

Corrected calibration selected four parallel scalar CPU chains for the NUTS
comparison. The median direct costs for Gaussian32, banana8 and regression64
were .118, .145 and 1.486 seconds; CPU batch measured .161, .160 and 4.855.
The one-seed GPU batch costs were 1.780, 87.059 and 3.262. GPU banana was
especially expensive even after compilation and with ordinary adapted epsilon;
the cause has not been profiled. This selects a practical measured layout
for these fixtures, not a universal CPU advantage or a production-policy change.
GPU full-kernel compatibility remains separately verified. The corrected
GPU calibration's TensorFlow allocator peak was 19,409,664 bytes, with peak
host RSS 2,003,692 KiB; growth was verified on the RTX 4080 SUPER before
initialization. The CPU main comparison peaked at 3,748 MiB host RSS.

## Fixed-HMC comparison and timing boundaries

The fixed-HMC arm uses `tune_hmc_kernel` with the public standard automatic
windowed mass preparation and candidate-set search. Its acceptance policy,
measurement counts, verification counts and trajectory-length exploration
come from that protocol. The per-fit search allowance is 180 seconds. Every
verified member receives 2000 fresh retained draws per chain and the same
posterior screen as NUTS. No retained result feeds back into tuning or chooses
an efficiency winner. The public procedure creates its own start bank from
the first common starting point; it does not use the four NUTS starts verbatim.
Its own preparation schedule uses 800 transitions for the 32-dimensional
targets, 200 for banana8, 250 for the two funnel targets and 1600 for
regression64, with an inherited acceptance target of .7. The harness's
`--warmup 1000` controls NUTS; it does not override this public HMC protocol.
Thus this comparison prices the two specified tuning procedures, rather than
holding their adaptation work or acceptance targets equal.

There are three distinct costs. Preparation/search is shared by all verified
members. A member's reported total adds the grouped export/reload and that
member's retained run, including archive writing and retained health checks.
The aggregate fit cost additionally includes retaining and assessing every
other member. That aggregate buys a candidate-set assessment and cannot be
read as the price of one frozen HMC chain. The NUTS table excludes compilation
and common diagnostics, whereas the public HMC procedure includes cold
compilation during preparation and search. These are recorded workflow costs,
not a matched comparison of transition speed or cold-start cost.

The first Gaussian preflight's per-member totals omitted export/build time;
they are explicitly marked as lower bounds in its timing correction artifact.
Its 264.691-second aggregate includes that work. Subsequent fits use the public
grouped export/reload and record the shared cost explicitly. The numerical
preparation/search protocol is unchanged. Exact public numerical sources are
archived for each fit; the preflight's source reconstruction matched all 67
hashes in its recorded dependency closure.

The following counts combine seeds 20261027, 20261028 and 20261029. Members
within a fit share preparation and are not independent statistical
replications. A passing member remains viable under the declared screen;
this does not select it for future use or establish superiority.

| Target | Verified and assessed members | Posterior-passing members | Fits with a passing member | Median preparation/search seconds | Median whole candidate-set seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gaussian32 | 63 | 0 | 0/3 | 127.386 | 188.924 |
| Rotated Gaussian32 | 37 | 0 | 0/3 | 52.453 | 80.042 |
| Banana8 | 8 | 3 | 2/3 | 96.595 | 112.123 |
| Centered funnel10 | 0 | 0 | 0/3 | 19.058 | 19.637 |
| Noncentered funnel10 | 47 | 4 | 2/3 | 71.014 | 95.289 |
| Regression32 | 26 | 3 | 1/3 | 37.713 | 54.104 |
| Regression64 | 35 | 3 | 2/3 | 154.425 | 380.796 |

All 21 fits returned a recorded result, with no preparation/consumer exception.
The three centered-funnel fits, regression32 seed 20261028 and noncentered
funnel seed 20261029 produced no verified candidate, so they have no retained
posterior assessment. The regression64 search for seed 20261027 was
`partial_budget`: its 181.512-second measured preparation/search crossed the
180-second allowance at a work boundary. All six verified members from that
partial search were assessed and failed the posterior screen. The other 20
searches completed their configured work. This does not establish the best
achievable fixed-HMC performance with a different preparation or budget.

Across the 13 passing members, per-member preparation/search/export/retention
costs ranged from 86.832–107.033 seconds on banana, 78.721–110.487 on the
noncentered control, 160.629–162.151 on regression32 and 146.402–168.178 on
regression64. Retained run costs alone, including archive writing and health
checks, were .679–1.958, .725–.887, 1.510–3.032 and 4.929–6.430 seconds
respectively. These ranges report every passing member rather than selecting
the fastest. They are descriptive and use different timing boundaries from
the NUTS numerical-kernel table.

On the rotated Gaussian, all three HMC preparations retained identity mass:
every slow window failed the public temporal-information requirement for a
metric update. For seed 20261028, minimum window ESS was 3.76, 3.58 and 3.14,
below that protocol's diagonal threshold of 6 and dense threshold of 33.
The declared `allow_valid_incumbent` policy allowed the existing identity
metric to survive. NUTS adapted dense metrics on this target. This is a
specific preparation limitation of the tested public procedure, not a
comparison that isolates fixed versus dynamic trajectory length. A separate
comparison with equally effective geometry preparation would be needed for
that inference; merely increasing the number of candidate trajectories does
not resolve the metric difference.

Acceptance qualification is not posterior qualification. A fixed trajectory
can return close to its starting point in a Gaussian mode despite acceptable
integration error. The independent Gaussian archive diagnostic measures
leapfrog phases as an explanation only; it does not change the candidate set
or select a new trajectory length. Longer retained runs, a separate prospective
mixing-aware comparison, or the repository's randomized-HMC method could
address different failure mechanisms. Their outcomes are unevaluated here.

The final independent archive check covered all 20 Gaussian preflight members.
Maximum affine-coordinate residual was 1.776e-15, fresh-target/cache residual
was zero, reconstructed MH energy-ratio residual was 3.553e-14, and
accepted/rejected selected-state residual was zero. Retained score flags were
finite. These checks exclude those specific coordinate, cache and MH-selection
errors as explanations for the Gaussian failures. They do not prove that
every HMC implementation detail is correct. The observed minimum absolute
sine of Gaussian leapfrog mode phase ranged from .0056 to .1334 across those
members, consistent with some nearly returning modes; that remains an
explanatory diagnostic rather than a causal attribution or tuning rule.

## Engineering findings and repairs

The first audit failed at tracing because a minimum operation inferred an FP32
literal in an FP64 expression. Reordering its operands fixed that type error.
The second reached all 180 common-variate native transitions and the normal
lifecycle tests, then exposed an XLA limitation on empty TensorArrays for
zero warm-up. A static empty-result branch fixed that case. Attempt 03 passed
all 16 focused edge tests. Attempt 04 passed all 113 tests across the new
port, prior adapter, native targets and retired-route scope regressions.
The final suite in attempt 16 passed **128 tests, with no skips**, including
six direct-search/native cases, duplicate-key rejection and eight native
initialization-service cases. The tests preserve Stan's actual code in the
oracle and include finite-gradient-sum overflow as an initialization failure.

Attempt 05 executed 182 native transition fixtures on GPU with exact agreement
on discrete results and random consumption. The maximum scaled state/energy
residual was 2.775564107262884e-16. Full unit/diagonal/dense warm-up and sampling
compiled to XLA. GPU allocator peak was 160000 bytes for this tiny fixture;
host peak RSS was 1770828 KiB. The artifact verifies memory growth. These are
engineering fixtures, not posterior validation or general memory bounds.

The final CPU graph-size test produced 1089 nodes at maximum depths 3, 10 and 20
(1045 before the independent-key guard),
showing that the implementation does not exponentially unroll the tree.
The stack holds depth times dimension state. Transition work can still grow
exponentially with depth, as in Stan.

Review found that the initial benchmark launcher used a shared stream key
with different counters for its direct-port chains. This creates overlapping
random streams and is wrong for independent-chain/seed claims. Attempts 06
and 08 were marked invalid for direct-arm statistical and layout promotion;
attempt 09 was stopped and retained. Attempt 07 had independently failed on
the initialization synchronization token; that reporting bug was fixed.
The kernel now rejects duplicate keys in a batch. Launchers use distinct
4*replication_seed+chain keys with counters zero. Attempt 10 passed six focused
independence, continuation and graph regressions. Calibration restarted at
attempt 11. The native common-variate results remain applicable; the first
GPU lifecycle timing is only mechanics evidence, not chain independence.

Every attempt, command, exact source snapshot, failure, wall charge and
remaining budget is preserved under
`docs/benchmarks/artifacts/stan-full-port-20261027/`. The campaign used 23
launches and 5052.046 worker seconds (84.201 minutes), including failed and
invalid attempts and conservatively including native-oracle builds. This is
within the 5400-second/30-launch ceiling, leaving 347.954 seconds unused.
No worker is left running. Final reporting also checked that the HMC numerical
dependency hashes were unchanged across the snapshotted fits.

## Compatibility scope

The transfer targets Stan 2.37.0 Euclidean multinomial NUTS on generic flat
unconstrained TensorFlow densities. It includes the two proposal rules,
three stopping criteria, all three metrics, jitter, step search, adaptation,
explicit initial metrics, initialization masks/retries, thinning and resume.
It uses TF Philox rather than MIXMAX/Boost. The zero-dimensional interface is
an explicitly local constant-evaluation convention, not native-service parity.
Stan's model compiler, constrained named-variable initialization, generated
quantities and exact seeded streams are outside this sampler scope.
The complete [source coverage matrix](../reference/nuts-stan-coverage.md)
identifies the source, test evidence and deliberate interface differences.

## Decision record

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the standalone direct port within its declared interface | 128 tests and native-oracle/XLA checks passed | Seed defect repaired; contaminated calibration excluded | Unseen targets, floating-point boundaries and explicit API differences | Retain as an optional research sampler | All of Stan ported, exact native seeds or universal correctness |
| Do not promote a faster NUTS default | Transition speed objective not achieved on these fixtures | Six targets passed; centered funnel vetoed | No time-to-precision uncertainty comparison | Profile fixed-kernel tree, RNG and gradient costs before another optimization | General superiority of any method |
| Reject the centered-funnel configuration for posterior reporting | Fixed-budget posterior screen failed in every NUTS engine/seed | R-hat/ESS/precision failures, including overdispersed starts | Target-specific adaptation and required draw budget | Reparameterize or separately retune; noncentered control is viable | Failure of the NUTS research direction |
| Retain fixed-HMC results as a bounded public-protocol comparison | All 21 fits recorded; all 216 verified members assessed; 13 passed in 7 fits | Gaussian cohorts failed; five fits yielded no member; one search was partial | Preparation adequacy, retained budget, seed variation and unmatched cold/I/O costs | Investigate geometry preparation and validate prospective members with adequate retained precision | Best achievable HMC, isolated fixed-L versus NUTS effects, or a selected efficiency winner |

| Inference status | Current evidence |
| --- | --- |
| Hard veto screen | Native source fixtures and Gaussian archive checks pass; contaminated calibration excluded; centered-funnel NUTS and nonpassing HMC cohorts are ineligible for posterior/efficiency promotion |
| Statistically supported ranking | None |
| Descriptive-only differences | All runtime, ESS/MCSE and between-engine differences in this campaign |
| Default-readiness | No production-default change |
| Next evidence needed | Matched cold/reuse cost boundaries, replicated time to precision, and target-specific repairs where posterior screens fail |

## Terminal review topics

The checks test the actual Stan headers and native model, not a second Python
implementation. Density/gradient parity precedes posterior comparisons, and
failed posterior screens precede efficiency interpretation. Calibration and
main seeds are distinct, chain keys are distinct, all warm-up draws are
excluded, and overdispersed-start results are separately preserved.

The strongest alternative explanation for the regression total-time change is
different warm-up integration work and wrapper costs, rather than a better
tree traversal. Frozen-kernel profiles could overturn that interpretation.
The weakest performance evidence is the small number of repetitions and the
different native/TF I/O boundaries. GPU banana needs profiling before causal
claims. The fixed-HMC tuner searches a candidate set and performs explicit
verification; Stan adaptation yields one kernel per chain. Charging both
procedures is necessary, but their costs are not pure transition timings.
The rotated-Gaussian metric mismatch and HMC cold compilation prevent a clean
fixed-versus-dynamic trajectory attribution. No failed screen is repaired by
choosing a favorable seed, relaxing a threshold or silently changing a target.
These limits prevent sampler superiority and default-readiness conclusions.

The completed experiment rejects particular configurations and precision
budgets, not either research direction. The planned noncentered geometry
control was executed and passed for NUTS. The public HMC expansion continued
after its Gaussian failures and found viable members on other targets. There
is no unexecuted repair phase hidden behind an early stop. A sequential
time-to-precision study and performance optimization remain unevaluated
follow-up work, rather than deliverables claimed by this bounded comparison.

Plan and terminal review are implementing-agent self-review; no independent
reviewer was used. The native executable remains an independent numerical
comparator despite that review limitation.

## Reproduction and preserved evidence

The base Git commit was `76b862db28098fc44d20cc1621b391847c0f78f7`; the
worktree was dirty. Each attempt retains exact source hashes and copies, so
that commit alone must not be treated as the executed source. Python was
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, with TensorFlow 2.20.0 and
TFP 0.25.0. The CPU host is an AMD EPYC 7742; the GPU calibration used an
RTX 4080 SUPER. CPU workers explicitly hid GPUs, and GPU workers verified
memory growth before initialization. FP64, XLA on and TF32 off were recorded.
Regression fixtures use stateless design seed `(20261027, 61)`; main sampler
seeds, calibration seeds and stress-start settings are recorded separately.

- [Campaign manifest and attempt ledger](../benchmarks/artifacts/stan-full-port-20261027/campaign-manifest.json) retains exact commands, charges, environment, source provenance and output paths.
- [Main comparison summary](../benchmarks/artifacts/stan-full-port-20261027/comparison-summary.json) includes all 84 engine/target/seed fits and every fixed-HMC member result.
- [Overdispersed summary](../benchmarks/artifacts/stan-full-port-20261027/overdispersed-summary.json) keeps the 27 stress-start fits separate.
- [Final Gaussian archive audit](../benchmarks/artifacts/stan-full-port-20261027/23-final-hmc-archive-audit/audit.json) records the coordinate, cache and MH residuals.
- [Rotated-Gaussian metric diagnostic](../benchmarks/artifacts/stan-full-port-20261027/fixed-hmc-rotated-metric-diagnostic.json) records the three preparation outcomes and window evidence.
- [Reset memo](bayesfilter-full-stan-nuts-port-reset-2026-10-27.md) records scope and continuation limits.

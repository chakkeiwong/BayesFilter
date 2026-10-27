# Wider NUTS geometry and execution overhead: diagnostic results

The completed standalone study is committed as `c72299c10`. This follow-up
adds six analytic targets and directly counts the work hidden by TFP's
chain-local leapfrog statistic. It does not establish a sampler ranking or
replace a production inference route. All numerical sampler/adaptation loops
ran on CPU with explicit signatures and XLA, float64, identity mass, and four
chains unless the batch-width diagnostic says otherwise. GPU devices were
intentionally hidden. The wider target set has not undergone a new CPU/GPU
comparison, so CPU remains a provisional choice inherited from the first pilot.

## Why the extra 1.83 steps understated the cost

The old comparison reported 4.83 NUTS leapfrogs per chain versus HMC L=3.
That is not the number of dense gradient evaluations executed by the batched
program. Installed TFP 0.25.0 keeps both tree loops active while any chain
continues, calls the integrator on the complete batch, then increments the
reported leapfrog count only for chains whose active mask remains true. The
source anchors are `_build_sub_tree`, `_loop_build_sub_tree`, and
`_loop_tree_doubling` in the archived `nuts.py`.

A separate resource-counter replay counted target calls without instrumenting
the timing function. Every output matched the corresponding uninstrumented
run exactly. Each call returns both the target value and its autodiff score;
the one bootstrap evaluation is subtracted below. Same fixed epsilons as the
previous banana comparison, 1,024 retained transitions, depth cap 6, seven
repeated timings of identical inputs. Different batch widths have different
RNG streams and trajectories; they are workload probes, not paired sampler
comparisons.

| Method / batch | Reported leapfrogs per chain | Dense gradient calls per transition | Median batch time |
| --- | ---: | ---: | ---: |
| HMC / 4 chains | 3.000 | 3.000 | 7.47 ms |
| NUTS / 1 chain | 4.782 | 4.782 | 32.64 ms |
| NUTS / 4 chains | 4.869 | 9.395 | 25.24 ms |
| NUTS / 16 chains | 4.920 | 16.532 | 132.30 ms |

For four chains, roughly 48% of dense row evaluations occur after those rows
stop contributing to the reported leapfrog count; at sixteen chains the
fraction is about 70%. These are derived from 1 - reported/dense-call counts.
The measured maximum chain count happens to equal the counted batch calls in
these probes, but is not assumed to be an exact counter for every tree path.

NUTS also performs RNG splitting, direction decisions, weighted candidate
selection, momentum accumulation, subtree U-turn tests, and tensor memory
updates. HMC only selects its terminal proposal. Those operations remain
necessary work after XLA compilation. Their individual runtime contributions
have not been profiled. Static optimized-HLO operation counts are not runtime
weights and are not evidence that one operation dominates.

The new probe's observed NUTS/HMC time ratio is 3.38, whereas the earlier
12-seed comparison was 5.15. Different starts, trajectories, timing length,
cache state and machine load prevent treating this change as an optimization.
No numerical NUTS optimization was introduced.

## Execution layout experiment

Four scalar NUTS chains were executed serially and through four persistent
CPU worker threads, with identical initial positions, epsilons and RNG seeds
for each chain. All outputs matched exactly. Method order was randomized for
seven paired timing repetitions. Median elapsed times were 145.59 ms serial
and 36.63 ms threaded, a descriptive 3.97 ratio. This confirms that the
compiled scalar functions can execute concurrently. It is not faster than
the observed 25.24 ms batched call; batch and scalar RNG streams differ, so
even that comparison is descriptive. Do not replace batching by default.

The plausible engineering candidates are smaller chain groups or independent
chains for targets with uneven trajectory lengths; compact, reusable tree
state and reduced RNG/control overhead; and hardware-specific layouts when
target gradients are expensive. Preserve tree stopping and selection
semantics. Increasing unrolled leapfrog steps or truncating tree depth changes
the trajectory policy and cannot be presented as a free compiler optimization.
A profiler attribution and matched-work experiment should precede a custom
NUTS implementation.

## Broader geometry pilot

Each method calibrated its own epsilon. HMC searched L=3,8,16 and both methods
searched acceptance targets .8,.9. A single tuning seed used 256 warm-up plus
256 retained transitions. Three fresh validation seeds used 512 discarded
fixed-kernel transitions plus 1,024 retained transitions per chain. Starts
were exact reference draws; this measures stationary exploration rather than
robustness to poor initialization. The depth cap was 8. These convenience
budgets are too small to support scientific superiority.

| Target | HMC L | HMC batch ms | NUTS batch ms | Maximum R-hat HMC / NUTS | Short R-hat/movement screen |
| --- | ---: | ---: | ---: | ---: | --- |
| Isotropic Gaussian, d32 | 3 | 20.79 | 94.83 | 1.0063 / 1.0055 | Both pass |
| Rotated Gaussian, d32, covariance condition 1000 | 8 | 39.88 | 929.97 | 1.1754 / 1.0070 | HMC fails; NUTS passes |
| Product banana, d8, bend 1 | 16 | 19.73 | 133.73 | 1.0051 / 1.0045 | Both pass |
| Centered funnel, d10, neck SD 3 | 16 | 19.04 | 889.64 | 1.3254 / 1.1060 | Both fail |
| Exact noncentered funnel coordinates, d10 | 3 | 10.69 | 80.77 | 1.0025 / 1.0042 | Both pass |
| Gaussian regression, 512x32 design | 3 | 162.38 | 728.62 | 1.0051 / 1.0044 | Both pass |

All times are medians across the three retained batches. R-hat includes
physical coordinates, analytic standard-normal reference coordinates, and
squared reference coordinates. Bulk/tail ESS and reference moments are saved.
A passed short screen is not posterior qualification, and no reference-
agreement uncertainty gate or efficiency uncertainty interval was designed
for this pilot. ESS-rate differences therefore remain descriptive.

The harder banana illustrates that masked work is not the whole explanation:
NUTS reports 17.52 steps versus HMC L=16; its batch maximum averages 26.15, while
its measured execution costs 6.78 times as much. Profiling is required to
separate tree/RNG overhead from other lowering and hardware effects.

The Gaussian regression evaluates the 512x32 design matrix at every target
call rather than the equivalent collapsed quadratic. Its analytic posterior
provides a reference while retaining matrix work. It is not a real-data
filtering example or a demonstrated GPU crossover workload.

## Longer-HMC-grid repair

The anisotropic Gaussian invalidates any conclusion based only on raw speed.
The small-grid HMC nominee fails its mixing screen, whereas NUTS passes and
uses about 66 leapfrogs. The third worker therefore expanded HMC's grid to
3,8,16,32,64,128 and repeated NUTS calibration, using fresh tuning and
validation streams with seed offset 10000. The first pilot remains unchanged.

The short tuning criterion selected HMC L=3, epsilon 1.19824. It again fails
validation (maximum R-hat 1.4574); NUTS epsilon 1.20969 passes the short screen
(maximum R-hat 1.0062). Observed minimum reference-ESS rates are 223/s and
1,201/s respectively, but the HMC failure prevents a valid comparison.
Longer L alone did not repair this finite-grid nomination procedure. That
result does not show that well-tuned or preconditioned fixed HMC cannot solve
the target. A stronger tuning sample, trajectory randomization as a separate
baseline, and matched metric adaptation deserve a new bounded experiment.
Do not retune on these saved validation seeds.

## What the Stan comparison can establish

Official Stan source was downloaded and inspected locally, including
`base_nuts.hpp` transition and recursive `build_tree`, and the reference manual
sections on automatic tuning, step-size initialization, metric adaptation and
NUTS configuration. Source URLs, retrieval time and SHA-256 hashes are in
`audit-01/source-index.json`. No paper-level or latest-release claim follows
from a source checkout on a moving branch.

Stan's inspected implementation builds one chain's tree with native C++
recursion and scalar control decisions, avoiding this TFP cross-chain masking
pattern. Stan also supports windowed metric and step-size adaptation. Our
identity-mass toy comparison is not matched to that end-to-end configuration.
Those are source-backed distinctions, not a measured Stan speed advantage.
No CmdStan installation was found in the standard local paths, and Stan was
not run. Its C++ sampler is not XLA, so it must be labeled an external
reference if added to the user's fully-XLA implementation comparison.

A fair Stan experiment needs two separate questions: (1) execution cost with
identical target, precision, metric, epsilon, depth and core allocation,
reporting time per actual gradient; (2) end-to-end efficiency with each
method's adaptation, all setup charged and posterior checks required. Compile,
warm-up, retained sampling and CSV/output time must remain separate. RNG and
U-turn implementations need not produce identical paths, so use replications
and uncertainty rather than asserting stepwise equality across libraries.

## Decisions, uncertainty and reproducibility

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Correct leapfrog cost interpretation | Exact replay counters pass | No replay/finite-cache failure | Timing contributions beyond dense calls | Profile RNG/tree/control and target cost separately | All overhead attributed |
| Keep batched NUTS available | Lowest observed cheap-target layout time | Scalar output parity passes | Different batch RNG paths; shared-host variation | Matched stream grouping experiment | Optimal layout or universal CPU preference |
| Retain NUTS on hard geometry | Rotated-Gaussian short screen passes twice | Selected HMC fails twice | Short tuning and validation; identity metric | Better calibrated HMC/metric comparison | NUTS superiority over tuned HMC |
| Reject centered-funnel nominees | Both fail short mixing screen | Posterior screen failure | Neck coverage and adaptation | Noncentered/metric controls with longer checks | NUTS research direction rejected |
| End current pilot | Three planned worker launches completed | Attempt cap reached | No Stan timings or profile attribution | Separate bounded follow-up | Unused time authorizes another launch |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Source/shape/score/XLA checks and counted replay pass; mixing screens fail for stated candidates |
| Statistically supported ranking | None |
| Descriptive differences | All wall times, ESS rates, short R-hat and serial/thread speed ratio |
| Default readiness | Not evaluated; no production defaults changed |
| Next evidence | Valid matched comparators, longer independent chains, adaptation and runtime profiles, replicated uncertainty |

Prelaunch target tests passed 8 checks, including inverse identities, exact
transformed density and finite-difference scores for all seven target/control
variants, plus a full-XLA centered-funnel sampler smoke. Final combined tests
passed 22 and failed the executable-source inventory guard because archived
source snapshots ended in .py. Renaming those 12 snapshots to .py.txt
preserved every byte/hash and launch manifest; the focused guard then passed.
The launcher now archives source as text. This was an archive-path defect, not
a numerical repair; no sampler rerun was needed.

Artifacts: `docs/benchmarks/artifacts/nuts-geometry-20261027/`, attempts
`overhead-01`, `geometry-01`, `geometry-repair-01`. The preservation audit
verifies 12 source snapshots, 157 recorded optimized-HLO hashes, one trace per
recorded function and 216 saved tensor files. TensorFlow's warning about many
functions does not indicate retracing within one recorded static function.
The counter diagnostic additionally compiles instrumented copies whose HLO
is not saved; the 157 count refers to the uninstrumented functions in manifests.

Worker time is 24.1602 + 291.8212 + 79.4594 = 395.4408 seconds under the 1200-second
cap, with all three launches consumed. Console logs, exact commands, source
snapshots, seeds, selections, CPU settings, versions and tensor checksums are
preserved. Initial commit c72299c10 contains the preceding completed study;
large raw tensors/HLO remain local while compact reports and hashes are
versioned. No package, environment, GPU or production sampler change occurred.

Post-run red team: this workload is dominated by synthetic FP64 targets,
short chains and timing-dependent one-seed nomination. An expensive filtering
target, learned/estimated metric, GPU matrix work or better HMC tuning could
change the observed ordering. The strongest present conclusion is the exact
masked-work count; the weakest is extrapolation of toy execution timings.
The outcomes invalidate particular pilot candidates and one archive filename
choice, not the checked target mathematics or NUTS research direction.

Checkpoint: the next engineering task is runtime attribution and matched
chain-group scheduling; the next sampling task is a stronger independently
validated HMC/metric baseline on anisotropy and a parameterization study on
the funnel. Stan remains an unmeasured external reference.

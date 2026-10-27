# Tuned fixed-length HMC versus NUTS on the banana

The October 27 follow-up asks how the standalone NUTS benchmark compares with
tuned fixed-leapfrog HMC. This extends the existing standalone exception using
the same `benchmark_nuts_xla.make_chain` numerical authority. It does not issue
a public HMC tuning artifact or change q20, production tuners or defaults.
The previous eight-launch pilot is closed; this is a new bounded comparison.

## Question and evidence contract

For the same 2D banana (b=.5), identity mass, float64 and CPU/XLA execution,
does a fixed-L HMC configuration nominated by a separate tuning sample deliver
more effective samples of relevant observables per second than tuned stock
TFP NUTS? The prior device pilots nominate CPU for this cheap workload, not for
all models. Compare the algorithms at their own calibrated epsilon; equal
epsilon would not equalize accuracy or efficiency.

Baseline ladder: HMC L=1; prior contextual L=8; grid-nominated fixed-L HMC;
stock NUTS with its own acceptance-target search. There is no enhanced NUTS
implementation yet. HMC L is fixed throughout every trajectory and retained
run, with no per-transition random length. Both methods use fresh momenta and
the same target, precision, chain count and start distribution.

Primary held-out metric: minimum mean ESS over x, y, x^2, y^2 and x^2*y, divided
by synchronized retained sampling seconds. This avoids choosing HMC from only
an easy/antithetic coordinate mean. Report raw throughput, ordinary coordinate
bulk/tail ESS/sec, warm-up and compilation costs separately. Timing excludes
diagnostics and I/O for both methods. The expected observables are
(0,0,1,1+2b^2,2b)=(0,0,1,1.5,1), derived in the earlier benchmark plan.

Tuning records every candidate. Nominate each method by median primary metric
on three tuning replicas, then freeze the selected L, acceptance target and
median adapted epsilon before opening the held-out seeds. This is finite-grid
benchmark tuning, not global optimality or canonical tuning admission.

Held-out comparisons pair seed IDs and common initial positions, with independent
RNG substreams for starts, warm-up and retention. Randomize method order for
each replica. For each pair compute log(HMC primary metric/NUTS primary metric).
Report a paired bootstrap 95% interval for the geometric-mean ratio (4,096
bootstrap resamples in TensorFlow), and an exact two-sided sign test as a
small-sample robustness check. Claim a within-target efficiency difference
only if the interval excludes one, sign p<.05, and held-out validity screens
pass for both arms. No cross-target or default-readiness inference follows.

## Diagnostic roles and stop conditions

| Diagnostic | Role | Action |
| --- | --- | --- |
| Full-chain XLA failure, wrong device, invalid retained state/cache or acceptance telemetry | Continuation veto for affected cell | Repair and retry in a fresh attempt; no eager fallback |
| Finite state and score; nonconstant observables | Engineering validity veto | Do not calculate a ranking from invalid draws |
| Rank/folded R-hat | Reporting only during tuning | Never reject, rank or steer tuning from R-hat |
| Held-out R-hat <=1.01 for coordinates and five observables; pooled reference residuals within 4 MCSE for both methods | Promotion veto only | Without this, efficiency remains descriptive; failure does not reject the research direction |
| Native divergence, energy flags, tree-depth hits | Explanatory only | Report; never prescribe epsilon direction or remove rejections |
| Failed moment screen | Repair trigger for a later longer validation plan | Do not tune on the held-out draws |
| Total budget or attempt cap | Continuation veto for further launches | Report partial/unresolved evidence |

Reference residuals use independent replicated estimator means: MCSE is sample
SD of the held-out replica means divided by sqrt(replica count), not an IID
assumption across correlated draws. The four-MCSE band is a conservative
convenience screen for five moments, not an exact simultaneous confidence
guarantee or proof of unbiasedness. All seeds/replicas remain in the report.

## Defaults, provenance and pre-mortem

| Choice | Provenance/status | Risk / early diagnostic |
| --- | --- | --- |
| Banana b=.5, 2D, float64, identity mass | Same existing benchmark target | Narrow generality; exact moments and score tests |
| CPU, four requested intra-op threads | Prior pilot nominated CPU; thread count inherited | Environment/load effects; random method order, record load |
| HMC L in 1,3,5,8,9,13,18,25 | 3,5,9,13,18,25 from public registry; 1 naive, 8 prior arm | Finite grid may miss optimum/resonance; held-out check |
| Acceptance targets .65,.8,.9 for both | Convenience search hypotheses, not asserted optimum | DA can miss target; record realized epsilon and acceptance |
| NUTS depth6, unroll1, energy threshold1000 | Previous pilot / installed TFP settings | Report cap hits and proposal flags |
| Tuning 512 warm-up + 1,024 retained, 4 chains, 3 seeds | Bounded convenience budget | No tuning efficiency claim; independent held-out confirmation |
| Held-out 1,024 frozen warm-up + 4,096 retained, 4 chains, 12 seeds | Convenience precision ladder | R-hat and exact-moment screens may still fail; no silent extension |
| Initial positions 2*N(0,I); distinct RNG streams | Prior spread with repaired stream separation | Initial transients; frozen warm-up and reference screens |
| DA remaining settings | Stock TFP .05 shrinkage,10 smoothing,.75 decay,10*epsilon anchor | Shared baseline, not universal default |
| R-hat1.01; confidence95%; sign p<.05; 4,096 resamples | Repository posterior screen / conventional inference / convenience bootstrap count | Few replicas and post-selection variation; qualify inference |
| Max 900 worker seconds, 3 attempts, 600s per worker | Conservative new compute cap | Failures charged, no unbounded retry |

Skeptical audit: the old L=8 arm adapted epsilon only, so calling it tuned in L
would be wrong. Optimizing seconds/draw alone ignores dependence and may select
short, sticky trajectories. Comparing identical epsilons would unfairly constrain
one algorithm. Reusing selection seeds for confirmation would overstate evidence.
The revised plan independently tunes both methods, freezes candidates, includes
nonlinear observables, and uses fresh paired repetitions and reference screens.
The public tuner registry was inspected; this standalone TFP experiment has no
canonical artifact authority and is not a replacement public tuning route.
Audit passes for this bounded single-target comparison.

## Execution and preservation

Driver: `docs/benchmarks/compare_nuts_fixed_hmc.py`. Interpreter:
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. Set CUDA_VISIBLE_DEVICES=-1 and
TF_FORCE_GPU_ALLOW_GROWTH=true before framework import. GPU devices are
intentionally hidden. Pure numerical warm-up and retained loops reuse full
XLA functions; host setup, scheduling, timing, diagnostics and artifacts are
outside the sampler timings.

Fresh root: `docs/benchmarks/artifacts/nuts-fixed-hmc-20261027/attempt-01/`.
Preserve source snapshots/hashes, Git HEAD, versions, commands, device settings,
every tuning candidate, frozen selection, held-out states/telemetry, optimized
HLO and result JSON. Use fresh directories for any localized repair. Record
actual attempts, time, interpretation, inference/decision tables and limitations
in `bayesfilter-nuts-fixed-hmc-result-2026-10-27.md`.

## Local repair after attempt 01

All 27 tuning candidates completed and the selection was frozen before heldout
generation. Attempt 01 stopped while writing replica 1009: one NUTS chain
never moved, giving an undefined upper-tail ESS and NaN that strict JSON
rejected. State/cache and acceptance telemetry remained finite; this is a
validity failure of the current candidate and a serialization repair trigger,
not a reason to alter the target, candidate, seed or selection criterion.
Serialize unavailable diagnostics as null with explicit field paths, preserve
zero-movement counts, and fail the predeclared posterior screen. Attempt 02
reuses the exact saved frozen selection and reruns the same 12 heldout seed
IDs in a new directory. These are not 12 additional independent replications.
No retuning or adjustment based on heldout results is permitted. Attempt 01
consumed 91.831236 worker seconds; 808.168764 seconds and two launches remain.
The localized source/timing repair leaves sampling graphs and method contracts
unchanged. Focused serialization regression must pass before retry.

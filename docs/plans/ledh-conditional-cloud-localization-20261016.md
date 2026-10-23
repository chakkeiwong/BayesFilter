# Conditional particle-cloud and particle-count localization

Status: COMPLETE, 17 October 2026, on `ledh-refactor-with-policy-fix`, starting
commit `ea1018de1`. The [result](artifacts/ledh-conditional-clouds-20261016/result.md)
records 832 valid evaluations, 27 passing focused tests and the failed guided
promotion screen. This bounded stage of the score master was
authorized by the owner's request to continue execution. Previous final
datasets remain untouched. The factorial result motivates this design but
does not select its data or tune its parameters.

## Research question and evidence contract

Does the persistent-condition model-score error mainly reflect particle-cloud
variability, a conditional mean error that persists across clouds, or both?
Does increasing N from 32 to 64 reduce either component? Compare current-only
and exact-future guidance at identical beta, data and particle randomness in
the existing shared covariance-mixture consumer. The exact Kalman model score
is the reference; an analytical derivative of the finite particle program is
the evaluated estimator. The primary coordinate is log process-noise SD;
report both drift coordinates and log observation-noise SD as well.

The primary contrasts are future-global minus current-global squared error,
and their N=64 minus N=32 changes, separately for each fixed dataset. A second
fixed-beta contrast uses the independently future-calibrated beta in both
guidance states. All conditional paired intervals and mean-error MCSEs use
particle clouds as the sampling unit. With only two datasets per regime,
population-level ranking, convergence, canonical LEDH admission, default
selection and cross-model/KDM claims are forbidden, regardless of outcomes.
No score metric selects a setting in this run.

Particle-count contrasts at fixed global beta isolate the count change.
Calibrated-arm count contrasts additionally include scope-specific beta
changes and are explanatory combined interventions. They must not be called
pure particle-count effects. Conditional variance separates particle noise
only at the tested datasets; two datasets cannot estimate population dataset
variability reliably.

Engineering pass requires every planned row, finite values, valid shared
consumer diagnostics, correct pairing and frozen beta. Missing/invalid rows,
wrong oracle/coordinates, source or random-input mismatches are continuation
vetoes. Calibration failure is a localized repair trigger: preserve the
attempt and retry the same objective/certificate within budget. Numerical
roundoff warnings remain observability, not automatic vetoes. No failed
candidate is a continuation veto.

The constructed heuristic adversaries are identity proposal (prior dynamics),
current-global (a stable pooled flow), current-local (local geometry) and
current-equal (untuned diversification). Add current-calibrated beta as the
optimized classical comparator. The eight arms are these five, future-global,
current with future-calibrated beta, and future with that same beta. The last
pair isolates guidance after calibration. Compare each future arm with all
four heuristics separately at every dataset and N; any observed MSE loss is
a descriptive promotion veto. Passing this screen cannot promote a default.
ESS, largest weight, likelihood error, runtime, moment residuals, and
bias/variance decomposition explain results; they never select a winner.

## Frozen design and assumptions

Use the same diagonal AR(2) model and predict-first convention, T=5,
theta=(0.9,0.9,0.3,0.5), Q=0.09 I and R=0.25 I. Evaluate ordinary observations
and a three-marginal-SD impulse at time index 2, coordinate 0. These are the
two persistent regimes implicated by the prior result. Leaving alpha=0.2
out narrows the conclusion and avoids spending compute on unaffected regimes.

Each (N, regime) has two independent calibration datasets and two independent
validation datasets with one particle cloud each. Freeze both current/future
betas before validation. Validation checks validity only and does not choose
hyperparameters. Final evaluation uses two fresh fixed datasets per regime
and twelve independent clouds per dataset, shared across all eight arms.
Total: 416 rows per N, 832 across both counts; 16 pilot calls in total.

Data seeds are 317000+1000*regime+100*partition+dataset, with partitions
0=calibration, 1=validation, 2=final. Cloud seeds are
417000+10000*regime+1000*partition+100*dataset+replicate. Calibration adds
100000*N to its cloud seed so each scope has independent calibration noise.
Evaluation clouds are shared across N: draw arrays of size 64 and use their
first N particles at every time, rather than rely on shape-dependent random
generation. Data hashes must agree across N; full-cloud hashes must match
within N and 32-particle prefix hashes must match across N. Guide coefficients
are recomputed from each dataset at nominal theta, then held fixed under
parameter differentiation, exactly as in the manuscript proposition.

| Choice | Provenance and status | Failure mode and earliest check |
|---|---|---|
| Epsilon 1 / Sinkhorn 5 | Owner's development baseline, not tuned/default evidence | Poor finite-N reset may remain; save mass/moment/score diagnostics |
| N=32,64; T=5 | Existing diagnostic scale plus one doubling; budget hypothesis | Two counts cannot establish an asymptotic rate; compare paired errors only |
| Two datasets, twelve clouds each | Bounded conditional diagnostic | Tail-sensitive intervals and weak dataset coverage; report every cell and maxima |
| Exact Gaussian future messages | Checked mathematical control; previous fitted/exact parity | Not evidence for a learned KDM guide; zero terminal precision and shared consumer tests |
| Beta calibration | Existing normalized multi-context second-moment objective; own scope | Weight objective may harm scores; frozen controls and heuristic comparisons |
| Beta floor 0.1, tolerance 1e-7, cap 20000 | Existing floor/certificate; cap supported by prior 4022-step repair | New contexts may need more iterations; preserve gap, objective, inputs, iterations |
| Flow 16, correction 2, moment policy | Existing shared diagnostic baseline, no numerical modification | Optional reset is not Contract E; validity/derivative call-chain checks retained |
| FP64 GPU/XLA, TF32 off | Explicit numerical-reference exception | Does not demonstrate production TF32 behavior or fair throughput |
| Exact Kalman CPU reference | Independent model-score authority | Coordinate/timing mismatch; test scale and preserve observations/reference |
| Bootstrap 10000, seed 517000 | Diagnostic uncertainty approximation | Twelve clouds poorly resolve tails; pointwise intervals, no population ranking |

For a fixed dataset write cloud error as e_r = score_r - exact score, with
R=12, mean error m and sample variance v=sum((e_r-m)^2)/(R-1). The exact
sample identity is MSE=m^2+(R-1)v/R. Report v, m, sqrt(v/R), m^2-v/R and MSE.
Under independent clouds E[v]=Var(e), and E[m^2]=E[e]^2+Var(e)/R, so m^2-v/R
estimates squared conditional bias without the finite-replication upward bias.
Its estimate can be negative; retain that value rather than clip it. A large
conditional mean alone is not evidence of bias unless its MCSE is considered.

## Skeptical audit and execution

Audit passes for this bounded diagnostic. The baseline, oracle and score
coordinates match; pairing separates clouds from data and preserves common
random numbers across N. Each scope calibrates independently without final
data. The heuristic set includes three distinct proposal choices and an
identity baseline. No weight, runtime or derivative-parity proxy is promoted
to model-score success. The central limitations are deliberately small data
coverage, pointwise bootstrap intervals, one particle-count doubling and the
optional noncanonical reset. These limit interpretation, not execution.

Pre-mortem: a lucky cloud or one dataset could suggest improvement, or a
calibration mismatch could masquerade as a guidance effect. Save all twelve
clouds, enforce input/beta pairing and report each dataset separately. The
N=64 scope must not import the N=32 calibration. Shared-consumer numerical
failure invalidates that cell and stops admission; poor statistical results
instead guide the next investigation. No algorithmic protection or tolerance
is changed in this stage.

New aggregate budget: 1800 GPU wall seconds, 180 CPU test/analysis seconds,
at most four GPU launches (two planned count scopes and two repair attempts).
Use separate processes for N=32 and N=64, on the available RTX 4080 SUPER with
escalated access and verified memory growth. Stop if no compatible shared GPU
is available, remaining budget cannot finish a scope, or a true validity veto
remains after a focused repair. A per-scope timeout is bounded by the remaining
aggregate budget. No environment changes, HMC, training, external agents,
commits, pushes or destructive work.

Environment: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`, `PYTHONPATH=.`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, UUID-selected GPU, XLA on, FP64/TF32 off.
CPU checks set `CUDA_VISIBLE_DEVICES=-1`. Commands:

```
python scripts/run_ledh_conditional_clouds.py --particles 32 --output docs/plans/artifacts/ledh-conditional-clouds-20261016/n32-attempt01 --budget-seconds 800
python scripts/run_ledh_conditional_clouds.py --particles 64 --output docs/plans/artifacts/ledh-conditional-clouds-20261016/n64-attempt01 --budget-seconds <remaining-minus-shutdown-reserve>
python scripts/analyze_ledh_conditional_clouds.py --n32 docs/plans/artifacts/ledh-conditional-clouds-20261016/n32-attempt01 --n64 docs/plans/artifacts/ledh-conditional-clouds-20261016/n64-attempt01 --output docs/plans/artifacts/ledh-conditional-clouds-20261016/decision.json
```

Each launch writes a manifest, frozen calibration, row stream, validity failures
and terminal status under a new directory. Manifests bind commit, source hashes,
command, environment, device/memory policy, data/seeds, controls, chunk extent
K=N, wall time, plan and result paths. Final decision and inference tables go
in `docs/plans/artifacts/ledh-conditional-clouds-20261016/result.md`; checkpoint
and master updates preserve the exact next action and remaining budget.

Pre-run verification: 21 focused CPU-only tests pass in 28.48 wall seconds,
including independent conditional-moment algebra, corrupted comparison
rejection, exact TensorFlow cloud prefixes and executable shared-filter/pilot
wiring. Harmless roundoff is accepted and material injected moment errors
are rejected. The RTX 4080 SUPER is available; GPU 0 is occupied and excluded.
The first launch uses an 800-second internal budget and 820-second supervisor
timeout; both are within the 1800-second aggregate campaign allowance.

Additional artifact-boundary checks reject reused clouds, changed stored
observations, mismatched calibration scope/beta and failed solver certificates.
There are now 27 distinct passing focused tests (21 analysis/cloud tests plus
six shared-consumer checks), at 35.95 aggregate CPU wall seconds. The runtime
source closure includes the lazily used moment-repair module explicitly;
all 79 recorded source files match at launch inspection.

N=32 attempt01 completes 416 valid rows; both ordinary and impulse calibrations
converge (11, 15, 14, 17 iterations). The artifact reader verifies scope,
observations and frozen calibration. Charge 306.874 GPU seconds, the larger
of the internal 306.874 and supervisor 281.55 clocks. The N=64 scope therefore
has 1493.126 seconds available; launch with internal cap 1400 and supervisor
cap 1420, preserving a shutdown margin and staying inside the aggregate cap.

Terminal audit, 17 October: N=64 also completes all 416 rows; its calibrations
converge in 6/16/15/17 iterations. Both scopes have zero invalid or warning
rows; all 79 source hashes and the analysis hash match. The paired analysis
passes data, cloud-prefix, calibration and full-design integrity checks.
There are 35 descriptive heuristic losses among 64 comparisons. No fixed-beta
primary guidance reduction is established; one global-beta interval indicates
conditional harm. Increasing N lowers observed global-arm MSE in seven of
eight cells, with five pointwise intervals excluding zero. Mean error and
variance both decrease descriptively in the stress cases. These are conditional
diagnostics, not a population ranking or an asymptotic-bias result.

Charge 815.378/1800 GPU seconds over two successful launches and 45/180 CPU
seconds conservatively (35.95 tests, 1.16 analysis, plus terminal inspections).
The unused budget is 984.622 GPU and 135 CPU seconds. No retry or numerical
core change was needed. The completed Gaussian localization has no remaining
continuation veto; the next master action is the nonlinear target/consumer
audit and its own fitting/comparison protocol, not another automatic Gaussian
ladder or an investigation of harmless roundoff.

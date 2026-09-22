# M21: actual warmup stops and interval calibration

M20 found no verified centered-funnel pair after the declared grid expansion.
The other two targets retained 39 verified members, but none of the eight
preselected posterior members met all declared checks. These results motivate
separating finite precision, initialization bias and the behavior of the
stopping controller. They do not justify relaxing acceptance or precision.

## Question and evidence contract

Does the actual shared sequential controller distinguish sufficient recent
mixing from delayed equilibration and a warmup cap, and how do its reported
intervals behave compared with independent fixed-count chains? First isolate
the controller using an exact Gaussian AR(1) transition; then exercise the
public ordinary HMC pipeline on Gaussian and beta-binomial controls. AR(1)
is a diagnostic fixture, not an additional tuner or a production sampler.

The fixture has stationary law N(0,1) and transition
`X[t+1] = rho X[t] + sqrt(1-rho^2) Z[t+1]`. It is exact, preserves chain
boundaries and uses independently seeded innovations. For fixed initialization
`x0`, its mean is `rho^t x0` and covariance is
`rho^abs(s-t) - rho^(s+t)`. Stationary random initialization instead gives
covariance `rho^abs(s-t)`. Thus both residual initialization bias and exact
fixed-count mean variance can be computed without a competing MCMC estimate.
These are local derivations from the linear recursion, not imported defaults.

Use `run_sequential_exact_transition` itself, its real posterior assessments,
and archived warmup/retained tensors. Do not replay precomputed diagnostics as
if the controller had run. A separate fixed arm starts from an independent
stationary draw or the same declared deterministic start pattern and uses
independent innovations, 10000 warmup and 10000 retained draws per chain.
Its count never depends on the stopped arm. Compare 95% normal mean intervals
using both the repository lugsail estimator and the exact fixed-count Gaussian
variance. The latter separates estimator error from known transient bias. It
does not confer an exact interval on a random stopping time.

Engineering success requires exact recurrence parity, disjoint streams,
correct final-state handoff, actual stopping decisions, all warmup excluded,
complete denominators and conserved budget. Statistical adequacy requires a
fixed untouched inventory, pointwise binomial intervals and explicit caps and
unavailable outcomes. R-hat/ESS/MCSE remain separate posterior checks. Neither
a failure to reject nor a short pilot closes calibration.

## Pilot and numerical assumptions

Run two independent development replications for each of six cases:

| Case | rho and start | Purpose and provenance |
| --- | --- | --- |
| iid | 0, stationary | Exact independent baseline |
| correlated | 0.8, stationary | Existing diagnostic baseline; integrated autocorrelation time 9 |
| slow_stationary | 0.98, stationary | Stress hypothesis; integrated autocorrelation time 99 |
| slow_shifted | 0.98, all chains at 8 | Eight stationary SD initial displacement; shared-start stress hypothesis |
| dispersed | 0.995, starts (-8,-4,4,8) | Delayed/capped readiness hypothesis; integrated autocorrelation time 399 |
| common_shift | 0.9999, all chains at 8 | Deliberate nearly immobile common transient; expected mean at 10000 is about 2.94 |

These values are diagnostic hypotheses, not HMC defaults. The first pilot
checks whether the cases actually exercise delayed stops, caps or false
readiness; labels never determine the outcome. Preserve cases that fail to
exercise the intended mechanism. A revised fixture uses fresh development
streams and documents the analytical reason before execution.

Use four chains and the inherited production-controller counts: warmup minimum
2000, recent window 1000, chunks 500, warmup cap 10000; retained minimum 1000,
chunks 500, cap 10000. Thresholds remain 1.05 and 1.01. Mean MCSE tolerance 0.05
is inherited from M20 for a unit-variance quantity. Lugsail uses the existing
sqrt(n), r=3, c=0.5 baseline; finite-sample calibration is under test, not
assumed. Stationary iid planning at 40000 total draws gives SE 0.005, whereas
rho=0.98 gives approximately 0.04975. This supplies an affordable comparator
and an intentional precision-boundary case without changing the tolerance.

Root seeds 2026092211 (development) and 2026092212 (confirmation) are convenience
identifiers. Derive separate case/replication/initialization/warmup/retained/fixed
streams. Save every seed and outcome. CPU execution hides GPUs before import,
uses one intra/inter-op thread and at most two workers. This is explicitly
controller/reference development, not GPU-default evidence. Numerical kernels
use stable TensorFlow signatures, with graph-only CPU reference execution;
GPU repetitions require XLA and verified memory growth. No training is involved.

## Replication, repair and stop rules

Pilot ceiling: 3600 CPU worker-seconds, including failures and focused tests.
M21 total ceiling remains 64800 CPU and 18000 GPU seconds. Each pilot worker
has a 300-second cap. The pilot estimates cost only; it is excluded from any
confirmation rate. Before confirmation freeze a separate manifest with cases,
counts, seeds, pass criteria and total reserved cost. At n=400, the normal
planning half-width is at most 0.049 at probability 0.5 and about 0.022 near
0.95. These are planning approximations; report exact binomial intervals.
If 400 replications per selected case are unaffordable, record the achievable
precision and leave the stronger claim open instead of sampling until a
desired interval passes. Keep all planned fits in the denominator.

Missing intervals count as failures of delivery and appear separately from
interval noncoverage. Report conditional coverage too. A stationary fixed
oracle discrepancy beyond a declared uncertainty screen invalidates the
harness and triggers repair before interpretation. A fixed lugsail discrepancy
with an accurate oracle motivates estimator diagnosis. Stopped-only discrepancy
motivates stopping-rule diagnosis. A readiness pass while the known transient
mean is material is a diagnostic limitation, not a tuning failure. Any proposed
rule or estimator change needs untouched confirmation before promotion.

Finite state or handoff corruption stops that experiment. A candidate cap,
warmup cap, precision cap or insufficient statistical power continues to its
planned diagnosis. Do not extrapolate one-dimensional AR(1) results to unknown
multimodal targets. The public HMC control fits use the existing ordinary tuner,
retain every verified candidate and choose the first candidate ID at L=3
before posterior data; their resolved inventory follows the pilot cost audit.

Artifacts live under `artifacts/hmc-repair-master-2026-09-16/m21-r1/`. Commands
use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; `run_controller_probe.py`
records the exact arguments, source snapshot, environment, tensors, elapsed
time and plan/result paths. Reuse M20's immutable merged-source snapshot;
new diagnostic code and tests receive their own source hashes.

Skeptical audit: the previous fixed-array diagnostic did not exercise stopping.
This design calls the real controller and distinguishes stationary from
conditional transient variance. A fixed comparator has independent streams
and counts. The deliberate high-correlation cases may cap even after their
initial bias has decayed; that distinction is reported. False readiness is
not tested by R-hat alone. Replication and cost are resolved before confirmation,
and pilot outcomes cannot alter a confirmation denominator. No numerical
default, universal burn-in certificate, sampler ranking or anytime coverage is
being promoted by this design.

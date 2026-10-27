# Wider NUTS geometry and execution-cost investigation

Owner request: commit the completed study, benchmark harder geometry, explain
why 4.83 reported NUTS leapfrogs cost much more than fixed HMC L=3, and find
an execution improvement. This is an extension of the standalone exception,
not a reopening of production NUTS routes.

## Research intent and evidence contract

Question: how much of the observed cost comes from masked batched work,
per-trajectory control, or target/gradient computation, and does scheduling
independent scalar CPU/XLA chains reduce elapsed time? Compare stock TFP NUTS
with the same stock kernel executed as four independent compiled chains; use
separately calibrated fixed-L HMC as a contextual comparator. Do not change the
NUTS stopping rule, RNG, multinomial selection, depth limit or integrator.

The wider pilot includes isotropic Gaussian, rotated anisotropic Gaussian,
strong product banana, centered funnel, its noncentered control, and a
Gaussian linear-regression target with matrix-based gradient work. These have
analytic standard-normal reference coordinates and cover global conditioning,
curvature, position-dependent scale, parameterization and target cost. They
are synthetic controls, not substitutes for a repository filtering-model run.

Primary engineering measurements: synchronized elapsed time per retained
batch and actual dense target/gradient calls from a separate counted replay.
The counter is an explicit diagnostic instrument and is absent from timings.
Its replay must match the uninstrumented outputs exactly before call counts
are interpreted. Report chain-local leapfrogs separately: max chain count is
only a lower bound on executed batch work, not an exact count.

Sampling diagnostics: coordinate and reference-coordinate rank/folded R-hat,
bulk/tail ESS, reference moments, movement, finite retained cache, acceptance
and tree saturation. All short-run ESS/rates are descriptive. No posterior
superiority or new default follows from this pilot, even if R-hat passes.
No observed failure may be removed from reports. Native divergence flags are
reporting-only and do not prescribe step-size direction.

Promotion criterion: this pilot can nominate an execution layout only. A
later claim requires independent repetitions, a predeclared uncertainty
comparison and joint reference/convergence screens. Nonfinite retained state,
incorrect score, counted-replay mismatch, non-XLA execution, wrong device or
missing provenance are continuation vetoes for the affected route. Failed
mixing is a candidate failure and diagnostic for later repair, not a reason to
abandon the other geometries. Expired budget stops further launches.

## Defaults, assumptions and skeptical audit

| Choice | Provenance/status | Failure mode and early check |
| --- | --- | --- |
| TensorFlow 2.20 / TFP 0.25, FP64, identity mass | Installed baseline; prior pilot | Version-specific cost; archive source hashes |
| CPU/XLA, 4 chains, 4 intra-op threads | Previous cheap-target pilot nomination | New matrix workload may favor GPU; no universal CPU claim |
| Gaussian d32, condition number 1000 | Convenience conditioning stress | Identity mass unfavorable; paired whitening control |
| Banana d8, bend 1 | Convenience stronger curvature | Tuning too short; exact inverse and gradient check |
| Funnel d10, neck scale 3 | Conventional funnel hypothesis | Neck exploration failure; centered/noncentered paired reference coordinates |
| Regression 512 observations x32 coefficients | Convenience compute stress | Cost too small for GPU; record target-gradient timing |
| HMC L=3,8,16; acceptance .8,.9 for both methods | Bounded inherited grid hypotheses | May miss optimum; label finite-grid pilot |
| One 256-warm-up/256-draw tuning seed; 512 frozen warm-up +1024 retained, 3 fresh validation seeds | Convenience bounded pilot | Noisy nomination and too short for difficult geometry; no convergence or superiority claim |
| Independent scalar chains in a four-worker thread pool | Execution hypothesis | Thread contention/dispatch; synchronize all four, record wall time |
| 1200 worker seconds, max 3 attempts, 600 seconds per worker | Bounded new local budget | Compile cost exhausts budget; preserve partial evidence |

Skeptical audit: interpreting 4.83 as executed batched work is wrong because
TFP continues dense integration while any chain remains active. Counting
target calls in the timed function would perturb the measurement; use a
separate exact-output replay. Scalar scheduling changes RNG streams and may
change trajectories, so its timings cannot establish an optimization benefit
without matched per-chain streams and validity checks. Stan C++ NUTS is not
XLA; it is an external comparator, not an eligible implementation of the
user's XLA requirement. No Stan installation was found in the standard local
locations. Source inspection is useful, but cannot establish a measured Stan
speed advantage. Audit passes for this bounded diagnostic scope.

## Execution and artifacts

Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, CPU hidden with
`CUDA_VISIBLE_DEVICES=-1` and growth environment set before import. No pfor,
NumPy numerical runtime, package installation, or uncompiled sampler fallback.
All sampler/adaptation/score loops retain explicit signatures and XLA.
Host scheduling, file writing and untimed diagnostics remain outside XLA.

Driver: `docs/benchmarks/benchmark_nuts_geometry.py`; target definitions in
`docs/benchmarks/nuts_benchmark_targets.py`. Reuse the existing compiled
`benchmark_nuts_xla.make_chain` with an optional target callable, not a copied
sampler. Save exact commands, source snapshots, seeds, HLO, environment,
timings, settings, samples and diagnostics under fresh attempts in
`docs/benchmarks/artifacts/nuts-geometry-20261027/`. Retain prior evidence.

Before any broad sweep, run score/transform checks and a small execution-cost
probe on the original banana. Then run the six-target pilot within the shared
budget. Record actual attempts, findings, decision/inference tables and the
next discriminating check in a result note. A source audit of installed TFP
and downloaded official Stan code provides mechanism evidence only.

Reference starts for the wider geometry pilot are drawn from the exact target
using independent stateless streams. This isolates stationary exploration from
the separate dispersed-start failure in the earlier banana study; it does not
test robustness to poor initialization. Reference coordinates and their squares
are the tuning observables. The noncentered funnel is sampled in its normal
coordinates; its speed is a parameterization result, not a new physical target.
The 32D Gaussian control also supplies the exact-whitened control for the rotated
Gaussian. All nominally positive ESS configurations remain in nomination;
tuning R-hat never rejects or steers a candidate.

## Bounded baseline repair after geometry-01

The initial six-target pilot completed in 291.8212 seconds; the earlier
counted-work probe used 24.1602 seconds. Both methods fail the centered-funnel
short mixing screen. The anisotropic Gaussian exposes an inadequate fixed-HMC
search: selected L=8 fails the short R-hat screen while NUTS uses roughly
66 leapfrogs. This does not establish NUTS superiority over tuned HMC.

Use the third and final planned worker for a separately labeled exploratory
repair on the rotated Gaussian: L=3,8,16,32,64,128 with the same acceptance
grid, and repeat the NUTS calibration. Geometric doubling is a convenience
extension of the earlier grid, not a proven optimum. Use seed offset 10000
for fresh tuning and validation streams; do not modify or pool the first
pilot. Preserve both versions. This consumes the existing budget, with
884.0186 worker seconds remaining and the same 600-second worker limit.

Skeptical repair audit: long HMC paths can resonate, so evaluate reference
coordinates and squares and retain all candidates. Three short validation
replicas cannot establish superiority. This repair tests whether the observed
failure was merely the too-short HMC grid; it does not repair the centered
funnel or convert pilot diagnostics into a confirmatory comparison. No other
scientific settings or gates change.

Exact launch: `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true timeout
600 /home/ubuntu/anaconda3/envs/tfgpu/bin/python
docs/benchmarks/benchmark_nuts_geometry.py --phase geometry --targets
rotated_gaussian32 --lengths 3,8,16,32,64,128 --seed-offset 10000 --output
docs/benchmarks/artifacts/nuts-geometry-20261027/geometry-repair-01`.

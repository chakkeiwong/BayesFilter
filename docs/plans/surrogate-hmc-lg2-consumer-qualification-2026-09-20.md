# LG2 full-consumer qualification

The research goal remains testing imperfect full canonical DCTR proposal fields
in endpoint-corrected HMC for model-parameter inference. This bounded diagnostic
checks whether the new LG2 consumer actually reaches that full implementation
with the correct coordinates, sign, independent exact-model endpoint and
per-chain arithmetic. It does not run retained inference.

Use the LG2 statistical scope and frozen attempt1 calibration data in
`surrogate-hmc-all-models-20260920-lg2-reference-01/reference`. The qualified
FP64 matrix Kalman reference uses precisely the same once-rounded FP32
observations as DCTR, with parameter-dependent stationary initialization and
the uniform-prior chart Jacobian exactly once. DCTR uses N=1024, K=1024,
FP32/TF32/GPU/XLA, the full shared analytical recursion, Contract E, GenUT,
UKF per-particle covariance and dual-cap correction. No reduced force is added.

The first consumer check explicitly supplies the existing post-invalidation
control hypothesis with protection multiplier1. This is an unselected warm
start. Its damping, ridges and caps can alter the field and may cause failures
on LG2. A dedicated LG2 calibration/validation non-harm curve is required
before this field can enter public tuning; LG1's selected artifact cannot
authorize it. Force accuracy versus Kalman is explanatory only. Poor force
accuracy does not invalidate endpoint-corrected HMC or cancel calibration.

At four fixed physical positions (.7,-.5), (1,0), (1.1,.5), (.55,.8), check:
actual consumer force/value/status versus direct full canonical fixture;
bitwise replay and permutation; row independence when one other row changes;
unchanged endpoint after force evaluation; sign and exact Kalman endpoint;
typed public binding; a two-step proposal involution and two corrected HMC
transitions. Step size .02 and identity inverse mass are explicit mechanics
hypotheses, not tuned geometry. Momentum rows (.2,-.1),(-.4,.3),(.6,-.5),
(-.8,.7) and seeds (20260920,201/202) are fixed before execution.

Involution tolerance is32 FP32 ulps at the observed coordinate/momentum scale,
matching the mixed-precision construction; record the measured error as well.
Per-chain energy and cached endpoint tolerances are1e-10 in FP64. Hard wiring,
determinism, numerical or per-chain failures invalidate this consumer check and
trigger repair. Finite zero-extension fallback must be recorded independently;
an invalid original field blocks tuning at this scope, without rejecting the
surrogate-HMC research direction. There is no posterior or efficiency promotion
criterion in this diagnostic. Heuristic dominance remains NOT_RUN.

Skeptical audit: a plausible endpoint with a partial initial-law derivative,
twice-added Jacobian or changed observations would test a different target.
The independent joint-Gaussian reference check and explicit endpoint equality
address those risks. A mocked wiring unit test cannot establish actual GPU
call-chain parity; the separate N1024 execution is required. Construction of
the full fixture alone is not proof that the consumer calls it.

Budget: use at most600 GPU seconds and450 CPU-worker seconds for one real
qualification, plus at most100 CPU-worker seconds for focused wiring tests.
These consume the existing LG2 P3 allocation, not additional campaign compute
or a retained-HMC attempt. Run GPU work only after the active LG1 C worker
finishes, with escalated access and verified memory growth. Preserve logs,
source snapshot, data/binding identities, every original status, timing,
allocator statistics, manifest and incremental accounting in fresh roots.
The watchdog stops on either budget; incomplete qualification is not a pass.

Execute `scripts/qualify_surrogate_hmc_linear_consumer.py --model LG2
--reference-root <frozen-reference-root> --output <fresh-output>` through the
master's watchdog with the tftwogpu Python. The next justified stage is LG2
control calibration if these checks pass; only a subsequent public
`tune_hmc_kernel` result with `VerifiedTensorFlowHMCConfig` can issue a retained
runner. The checked capability registry supports that exact endpoint-corrected
position-field binding in ordinary coordinates.

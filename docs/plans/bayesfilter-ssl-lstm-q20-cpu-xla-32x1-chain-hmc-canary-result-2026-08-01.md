# q=20 CPU-XLA 32x1-Chain NeuTra-HMC Canary Result

Date: 2026-08-01
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-cpu-xla-32x1-chain-hmc-canary-plan-2026-08-01.md`
Artifact: `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-chain-hmc-canary-2026-08-01/r1/summary.json`
Status: `PASSED`

## Outcome

The requested 33-core topology works: 32 independent one-chain CPU/XLA HMC
workers ran on physical CPUs `0..31`, while the supervisor remained pinned to
physical CPU `32`. All workers restored the same Seed-A best checkpoint,
compiled the same batch-native transformed q=20 HMC graph, completed two
synchronized warm calls with finite samples and traces, and exited with code
zero.

| Arm | Workers x chains | Cold compile + first call, max | Warm window, mean | Aggregate process calls/s | Aggregate chain transitions/s | Worker RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Matched baseline | `1 x 1` | `20.334 s` | `3.980 s` | `0.2512` | `0.7537` | `0.924 GiB` |
| Candidate | `32 x 1` | `33.031 s` | `5.185 s` | `6.1712` | `18.5136` | `29.527 GiB` aggregate |

The candidate achieved a descriptive `24.56x` aggregate speedup relative to
the exact one-worker/one-chain comparator, or `76.76%` parallel efficiency at
32 workers. Per-worker warm latency increased from `3.980 s` to `5.185 s`, a
`30.3%` contention penalty, while aggregate throughput increased by `24.56x`.

The two p32 warm windows were `5.164 s` and `5.207 s`, so the short-run rate was
internally stable. This is descriptive performance evidence from two
repetitions, not a statistically supported topology ranking.

## Time Estimate

Each timed call executed three transitions in one chain at one leapfrog step.
Using the measured p32 mean gives:

`5.185457806 / 3 = 1.728485935 seconds per transition-leapfrog`

For the policy minimum of `2,000` warm-up plus `1,000` retained transitions
per chain, using the current four-leapfrog kernel hypothesis:

`1.728485935 * 3,000 * 4 = 20,741.83 s = 5.76 h`

This `5.76 h` value is a derived linear estimate, not a measured long-chain
runtime. It excludes fresh kernel tuning, XLA compilation, 500-transition
chunk diagnostics and artifact I/O, any additional warm-up or retained draws
needed for R-hat/ESS, and possible nonlinear timing effects from a longer HMC
graph. A longer representative chunk must be timed before freezing a serious
campaign cap.

The 32 workers can cover 32 independent chains concurrently. The planned
two-transport validity test needs only eight chains (`2 transports x 4
chains`), so 32 workers are most useful for running independent tuning arms or
replications concurrently. They do not make the sequential transitions inside
one chain 32 times faster.

## Validity And Resource Screens

- Exact checkpoint SHA-256:
  `c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff`.
- Best trainer state: Seed A step `1,500`, hash
  `26aeca40e95cd9d18e52a340ea97b1c34377b6b926ca20aad4e3821930c008ac`.
- Every process recorded the same target and frozen transport hashes.
- Every worker recorded `CUDA_VISIBLE_DEVICES=-1`, no TensorFlow GPU,
  `jit_compile=true`, FP64, and its exact one-core affinity.
- All 33 worker logs contain an XLA compiled-cluster receipt.
- Every sample tensor had shape `[2,1,4]`; all samples and floating traces were
  finite.
- All 33 worker processes exited with code zero.
- Aggregate p32 worker RSS was `29.53 GiB`, below the declared `64 GiB` veto
  and far below the 251 GiB host capacity.
- Total p1+p32 canary wall time was `89.37 s`, below the `1,200 s` cap.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept 32+1 CPU process topology as mechanically feasible | All 32 one-chain workers completed finite CPU/XLA calls | No checkpoint, XLA, affinity, finite, memory, or worker veto | Only two short warm repetitions | Use this topology for bounded independent HMC tuning arms/replications | No convergence or posterior correctness |
| Retain p32 as a promising throughput lane | `24.56x` descriptive speedup and `76.76%` efficiency | No resource veto | Short graph may not predict long chunks | Benchmark a representative 500-transition/four-leapfrog chunk before the serious run | No statistically supported superiority or linear long-run scaling |
| Treat `5.76 h` only as a planning estimate | Exact arithmetic from the measured short-call rate | Not a pass/fail gate | Excludes overhead and assumes linear work scaling | Replace with a measured chunk rate before setting cap | No claim that the real validation will finish in 5.76 hours |
| Preserve CPU artifact classification | Mechanics and speed canary passed | Existing artifact remains `hmc_eligible=false` | Claim-bearing route admission remains unresolved | Build and review a payload-bound admission/tuning bridge | No transport promotion or CPU default |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for p1 and p32 mechanics |
| Statistically supported ranking | None; timings are descriptive |
| Descriptive-only differences | Cold compile, warm latency, throughput, speedup, efficiency, and RSS |
| Default readiness | Not assessed; CPU remains a diagnostic exception |
| Next evidence needed | Fresh payload-bound tuning and a measured representative longer HMC chunk, followed by sequential R-hat/ESS validation |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `882679796e8ee684b6b020b7cd84e3cfc1d92d58` with unrelated dirty worktree preserved |
| Environment | `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU | Workers CPU `0..31`; supervisor CPU `32`; GPU intentionally hidden before every child import |
| XLA/dtype | XLA enabled and receipt-verified in all 33 processes; FP64 |
| HMC shape | One chain/process, one burn-in plus two results, one leapfrog step, fixed step size `0.01` |
| Seeds | Cold `61000..61031`; warm folds `62000..65101` under root `20260801` |
| Wall/cap | `89.37 s` / `1,200 s` |
| Memory | Maximum aggregate p32 worker RSS `31,704,313,856` bytes |
| Artifacts | `r1/summary.json`, `r1/p{1,32}x1/result.json`, and 33 per-worker stderr logs under the artifact root |

## Post-Run Red Team

The strongest alternative explanation is that the p32 result looks favorable
because each compiled graph contains only three transitions; process startup,
chunk boundaries, and TensorFlow call overhead have a different relative weight
than in a 500-transition chunk. This is why the short-call rate is not adequate
to freeze the serious-run budget.

The result would be overturned for this topology by a current-source
representative-chunk run showing throughput collapse, nonfinite state, target
status failures, or memory growth beyond the declared cap. The weakest evidence
is the untested linear scaling from one to four leapfrog steps and from three to
3,000 transitions. The next benchmark should measure the actual proposed chunk
shape rather than repeating another tiny mechanics call.

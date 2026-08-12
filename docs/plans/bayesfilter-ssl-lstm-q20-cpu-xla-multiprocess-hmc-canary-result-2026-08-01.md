# q=20 CPU-XLA Multiprocess NeuTra-HMC Canary Result

Date: 2026-08-01
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-cpu-xla-multiprocess-hmc-canary-plan-2026-08-01.md`
Artifact: `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-multiprocess-hmc-canary-2026-08-01/r1/summary.json`
Status: `PASSED`

## Outcome

Independent CPU processes can run the same XLA-compiled q=20 NeuTra-HMC
mechanics concurrently. All `1`, `2`, and `4` process arms completed three
synchronized warm calls per worker with finite samples and traces, exact
checkpoint/target/transport binding, expected one-core affinity, no visible
TensorFlow GPU, XLA compile receipts, and zero worker exit codes.

| Processes | Cold compile + first call, max | Warm concurrent window, mean | Aggregate process calls/s | Speedup vs 1 | Parallel efficiency | Aggregate chain transitions/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `38.111 s` | `18.933 s` | `0.05282` | `1.000x` | `100.0%` | `0.6338` |
| 2 | `39.281 s` | `19.355 s` | `0.10333` | `1.956x` | `97.8%` | `1.2400` |
| 4 | `41.476 s` | `19.829 s` | `0.20173` | `3.819x` | `95.5%` | `2.4208` |

Each process call contains four chains and three transitions per chain: one
burn-in transition plus two returned transitions, with one leapfrog step per
transition. Thus one process call represents twelve chain transitions. The
clean independent-work scaling measure is aggregate process calls per second;
the chain-transition rate is the same result multiplied by twelve.

The observed four-process warm window is only `4.73%` slower than the
one-process warm window, while aggregate throughput is `3.82x`. This is strong
descriptive feasibility evidence for independent process-level CPU parallelism
at this small topology. It is not a statistically supported topology ranking
and does not establish scaling to dozens of processes.

## Validity And Resource Screens

- The exact Seed-A best checkpoint at step `1,500` was validated, restored,
  frozen, and reloaded independently in every process.
- Checkpoint SHA-256:
  `c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff`.
- Target signature:
  `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`.
- Frozen transport hash:
  `017c89c25a92ee394e3b181ee661e91b793d342f333f9f8f53b6ff925b5e4e16`.
- Every child recorded `CUDA_VISIBLE_DEVICES=-1`, no physical TensorFlow GPU,
  `jit_compile=true`, TensorFlow `2.20.0`, FP64, and affinity to its single
  assigned CPU.
- Every child log contains `Compiled cluster using XLA!`.
- Every warm sample tensor had shape `[2,4,4]` and passed the finite
  sample/trace screen.
- Maximum per-worker RSS was about `0.99 GiB`; four workers therefore required
  about `4 GiB` of worker memory, far below host capacity.
- Total canary wall time was `316.05 s`, below the `1,800 s` cap.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept CPU multiprocess XLA-HMC mechanics as feasible | All processes compiled and completed finite synchronized calls | No canary veto fired | Only 1/2/4 processes and three warm repetitions | Use a fresh payload-bound HMC tuning canary with process-level independent candidates | No convergence, posterior correctness, or CPU default |
| Treat process parallelism as a promising throughput lane | Four-process descriptive speedup `3.819x`, efficiency `95.5%` | No affinity, memory, or worker failure | Scaling beyond four processes is not measured | Run a prospectively bounded `8`-process rung before projecting large campaigns | No universal linear scaling or statistical superiority |
| Preserve CPU training eligibility classification | Mechanics canary passed | Existing CPU result still says `hmc_eligible=false` | Claim-bearing GPU/default policy remains separate | Create a reviewed input/payload admission bridge before retained HMC | No transport promotion or posterior evidence |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the bounded mechanics canary |
| Statistically supported ranking | None; three timing repetitions per topology are descriptive |
| Descriptive-only differences | Cold compile time, warm latency, throughput, speedup, efficiency, and RSS |
| Default readiness | Not assessed; CPU remains an explicit diagnostic exception |
| Next evidence needed | Fresh payload-bound kernel tuning, then sequential multi-chain R-hat/ESS validation under the eligible route |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `882679796e8ee684b6b020b7cd84e3cfc1d92d58` with unrelated dirty worktree preserved |
| Command | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/benchmark_ssl_lstm_q20_cpu_xla_multiprocess_hmc_2026_08_01.py --output-root docs/plans/artifacts/ssl-lstm-q20-cpu-xla-multiprocess-hmc-canary-2026-08-01/r1` |
| Environment | `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| CPU/GPU | AMD EPYC 7742; CPUs `0..3`; GPU intentionally hidden in every child |
| XLA/dtype | `jit_compile=true`; FP64; each child emitted an XLA compile receipt |
| Seeds | Root HMC seeds `20260801:51000..51003`; warm folds `52000..52302` |
| Wall/cap | `316.05 s` / `1,800 s` |
| Artifact paths | `r1/summary.json`, `r1/p{1,2,4}/result.json`, and per-worker stderr logs under the artifact root |
| Plan/result | This result and the plan named above |

## Post-Run Red Team

The strongest alternative explanation is that four-way scaling looks good only
because the topology is small relative to the 128 physical cores and remains
inside one NUMA node. That is plausible and is why this result does not support
a 25-, 50-, or 75-process projection. An eight-process rung with the same
synchronized protocol is the next smallest discriminating test.

The result would be overturned for this topology by a repeated current-source
run showing worker failures, nonfinite HMC state, missing XLA receipts, or
material aggregate-throughput collapse. The weakest evidence is the three warm
timing repetitions; they establish feasibility and a stable descriptive rate,
not a population-level performance ranking.

The canary tested actual transformed-target HMC mechanics, but the short draws
cannot evaluate convergence or posterior validity. The CPU training receipt
remains diagnostic and ineligible for posterior promotion until a separately
reviewed admission and HMC validation campaign succeeds.

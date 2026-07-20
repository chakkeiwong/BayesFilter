# SSL-LSTM q=20 Process-Grid HMC Tuning Result

Date: 2026-07-20

Status: `RESOURCE_PROJECTION_STOP_BEFORE_TUNING`

Plan: `docs/plans/bayesfilter-ssl-lstm-q20-process-grid-hmc-tuning-plan-2026-07-20.md`

## Outcome

The reviewed plan, focused implementation checks, contract smoke, and trusted
GPU/XLA rate/topology canaries completed. The complete q=20 fixed-metric grid
was **not launched** because the prospective eight-GPU-hour resource gate
failed. Therefore tuning did not succeed or fail: no grid candidate was tuned,
no survivor was selected, and the conditional HMC test was not authorized.

The result is direct. Process-parallel independent candidates do not accelerate
this q=20 workload on one RTX 4080 SUPER. The one-worker warm rate was
`1.47094` seconds per four-chain transition-leapfrog. Under the later common
source signature, two workers required `2.95259` seconds per worker unit and
four required `5.92838` seconds. Aggregate throughput was therefore essentially
flat:

| Worker processes | Max warm seconds per worker transition-leapfrog | Aggregate transition-leapfrogs/second |
| ---: | ---: | ---: |
| 1, earlier rate context | 1.47094 | 0.67984 |
| 2, common bound source | 2.95259 | 0.67737 |
| 4, common bound source | 5.92838 | 0.67472 |

Additional processes time-shared a saturated GPU. They did not reveal a memory
problem: each worker recorded only `381,568,512` bytes allocator peak and about
`4.47` GB conservative host `ru_maxrss`; four workers remained well below the
32 GB VRAM and 64 GiB host-RAM ceilings.

Using the measured four-worker maximum rate, the complete reviewed Round-0
grid `L=(3,5,9,13,18,25)` projects to:

| Prospective workload | Transition-leapfrog accounting | Projected wall with 50% margin |
| --- | ---: | ---: |
| No evidence extensions | `18,980` | `43,130.7 s` = `11.98 h` |
| All allowed 128-draw extensions | `47,231` | `105,936.7 s` = `29.43 h` |

Both exceed the eight-hour ceiling. The completed rate/topology canaries charged
`530.83 s` (`0.1475 h`), leaving `28,269.17 s` (`7.8525 h`). Even the
no-extension projection cannot fit. Launching the grid would have violated the
prospective evidence contract.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not launch the q=20 Round-0 grid under this cap | Not evaluated; candidate tuning never started | `RESOURCE_PROJECTION_STOP` fired before tuning | Whether a materially faster target evaluator, multiple physical GPUs, a statistically redesigned smaller grid, or a larger owner-approved cap makes tuning feasible | Redesign the cost/algorithm contract before another material run; train q=20 NeuTra separately only under its own authority and budget | No HMC tuning failure, no candidate rejection, no geometry verdict, no NeuTra verdict, no convergence or posterior claim |
| Do not launch the conditional HMC test | Required survivor does not exist because tuning was not launched | HMC-test gate correctly remained closed | None about the gate; sampler quality remains unknown | Run HMC test only after a valid complete-grid survivor exists | No HMC-test failure |
| Do not select process-parallel candidates as a one-GPU acceleration route for the bound source | Two- and four-worker aggregate throughput did not improve | No memory-growth, target-signature, finiteness, allocator, or host-memory veto fired | Tail variability of rate measurements; continuous timing is descriptive rather than a statistical ranking | Prefer one process per physical GPU or optimize the target/transition implementation before candidate parallelism | No universal process-parallel or multi-GPU performance claim |

## Research Question Guardian

| Question | Verdict |
| --- | --- |
| Did the result invalidate the harness? | No. Focused tests passed and every trusted canary produced finite, target-bound structured evidence. |
| Did it invalidate the implementation or target? | No. Target and adapter signatures matched across every worker in the two/four-worker artifacts. |
| Did it invalidate the data or mathematics? | No such evidence was collected. |
| Did the current candidate fail? | No candidate ran. This is a pre-run resource veto, not candidate rejection. |
| Does it reject plain HMC or NeuTra? | No. Plain-HMC tuning and NeuTra-HMC viability remain unevaluated. |
| What repair does it trigger? | A resource/algorithm redesign: one candidate process per physical GPU, faster per-transition target evaluation, a prospectively justified lower-cost tuning policy, or a larger explicit cap. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Result |
| --- | --- |
| Engineering correctness | `38 passed` focused tests after repairs, Python compilation passed, contract smoke passed, and lane `git diff --check` passed. The tests cover parent launch environment, target/start/metric lineage, transition-work projection, resource fail-closed behavior, process-grid semantics, and conditional HMC-test gating. |
| Numerical/sampler validity | Rate canaries executed the current q=20 plain target in FP64 on GPU/XLA with four batched chains, finite samples, exact target/adapter signatures, and verified memory growth. They are mechanics/resource evidence only. No candidate tuning or retained sampling ran. |
| Scientific interpretation | The bound single GPU is compute-saturated for this workload. Nothing was learned about posterior convergence, correctness, model adequacy, identity-metric viability, NeuTra quality, or predictive validity. |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | GPU visibility, XLA, memory growth, target binding, finiteness, host RSS, and allocator checks passed in the canaries; the prospective resource projection veto fired before tuning. |
| Statistically supported ranking | None. The continuous timing values are descriptive, although their near-exact inverse scaling is sufficient for the prospective cost stop. |
| Descriptive-only differences | One/two/four-worker warm rate, cold compile time, wall time, host RSS, and allocator peak. |
| Default-readiness | Not established. Serial remains the fixed-grid API default; this result does not change it. |
| Next evidence needed | A reviewed resource/algorithm redesign whose current-source worst-case projection fits its declared cap, then complete Round-0 tuning, followed by the separately gated HMC test. |

## Provenance And Source Boundary

The initial one-worker rate probe used source signature
`4afa97d8a263ec6c06d01600e17f8971cab9abe361f9240c05257cc1668cbf82`
and predates the worker-runner cache repair. It is retained as historical rate
context only.

The two- and four-worker topology canaries share the later exact execution
source signature
`ca66d7a7c124ac0a6657b9c55ac1ca540d3d4d2c7c5f5a190d47f012425fa079`.
They are the paired current-at-execution topology evidence used for the resource
decision. Both bind target signature
`302d50b16ac4804e1656527bbbfdb535ce46049536b3e7187fe5bd223e1cdb71`
and adapter signature
`0e873acd2543335eaf6e50c406b5e7995b073825011a8b5e006f9d71ed3d1925`.

Other agents changed transitive shared-worktree source after those canaries.
The completed receipts remain valid for their hashed bound source; they do not
authorize launching a grid against later source without a fresh rate/provenance
check. No concurrent agent file was edited or reverted by this lane.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `3250e0cb708eef7f8cbeafb62b2fd27741e3554f` with dirty shared worktree preserved |
| Interpreter/environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; conda env `tfgpu`; Python 3.13.13 |
| Device | Physical GPU 1, NVIDIA GeForce RTX 4080 SUPER; GPU 0 left untouched for other/desktop work |
| GPU policy | `CUDA_VISIBLE_DEVICES=1`; `TF_FORCE_GPU_ALLOW_GROWTH=true`; every worker verified TensorFlow memory growth before logical-device initialization |
| Backend | TensorFlow/TFP, GPU/XLA, `float64`; TF32 enabled and recorded but not used for FP64 tensor arithmetic |
| Seeds | Single probe `(20260720,9700)`; two-worker `(20260720,9710..9711)`; four-worker `(20260720,9710..9713)`; no sampler seed retained as posterior evidence |
| Trusted material wall | Single probe `160.8577 s`; two-worker `171.1488 s`; four-worker `198.8211 s`; cumulative `530.8277 s` |
| Host/device resource | Four-worker per-process host `ru_maxrss` about 4.47 GB; per-process TensorFlow allocator peak `381,568,512` bytes |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-process-grid-hmc-tuning-plan-2026-07-20.md` |
| Result | `docs/plans/bayesfilter-ssl-lstm-q20-process-grid-hmc-tuning-result-2026-07-20.md` |
| Contract artifact | `docs/plans/artifacts/ssl-lstm-q20-process-grid-hmc-tuning-2026-07-20/contract-smoke.json` |
| Historical single-rate artifact | `docs/plans/artifacts/ssl-lstm-q20-process-grid-hmc-tuning-2026-07-20/rate-probe.json` |
| Paired topology artifacts | `docs/plans/artifacts/ssl-lstm-q20-process-grid-hmc-tuning-2026-07-20/rate-topology-w2.json`; `docs/plans/artifacts/ssl-lstm-q20-process-grid-hmc-tuning-2026-07-20/rate-topology-w4.json` |

Exact material commands are recorded in each JSON `run_manifest.command`.

## Post-Run Red Team

The strongest alternative explanation is that process startup/XLA compilation,
rather than steady HMC execution, caused the failure. That does not explain the
warm result: after compilation, per-worker warm latency still doubled at two
workers and quadrupled at four, leaving aggregate throughput flat. The evidence
therefore supports GPU compute saturation for this bound implementation.

The result would be overturned by a current-source canary showing materially
higher aggregate warm throughput at multiple workers, or by a target/runtime
change that lowers the worst-case complete-grid projection beneath the declared
cap. The weakest part of the evidence is that each topology has only two warm
repeats; continuous timing is descriptive. The resource veto remains reasonable
because the no-extension four-worker projection (`11.98 h`) exceeds the
remaining cap (`7.85 h`) by more than 50%, while the all-extension projection
is `29.43 h`.

## Exact Next Step

Do not merely raise worker count on one GPU. Choose and review one of these
repairs before another run:

1. Map independent candidates to independent physical GPUs, one worker per GPU.
2. Reduce q=20 per-transition cost in the target/filter/HMC path and refresh the
   bound-source rate.
3. Redesign the grid/screen policy prospectively with statistical justification
   for fewer operations, without post-hoc threshold or evidence weakening.
4. Increase the owner-approved cap enough to cover the current-source worst
   case, explicitly acknowledging that one-GPU candidate processes do not
   reduce aggregate work.

Separately, NeuTra-HMC still requires target-specific q=20 Phase-3 training;
the scalar trained transports remain inadmissible for this target.

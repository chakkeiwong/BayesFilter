# SSL-LSTM q=20 Process-Grid HMC Tuning Plan

Date: 2026-07-20

Status: `CLOSED_RESOURCE_PROJECTION_STOP_BEFORE_TUNING`

## Scope Correction And Objective

Tune fixed-metric HMC for the controlled synthetic q=20 SSL-LSTM posterior
with the new spawned process-candidate grid, then run one short fixed-kernel
four-chain HMC mechanics test if at least one candidate survives.

This is a **plain-posterior HMC baseline**, not NeuTra-HMC. The q=20 artifact
inventory contains target, derivative, timing, and topology receipts but no
admitted q=20 frozen NeuTra payload. The only existing 5,000-step SSL-LSTM
payloads bind the earlier scalar target and are inadmissible here. The existing
transformed-HMC Phase-4 runner correctly requires two independently admitted,
target-matched Phase-3 transports. This run will not bypass that gate.

## Research Intent Ledger

| Role | Prospective definition |
| --- | --- |
| Main question | Can the q=20 plain SSL-LSTM posterior be tuned by the reviewed fixed-metric grid without changing candidate semantics, and does spawned candidate execution have an admissible one-GPU topology? |
| Candidate mechanism | Complete independent `L` candidates run in spawned processes, each with a q=20 GPU/XLA target, fixed identity metric, fixed-mass dual averaging, and fresh fixed-kernel screens. |
| Exact baseline | Serial execution of the same public fixed-metric candidate with identical target, starts, seeds, tuning policy, and screen policy. The legacy scale-first `(2,4,8,16)` runner is not the comparator. |
| Expected failure mode | Single-GPU process contention may remove speedup; identity-metric plain HMC may produce no survivors; worker bootstrap or memory-growth setup may fail. |
| Promotion criterion | Engineering parity on a bounded real-target candidate, followed by at least one complete-grid survivor under the predeclared acceptance policy. |
| Promotion veto | Target/signature mismatch, nonfinite target or telemetry, candidate-data invalidity, positive exposed native divergence, failed movement/path-return screen, or serial/process payload mismatch. |
| Continuation veto | Shared worker/bootstrap invalidity, memory growth not established before GPU initialization, target-health failure shared across candidates, host RSS above 64 GiB, or prospective resource cap exceeded. A no-survivor candidate result is not evidence against NeuTra. |
| Repair trigger | One-worker viability with two-worker contention triggers serial execution for correctness evidence only; no-survivor plain HMC triggers q=20 target-specific NeuTra training, not post-hoc grid or threshold changes. |
| Explanatory only | Wall time, allocator peak, acceptance means, leapfrog count, tuned step size, jump summaries, and worker-count timing. |
| Must not be concluded | No posterior correctness, convergence, model adequacy, process speedup in general, worker-count superiority, NeuTra quality, NeuTra-HMC failure, or production/default readiness. |

## Evidence Contract

| Item | Contract |
| --- | --- |
| Scientific/engineering question | Does the new process grid preserve real q=20 HMC candidate semantics, and can it yield a viable fixed kernel within the bounded resource envelope? |
| Comparator | Public serial `run_fixed_metric_candidate` for the same canary `L`, not the old local tuner. |
| Primary tuning pass | At least one survivor after the complete reviewed broad Round-0 grid. Midpoint refinement is disabled because it enriches an already viable set but is unnecessary for the non-ranking viability question. |
| Veto diagnostics | Exact target/start/metric/seed lineage; finite target, samples, target log probability, and log acceptance ratio; native divergence when exposed; acceptance-policy promotion vetoes; GPU memory growth; 64 GiB host-RAM ceiling; shared process invalidity. |
| Explanatory diagnostics | Per-stage wall time, TensorFlow allocator current/peak bytes, process count, tuned step, acceptance summaries, and descriptive movement. |
| Conditional HMC test | Select a deterministic representative survivor by smallest `L`, then tuned step size as a tie breaker. Run a fresh 64-draw, 64-burn-in, four-chain fixed-kernel test with a disjoint seed. Require finite values, all chains moving, no exposed positive divergence, and a promotion-eligible acceptance receipt. |
| Preserved artifact | `docs/plans/artifacts/ssl-lstm-q20-process-grid-hmc-tuning-2026-07-20/` plus the result note named below. |
| Nonclaim | The tuning draws are discarded. The short conditional test is mechanics/viability evidence, not retained posterior sampling or convergence evidence. |

## Frozen Mechanics

- Target: `complexity_posterior_target(q=20, jit_compile=True)`, four free
  coordinates, FP64 filter/score and HMC state.
- Coordinates: plain target coordinates; no learned or untrained transport.
- Metric: identity, recorded as a baseline hypothesis.
- Starts: the common prior center plus the four already reviewed relative
  offsets `(0,0,0,0)`, `(0.5,-0.5,0.5,-0.5)`,
  `(-0.5,0.5,-0.5,0.5)`, and `(0.5,0.5,-0.5,-0.5)`.
- Grid: reviewed Round-0 `L=(3,5,9,13,18,25)`. The existing midpoint
  refinement mechanism is disabled for this bounded viability question.
- Tune: each candidate independently uses 64 fixed-mass dual-averaging burn-in
  updates and one discarded post-adaptation result, target acceptance 0.70,
  and initial step 0.05. The last adapted step is extracted; all tune draws are
  discarded.
- Screens: three independent fixed-kernel replications of 64 draws with one
  initialization transition. The existing API may replace an inconclusive screen
  with one fresh 128-draw extension. Screens use `HMCAcceptancePolicy()`:
  target 0.70, practical region `[0.65,0.75]`, repair region `[0.55,0.85]`,
  four chains, four blocks, minimum block size 16, and the existing movement,
  recurrence, and native-divergence rules.
- Backend: TensorFlow/TFP GPU, XLA on, FP64, one physical GPU, TF32 setting
  recorded but immaterial to FP64 tensors.
- Processes: `spawn` only. Every child receives explicit
  `CUDA_VISIBLE_DEVICES` and `TF_FORCE_GPU_ALLOW_GROWTH=true`, sets and verifies
  memory growth before target construction, and records allocator telemetry.
  The parent validates those environment settings before importing BayesFilter;
  it may import the TensorFlow package through BayesFilter's package initializer
  but must not initialize a GPU device or construct the q=20 target.

## Resource Ladder And Stops

The prospective cumulative ceiling is eight trusted GPU-hours for this
bounded plain-HMC tuning/test lane. This is a convenience ceiling inherited
from the owner's prior bounded HMC materiality threshold, not a claim that
eight hours is statistically sufficient. Sequential stopping returns unused
time.

1. CPU-hidden contract and focused tests; no scientific result.
2. Current-source q=20 GPU/XLA value/score and one-transition rate canary.
3. Generic process-grid serial/process payload equality remains covered by the
   focused spawn tests. Run a real q=20 concurrent rate canary with two workers,
   and a four-worker canary only if memory growth, target signatures, finiteness,
   and aggregate resource evidence pass. Each worker compiles the current q=20
   HMC graph and performs two warm calls; no grid survivor can be created.
4. If projected complete-grid work plus a 50% margin does not fit the remaining
   ceiling, record `RESOURCE_PROJECTION_STOP` and do not launch the grid.
5. The current-source contended rate decides topology feasibility. If the
   all-extension Round-0 projection using the measured four-worker maximum warm
   rate and cold compile time cannot fit under the remaining cap, stop. One- and
   two-worker full-screen topologies are resource-infeasible, not performance-
   rejected. Four workers remain a budget-feasibility hypothesis rather than a
   promoted topology.
6. Run the complete Round-0 process grid with four workers. A barrier already
   in flight is non-preemptive. Midpoint refinement is disabled prospectively;
   it is unnecessary for survivor viability and cannot support ranking under
   the current evidence.
7. Only after a survivor exists, run the conditional fixed-kernel HMC test.

Timeout, worker crash, missing artifact, source drift during execution, or
aggregate RSS above 64 GiB closes the run without converting partial evidence
into a survivor. Do not lower screen lengths, alter acceptance regions, remove
large `L` values, or reuse tuning traces after observing results.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode | Earliest diagnostic |
| --- | --- | --- | --- | --- |
| q=20 | User-selected target rung; reviewed experiment target | Current scalability question | Synthetic four-coordinate target may not represent broader q=20 estimation | Target signature and explicit nonclaim |
| Plain target | Necessary baseline; not NeuTra substitute | No q=20 admitted transport exists | Could fail from geometry NeuTra is intended to repair | Classify no-survivor as plain-HMC failure only |
| FP64 | Measured reviewed default | Mixed FP32 was not faster; all-FP32 changed numerical behavior | Slower than lower precision | Current-source rate canary |
| Identity metric | Baseline hypothesis | Exact fixed-metric comparator with no unadmitted learned object | Poor geometry can reject every candidate | Broad `L` grid and per-candidate tune result |
| `L` grid | Reviewed API default for Round 0 | Broad independent trajectory coverage | Too expensive or still misses useful values | Prospective cost projection; no midpoint refinement is launched in this lane |
| 0.70 acceptance | `HMCAcceptancePolicy` reviewed default | Owner-confirmed HMC target in prior discussion | Short screens may remain inconclusive | Three 64-draw screens and existing 128-draw extension |
| 64 tune updates plus one post-adaptation result | Reviewed warm-start hypothesis for this bounded run | TFP completes all dual-averaging updates during burn-in; the post-adaptation result exposes the frozen step without extra unused draws | Step may not stabilize or may differ by chain | Finite common final step and independent screen replications; no convergence claim |
| Three 64/128 screen replications; one initialization transition | Replication count and retained sizes are current reviewed grid defaults; one transition is a bounded mechanics-screen choice | Preserves independent screen replication and the minimum four-by-16 acceptance evidence without pretending the short path is equilibrium sampling | Initial-state dependence or low power can make the screen unstable | Three fresh replications, extension path, per-block evidence, and explicit no-convergence claim |
| Four candidate workers after concurrent rate canaries | Budget-feasibility hypothesis selected only if measured contended projection fits | One/two-worker worst-case Round 0 cannot fit; measured per-worker allocator/RSS permits a bounded four-worker probe | Contention can erase projected overlap or aggregate memory can exceed the cap | Generic serial/process equality tests, real two/four-worker q=20 rate receipts, command timeout, allocator/RSS telemetry |
| Eight GPU-hours | Inherited bounded-HMC ceiling, convenience cost stop | Prevents an open-ended grid | May be insufficient for q=20 | Current-source rate projection before grid |
| Smallest-`L` representative | Deterministic non-ranking handoff | Avoids descriptive stochastic ranking | May not be most efficient | Explicit representative/non-best label |

## Pre-Mortem

- The command could succeed while evaluating the wrong target. Bind target,
  adapter, common starts, and metric content signatures in every callback.
- Process execution could appear faster because it changed seeds or omitted a
  screen. Compare complete canary candidate payloads, not only timing.
- Multiple GPU children could initialize correctly but contend badly. The
  four-worker barrier has a command timeout and resource telemetry; do not
  interpret completion time as a general topology ranking.
- Dual averaging could return a finite but poor step. Only independent fixed-
  kernel screens can create a survivor.
- Short screens could pass while chains have not converged. Preserve the
  mechanics-only nonclaim and do not retain their samples.
- Plain HMC could fail for the geometry NeuTra is meant to repair. Treat that
  as the stated repair trigger, not as rejection of SSL-LSTM or NeuTra.

## Skeptical Pre-Execution Audit

- **Wrong baseline:** repaired. The comparator is the same public fixed-metric
  candidate in serial mode, not the legacy scale-first tuner.
- **Proxy promotion:** timing and acceptance means alone are explanatory;
  survivor status requires the complete policy receipt.
- **Missing stop:** current-source projection, eight-hour cumulative ceiling,
  memory growth, RSS, source drift, and process invalidity all stop launch.
- **Unfair comparison:** target, starts, metric, seeds, tune, and screen policy
  are content-bound and identical between serial and spawned canaries.
- **Hidden assumptions:** plain-target scope, identity metric, tune length,
  screen power, and worker count are recorded above as hypotheses.
- **Stale context:** the plan uses the new broad process-grid API and current
  matrix-free q=20 source, not stale rate or legacy tuner results.
- **Environment mismatch:** every GPU process must establish on-demand memory
  growth before logical device initialization and use XLA.
- **Artifact insufficiency:** structured receipts preserve private candidate
  mechanics, public summary, exact manifest, resource projection, and the
  conditional test separately.

Audit decision: `PASS_FOR_BOUNDED_EXECUTION`. The plan can answer process
semantic correctness and bounded plain-HMC viability. It cannot answer q=20
NeuTra-HMC viability; that remains gated by target-specific Phase-3 training.

## Planned Files And Commands

Implementation files:

- `bayesfilter/testing/ssl_lstm_q20_fixed_metric_worker.py`
- `docs/benchmarks/run_ssl_lstm_q20_process_grid_hmc_tuning_2026_07_20.py`
- `tests/test_ssl_lstm_q20_process_grid_hmc_tuning.py`

Focused CPU-hidden verification will use the `tfgpu` Python executable with
`CUDA_VISIBLE_DEVICES=-1`. Trusted GPU commands will set
`TF_FORCE_GPU_ALLOW_GROWTH=true`, select an unoccupied GPU (GPU 1 preferred,
GPU 0 fallback), and write only beneath:

`docs/plans/artifacts/ssl-lstm-q20-process-grid-hmc-tuning-2026-07-20/`.

## Result Contract

Result note:

`docs/plans/bayesfilter-ssl-lstm-q20-process-grid-hmc-tuning-result-2026-07-20.md`

It must contain a decision table, engineering/numerical/scientific ledgers,
inference-status table, exact run manifest, candidate rejection versus research
direction classification, and a post-run red team. If projection or another
continuation veto stops execution, it must say what prerequisite or budget
would be needed next rather than labeling the algorithm unsuccessful.

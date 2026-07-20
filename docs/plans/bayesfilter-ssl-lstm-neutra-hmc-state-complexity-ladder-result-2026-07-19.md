# SSL-LSTM NeuTra-HMC State-Complexity Ladder Result

Date: 2026-07-19  
Decision: `PHASE_2_COMPLETE_BATCHED_XLA_HMC_RETAINED_MATERIAL_LADDER_NOT_RUN`

## Scope

The executed work repaired the dimension-general four-coordinate target and
selected-direction analytic score, then ran trusted TensorFlow/XLA target
preflights and full NeuTra/HMC mechanics canaries for `q=1,2,5,10,20`, then
compared batched-chain XLA against the actual DSGE-HMC independent scalar-chain
threaded-XLA production topology at q=1 and q=20. The 64 GiB host-RAM ceiling
was applied. No material NeuTra training,
Optuna study, HMC tuning, retained posterior acquisition, or predictive
validation was executed.

The later Phase-2 repair replaced the serialized batch evaluator with the
DSGE-HMC-style persistent CPU-worker/custom-gradient boundary. The target,
estimand, prior, observations, filter equations, and NeuTra family did not
change during those timing runs. A subsequent q=5 canary exposed a target
factorization inconsistency; the current target now uses principal-square-root
value and score consistently. All final timing receipts were then replayed under
the principal-root target signature.

## Target And Score Result

The new target estimates the same four homologous coordinates at every rung:
`latent_mean_weight.0.0`, `latent_mean_bias.0`,
`observation_weight.0.0`, and `observation_bias.0`. The remaining chart
coordinates are fixed by a deterministic rung fixture. This avoids pretending
that 3,862 q=20 parameters can be estimated from 30 scalar observations.

The selected score path constructs derivative surfaces with leading dimension
four instead of the full `9q^2+13q+2` chart dimension. Focused checks passed:

- existing full-route and structural adapter tests: `19 passed`;
- selected-target focused suite: `7 passed, 5 deselected`;
- q=1 sensitivity to the locked scalar target: maximum fixed-point absolute
  log-density difference `9.45e-6` and score-coordinate difference `9.26e-5`;
- q=2 selected score versus finite differences: passed;
- q=1,2,5,10,20 directional derivative surfaces have leading dimension 4;
- deterministic fixture and synthetic observation replay: passed;
- isolated q=20 target/XLA preflight: finite repeated score, four-coordinate
  score, host RSS `3.03 GiB`, GPU allocator peak `0.31 GiB`.

The q-ladder target receipt is
`docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/phase-1-2/target-preflight.json`.
The isolated q=20 receipt is
`docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/phase-1-2/q20-isolated-target-preflight.json`.

## Q=5 Target Failure And Repair

The first 16-worker q=5 canary stopped on the first stream before any material
claim. At deterministic batch row 9, theta
`[1.7911783478, 2.1641609074, -0.4040704194, -0.1681555034]`, the historical
eigenvector derivative found a minimum placement spectral gap `3.17e-11`,
below its hard `1e-10` threshold. This was not a memory, GPU, process-pool,
architecture, or tuning failure.

Root cause: the score-bearing path differentiated an eigenvector-oriented SVD
factor, while value-only validation used the principal square root. These are
different nonlinear sigma-point approximations, and only the former requires a
simple eigenvalue spectrum. Lowering the threshold would not repair the target
definition.

The repair uses `tf_principal_sqrt_ukf` for values and
`tf_principal_sqrt_ukf_score` for analytic scores, differentiating the unique
strict-SPD square root through the Sylvester equation. The target signature now
binds this factorization. Focused CPU tests passed:

- principal analytic score versus principal finite differences at q=2;
- the exact q=5 failure point, including four-coordinate finite differences;
- selected versus full principal-root score directions;
- 7 target tests total;
- 6 process-boundary tests, including worker/native parity and exact
  identical-state native/external trainer updates.

The failure receipt is
`docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/process-parallel/canary-q5-w16-eigenderivative-failure.json`.
Because the target signature changed, Phase-2 canary replay restarts at q=1.

## Superseded Serial Canary

Each canary used the real 32x32 three-stage dense-IAF trainer, two independent
seeds, batch size 480, 10 training steps per seed, frozen transport reload,
and a two-draw transformed-HMC mechanics call. These are mechanics/timing
artifacts only.

| q | Canary wall | Host RSS | Warm NeuTra step | HMC samples | Status |
|---:|---:|---:|---:|---|---|
| 1 | 510.17 s | 8.48 GiB | 6.79 s | finite `[2,4,4]` | mechanics passed; resource projection veto |
| 2 | 630.78 s | 8.72 GiB | 12.06 s | finite `[2,4,4]` | mechanics passed; resource projection veto |

Receipts:

- `docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/phase-1-2/canary-q1.json`;
- `docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/phase-1-2/canary-q2.json`.

The q=1 two-seed 5,000-step training projection is `74,271 s` before the
declared 50% margin and `111,407 s` with margin, approximately `30.9 GPU-hours`.
The q=2 projection is `123,436 s` before margin and `185,155 s` with margin,
approximately `51.4 GPU-hours`. Neither projection includes Optuna trials,
HMC tuning/confirmation, four-chain retained draws, or predictive validation.

The q=1/q=2 serial canary execution cost was `1,140.95 s` wall in fresh trusted GPU
processes. The q=5, q=10, and q=20 target preflights were already complete, but
their material canaries were correctly skipped after the q=1 resource gate.

These projections are superseded because `tf.map_fn` serialized 480 independent
30-step nonlinear filter/score evaluations. They no longer support a resource
stop for the repaired execution topology.

## Process-Parallel Repair

The repaired route uses a persistent `spawn` pool. Every child inherits
`CUDA_VISIBLE_DEVICES=-1` and one CPU thread before importing TensorFlow, warms
one non-XLA scalar value/analytic-score graph at the valid prior center, and
returns ordered values and scores. The parent retains the dense-IAF transport,
`tf.custom_gradient` score bridge, Adam update, and HMC on physical GPU 1. GPU
0 is the automatic fallback when GPU 1 is absent.

Correctness checks passed:

- worker scalar values/scores match the native target;
- the external-score bridge produces the same loss, gradient norm, and updated
  transport parameters as the native trainer from identical state;
- value-only validation matches the score-bearing target value without
  derivative propagation;
- worker receipts prove CPU-only visibility, request/full-batch/shard identity,
  contiguous coverage, finite shapes, and worker PIDs;
- one readiness task per configured process holds at a startup barrier until
  all workers have completed target construction and warmup; startup PIDs are
  recorded separately from the workers active on a particular request;
- a first request with two rows and four configured workers completes with two
  active shards after all four startup PIDs are verified.

The startup barrier repaired a real timing bug. Before the barrier, q=2 stream
0 was served by only 5--7 live workers while the rest of the pool was still
initializing, yielding 1.4--1.8 s steps and a false 7.46-hour projection. After
the barrier, both streams used all 32 workers from their first batch.

### Post-Canary API Hardening

After the timing canaries, the barrier was moved from worker initialization to
an explicit one-task-per-worker readiness round. This closes a generic API edge
case: `ProcessPoolExecutor` starts workers lazily, so a first request with fewer
rows than the configured worker count could previously leave the barrier
waiting for processes that had no submitted task. The serious batch-480 route
already submitted at least 32 shards and was not affected.

CPU-only regression commands were run with GPU devices intentionally hidden:

```text
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_ssl_lstm_process_parallel.py
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_ssl_lstm_complexity_target_tf.py tests/test_ssl_lstm_complexity_ladder.py tests/test_neutra_reverse_kl_training.py tests/test_neutra_training_control.py
```

Results: `6 passed` in `264.77 s` and `43 passed` in `168.58 s`. The first
suite includes worker/native parity, value-only parity, full-pool readiness,
the 2-row/4-worker edge case, and exact identical-state native/external trainer
updates. `git diff --check` and Python compilation of the modified pool,
target, runners, and test module also passed.

Source state for these post-canary checks:

| Field | Value |
| --- | --- |
| Git commit | `3250e0cb708eef7f8cbeafb62b2fd27741e3554f` with the documented working-tree changes |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`, deliberate CPU-only `CUDA_VISIBLE_DEVICES=-1` |
| Pool SHA-256 | `1c7ea099dccc2566609b4daa38dbc85f5d2f6bec106b4d068fa243dff9dfa8f4` |
| Target SHA-256 | `80da2f1c18a622c5cda43c681fca68f1256bec70c6f02ccf886a341c9b1ba72f` |
| Focused test SHA-256 | `021d467d0de3e6951eee2cd849fd387ea00784d5fcefd747b403f514b1749237` |
| Result artifact | This result note |

The q=1/q=2 timing receipts retain pool hash
`99749085b16ce493f0bda535bb6bb143d516f838884aad15fe565094c1182fa4`.
They were not regenerated after the generic small-first-batch hardening, so
their timings apply to the earlier barrier placement. Because the material
batch size is 480 with 32 workers, the repair does not invalidate those timing
observations, but the old receipts must not be represented as executions of the
new pool hash.

## Final Training Topology

Worker-count timing is descriptive execution evidence, not a statistical
ranking of scientific methods. Every selected topology passed finite mechanics,
CPU-only worker visibility, startup PID, ordered-shard, and 64 GiB host checks.

| q | Selected topology | Warm target/update max | Conservative host bytes | Selection reason |
|---:|---:|---:|---:|---|
| 1 | 32x1 | 0.465 s | 64.43 GB | faster than 16x1 and below cap |
| 2 | 32x1 | 0.436 s | 65.79 GB | faster than 16x1 and below cap |
| 5 | 32x1 | 0.491 s | 62.04 GB | faster than passing 16x1 canary |
| 10 | 32x1 | 1.044 s | 61.56 GB | faster than passing 16x1 canary |
| 20 | 16x1 | 12.059 s | 36.25 GB | 32x1 was only 0.3% faster; select smaller topology within 10% |

At q=20, 32x1 took 12.023 seconds with 61.91 GB conservative host memory,
while 16x2 took 35.336 seconds. The selected topology therefore remains 16
single-thread workers. The principal-root receipts are under
`docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/process-parallel/`.

The earlier `119.59`-hour warm-rate total omitted explicit fresh pool and graph
startup. The corrected receipt retains 19,800 prospective training steps per
rung and adds three full-canary launch reserves plus nine second-trainer compile
excesses per rung, all under the same 50% margin. The resulting Phase 3 cap is
`121.2874392020` hours. Sequential pruning and plateau stopping may reduce
this; the estimate does not authorize execution.

The audit also found that the allowed `seed-c` confirmation was declared and
budgeted but not reachable from the CLI. Phase 3 now has a separate
`confirmation` mode. It requires a completed final summary with exactly one
`ADMITTED` and one `VETOED` stream, the matching failed-result path/hash, the
same nominated parameters, and the fixed fresh seed. A resource-stopped
confirmation may resume; a host-memory hard veto may not.

## HMC Topology Transfer

Direct source inspection corrected the informal premise. DSGE-HMC does not use
an OS process pool for HMC target calls. It compiles one scalar-chain graph per
production chain and schedules the graphs through `ThreadPoolExecutor`; warmup
remains TensorFlow-batched. BayesFilter implemented that exact production
boundary as a diagnostic runner and compared it with the existing one-call
four-chain XLA runner.

| q | Batched XLA median | Scalar serial median | Scalar threaded median | Decision |
|---:|---:|---:|---:|---|
| 1 | 0.591 s | 0.682 s | 1.138 s | retain batched; threaded much slower |
| 20 | 14.202 s | 14.200 s | 13.975 s | retain batched; 1.6% difference misses 10% rule |

All samples and traces were finite, every chain moved, shapes were `[4,4,4]`,
and scalar serial/threaded samples and traces were exactly equal for identical
folded seeds. Native divergence telemetry was not exposed by this TFP kernel and
was recorded as unavailable, not zero. First-call setup was 431.32 seconds at
q=1 and 515.20 seconds at q=20 because the candidate compiled four scalar graphs
in addition to the batched baseline. The q=1/q=20 receipts are
`hmc-topology/hmc-topology-q1.json` and `hmc-topology/hmc-topology-q20.json`,
with SHA-256 `4512ba65...f72f` and `e09154aa...f30b` respectively.

The tiny benchmark used two leapfrog steps. Its two-transport, all-rung Phase
4--5 reserve is 19.67 `L=2`-equivalent hours with 50% margin under four tuning
probes, one fresh confirmation, and 4,096 retained draws per chain. Actual HMC
cost scales approximately as `L/2`, and the tuned leapfrog count is unknown.
This is therefore a normalization, not a frozen budget or convergence forecast.

## Decision Table

| Decision | Primary criterion | Veto status | Next justified action | Not concluded |
| --- | --- | --- | --- | --- |
| Admit target implementation for further planning | q=1 parity, q=2 finite differences, all-rung four-direction shapes, q=20 isolated memory/finite score | No target or host-RAM veto | Retain the generalized target and directional-score repair | Posterior correctness or HMC convergence |
| Select training topology | All-rung topology/canary mechanics and memory checks | No selected-topology veto | Use 32x1 for q=1/2/5/10 and 16x1 for q=20 | Training quality or universal topology superiority |
| Select HMC topology | q=20 threaded scalar chains did not beat batched XLA by 10% | No mechanics veto; candidate misses timing criterion | Retain batched four-chain XLA | Convergence, posterior validity, or general DSGE-HMC inefficacy |
| Hold material ladder pending authority | Complete receipt replay freezes a 544.0235 h sequential cap | No receipt/source/operation-count/resource veto; material authority absent | Request separate explicit launch authority | Scientific validity, convergence, or expected runtime |

## Phase 3 Runner Readiness

The q-general Phase 3 harness now exists at
`docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py`.
It is engineering-ready but has not run a study or final stream. Its contract
smoke starts no workers, evaluates no target, and trains no transport.

The focused audit found and repaired: GPU selection after TensorFlow import,
missing support checks at plateau checkpoints, LR repair without best-state
restore, non-resumable final streams, Optuna resume adding excess trials,
in-memory-only trial evidence, repeated embedded state/transport payloads, and
memory breaches being misclassified as resumable resource stops. The final
contract restores best trainer/Adam state before halving LR, preserves
controller patience, persists joint resume checkpoints, externalizes large
payloads, writes per-trial records, and treats host RSS above 64 GiB as a
continuation veto.

Verification:

- q-general runner plus plateau-controller tests: `18 passed`;
- full HMC runner module: `29 passed`;
- target/NeuTra/process-boundary regression group: `50 passed`;
- Python compilation and `git diff --check`: passed.

These are engineering checks only. They do not nominate hyperparameters,
establish transport quality, or authorize either the superseded 119.59-hour
warm-only estimate or the final 121.2874-hour Phase 3 cap.

## Inference Status

| Row | Status |
| --- | --- |
| Hard veto screen | Target, worker bridge, value-only validation, selected memory, HMC mechanics, forecast replay, source/receipt identity, and budget arithmetic passed; 64/96-worker training arms resource-vetoed |
| Statistically supported ranking | None |
| Descriptive-only differences | Worker/HMC topology wall times, RSS, GPU telemetry, acceptance, and training-loss rows |
| Default readiness | Not established |
| Next evidence needed | Separate material authority, then sequential Optuna nomination, two 5,000-step streams, admitted HMC, calibration, and predictive-moment validation |

## Phase 4 Runner Readiness

The q-general transformed-HMC preflight/tuning harness now exists at
`docs/benchmarks/run_ssl_lstm_neutra_complexity_hmc_tuning_2026_07_19.py`.
It is engineering-ready but has not loaded a trained transport or run HMC. Its
contract smoke hides GPUs and performs no target evaluation.

The focused skeptical audit found and repaired five material risks: direct
payload input could bypass Phase 3 admission; resume receipts were not bound to
the frozen transport or execution source; movement/RMS reference values had
silently become hard vetoes; a coarse scale grid could reject a bracketed
acceptance solution; and a fixed 60-second reserve did not scale with q,
transition count, or leapfrog count. Phase 4 now requires two distinct admitted
Phase 3 result receipts, verifies payload hashes and serialized replay, binds
every arm receipt to the exact experiment contract, retains jump diagnostics as
explanatory, permits one geometric-midpoint bracket repair, and scales later
arm reserves from observed seconds per transition-leapfrog with 50% margin.

Verification:

- Phase 3/4 focused runner tests: `17 passed`;
- Phase 4 contract smoke at q=20: passed with GPU intentionally hidden;
- Python compilation and `git diff --check`: passed.

These checks establish runner mechanics only. They do not admit a NeuTra
transport, freeze an HMC kernel, establish convergence, or authorize material
execution. The next engineering artifact is the q-general Phase 5 retained
four-chain runner.

## Phase 5 Runner Readiness

The q-general immutable retained-HMC harness now exists at
`docs/benchmarks/run_ssl_lstm_neutra_complexity_retained_hmc_2026_07_19.py`.
It requires a `KERNELS_FROZEN` Phase 4 summary, preserves 256-draw private
archives and exact segment lineage, evaluates frozen checkpoints at
512/1024/2048/4096 draws per chain, and applies the plan's R-hat, bulk/tail ESS,
MCSE/SD, finite-value, movement, and exposed-divergence gates in both `z` and
`theta`. Cross-chart means and ten raw second moments use the frozen combined
MCSE screen. Acceptance and continuous jump/runtime differences are
descriptive after the Phase 4 kernel confirmation.

The audit repaired first-segment budget underestimation, q=20 derivative
materialization in post-archive validation, stale resume lineage, and source
drift between tuning and acquisition. The first retained reserve now inherits
actual Phase 4 HMC rates; complete segment-plus-audit rates update later
reserves; audits run in fixed four-point XLA batches; and every resumed archive
checks Phase 4 identity, executable-source identity, prior manifest, prior
final-state hash, finite telemetry, and divergence status.

Verification:

- Phase 5 focused tests: `9 passed`;
- combined Phase 3--5 runner/archive/diagnostics suite: `39 passed`;
- Phase 5 q=20 contract smoke: passed with GPU intentionally hidden;
- Python compilation and `git diff --check`: passed.

No retained HMC ran. A subsequent cross-phase audit found that the locked
predictive procedure requires 12,288 draws per chain while Phase 5 admits at a
maximum of 4,096. The live plan now keeps 4,096 as the sampler-admission cap and
allows only already-admitted fixed kernels to extend to 12,288 in Phase 6.

## Phase 6 Runner Readiness

The q-general recovery/predictive runner now exists at
`docs/benchmarks/run_ssl_lstm_neutra_complexity_predictive_validation_2026_07_19.py`,
with supporting code in
`bayesfilter/nonlinear/ssl_lstm_complexity_predictive_tf.py` and
`bayesfilter/inference/cpu_forecast_pool.py`. Calibration is a separately
resumable CPU-hidden truth-fixture lane. Validation extends only Phase 5-admitted
fixed kernels to 12,288 draws per chain on GPU/XLA, renews z/theta sampler
diagnostics, maps retained draws in bounded chunks, and evaluates forecasts in
persistent spawned CPU workers. It applies the locked Rao-Blackwell conditional
mean/log-variance influence method, Bartlett HAC multiplier 3.0, zero ridge,
split alpha allocation, and frozen acceptable loss. It writes replay-bound
forecast blocks plus JSON/PNG/PDF simulated-path artifacts.

The skeptical audit repaired dynamic Phase 5 admission counts, Phase 5 archive
replay, extension source/kernel/seed identity, calibration bank replay,
resumable calibration checkpoints, renewed chain movement, aggregate
parent-plus-worker memory enforcement, disjoint seed domains, forecast-block
theta/calibration/source binding, inadmissible-HAC handling, and a misleading
conditional-mean plot label. Calibration does not access retained A/B values.

Verification:

- Phase 6 focused contract/statistical tests: `11 passed`;
- real spawned CPU forecast pool parity/barrier tests: `2 passed`;
- combined affected Phase 3--6 suite: `55 passed`;
- q=1 and q=20 Phase 6 contract smokes: passed with GPU intentionally hidden;
- Python compilation and `git diff --check`: passed.

No calibration bank, HMC extension, retained forecast bank, or predictive
decision ran. The production-block CPU forecast timing canaries and numerical
Phase 3--6 budget freeze are now complete.

The first timing attempt passed q=1 but failed q=2 before emitting a receipt:
every worker rejected the deterministic warmup forecast. A focused diagnostic
showed that the q=2 terminal covariance was finite and strict SPD, and the
forecast dynamics were finite eagerly. The failure was CPU XLA eigensolver
reconstruction error in the terminal Gaussian sampling factor, not invalid
target/filter geometry. Cholesky represented the same covariance law with
relative reconstruction error `2.12e-21` at q=2 and at most `1.58e-16` across
q=1,2,5,10,20. The q=1 timing receipt is therefore superseded by source drift,
and the ordered canary was restarted after the focused repair tests. All five
repaired receipts subsequently passed.

## Final Budget Freeze

The five repaired CPU forecast receipts bind source signature
`c277d55333f93fa5906242366664db5be0998422ff7726212adf3dbd6bf68f55`.
The five current-source HMC receipts bind signature
`12485b7728c16f6333047ed3661989a79fd7a54c4dec086f6479af2b3b2f489a`.
The replay harness also verifies the current target, CPU value/score pool, and
trainer hashes against the selected Phase 3 timing receipts.

| q | Phase 3 h | HMC h | CPU forecast h | Sequential total h |
|---:|---:|---:|---:|---:|
| 1 | 4.1338 | 12.3418 | 0.0707 | 16.5463 |
| 2 | 3.8609 | 18.3834 | 0.0783 | 22.3227 |
| 5 | 4.3261 | 48.4203 | 0.0811 | 52.8275 |
| 10 | 8.9348 | 85.8525 | 0.1151 | 94.9024 |
| 20 | 100.0319 | 256.5852 | 0.8076 | 357.4247 |
| **Total** | **121.2874** | **421.5832** | **1.1529** | **544.0235** |

The GPU-active subtotal is `542.8706291745` hours. CPU-only forecasts add
`1.1528584148` hours, producing a sequential wall cap of
`544.0234875893` hours. The q=5/q=10 startup reserve deliberately uses the
slower 16-worker two-stream full-canary wall while warm work uses the selected
32-worker rate; this is conservative. The HMC cap uses the maximum `L=16`
operation count and a 9,000-second cold reserve per rung. Sequential stopping
returns unused budget when pruning, plateau stopping, smaller selected `L`,
early sampler admission, or a veto ends work; it never authorizes exceeding the
cap.

Artifact:
`docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/budget-freeze/budget-freeze-phase3-6.json`.
Close SHA-256: `5cc96c62f8fb1fc8aaf7d14e28f19257610c6523d2c31f24947d528eedf07172`.
The CPU-only replay command was
`python docs/benchmarks/freeze_ssl_lstm_neutra_complexity_budget_2026_07_19.py --output docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/budget-freeze/budget-freeze-phase3-6.json`
in conda environment `tfgpu`, Python `3.13.13`, git commit
`3250e0cb708eef7f8cbeafb62b2fd27741e3554f` with a dirty shared worktree.
It initialized no GPU, used no random seeds, and completed receipt parsing and
SHA-256 replay in `0.0511` seconds.

Final engineering verification after the confirmation and freeze repairs:

- affected Phase 3--6 suite: `53 passed` in `182.81 s` with GPUs intentionally hidden;
- focused confirmation/freeze suite: `10 passed`;
- q=20 Phase 3 and Phase 6 CPU-hidden contract smokes: passed;
- Python compilation and `git diff --check`: passed.

These checks support engineering readiness and budget identity only. No
stochastic candidate was ranked, no default was promoted, and no scientific
claim was tested.

## Why Material Execution Has Not Started

The engineering and budget questions are resolved. Material execution has not
started because the frozen receipt is explicitly non-authorizing and the
maximum sequential cap is large: `544.0235` hours. A separate explicit launch
decision is required before Optuna, 5,000-step training, HMC, calibration, or
predictive validation.

## Nonclaims And Red Team

No posterior oracle was used. The canaries do not establish NeuTra quality,
HMC convergence, posterior correctness, model adequacy, full-parameter
estimability, predictive validity, or superiority over plain HMC. The strongest
alternative explanation for the observed timing differences is shared-host CPU
scheduling; therefore worker-count differences remain descriptive. The result
would be overturned if a source-consistent replay failed worker/native parity,
the startup barrier failed to activate all workers, or a full run exceeded the
64 GiB cap. For HMC topology, the strongest alternative explanation is that
longer trajectories or another GPU runtime could change relative scheduling
cost; this result applies to the measured TensorFlow 2.20 single-GPU route. The
weakest evidence remains extrapolation from short canaries to 5,000 training
steps and from tiny HMC mechanics to production transition counts. The cap
handles this conservatively through 50% margins, explicit Phase 3 startup
reserves, the allowed `L=16` maximum, 9,000-second HMC cold reserves, and
sequential stopping. These controls bound authority; they do not turn timing
evidence into a convergence or scientific-validity claim.

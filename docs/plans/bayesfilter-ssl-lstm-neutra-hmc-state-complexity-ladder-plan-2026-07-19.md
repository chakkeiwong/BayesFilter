# SSL-LSTM NeuTra-HMC State-Complexity Ladder Plan

Date: 2026-07-19  
Status: `PHASE_6_RUNNER_ENGINEERING_READY_FORECAST_TIMING_CANARY_PENDING_MATERIAL_LADDER_NOT_RUN`
Tier: Tier 3 bounded GPU/XLA research experiment

## Question And Estimand

Does the same four-coordinate Bayesian SSL-LSTM estimand remain numerically
valid and yield converged NeuTra-transformed HMC as filtering-state complexity
increases through `q in {1,2,5,10,20}`?

For rung `q`, use `latent_dim=hidden_dim=q`, scalar observation dimension,
augmented filtering-state dimension `3q`, and the general SSL-LSTM parameter
chart of size `9q^2+13q+2`. Estimate exactly these homologous coordinates:

1. `latent_mean_weight.0.0`;
2. `latent_mean_bias.0`;
3. `observation_weight.0.0`; and
4. `observation_bias.0`.

All other coordinates are fixed by a deterministic, hashed rung fixture. This
is a controlled **state/filter-complexity ladder**, not a full-parameter
dimension ladder. Estimating all 3,862 possible q=20 coefficients from 30
scalar observations would be underidentified and would not answer the stated
question. A full-parameter ladder requires a separate data-size and
identifiability design.

Every rung, including q=1, uses the strict-SPD principal-square-root UKF value
and its analytic Sylvester-equation score. The existing locked q=1
eigenvector-oriented SVD-UKF target remains a historical sensitivity comparator,
not the ladder target, because the two square-root choices define distinct
nonlinear sigma-point approximations. The q>1 rungs use deterministic synthetic
observations generated prospectively from their fixed full fixture at the same
four-coordinate truth. Synthetic data identity, truth, prior, horizon,
free-coordinate mask, and square-root backend are frozen before training or HMC.

## Research Intent Ledger

| Field | Frozen contract |
| --- | --- |
| Main question | Does NeuTra-HMC convergence survive increasing nonlinear filtering-state complexity for one controlled four-coordinate estimand? |
| Exact baseline | The q=1 four-coordinate principal-square-root UKF target and its 32x32 three-stage dense-IAF NeuTra procedure. The locked q=1 eigensquare-root target is a sensitivity comparator; the original serial `tf.map_fn` evaluator is only a historical timing comparator. |
| Candidate mechanism | Persistent spawn-based CPU workers evaluate independent scalar filter value/score rows; the selected GPU retains the 32x32 three-stage dense-IAF transport, custom-gradient bridge, optimizer, and later HMC. |
| Expected failure modes | Wrong generalized target, full-chart derivative materialization, nonfinite score, XLA/device failure, NeuTra saturation or seed instability, poor transformed geometry, HMC divergence, nonconvergence, or forecast-moment instability. |
| Primary promotion criterion | A rung passes target validity, two-seed NeuTra admissibility, transformed-target validity, and four-chain retained HMC diagnostics, then passes the declared synthetic recovery/predictive checks. |
| Promotion veto | Any required target/transport/sampler/predictive gate fails after its prospectively allowed repair. |
| Continuation veto | Invalid target math or fixture, worker/native value-score mismatch, custom-gradient update mismatch, corrupted artifact, unavailable trusted GPU 1 and GPU 0, host RSS above 64 GiB, GPU allocation failure, source drift during a rung, or exhausted declared wall/GPU budget. |
| Repair trigger | A candidate failure triggers the smallest declared repair: directional-score correction, one Optuna/plateau repair, HMC retuning, or additional retained segments. It does not reject the research direction. |
| Explanatory only | Training loss, runtime, RSS below cap, GPU allocator peak, acceptance within a viable band, jump size, and continuous cross-seed differences without uncertainty support. |
| Must not conclude | Posterior truth from another sampler, complete mode/tail coverage, full-parameter SSL-LSTM estimability, superiority of NeuTra, production readiness, or model adequacy outside the synthetic design. |

## Evidence Contract

| Role | Requirement |
| --- | --- |
| Scientific question | Whether one fixed four-coordinate Bayesian problem continues to admit valid NeuTra-HMC sampling as the latent/filter state grows. |
| Comparator | q=1 principal-root ladder target; all rungs share horizon 30, scalar observations, prior SD 4, free-coordinate names/order, dtype, principal-root filter/score backend, NeuTra family, validation rules, and HMC diagnostics. The locked q=1 eigensquare-root target is reported only as sensitivity evidence. For execution topology, compare serialized `tf.map_fn` against the persistent scalar CPU-worker route on identical q, batch, target points, and trainer state. |
| Primary pass | Every retained chain coordinate has rank-normalized split R-hat `<=1.01`, bulk ESS `>=400`, tail ESS `>=400`, and mean MCSE/SD `<=0.05`, with zero exposed native divergences and finite target values/scores. |
| Training admission | Both independent NeuTra seeds have finite value/score/round-trip probes, saturation `<=0.05`, and a one-sided paired 95% heldout reverse-KL improvement bound below zero. Training metrics nominate transports; they do not establish posterior correctness. |
| HMC tuning | Tune each frozen transport separately toward acceptance 0.70. Tuning samples are excluded from retained evidence. A fixed kernel is admitted only after a fresh confirmation. |
| Predictive/recovery check | On synthetic q>1 data, report truth coverage by marginal 95% intervals and standardized posterior-mean errors; compare independent retained-chain forecast laws using predeclared one-to-ten-step mean/log-variance influence statistics with MCSE-aware uncertainty. These are calibration and replication checks, not an oracle posterior. |
| Hard vetoes | Nonfinite values/scores/samples, wrong score, failed round trip, positive native divergence, any unmoved chain, invalid lineage, host RSS `>64 GiB`, GPU OOM, or artifact failure. |
| Explanatory only | Loss curves, runtime, memory below cap, acceptance, observed truth errors, and point differences without declared uncertainty. |
| Artifact | `docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/` plus a result note beside this plan. |

Passing a hard screen means the rung is viable under this design. It does not
statistically rank q values or establish that NeuTra is superior to plain HMC.

## Phase 1: Dimension-General Target And Directional Score

Implement a target that embeds four free coordinates into the rung's full
fixture and returns only the corresponding four analytic score directions.
Value and score must both use `tf_principal_sqrt_ukf` semantics; differentiating
the historical eigenvector-oriented SVD factor is not admitted for this ladder.
The derivative engine must construct derivative tensors with leading dimension
four directly. It must not construct all `9q^2+13q+2` directions and gather
afterward.

Checks:

- q=1 principal-root value/score self-consistency plus quantified sensitivity
  against the locked historical target at the prior center and fixed shell
  points;
- q=1,2,5,10,20 score finite differences on all four free coordinates;
- eager/XLA and scalar/batch parity;
- free-coordinate ordering, fixture embedding, prior, and target signature;
- q=20 isolated host/GPU memory measurement under the 64 GiB host ceiling;
- negative controls for wrong coordinate order and perturbed fixture identity.

Handoff: all target checks pass and q=20 does not allocate full-chart derivative
tensors. Otherwise write a blocker result and do not train.

## Phase 2: Process-Parallel Timing Canary And Budget Freeze

The previous canary serialized 480 independent filters through `tf.map_fn`; its
30.9/51.4 GPU-hour projections are measurements of that rejected execution
topology and are not current ladder budgets.

Implement the DSGE-HMC execution boundary prospectively:

1. start a persistent `spawn` process pool before the first target batch;
2. make every child inherit `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import;
3. build and warm one non-XLA scalar TensorFlow value/analytic-score graph per
   worker at the target's valid prior center;
4. submit one blocking readiness task per configured worker and release them
   through a full-pool startup barrier before sharding the first real request;
   record every startup PID, and allow a real request to contain fewer rows
   than configured workers without deadlock;
5. shard each current transport batch into ordered, identity-hashed worker
   payloads and reject missing, duplicate, reordered, stale, nonfinite, or
   shape-invalid results;
6. in the parent, attach returned theta-scores through `tf.custom_gradient`,
   then execute the transport gradient and Adam update on the selected GPU;
7. use the same CPU-worker route for a value-only SVD-UKF validation call;
   validation must not propagate analytic score derivatives that the heldout
   reverse-KL loss does not use;
8. prefer physical GPU 1 before TensorFlow import, falling back to physical GPU
   0 only when GPU 1 is absent; record the physical selection and visible
   logical device. Spawned workers must never run this GPU selection.

First prove native/worker scalar value-score parity and exact native/external
trainer-update parity from identical state. Then benchmark worker counts
`{16,32,64,96}` on q=1 with batch 480 and at least two current-step calls after
worker warmup. This includes the 96-worker DSGE-HMC reference topology rather
than treating a small worker count as the transferred method. Select the
smallest count statistically tied within 10% of the fastest observed warm wall
time, subject to finite outputs and the 64 GiB aggregate active-worker RSS cap.
Repeat the selected count at q=2. The observations are descriptive timing
evidence; they choose execution capacity but do not rank scientific methods.
If q=1 leaves less than 25% host-RAM headroom at the selected count, preflight
q=2 at the next lower tested count and extrapolate from measured active-worker
RSS before launching the selected count. Do not launch a worker rung whose
conservative parent-plus-worker projection exceeds 64 GiB.

After topology selection, run one trusted canary per new q in increasing order.
Each canary executes 10 NeuTra training steps for each of two independent seeds,
transport reload, and a minimal transformed-HMC mechanics call. Record pool
startup, worker warmup, per-step wall, maximum worker shard wall, parent GPU
update wall when available, host RSS, GPU allocator memory, worker PIDs, and
CPU/GPU visibility.

Project the full cost using the slower warm rate with a 50% margin and include
fresh worker startup/warmup. This repair is bounded to focused tests plus q=1
and q=2 topology/timing canaries; it does not authorize 5,000-step training,
Optuna, HMC tuning, or retained acquisition. The 64 GiB host cap remains
binding. A material ladder budget is frozen only from the repaired topology.

Phase-2 decision: 32 workers are the selected viable topology at q=1 and q=2.
The measured q=1/q=2 full-canary warm maxima are 0.325/0.354 seconds and the
two-seed 5,000-step projections with the declared 50% margin are 5,039/5,487
seconds. Conservative combined host high-water marks are 66.83/68.05 GB under
the exact 68.72 GB cap. The q=2 headroom is only 0.67 GB, so 64 and 96 workers
are resource-vetoed by measured 32-worker scaling and must not be launched
under this cap. A startup barrier is mandatory: without it, early batches were
served by only 5--7 initialized workers and produced invalid timing projections.
The barrier must be exercised by explicit readiness tasks rather than the first
data shards so that pool initialization is independent of first-request batch
size.

Handoff: Phase 3 may use 32 workers, GPU 1 with GPU 0 fallback, and the same
startup barrier/value-only validation route. Material Optuna and 5,000-step
training still require a separately frozen cumulative budget covering all
trials, both final seeds, HMC tuning/retention, and predictive validation.

### Phase-2B Higher-Rung Budget Canaries

The historical q=2 32-worker canary left only 0.67 GB below the 64 GiB cap, so
the first higher-rung pass uses 16-worker canaries at q=5,10,20 in increasing
order, with the same two streams, 10 steps, batch 480, readiness barrier,
value-only validation, transport reload, and minimal HMC mechanics call.

The prospective memory bound is conservative: the isolated q=20 target used
3.03 GiB and the largest observed canary parent used 8.72 GiB, so
`16 * 3.03 + 8.72 = 57.20 GiB`, leaving 6.80 GiB under the 64 GiB cap before
launch. After each actual rung, stop before the next rung if any hard veto is
present, startup PID count is not 16, a worker sees a GPU, combined host memory
exceeds 64 GiB, or the artifact/source binding is invalid. Otherwise continue;
a failed training-loss row is explanatory in this canary and is not a stop
unless it is nonfinite or invalidates the mechanics.

Use each rung's slower warm stream with the existing 50% margin to project its
two final 5,000-step streams. These projections still exclude Optuna, HMC
tuning/confirmation, retained acquisition, and predictive validation. After
q=20, write a cumulative Phase 3--6 budget table before any material launch.

After all 16-worker canaries pass, a 32-worker topology check is admissible for
each higher rung only when `parent_rss + 2 * measured_16_worker_rss` remains
below 64 GiB. Run q=5,10,20 sequentially; an actual cap breach stops later
32-worker checks. Select 32 workers for material training only when it remains
under the cap and is descriptively faster than 16. This topology selection does
not alter HMC or scientific promotion criteria.

For q=20 only, the 80-dimensional augmented principal-root/Sylvester algebra
may benefit from limited intra-worker threading. If 32 one-thread workers are
not at least 10% faster than 16 one-thread workers, test 16 workers with two
threads each. Stop unless this is at least 10% faster than 16x1; only then test
16x4. Select the smallest thread count within 10% of the fastest result, under
the same 64 GiB cap. Timing differences remain descriptive topology evidence.

### Phase-2C Four-Chain HMC Execution Topology

The source transfer boundary must follow the implementation actually present in
`dsge_hmc`, not the informal label "multiprocess HMC". The inspected
`MultiChainHMC` route batches chains inside TensorFlow during adaptation, then
builds one scalar-chain `tf.function`/XLA callable per production chain and
invokes those independent callables through a four-worker
`ThreadPoolExecutor`. It does not send HMC target evaluations through an OS
process pool. BayesFilter therefore retains the spawned CPU process pool for
independent NeuTra training rows, but tests scalar-chain threaded XLA within the
single GPU-owning parent for HMC.

Evidence contract:

| Role | Frozen Phase-2C contract |
| --- | --- |
| Question | Does DSGE-HMC-style independent scalar-chain XLA reduce four-chain HMC wall time relative to BayesFilter's current one-call batched-chain XLA topology at q=1 and q=20? |
| Exact baseline | The current `ReusableFullChainHMCRunner` with state shape `[4,4]`, one `tfp.mcmc.sample_chain` call, identical target, deterministic frozen 32x32 three-stage dense-IAF initialization, initial states, fixed kernel, result/burn-in counts, and root seed. TFP's batched random stream is not claimed to equal four scalar streams draw-for-draw. |
| Candidate | Four scalar-state reusable XLA runners in one GPU-owning process, one per chain, compiled/warmed before measurement and invoked either serially or with `ThreadPoolExecutor(max_workers=4)`. Each scalar seed is deterministically folded from the same root seed. |
| Primary topology criterion | Select threaded scalar-chain execution only if its repeated warm wall time is at least 10% below both the batched-chain and scalar-chain-serial warm walls at q=20, while all mechanics/parity/resource checks pass. Otherwise retain batched-chain XLA. q=1 is a lower-cost consistency comparator, not the deciding rung. |
| Hard vetoes | Nonfinite sample/trace, wrong shape or chain order, scalar serial/threaded replay mismatch for identical per-chain seeds, unmoved chain, invalid target/transport binding, worker/process GPU misuse, host RSS above 64 GiB, GPU OOM, or inability to expose required HMC health telemetry. Native divergence unavailability is reported as unavailable and is not converted into zero divergences. Batched/scalar draw equality is not a veto because their TFP random-stream shapes differ. |
| Explanatory only | Compile wall, per-chain wall, acceptance, log-accept tails, target-log-probability range, GPU allocator peak, and q=1 timing. |
| Nonclaims | No convergence, posterior correctness, kernel tuning, retained-sample validity, NeuTra quality, or scientific superiority claim. |
| Artifact | `docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/hmc-topology/` plus this plan's result note. |

Use a tiny fixed mechanics contract for the topology preflight and run q=1
before q=20 in fresh trusted GPU processes. Record physical GPU selection,
logical device, XLA/TF32, exact seeds, source hashes, first and repeated warm
walls, per-chain walls, sample/trace parity, movement, finite counts, RSS, GPU
allocator peak, and artifact paths. This preflight is bounded to topology
selection; it does not authorize tuning, convergence diagnostics, retained HMC,
or material training.

Handoff: freeze the Phase 4--6 HMC budget only after the q=20 topology result.
If threaded scalar chains pass and meet the timing criterion, q-general tuning
and retained runners may adopt that execution topology. Otherwise retain the
existing batched-chain XLA route and budget from its measured q=20 warm rate.

Phase-2C result: retain the existing batched-chain XLA route. All q=1 and q=20
mechanics checks passed, and scalar serial/threaded samples and traces were
exactly replayable for identical folded per-chain seeds. The q=20 deciding
medians were 14.202 seconds for batched chains, 14.200 seconds for scalar-chain
serial execution, and 13.975 seconds for scalar-chain threaded execution. The
threaded difference was only 1.6%, below the prospective 10% threshold, while
the independent topology required four compiled graphs and much larger one-time
setup. At q=1, threaded scalar chains were 1.138 seconds versus 0.591 seconds
for the batched baseline. No HMC process pool is admitted.

The selected NeuTra training topologies are q=1/2/5/10 with 32 one-thread CPU
workers and q=20 with 16 one-thread CPU workers. q=20 32x1 was within 1% of
16x1, so the plan's smallest-within-10% rule selects 16x1; 16x2 was materially
slower. Using selected warm maxima and the declared 50% margin gave this
preliminary warm-work-only Phase 3 envelope, later superseded by the startup-
inclusive freeze:

| Component | Maximum hours |
| --- | ---: |
| Six two-stream Optuna trials per rung through 400 steps | 28.99 |
| Two final 5,000-step streams per rung | 60.40 |
| One allowed fresh 5,000-step confirmation per rung | 30.20 |
| Preliminary warm-only Phase 3 total | 119.59 |

For HMC, the measured batched-chain mechanics rates imply an `L=2`-equivalent
19.67 hours with 50% margin for two transports per rung under a reserve of four
512-transition tuning probes, one 512-transition fresh confirmation, and one
4,352-transition retained path (256 initial burn-in plus 4,096 retained draws)
per transport. The q=20 `L=2`-equivalent share is 13.63 hours. This is not yet a
frozen HMC budget: actual tuned cost scales approximately as `L/2`, with added
target bootstrap and compilation overhead, and the admissible leapfrog count is
not known before Phase 4 tuning. Runner compilation, repair reruns beyond the
declared reserve, and Phase 6 predictive simulation were also unfrozen at this
stage. The later freeze supersedes the `119.59`-hour warm-only estimate with a
`121.2874392020`-hour startup-inclusive Phase 3 cap and freezes Phase 4--6.

### Phase-2D CPU Forecast-Pool Timing And Final Budget Freeze

The q-general Phase 6 runner fixes the forecast workload exactly: four
calibration banks of 256 parameter draws plus two charts with four chains and
12,288 draws per chain, always with two forecast replications and horizon 10.
This is 99,328 forecast draws, 388 blocks of 256, and three fresh persistent
pool startups per rung. Before material execution, run one bounded CPU-hidden
timing canary per q with the production worker topology, one 256-draw truth
block, and two exact warm replays under a seed family excluded from material
calibration and validation.

| Role | Frozen Phase-2D contract |
| --- | --- |
| Question | What wall and aggregate host-memory envelope does the implemented persistent scalar principal-root forecast pool require at each q? |
| Exact comparator | The same q-specific scalar forecast worker, 256 identical truth rows, identical ordered seeds, two forecast replications, horizon 10, and the Phase 6 production worker count. |
| Primary budget evidence | First block wall, maximum of two warm replay walls, and `1.5 * (3 * first + 385 * warm_max)` seconds per rung. |
| Hard vetoes | Wrong startup PID count, worker-visible GPU, nonfinite or nonpositive output, replay mismatch, wrong shape/order/hash, aggregate parent-plus-worker `ru_maxrss` above 64 GiB, source drift, or missing structured receipt. |
| Explanatory only | Per-worker timing dispersion and continuous differences between q values. |
| Nonclaims | No forecast accuracy, calibration, predictive equivalence, HMC, NeuTra, model-adequacy, or production-readiness claim. |
| Artifact | `docs/plans/artifacts/ssl-lstm-neutra-hmc-state-complexity-2026-07-19/forecast-timing/forecast-timing-q<q>.json`. |

The HMC operation count is already prospectively bounded without knowing the
selected kernel. At any Phase 5 admission checkpoint, Phase 5 plus the allowed
Phase 6 extension totals 12,544 post-kernel transitions per chart. Combining
the implemented Phase 4 maximum arm ladder with two charts gives 408,800
transition-leapfrogs per rung at the allowed worst-case `L=16`. Apply the
recorded q-specific warm seconds per transition-leapfrog with the declared 50%
margin and add the full 9,000-second cold-runner reserve per rung. Sequential
stopping returns unused budget when tuning selects `L<16`, Phase 5 admits early,
or an earlier veto fires; it does not authorize exceeding the frozen cap.

Skeptical audit: the canary uses the actual Phase 6 worker factory, block size,
worker topology, replication count, horizon, and CPU visibility. It cannot
silently promote replay or throughput into a scientific criterion. The
`L=16` HMC envelope is deliberately conservative and avoids conditioning
launch authority on an unknown future tuned kernel.

Audit decision: `PASS_FOR_BOUNDED_FORECAST_TIMING_CANARY`.

The first ordered canary exposed an XLA-only terminal-sampling defect at q=2.
The principal-root filter returned a finite strict-SPD covariance with minimum
eigenvalue `1.09e-9`; eager eigendecomposition reconstructed it to `8.17e-16`,
but CPU XLA's symmetric eigensolver reconstructed it only to `5.35e-9`, causing
the forecast validity gate to reject a valid covariance. This does not change
the principal-root UKF target. For sampling `x_T ~ N(m_T,P_T)`, any factor
`F F' = P_T` defines the same Gaussian law. Replace only this terminal sampling
factor with XLA-stable Cholesky; keep the filter, target, conditional moments,
and all statistical gates unchanged. Rerun focused q=1/q=2 forecast tests and
restart the timing ladder from q=1 because the executable source signature
changes.

### Phase-2E Current-Source Batched-HMC Rate Refresh

The HMC mechanics receipts used for the preliminary envelope bind an older
`bayesfilter/inference/hmc.py` hash. The current diff only adds the rejected
independent-chain topology and does not modify `ReusableFullChainHMCRunner` or
the retained-archive path, but material Phase 4--6 runners bind the current
file identity. Refresh one tiny batched-chain GPU/XLA rate receipt per q under
the current source before freezing the final cap.

Use the selected one-call batched four-chain topology, deterministic untrained
32x32 three-stage dense-IAF transport family, two results, one burn-in, one
leapfrog, and two warm repeats. Record the maximum warm seconds per
transition-leapfrog. The HMC envelope per rung is
`1.5 * warm_rate * 408800 + 9000` seconds, where 9,000 seconds preserves every
possible fresh compiled runner in Phase 4, Phase 5, and Phase 6. The operation
count includes the maximum seven scale arms, four trajectory arms, the largest
allowed confirmation-plus-adjacent-repair pair, and the full 12,544
post-kernel transitions per chart through Phase 6.

Hard vetoes are nonfinite or wrongly shaped samples, an unmoved chain, positive
exposed divergence, missing GPU placement, host RSS above 64 GiB, source drift,
or a missing immutable receipt. Acceptance and continuous timing differences
are explanatory. This canary cannot tune a kernel, retain posterior evidence,
or support convergence, posterior, transport-quality, or sampler-ranking
claims.

Audit decision: `PASS_FOR_CURRENT_SOURCE_HMC_RATE_REFRESH`.

## Phase 3: NeuTra Hyperparameter Nomination And Training

Engineering status: the q-general runner
`docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py`
implements this phase without launching it. It uses the selected per-rung worker
counts, one persistent spawned CPU pool across trials/streams, detached
value/score training updates on the parent GPU, value-only validation, six-trial
Optuna nomination, and the paired plateau controller. Study/final modes require
an explicit material-run flag, positive cumulative cap, and explicit output
root; contract-smoke mode hides GPUs and starts no pool.

Focused review repaired these failure modes before material use: GPU selection
now occurs before TensorFlow import; every plateau checkpoint is support-probed
before it can become best; learning-rate repair restores the best transport and
Adam state before halving LR while retaining controller patience; final streams
resume from a joint trainer/controller/best-state checkpoint and exact next
program step; Optuna resumes to six total trials rather than adding six; trial
evidence is persisted incrementally; large trainer/transport payloads are
externalized; and a 64 GiB breach is a continuation veto rather than a
resumable time-cap stop. The focused runner/controller tests pass. No study or
final training has run.

For each rung sequentially:

1. Use the existing 32x32, three-stage dense-IAF family with ELU activation,
   per-variable clipping, and `s_max=1`; do not substitute the historical 4x4
   capacity.
2. Run bounded Optuna nomination over learning rate `[1e-4,2e-3]` (log scale),
   initialization scale `{0.005,0.01,0.02}`, and clip norm `{5,10}` using two
   independent training/validation streams. Rungs `50,100,200,400` are
   nomination proxies only.
3. Train two fresh independent seeds under the nominated configuration to a
   maximum of 2,000 steps, batch size 480, with validation every 250 steps.
4. Use the statistically defined paired heldout-improvement rule at every
   250-step boundary. After one cycle without meaningful improvement, restore
   the best trainer and Adam state and halve the learning rate. If improvement
   remains absent, continue through two additional 250-step cycles and stop at
   the third no-improvement boundary (750 steps since the best checkpoint), at
   the 2,000-step maximum, or at a hard/resource veto. Any meaningful
   improvement becomes the new best and resets this sequence. This amendment
   supersedes the earlier 5,000/100/500 final-training cadence but does not
   change Optuna rungs, batch sizes, architecture, seed policy, or promotion
   gates.
5. Freeze the best replayable transport from each seed. Training loss and an
   Optuna objective may nominate but cannot promote a transport.

If one seed fails, perform at most one bounded fresh-seed confirmation with the
same nominated hyperparameters. Do not search architectures within this plan.

## Phase 4: Transformed-Target Preflight And HMC Tuning

Engineering status: the q-general runner
`docs/benchmarks/run_ssl_lstm_neutra_complexity_hmc_tuning_2026_07_19.py`
implements this phase without launching material HMC. It accepts only two
distinct `ADMITTED` Phase 3 result receipts, verifies each externalized frozen
payload hash and q binding, replays the serialized transport, and binds every
resumable arm to the exact receipt, target/transport, runner contract, and
execution-source signature. The selected topology is the existing one-call
batched four-chain XLA runner. A cumulative cap guard uses observed seconds per
transition-leapfrog with 50% margin and a prospective first-compiled-arm
reserve; because HMC arms are non-preemptive, an active arm may overrun its
reserve, but the next arm cannot start after the guard fires.

For every admitted transport:

- verify change-of-variables value and score identities, finite differences,
  forward/inverse round trip, serialization replay, and source/fixture binding;
- tune step size and leapfrog count separately with disjoint seeds;
- target acceptance is `0.70`; use a viable confirmation band `[0.60,0.80]`;
- freeze only a kernel that passes a fresh four-chain confirmation with finite
  telemetry, movement in every chain, and no exposed native divergences;
- tuning/confirmation samples are permanently excluded from retained evidence.

One adjacent trajectory-length repair is allowed for acceptance-only failure.
Geometry, target, or divergence failures return to the relevant earlier phase.
If the coarse scale grid strictly brackets the pilot acceptance band without a
viable point, one geometric-midpoint scale repair is allowed before rejecting
the transport. Movement rate and RMS jump are explanatory diagnostics; finite
telemetry, at least one move in every chain, exposed positive divergence, and
the declared acceptance bands determine the tuning vetoes.

## Phase 5: Four-Chain Retained HMC With Sequential Stopping

Engineering status: the q-general runner
`docs/benchmarks/run_ssl_lstm_neutra_complexity_retained_hmc_2026_07_19.py`
implements immutable 256-draw segments, first-segment burn-in 256, exact Phase 4
receipt/kernel binding, resumable manifest/final-state lineage, streaming XLA
post-archive value/score audits, and cumulative diagnostics in both NeuTra `z`
and common model `theta` coordinates. Its first segment reserve is seeded from
the observed Phase 4 seconds per transition-leapfrog and later reserves use the
maximum observed complete segment-plus-audit rate with 50% margin. It has not
run material HMC.

For each of the two independent transports, acquire immutable four-chain
segments of 256 retained draws per chain. Burn in 256 transitions on the first
segment only. Evaluate cumulative checkpoints at `512,1024,2048,4096` retained
draws per chain. Continue after a diagnostic miss unless a hard/resource veto
fires.

A chart passes only at a checkpoint where all free coordinates satisfy:

- rank-normalized split R-hat `<=1.01`;
- rank-normalized bulk ESS `>=400`;
- rank-normalized tail ESS `>=400`;
- posterior-mean MCSE/SD `<=0.05`;
- finite mapped and transformed samples/value/score;
- zero exposed native divergences and movement in every chain.

Compare the two admitted charts in the common four-coordinate theta chart.
Require the absolute difference in each of four means and ten raw second
moments to be at most three combined MCSEs. This is a replication-stability
screen, not an oracle comparison or equivalence proof.

## Phase 6: Recovery And Predictive-Moment Validation

The locked controlled predictive procedure from the July 18 directional-region
audit requires 12,288 retained draws per chain, not 4,096. Preserve the phase
boundary as follows: 4,096 remains the maximum Phase 5 sampler-admission
opportunity. Only after both charts pass Phase 5 may Phase 6 continue the exact
same immutable fixed kernels and lineages to 12,288 draws per chain. This
extension cannot rescue or reinterpret a Phase 5 sampler-admission failure; it
exists only to meet the prospectively validated predictive sample-size
contract. The extension cost and forecast cost must be frozen before launch.

For q>1 synthetic rungs, report marginal 95% interval coverage of the frozen
truth and standardized mean error `abs(E[theta_j]-theta*_j)/posterior_sd_j`.
These diagnostics are descriptive for one synthetic dataset and cannot prove
frequentist calibration. A future multi-dataset SBC program is required for
that claim.

Using fresh stateless forecast innovations, compare the two independently
trained/sampled charts over horizons 1 through 10 using posterior-predictive
means and log variances. Use the documented influence-function/HAC procedure
and frozen calibration scales. Pass only when no predeclared MCSE-aware region
rejects replication stability. Preserve path plots from dispersed retained
draws as intuitive explanatory evidence.

Use the locked split-region/Rao-Blackwell design: one 20-dimensional average
region at alpha 0.025, ten two-dimensional horizon regions at alpha 0.0025,
familywise alpha 0.05, growing Bartlett HAC multiplier 3.0, zero ridge, and the
frozen acceptable loss computed from the 0.05/1.05 negligible anchors and the
0.20/1.25/0.80 material anchors. Calibration scales must come from a q-specific
calibration-only forecast bank at the known synthetic truth under disjoint
stateless seeds; retained A/B forecast values cannot set the scales.

## Sequential Program Stop And Handoff

Execute `q=1,2,5,10,20` in order. A target-invalidity or resource veto stops
the program. A NeuTra or HMC failure rejects that rung under the current
candidate and records the exact repair trigger; it does not claim the research
direction is invalid. Later rungs normally stop after a failed rung because
the ordered ladder no longer supports a clean complexity boundary, unless the
failure is repaired within the prospectively allowed single repair.

The 64 GiB host-RAM ceiling is `68,719,476,736` bytes. Record both isolated
process `ru_maxrss` and TensorFlow GPU allocator peak. The GPU allocator is
limited by the physical device; 64 GiB is not a GPU-memory claim. Run rungs in
fresh processes so host high-water marks and XLA caches are isolated.

## Skeptical Pre-Execution Audit

| Risk | Finding and repair |
| --- | --- |
| Wrong baseline | Repaired: q=1 is the common principal-root ladder target; the locked eigensquare-root target is a sensitivity comparator only. All rungs use the same coordinate semantics, prior, and square-root backend. |
| Value/score target mismatch | Repaired: both routes use the principal square root; the analytic score differentiates it through the Sylvester equation and no longer requires separated eigenvectors. |
| Proxy promoted | Repaired: Optuna/loss/canaries only nominate or veto training mechanics; retained four-chain diagnostics and predictive replication are primary. |
| Hidden estimand drift | Repaired: q scales state/filter complexity while the estimated block remains four homologous coordinates. Full-parameter estimation is explicitly out of scope. |
| Underidentified comparison | Repaired: do not estimate 64--3,862 coefficients from 30 scalar observations. |
| Memory cap masks bad algorithm | Repaired: directional derivatives must have leading dimension four; full-chart score materialization is forbidden. |
| Serial batch treated as necessary | Repaired: independent filter rows move to persistent CPU processes; the main process keeps only transport/optimizer/HMC GPU work. |
| Informal "multiprocess HMC" treated as source fact | Repaired: direct source inspection shows batched TensorFlow adaptation and per-chain threaded XLA production. Phase 2C compares that exact production boundary against BayesFilter's batched-chain XLA baseline; it forbids forked/shared-GPU TensorFlow processes. |
| Worker score breaks autodiff | Repaired by contract: worker analytic scores enter a parent `tf.custom_gradient`; identical-state native/external update parity is a veto test. |
| Worker imports GPU | Repaired by contract: spawn inherits CPU-hidden environment before TensorFlow import, and every receipt records an empty worker GPU list. |
| Stale or reordered worker payload | Repaired by request id, full-batch hash, shard hash, contiguous range validation, shape checks, and finite checks. |
| First batch smaller than worker pool | Repaired: explicit one-task-per-worker readiness initializes the full pool before real sharding; a focused 2-row/4-worker regression must pass. |
| Wrong GPU fallback | Repaired: physical GPU 1 is preferred before TensorFlow import; GPU 0 is used only when GPU 1 is absent. CPU workers bypass GPU selection. |
| Missing stop | Repaired: 64 GiB host ceiling, physical GPU OOM, one-hour preflight authority, canary projection, per-rung sequential gates, and artifact/source failures are explicit. |
| Unfair comparison | Horizon, observation dimension, free coordinates, prior, filter, NeuTra family, seed count, HMC gates, and maximum retained opportunity are held fixed. Synthetic observations necessarily differ by q and are hashed before fitting. |
| No posterior oracle | Repaired: peer transport replication, truth recovery on synthetic data, forecast moments, and future SBC replace parameter agreement with an unavailable oracle. |
| Stale scalar code | Repaired: q-general Phase 3--6 runners bind the principal-root target, q-specific fixtures, immutable transports/kernels, and common four-coordinate estimand. |
| Misleading pass | Repaired: engineering, training, sampler, and scientific ledgers remain separate; no rung is called converged from short timings. |
| Vetoed or arbitrary transport enters HMC | Repaired: Phase 4 requires two distinct admitted Phase 3 result receipts and verifies the externalized frozen-payload hashes. |
| Resume reuses stale HMC evidence | Repaired: each arm receipt binds the Phase 3 receipt, payload/artifact signature, q-specific target, static runner contract, and execution-source signature. |
| Arbitrary jump threshold rejects a candidate | Repaired: movement rate and RMS jump remain explanatory; only no movement in a chain is a hard movement veto. |
| Coarse scale grid creates false rejection | Repaired: one prospective geometric-midpoint bracket repair is permitted only when adjacent scales strictly bracket the pilot band across all chains. |
| Declared cap ignores q/L cost | Repaired: the final envelope uses the implemented maximum `L=16` HMC operation count, q-specific measured transition-leapfrog rates, full cold-runner reserves, the Phase-2D production-block forecast timing formula, and explicit Phase 3 pool/graph startup reserves. |
| Fresh confirmation budgeted but unreachable | Repaired: the Phase 3 runner now has a separate `confirmation` mode bound to a completed final summary with exactly one `ADMITTED` and one `VETOED` stream, the matching failed-result hash, the same nominated parameters, and the fixed `seed-c` contract. |
| Training startup omitted from warm-rate arithmetic | Repaired: the freeze receipt charges three fresh full-canary launch reserves per rung plus nine additional second-trainer compile excesses for the twelve Optuna trainers not all covered by those three launches. The same 50% margin covers warm work and startup reserve. |

Audit decision: `PASS_FOR_BUDGET_FREEZE_AND_SEPARATE_MATERIAL_AUTHORITY`. The
target, training, HMC, retention, and predictive runner questions are bounded.
The five repaired forecast receipts, five current-source HMC receipts, selected
Phase 3 rates, explicit startup reserves, and exact operation counts replay in
`budget-freeze/budget-freeze-phase3-6.json`. Timing remains descriptive and
cannot promote HMC convergence or posterior validity. Material
Optuna/training/HMC/predictive execution now requires explicit launch authority;
the budget receipt itself provides none.

## Required Result Record

The result note must include a run manifest, per-rung decision table, inference
status table, exact commands/environment/device/JIT/TF32/seeds/wall/RSS/output
paths, hard vetoes, viable candidates, whether any ranking is supported,
descriptive-only differences, the strongest alternative explanation, what
would overturn the conclusion, and the next evidence needed.

## Phase Close

Phase 1 and Phase 2A--2E engineering/timing work are complete. The selected
training topology is 32x1 for q=1/2/5/10 and 16x1 for q=20; the selected HMC
topology is the existing one-call batched four-chain XLA runner. The final
conservative Phase 3 subtotal is `121.2874392020` hours, the HMC subtotal is
`421.5831899725` hours, and the CPU-only Phase 6 forecast subtotal is
`1.1528584148` hours. The frozen sequential wall cap is therefore
`544.0234875893` hours, of which `542.8706291745` hours are GPU-active.
Sequential stopping returns unused budget and cannot exceed this cap.

No 5,000-step NeuTra training, Optuna study, transformed HMC tuning, retained
four-chain acquisition, calibration bank, or predictive validation has run.
The q-general Phase 3--6 runners are engineering-ready and contract-smoked
only. The next step is a separate explicit material launch decision; material
execution remains closed.

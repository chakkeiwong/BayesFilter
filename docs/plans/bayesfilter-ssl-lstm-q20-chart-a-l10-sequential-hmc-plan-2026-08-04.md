# q=20 Chart A L=10 Sequential Fixed-HMC Validation Plan

Date: 2026-08-04
Status: `REVIEWED_READY_TO_EXECUTE`
Scope: Chart A only; frozen trained transport; frozen fixed-HMC kernel

## Research Intent Ledger

| Item | Prospective definition |
| --- | --- |
| Main question | Does the tuning-nominated Chart A fixed-HMC kernel produce four numerically healthy chains that satisfy the repository sequential warm-up, R-hat, and ESS screens? |
| Candidate | `L=10`, step size `0.4148806556986277`, identity mass in trained-transport `z` coordinates |
| Exact baseline | The completed six-`L` tuning artifact. No new tuner, NUTS, mass adaptation, or step adaptation is run. |
| Expected failure mode | The 64-transition tuning pass may not persist: a chain may leave the acceptance bounds, emit invalid status/nonfinite tensors, or fail warm-up/retained R-hat and ESS within the caps. |
| Promotion criterion | At least 2,000 discarded warm-up transitions per chain; latest 1,000 warm-up transitions pass maximum rank-normalized split/folded R-hat `<=1.05`; then at least 1,000 retained transitions per chain pass maximum R-hat `<=1.01`, bulk ESS `>=400`, and tail ESS `>=400` in both HMC and model coordinates. |
| Promotion veto | Any nonfinite state, accepted/proposed target value, target score, log-acceptance ratio, or energy difference; invalid per-transition target status; available positive native divergence; or per-chain chunk acceptance probability outside `[0.35,0.95]`. |
| Continuation veto | Candidate/source identity mismatch, XLA or CPU-isolation failure, worker failure, corrupt/missing archive, campaign wall cap, or inability to preserve the four chain states and random streams. |
| Repair trigger | Failure confined to initialization, kernel acceptance, process topology, or campaign budget is recorded by category. It does not reject the target, transport, trained model, or NeuTra direction. |
| Explanatory diagnostics | Runtime, RSS, chain movement, binary acceptance, finite energy-tail magnitude, and unavailable native-divergence telemetry. |
| Must not be concluded | Posterior correctness, model validity, scientific adequacy, sampler superiority, GPU equivalence, Chart B behavior, or default readiness. |

If this sequential screen passes, the next separate phase is an untouched
posterior/reference validation. Sequential R-hat and ESS are necessary sampler
evidence, not proof that the target or learned transport is scientifically
correct.

## Evidence Contract

- Question: the main question in the ledger above.
- Comparator: the frozen six-`L` nomination, not a historical hand-written grid
  and not the deleted 16-transition run.
- Primary criterion: the prospectively fixed warm-up, retained R-hat, and ESS
  gates above.
- Veto diagnostics: only the promotion and continuation vetoes above.
- Explanatory only: finite energy tails, movement, runtime, and RSS. A finite
  large `abs(log_accept_ratio)` is not native divergence and is not a veto.
- Nonclaim: passing does not establish posterior/reference agreement.
- Artifact root:
  `docs/plans/artifacts/ssl-lstm-q20-chart-a-l10-sequential-hmc-2026-08-04/r1/`.

## Frozen Inputs

| Input | Required identity |
| --- | --- |
| Merged tuning artifact | `docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1/merged-tuning-result.json` |
| Merged artifact SHA-256 | `c3018064fcbbe040b3510165138bc7db7de1b378dd0eb4c1a1b8155af796fb19` |
| Kernel hash | `34b89acd551dd25bee9dd0a463be67ff9d06f08ea3f970da5ffa97b44438ca4d` |
| Checkpoint SHA-256 | `c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Base adapter signature | `a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3` |
| Transformed adapter signature | `9772c5988104a9548e34eb138ffe4e950fb8354580f2395fd96718a35e60103e` |
| Fixed transport manifest hash | `dcb1ec65e7d91a382518a0eef382e3cd8efec78341445f22d4d6ac899ea685eb` |
| Transport hash | `caf6c9ec1a46d04253b2ae3922d83e619f38c824cea955d5da8ac419d2dfed7f` |

The launcher fails closed if any identity changes. The invalid deleted
16-transition run is neither an input nor evidence.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| Four chains | Repository sequential policy; reviewed default | Minimum multi-chain bank for rank-normalized split/folded diagnostics | Too few independent modes represented | Preserve four distinct stateless random streams and report per-chain diagnostics |
| `L=10`, step `0.4148806556986277` | Measured tuning nomination | Only arm passing the six-`L` screen and fresh verification | Short tuning pass was stochastic and had a large finite energy tail | First real 500-draw warm-up chunk applies full health/status/acceptance gates |
| Identity `z` mass | Bound by candidate artifact | Matches tuning and trained-transport coordinate system | Transport does not sufficiently regularize geometry | Warm-up R-hat failure or cap hit is a kernel/transport repair trigger |
| Initial `z` states `(0,0,0,0)`, `(0.5,-0.5,0.5,-0.5)`, `(-0.5,0.5,-0.5,0.5)`, `(0.5,0.5,-0.5,-0.5)` | Inherited from the prior q=20 sequential plan; warm-start hypothesis | Deterministic modest dispersion in nominal NeuTra coordinates | Spread may be too small or start in a pathological region | Prelaunch value/score/status finite check for all four states; warm-up readiness remains decisive |
| 500 draws per chunk | Repository archived-controller default; reviewed execution choice | Preserves meaningful progress and limits diagnostic/archive cadence overhead | Long first chunk delays feedback or exceeds cap | Supervisor records live worker logs; first completed chunk supplies the measured rate |
| 8 cores per chain | Derived from available 33-core lane and earlier 8-core canary | Four chains run concurrently while each batch-native target gets CPU parallelism | Scaling may saturate or contention may dominate | Record affinity, threads, per-worker chunk times, and RSS |
| CPU/XLA FP64 | User-directed validation topology; explicit exception to GPU default | Existing q=20 CPU/XLA path is measured and avoids GPU contention | CPU route may be too slow; XLA receipt may be absent | GPUs hidden before TensorFlow import; real worker log must contain XLA compilation evidence |
| Acceptance bounds `[0.35,0.95]` | Inherited repository sequential controller policy and user-directed gate | Detects persistently unusable fixed-kernel behavior without treating target acceptance as convergence | A single 500-draw stochastic chunk may land outside the bounds | Treat as declared promotion veto only; do not generalize to target or transport invalidity |
| R-hat/ESS thresholds | Repository sequential policy | Standard operational finite-sample screen | Passing can coexist with target or posterior misspecification | Explicit posterior/reference follow-up and nonclaim |
| 24-hour cap (`86,400 s`) | Convenience-chosen bounded campaign cap, informed by measured `L=10` timing | Allows the minimum a plausible chance while bounding compute | Minimum may remain under-budgeted | Forecast before every next chunk; preserve completed shards and classify cap separately |

## Execution Topology

- Supervisor: CPU `32`.
- Chain 0: CPUs `0..7`; chain 1: `8..15`; chain 2: `16..23`;
  chain 3: `24..31`.
- Each chain is one persistent CPU/XLA process. The shared repository controller
  owns phase order, gates, diagnostics, archives, and stop decisions.
- The controller supplies one deterministic chunk seed; the harness folds it
  into four deterministic disjoint chain seeds. No worker adapts the step,
  trajectory length, mass, transport, or target.
- Worker samples and traces are reassembled as `[draw, chain, parameter]`
  before the shared controller sees them.
- `CUDA_VISIBLE_DEVICES=-1` is set before TensorFlow import in supervisor and
  workers. GPU 0 and GPU 1 are not initialized or used.
- Every completed chunk is archived before another chunk starts. Warm-up shards
  are retained but excluded from posterior summaries.

## Sequential Policy

| Field | Value |
| --- | ---: |
| Warm-up chunk | 500 transitions per chain |
| Warm-up minimum | 2,000 per chain |
| Warm-up check window | latest 1,000 per chain |
| Warm-up maximum | 10,000 per chain |
| Warm-up R-hat | `<=1.05` |
| Retained chunk | 500 per chain |
| Retained minimum | 1,000 per chain |
| Retained maximum | 10,000 per chain |
| Retained R-hat | `<=1.01` |
| Bulk ESS | `>=400` |
| Tail ESS | `>=400` |

The minimum valid execution is 3,000 transitions per chain, 12,000 total, and
30,000 leapfrog steps per chain. Fewer completed draws cannot accept or reject
the candidate on convergence grounds.

## Compute Budget And Timing

The strongest current timing anchor is the exact `L=10` tuning worker: a
64-result four-chain verification on six cores took `6,188.897 s`. Linear
projection to 3,000 transitions is about 80.6 hours for that batched topology.
That projection is descriptive and includes static-call effects; it does not
measure four concurrent 8-core one-chain workers.

The earlier one-chain CPU/XLA canary showed that independent processes scale
well, but it used `L=1` mechanics and cannot predict the exact `L=10` rate.
Therefore:

1. The first 500-draw chunk is real archived warm-up, not a disposable canary.
2. Its slowest worker time becomes the forecast for later chunks.
3. Before a later chunk, the supervisor requires the forecast plus a 25% margin
   and 10-minute shutdown/archive reserve to fit the remaining cap.
4. Cap refusal is `UNDER_BUDGETED_PARTIAL`, not candidate rejection.
5. The sequential minima are never reduced to fit the cap.

Attempt budget: one focused test pass, one preflight, one material launch, and
at most one localized retry if scientific target, kernel, topology, gates, and
total cap remain unchanged.

## Pre-Mortem

| How the run could mislead or fail incorrectly | Discriminator |
| --- | --- |
| Tuning artifact drift silently changes the kernel | Validate artifact and kernel hashes plus all adapter/transport identities in every worker |
| Independent workers accidentally use different kernels | Worker readiness payloads must match exact step, `L`, mass, target, and transport bindings |
| Status-invalid finite rows pass | Trace actual target `status_code` and `valid_pre_regularized_score` at every retained transition |
| Large finite energy tails are mislabeled divergence | Preserve `delta_h` as explanatory; native divergence stays unavailable/null unless TFP exposes it |
| Same starting point creates artificial agreement | Use four declared dispersed starts and require latest-window warm-up R-hat |
| A worker restarts and loses chain state | Persistent workers only; any unexpected exit is a continuation veto |
| A 500-draw chunk overruns the wall cap | Forecast gate plus external service runtime cap; completed prior chunks remain immutable |
| R-hat/ESS pass is described as posterior validity | Result note separates sampler, numerical, and scientific ledgers and requires posterior/reference follow-up |

## Skeptical Pre-Execution Audit

1. Wrong baseline: corrected. The exact merged six-`L` artifact is the only
   candidate source; historical grids and the deleted 16-draw run are excluded.
2. Proxy promotion: corrected. Acceptance is only the declared broad kernel
   gate; energy tail, movement, and runtime are explanatory. R-hat/ESS cannot
   establish target correctness.
3. Missing stops: corrected. Identity, finite/status/divergence/acceptance,
   worker, archive, forecast, 24-hour, warm-up, and retained caps are explicit.
4. Unfair comparison: not applicable. Only one frozen candidate is tested; no
   method or chart ranking is attempted.
5. Hidden assumptions: initial states, chunk size, core count, acceptance band,
   and cap are classified above with failure diagnostics.
6. Stale context: corrected. Current authoritative artifact hashes match the
   reboot memo; no BayesFilter/TensorFlow workload was active at audit time.
7. Environment mismatch: corrected. CPU-only intent and XLA are explicit;
   both GPUs are hidden before framework import.
8. Artifact adequacy: corrected. Per-chunk samples, full required traces,
   receipts, seeds, state continuation, diagnostics, hashes, commands, affinity,
   environment, and final summary are preserved.
9. Statistical adequacy: corrected. No decision is allowed before policy
   minima; retained inference uses four chains, modern R-hat, and ESS.
10. Route-policy gap: the repository test references
    `docs/plans/artifacts/neutra-hmc-core-consolidation-and-robustness-2026-07-15/c0/route_ledger.json`,
    but that file is absent from the current shared worktree. This plan directly
    binds the launcher to `run_sequential_neutra_hmc`; the missing pre-existing
    ledger is recorded as a governance-test limitation and is not repaired in
    this concurrent lane.

Audit verdict: `PASS_WITH_RECORDED_ROUTE_LEDGER_LIMITATION`. The missing ledger
does not change the sampler math or execution binding, but the result cannot
claim that the repository-wide discovery test passed.

## Planned Commands

Focused tests:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_neutra_sequential_hmc.py \
  tests/test_ssl_lstm_q20_chart_a_l10_sequential_hmc.py
```

Preflight:

```bash
taskset -c 32 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_chart_a_l10_sequential_hmc_2026_08_04.py \
  --mode preflight
```

Material launch:

```bash
systemd-run --user --collect \
  --unit=bayesfilter-q20-chart-a-l10-sequential-hmc-r1 \
  taskset -c 32 /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_chart_a_l10_sequential_hmc_2026_08_04.py \
  --mode run \
  --cap-seconds 86400 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-chart-a-l10-sequential-hmc-2026-08-04/r1
```

## Result Requirements

The terminal note must include a run manifest, decision table, inference-status
table, separate engineering/numerical/scientific ledgers, and post-run red team.
It must state hard veto evidence, viable-candidate status, whether any ranking is
statistically supported (none is planned), descriptive-only observations, and
the next evidence needed.

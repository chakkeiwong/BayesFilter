# q=20 Chart A Six-L Fixed-HMC Tuning Plan

Date: 2026-08-03  
Status: `COMPLETED_CANDIDATE_NOMINATED`

## Research Intent Ledger

| Field | Predeclared value |
| --- | --- |
| Main question | Does the trained Chart A transport yield one or more viable fixed-HMC kernel candidates when BayesFilter tunes `L in {3,5,10,15,20,25}`? |
| Mechanism under test | Public `tune_fixed_transport_hmc_kernel` dual-averaging and fresh-screen procedure, sharded by `L` across six CPU/XLA processes. |
| Expected failure mode | No tuned handoff lands in the acceptance band, a longer trajectory encounters a numerical/status veto, or shared-host contention exhausts the wall cap. |
| Candidate criterion | An arm passes the public tuner's fresh screen and its configured fresh verification with acceptance in `[0.65,0.75]`, finite required telemetry, valid target status, and no available positive native divergence. |
| Candidate veto | Nonfinite state/target/score/log acceptance, invalid target status, available positive native divergence, runtime failure, or acceptance outside the configured bound. |
| Continuation veto | Source/config mismatch, wrong seed offset, wrong CPU affinity/thread count, visible GPU, XLA disabled, artifact collision/corruption, supervisor failure, or `43,200 s` wall cap exhaustion. |
| Repair trigger | A clean acceptance miss nominates a fresh-step repair; it does not reject fixed HMC, the transport, target, or chart. |
| Explanatory diagnostics | Tuned steps, acceptance, binary acceptance, finite log-accept/energy tails, runtime, and per-arm scaling. |
| Must not be concluded | Chart B behavior, posterior convergence/validity, model adequacy, sampler superiority, GPU equivalence, or default-readiness. |

## Evidence Contract

- Exact target: Chart A checkpoint `1500`, q=20 SSL-LSTM target, frozen trained
  transport, identity mass in transport coordinates, four chains in one rank-2
  batch, and shared scalar step size.
- Exact comparator: no candidate is ranked against the prior `L=2` result.
  That result is timing context only and is outside this candidate grid.
- Candidate criterion and vetoes are those in the intent ledger. Acceptance is
  a tuning criterion here, not convergence evidence.
- If multiple candidates pass, the merged artifact applies the public tuner's
  deterministic selection tuple: distance from target acceptance `0.70`, then
  smaller `L`, smaller step, and canonical candidate index. This selects a
  representative Chart A kernel candidate; it is not a statistical ranking.
- Artifact root:
  `docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1/`.
- No sequential HMC is launched by this plan.

## Grid And Resources

The canonical grid order is `(5,10,15,20,25,3)`. `L=3` is appended rather than
inserted first so the five BayesFilter default arms retain their original
candidate indices and public-API random streams.

| Canonical index | L | CPU cores | Affinity |
| ---: | ---: | ---: | --- |
| `0` | `5` | `6` | `0-5` |
| `1` | `10` | `6` | `6-11` |
| `2` | `15` | `8` | `64-71` |
| `3` | `20` | `16` | `12-27` |
| `4` | `25` | `16` | `72-87` |
| `5` | `3` | `6` | `88-93` |

The six workers use `58` logical CPU cores. The supervisor uses CPU `127`.
Assignments are split across the two sockets to reduce avoidable concentration;
the CPU allocation itself is user-specified except for exact affinity placement.

## Default And Assumption Audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
| --- | --- | --- | --- | --- |
| Chart A only | User-directed campaign scope | One chart can yield a Chart A kernel candidate | Cannot generalize to Chart B | Explicit Chart B nonclaim |
| `L=(5,10,15,20,25)` | BayesFilter API reviewed default | Uses intended multi-length tuning procedure | Long arms may exceed budget | Concurrent sharding and cap |
| Additional `L=3` | User-directed hypothesis | Tests a trajectory longer than forbidden `L=1` and shorter than default `L=5` | Perturbing grid order would alter all default seeds | Append at canonical index `5` |
| Core allocation `6,6,8,16,16,6` | User-directed | Allocates more cores to longer trajectories | Runtime need not scale linearly in `L` or cores | Per-arm wall and call timing |
| Candidate-index seed offsets | Derived from inspected public tuner implementation | Makes sharded calls numerically equivalent to their positions in one grid call | Naive singleton calls would reuse index-zero seeds | Preflight verifies exact bases and emitted tune seeds |
| `(8,16,32)` DA budget ladder | Inherited from the prior target-specific campaign; baseline hypothesis | Preserves prior workload while screening new `L` arms | Short screens may be noisy and miss candidates | Preserve all arms and do not rank failures |
| 64-result/16-burn-in fresh verification | Inherited target-specific campaign setting | Stronger candidate check than API minimum while still bounded | Not a convergence test | Explicit posterior/convergence nonclaim |
| Acceptance band `[0.65,0.75]` and target `0.70` | Inherited reviewed campaign policy | Existing tuning criterion | Short-run Monte Carlo error can reject a usable step | Clean misses are repair triggers only |
| CPU-only FP64 XLA | User-selected campaign lane plus repository XLA policy | Matches measured CPU path | GPU policy does not apply; CPU path is not repository default | Hide CUDA and verify empty GPU list plus XLA log |
| `43,200 s` wall cap | User-directed on 2026-08-03 | Bounds the concurrent campaign at 12 hours | A scientifically valid slow arm may time out | Preserve partial arms; do not merge incomplete grid |

## Sharding Equivalence

The public tuner uses candidate index `i` in its random streams:

```text
tune round r: tune_seed_base + 100*i + r
screen round r: screen_seed_base + 100*i + r
fresh verification: verification_seed_base + i
```

Each singleton worker therefore shifts its three seed bases by those exact
offsets before calling the public API. The merged artifact restores the
canonical unshifted config and canonical candidate indices. Candidate payloads,
target/transport identities, and selection are checked before merge.

## Skeptical Plan Audit

- Wrong baseline: avoided. The prior `L=2` run is timing context, not a
  promotion comparator or a member of this grid.
- Proxy promotion: acceptance only tunes/nominates a kernel; it does not prove
  convergence or posterior validity.
- Missing stop conditions: explicit numerical, status, device, source,
  affinity, seed, artifact, process, and 12-hour stops are present.
- Unfair comparison: all arms hold chart, target, transport, dtype, chain bank,
  ladder, verification, and public API fixed. Only `L`, requested cores, and
  the required candidate-index random stream differ.
- Hidden assumption: simultaneous jobs can contend for cache, memory bandwidth,
  and unrelated host resources. Runtime differences remain descriptive.
- Stale context: the earlier `(2,)` target-specific override is explicitly
  superseded for this grid. `L=1` remains forbidden.
- Artifact adequacy: each arm writes the public-API result plus a worker
  manifest; the supervisor writes exact commands, terminal state, hashes, and
  one merged six-candidate artifact only if all arms complete validly.

The plan passes for one Chart A candidate-tuning campaign. It does not pass as
posterior validation or as a Chart A/B comparison.

## Pre-Mortem

- Misleading pass: a short verification can land in the band by chance. The
  output remains a kernel candidate requiring sequential HMC validation.
- Non-scientific failure: CPU contention, XLA compilation, or seed-sharding
  error could fail an arm. Affinity, XLA, source hashes, exact seeds, per-call
  timings, and terminal codes distinguish these cases.
- Tuning rather than idea failure: every acceptance miss triggers fresh-step
  repair; it does not reject fixed HMC or NeuTra.

## Budget

- Prior campaign charge: `6440.7955 s`.
- New concurrent wall cap: `43,200 s`.
- Maximum cumulative charge after this run: `49,640.7955 s`.
- Attempts: one six-arm launch. Localized harness failure may be repaired and
  retried only within the same 12-hour concurrent wall allowance.

## Launch

```bash
CUDA_VISIBLE_DEVICES=-1 \
PYTHONDONTWRITEBYTECODE=1 \
timeout 43500s taskset -c 127 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_chart_a_six_l_fixed_hmc_tuning_2026_08_03.py \
--mode supervisor \
--cap-seconds 43200 \
--output-root \
docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1
```

# SSL-LSTM q=20 Phase 9A R2a canary repair result

Date: 2026-09-02  
Controlling subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-subplan-2026-09-01.md`  
Master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Status: `R2A_REPAIR_COMPLETE_RETRY_CONSUMED_M3_VETO`

## Decision

The first R2 canary (`attempt-01`) is preserved as failed infrastructure/resource
evidence. It wrote to a literal-brace artifact directory because the launcher
had an extra `}` in its source-owned root. That violated the execution envelope,
so the attempt cannot be used as a candidate or as a valid R2 closeout.

The timing records are nevertheless useful for repair. They show that the
runner pool created exactly two static graphs (screen and selection/held-out)
and reused each after its first trace. Twenty-one full-chain calls completed in
`1645.963995` seconds before the outer `1800` second cap interrupted call 21.
The completed records include all eight screen calls and thirteen of the
sixteen required replicated-selection calls. The remaining selection calls and
the post-selection held-out call therefore could not fit within the observed
cap with the current route. This is a performance repair trigger, not a
scientific or tuning conclusion.

The launcher root was repaired and covered by a focused regression. Under the
unchanged target, data, route, GPU, seed contract, and `1800` second cap, the
one permitted corrected retry was executed as attempt-02. It reached the same
bounded resource failure, so the declared R2 continuation veto is now active
and blocks the six-scope replay; no cap widening or profile substitution is
implicit.

## Evidence contract

| Item | Binding definition |
|---|---|
| Question | Can the declared eight-pair chart-1/beta-0 measured-grid canary complete within its fixed cap after the launcher repair? |
| Target and route | Frozen q=20 target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`, `tensorflow_eigh_strict`, C5 `phase8-k2-compact-high-l3-pure`, fixed-kernel `measured_joint_grid_v1` |
| Comparator | None in this infrastructure retry; attempt-01 is a preserved diagnostic, not a candidate comparator |
| Primary criterion | Corrected canary writes a complete manifest and all declared pair, selection, and held-out receipts before `1800` seconds |
| Hard vetoes | Wrong root/profile/seed/target, output collision, missing receipt, nonfinite route, memory-growth/XLA failure, or corrected bounded resource failure |
| Explanatory diagnostics | Per-call timing, first-trace versus steady role, runner count, allocator telemetry, acceptance, ESS, R-hat, and movement |
| Nonclaims | No whitening, mode discovery, convergence, posterior correctness, sampler ranking, high-dimensional scaling, or Phase 9B readiness |

## Attempt-01 audit

Artifact root (preserved exactly):
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/phase9a-full-replay}/attempt-01/`.

| Measurement | Observed value | Role |
|---|---:|---|
| Profile | `phase9a_full_replay_canary_v1`, scope `3/1` | Identity check |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` | Identity passed |
| Material cap | `1800.0` s | Fixed; not widened |
| Completed calls | `21` | Partial execution only |
| Completed-call sum | `1645.963995` s | Lower bound on required work |
| First traces | Calls `0` and `8` only | Pool reuse evidence |
| Runner count | Screen and selection static contracts | Reuse diagnostic |
| Failure | signal 15 during call `21`; call `22` was only started | Resource/execution; no candidate result |

The completed-call decomposition is:

| Role | Calls | Seconds |
|---|---:|---:|
| Eight screen pairs (`4` results, `2` burn-in) | 8 | `280.857626` |
| Completed replicated selection (`16` results, `4` burn-in) | 13 | `1365.106368` |
| All completed calls | 21 | `1645.963995` |

Selection calls with `L=3` averaged about `59--66` seconds after the first
trace; calls with `L=8` averaged about `155--156` seconds. The required three
uncompleted selection calls plus held-out verification leave materially less
than the observed time needed under the cap. This projection is a resource
diagnostic only; it does not alter the frozen schedule.

## Repair performed

- Removed the extra closing brace from
  `scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu.sh` so the output root is
  exactly the path named by the subplan.
- Added a source-contract regression to
  `tests/test_ssl_lstm_q20_phase9a_repair_runner.py` that rejects the literal
  brace and checks both source-owned profile identifiers.
- Preserved the runner's timing and reusable-pool telemetry; no target,
  bridge, map, HMC kernel, seed, or cap was changed.

Source checksums after repair:

| File | SHA-256 |
|---|---|
| `scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu.sh` | `a5ca1d89ffc78b10e2c42b12795cfa8b78eabd68affbf5cfb91a65dc95c302bb` |
| `docs/benchmarks/run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py` | `72abb608493222b95e293124793a7d79ad2e0b05c7aee066180b4f4dfcdd4b5f` |
| `tests/test_ssl_lstm_q20_phase9a_repair_runner.py` | `8fffb039757cef07461e461423398dea918d96ea9e025b2ae279eda993e6cb77` |

## Verification

| Check | Result |
|---|---|
| `python -m pytest -q tests/test_ssl_lstm_q20_phase9a_repair_runner.py` | `13 passed` |
| `bash -n scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu.sh` | pass |
| `git diff --check` | pass |
| Attempt-01 artifact preservation | pass; no files overwritten |

## Decision and inference status

| Decision | Primary criterion | Veto status | Uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Repair launcher and retry once | Exact root contract and focused test | No post-repair veto | Timing projection comes from one interrupted run | Launch fresh `attempt-02` canary with the same cap | Route is not faster or valid for posterior use |
| Treat attempt-01 as evidence | Durable call records and failure manifest | Invalid artifact root makes R2 candidate status ineligible | Missing final selection/held-out calls | Preserve and cite as resource diagnostic | No candidate ranking |

| Inference class | Status |
|---|---|
| Hard veto screen | Attempt-01 is a failed infrastructure envelope, not a scientific veto; corrected retry remains required |
| Statistically supported ranking | None; no candidate result |
| Descriptive-only differences | Per-call timing and runner reuse only |
| Default readiness | Not assessed |
| Next evidence needed | One corrected canary; a repeated cap failure blocks R3/full replay |

## Pre-mortem and red team

The corrected retry could still fail for an unrelated environment or allocator
reason; the manifest must distinguish that from the already diagnosed path
defect. It could also complete by changing hidden defaults or omitting a pair;
the exact profile payload, call count, and source-owned root checks prevent that
from becoming a pass. Conversely, a pass would establish only bounded execution
and mechanics receipts, not adequate statistical tuning.

The strongest alternative explanation for the timing is transient GPU
contention. A trusted pre/post device snapshot and allocator record in the
retry can test that explanation. If the corrected retry again reaches the cap
with the same two-runner timing pattern, the M3 continuation veto is supported;
the route must then be redesigned under a new plan rather than silently
lengthened.

## Executed retry command

```text
BAYESFILTER_PHASE9A_ATTEMPT_ID=attempt-02 \
bash scripts/run_ssl_lstm_q20_phase9a_full_replay_gpu.sh \
  --profile phase9a_full_replay_canary_v1 \
  --scope-start 3 --scope-limit 1
```

This command used a fresh output directory and the same source-owned seed
namespace. Its terminal outcome is recorded in the M3 result/reset memo; no
further retry is authorized by the current plan.

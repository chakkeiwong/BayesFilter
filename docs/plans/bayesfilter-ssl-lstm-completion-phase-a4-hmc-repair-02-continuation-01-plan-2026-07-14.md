# Phase A4 HMC Repair-02 Continuation-01 Plan

Date: 2026-07-14 (Asia/Shanghai)

Status: `AUTHORIZED_PROSPECTIVE_EXACT_CONTINUATION`

## Objective

Determine whether one exact 250-draw continuation of the viable repair-02
frozen kernel is sufficient to pass the unchanged A4 cumulative sampler-
admission gate at 500 retained draws per chain.

## Entry Conditions

- Repair-02 adaptation is `SELECTED` and hash verified.
- Repair-02 segment 0 is `NOT_ADMITTED` with decision
  `PROMOTION_VETO_EXTEND_IF_BUDGET_ALLOWS`.
- Segment 0 has no hard veto: all four chains moved, samples and telemetry are
  finite, and per-chain acceptance is within `[0.20,0.95]`.
- Target semantic SHA-256 remains
  `549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e`.
- Total trusted GPU time before continuation is `2040.799946242012s`; the
  remaining shared budget is `26759.200053757988s`.

## Frozen Continuation Contract

| Field | Value |
| --- | --- |
| Current state | Exact private final state from repair-02 segment 0 |
| Existing retained draws | Exact private repair-02 segment-0 shard `[250,4,4]` |
| Step size | `0.37613058552609946` |
| Leapfrog steps | `4` |
| Trajectory length | `1.5045223421043978` |
| New burn-in | `0` |
| New retained draws | `250` per chain |
| Seed | `[20260714,1640]` |
| Cumulative shape | `[500,4,4]` |
| Execution | TensorFlow/TFP `float64`, trusted GPU/XLA, TF32 enabled |

No adaptation, warmup, step change, leapfrog change, restart, chain deletion,
seed search, or prior-sample mutation is allowed.

## Evidence Contract

| Field | Requirement |
| --- | --- |
| Question | Do cumulative exact-continuation draws satisfy the existing A4 calibration-input admission criteria? |
| Comparator | The same repair-02 250-draw cumulative diagnostics; used to explain direction only, not rank a sampler |
| Primary admission | All chains move; aggregate and per-chain acceptance `[0.20,0.95]`; max rank-normalized split R-hat `<=1.05`; every bulk/tail ESS `>=100`; every mean MCSE/SD `<=0.10` in latent and free coordinates |
| Hard vetoes | Any hash/source/target/kernel/state drift; malformed or nonfinite artifact; unmoved cumulative chain; positive native divergence if exposed; nonfinite target/log-accept telemetry; output collision; budget failure |
| Explanatory only | Change in R-hat, ESS, MCSE/SD, chain means, initialization memory, acceptance, runtime, and finite telemetry extrema |
| Nonclaims | No convergence proof, posterior correctness, sampler superiority, predictive equivalence, NeuTra readiness, model adequacy, or default readiness |
| Result | `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-repair-02-continuation-01-result-2026-07-14.md` |

## Sequential Stop

1. Verify all repair-02 public/private hashes, source bindings, frozen kernel,
   segment status, and final-state/sample shapes.
2. Run exactly one new 250-draw, zero-burn-in block from the exact segment-0
   final state.
3. Concatenate the immutable old and fresh shards in draw order and recompute
   cumulative admission diagnostics.
4. Stop HMC after this block regardless of result.
5. If admitted, refresh the forecast-calibration plan with exact executable
   counts, seed domains, ridge/condition ladder, and resource stop before any
   calibration run. If not admitted, write a blocker; do not extend again under
   this plan.

## Budget

The conservative continuation projection is `1800s`, well within the remaining
`26759.200053757988s`. Actual trusted wall time is charged to the shared `8h`
cap. Unspent time after this block does not authorize another extension.

## Fresh Namespace

| Artifact | Path |
| --- | --- |
| Runner | `docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_continuation_01_2026_07_14.py` |
| Tests | `tests/test_ssl_lstm_a4_hmc_repair_02_continuation_01.py` |
| Public receipt | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/repair-02-continuation-01/segment-1.json` |
| Private shard root | `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/repair-02-continuation-01/private/` |
| Review | `docs/reviews/bayesfilter-ssl-lstm-a4-hmc-repair-02-continuation-01-native-review-2026-07-14.md` |

## Skeptical Audit

| Challenge | Disposition |
| --- | --- |
| Wrong baseline | Same target, kernel, chains, and exact earlier draws; no weak external comparator |
| Proxy promotion | Only cumulative retained admission criteria can pass; trend diagnostics remain explanatory |
| Missing stop | Exactly one block, then stop whether admitted or not |
| Unfair comparison | No restart, new warmup, retuning, or seed selection; continuation begins at the exact prior final state |
| Hidden assumption | More draws may not repair chain separation; failure hands off to mass geometry rather than repeated extension |
| Artifact relevance | The cumulative archive directly answers whether A4 calibration input is admissible |
| Resource risk | Conservative projection fits with more than 7 GPU-hours remaining |

Audit decision: `PASS_FOR_ONE_EXACT_CONTINUATION_BLOCK`.

## Exact Commands

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-cont01-pycache \
  TMPDIR=/tmp/bayesfilter-a4-cont01-tmp \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_ssl_lstm_a4_hmc_repair_02_continuation_01.py

PYTHONDONTWRITEBYTECODE=1 \
  PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a4-cont01-pycache \
  TMPDIR=/tmp/bayesfilter-a4-cont01-tmp \
  CUDA_CACHE_PATH=/tmp/bayesfilter-a4-cont01-cuda-cache \
  XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_a4_hmc_repair_02_continuation_01_2026_07_14.py run
```

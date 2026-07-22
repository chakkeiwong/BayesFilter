# LEDH LGSSM T=50 Scope Tuning Subplan

Date: 2026-07-19  
Parent: `bayesfilter-ledh-per-model-scope-tuning-master-program-2026-07-19.md`  
Status: `CLOSED_SCOPE_CLAIM_PASS`

## Objective And Entry Conditions

Tune the exact LGSSM scope `T=50,N=1024,float32/TF32,GPU/XLA,K=1024` before
its claim run. The historical `(20,3)` T=50 execution is an untuned baseline;
its seeds `81720..81735` are excluded from tuning and claim selection.

Entry conditions are the fused one-solve Contract E--Chol graph, direct
`TV_col`/`E_row` metrics, exact chunk policy, 8192 MiB logical-device limit,
and the repository-owned `LEDHTuningScope` guard.

## Evidence Contract

- Calibration seeds: `81800..81807`.
- Validation seeds: `81808..81815`.
- Untouched claim seeds: `81820..81835`.
- Warm-start hypothesis: `sinkhorn_steps=20`, `balance_steps=3`.
- Cheaper-first balance ladder: `3,5,8,12,16,25,32`.
- Sinkhorn ladder, only after exhausting balance at a rung: `20,25,30,40`.
- Gates: `TV_col <= 1e-4`, `E_row <= 0.01`, finite value/score, valid
  chart/reset, exact work counts, `StatelessWhile`, replay on the selected claim,
  correct scope hash, and node/resource caps.
- Selection does not use Kalman value or score.

Promotion means only that the selected fixed controls pass this exact T=50
scope. It does not promote another horizon, model, route, N, data regime, or
HMC use.

## Default And Assumption Audit

| Choice | Status | Justification | Failure mode / diagnostic |
| --- | --- | --- | --- |
| `(20,3)` first candidate | Warm start from T=10, not a default | Cheapest known nearby candidate | Expected to reproduce an occasional row failure; next balance candidate follows |
| Balance ladder | T=50 hypothesis | Cheaper control is exhausted first | No passing count means advance Sinkhorn, not relax gates |
| Sinkhorn ladder | Second-stage hypothesis | Tests initialization after balance exhaustion | No pair within grid is a bounded tuning failure |
| 16 tuning + 16 claim seeds | Bounded evidence | Covers 800 states in each partition while retaining holdout separation | No population/universal claim |

## Skeptical Audit

Verdict: `PASS`.

- The baseline and claim scopes now match exactly; no cross-horizon transfer.
- Previous T=50 failure seeds are not tuning inputs.
- Both candidate ladders are explicit CLI inputs, not hidden defaults.
- The offline Python supervisor does not enter the XLA/HMC finite program.
- A failed candidate advances the declared ladder; a failed tuned claim
  triggers a fresh T=50 repair phase rather than threshold relaxation.
- Artifacts answer the tuning question through direct marginals and exact scope
  identity; Kalman diagnostics cannot select controls.

## Artifacts, Checks, And Stop Conditions

Required artifacts are one JSON per candidate, `selected_pair.json`, the T=50
claim JSON, campaign result, run manifest, and a result note. Before launch:
compile/tests/diff check and trusted GPU preflight. After launch: validate JSON,
scope hash, selected-control binding, per-seed/time residuals, work counts,
runtime/memory, and replay.

Stop on no pair within the finite grid, implementation/resource invalidity,
total 90-minute budget, or the untouched claim veto. Candidate marginal
failure is not a stop condition.

## Initial Command (Attempt 01)

```bash
/home/chakwong/anaconda3/bin/conda run -n tf-gpu python \
  docs/benchmarks/run_ledh_offline_ot_tuning_campaign.py \
  --output-root docs/benchmarks/artifacts/ledh_per_scope_tuning_20260719/lgssm_t50_attempt01 \
  --attempt-id lgssm-t50-attempt01 \
  --time-steps 50 \
  --tuning-seed-start 81800 \
  --claim-seed-start 81820 \
  --sinkhorn-candidates 20,25,30,40 \
  --balance-candidates 3,5,8,12,16,25,32
```

## Execution Outcome

Attempt 01 stopped before candidate execution because importing the scope module
initialized TensorFlow before the 8192 MiB logical-device limit was installed.
The harness was repaired to delay that import until after GPU configuration.
Attempt 02 used the same scientific scope, data partitions, search order, gates,
hardware class, and budget in a fresh output directory.

The successful retry changed only the versioned attempt identity and output
root:

```bash
/home/chakwong/anaconda3/bin/conda run -n tf-gpu python \
  docs/benchmarks/run_ledh_offline_ot_tuning_campaign.py \
  --output-root docs/benchmarks/artifacts/ledh_per_scope_tuning_20260719/lgssm_t50_attempt02 \
  --attempt-id lgssm-t50-attempt02 \
  --time-steps 50 \
  --tuning-seed-start 81800 \
  --claim-seed-start 81820 \
  --sinkhorn-candidates 20,25,30,40 \
  --balance-candidates 3,5,8,12,16,25,32
```

The T=50-specific ladder rejected `(20,3)` and `(20,5)` on validation, selected
`(20,8)`, and passed the untouched 16-seed claim. See
`bayesfilter-ledh-lgssm-t50-scope-tuning-result-2026-07-19.md` for the evidence,
manifest limitation, decision, and nonclaims.

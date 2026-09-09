# SSL-LSTM q=20 Phase 9B P1 Sequential Canary Repair Result

Date: 2026-09-06  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9b-p1-sequential-canary-plan-2026-09-05.md`  
Parent master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Status: `P1_HARNESS_REPAIR_COMPLETE_BUDGET_REASSESSMENT_REQUIRED_P2_BLOCKED`

## Result

The reviewed P1 plan did not complete a valid sequential canary. Three fresh
GPU attempts are preserved under
`docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/`.
The first two stopped at a runner chart-object indexing defect. The third
repaired that defect, completed factor chart construction and factor tuning,
and reached the first 500-transition report before stopping at the shared
controller boundary. The runner passed a telemetry evaluator that accepts
sample states where the controller requires a status-mapping summarizer.

No attempt produced a complete sequential arm, a retained posterior stream, a
strict comparator result, or a P1 pass. The failures are classified as
harness/infrastructure failures. They do not reject the target, the factor
numerical-backend direction, or the broader transport research question.

The runner repair is now implemented and covered by focused CPU tests. It uses
the controller's standard target-status mapping, converts shared controller
samples from `[draw, chain, parameter]` to the diagnostic API's
`[chain, draw, parameter]` layout, computes mean MCSE, separates role/arm/
attempt seed namespaces, records `run_start.json` and incomplete failure
provenance, requires the current P1 audit receipt, and performs a pre-chunk
budget forecast.

## Attempt ledger

| Attempt | Observed outcome | Classification | Evidence boundary |
|---|---|---|---|
| `p1-attempt-20260905T120000Z` | Chart construction stopped at `ReferenceAffineTransport` indexing | Harness defect | No HMC evidence |
| `p1-attempt-20260906T104500Z` | Same indexing defect reproduced with traceback | Harness defect | No HMC evidence |
| `p1-attempt-20260906T105000Z` | Factor tuning completed; first 500-transition report failed at callback contract | Harness defect | No durable sequential archive or posterior stream |
| `p1-budget-preflight-20260906` | Repaired launcher rejected the measured 1,423-second chunk forecast before TensorFlow import | Budget veto | No GPU initialization or scientific run |

The first three attempt durations are estimated from artifact birth and
failure-file modification timestamps, not durable run timers. Their aggregate
is approximately `1,832.61` seconds. Against the declared `5,200`-second P1
budget, the nominal remainder is approximately `3,367.39` seconds. The first
sequential chunk took approximately `1,423` seconds after tuning, while the
minimum schedule requires six sequential chunks per arm. That forecast is
`8,538` seconds per arm before any strict comparator completion and therefore
does not fit the declared `2,600`-second arm cap.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| P1 sequential canary | Failed to complete | Harness callback defect; budget forecast vetoes another launch under current schedule | Compile versus steady-state chunk cost is not separated | Keep P2 blocked; measure cost with a reviewed bounded diagnostic and refresh the budget | No sequential sampler readiness |
| Factor candidate | No new P1 evidence | No factor numerical-backend veto fired | Longer sequential behavior remains unmeasured | Retain as candidate only | No superiority, posterior correctness, or default readiness |
| Strict comparator | Not reached | No comparator result exists | Fresh strict tuning and sequential cost remain unknown | Run only under a valid refreshed budget | No factor-versus-strict ranking |

## Inference status

| Evidence class | Status | Interpretation |
|---|---|---|
| Hard veto screen | P1 incomplete; harness and budget vetoes supported | The attempted P1 cannot pass or support sampler claims |
| Statistically supported ranking | None | No paired posterior streams or uncertainty analysis exist |
| Descriptive-only differences | None from sequential sampling | Factor tuning completion is not a posterior comparison |
| Default readiness | Not ready | The factor route remains a q=20 candidate backend; strict remains the default/fallback |
| Next evidence needed | Separate compile/steady timings, valid budget, complete factor and strict sequential arms, chart thresholds, and downstream checks | P2 remains closed |

## Validation performed

The focused CPU regression is:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
tests/test_ssl_lstm_q20_phase9b_p1_canary.py
```

It passed six tests. The current source-only P1 audit is:

`docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-plan-audit-20260906-r4/run_manifest.json`

It passed with status `PASS_PHASE9B_P1_PLAN_AUDIT` after the P1 plan was linked
to M4-P0. The budget preflight is:

`docs/plans/artifacts/ssl-lstm-q20-phase9b-p1-sequential-canary-2026-09-05/p1-budget-preflight-20260906/failure.json`

It is an intentional fail-closed infrastructure receipt, not a scientific
failure. No new GPU process is active.

## Red-team note

The strongest alternative explanation is that the approximately 1,423-second
observation includes compilation or resource contention and overstates the
steady-state cost. The current evidence does not distinguish those causes,
which is why the next step is a separately reviewed compile/steady-state
diagnostic rather than a full P1 retry. A valid future run must not infer
posterior or performance conclusions from the existing timestamp estimates.

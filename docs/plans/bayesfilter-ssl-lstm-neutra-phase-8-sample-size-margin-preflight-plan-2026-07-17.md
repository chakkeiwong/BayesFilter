# SSL-LSTM NeuTra Phase 8 Sample-Size And Margin Preflight Plan

Date: 2026-07-17

Status: `MATERIAL_PREFLIGHT_COMPLETE_MARGIN_AND_DIRECT_VALIDATION_REQUIRED`

## Research Intent And Evidence Contract

| Field | Contract |
| --- | --- |
| Main question | Before acquiring more HMC draws, which draw-count and labeled margin scenarios have adequate operating characteristics for the complete 20-feature one-to-ten-step predictive decision? |
| Exact baselines | Failed 448-draw receipt `ec112880...` and failed 1984-draw receipt `56a34c4a...`; neither is rerun or reinterpreted as a successful design |
| Candidate mechanism | Prospective draw grid `1984, 3072, 4096, 6144, 8192`, analytical precision scaling, and fresh synthetic operating-characteristic simulation |
| Primary feasibility stress | Persistent true-equivalent variance ratio `1.05`, evaluated jointly with every other required mean/variance family |
| Feasibility targets | Simultaneous coverage at least `0.90`, required-decision probability at least `0.80`, and false-decision probability at most `0.05` for every required family |
| Promotion vetoes | Any required family misses a target; covariance/MMD invalidity; historical binding drift; or a result that depends on G/H confirmation values |
| Continuation vetoes | Nonfinite output, GPU/XLA failure, malformed receipt, resource-cap exhaustion, or a discovered error in the historical feature-decision algebra |
| Explanatory only | Per-family probabilities, interval widths, analytical draw approximations, MMD tolerance behavior, sensitivity margins, and resource projections |
| Nonclaims | No margin is scientifically selected; no HMC acquisition is authorized; no G/H equivalence, posterior correctness, sampler ranking, model adequacy, or default readiness is established |
| Result artifacts | JSON under `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/` and a Markdown result beside this plan |

## Why The Previous Design Cannot Be Reused

The earlier draw counts were not derived from a prospective sample-size
calculation. The 1984-draw rung was the previously contemplated 2048 retained
draw checkpoint minus the excluded 64-draw pilot. That is a resource-derived
number, not a power-derived number.

The earlier “midpoint” margins also did not bisect the declared equivalent and
material anchors. For the mean stress cases the arithmetic midpoint between
`0.05` and `0.20` is `0.125`, not `0.10`. On the log-variance scale the midpoint
between `log(1.05)` and `log(1.25)` is approximately `0.13597`, corresponding to
a variance ratio of approximately `1.1457`. These arithmetic midpoints are
useful feasibility sensitivities only. They do not supply the forecasting-loss
or application-utility argument required to choose a scientific equivalence
margin.

## Prospective Layers

### 1. Analytical approximation

Recover worst-coordinate standard errors from the immutable 1984-draw
interval widths and use

\[
  SE_N \simeq SE_{1984}\sqrt{1984/N}.
\]

For each stress requirement, report both an optimistic single-coordinate
approximation and a conservative 20-coordinate Bonferroni power lower-bound
approximation. The latter is essential because the actual equivalence claim
requires all 20 horizon/feature coordinates to pass together. Round required
draws upward to a multiple of the fixed block length 16.

### 2. Fresh synthetic pilot

Use fresh Philox domains to generate controlled paths at 8192 draws per chain.
For each required and explanatory family, estimate the complete 20-feature
long-run covariance, the linear-MMD center/variance, and the MMD degrees-of-
freedom scaling. The target-pilot receipt may supply only the already frozen
bandwidth lineage. No retained archive or G/H forecast outcome may be read.

### 3. Operating-characteristic simulation

Use fresh Monte Carlo seeds, the joint 20-dimensional covariance, and the
prospective `1/N` precision scaling to evaluate each draw count. Simulate the
whole feature decision, not 20 independent marginal screens. MMD and feature
decisions are combined with sharp dependence-agnostic Frechet bounds rather
than an unverified independence assumption. Report Wilson uncertainty
intervals for Monte Carlo probabilities.

The following are labeled scenarios, not selectable scientific margins:

| Scenario | Mean margin | Log-variance margin | Equivalence rule | Role |
| --- | ---: | ---: | --- | --- |
| `historical_original_symmetric` | `0.15` | `log(1.15)` | symmetric Bonferroni | Exact original contract sensitivity |
| `historical_repair_tost` | `0.10` | `0.5*log(1.25)` | IUT/TOST | Exact enhanced repair sensitivity |
| `anchor_midpoint_tost` | `0.125` | `0.5*(log(1.05)+log(1.25))` | IUT/TOST | Arithmetic feasibility sensitivity only |

MMD tolerances remain the historical feasibility grid
`0.005, 0.01, 0.02, 0.04, 0.08, 0.16`. No tolerance is selected by this
preflight.

## Skeptical Pre-Execution Audit

| Risk | Disposition |
| --- | --- |
| Wrong baseline | Prevented by SHA-binding both failed receipts and preserving their decisions as failures |
| Proxy promoted to criterion | Analytical calculations nominate no design; the synthetic result is feasibility evidence only |
| Margin chosen from outcomes | Prevented: scenarios were declared from historical contracts and anchor arithmetic; no G/H confirmation input is accepted |
| Unfair comparison | All scenarios reuse the same synthetic pilot covariance and fresh Monte Carlo draws at each family/draw rung |
| Hidden independence | Prevented in the combined feature/MMD screen by reporting Frechet lower/upper bounds |
| Hidden Gaussian/scaling assumption | Explicitly labeled; any feasible design still requires a smaller direct finite-sample validation before HMC acquisition |
| Stale environment | Serious execution is TensorFlow/TFP `float64`, XLA-default, trusted GPU 1, with device and trace metadata |
| Artifact cannot answer question | Receipt records full family/draw/scenario curves, uncertainty, analytical requirements, and estimated HMC cost |
| Expected candidate failure treated as direction failure | Prevented: an infeasible scenario triggers margin-science or larger-sample redesign, not rejection of predictive validation |

Audit disposition: `PASS_IMPLEMENTATION_AND_FOCUSED_CHECKS`. A material GPU
run is permitted only after the standalone runner and focused tests pass. The
run may establish feasibility or infeasibility under its assumptions; it may
not freeze a scientific margin or authorize HMC.

## Resource And Stop Contract

- Device: physical GPU 1 only; do not use or interrupt GPU 0.
- Runner: TensorFlow/TFP `float64`, XLA JIT, four chains, two forecast
  replications, horizon 10, block length 16.
- Synthetic pilot: four fresh replications per family for the material run.
- Parametric Monte Carlo: 20,000 fresh draws per family/draw/scenario.
- Material runner wall cap: 2400 seconds plus 60 seconds outer cancellation
  margin.
- Stop without retry on binding drift, covariance/MMD inadmissibility,
  nonfinite output, GPU placement failure, retracing outside the declared
  fixed surfaces, serialization failure, or wall-cap exhaustion.
- Resource projections use only the public Phase 7 warm-segment timing receipt
  and are estimates, not acquisition authority.

No HMC command, retained sample shard, G/H confirmation forecast, margin
selection, or Phase 9 decision is bundled into this plan.

## Planned Checks And Handoff

1. Focused CPU-hidden unit tests for bindings, formulas, joint feature logic,
   Frechet combination, scenario labels, seed separation, and prohibited paths.
2. Python compilation and diff hygiene.
3. One small trusted GPU/XLA smoke receipt that cannot emit feasibility.
4. One material preflight receipt under the frozen cap.
5. Result note with decision and inference-status tables plus post-run red team.

Focused native review:
`docs/reviews/bayesfilter-ssl-lstm-neutra-phase-8-sample-size-margin-preflight-native-review-2026-07-17.md`,
verdict `AGREE_GPU_SMOKE_ONLY`. Focused checks passed: `12` tests, Python
compilation, and scoped `git diff --check`.

The smoke uses only the decisive persistent true-equivalent variance-ratio
`1.05` family, one fresh 8192-draw pilot replication, and 512 parametric Monte
Carlo draws. It exercises the exact material fixed shapes and all compiled
surfaces but must return
`PHASE8_SAMPLE_SIZE_PREFLIGHT_SMOKE_PASSED_MATERIAL_REQUIRED` regardless of
its descriptive operating probabilities.

Frozen smoke command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-sample-size-preflight-smoke-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-sample-size-preflight-smoke-cuda timeout 660s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_sample_size_margin_preflight_2026_07_17.py --mode smoke --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/sample-size-margin-preflight-smoke.json --wall-cap-seconds 600
```

Resource contract: one trusted invocation on physical GPU 1, at most 600
runner seconds plus 60 seconds cancellation margin. Stop without retry on
binding drift, invalid covariance/MMD output, nonfinite value, non-GPU
placement, XLA retracing, serialization failure, or cap exhaustion. A passing
smoke authorizes only receipt audit and material-command freeze.

The first non-trusted invocation reported `CUDA_ERROR_NO_DEVICE` before runner
entry and wrote no artifact; under the GPU trust policy this is sandbox
evidence only. The exact command was rerun in the trusted context. It passed in
`21.4779` seconds with receipt
`sample-size-margin-preflight-smoke.json`, SHA-256
`7eaf2b17c56cad4b523f981e0016e6b10366573d160efae92be88cdb3af4c224`.
All seven compiled surfaces traced once on GPU 1, and margin/tolerance
selection remained null. GPU 1 was shared with a separate MacroFinance lane,
so elapsed time is not performance evidence.

Material mode now hard-binds that passing smoke receipt. The binding addition
does not alter the numerical surfaces. The material run uses all 13 declared
families, four fresh 8192-draw pilot replications per family, 20,000 fresh
parametric Monte Carlo draws, and the complete prospective draw/scenario/
tolerance grid. It may label scenarios feasible under the declared assumptions
but cannot select a margin or tolerance.

Material skeptical-audit disposition: `PASS_MATERIAL_PREFLIGHT_ONLY`. The
failed baselines remain exact, G/H confirmation remains unopened, all
feasibility scenarios and targets predate the material outcomes, the smoke is
not promoted to evidence, Frechet bounds avoid a feature/MMD independence
assumption, and the output cannot authorize HMC or Phase 9.

Frozen material command:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX=/tmp/bayesfilter-phase8-sample-size-preflight-material-pyc CUDA_CACHE_PATH=/tmp/bayesfilter-phase8-sample-size-preflight-material-cuda timeout 2460s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_phase8_sample_size_margin_preflight_2026_07_17.py --mode material --output docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/sample-size-margin-preflight-material.json --wall-cap-seconds 2400
```

Resource contract: one trusted invocation on physical GPU 1, at most 2400
runner seconds (`0.6667` GPU-hour) plus 60 seconds cancellation margin. Stop
without retry on any declared hard veto. No HMC acquisition, direct G/H
forecast, margin selection, or Phase 9 action is bundled.

The resource table will separately show HMC and forecast costs. Required
retained draws equal confirmation draws plus the fixed 64-draw pilot and are
rounded upward to the existing 256-draw acquisition segment; any surplus is
excluded from the planned confirmation statistic. Forecast cost uses the
target-pilot warm rate plus one observed compile overhead. These projections
are planning estimates, not evidence that the future target run will match the
same timing.

## Material Result

The trusted material run passed execution in `146.4031` seconds. Receipt:
`sample-size-margin-preflight-material.json`, SHA-256
`ad13cede2f7ab23f18f956eb7eb39e729f1ed987e4175292cafd7ee59786d89d`.
All seven compiled surfaces traced once; covariance and MMD checks were
admissible; margin and tolerance selections remained null.

No scenario was feasible at 1984 draws. The arithmetic-midpoint sensitivity
first passed all dependence-robust screens at 4096 draws and MMD tolerance
`0.01`; both historical contracts first passed at 8192 draws and tolerance
`0.005`. These are feasibility envelopes, not selected contracts. The
analytical worst-realization calculation is materially more conservative,
including 80% all-20-coordinate lower bounds of 9904 draws for the repair mean
and 16784 for the original material mean. This spread triggers direct finite-
sample validation after, not before, a scientific margin choice.

Closeout:
`docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-sample-size-margin-preflight-result-2026-07-17.md`.

If no scenario reaches all targets by 8192 draws, Phase 9 and HMC acquisition
remain closed pending a scientific estimand/margin revision or a newly budgeted
larger grid. If a scenario reaches all targets, the next action is a direct
finite-sample validation at the smallest apparently feasible rung and an
independent scientific justification of the margin. It is not immediate HMC
acquisition.

# LGSSM Selected-Control Kalman Certification Result

Date: 2026-07-19

Status: `ENGINEERING_PASS_T10_INCONCLUSIVE_T50_SCORE_FAIL_NONLINEAR_TRANSFER_NOT_EXECUTED`

Plan:
`docs/plans/bayesfilter-lgssm-selected-controls-kalman-certification-plan-2026-07-19.md`

Aggregate:
`docs/benchmarks/artifacts/lgssm_selected_controls_kalman_20260719/attempt01/aggregate.json`

## Outcome

The exact selected production-shaped scopes were tested against the
differentiated Kalman likelihood:

- `T=10,N=1024,float32/TF32,GPU/XLA`, `sinkhorn_steps=20`,
  `balance_steps=3`, claim seeds `81700..81715`;
- `T=50,N=1024,float32/TF32,GPU/XLA`, `sinkhorn_steps=20`,
  `balance_steps=8`, claim seeds `81820..81835`.

Both executions pass finite, replay, chart, reset, direct marginal, work,
chunk-policy, XLA-loop, GPU, and 8192 MiB memory-cap gates. Neither horizon is
scientifically closed against Kalman under the frozen simultaneous center
screen:

- `T=10` is `inconclusive`. Mean errors are small, but the 16-seed
  simultaneous intervals are not contained within every frozen region.
- `T=50` is `screen_fail`. The `q_scale` HMC-score mean relative error is
  `-31.65%`, with simultaneous 95% interval `[-43.59%,-19.71%]`, wholly
  outside the frozen `[-5%,+5%]` region. The value interval also crosses the
  `0.1%` boundary, but is inconclusive rather than wholly outside it.

Therefore the user's conditional nonlinear extension was not executed. The
prerequisite was not successful, and extending/tuning nonlinear horizons now
would test transfer from an LGSSM route with a known long-horizon score bias.

## Detailed Results

| Horizon | Controls | Engineering | Kalman screen | Value mean relative error | Main score result |
| ---: | --- | --- | --- | ---: | --- |
| 10 | `(20,3)` | pass | `inconclusive` | `0.08425%`; CI `[-0.08472%,0.25323%]` | all mean score errors small; `phi2` and `phi3` intervals cross `5%` boundaries |
| 50 | `(20,8)` | pass | `screen_fail` | `0.08472%`; CI `[0.02758%,0.14186%]` | `q_scale=-31.65%`; CI `[-43.59%,-19.71%]` |

The mean physical and HMC-coordinate results were:

| Horizon | LEDH value | Kalman value | LEDH HMC score | Kalman HMC score |
| ---: | ---: | ---: | --- | --- |
| 10 | `-32.0256103` | `-32.0526157` | `[5.42740,-0.19968,-1.13431,3.28921,6.32127]` | `[5.43244,-0.21207,-1.14517,3.32103,6.33075]` |
| 50 | `-135.9606886` | `-136.0759746` | `[2.72415,-2.63607,0.22185,-0.88340,1.95371]` | `[2.72366,-2.67495,0.26532,-0.67101,1.95942]` |

The OPG diagnostics are secondary and did not affect the classification:

| Horizon | Mean RMS total-metric error | Maximum diagonal standardized error |
| ---: | ---: | ---: |
| 10 | `0.009503` | `0.008275` |
| 50 | `0.026837` | `0.028251` |

These small regularized values do not override the `q_scale` failure. They use
the previously documented diagnostic convenience ridge and have no scientific
acceptance threshold.

## Localization Diagnostic

One predeclared-scope explanatory `T=50` no-reset run used the same
observations, seeds, proposal construction, numerical controls, dtype, and
GPU/XLA environment. It was much worse than active Contract E:

- `phi1` interval `[-78.55%,-12.82%]`, wholly outside;
- `r_scale` interval `[45.69%,116.33%]`, wholly outside;
- several other coordinates were wide and inconclusive.

Thus the evidence does not support a reset-only explanation. Contract E
materially repairs the shared finite-particle proposal/weight recursion, but
the repaired route still retains a long-horizon `q_scale` bias. The next repair
must inspect the time-local score accumulation and finite-particle
normalization/proposal contributions for `q_scale`, comparing active reset,
no-reset, and Kalman increments on identical random streams. Increasing
`balance_steps` or `sinkhorn_steps` is not justified by this result because all
direct OT marginal gates already pass; tuning OT controls against Kalman would
also violate the selection contract.

## Source And Harness Findings

The fresh current-source outputs were not bitwise identical to the historical
selected-control claims. The repository remained at Git commit
`9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b`, but shared uncommitted generic
transport optimization changed source hashes and float32 evaluation order.
At `T=10`, the aggregate physical-score drift was only `8.57e-6` relative and
did not change the old or new classification (`inconclusive`).

The initial aggregate also exposed a target-identity bug: it recomputed Kalman
on the original float64 observation tensor, while the production candidate
consumes float32-rounded observations and the node correctly casts those bytes
back to float64 for the oracle. The aggregator was repaired to use the exact
production-rounded observation target. Focused tests now bind that oracle.

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Established | Not established |
| --- | --- | --- |
| Engineering correctness | exact selected controls/seeds; current-source claim revalidation; finite/replay/chart/reset/marginal/work gates; `StatelessWhile`; TF32/GPU/XLA; `K=N`; 8 GiB cap | bitwise equality to the pre-optimization float32 source |
| Numerical validity | direct `TV_col` and `E_row` pass at both horizons; exact Kalman target identity; simultaneous interval calculation; average-OPG diagnostics | score equivalence at `T=10`; value equivalence at either horizon; `q_scale` score validity at `T=50` |
| Scientific interpretation | `T=50 q_scale` bias is statistically outside the frozen center region; no-reset is worse, so the failure is not reset-only | parameter-region validity, nonlinear carryover, HMC energy accuracy, posterior validity, method superiority |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| retain selected OT controls for their exact transport scopes | direct numerical/engineering claim passes | no transport veto | OT validity does not imply score validity | keep `(20,3)` and `(20,8)` as scope-specific numerical settings only | Kalman score correctness |
| do not close LGSSM `T=10` | simultaneous screen inconclusive | no hard engineering veto | 16-seed power, especially small `phi2` scale | paired time-local score diagnostics and a predeclared precision design | candidate is wrong or correct |
| reject current `T=50` score for Kalman/HMC-facing use | `q_scale` interval wholly outside `5%` region | scientific score veto | which shared recursive contribution creates the bias | decompose per-time normalization, proposal, carried-weight, and reset score contributions | Contract E idea rejected |
| do not execute nonlinear transfer program | prerequisite required both LGSSM horizons to pass | continuation veto fired | nonlinear behavior not tested in this campaign | repair LGSSM long-horizon score, then write/review a fresh nonlinear per-scope tuning program | nonlinear models fail |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto evidence | `T=50 q_scale` simultaneous interval is wholly outside the frozen region |
| Viable candidates | the finite transport implementations remain engineering-valid; `T=10` remains scientifically unresolved |
| Statistically supported ranking | none |
| Descriptive-only differences | OPG diagnostics, runtimes, memory, no-reset comparative magnitudes |
| Default readiness | no new default; selected controls remain exact-scope settings only |
| Next evidence needed | time-local paired score decomposition at `T=10,50`, same-scalar checks on each contribution, and a precision design that separates Monte Carlo uncertainty from persistent bias |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b` with shared uncommitted transport optimization and this campaign's new files |
| Environment | conda `tf-gpu`; TensorFlow `2.19.1` |
| Hardware | NVIDIA GeForce RTX 4080 SUPER; TensorFlow logical limit `8192 MiB`; TF32 enabled |
| T=10 wall / compile+first / warm | `44.485 / 24.216 / 7.892 s` |
| T=50 wall / compile+first / warm | `123.381 / 61.135 / 45.987 s` |
| T=50 no-reset wall / warm | `16.280 / 0.485 s` |
| Peak TensorFlow allocation | `T=10: 897,992,704`; `T=50: 939,740,928` bytes |
| Main aggregate SHA-256 | `dc6f0422bc6cf4117fcfc83e70b7d1057817f68367d816ec4c5e9237262067d6` |
| T=10 node SHA-256 | `7d17da894af1335b7df57c4fa51d2b8d78feebd314dbc6ae60013e040c3b7c50` |
| T=50 node SHA-256 | `8a8f517c1343c3a293f63651dc8b0c128e748dbd955b89002b88ca950d889ff2` |
| No-reset node SHA-256 | `31b54044f54941be0a002b503af6869db1520eca12b1b9f5260047ee42192e86` |

## Checks

- focused preflight: `31 passed`, two dependency deprecation warnings;
- post-repair certification tests: `4 passed`, two dependency warnings;
- Python compilation: pass;
- diff whitespace checks: pass;
- trusted `nvidia-smi` and TensorFlow GPU probe: pass;
- both claim nodes and the no-reset diagnostic completed with structured JSON.

## Post-Run Red Team

The strongest alternative explanation for the `T=50 q_scale` failure is not a
wrong manual derivative of the executed scalar: the current route has prior
same-scalar derivative evidence, while this campaign compares that finite
scalar's score with a different exact Kalman target. The failure can arise from
finite-particle approximation bias in the shared proposal/importance-weight
recursion, from Contract E's remaining approximation, or their interaction.
The no-reset result rules out the simple claim that removing Contract E yields
an otherwise correct filter.

The result would be overturned by a source-matched time-local decomposition
showing a harness coordinate/target error or by a predeclared higher-precision
replication whose simultaneous `q_scale` interval lies within `[-5%,5%]`.
Neither evidence exists. The weakest evidence is `T=10`, where the mean is
close but the interval is too wide; it must remain inconclusive.


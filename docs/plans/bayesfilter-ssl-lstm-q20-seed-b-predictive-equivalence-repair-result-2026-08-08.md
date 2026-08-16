# SSL-LSTM q=20 seed-B predictive-equivalence repair result (2026-08-08)

## Outcome

The repaired predictive-equivalence harness was implemented and executed through
two prospective calibration attempts. Both canaries and both null-only scale
stages passed. Attempt `r2` showed that the initial 1,024-draw calibration was
underpowered. Attempt `r3` increased calibration to 16,384 draws per lane and
repaired feature-interval power, but no frozen MMD tolerance simultaneously
passed the negligible-mean equivalence family and the shape-only material
family. Nomination therefore failed, as predeclared.

Fresh validation, the posterior-mean-versus-true material comparison, and audit
were not run. The scientific question remains unanswered. This is negative
evidence about the current combined calibration design, not evidence that the
posterior-mean plug-in is predictively equivalent or materially different.

## Claimed and computed targets

| Item | Verdict |
|---|---|
| Claimed target | Practical equivalence of the ten-step q=20 output law at the seed-B posterior-mean physical parameter versus the true-control parameter. |
| Quantity actually computed | Prospective operating behavior of the repaired simultaneous standardized-mean/log-variance plus cross-lane linear-MMD decision on synthetic calibration families. |
| Relation | Different. Calibration is required to open the material target but is not the material target. |
| Supporting artifacts | `r2/nominate.json` and `r3/nominate.json` under the repair artifact root. |
| Unproved | The posterior-mean material output-law status, audit replication, posterior correctness, parameter identification, model adequacy, NeuTra superiority, and default readiness. |

## Engineering repairs

The new runner:

- applies the frozen true-control center and scale through XLA
  `standardize_forecast_paths` to every calibration and material arm;
- derives bandwidths only from a standardized null true-control scale bank;
- uses iid fixed-parameter forecasts with block length `1`;
- uses the correct equal-arm `+2/-2` influence covariance construction;
- calibrates the complete feature-plus-MMD decision over all six families;
- keeps calibration independent of the NeuTra transport and HMC archive;
- requires passed, source-bound scale, nomination, and validation receipts
  before material execution; and
- uses TensorFlow/TFP only, CPU/XLA, and `CUDA_VISIBLE_DEVICES=-1`.

Five focused enforcement tests pass. The first `r1` canary computation exposed a
`TensorShape` JSON serialization error after computation but before artifact
write. The serializer was repaired and retested. A later pre-scale audit found
an inconsistent population/sample denominator in the correlation construction;
it was repaired before any shape-family calibration. `r1` remains superseded
engineering evidence. No material seed was opened by either repair.

## Scale result

The coherent `r3` null-only scale stage passed in `32.2322 s`:

- 4 lanes x 2,048 draws/lane x 2 forecast replications;
- all ten scales strictly exceeded `1e-8`; floor use was false;
- complete-path median standardized distance: `4.4083772261`;
- frozen bandwidths: `2.2041886130`, `4.4083772261`, and `8.8167544522`;
- scale-bank correlation diagonal equaled one within `1e-12`; and
- archive/transport loading was false.

## Calibration results

### Attempt r2

`r2` used 20 replications, 1,024 draws per lane, two forecast replications, and
the grid `0.004` through `0.020`. Nomination failed in `207.4642 s` with zero
hard vetoes.

- At tolerance `0.008`, identical and negligible mean passed `20/20`.
- Negligible variance passed `1/20`; feature intervals were too wide.
- Shape-only skew produced `1/20` material decisions; MMD intervals were too
  wide.
- Material mean and material variance produced `20/20` material decisions at
  every tolerance because their feature intervals rejected equivalence.
- All known feature truths were covered `20/20` in every family.

This was an underpowered calibration result and triggered the pre-material `r3`
power repair.

### Attempt r3

`r3` used fresh seeds, 20 replications, 16,384 draws per lane, two forecast
replications, and the prospective grid `0.0005` through `0.003`. Nomination
failed in `467.0891 s` with zero hard vetoes.

Feature power was repaired:

- all six families had `20/20` feature decisions in the required direction;
- the median maximum simultaneous feature width fell to `0.03523` for forecast
  families and `0.05119` for shape-only skew; and
- known feature truth coverage was `18/20` for the five forecast-derived
  families and `20/20` for analytic shape-only skew. The `18/20` descriptive
  count was not a nomination gate but would have failed the later 54/60 gate if
  that rate persisted.

MMD operating separation still failed:

| Tolerance | Identical pass | Negligible mean pass | Negligible variance pass | Shape material | Material mean | Material variance |
|---:|---:|---:|---:|---:|---:|---:|
| `0.00125` | 16/20 | 3/20 | 13/20 | 19/20 | 20/20 | 20/20 |
| `0.00150` | 19/20 | 7/20 | 16/20 | 17/20 | 20/20 | 20/20 |
| `0.00175` | 20/20 | 11/20 | 20/20 | 12/20 | 20/20 | 20/20 |
| `0.00200` | 20/20 | 16/20 | 20/20 | 7/20 | 20/20 | 20/20 |
| `0.00225` | 20/20 | 20/20 | 20/20 | 5/20 | 20/20 | 20/20 |

No tolerance met every required `16/20` gate. The limiting conflict was
negligible mean versus shape-only skew. Descriptively, the median MMD estimates
were `0.000067` for identical, `0.000764` for negligible mean, and `0.002818`
for shape-only skew. Those point estimates do not establish a ranking or a
valid tolerance by themselves.

## Decision table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Repair implementation | Passed focused tests and canary mechanics | No nonfinite paths, inadmissible covariance, stale receipt, GPU use, or XLA failure | Source tests are focused, not repo-wide | Preserve runner as a fail-closed research harness | General production readiness |
| Scale receipt | Passed | No floor use; null-only provenance passed | Scale is one frozen bank | Reuse only with an explicitly reviewed new calibration design | Universal q=20 scale |
| `r2` nomination | Failed | Zero hard vetoes | Insufficient feature and MMD power | Executed prospective `r3` power repair | Candidate failure |
| `r3` nomination | Failed | Zero hard vetoes | Current MMD statistic cannot meet both negligible-mean and shape-only count gates at this budget | Stop current campaign; redesign the shape/dependence discriminator or family/margin evidence contract prospectively | Candidate equivalence or material difference |
| Validation/material/audit | Closed | Required nomination receipt absent | Entire material question unopened | Do not infer a result | Any posterior-mean output-law verdict |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | No invalid numerical, target, provenance, XLA, or artifact veto occurred in `r2` or `r3`; nomination failure itself closed later stages. |
| Statistically supported ranking | None. The campaign calibrated a decision and did not compare competing stochastic methods. |
| Descriptive-only differences | Per-family pass counts, MMD estimates/intervals, feature widths, and runtimes are descriptive calibration evidence. |
| Default readiness | Not established. |
| Next evidence needed | A new pre-material calibration plan with a discriminator that separates the predeclared negligible-mean law from the shape-only material law, followed by fresh nomination and validation. |

## Research ledgers

Engineering correctness: the repaired orchestration passed five focused tests,
two coherent canaries, null-only scale checks, XLA compilation, receipt binding,
and all numerical validity checks exercised by nomination.

Numerical/statistical validity: both nomination attempts completed without hard
vetoes. The current decision did not achieve its predeclared calibration
operating characteristics; validation was correctly not opened.

Scientific interpretation: no posterior-mean candidate path was generated by
the repaired formal harness. The result rejects the current calibration design
at its tested budgets. It does not reject the posterior-mean candidate, NeuTra,
the SSL-LSTM model, or the predictive-equivalence research direction.

## Post-run red team

Strongest alternative explanation: the linear-MMD estimator and its two-pair
interval, rather than intrinsic law overlap, may be an inefficient discriminator
for this combination of raw `+0.05` mean shift and analytic skew. More draws
could narrow intervals, but `r3` already used 16,384 draws/lane and still showed
no tolerance meeting both count gates; blindly escalating draws would be local
optimization without a reviewed power model.

What would overturn this result: a fresh prospective design whose calibrated
complete decision passes the equivalence and material families on nomination
and a disjoint 60-replication validation. That would overturn only the harness
failure, not establish material equivalence until the material and audit stages
also pass.

Weakest evidence: the selected calibration families and transferred margins are
working hypotheses. In particular, whether raw `+0.05` is the correct q=20
negligible mean boundary and whether coefficient `0.35` is the correct minimum
shape alternative remain policy choices, not universal truths.

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` with dirty concurrent worktree |
| Environment | `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` |
| Hardware | CPU only; `CUDA_VISIBLE_DEVICES=-1`; GPU deliberately hidden |
| XLA | Enabled for forecast and statistical kernels |
| Data/target | Synthetic q=20 target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Adapter | `a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3` |
| Seeds | Recorded in the plan and structured artifacts; `r3` nomination used fresh namespaces `900000` and `1100000` |
| Serious commands | Recorded exactly in each structured artifact manifest |
| Artifact root | `docs/plans/artifacts/ssl-lstm-q20-seed-b-predictive-equivalence-repair-2026-08-08/` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-predictive-equivalence-repair-plan-2026-08-08.md` |
| Result | This file |

Important SHA-256 receipts:

- `r2/nominate.json`: `4dd791bf0d89dae8a3b3e3f8154244dcc36566c261f6d7f6ff069f6244634b7b`
- `r3/canary.json`: `e28cd27f4e5374f973a5fb092e1aa26e281bf9b740b39cf65dbb7eb318e3a194`
- `r3/scale.json`: `d0db6b46446810b5a3354af7dc14cad351818ab8cdeaedc236180730a230525e`
- `r3/nominate.json`: `90f7825d986f030c690a23c72e5ede4ca2210a7370e3ba181b598094fb414068`


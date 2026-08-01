# LGSSM N=10000 Tuned Kalman Certification Result

Date: 2026-07-20

Status: `ENGINEERING_PASS_KALMAN_SCREEN_FAIL`

Plan:
`docs/plans/bayesfilter-lgssm-n10000-tuned-kalman-certification-plan-2026-07-20.md`

Aggregate:
`docs/benchmarks/artifacts/lgssm_n10000_tuned_kalman_20260720/attempt01/aggregate.json`

## Verdict

The proper independently tuned `T=50,N=10000` run is engineering-valid, but
it does not improve the frozen Kalman certification issue relative to the
independently tuned `N=5000` claim. The value and `q_scale` biases are both
larger in absolute mean relative error at `N=10000`:

| Scope | Controls | Mean value relative error | Mean `q_scale` relative error |
| --- | --- | ---: | ---: |
| `N=5000` | `(20,5)` | `+0.1482%` | `-9.91%` |
| `N=10000` | `(20,8)` | `+0.1735%` | `-15.90%` |

This is not evidence that increasing `N` is harmful in general. It is a
fresh, independent 16-seed result showing that the expected monotone bias
improvement did not occur at this `T=50` scope. Several score-coordinate
standard deviations decreased, but the key mean biases did not.

## Claimed And Computed Quantities

| Item | Exact classification |
| --- | --- |
| Claimed target | Exact differentiated Kalman log likelihood and five-coordinate HMC score on the float32-rounded production observation prefix at `T=50`. |
| Candidate quantity | Canonical Contract E--Chol finite-particle value and total derivative, with independently tuned streaming OT controls for `T=50,N=10000,K=2500`, `4 x 4` blocks. |
| Equality verdict | Different relative to the frozen certification target; the simultaneous screen fails. |
| Supporting artifact | `aggregate.json`, SHA-256 `edb67e93523d91b4ececebc9354aa962e3259265fbc180bbc7edacb88173739a`. |
| Not proved | A `1/N` rate, asymptotic behavior, method superiority, nonlinear validity, HMC/posterior readiness, or universal controls. |

## Frozen Scope And Tuning

The run used float32, TF32, GPU/XLA, singleton seed microbatches, the exact
chunk policy `dpf_transport_exact_divisor_cap3000_v1`, and `K=2500` with a
`4 x 4` block grid. Tuning was blind to Kalman and used calibration seeds
`82400..82407`, validation seeds `82408..82415`, and untouched claim seeds
`82420..82435`.

The warm-start candidate `(20,5)` failed validation with
`E_row=0.0112653`. The first direct-gate-valid pair was `(20,8)`:

| Partition | Maximum `TV_col` | Maximum `E_row` | Status |
| --- | ---: | ---: | --- |
| Calibration | `3.0496e-7` | `5.3406e-5` | PASS |
| Validation | `3.6288e-7` | `0.0027720` | PASS |

No Kalman values or scores were used in selecting the pair. The tuning history,
candidate hash, selected-control hash, claim artifact, and run manifest all
passed exact binding checks.

## Untouched Claim

The claim itself passed every hard engineering gate:

- `TV_col=3.0400e-7` and `E_row=2.6166e-5`;
- finite value and total score;
- bitwise replay;
- valid chart and reset;
- exact work accounting;
- `StatelessWhile` with no Python horizon unroll;
- exact scope/chunk/device identity; and
- singleton microbatch size one for all 16 claim seeds.

The peak TensorFlow allocator value was `400,609,536` bytes (about `382.1`
MiB), with `20,082,432` bytes current at artifact time. The claim wall time
was `4096.77 s` and the full campaign wall time was `8412.73 s`.

## Kalman Screen

The screen uses two-sided Bonferroni-Student simultaneous 95% intervals over
six outputs, critical value `3.036283222821165`, value margin `0.001`, and
score margin `0.05`.

| Output | Mean relative error | Simultaneous interval | Required region | Status |
| --- | ---: | --- | --- | --- |
| Value | `+0.1735%` | `[+0.1502%,+0.1968%]` | `[-0.1%,+0.1%]` | FAIL, wholly outside |
| `phi1` | `-0.2881%` | `[-1.4388%,+0.8627%]` | `[-5%,+5%]` | contained |
| `phi2` | `+0.0514%` | `[-1.0487%,+1.1516%]` | `[-5%,+5%]` | contained |
| `phi3` | `+0.3706%` | `[-18.3272%,+19.0684%]` | `[-5%,+5%]` | inconclusive |
| `q_scale` | `-15.8989%` | `[-22.0036%,-9.7941%]` | `[-5%,+5%]` | FAIL, wholly outside |
| `r_scale` | `-3.6106%` | `[-5.2386%,-1.9826%]` | `[-5%,+5%]` | inconclusive |

The exact Kalman value was `-136.0759746346`; the candidate mean was
`-135.8398780823`, an absolute error of `0.2360965523`. The candidate mean
HMC score was `[2.7158171,-2.6735765,0.2663058,-0.7776950,1.8886773]`,
compared with Kalman `[2.7236629,-2.6749522,0.2653226,-0.6710117,1.9594240]`.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept exact `N=10000` route as engineering-valid | PASS | Kalman value and `q_scale` hard veto; `phi3`/`r_scale` inconclusive | Long-horizon finite-particle score cancellation | Preserve as negative evidence and run the predeclared time-local score decomposition | No Kalman correctness or default readiness |
| Claim larger `N` fixes the issue | FAIL | `N=10000` value and `q_scale` intervals remain outside | Independent seeds and cancellation-sensitive score | Diagnose initial/transition/weight/carried-weight/Contract-E reset components on identical streams | No monotone `1/N` trend |
| Transfer controls to another scope | VETO | Tuning scope is exact and `(20,8)` differs from `N=5000` `(20,5)` | Scope-specific numerical sensitivity | Tune every new horizon/model/particle scope independently | No universal control setting |
| Continue nonlinear testing | BLOCKED | LGSSM prerequisite not met | Long-horizon score mismatch remains unresolved | Do not launch nonlinear transfer yet | No nonlinear failure claim |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto evidence | Value interval wholly outside; `q_scale` interval wholly outside. |
| Viable candidates | The finite `N=10000` program is engineering-valid for its exact scope. |
| Statistically supported ranking | None across `N=5000` and `N=10000`; seeds are independent and no paired cross-`N` analysis was declared. |
| Descriptive-only differences | `N=10000` has lower seed SD for value, `phi1`, `phi2`, `phi3`, `q_scale`, and `r_scale`, but larger absolute mean value, `q_scale`, and `r_scale` errors. |
| Default readiness | No new default and no HMC-facing status. |
| Next evidence needed | Same-observation/same-stream time-local score decomposition and same-scalar derivative checks for active Contract E, no-reset recursion, and Kalman increments. |

## Post-Run Red Team

The strongest alternative explanation for the worse `N=10000` mean is ordinary
between-seed variation or cancellation-sensitive finite-particle error rather
than a monotone particle-count effect. The two scopes use independent seed
blocks, so the cross-`N` mean comparison is descriptive only. The strongest
evidence against a harness failure is that all direct gates, replay, exact work,
scope identity, and Kalman-independent tuning checks passed.

The result would be overturned as a scientific diagnosis if the time-local
decomposition identifies a target or reset implementation mismatch, or if a
predeclared paired common-random-number study demonstrates a supported
cross-`N` improvement. Neither exists yet.

## Run Manifest

| Field | Value |
| --- | --- |
| Campaign artifact | `docs/benchmarks/artifacts/lgssm_n10000_tuned_kalman_20260720/attempt01/campaign_result.json` |
| Campaign SHA-256 | `e9e155b8b15408114d5395964f60a34b59042f9f81f4f7ac88c0d21f112f7317` |
| Claim artifact SHA-256 | `5fdcb22cd37d885177cccfcf3d36e8fb4822bd3c080bde043b47519609de5aba` |
| Selected pair SHA-256 | `d4cdd159e75a4aed3cff69e379bc5ac18ab200b6896777bb0badec294faa0e1c` |
| Manifest SHA-256 | `e7f13f47d09383c7d0a0e880c74ae02b4e6d499d5eb64a05043c721dd3d1a30e` |
| Git commit | `9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b` plus preserved dirty worktree |
| Environment | conda `tf-gpu`, TensorFlow `2.19.1`, Python `3.11.14` |
| Device policy | GPU, float32, TF32, XLA, fixed logical 8192 MiB limit |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |

# LGSSM N=10000 Single-Seed Kalman Diagnostic Result

Date: 2026-07-20
Status: `ENGINEERING_PASS_MIXED_ONE_SEED_NO_BIAS_IMPROVEMENT`
Plan:
`docs/plans/bayesfilter-lgssm-n10000-single-seed-kalman-diagnostic-plan-2026-07-20.md`

## Outcome

The canonical `T=50,N=10000,K=2500` singleton route is easy to run on the
current GPU. It compiled and completed a cold execution plus bitwise replay in
`314.66 s`, with a peak TensorFlow allocator value of `400,607,744` bytes
(`382.05 MiB`). GPU/TF32/XLA, exact `4 x 4` chunks, finite value/total score,
chart, reset, marginals, replay, graph structure, and work accounting all
passed.

For paired seed `82220`, increasing from `N=5000` to `N=10000` did **not**
consistently move the result closer to Kalman. It was closer on `phi1`, `phi3`,
and `r_scale`, but worse on value, `phi2`, and especially `q_scale`.
`q_scale` relative error changed from `+5.02%` at `N=5000` to `+22.63%` at
`N=10000`.

This is one realization with controls `(20,5)` transferred from the `N=5000`
scope. It is not a bias estimate and cannot establish a particle-count trend.

## Paired Comparison

| Output | Kalman | N=5000 | N=10000 | N=5000 rel. error | N=10000 rel. error | N=10000 closer? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Value | `-136.075975` | `-135.894852` | `-135.838196` | `-0.1331%` | `-0.1747%` | No |
| `phi1` | `5.655446` | `5.515753` | `5.538677` | `-2.4701%` | `-2.0647%` | Yes |
| `phi2` | `-3.835057` | `-3.851893` | `-3.862609` | `+0.4390%` | `+0.7184%` | No |
| `phi3` | `0.302362` | `0.250258` | `0.252287` | `-17.2323%` | `-16.5612%` | Yes |
| `q_scale` | `-1.917176` | `-2.013382` | `-2.351036` | `+5.0181%` | `+22.6301%` | No |
| `r_scale` | `4.354276` | `4.440805` | `4.275992` | `+1.9872%` | `-1.7978%` | Yes |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Record singleton `N=10000` as engineering-feasible | PASS | No engineering veto | Current external load affects timing only | Reuse `K=2500`, singleton as a resource witness | No default or HMC readiness |
| Do not claim lower bias at `N=10000` | Mixed; `q_scale` and value worsened | One-seed/statistical-evidence veto | Monte Carlo variation and cross-scope warm-start controls | Independently tune `N=10000`, then run fresh multi-seed evidence if the particle hypothesis remains worth testing | No monotonic convergence or bias-rate claim |
| Preserve time-local decomposition as the more discriminating diagnosis | Prior `N=5000` screen failed and this rung is mixed | None for diagnostic continuation | Structural versus finite-particle score error | Localize timewise active/no-reset/Kalman score contributions before another expensive ladder | No rejection of the LEDH direction |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | All `N=10000` engineering gates passed. |
| Statistically supported ranking | None; one paired seed cannot rank particle counts or estimate bias. |
| Descriptive-only differences | Three outputs closer and three worse; `q_scale` changed from `+5.02%` to `+22.63%` relative error. |
| Default readiness | Not established; controls are not tuned for `N=10000`. |
| Next evidence needed | Scope-specific tuning plus fresh multi-seed claim evidence, or first the planned time-local score decomposition. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `9fd0b97fccd8ba216407eb8ff0a727bdc5a2709b` |
| Command | Recorded verbatim in `node.json` |
| Environment | `tf-gpu`, TensorFlow `2.19.1` |
| GPU | NVIDIA GeForce RTX 4080 SUPER; fixed logical limit 8192 MiB |
| Backend | float32, TF32 enabled, XLA JIT |
| Shape | `T=50,N=10000,K=2500`, `4 x 4`, singleton seed `82220` |
| Controls | `(sinkhorn_steps=20,balance_steps=5)`, cross-scope warm start |
| Wall time | `314.656 s` |
| Peak allocator | `400,607,744` bytes |
| Node | `docs/benchmarks/artifacts/lgssm_n10000_single_seed_kalman_20260720/attempt01/node.json` |
| Comparison | `docs/benchmarks/artifacts/lgssm_n10000_single_seed_kalman_20260720/attempt01/comparison.json` |
| Node SHA-256 | `43e670dd9cc156391d2329661b17a2406cd69fe93247fd884fe72203384f8819` |
| Comparison SHA-256 | `49b1d57d1700be15c81058782602e447e9f2222a022ace73c47b7dc51766f78b` |

## Post-Run Red Team

The strongest alternative explanation for the poor `q_scale` result is that
`(20,5)` is not tuned for the new `N=10000,4 x 4` scope, not that more particles
intrinsically worsen the method. Conversely, selectively emphasizing the three
coordinates that improved would be misleading because value, `phi2`, and
`q_scale` worsened on the same paired realization. Independent tuning and
multi-seed uncertainty are required before making a bias statement.


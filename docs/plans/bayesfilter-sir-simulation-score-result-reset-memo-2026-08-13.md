# Austria-SIR Simulation Score Result Memo

Date: 2026-08-13  
Plan: `docs/plans/bayesfilter-sir-simulation-score-plan-2026-08-13.md`  
Artifact: `docs/benchmarks/artifacts/sir_simulation_score_20260813/`

Status correction, 2026-08-13: `OFF_TARGET_WRONG_METHOD`. This result is only
evidence that unconditional latent-path importance sampling collapsed. It did
not execute the requested observation-only classifier likelihood-ratio method.

## Result

The generic Fisher-identity simulation-score utility and Austria-SIR adapter
executed on the trusted `tftwogpu` GPU/XLA path. All `24` merged replicate rows
were finite, but all `24` failed the predeclared reliability screen:
effective-sample-fraction `< 0.01`.

| Horizon | Score component 0 mean (SE) | Component 1 mean (SE) | Component 2 mean (SE) | ESS fraction range | Maximum normalized weight range |
|---|---:|---:|---:|---:|---:|
| `T=20` | `-35.05 (50.09)` | `27.18 (4.95)` | `12.48 (0.56)` | `0.000467..0.001690` | `0.158..0.494` |
| `T=40` | `-150.13 (45.59)` | `-49.10 (38.53)` | `9.98 (0.37)` | `0.000438..0.000873` | `0.280..0.485` |
| `T=50` | `-98.28 (40.75)` | `-66.13 (8.97)` | `5.06 (0.32)` | `0.000679..0.002038` | `0.133..0.390` |

The estimates are therefore finite descriptive outputs from a severely
degenerate prior-simulation importance sampler, not usable observed-data score
references. The high-dimensional complete-path likelihood makes prior paths
rarely representative of the fixed observation, even at `T=20`.

## Attempt Provenance

- `smoke_attempt01`: failed closed because memory growth was configured after GPU initialization.
- `smoke_attempt02`: failed closed on a positional `tf.random.stateless_normal` dtype argument.
- `smoke_attempt03`: failed in the compiled path because the eager-only `scaled_model` constructor was traced.
- `smoke_attempt04`: compiled GPU/XLA estimator completed; artifact serialization hit a tuple/list bug.
- `smoke_attempt05`: clean GPU/XLA smoke completion.
- `attempt01`: all `T=20/40` rows and `T=50` rows `0,1` plus later partial rows; no terminal summary.
- `attempt02`: fresh continuation for `T=50` replicate indices `2..7`; terminal completion.

The merged result is `docs/benchmarks/artifacts/sir_simulation_score_20260813/merged_result_attempt01_attempt02.json`.

## Decision Table

| Decision | Status | Reason |
|---|---|---|
| Hard finite/shape screen | Pass | All `24` rows finite with valid shapes. |
| Simulation-reference reliability | Fail | Every row has ESS fraction below `0.01`. |
| Statistically supported algorithm ranking | Not available | The reference estimator is vetoed; no comparison is valid. |
| Descriptive score differences | Not interpretable as oracle agreement | Point estimates are dominated by a few paths and replicate spread. |
| Default/HMC/leaderboard readiness | Not claimed | Outside scope and unsupported. |
| Next justified action | Repair the proposal distribution | Use conditional/bridge simulation or classifier/regression reference, with ESS/calibration gates unchanged. |

## Reset For Next Agent

The generic function is reusable, but raw prior simulation is not an adequate
reference for this observed SIR path. Do not increase path count mechanically:
with maximum normalized weights up to `0.49`, millions of additional prior paths
would be required before the effective sample size becomes useful.

The next smallest discriminating route is a conditional simulation reference:
either (a) a bridge/particle proposal targeting the fixed observation with
incremental importance weights, or (b) the independent local likelihood-ratio
classifier with perturbation-size calibration. The latter must report held-out
log-loss/calibration and estimate the score at the actual observation path;
chance/perfect separation and perturbation instability are vetoes. Any repaired
route needs a fresh artifact root and must retain this failed prior-simulation
result as evidence.

No conclusion is authorized about repaired-fixed versus repaired-permutation,
the correctness of the repository standard score, or the scientific model from
this failed reference attempt.

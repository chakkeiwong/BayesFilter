# Austria-SIR Simulation Score Plan

Date: 2026-08-13  
Status: `OFF_TARGET_WRONG_METHOD_NOT_REQUEST_COMPLETION`

Supersession note, 2026-08-13: this plan substituted latent-path Fisher
importance sampling for the requested observation-only classifier likelihood
ratio. It is preserved as a negative diagnostic only and must not be cited as
execution of the classifier-ratio request.

## Research Intent Ledger

| Field | Frozen definition |
|---|---|
| Main question | Can an independent simulation-based Fisher-identity estimator provide a useful reference for the observed-data parameter score at Austria-SIR horizons `T=20`, `T=40`, and `T=50`? |
| Quantity under test | `s_theta(y) = grad_theta log p_theta(y)` for the fixed observed path, evaluated at `theta=(0,0,0)` in the repository's three log-scale coordinates. |
| Mechanism | Simulate latent pre-clipping paths from the model prior at `theta`, evaluate their likelihood under the fixed observed path, and self-normalize complete-data parameter scores. |
| Baseline/comparator | The existing repository standard pairwise backward-filtering score emitted by the GenUT campaign, read only as a descriptive comparator. |
| Primary pass criterion | For each horizon, the simulation estimator is finite, its independent replicate estimates are finite, and its effective sample size (ESS) and replicate dispersion are reported. No numerical score-agreement threshold is promoted before seeing the diagnostic. |
| Hard vetoes | Non-finite path, likelihood, score, estimate, invalid shape, invalid seed partition, or severe importance-weight collapse (`ESS / paths < 0.01`), which makes the fixed-path estimate unreliable. |
| Explanatory diagnostics | Weight ESS, log-weight range, normalized-weight maximum, replicate spread, prefix drift, and manual-score versus autodiff-score parity on a tiny independent fixture. |
| What will not be concluded | This is not an exact oracle, proof of particle-score correctness, statistical superiority, HMC readiness, default readiness, or a claim that any algorithm is best. |
| Artifact | `docs/benchmarks/artifacts/sir_simulation_score_20260813/` with manifest, per-replicate records, summary, and result/reset memo. |

## Method

For latent paths `x^(m)` simulated from `p_theta(x)`, compute

`ell_m = log p_theta(y_obs | x^(m))` and `u_m = grad_theta log p_theta(x^(m), y_obs)`.

The Fisher identity gives

`grad_theta log p_theta(y_obs) = E[u_m | y_obs]`.

The estimator is the self-normalized importance average

`sum_m exp(ell_m - logsumexp(ell)) u_m`.

The same weights give the simulated log-marginal estimate
`logsumexp(ell) - log(M)`. The implementation is generic over a path simulator
and a complete-data evaluator; the Austria-SIR adapter is diagnostic-only.

## Data And Seeds

- Model: `latent_preclip_zhao_cui_sir_austria_model()`.
- Parameter: `theta=[0,0,0]`.
- Observed path: source simulator `base_model.simulate(final_time=50, seed=81120)` at the true/base parameter; use observations `y_1:y_T` for each prefix.
- Horizons: `20`, `40`, `50`, paired prefixes of the same generated path.
- Replicates: `8` independent streams per horizon.
- Paths per replicate: `8192` (total `65536` paths per horizon).
- Simulation seeds: `86100 + 100*horizon + replicate_index`.
- No observed path, score output, or particle result is used to tune the estimator.

## Skeptical Audit Before Execution

1. **Wrong target risk:** the path is the latent pre-clipping simulator state and observations are `y_1:y_T`; this matches the existing Austria-SIR callback time order, not the base simulator's `y_0` prefix.
2. **Reuse risk:** the estimator does not call GenUT, LEDH, transport, resampling, or particle ancestry code. Its only model dependency is the declared simulator and complete-data density/score surface.
3. **Proxy risk:** ESS and classifier-style separation are diagnostics, not promotion criteria. A finite estimate with low ESS is rejected as unreliable.
4. **Uncertainty risk:** replicate-level dispersion is retained. Eight replicates are descriptive; no ranking or superiority claim is allowed.
5. **Scale risk:** all three coordinates use the model's declared log-scale parameterization; no physical-scale perturbation or cross-model transfer is introduced.
6. **Compute risk:** the campaign is bounded to `3 * 8 * 8192` simulated path-prefix evaluations, with a fresh artifact root and no overwrite. A non-finite or collapsed pilot stops interpretation and is recorded.

Audit verdict: `PASS_WITH_NONCLAIMS`; execute the bounded implementation and campaign.

## Implementation And Verification

1. Add `bayesfilter/highdim/simulation_score_tf.py` with a generic batched Fisher-identity estimator and typed result payload.
2. Add an Austria-SIR adapter/runner under `docs/benchmarks/` that uses the fixed paired path and writes structured artifacts.
3. Add focused tests for normalized weights, ESS/collapse handling, shape/finite vetoes, and manual complete-score versus autodiff parity on a tiny path set.
4. Run focused tests, then the GPU/XLA campaign in the `tftwogpu` environment with memory growth enabled. If GPU execution is unavailable after an escalated probe, run no silent CPU substitute.
5. Write a result/reset memo separating hard veto status, descriptive estimates, uncertainty, and the next justified action.

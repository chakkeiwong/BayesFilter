# Austria SIR Pairwise-Moment GenUT Reset Memo

Date: 2026-07-30

## Current State

Pairwise co-skewness/co-kurtosis correction is implemented as an opt-in
extension of `higher_moment_shape_jvp`. Existing callers remain diagonal-only
because all new controls default to zero.

Implemented moment families in whitened coordinates:

```text
ordered co-skewness:   E[z_i^2 z_j], i != j
pair co-kurtosis:      E[z_i^2 z_j^2], i != j
```

The correction uses matrix contractions, projects its update out of the
mean/covariance tangent space, and restandardizes every iteration. The full
executed map has a manual JVP.

## Canonical Trial

Plan:

`docs/plans/bayesfilter-austria-sir-pairwise-moment-genut-score-trial-plan-2026-07-30.md`

Result:

`docs/plans/bayesfilter-austria-sir-pairwise-moment-genut-score-trial-result-2026-07-30.md`

Artifact:

`docs/benchmarks/artifacts/austria_sir_pairwise_moment_genut_score_20260730/attempt01/result.json`

GPU smoke:

`docs/benchmarks/artifacts/austria_sir_pairwise_moment_genut_score_20260730/smoke_attempt01/result.json`

## Verdict

Status: `PAIRWISE_SCORE_VARIANCE_PROMOTION_FAIL`.

The selected controls were pairwise steps `4`, strength `0.02`, floor `1e-5`
on top of the July diagonal/OT controls. All score SDs fell dramatically and
the aggregate variance-ratio bootstrap interval was below one. Promotion
failed because the mean finite value shifted by `1.260`, exceeding the
predeclared one-baseline-SE gate. The candidate is promising and opt-in, not a
new default.

The candidate score means/SDs were:

```text
log_kappa_scale:                 -16.304 / 36.512
log_nu_scale:                   -109.627 / 17.762
log_observation_noise_scale:      15.907 / 20.635
```

SGQF lies inside the latter two 95% intervals but outside `log_kappa_scale`.
Do not claim exact score agreement.

## Code And Tests

Touched execution files:

- `bayesfilter/highdim/higher_moment_contract_e.py`;
- `bayesfilter/highdim/cubature_genut_filter.py`;
- `docs/benchmarks/run_moment_retuned_genut_whole_leaderboard.py` only for
  backward-compatible zero-default control/diagnostic plumbing; and
- `docs/benchmarks/run_austria_sir_pairwise_moment_genut_score_trial.py`.

Focused tests:

```text
tests/highdim/test_higher_moment_contract_e.py
tests/highdim/test_cubature_genut_candidate.py
tests/highdim/test_cubature_genut_adapters.py
tests/highdim/test_cubature_genut_filter.py
```

Result: `34 passed`; CPU-only was deliberate with CUDA hidden. The nonzero
pairwise GPU/XLA path passed the trusted smoke and final campaign.

## Next Smallest Valid Step

Do not simply increase pairwise strength. Run a fresh, disjoint tradeoff study
around steps `4`, strengths approximately `0.01-0.02`, with explicit value
stability and `log_kappa` diagnostics. A stronger reference could be the
independent online SIR score teacher if its target/event order can be matched
and its cost is bounded. SGQF remains diagnostic, not an oracle.

Do not overwrite July 23 or July 30 artifacts, do not promote the pairwise path
as default, and do not interpret lower variance as proof of lower bias.

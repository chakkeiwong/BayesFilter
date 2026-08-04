# Higher-Moment GenUT Retuning Reset Memo

Date: 2026-07-23

## State To Preserve

The current experimental route remains:

`transition -> likelihood weighting -> entropic OT -> Contract E reset ->
diagonal higher-moment correction -> exact mean/covariance rewhitening ->
equal-weight continuation`.

No new sigma-point or assumed-density filter was introduced. The canonical
Contract E route was not changed.

## Code Changes

- `bayesfilter/highdim/higher_moment_contract_e.py` now reports the actual
  target-versus-output skewness/kurtosis residual even when correction steps
  are zero. The zero-step particles and tangents remain exactly unchanged.
- `bayesfilter/highdim/cubature_genut_filter.py` records normalized moment
  objective and correction displacement diagnostics.
- `docs/benchmarks/run_higher_moment_genut_retuning_trial.py` is a separate
  oracle-free retuning harness. It does not overwrite prior artifacts.
- Focused tests pass: `24 passed` in the CPU-hidden test suite.

## Executed Trial

Claim artifact:
`docs/benchmarks/artifacts/higher_moment_genut_retuning_20260723/attempt03/result.json`

The trial used FP32, TF32, XLA, GPU memory growth, `N>1000`, the existing
model fixtures, and 16 untouched claim seeds on LGSSM, fresh transformed SV,
and predator-prey. Two earlier attempts are preserved as harness failures:

- `attempt01`: missing wrapper import;
- `attempt02`: unsupported route-identity design-family label during finalization.

Neither failed attempt is scientific evidence.

## Selected Controls And Verdict

All scopes selected the grid boundary:

```text
epsilon=2, sinkhorn_steps=8, balance_steps=8, ridge=1e-5,
higher_moment_correction_steps=4, higher_moment_strength=0.2,
higher_moment_floor=1e-5
```

Mean maximum skewness and kurtosis residuals decreased on every claim scope.
This supports the narrow diagnostic claim that stronger existing correction
controls improve the selected diagonal moment match. It does not establish
likelihood or score improvement. The boundary hit means the controls are not
promoted as defaults.

## Reboot Guidance

Do not relabel this result as a GenUT likelihood or score win. Do not use the
new boundary controls as an unreviewed default. If work resumes, the next
smallest valid trial is a fresh boundary search with controls beyond `0.2` and
`4` steps, plus a numeric predeclared variance/displacement veto. Keep oracle
comparisons post-run only. The Austria SIR convention remains the parameterized
score-capable `J=9`, latent `d=18` route; no GenUT SIR score run exists yet.

## Nonclaims

No exactness, unbiasedness, score correctness, HMC readiness, leaderboard
promotion, or NAWM result follows from this memo.

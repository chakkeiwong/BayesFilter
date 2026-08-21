# SSL-LSTM q=20 NeuTra R-hat trajectory diagnostic reset memo (2026-08-21)

This memo supersedes
`docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-continuation-reset-memo-2026-08-20.md`
for the narrow question of whether the rejected seed-2, `L=5` verification was
merely too short. The earlier memo remains binding historical provenance for
the `r2` campaign.

## Terminal state

- The diagnostic is terminal with `DIAGNOSTIC_SCREEN_FAILED_AT_4000` and
  `DOUBLING_TO_4000_INSUFFICIENT_FOR_DECLARED_SCREEN`.
- Cumulative observation-weight R-hat fell from `1.0875996310350042` at 2,000
  draws to `1.0489500982500948` at 4,000, but remained above `1.01`.
- The endpoint sign-indicator R-hat was `1.0761213959625822`. The final
  trailing-1,000 observation-weight and sign R-hat were `1.0595875339465644`
  and `1.096654664213271`.
- The trajectory was not monotone: five adjacent cumulative decreases and two
  increases. Recent-window observation-weight R-hat ranged from
  `1.0595875339465644` to `1.3671718611526078` across eligible checkpoints.
- Every chain visited both signs and transitioned. Endpoint transition counts
  were `[33,49,29,46]`; final-window counts were `[13,19,11,16]`.
- All run-validity checks passed, including finite target/score/log acceptance,
  target status, all-chain movement, memory growth, raw archive, and artifact
  integrity. Native divergence telemetry was unavailable and is not zero.
- No candidate was reinstated. No canonical sequential HMC, ESS, posterior,
  mode-weight, predictive, scientific, or default-readiness result exists.

## Binding files

- Plan:
  `docs/plans/bayesfilter-ssl-lstm-q20-neutra-rhat-trajectory-diagnostic-plan-2026-08-21.md`
- Result note:
  `docs/plans/bayesfilter-ssl-lstm-q20-neutra-rhat-trajectory-diagnostic-result-2026-08-21.md`
- Runner:
  `docs/benchmarks/run_ssl_lstm_q20_neutra_rhat_trajectory_diagnostic_2026_08_21.py`
- Terminal result:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01/result.json`
- Checkpoint diagnostics:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01/checkpoint-diagnostics.json`
- Raw archive receipt:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01/raw-archive.json`
- Terminal manifest:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01/manifest.json`
- Terminal inventory:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory-retry-01/artifact-hashes.json`
- Preserved first launcher failure:
  `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r3-rhat-trajectory/result.json`

Launch-bound identities are plan SHA-256
`b1d72d761cf17b58b334541bcc181429b5754b9c83ec60d8a082fd02391cb83b`
and runner SHA-256
`b4d83c0c49a215e4d031c1cd60016ca363b20a69a13dec20bddbb6b26c746c7a`.
The terminal result, manifest, and inventory SHA-256 values are respectively
`8899fa82f988ffd86e8c68076be77bbe4e2d6c988912e526ca368ccd8ab2c200`,
`84e0d1052660668a63e57d03d4b4604505e50bf7e8c9dcdd93412063b9d521cd`,
and `038d7d59095d69e4d6898df2d123b0bc410cc70655ae70a48044cd8ea2e3ad33`.

## Resume boundary

Both `r3-rhat-trajectory` and `r3-rhat-trajectory-retry-01` are terminal,
immutable evidence roots. Do not overwrite, resume, append to, or reinterpret
them as posterior storage. The plan's launcher and GPU-attempt schedule is
exhausted. The remaining `15705.352527493014 s` of the user's 18-hour grant is
unused budget, not authorization for another run.

The direct answer to carry forward is:

1. More samples lowered cumulative R-hat for the slow coordinate.
2. Four thousand draws remained insufficient for both continuous and sign
   R-hat.
3. The chains were not sign-locked; they crossed repeatedly.
4. Recent-window volatility means sample shortage alone is unsupported as the
   root cause.

## Next justified action

Do not merely lengthen this exact chain again. A future serious continuation
should use a new plan and output root to investigate target-specific
observation-weight geometry and kernel repair. The smallest useful precursor is
diagnostic-only localization on the preserved raw archive, such as sign dwell
lengths and coordinate autocorrelation, followed by a predeclared mass/path or
transport repair with disjoint tuning and validation. Any passing candidate
must still enter `bayesfilter_neutra_sequential_hmc_v1` and pass retained
R-hat, ESS, status, movement, energy, and direct cross-sign gates before
posterior or predictive use.

## Do not conclude or reuse

- Do not call the 4,000 diagnostic draws converged posterior samples.
- Do not say R-hat decreased monotonically or extrapolate a passing draw count.
- Do not say sign traversal establishes adequate mixing.
- Do not claim sample count is the sole problem.
- Do not claim native divergence count zero; it was unavailable.
- Do not reinstate seed 2, `L=5`, rank it against another candidate, or use it
  for predictive work.
- Do not reject the SSL-LSTM target, frozen transport family, NeuTra, or the
  research direction from this scoped failure.

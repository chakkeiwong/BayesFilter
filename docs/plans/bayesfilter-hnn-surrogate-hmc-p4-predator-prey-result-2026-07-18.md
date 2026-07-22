# P4 Result: Predator-Prey UKF And SGQF

Decision: `P4_PASSED_TWO_INDEPENDENT_ONE_SEED_LEARNED_FORCE_VALIDITY_CONFIRMATIONS`.

The corrected position-only learned-force kernel passed separately on the named
PP-UKF and PP-SGQF deterministic filter posteriors. The zero-residual control
also passed both cells. These are viability results on one frozen fixture and
one seed per cell; no between-arm or between-filter ranking is statistically
supported.

## Evidence

| Cell / arm | eps, L | Acceptance | Warm-up / retained per chain | max modern R-hat | min bulk / tail ESS | min truth-tail | Plain-HMC mean agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| PP-UKF zero | 0.8, 10 | 0.7987 | 2000 / 1000 | 1.00694 | 2573.8 / 1233.0 | 0.2062 | pass |
| PP-UKF learned | 0.2, 10 | 0.9698 | 2000 / 1000 | 1.00534 | 5187.8 / 2197.5 | 0.2062 | pass |
| PP-SGQF zero | 0.8, 10 | 0.8320 | 2000 / 1000 | 1.00551 | 3332.5 / 1760.0 | 0.2232 | pass |
| PP-SGQF learned | 0.2, 10 | 0.9699 | 2000 / 1000 | 1.00179 | 6679.8 / 3186.9 | 0.2177 | pass |

The full energy identity reconstructed exactly in all four arms. Endpoint
parity error was `2.842170943040401e-14` for PP-UKF and exactly zero for
PP-SGQF. Each endpoint uses the separate value-only scalar filter route, not an
unused filtering gradient. Training used 2,048 preserved target-matched latent
positions and 1,024 disjoint heldout positions; the selected per-cell recipe
was two width-24 layers, learning rate 0.005, batch size 256. Its heldout
standardized force RMSE was 0.1340 (UKF) and 0.1574 (SGQF). Those losses are
nomination evidence only.

Primary artifacts:

- PP-UKF: `docs/plans/artifacts/corrected-neural-force-hmc-20260717/phase-p4/PP-UKF/attempt-01-20260717T165000Z/result.json`, SHA-256 `4acc4435f2a3487500bfa9713f93b2d598c2e2b40807b59d873bec6c6bed8079`.
- PP-SGQF: `docs/plans/artifacts/corrected-neural-force-hmc-20260717/phase-p4/PP-SGQF/attempt-01-20260717T171500Z/result.json`, SHA-256 `865a93fbe77640826f9b365bae17008252fe3bcfd8f3169682888b8a6d26fecf`.

## Repairs And Review

PP-UKF smoke attempt 1 failed before scientific execution because the target
module initialized TensorFlow before memory growth was configured. Attempt 2
correctly detected that adding endpoint-only functions changed a whole-module
provenance hash. The repair preserved the historical execution identity bound
to the frozen transport and independently replayed the unchanged mathematical
target, adapter signature, registry, artifact hashes, and scalar parity. Smoke
attempt 3 passed. PP-SGQF passed on its first smoke.

Claude returned the tiny liveness token but timed out twice on bounded
substantive prompts. Under the repository review-proportionality rule this is a
recorded advisory limitation, not a scientific veto. Codex's focused audit and
14 target tests plus 21 kernel/campaign tests passed.

## Inference Status

| Question | Status |
|---|---|
| Hard veto screen | passed for all four arms |
| Statistically supported arm ranking | none |
| Descriptive differences | learned arms had larger observed ESS but smaller tuned step size |
| Default readiness | not established |
| Next evidence | independent SIR-SGQF and structural-UKF P5 cells |

Strongest alternative explanation: both existing NeuTra charts are already so
effective that the learned residual's apparent ESS gain is seed-specific and
not worth its training or tuning complexity. This does not invalidate the
corrected learned-force kernel; it limits the performance claim.


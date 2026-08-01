# P5 Result: SIR-SGQF And Structural-UKF

Decision: `P5_PASSED_TWO_INDEPENDENT_ONE_SEED_LEARNED_FORCE_VALIDITY_CONFIRMATIONS`.

The corrected position-only learned-force kernel passed separately on the
named parameterized SIR fixed-SGQF posterior and Chapter 18b deterministic
structural fixed-UKF posterior. The zero-residual control also passed both
cells. These are one-fixture, one-seed viability results. They do not support
an arm ranking, filter exactness, calibration, or a general model-family
claim.

## Evidence

| Cell / arm | eps, L | Acceptance | Warm-up / retained per chain | max modern R-hat | min bulk / tail ESS | min truth-tail | Plain-HMC mean agreement |
|---|---:|---:|---:|---:|---:|---:|---:|
| SIR-SGQF zero | 0.2, 10 | 0.9770 | 2000 / 1000 | 1.00247 | 6478.7 / 2988.1 | 0.3777 | pass |
| SIR-SGQF learned | 0.8, 10 | 0.8983 | 2000 / 1000 | 1.00055 | 5770.4 / 3518.7 | 0.3697 | pass |
| STR-UKF zero | 0.2, 8 | 0.8308 | 2000 / 1000 | 1.00356 | 2188.3 / 1689.0 | 0.3102 | source-geometry blocked |
| STR-UKF learned | 0.2, 8 | 0.8957 | 2000 / 1000 | 1.00543 | 2203.6 / 2121.4 | 0.2777 | source-geometry blocked |

R-hat is the maximum of rank-normalized split R-hat and folded
rank-normalized split R-hat. Warm-up samples were retained. Every arm passed
the prospective retained thresholds: R-hat at most `1.01`, bulk ESS at least
`1000`, and tail ESS at least `400`. The historical posthoc structural
bulk-ESS threshold of `900` was not used.

The full endpoint energy identity reconstructed with zero error in all four
arms. Endpoint parity error was exactly zero for SIR-SGQF and
`2.842170943040401e-14` for STR-UKF. The structural deterministic completion
remained intact and no artificial process noise was added to `k_t`.

SIR-SGQF passed the admitted three-physical-mean plain-HMC comparison. The
structural raw-coordinate plain-HMC comparator remains honestly blocked by
the source geometry recorded in the preserved comparator artifact. No proxy
comparator was substituted. This does not veto the predeclared structural
truth-tail result, and it prevents a claim of same-target plain-HMC mean
agreement for that cell.

Primary artifacts:

- SIR-SGQF: `docs/plans/artifacts/corrected-neural-force-hmc-20260717/phase-p5/SIR-SGQF/attempt-01-20260718T014000Z/result.json`, SHA-256 `2e4aaf438f3117734dda993ec936370e36f0163a3f6c396f7e61e5ac56438f83`.
- STR-UKF: `docs/plans/artifacts/corrected-neural-force-hmc-20260717/phase-p5/STR-UKF/attempt-01-20260718T015500Z/result.json`, SHA-256 `8d59d0831b679a9650346f7018d0edcdf1d84d71de25a9c1b4f4aad3117fe694`.

P5 used `0.643987` GPU wall-hours including both smoke and serious attempts.
No P5 CPU-only run was charged. All runs recorded TensorFlow 2.19.1, TFP
0.25.0, GPU memory growth, GPU/XLA, TF32, and the managed-session trust basis.

## Repairs And Review

The phase added independent value-only endpoint programs for SIR-SGQF and
STR-UKF and checked them against the complete transformed value/score targets.
Adding these functions changed whole-module source hashes without changing the
mathematical target or frozen transport. The execution therefore preserved
the historical identity bound to each transport, replayed target and adapter
signatures independently, and recorded the provenance refresh rather than
falsely relabelling a frozen artifact.

Four focused SIR/structural parity, invariant, and loop-policy tests passed,
as did the shared kernel/campaign regressions and both GPU/XLA smokes. Claude
answered the liveness probe but timed out on two earlier bounded substantive
reviews. This is an advisory review limitation under the repository's
proportional-review policy, not a scientific veto; the local evidence was
audited directly.

## Decision Table

| Decision field | Status |
|---|---|
| Primary corrected-kernel criterion | passed independently in both cells |
| Veto diagnostics | no target, energy, convergence, truth-tail, or structural veto fired |
| Main uncertainty | one fixture and one seed per cell; structural raw-HMC agreement unavailable |
| Next justified action | proceed to P6 target-by-target requalification |
| Not concluded | superiority, calibration, filter exactness, broad readiness, or DSGE validity |

## Inference Status

| Question | Status |
|---|---|
| Hard veto screen | passed for all four arms |
| Statistically supported arm ranking | none |
| Descriptive-only differences | acceptance, ESS, loss, and runtime differences between arms |
| Default readiness | not established |
| Next evidence | exact-source P6 requalification or an honest cell-local blocker |

Strongest alternative explanation: the frozen NeuTra charts already remove
enough geometry that the residual learner adds no reproducible computational
benefit. That explanation does not invalidate the corrected kernel, but it
limits the scientific result to target validity and one-seed viability.


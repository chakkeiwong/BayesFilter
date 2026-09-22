# A09 skeptical plan review

Reviewer: Codex, executor self-review; no independent review claimed.
Reviewed before implementation/numerical execution on 2026-09-16.

## Findings and repairs

| Issue examined | Finding and disposition |
|---|---|
| Wrong baseline | A08 generic-start and hybrid are not standalone SGQF-start. A09 constructs a new frozen warm baseline; generic-start remains a fresh secondary comparator. |
| Proxy metric replacing objective | Fit loss cannot promote filtering. Calibration selects using actual corrected-filter MSE; fresh MSE with uncertainty is primary. Same-target fits explain regression mechanisms only. |
| Changing recursive targets | Each final arm must carry its own retained TT. Comparing their fit losses alone is confounded; added identical incoming-target check at three times. |
| Candidate failure selection bias | Initial draft did not define missing calibration arms. Repaired: common guide-valid sequences, >=2/dimension, candidate must complete all, failed reference stops selection. |
| Audit leakage | Historical A06/A08 data excluded. Choices frozen before controlled audit and independent confirmation; no retuning after either. |
| Regularization mathematics | Shifted proximal formula and KKT checked algebraically by substitution v=c-a. Lambda-zero parity and known one-core solution require executable tests. Gauge remains explicit and fixed. |
| Smoothness assumption | Fixed L1 settings do not make soft threshold/SVD/guide selection globally smooth. This plan does not promise total derivatives or HMC suitability. |
| Source route assumption | New regression classified extension/invention. No source-faithfulness claim or author-method modification is being approved. |
| Wrong density in consumer | Existing joint dispatcher supports PairTTStep. New path must enter that dispatcher for all later times, with first Gaussian and subsequent TT types tested. No hybrid selection function may be called. |
| Reference uncertainty | Existing independent scalar/grid and multiscale-particle screens retained. Paired sequence units and combined evidence MCSE specified. |
| Multiple comparisons | Four primary contrasts, simultaneous bootstrap; independent dimensions get independent bootstrap indices. Duplicates/zero-SE do not imply superiority. |
| Heuristics | Four concrete proposals and conditional observation regimes retained. They restrict promotion, never veto the scientific repair direction or select fit controls. |
| Runtime fairness | Different degrees compile differently. Record build/particle timings; duplicate aliases share real costs; no statistical speed ranking or compilation-free claim. |
| Guide failure | Existing signed-SPD failure remains possible. No covariance alteration smuggled into TT comparison; report coverage separately. |
| Exponential initializer | Degree-4/d=4 is bounded at 390625 coefficients. Larger dimensions/degrees and production promotion excluded. |
| Solver adequacy | Fixed iteration budget is a comparator hypothesis. KKT/descent/condition are mandatory, and failure cannot prove the representation is inadequate. |
| Budget/stops | Six active hours with 14400 numerical seconds, bounded attempts, fresh outputs and recorded failures. No open-ended sweep or repeated approval chain. |

## Verdict

PASS FOR BOUNDED DIAGNOSTIC EXECUTION after the above clarifications.
The plan answers the requested improvement question while leaving guide repair,
scalable initialization and total analytical differentiation open. Engineering
checks are still required before the serious numerical stage. A candidate
failure blocks its promotion, not the remaining planned comparisons.

The weakest planned evidence is the small calibration set and n=12 confirmation
bootstrap. Calibration winners may be noise. Fresh confirmation can leave all
TT nominees statistically indistinguishable; this is a valid campaign result.

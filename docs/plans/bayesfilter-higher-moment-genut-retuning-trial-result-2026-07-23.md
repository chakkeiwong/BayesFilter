# Higher-Moment GenUT Retuning Trial Result

Date: 2026-07-23
Artifact: `docs/benchmarks/artifacts/higher_moment_genut_retuning_20260723/attempt03/result.json`
Plan: `docs/plans/bayesfilter-higher-moment-genut-retuning-trial-plan-2026-07-23.md`

## Outcome

The existing OT + Contract E + higher-moment particle-filter route was
retuned with an oracle-free moment-residual objective. The run was finite and
GPU/XLA/TF32-valid. The selected controls reduced the emitted diagonal
skewness and kurtosis residuals on all tested scopes, including the untouched
16-seed claim rows. This is a real improvement in the declared moment
matching diagnostic, but it does not establish improved likelihood or score
accuracy.

The selected controls hit the top of the tested grid in every scope:

```text
epsilon=2, sinkhorn_steps=8, balance_steps=8, ridge=1e-5,
higher_moment_correction_steps=4, higher_moment_strength=0.2,
higher_moment_floor=1e-5
```

Therefore the trial identifies a promising direction but does not identify an
optimum. Further increases in strength or iteration count were not tested.

## Claim-Scope Diagnostics

Mean maximum residuals over the 16 untouched claim seeds are shown below.
The old row is the previous variance-tuned candidate; the new row is this
moment-retuned candidate.

| Scope | Max skew old | Max skew new | Max kurtosis old | Max kurtosis new |
|---|---:|---:|---:|---:|
| LGSSM `T=2` | 0.3586 | 0.1611 | 0.4717 | 0.1906 |
| LGSSM `T=10` | 0.3665 | 0.1679 | 0.5464 | 0.2306 |
| LGSSM `T=50` | 0.5807 | 0.2403 | 1.1720 | 0.3885 |
| Fresh transformed SV `T=50` | 0.4658 | 0.1894 | 0.5064 | 0.1978 |
| Predator-prey `T=20` | 0.2540 | 0.1161 | 0.4988 | 0.2166 |

Paired 95% intervals for the new-minus-old residuals were entirely below zero
for every scope. For LGSSM `T=50`, for example, the skewness residual change
was `-0.3404` with CI `[-0.3816,-0.2993]`, and the kurtosis residual change was
`-0.7834` with CI `[-0.9675,-0.5993]`.

The maximum normalized shape displacement remained below the declared veto
of `2.0`; the observed maxima were approximately `0.18`, `0.17`, `0.25`,
`0.12`, and `0.18` for the five scopes respectively.

## Value And Score Diagnostics

The retuning changed the finite approximation. On the oracle-backed scopes,
the changes were descriptive and not uniformly favorable:

- LGSSM `T=50`'s error of the 16-seed mean moved from about `0.0119` to
  `0.0055`, but this is not the paired accuracy criterion. The mean paired
  per-seed absolute-error change was `+0.0192`, with 95% CI
  `[0.0026,0.0358]`, so this finite value diagnostic regressed even though the
  aggregate mean happened to move closer. Score-coordinate changes were mixed.
- Fresh transformed SV score means moved to
  `(-0.8393,-2.2697)` from `(-0.8124,-2.2835)`; the dense reference is
  `(-0.8520,-2.2316)`. The changes are not a statistical certification of
  score improvement.
- No exact predator-prey score oracle exists. Its values/scores remain
  descriptive.

The result must not be interpreted as evidence that lower higher-moment
residuals imply lower likelihood or score bias.

## Audit Findings

| Item | Status |
|---|---|
| Existing filtering algorithm changed | `NO` |
| Oracle used for tuning | `NO` |
| Claim seeds used for selection | `NO` |
| FP32/TF32/XLA and memory growth | `PASS` |
| Value/score finite and score-increment identity | `PASS` |
| Moment residual reduction | `PASS descriptively` |
| Candidate optimum found | `NO; grid boundary hit` |
| Secondary variance veto | `NOT ENFORCED` |

The plan mentioned a variance/displacement secondary veto but did not declare a
numeric variance threshold. The implementation enforced the displacement and
hard numerical vetoes and recorded variance as a diagnostic; it did not invent
a threshold after seeing the data. This is a contract limitation, not a hidden
pass.

## Decision

| Decision | Status | Meaning |
|---|---|---|
| Engineering validity | `PASS` | The finite route and artifacts are valid. |
| Oracle-free moment matching | `PASS descriptive` | Residuals fell across all claim scopes. |
| Likelihood/score improvement | `UNSUPPORTED` | No oracle-free criterion proves this, and the available oracle diagnostics are mixed. |
| LGSSM `T=50` value no-regression | `FAIL` | Paired per-seed absolute oracle error increased. |
| Promotion to default/leaderboard/HMC | `NOT READY` | Boundary selection and missing variance veto remain. |

The candidate remains opt-in. The evidence supports a follow-up boundary
search with predeclared stronger controls and an explicit variance/displacement
acceptance rule, not promotion.

## Nonclaims

No exact higher-moment projection, exact nonlinear likelihood, unbiasedness,
exact posterior score, method superiority, default readiness, HMC readiness,
leaderboard promotion, or NAWM conclusion is made.

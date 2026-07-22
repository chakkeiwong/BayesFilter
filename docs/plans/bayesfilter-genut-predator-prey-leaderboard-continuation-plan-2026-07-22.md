# GenUT Predator-Prey Leaderboard Continuation Plan

Date: 2026-07-22

Status: `AUTHORIZED_FOR_BOUNDED_EXECUTION_AFTER_SKEPTICAL_AUDIT`

## Research Intent Ledger

| Field | Frozen decision |
|---|---|
| Main question | On the canonical additive-Gaussian predator-prey `T=20` target, does scope-tuned positive Gaussian GenUT approach an independently refined observed-data likelihood value as `N` increases, while its hand-derived recursive score remains numerically valid and stable? |
| Candidate | Non-fused positive Gaussian GenUT residual, OT barycentric update, and Cholesky restoration, using the existing finite value program and its recursive forward sensitivity. |
| Canonical claim dataset | Leaderboard row `zhao_cui_predator_prey_T20`, DGP seed `81104`, physical truth `(r,K,a,s,u,v)=(0.6,114,25,0.3,0.5,0.5)`. |
| Value reference | Same-target stateless bootstrap particle filter at `N_ref={65,536,262,144}` with 16 independent seeds. It is an independent stochastic reference, not an exact oracle. |
| Score diagnostics | GenUT recursive-score same-scalar finite-difference audit during tuning; `N`-stability intervals; principal-square-root UKF score as a same-target diagnostic, not truth. The generic fixed-SGQF route is excluded because it transitions before `y0` and is target-mismatched for this row. |
| Expected failure mode | Reset bias may persist with `N`; score variance may mask convergence; controls may transfer poorly from SV/LGSSM; dense `N^2` OT may be slow or memory-bound; the PF reference may not be refined enough. |
| Primary value criterion | The 95% interval for tuned `N=1002` GenUT mean minus the refined PF mean contains zero, and the two PF reference rungs are mutually compatible at 95%. |
| Primary score criterion | Explanatory stability only: for every physical coordinate, the 95% interval for the `N=1002` minus `N=384` GenUT mean score contains zero. This cannot establish score truth. |
| Promotion veto | Nonfinite output; GPU/XLA/TF32 or memory-growth failure; reset or marginal residual above `5e-4`; recursive score versus same-scalar FD error above 5%; invalid repository route identity; claim-data leakage into tuning; or an unrefined PF reference. |
| Continuation veto | Target-law mismatch, timing mismatch, corrupted artifact, unavailable trusted GPU, allocator peak above 14 GiB, wall time above 30 minutes, or two failed campaign attempts. |
| Repair trigger | Local compilation, serialization, shape, identity, or resource failure that leaves the scientific contract unchanged. |
| Not concluded | No exact score, unbiasedness, superiority, HMC performance, high-dimensional feasibility, Zhao-Cui source-faithfulness, GenUT default promotion, or completed leaderboard cell. |

## Exact Target

The initial law is `x_0 ~ N((50,5), I)`. Observation `y_0` is assimilated
before any transition. For `t=1,...,19`, the deterministic predator-prey flow
is integrated over `delta=2.0` by RK4 with internal step `0.1`, followed by
additive `N(0,4I)` process noise. Observations have additive `N(0,4I)` noise.
The parameter order is `(r,K,a,s,u,v)`.

The GenUT design is the positive dimension-two Gaussian rule. Its weights are
`(1/3,1/6,1/6,1/6,1/6)`, so every particle count must be divisible by six.
The descriptive ladder is `N={96,384,1002}`. Only `N=1002` is scope-tuned and
eligible for the primary value criterion; lower rungs are convergence
diagnostics and cannot be promoted.

## Offline Tuning

Calibration DGP seeds are `{95101,95102}` and validation seeds are
`{95201,95202}`. They are disjoint from canonical claim seed `81104`.
Particle seeds `{96101,...,96104}` are used only for tuning. The full bounded
control family is:

```text
epsilon        in {2,4}
sinkhorn_steps in {4,8}
ridge          in {1e-6,1e-5}
```

These values are inherited warm-start hypotheses from the prior GenUT
campaigns, not defaults. Their failure mode is under-balanced OT or excessive
regularization. The early diagnostic is the reset/marginal residual plus the
same-scalar score FD audit. Eligible candidates minimize validation value error
against a disjoint `N_ref=65,536` PF mean, then the maximum scaled conditional
value/score variance. Ties prefer calibration performance, lower FD error,
fewer Sinkhorn steps, larger ridge, and smaller epsilon. The claim dataset is
not read until controls freeze.

The tuning score is the recursive analytical score. Finite differences are
used at representative parameter points only to audit that score against the
identical finite scalar; no finite-difference score is emitted by the claim.

## Claim And Uncertainty

The canonical dataset is evaluated with 16 fixed GenUT particle seeds at each
ladder rung. Means, sample standard deviations, standard errors, and two-sided
95% Student intervals are reported for value and every physical score
coordinate. The PF reference uses 16 seeds at each reference rung and reports
the same statistics. Difference intervals conservatively use the 15-degree
Student critical value and the root-sum-square standard error.

The principal-square-root UKF is a same-target deterministic approximation with
an analytical score. It triangulates value and score behavior but is not an
oracle. The generic fixed-SGQF route is not a comparator for this row: its
implementation transitions before every observation, whereas the canonical
target assimilates `y0` before the first transition. The historical
retained-grid Zhao-Cui result is excluded because repository policy classifies
it as diagnostic/historical and not the production fixed-variant route.

## Skeptical Plan Audit

| Risk | Audit result |
|---|---|
| Wrong baseline | Repaired: the historical retained-grid Zhao-Cui value is excluded as an oracle, and generic SGQF is excluded for the `y0` timing mismatch. A same-target refined bootstrap PF is the value reference; UKF is a diagnostic. |
| Proxy promoted | Repaired: FD, residuals, `N`-stability, and UKF agreement cannot establish score truth or default readiness. |
| Missing stop conditions | Repaired: 30-minute, 14-GiB, two-attempt, validity, and reference-refinement vetoes are explicit. |
| Hidden assumption | The refined PF log-likelihood has finite-`N` bias. Two rungs and independent-seed uncertainty are required before using the high rung as reference. |
| Stale controls | Repaired: prior controls seed a full target-specific grid; they are not silently transferred. |
| Environment mismatch | GenUT claims use FP32, TF32, GPU/XLA, and memory growth. The PF is a deliberate FP64 GPU/XLA independent-reference lane. |
| Artifact insufficiency | Raw per-seed values/scores, tuning rows, reference rungs, route identities, source hashes, device/memory data, and decision ledgers are preserved. |
| Misleading success | Even perfect value agreement cannot validate the score or high-dimensional scaling; those remain explicit gaps. |

Audit verdict: `PASS_FOR_BOUNDED_PREDATOR_PREY_VALUE_AND_SCORE_STABILITY_TEST`.

## Budget And Artifacts

- One normal execution plus one localized repair attempt.
- Maximum 8 tuning control combinations, four tuning DGPs, four particle seeds,
  three claim particle rungs, and two PF reference rungs.
- Versioned output root:
  `docs/benchmarks/artifacts/genut_predator_prey_leaderboard_continuation_20260722/attempt01/`.
- Result note:
  `docs/plans/bayesfilter-genut-predator-prey-leaderboard-continuation-result-2026-07-22.md`.

Planned commands:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/highdim/test_cubature_genut_candidate.py \
  tests/highdim/test_genut_predator_prey_leaderboard_continuation.py \
  tests/test_predator_prey_bootstrap_pf_reference.py

TF_FORCE_GPU_ALLOW_GROWTH=true python \
  docs/benchmarks/run_genut_predator_prey_leaderboard_continuation.py \
  --output-root \
  docs/benchmarks/artifacts/genut_predator_prey_leaderboard_continuation_20260722/attempt01
```

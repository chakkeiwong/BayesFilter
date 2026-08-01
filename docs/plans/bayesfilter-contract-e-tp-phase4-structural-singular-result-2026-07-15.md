# Contract E--TP Phase 4 Structural Singular-Dynamics Result

metadata_date: 2026-07-15
phase: 4
status: PASS_PHASE4_STRUCTURAL_SUPPORT
master_plan: `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md`

## Outcome

The experimental structural teacher is implemented in
`bayesfilter/highdim/ledh_contract_e_tp_structural_tf.py`. It consumes the
existing `TFStructuralStateSpace` contract, requires innovation integration and
declared deterministic completion for mixed models, forms only parent-by-
innovation candidates, and calls the model-owned structural transition.

No full-state artificial noise, covariance jitter, Cholesky factor, or NumPy
algorithmic backend is used. The deterministic-completion coordinates are
outputs of the structural map and therefore remain on the declared support.

## Fixture

The fixture has state `(m,k)`, stochastic index `(0,)`, deterministic index
`(1,)`, and one innovation. Its transition is

```text
m_next = rho*m_prev + sigma*epsilon
k_next = alpha*k_prev + beta*tanh(m_next)
```

Features are mass, stochastic state, stochastic square, and the observed
functional `m_next+k_next`. A deterministic positive basic feasible chart is
selected from the 12 teacher candidates at the center; the controlling anchors
are `(1,4,6,11)`, with minimum weight approximately `0.1166` and scaled
condition number approximately `16.60`.

## Evidence

| Check | Result |
| --- | --- |
| Innovation-space metadata and teacher count | Pass: 12 candidates, required completion, innovation integration |
| Pointwise deterministic residual | Pass: zero to `2e-16` |
| Total parameter tangent of completion residual | Pass: zero to `4e-16` |
| Student support | Pass: every retained point is a teacher candidate |
| Selected feature values | Pass: student and teacher targets tie to `2e-15` |
| Projected structural residual | Pass: zero to `2e-16` |
| Structural projected tangent and same-scalar FD | Pass |
| Hidden full-state route | Pass negative control: rejected without innovation integration |
| Backend policy | Pass: owned module has no NumPy, `.numpy()`, jitter, or Cholesky |

Focused structural tests: `6 passed`; combined Phase 1--4 suite: `30 passed`.

## Interpretation

The support theorem is established for this finite structural fixture: a
positive fixed chart can preserve selected structural features and their total
tangents without inventing off-support states. This does not prove SIR, NAWM,
or any large DSGE adapter is ready. It establishes the required interface and
negative control before those models are attempted.

## Decision And Handoff

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close structural phase | support, residual, tangent, and negative-control checks pass | none | SIR-specific structural map and feature capacity | implement Phase 5 model adapters | SIR/NAWM or leaderboard validity |

Phase 4 gate: `PASS`. Phase 5 may add model-owned adapters and preparation
records. It must preserve each registry target, use the existing model
transition/observation APIs, and stop a row independently when its support,
chart, or same-scalar derivative gate fails.

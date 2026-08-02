# SVX-ZC Score Capacity Validation Result

Date: 2026-08-02

Plan:
`docs/plans/bayesfilter-svx-zc-score-capacity-validation-plan-2026-08-02.md`

Artifact:
`docs/plans/artifacts/bayesfilter-svx-zc-score-capacity-validation-20260802/attempt01/`

## Outcome

The run completed all five frozen cells within the bounded CPU-only budget.
The result is `SCORE_VALIDATION_BLOCKED` under the declared all-cell hard gate.
This does not mean that autodiff is failing everywhere. It means three cells
passed the same-scalar derivative check, while two cells did not satisfy the
fixed-branch contract.

## Cell summary

| Capacity `(degree, rank, order)` | Value | Score `[gamma, log_beta]` | FD/branch status |
|---|---:|---|---|
| `(10, 2, 25)` | `-20.4638454838` | `[-0.8902664233, 0.9057606733]` | PASS |
| `(10, 4, 25)` | `-20.4475727494` | `[-0.8650604651, 0.9114701941]` | PASS |
| `(10, 2, 29)` | `-20.4595034015` | `[-0.8895383954, 0.8816570360]` | PASS |
| `(10, 2, 33)` | `-20.4626701753` | `[-0.8891865302, 0.8626110567]` | BLOCKED: one `log_beta` FD plus branch changed |
| `(12, 2, 25)` | `-20.1963095256` | `[-0.8120689471, 0.7271172296]` | BLOCKED: perturbation branch changes and no stable `log_beta` window |

For every PASS cell, values and scores were finite, all 8 finite-difference
rows were valid, the base/plus/minus compatibility hashes were identical, and
both parameters had a stable decreasing finite-difference window. At the
center nominee the selected central differences were
`[-0.8902664412, 0.9057606373]`; the componentwise relative discrepancies
from autodiff were below `2e-8`.

## What caused the two blocks

The route includes the discrete `log_scale_shift_index` sequence in its
fixed-branch compatibility identity. At `(10,2,33)`, the `log_beta + 0.01`
perturbation changes one such branch hash. At `(12,2,25)`, perturbations change
the branch hash for multiple steps and both parameter directions; the
`log_beta` rows consequently have no valid adjacent window even though their
raw central differences approach the autodiff score closely.

These are real fixed-branch admissibility failures for the tested perturbation
ladder. They are not evidence that TensorFlow differentiation is numerically
wrong, and they must not be bypassed by caller-stamping a compatible hash.

## Capacity sensitivity

Relative to `(10,2,25)`, the observed score changes are descriptive:

| Comparison | Gamma relative change | Log-beta relative change |
|---|---:|---:|
| `(10,4,25)` | `2.83%` | `0.63%` |
| `(10,2,29)` | `0.08%` | `2.66%` |
| `(10,2,33)` | `0.12%` | `4.76%` |
| `(12,2,25)` | `8.78%` | `19.72%` |

The value-prefix rule alone therefore does not establish score convergence.
In particular, degree 10 and degree 12 share the first two significant
likelihood digits under the requested value rule, while their score vectors
move substantially.

## Decision tables

| Decision | Primary criterion | Veto status | Next justified action | Not concluded |
|---|---|---|---|---|
| Autodiff at `(10,2,25)` | Same-scalar FD and branch contract | Passed | Use as derivative-consistency evidence for this finite program | Exact score |
| All five-cell score validation | Every cell passes FD/branch gate | Blocked by degree-12 and order-33 cells | Repair or redesign branch-stability criterion before broader score promotion | Score convergence |
| Value nominee | Upstream value-prefix nomination | Unchanged; score does not retroactively invalidate value nomination | Keep as value-only nominee, not score-ready default | Joint value/score capacity |

| Inference status | Verdict |
|---|---|
| Hard veto screen | Three cells passed; two were blocked by branch compatibility/stable-window requirements. |
| Statistically supported ranking | Not applicable; one deterministic frozen-data run. |
| Descriptive-only differences | Score deltas, relative changes, likelihood deltas, residuals, and runtime. |
| Default readiness | Not established. |
| Next evidence needed | A reviewed branch-stability repair or a score-specific capacity protocol that does not silently discard discrete branch changes, followed by fresh validation. |

## Research interpretation

The score implementation is internally consistent for the nominated cell and
two nearby controls that pass the fixed-branch contract. The broader result is
not a score admission: the derivative depends materially on capacity, and the
route's discrete scale-shift branch changes make two higher-capacity tests
inadmissible under the current finite-difference contract. This is a score
validation limitation, not evidence against the filtering research direction.

## Post-run red-team note

- Strongest alternative explanation: the score movement may be driven by the
  finite TT projection and its capacity-dependent fitted density, rather than
  by an autodiff defect. The current artifact cannot distinguish those causes.
- Result that would overturn the center-cell derivative conclusion: a fresh
  rerun with the same `(10,2,25)` scope producing a non-finite score, a
  branch-compatible finite-difference mismatch, or a failed stable window.
- Weakest part of the evidence: only one deterministic data/horizon/parameter
  scope was tested, and the two higher-capacity cells were blocked by branch
  identity rather than independently validated with a branch-stable protocol.
- Required follow-up before any joint value/score promotion: review whether
  the discrete `log_scale_shift_index` is a numerical-stabilization identity
  that should remain a fixed-branch veto or be separated from the mathematical
  score target. Any change requires a fresh, versioned score validation run.

## Nonclaims

This artifact does not establish exact score correctness, score convergence
outside the tested cells, exact likelihood accuracy, HMC validity or
convergence, posterior correctness, statistical superiority, GPU/XLA readiness,
production readiness, or transfer to another model, horizon, parameter region,
backend, dtype, or dataset.

# BayesFilter Custom Fixed-Metric Grid And Phase Separation Result

Date: 2026-07-20

Status: `COMPLETE`

Plan:
`docs/plans/bayesfilter_fixed_metric_custom_l_grid_and_phase_separation_plan_2026-07-20.md`

## Outcome

The fixed-metric search now accepts a caller-supplied grid of at least three
distinct integer leapfrog counts within `1 <= L <= 25`. The existing default
remains `(3, 5, 9, 13, 18, 25)`. This supports target-specific six-to-eight
point DZ5 searches without promoting the historical `(2, 7, 13)` or `(4, 5, 9)`
diagnostic grids to library defaults.

The grid runner remains a tuning/nomination phase. It independently tunes each
`(L, epsilon_L)` pair and runs discarded screens. It does not call fresh
confirmation. `confirm_fixed_metric_candidate()` remains an explicit separate
phase. The optional evidence extension is recorded as tuning evidence only.

Reasonable-epsilon search now treats a TensorFlow `InvalidArgumentError` as a
shrinkable target-domain failure only when the active target adapter explicitly
classifies it that way. Unclassified shape, dtype, XLA, and wiring failures
remain fatal runner errors.

## Decision Table

| Decision | Primary criterion | Veto status | Next justified action | Nonclaim |
| --- | --- | --- | --- | --- |
| Retain configurable grid API | Passed: default behavior, custom values, active-grid refinement, and phase separation | No schema, seed, duplicate, range, or hidden-confirmation veto | Choose and document a DZ5-specific six-to-eight point grid, then run a trusted-GPU canary | No HMC convergence, mass adequacy, ranking, or estimation readiness |

## Verification

- `43 passed` in `tests/test_hmc_fixed_metric_grid_search.py`.
- Covered default preservation, six- and eight-point custom grids, `L=2`,
  custom midpoint refinement, order-independent seeds, candidate-local failure,
  spawned scheduling semantics, and explicit no-confirmation behavior.
- Python compilation and `git diff --check` passed for changed files.
- No target-specific GPU or CPU scientific run was launched.

## Numeric And Assumption Audit

| Choice | Provenance | Status |
| --- | --- | --- |
| Default `(3,5,9,13,18,25)` | Existing reviewed API | preserved |
| Custom grid size at least three | Required to span a local search rather than a single candidate | implementation bound |
| `1 <= L <= 25` | Existing runtime safety bound; `L=2` is needed for historical DZ5 representation | reviewed configurable bound |
| Six-to-eight point DZ5 grid | User robustness requirement | target-specific choice still pending |
| Separate confirmation | Research-stage separation | enforced by API call boundary |

## GPU/Parallel Note

Process-parallel execution remains opt-in. Multiple workers are acceptable when
they are assigned to available physical GPUs and the target-specific canary
shows no harmful contention. The current generic process environment is shared
by all workers, so a two-GPU run must use an explicit device-aware launcher or
worker factory; this result does not claim multi-GPU performance.

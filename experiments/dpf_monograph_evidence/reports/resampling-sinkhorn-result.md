# IE5 resampling and Sinkhorn result

## Skeptical audit before execution

- Wrong baseline audit: the only allowed comparators are `closed_form_two_particle_soft_resampling_reference` and `manual_balanced_sinkhorn_marginal_reference`; no weaker proxy baseline was used.
- Proxy-metric audit: exact relaxed-target arithmetic and final-budget marginal residuals are the promotion criteria; categorical-reference deltas and residual trend remain bounded to their declared roles.
- Stop-condition audit: any non-finite value, tolerance miss, missing non-implication text, or budget-ladder regression would force a row failure with a structured repair trigger.
- Environment audit: the runner fixes `CUDA_VISIBLE_DEVICES=-1` before NumPy import and records a CPU-only manifest.
- Artifact audit: each canonical IE5 diagnostic writes one schema-valid JSON object, with IE3 and IE4 coverage carried forward as `passed` and later rows kept `missing`.

## Research-intent ledger

- Main question: do the bounded clean-room fixtures preserve deterministic relaxed-target expectation arithmetic for selected soft-resampling summaries and finite-budget marginal residual control for small Sinkhorn transport?
- Candidate under test: two-particle relaxed-probability bookkeeping and small balanced log-domain Sinkhorn scaling with epsilon `0.3` and budget ladder `[5, 20, 100]`.
- Expected failure mode: probability formula drift, probability normalization drift, relaxed-target expectation mismatch, unexpected categorical nonlinear-delta sign, marginal residual threshold miss, or budget-trend regression.
- Promotion criterion: soft-resampling relaxed-probability and relaxed-expectation checks pass with finite categorical-reference deltas and a nonzero categorical nonlinear delta, and final Sinkhorn marginal checks pass at budget `100` with no budget nonincrease violations.
- Promotion veto: missing comparator identity, source-support class, epsilon, budget ladder, stabilization mode, tolerance object, seed policy, or exact row-specific non-implication text.
- Continuation veto: inability to preserve the trusted marginal comparator and row-local relaxed-target caveat in schema-valid IE2 rows.
- Repair trigger: probability formula, probability normalization, relaxed expectation, categorical-delta finiteness, nonlinear-delta sign, row marginal, column marginal, mass, nonnegativity, finite-plan, or budget trend failure.
- What must not be concluded: categorical resampling law preservation, unbiasedness for nonlinear observables, posterior equivalence, exact unregularized OT equivalence, production bayesfilter correctness, banking use, model-risk use, or production readiness.

## Evidence contract

| Diagnostic | Comparator | Primary criterion | Veto diagnostics | Explanatory only | Source support |
| --- | --- | --- | --- | --- | --- |
| `soft_resampling_bias` | `closed_form_two_particle_soft_resampling_reference` | relaxed-probability and relaxed-expectation residuals finite and <= `1e-12`, categorical-reference deltas finite, and nonlinear delta nonzero | missing relaxed-target caveat, comparator id, source support, tolerance object, seed policy, or exact non-implication text | categorical-resampling, posterior, or production interpretation | `bibliography_spine_only` |
| `sinkhorn_residual` | `manual_balanced_sinkhorn_marginal_reference` | final row/column/mass/nonnegativity/finite-plan residuals finite and within tolerance at budget `100`, with zero budget nonincrease violations | missing epsilon `0.3`, budget ladder `[5, 20, 100]`, stabilization mode, comparator id, source support, tolerance object, or exact non-implication text | residual trend before final-threshold success and all posterior-equivalence commentary | `bibliography_spine_only` |

## Pre-mortem / failure-mode map

- A pass could still mislead if the harness preserves deterministic row-local arithmetic while categorical resampling behavior or downstream filter logic remains wrong.
- A fail could still be a harness artifact if the mixture rule, expectation ledger, or log-domain scaling loop is encoded incorrectly despite the intended clean-room arithmetic being valid.
- The cheapest discriminator is the row-level tolerance object family, which isolates probability formula, probability normalization, relaxed expectation, categorical-delta finiteness, nonlinear-delta sign, marginal residual, mass, nonnegativity, finite-plan, or budget-trend failure.

## Artifact list

- `experiments/dpf_monograph_evidence/reports/outputs/soft_resampling_bias.json`
- `experiments/dpf_monograph_evidence/reports/outputs/sinkhorn_residual.json`
- `experiments/dpf_monograph_evidence/reports/resampling-sinkhorn-result.md`
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie5-resampling-sinkhorn-result-2026-05-16.md`

## Residual summary

- `relaxed_probability_formula_abs`: observed `0.000e+00` against threshold `1.0e-12`, finite=`True`
- `relaxed_constant_expectation_abs`: observed `0.000e+00` against threshold `1.0e-12`, finite=`True`
- `relaxed_identity_expectation_abs`: observed `0.000e+00` against threshold `1.0e-12`, finite=`True`
- `relaxed_linear_summary_abs`: observed `0.000e+00` against threshold `1.0e-12`, finite=`True`
- `relaxed_nonlinear_expectation_abs`: observed `0.000e+00` against threshold `1.0e-12`, finite=`True`
- `categorical_identity_delta_abs`: observed `1.500e-01`, finite=`True`
- `categorical_linear_summary_delta_abs`: observed `3.000e-01`, finite=`True`
- `categorical_nonlinear_delta_abs`: observed `1.500e-01` with expected nonzero sign marker `0`
- `probability_sum_abs`: observed `0.000e+00` against threshold `1.0e-12`, finite=`True`
- `row_marginal_abs_max`: observed `1.994e-12` against threshold `1.0e-09`, finite=`True`
- `column_marginal_abs_max`: observed `5.551e-17` against threshold `1.0e-09`, finite=`True`
- `total_mass_abs`: observed `0.000e+00` against threshold `1.0e-12`, finite=`True`
- `nonnegative_plan_violation_abs`: observed `0.000e+00` against threshold `0.0e+00`, finite=`True`
- `finite_plan_violation_abs`: observed `0.000e+00` against threshold `0.0e+00`, finite=`True`
- `budget_residual_nonincrease_violations`: observed `0` against threshold `0`, finite=`True`

## Per-diagnostic decision table

| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |
| --- | --- | --- | --- | --- | --- |
| `soft_resampling_bias` | `pass` | `pass` | `not_triggered` | `not_triggered` | `none` |
| `sinkhorn_residual` | `pass` | `pass` | `not_triggered` | `not_triggered` | `none` |

## Inference-status table

| Row | Status |
| --- | --- |
| hard veto screen | both IE5 rows passed deterministic residual and finite checks (`soft_resampling_bias=pass`, `sinkhorn_residual=pass`) |
| statistically supported ranking | not applicable; IE5 is deterministic and has no candidate ranking problem |
| descriptive-only differences | categorical-reference deltas and budget-ladder trend are descriptive within the declared row-local caveats and do not support posterior or equivalence claims |
| default-readiness | not established; deterministic relaxed-target and finite-budget marginal checks are insufficient for any production or default claim |
| next evidence needed | IE6 must test the learned-map residual phase on its own artifacts |

## Run manifest

- command: `python -m experiments.dpf_monograph_evidence.runners.run_resampling_sinkhorn`
- branch: `main`
- commit: `0684d6fe4350664c34826e83df59b2413d7be89c`
- dirty-state summary captured: `M docs/plans/bayesfilter-dpf-monograph-reviewer-grade-reset-memo-2026-05-15.md
 M docs/plans/bayesfilter-student-dpf-bas...`
- python version: `3.13.13`
- numpy version: `2.1.3`
- cpu_only: `True`
- pre-import CUDA_VISIBLE_DEVICES: `-1`
- pre-import GPU hiding assertion: `True`
- seed policy: `deterministic_no_rng_resampling_sinkhorn_fixture`
- replication_count: `1`
- artifact paths: `experiments/dpf_monograph_evidence/reports/outputs/soft_resampling_bias.json, experiments/dpf_monograph_evidence/reports/resampling-sinkhorn-result.md`

## Source-support note

Both IE5 rows intentionally keep `source_support_class=row_level_source_support_class=bibliography_spine_only`. The deterministic clean-room fixtures provide local arithmetic evidence only and do not upgrade provenance beyond the bibliography-spine support allowed by the phase plan.

## Required non-implication text

- `soft_resampling_bias`: IE5 soft-resampling diagnostics validate only deterministic two-particle relaxed-target expectation arithmetic for selected affine and nonlinear test functions. They do not validate categorical resampling law preservation, unbiasedness for nonlinear observables, posterior equivalence, production bayesfilter code, banking use, model-risk use, or production readiness.
- `sinkhorn_residual`: IE5 Sinkhorn diagnostics validate only finite-budget marginal residuals for a small deterministic regularized transport fixture. They do not validate exact unregularized OT equivalence, exact EOT equivalence at finite epsilon/iteration budget, posterior equivalence, production bayesfilter code, banking use, model-risk use, or production readiness.

## Coverage semantics

Both JSON row files mark `linear_gaussian_recovery`, `synthetic_affine_flow`, and `pfpf_algebra_parity` as `passed`, both IE5 diagnostic IDs as `passed`, and all later-phase diagnostic IDs as `missing`, matching the required carry-forward coverage state after successful IE5 emission.

## Post-run red-team note

The strongest alternative explanation is that the harness preserves deterministic relaxed-target arithmetic and finite-budget marginal control while categorical resampling behavior, finite-epsilon target mismatch, or downstream filtering logic are still wrong. A contrary result that would overturn the current pass would be any later artifact showing the same mixture bookkeeping or marginal residual ledger breaks once the fixture is embedded in a richer filter or exact-equivalence comparator.

## Next-phase justification or blocker

IE5 passed its bounded relaxed-target and Sinkhorn residual checks, so the lane can advance to IE6 without upgrading any claim beyond deterministic arithmetic and marginal-residual evidence.

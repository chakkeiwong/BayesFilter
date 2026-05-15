# IE4 affine-flow PF-PF result

## Skeptical audit before execution

- Wrong baseline audit: the only allowed comparators are `analytic_affine_pushforward_density_reference` and `closed_form_pfpf_log_weight_reference`; no weaker proxy baseline was used.
- Proxy-metric audit: residual tolerances are the promotion criteria, while any narrative about nonlinear flow or filtering remains explanatory-only.
- Stop-condition audit: any non-finite or above-threshold residual would force a row failure with a structured repair trigger instead of silent renormalization.
- Environment audit: the runner fixes `CUDA_VISIBLE_DEVICES=-1` before NumPy import and records a CPU-only manifest.
- Artifact audit: each canonical IE4 diagnostic writes one schema-valid JSON object, with known coverage semantics carried into both row files.

## Research-intent ledger

- Main question: does the clean-room harness preserve affine pushforward-density and PF-PF proposal-correction algebra exactly on deterministic fixtures?
- Candidate under test: deterministic affine inverse-map, Jacobian-sign, and corrected-weight bookkeeping in the IE4 clean-room harness.
- Expected failure mode: sign mistakes in `log|det A|`, inverse-map reconstruction drift, or normalization hiding an unnormalized discrepancy.
- Promotion criterion: each required residual family stays finite and at or below `1e-12`.
- Promotion veto: missing comparator identity, source-support class, explicit sign convention, or exact row-specific non-implication text.
- Continuation veto: inability to record affine-only comparator identity, row-level source support, or separate unnormalized/normalized parity residuals.
- Repair trigger: any deterministic algebra mismatch or non-finite residual.
- What must not be concluded: anything about nonlinear flow integration, solver stability, PF-PF filtering correctness, posterior quality, banking use, model-risk use, or production readiness.

## Evidence contract

| Diagnostic | Comparator | Primary criterion | Veto diagnostics | Explanatory only | Source support |
| --- | --- | --- | --- | --- | --- |
| `synthetic_affine_flow` | `analytic_affine_pushforward_density_reference` | forward/inverse, log-det, and pushforward-density residuals all finite and <= `1e-12` | missing determinant sign, comparator id, source support, tolerance object, or exact non-implication text | nonlinear-flow interpretation | `bibliography_spine_only` |
| `pfpf_algebra_parity` | `closed_form_pfpf_log_weight_reference` | proposal-density, corrected-log-weight, normalized-weight, and probability-sum residuals all finite and <= `1e-12` | missing sign convention, determinant contribution, comparator id, source support, tolerance object, or exact non-implication text | filtering/posterior interpretation | `bibliography_spine_only` |

## Pre-mortem / failure-mode map

- A pass could still mislead if the harness is correct only for affine closed forms while nonlinear integrators remain broken.
- A fail could still be a harness artifact if the inverse map, determinant sign, or normalization ledger is encoded wrongly even though the underlying affine algebra is valid.
- The cheapest discriminator is the row-level residual object family, which localizes map inversion, determinant, proposal density, target density, or normalization failure.

## Artifact list

- `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_synthetic_affine_flow.json`
- `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_pfpf_algebra_parity.json`
- `experiments/dpf_monograph_evidence/reports/affine-flow-pfpf-result.md`
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie4-affine-flow-pfpf-result-2026-05-16.md`

## Residual summary

- `forward_reconstruction_abs_max`: observed `0.000e+00` against threshold `1.0e-12`, finite=`True`
- `inverse_reconstruction_abs_max`: observed `2.220e-16` against threshold `1.0e-12`, finite=`True`
- `log_det_abs_max`: observed `1.665e-16` against threshold `1.0e-12`, finite=`True`
- `pushforward_log_density_abs_max`: observed `8.882e-16` against threshold `1.0e-12`, finite=`True`
- `proposal_log_density_abs_max`: observed `4.441e-16` against threshold `1.0e-12`, finite=`True`
- `corrected_log_weight_abs_max`: observed `4.441e-16` against threshold `1.0e-12`, finite=`True`
- `normalized_weight_abs_max`: observed `1.110e-16` against threshold `1.0e-12`, finite=`True`
- `probability_sum_abs_max`: observed `2.220e-16` against threshold `1.0e-12`, finite=`True`

## Per-diagnostic decision table

| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |
| --- | --- | --- | --- | --- | --- |
| `synthetic_affine_flow` | `pass` | `pass` | `not_triggered` | `not_triggered` | `none` |
| `pfpf_algebra_parity` | `pass` | `pass` | `not_triggered` | `not_triggered` | `none` |

## Inference-status table

| Row | Status |
| --- | --- |
| hard veto screen | both IE4 rows passed deterministic residual and finite checks (`synthetic_affine_flow=pass`, `pfpf_algebra_parity=pass`) |
| statistically supported ranking | not applicable; IE4 is deterministic and has no candidate ranking problem |
| descriptive-only differences | any narrative about nonlinear flow integration or filtering remains descriptive-only and unsupported by IE4 |
| default-readiness | not established; affine parity is insufficient for any production or default claim |
| next evidence needed | IE5 must test later resampling/transport diagnostics on its own artifacts |

## Run manifest

- command: `python -m experiments.dpf_monograph_evidence.runners.run_affine_flow_pfpf`
- branch: `main`
- commit: `0684d6fe4350664c34826e83df59b2413d7be89c`
- dirty-state summary captured: `M docs/plans/bayesfilter-dpf-monograph-reviewer-grade-reset-memo-2026-05-15.md
 M docs/plans/bayesfilter-student-dpf-bas...`
- python version: `3.13.13`
- numpy version: `2.1.3`
- cpu_only: `True`
- pre-import CUDA_VISIBLE_DEVICES: `-1`
- pre-import GPU hiding assertion: `True`
- seed policy: `deterministic_no_rng_affine_fixture`
- replication_count: `1`
- artifact paths: `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_synthetic_affine_flow.json, experiments/dpf_monograph_evidence/reports/affine-flow-pfpf-result.md`

## Source-support note

Both IE4 rows intentionally keep `source_support_class=row_level_source_support_class=bibliography_spine_only`. The deterministic local affine fixture is clean-room evidence for algebra parity, but it does not upgrade source provenance beyond the bibliography-spine support allowed by the plan.

## Required non-implication text

- `synthetic_affine_flow`: IE4 synthetic affine-flow checks validate only closed-form affine pushforward-density, inverse-map, and Jacobian-sign parity on deterministic clean-room fixtures. They do not validate nonlinear flow integration, solver stability, PF-PF filtering correctness, production bayesfilter code, real DPF-HMC targets, posterior quality, banking use, model-risk use, or production readiness.
- `pfpf_algebra_parity`: IE4 PF-PF algebra parity checks validate only closed-form proposal-density, Jacobian-sign, unnormalized corrected-log-weight, and normalized-weight parity on deterministic affine clean-room fixtures. They do not validate nonlinear flow integration, solver stability, filtering correctness, production bayesfilter code, real DPF-HMC targets, posterior quality, banking use, model-risk use, or production readiness.

## Coverage semantics

Both JSON row files mark `linear_gaussian_recovery` as `passed`, both IE4 diagnostic IDs as `passed`, and all later-phase diagnostic IDs as `missing`, matching the within-program known coverage state after successful IE4 emission. No asymmetry is present because both row files passed.

## Post-run red-team note

The strongest alternative explanation is that the harness encodes the affine closed form correctly while later nonlinear flow integrators, solver stability, or filtering logic are still wrong. A contrary result that would overturn the current pass would be any later artifact showing the same sign convention or normalization ledger breaks once the proposal ceases to be affine.

## Next-phase justification or blocker

IE4 passed its bounded affine algebra checks, so the lane can advance to IE5 without upgrading any claim beyond affine clean-room PF-PF parity.

# Phase IE8 result: posterior sensitivity and research-evidence note

## Outcome

- Master-program exit label: `dpf_monograph_evidence_program_complete_with_blockers`
- Phase status: `complete_with_blockers`

## Summary

IE8 produced the aggregate research evidence note and summary JSON. It ran zero posterior sensitivity checks and preserved IE6 as a visible deferred evidence gap.

## Decision Table

| Diagnostic | Status | Claim ceiling | Reference class | Trusted reference | Repair trigger |
| --- | --- | --- | --- | --- | --- |
| `linear_gaussian_recovery` | `passed` | `controlled_fixture_supported` | `analytic_reference` | `True` | `Re-open IE3 if the schema stops carrying PF uncertainty fields, EDH special-case tolerances fail, or the CPU-only manifest proof drifts.` |
| `synthetic_affine_flow` | `passed` | `controlled_fixture_supported` | `analytic_reference` | `True` | `none` |
| `pfpf_algebra_parity` | `passed` | `controlled_fixture_supported` | `analytic_reference` | `True` | `none` |
| `soft_resampling_bias` | `passed` | `controlled_fixture_supported` | `analytic_reference` | `True` | `none` |
| `sinkhorn_residual` | `passed` | `controlled_fixture_supported` | `analytic_reference` | `True` | `none` |
| `learned_map_residual` | `deferred` | `deferred_evidence_gap` | `no_trusted_reference_exploratory_only` | `False` | `missing artifact provenance` |
| `hmc_value_gradient` | `passed` | `exploratory_only` | `no_trusted_reference_exploratory_only` | `False` | `none` |
| `posterior_sensitivity_summary` | `missing` | `not_tested` | `no_trusted_reference_exploratory_only` | `False` | `not tested` |

## Non-Implication

The DPF monograph implementation-evidence program provides clean-room controlled-fixture and governance evidence only. It does not validate real DPF-HMC targets, DSGE or MacroFinance posterior inference, banking use, model-risk use, production bayesfilter code, or production readiness.

## Artifacts

- `experiments/dpf_monograph_evidence/reports/outputs/dpf_monograph_evidence_summary.json`
- `experiments/dpf_monograph_evidence/reports/dpf-monograph-research-evidence-note.md`
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-final-closeout-2026-05-16.md`

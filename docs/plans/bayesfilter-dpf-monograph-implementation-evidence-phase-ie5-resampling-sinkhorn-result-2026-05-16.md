# Phase IE5 result: soft-resampling and Sinkhorn controlled tests

## Outcome

- Master-program exit label: `ie_phase_passed`
- Local exit label: `ie5_resampling_sinkhorn_passed`
- Phase status: `pass`

## Skeptical audit before execution

- Baselines stayed pinned to the two row-level deterministic comparators required by the plan.
- Promotion criteria stayed row-local and deterministic; no explanatory residual trend or posterior commentary was upgraded into a pass criterion.
- The runner records both relaxed-target arithmetic and final marginal thresholds, so categorical-reference deltas cannot hide relaxed-target failures and budget trend cannot hide a final-threshold failure.
- CPU-only execution was enforced with `CUDA_VISIBLE_DEVICES=-1` before NumPy import.

## Research-intent ledger

- Main question: whether deterministic soft-resampling relaxed-target arithmetic and bounded Sinkhorn marginal control meet the clean-room IE5 comparator contracts.
- Promotion criterion: the two IE5 rows pass their declared tolerance objects with seed policy `deterministic_no_rng_resampling_sinkhorn_fixture` and replication_count `1`.
- Promotion veto: missing comparator identity, relaxed-target caveat, source-support class, epsilon `0.3`, budget ladder `[5, 20, 100]`, stabilization mode, tolerance object, or exact row-specific non-implication text.
- Continuation veto: inability to encode the trusted Sinkhorn marginal comparator or the soft-resampling caveat in a schema-valid IE2 row.
- Repair trigger: any probability formula, probability normalization, relaxed expectation, categorical-delta finiteness, nonlinear-delta sign, marginal residual, mass, nonnegativity, finite-plan, or budget-trend failure.
- Not concluded: categorical law preservation, nonlinear unbiasedness, posterior equivalence, exact OT equivalence, production correctness, banking use, model-risk use, or production readiness.

## Evidence contract

- `soft_resampling_bias` vs `closed_form_two_particle_soft_resampling_reference` with deterministic relaxed-probability and relaxed-expectation thresholds, finite categorical-reference deltas, nonzero nonlinear-delta sign requirement, and `bibliography_spine_only` source support.
- `sinkhorn_residual` vs `manual_balanced_sinkhorn_marginal_reference` with epsilon `0.3`, budget ladder `[5, 20, 100]`, final marginal thresholds at budget `100`, zero nonincrease violations, and `bibliography_spine_only` source support.
- Explanatory-only diagnostics remain any categorical-resampling, posterior, or exact-equivalence commentary plus budget-trend interpretation beyond the threshold contract.

## Decision table

| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `soft_resampling_bias` | `pass` | `pass` | `not_triggered` | `not_triggered` | deterministic relaxed-target scope | carry IE5 pass into IE6 coverage state | categorical law preservation / posterior claims |
| `sinkhorn_residual` | `pass` | `pass` | `not_triggered` | `not_triggered` | deterministic finite-epsilon finite-budget scope | carry IE5 pass into IE6 coverage state | exact OT/EOT or posterior claims |

## Inference-status table

| Row | Status |
| --- | --- |
| hard veto screen | passed for both IE5 diagnostics |
| statistically supported ranking | not applicable |
| descriptive-only differences | categorical-reference deltas and budget-ladder trend remain descriptive within the declared caveats |
| default-readiness | not established |
| next evidence needed | IE6 artifacts |

## Residual highlights

- categorical nonlinear delta magnitude: `1.500e-01` with expected nonzero sign marker `0`
- relaxed identity residual: `0.000e+00`
- categorical identity delta magnitude: `1.500e-01`
- probability-sum residual: `0.000e+00`
- final row marginal residual: `1.994e-12`
- final column marginal residual: `5.551e-17`
- final total-mass residual: `0.000e+00`
- budget nonincrease violations: `0`

## Run manifest

- command: `python -m experiments.dpf_monograph_evidence.runners.run_resampling_sinkhorn`
- branch: `main`
- commit: `0684d6fe4350664c34826e83df59b2413d7be89c`
- python: `3.13.13` / numpy `2.1.3`
- cpu_only: `True`
- pre-import CUDA_VISIBLE_DEVICES: `-1`
- pre-import GPU hiding assertion: `True`
- seed policy: `deterministic_no_rng_resampling_sinkhorn_fixture`
- replication_count: `1`
- artifact paths: `experiments/dpf_monograph_evidence/reports/outputs/soft_resampling_bias.json, experiments/dpf_monograph_evidence/reports/resampling-sinkhorn-result.md`

## Post-run red-team note

IE5 evidence could still coexist with wrong categorical resampling behavior, wrong exact-equivalence interpretation at finite epsilon, or broken downstream filtering logic, so IE5 should only be cited for deterministic relaxed-target arithmetic and bounded marginal residual control.

## Next-phase justification

IE5 survived the skeptical audit and passed both bounded promotion criteria, so the clean-room implementation-and-evidence lane may proceed to IE6.

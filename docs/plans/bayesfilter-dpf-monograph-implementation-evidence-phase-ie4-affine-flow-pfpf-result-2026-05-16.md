# Phase IE4 result: affine-flow PF-PF density and log-det tests

## Outcome

- Master-program exit label: `ie_phase_passed`
- Local exit label: `ie4_affine_flow_pfpf_passed`
- Phase status: `pass`

## Skeptical audit before execution

- Baselines stayed pinned to the two row-level analytic comparators required by the plan.
- Promotion criteria stayed row-local and deterministic; no proxy metric was upgraded into a pass criterion.
- The runner records separate unnormalized and normalized parity residuals, so normalization cannot hide an upstream mismatch.
- CPU-only execution was enforced with `CUDA_VISIBLE_DEVICES=-1` before NumPy import.

## Research-intent ledger

- Main question: whether deterministic affine pushforward-density and PF-PF proposal-correction algebra match their closed-form references exactly.
- Promotion criterion: every required residual object remains finite and <= `1e-12`.
- Promotion veto: missing comparator identity, determinant sign, source-support class, sign convention, tolerance object, or exact row-specific non-implication text.
- Continuation veto: inability to encode affine-only non-implication, comparator identity, or separate unnormalized/normalized parity residuals.
- Repair trigger: any residual failure would localize map inversion, determinant sign, proposal density, target density, or normalization.
- Not concluded: anything about nonlinear integration, solver stability, filtering correctness, posterior quality, banking use, model-risk use, or production readiness.

## Evidence contract

- `synthetic_affine_flow` vs `analytic_affine_pushforward_density_reference` with deterministic `1e-12` residual thresholds and `bibliography_spine_only` source support.
- `pfpf_algebra_parity` vs `closed_form_pfpf_log_weight_reference` with deterministic `1e-12` residual thresholds and `bibliography_spine_only` source support.
- Explanatory-only diagnostics remain any nonlinear-flow or filtering commentary.

## Decision table

| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `synthetic_affine_flow` | `pass` | `pass` | `not_triggered` | `not_triggered` | affine-only scope | carry IE4 pass into IE5 coverage state | nonlinear-flow/filter correctness |
| `pfpf_algebra_parity` | `pass` | `pass` | `not_triggered` | `not_triggered` | affine-only scope | carry IE4 pass into IE5 coverage state | nonlinear-flow/filter correctness |

## Inference-status table

| Row | Status |
| --- | --- |
| hard veto screen | passed for both IE4 diagnostics |
| statistically supported ranking | not applicable |
| descriptive-only differences | all nonlinear-flow and posterior-quality commentary remains descriptive-only |
| default-readiness | not established |
| next evidence needed | IE5 artifacts |

## Residual highlights

- forward reconstruction max residual: `0.000e+00`
- inverse reconstruction max residual: `2.220e-16`
- log-det residual: `1.665e-16`
- pushforward log-density residual: `8.882e-16`
- proposal log-density residual: `4.441e-16`
- corrected log-weight residual: `4.441e-16`
- normalized-weight residual: `1.110e-16`
- probability-sum residual: `2.220e-16`

## Run manifest

- command: `python -m experiments.dpf_monograph_evidence.runners.run_affine_flow_pfpf`
- branch: `main`
- commit: `0684d6fe4350664c34826e83df59b2413d7be89c`
- python: `3.13.13` / numpy `2.1.3`
- cpu_only: `True`
- pre-import CUDA_VISIBLE_DEVICES: `-1`
- pre-import GPU hiding assertion: `True`
- seed policy: `deterministic_no_rng_affine_fixture`
- replication_count: `1`
- artifact paths: `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_synthetic_affine_flow.json, experiments/dpf_monograph_evidence/reports/affine-flow-pfpf-result.md`

## Post-run red-team note

Affine algebra pass evidence could still coexist with broken nonlinear integration or filtering logic, so IE4 should only be cited for closed-form affine parity.

## Next-phase justification

IE4 survived the skeptical audit and passed both bounded promotion criteria, so the clean-room implementation-and-evidence lane may proceed to IE5.

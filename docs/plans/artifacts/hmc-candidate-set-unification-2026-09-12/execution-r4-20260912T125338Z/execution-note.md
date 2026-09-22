# HMC candidate-set unification R4 execution note

Date: 2026-09-12
Plan: `docs/plans/bayesfilter-hmc-candidate-set-unification-plan-2026-09-12.md`
Plan revision: R4
Repository commit inspected: `9be4b8fe7bad711deea61e915c6f95bc0d37649f`
Execution root:
`docs/plans/artifacts/hmc-candidate-set-unification-2026-09-12/execution-r4-20260912T125338Z/`

This execution covers the bounded R4 controller, artifact/replay, and guide
alignment repairs. It does not run an HMC campaign, GPU process, downstream
MacroFinance/dsge_hmc code, or a public numerical-default migration.

## Skeptical pre-execution audit

The R3 state was checked against the actual source before edits. The pure
controller had no durable reconstruction path, the work reservation ledger did
not charge dispatch, and the reference guide still described
`select_fixed_transport_candidate_set` as a second recommended lifecycle.
These were material correctness/documentation defects. The R4 repairs were
limited to those boundaries; numerical adapter integration remains a separate
P2/P3/P5 task with its own known-target and TensorFlow qualification evidence.

## Changes executed

- Bound scope and candidate identity to backend, dtype, execution mode, adapter
  signature, and source-dependency hash.
- Charged each dispatched work item once, released unspent candidate reserve,
  exposed used/reserved/remaining budget, and made unfunded exploration partial.
- Added checked scope/config/record/receipt/action/work reconstruction and
  `resume_hmc_candidate_set` for persisted work-order resume.
- Strengthened artifact validation for candidate hashes, scope identity,
  parent/child repair invariants, directional epsilon, receipt uniqueness and
  draw-range overlap, status precedence, and shared-invalidity replay veto.
- Removed the legacy selector from the recommended reference workflow and
  labelled its example diagnostic-only.

## Commands and evidence

Environment: Python 3.13.13; focused TensorFlow imports use the repository's
CPU-hidden test convention where applicable. Git worktree contained unrelated
SSL-LSTM, sigma-point, and campaign-ledger edits; they were preserved.

Focused command:

```text
pytest -q tests/test_hmc_candidate_set_tuning.py tests/test_hmc_candidate_set_artifacts.py tests/test_hmc_tuning_documentation_contract.py
```

Result: 37 passed, 2 pre-existing TensorFlow Probability deprecation warnings.

Additional checks from the R3 execution root remain valid: controller/artifact
tests, route inventory, generated route-table check, and the 556-page guidebook
build passed. R4 also passed the HMC dispatch/contract/replay suite (37 tests),
the public ordinary/outer-loop suite (185 tests), and the fixed-transport suite
(70 tests, with dependency deprecation warnings). The R4 source changes add no
numerical or posterior evidence.

## Disposition

R4 controller/replay/documentation work is complete for this bounded phase.
P2 typed numerical adapters, P3 replacement of legacy active numerical
selection/replay consumers, and P5 per-adapter TensorFlow/XLA qualification
remain open. The current public numerical routes therefore retain their
existing authority classification and must not be described as using the
unified lifecycle until those phases pass.

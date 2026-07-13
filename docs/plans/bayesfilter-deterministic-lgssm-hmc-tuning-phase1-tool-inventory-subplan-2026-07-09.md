# Phase 1 Subplan: Tool Inventory And API Binding

Date: 2026-07-09

## Phase Objective

Inventory the current BayesFilter tuning APIs and bind the deterministic driver
to existing public tools instead of inventing a new tuning procedure.

## Entry Conditions Inherited From Previous Phase

- Phase 0 result passed or documented bounded fallback with no material blocker.
- Master program and runbook identify the deterministic tuning direction.

## Required Artifacts

- Result:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-phase1-tool-inventory-result-2026-07-09.md`
- API inventory table with exact module/class/function names.
- Driver dependency decision table.

## Required Checks / Tests / Reviews

- Read-only source inspection of tuning modules under `bayesfilter/inference`.
- No runtime HMC.
- Local check that planned imports are exported or otherwise intentionally
  internal with justification.
- Claude review if the inventory changes required API choices.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Which existing BayesFilter tools must the deterministic driver call? |
| Baseline/comparator | `quadratic_geometry`, `mass_matrix`, `hmc_kernel_tuning`, `hmc_budget_ladder`, and `hmc_diagnostics`. |
| Primary pass criterion | Every tuning decision point is assigned to an existing Python API or explicitly deferred as implementation gap. |
| Veto diagnostics | Agent-only tuning decision remains, missing quadratic initializer, missing mass/covariance handoff, non-XLA target path allowed. |
| Explanatory diagnostics | API line anchors and relevant tests/examples. |
| Not concluded | No tuning result, sampler convergence, or LGSSM recovery claim. |

## Forbidden Claims / Actions

- Do not run tuning.
- Do not choose step sizes, leapfrogs, budgets, or burn-in manually.
- Do not add new backend dependencies.

## Exact Next-Phase Handoff Conditions

- Phase 2 config schema can name exact APIs and config fields to drive them.

## Stop Conditions

- Existing APIs cannot support deterministic tuning without material code work
  not planned here.
- Public/private API boundary is unclear enough to require human direction.

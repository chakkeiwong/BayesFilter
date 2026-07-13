# Phase 0 Subplan: Source And Identifiability Inventory

Date: 2026-07-08

## Phase Objective

Determine the first admissible multidimensional LGSSM target contract by
source-anchoring the constrained state-space structure, stationarity mechanism,
local BayesFilter stationary/Lyapunov code, and identifiability limits.

## Entry Conditions Inherited From Previous Phase

- New program starts after the static QR LGSSM Phase 17-21 runbook completed
  only fixture-local readiness.
- No multidimensional triangular LGSSM runtime artifacts are assumed.
- User selected lower-triangular or block-lower-triangular dynamics as the
  serious next direction.

## Required Artifacts

- Phase 0 result:
  `docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase0-source-identifiability-result-2026-07-08.md`.
- Source/support ledger section for:
  - MARSS constrained state-space model;
  - statsmodels / Ansley-Kohn stationary transform as context/foil;
  - stationary VAR HMC parameterization context;
  - local `bayesfilter/linear/stationary_lgssm_derivatives_tf.py`.
- Recommendation: `lower_triangular_first`, `block_lower_triangular_first`,
  or `BLOCK_SOURCE_OR_IDENTIFIABILITY`.

## Required Checks/Tests/Reviews

- Read local stationary/Lyapunov implementation and tests.
- Search repo for existing stationary LGSSM utilities.
- Confirm no implementation/runtime edits are made in Phase 0.
- `git diff --check` on Phase 0 docs.
- Claude read-only launch/Phase 0 review if available; otherwise documented
  same-foreground Codex substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Which source-anchored constrained multidimensional LGSSM target should BayesFilter implement first? |
| Baseline/comparator | MARSS constrained form, stationarity literature/code context, and local stationary/Lyapunov utilities. |
| Primary criterion | A clear recommendation with supported and unsupported claims separated. |
| Veto diagnostics | Unsupported identifiability claim, no stationary initial law, source mismatch, hidden dense-A assumption, or treating stationarity as full identifiability. |
| Explanatory diagnostics | Local code inventory, test inventory, source anchors, open gaps. |
| Not concluded | That HMC works, that the model is globally identifiable in all settings, or that any source proves BayesFilter readiness. |
| Artifact | Phase 0 result and review record. |

## Forbidden Claims/Actions

- Do not implement model code.
- Do not run training or HMC.
- Do not claim full identifiability; say exactly what is enforced and what
  remains empirical or unchecked.
- Do not claim stationarity transform alone solves state-space equivalence.

## Exact Next-Phase Handoff Conditions

Phase 1 may begin only if Phase 0 names the first target contract and records:

- why the contract is stationarity-safe;
- how `H=I`, diagonal `Q/R`, fixed order, and triangular/block structure reduce
  latent-coordinate ambiguity;
- what remains unproved;
- required parameter names and constraints to formalize.

## Stop Conditions

Stop if source/design review finds the first target contract unsupported,
ambiguous, or likely non-identifiable in a way that invalidates HMC testing.

## Skeptical Plan Audit

The main risk is overclaiming identifiability. Phase 0 passes only if it states
that triangular structure plus fixed observation coordinates is an audited
benchmark design, not a theorem of global identifiability for all LGSSMs.

# Register: remaining original-plan blockers and promotion gates

## Date

2026-05-05

## Purpose

This register is the control layer for the remaining gaps of
`bayesfilter-structural-state-partition-core-plan-2026-05-04.md`.  It records
what BayesFilter can close locally, what remains client-owned, and which claim
labels are allowed.

| Gap | Current evidence | Required gate | Owner | Allowed label now | Promotion blocker |
| --- | --- | --- | --- | --- | --- |
| Workspace hygiene | Current pass has only untracked handoff/assets/templates before validation. | Scoped staging by path/hunk; reset memo phase log. | BayesFilter | `scoped_pass` | None after scoped commit. |
| Provenance/control layer | Source map contains the original plan and Phase 8 plans; new roadmap/audit/register are added in this pass. | YAML parse and reset-memo closed-vs-blocked summary. | BayesFilter | `control_gate_ready` | None after validation. |
| DSGE adapter pilot | `/home/chakwong/python` commit `8645623` exposes explicit metadata for SmallNK, Rotemberg, and SGU; focused client metadata tests pass. | BayesFilter gate confirmation plus model-specific residual evidence before nonlinear promotion. | BayesFilter plus DSGE client repo | `client_metadata_ready_for_structural_tests` | Rotemberg second-order/pruned identity evidence, SGU deterministic residual evidence, EZ timing/partition audit, and downstream derivative/JIT/HMC gates remain open. |
| Particle-filter semantics | BayesFilter has structural toy models and exact AR(2) Kalman reference. | Structural bootstrap particles sample innovations, preserve completion, and block unlabeled deterministic-coordinate noise. | BayesFilter | `monte_carlo_value_only` | Differentiable PF/PMCMC claims need separate literature and estimator audit. |
| Factor backend audit | Covariance Kalman is value-side exact; richer QR/SVD derivative paths live in client repos. | Backend classification separates value, derivative, compiled, approximation, and HMC status. | BayesFilter plus clients | `value_exact` or `blocked` | Client QR/SVD extraction needs backend-specific parity and derivative tests. |
| SVD/eigen derivative certification | Literature gate already records spectral-gap risk from Matrix Backprop. | Spectral gap telemetry plus finite-difference and JVP/VJP evidence. | BayesFilter plus clients | `not_certified` unless gate passes | Stress tests near small gaps and documented validity regions. |
| MacroFinance expanded-provider evidence | BayesFilter gates exist for large-scale masks, cross-currency coverage/oracles, production exposure, and HMC diagnostics. | Provider-owned masked derivative metadata, blockwise oracle checks, final readiness, sparse policy, and final identification evidence. | MacroFinance client repo | `gate_consumes_provider_evidence` | Final ten-country provider remains blocked until real provider evidence passes. |
| HMC sampler readiness | Target and diagnostic gates exist; no sampler is run by BayesFilter. | Real chain diagnostics: finite values, acceptance bounds, zero divergences, ESS, split R-hat, and backend sensitivity. | Client sampler workflows | `not_claimed` | Tiny and medium HMC runs with diagnostics. |
| Release docs/literature | Docs contain conservative chapters and source maps. | Docs build, stale-claim search, source-map parse, and primary-source checks for promoted claims. | BayesFilter | `drafted_with_blockers` | Full release requires client promotion blockers closed or documented. |

## Label Policy

- `filter-correct`: allowed only for a backend with value and derivative tests
  that match its claim scope.
- `sampler-usable`: allowed only after finite chain diagnostics pass on a real
  target.
- `converged`: allowed only after strict multi-chain diagnostics pass.
- `production-ready`: allowed only when final-data readiness, identification,
  sparse backend, and HMC diagnostics all pass.
- `approximation_only`: required for full-state mixed-model integration,
  artificial deterministic-coordinate noise, or any enlarged proposal space.

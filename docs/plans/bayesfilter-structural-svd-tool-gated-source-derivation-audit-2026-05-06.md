# Audit: structural SVD tool-gated source and derivation evidence

## Date

2026-05-06

## Purpose

This note executes Phase 1 of
`docs/plans/bayesfilter-structural-svd-final-execution-plan-2026-05-06.md`.
It is an addendum to the earlier source/code audit:

- `docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md`

The earlier audit established the BayesFilter-local structural contract and
code reuse decisions.  This addendum applies the stricter tool-gated evidence
policy: ResearchAssistant for source-support status and MathDevMCP for local
LaTeX obligation routing.

## Tool status

ResearchAssistant:

- Workspace status: available, read-only, offline, local workspace rooted at
  `/home/chakwong/research-assistant`.
- Parser/tool matrix: local lifecycle, metadata-only ingest, PDF text ingest,
  structured source inspection, and parser smoke workflows report ready.
- Broad local summary searches for the required structural SVD phase sources
  returned no matching paper summaries:
  - Julier/Uhlmann unscented-transform or nonlinear transformation support;
  - Matrix Backprop/SVD-eigen derivative support;
  - Hoffman/Gelman NUTS support;
  - Betancourt HMC conceptual support.

MathDevMCP:

- Tool matrix supports LaTeX search, derivation-backed claim routing, and
  document-code consistency workflows.
- `latex_label_lookup` succeeded for the BayesFilter structural labels listed
  below.
- `typed_obligation_label` on `prop:bf-structural-ukf-pushforward` located the
  proposition and routed the obligation to human review, not backend
  certification.  This is expected because the claim is a measure-theoretic
  pushforward statement over model maps, not a bounded symbolic identity.

## Structural contract labels checked

| Label | File | Tool status | Evidence label |
| --- | --- | --- | --- |
| `def:bf-structural-state-partition` | `docs/chapters/ch02_state_space_contracts.tex` | Found by MathDevMCP with paragraph context. | `derivation_checked` for location and contract extraction; semantic proof not required for a definition. |
| `asm:bf-approximation-labeling` | `docs/chapters/ch02_state_space_contracts.tex` | Found by MathDevMCP with paragraph context. | `derivation_checked` for location and policy extraction; treated as release policy. |
| `prop:bf-structural-ukf-pushforward` | `docs/chapters/ch18b_structural_deterministic_dynamics.tex` | Found by MathDevMCP; typed obligation routing requires human review. | `human_review_required` for theorem-level certification; prose proof and tests are required. |
| `eq:bf-structural-ukf-prop-deterministic-completion` | `docs/chapters/ch18b_structural_deterministic_dynamics.tex` | Found by MathDevMCP with proof context. | `derivation_checked` for extraction; value tests check finite-dimensional fixtures. |

## Source-support table

| Claim path | Exact LGSSM | Structural nonlinear | Labeled approximation | Source status | Derivation status | Promotion status |
| --- | --- | --- | --- | --- | --- | --- |
| Covariance Kalman prediction-error likelihood | Yes | No | No | `source_supported` by local monograph equations recorded in the 2026-05-04 source/code audit. | `derivation_checked` by existing exact Kalman tests. | `value_only` exact LGSSM reference. |
| Structural sigma-point pushforward doctrine | No | Yes | Sigma-point Gaussian closure remains approximate. | `source_missing` in ResearchAssistant local summaries for external UKF literature; local Chapter 18b contains a proof sketch. | `human_review_required` for theorem-level proof; structural tests check finite fixtures. | `value_only`, `approximation_only` for nonlinear likelihood. |
| Approximation labeling for mixed full-state integration | No | Policy for mixed structural models | Yes | `source_supported` as BayesFilter release policy in Chapter 2. | `derivation_checked` by MathDevMCP label extraction and config tests. | Gate closed as policy; not a model promotion claim. |
| SVD/eigen derivative safety | No | Potentially relevant to structural SVD backends | No | `source_missing` in ResearchAssistant local summaries for Matrix Backprop support. | `human_review_required` until finite-difference/JVP/VJP/spectral-gap tests exist. | `not_claimed`. |
| HMC/NUTS convergence readiness | No | Downstream sampler claim only | No | `source_missing` in ResearchAssistant local summaries for NUTS/HMC support. | Not a BayesFilter derivation claim; requires real chain diagnostics. | `not_claimed`. |
| Rotemberg/SGU/EZ structural DSGE promotion | No | Model-specific | No | `source_missing` for model-specific residual derivations in this BayesFilter audit. | `human_review_required` until client-owned residual tests exist. | blocked. |

## Interpretation

Phase 1 is strong enough to proceed to Phase 2 for BayesFilter-local code reuse
and value-side validation because the exact Kalman and structural-contract
claims are already tied to local docs and tests.

Phase 1 is not strong enough to support any of the following promotions:

- theorem-level certification of the structural UKF pushforward claim solely
  from MCP output;
- external-literature claims about UKF, Matrix Backprop, NUTS, or HMC unless
  source records are ingested/reviewed or otherwise explicitly cited;
- SVD/eigen gradient certification;
- Rotemberg, SGU, EZ, or NAWM nonlinear structural filtering promotion;
- HMC convergence or production readiness.

## Next justified phase

Phase 2 is justified as a code reuse and document-code audit.  Backend
implementation remains limited to localized fixes that are demanded by tests.

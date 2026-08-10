# Final crosswalk and blocker resolution note after the whole-document audit start

- **Date:** 2026-08-07
- **Purpose:** collect the remaining audit findings into a narrow rewrite crosswalk so the final release pass can proceed without losing the whole-document audit trail.
- **Status:** working blocker-resolution crosswalk; not yet the final release note.
- **Resolution addendum for R1, 2026-08-08:** the squared-TT mathematical blocker is closed for the declared fixed branch by `bayesfilter-fable-rewrite-squared-tt-block-certificate-result-2026-08-08.md`. Chapter 37 implementation/source-route parity remains a separate open audit.

---

## 1. What the full audit has already confirmed as stable enough to keep

These passages should be preserved rather than rewritten again unless a later check finds a new error:

- `ch05_prediction_error_decomposition.tex:78-104` — exact PED likelihood and missing-data convention.
- `ch09_kalman_score.tex:63-88` — solve-form score recursion.
- `ch10_kalman_hessian.tex:123-150` — Hessian decomposition.
- `ch11_structural_derivatives.tex:91-147` — transform chain rule and initial-condition cases.
- `ch12_factor_derivatives.tex:39-60, 65-107, 112-177` — factor reconstruction with Cholesky/QR branch caveats.
- `ch14_derivative_validation.tex:62-99, 137-153` — validation ladder and artifact schema.
- `ch19_particle_filters.tex:488-592` — weighted post-decision PF proof object and differentiability boundary.
- `ch32c_entropic_ot_sinkhorn.tex:688-694, 903-918, 1888-1944` — source-boundary language, contract-E design-space boundary, and partial-derivative warning.
- `ch32c2_ledh_pfpf_ot_custom_gradient.tex:107-119, 2287-2299` — canonical policy identifiers and chunk/tuning semantics.
- `ch32e_icnn_brenier_monge_gap_map_learning.tex:287-341` — fixed-`\varphi` explanation and schematic trainer status.
- `ch20_filter_choice.tex:80-84` — repaired later-book pointer.
- `ch28a_neural_network_state_space_model_applications.tex:457-485, 815-869` — HAC passage as audit direction; validation as candidate failure, not direction failure.
- `appendices/app_b_matrix_calculus.tex:8-30` — matrix identities with symmetry conditions.
- `appendices/app_c_factor_derivative_proofs.tex:1-14` — bounded placeholder status.
- `appendices/app_e_researchassistant_workflows.tex:52-55` — honest source-support blocker ledger language.

## 2. Remaining high-risk issues to rewrite before final release

### R1. Squared-TT retained-coordinate derivation (`ch36b`, `ch37`)
At the time of this 2026-08-07 crosswalk, this was the single most important mathematical blocker. The 2026-08-08 resolution addendum above records its bounded closure.

Current state:
- the rewrite now explicitly chooses the retained-first branch,
- uses right-side contractions,
- and includes a derivation note,
- but the release ledger still wants a scalar/vector identity certificate and a numerical check.

Rewrite action:
1. keep the retained-first/right-contraction convention,
2. add one compact scalar/vector identity check with a fixed numerical example,
3. ensure the derivative recursion and the query rule use the same retained/reference convention,
4. preserve the route-audit note as a release boundary, not as a theorem.

### R2. SR-UKF release posture (`ch17`, `ch12`, `app_c`)
The rewrite is honest, but the signed derivative lane remains conditional.

Rewrite action:
1. either finish the ordered signed update/downdate derivative primitive,
2. or explicitly narrow every summary/recommendation so the negative-weight analytical-score route is conditional/incomplete.
3. keep the current filtered-factor correction wording, but remove the lingering prose fragment before the filtered-factor derivative equation.

### R3. Source-support / bibliography closure (`references.bib`, `ch28a`, `ch19b`, `ch32c`, `app_e`)
This is now a curation problem, not a mechanical BibTeX problem.

Rewrite action:
1. keep only active bibliography records needed by the monograph,
2. move unresolved candidate papers into a ledger note if they are not needed for the active prose,
3. keep theorem-level claims at the exact source-support boundary already recorded,
4. make sure the HAC passage and Li-Coates source boundary remain explicit nonclaims until inspected support is available.

### R4. Canonical LEDH policy binding (`ch32c2`)
The current policy identifiers are present and should be preserved, but the policy subsection should be coherent and singular.

Rewrite action:
1. keep the route identifiers,
2. ensure the canonical policy statement appears once in its strongest form,
3. keep historical routes clearly labeled as historical/diagnostic-only,
4. avoid letting benchmark prose weaken policy text.

### R5. ICNN trainer/status (`ch32e`)
The fixed-`\varphi` explanation is repaired, but the trainer is still schematic.

Rewrite action:
1. either specify the trainer completely,
2. or label it clearly as schematic/direct-map pattern,
3. avoid “canonical trainer” language unless it is truly completed.

### R6. Final typesetting / prose cleanup
The rewrite still carries a warning and significant overflow debt.

Rewrite action:
1. normalize accent syntax and old-TeX fraction forms,
2. break the widest display/table/path-heavy passages,
3. remove the remaining `amsmath`/foreign-command warning,
4. remove remaining process-register language where it obstructs reading.

## 3. Equation-level release priorities

The highest-priority equations to certify or demote before release are:

- `eq:bf-srukf-filtered-factor`
- `eq:bf-hd-squared-tt-retained-numerator-contraction`
- `eq:bf-hd-ttkr-retained-query-rule`
- `eq:bf-custom-primal-target`
- `eq:bf-ssl-lstm-growing-hac`
- `eq:bf-neural-ot-direct-icnn-objective`
- `prop:bf-eot-stopped-normalization-partial`

## 4. Evidence artifacts to preserve with the release record

The next release note should preserve:
- the persistent audit ledger,
- the squared-TT derivation note,
- the paragraph/equation audit note,
- the final blocker-closure note,
- the curated promotion diff set,
- the final release-status note,
- MathDevMCP unverified artifacts for the high-risk labels,
- and the final build log.

## 5. What the next pass should produce

The next pass should end with a short final note that says:

- which blockers are closed,
- which are narrowed to explicit nonclaims,
- which still block canonical replacement,
- and whether the rewrite is now a replacement candidate or only a curated promotion base.

This note should be written after the last narrow rewrite and clean rebuild, not before.

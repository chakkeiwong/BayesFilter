# Actionable Math Document Rigor Audit

Source: `zlb_survey_labeled.tex`
Source SHA-256: `797a0679e1db48e30edfc1637e8daf939adc46872180fb028b80014529f41b37`
Coverage: `partial_coverage`; selected `25`; distinct issues `4`; open `4`; actionable proposals `1`; resolved by context `0`.

Detailed evidence pointer: `source_reports`; forensic rendering: `forensic_markdown`.

## Issue Ledger

### `eq:75/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['statistical_estimator']`
- Location: `zlb_survey_labeled.tex > 13. The MacroFinance shadow-rate contract > 13.1 The model as coded > eq:75 > line 1525`
- Unresolved obligations: `['obligation_1']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:76/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['unknown']`
- Location: `zlb_survey_labeled.tex > 13. The MacroFinance shadow-rate contract > 13.1 The model as coded > eq:76 > line 1537`
- Unresolved obligations: `['obligation_2']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:81/formalization-and-source-role`

- Status: `needs_formalization`
- Roles: `['estimator_objective', 'local_derived_claim']`
- Location: `zlb_survey_labeled.tex > 13. The MacroFinance shadow-rate contract > 13.3 Posterior geometry of the exact-censoring variant > eq:81 > line 1670`
- Unresolved obligations: `['obligation_1', 'obligation_3']`
- Boundary: Formalization status is diagnostic and does not establish truth or falsehood.

### `eq:85/matrix-domain-and-invertibility`

- Status: `unresolved`
- Roles: `['approximation_linearization']`
- Location: `zlb_survey_labeled.tex > 13. The MacroFinance shadow-rate contract > 13.4 An exact filter and an exactly weighted particle authority > eq:85 > line 1760`
- Unresolved obligations: `['dimension_contract', 'invertibility']`
- Repair status: `actionable_assumption_text`
- Candidate patch: State a condition ensuring that the displayed inverse operand is invertible.
- Patch boundary: `candidate_exposition_patch_not_certificate`; human review required.
- Boundary: This status reports whether the document states the scoped exposition conditions. It does not certify the matrix theorem or source-specific validity.

## Non-Claims

- Context closure means the document states the scoped exposition condition; it is not a proof certificate.
- Candidate patches are bounded human-review text and do not establish source-specific truth.
- This focused audit does not certify general readability, pedagogy, or whole-document correctness.

## Package note (2026-08-19)

This focused audit ran on the labeled-equation LaTeX export of the
expanded manuscript (survey equations 56a and 75--98; 25 of 102 labeled
equations selected). The single actionable proposal (state invertibility
of the inverse operand in eq. (85)) was applied to the manuscript on the
same date. The three remaining records are formalization-status
diagnostics for model-definition equations and, per the tool's own
boundary statements, do not establish truth or falsehood. The earlier
zero-scope audit of 2026-08-18 is preserved in `mathdevmcp_audit.md`.

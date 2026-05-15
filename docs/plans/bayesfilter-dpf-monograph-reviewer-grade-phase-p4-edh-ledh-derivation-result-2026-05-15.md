# Phase P4 result: EDH and LEDH derivation expansion

## Date

2026-05-15

## Governing inputs

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p4-edh-ledh-derivation-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`

## Governance reconciliation note

This result remains an in-lane provisional P4 artifact.  A later supervisor
audit tightened the governing execution order so that P3 is the next baseline
gate before P6.  Do not revert this P4 work solely because it was produced
before P3; after P3 completes, record whether P3 changes any baseline
definition, estimator-status claim, or differentiability boundary that requires
repairing this chapter before P6.

## Files changed

- `docs/chapters/ch19b_dpf_literature_survey.tex`

## Revision summary

The particle-flow chapter was revised from a mostly formula-led treatment into a
more derivation-led chapter.

Added:

- object and notation inventory for predictive/filtering laws, homotopy density,
  normalizer, velocity field, flow map, EDH coefficients, Gaussian closure, local
  Jacobian, local precision, and local information vector;
- source-role paragraph for Daum-Huang, Li-Coates, and Hu/van Leeuwen with
  explicit limits;
- normalized and unnormalized homotopy path distinction;
- normalizer derivative and centered log-likelihood identity;
- continuity-equation regularity and weak/test-function derivation;
- velocity-field non-uniqueness warning;
- EDH Gaussian-closure derivation through precision, covariance, information
  vector, mean evolution, affine velocity matching, and coefficient convention;
- explicit inverse-matrix derivative for the homotopy covariance equation;
- linear-Gaussian recovery for both covariance and mean endpoints;
- LEDH local linearization derivation through shifted local observation
  equation, local precision, local information vector, local mean, and
  particle-specific coefficients;
- exactness and approximation ledger separating homotopy endpoints,
  conservation equation, velocity choice, EDH closure, LEDH closure, and
  pseudo-time integration.

## Checks run

Static searches:

- `rg -n "exact|Gaussian closure|approx|HMC-ready|production|validated|suitable|first serious|guarantee|guaranteed|correct|target" docs/chapters/ch19b_dpf_literature_survey.tex`
- `rg -n -F '\label' docs/chapters/ch19b_dpf_literature_survey.tex`
- `rg -n "citep|citet" docs/chapters/ch19b_dpf_literature_survey.tex`

Build:

```text
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Result:

- Build passed.
- Output: `docs/main.pdf`
- Page count after P4: 202 pages.
- No undefined citation or undefined reference warnings were found by targeted
  log search.

MathDevMCP:

- `latex_label_lookup` succeeded for the covariance-derivative neighborhood
  before editing.
- `latex_label_lookup` failed once for `eq:bf-pff-ledh-b`; the chapter was not
  allowed to rely on this lookup.
- `derive_label_step` on `eq:bf-pff-homotopy-cov-derivative` returned a
  mismatch/symbol-context diagnostic.  The chapter was then repaired by adding
  the explicit derivation from
  `M_{t,\lambda}P_{t,\lambda}=I`.
- `latex_label_lookup` succeeded for the new mean endpoint label
  `eq:bf-pff-kalman-mean-endpoint`.
- `latex_label_lookup` succeeded for
  `eq:bf-pff-local-information-vector`.

## Audit result

P4 requirements:

- Notation/object inventory: passed.
- Homotopy endpoint and normalizer derivation: passed.
- Continuity-equation derivation and non-uniqueness: passed.
- EDH under Gaussian closure derived before formula statement: passed.
- Linear-Gaussian mean and covariance recovery: passed.
- LEDH local precision and information vector derivation: passed.
- Stiffness and discretization detail: passed.
- Claim-status limitations: passed.

Overclaim search result:

- `exact` occurrences are locally tied to model identities, conservation
  statements, Gaussian-family status, or linear-Gaussian recovery.
- `HMC-ready` appears only in a negative boundary statement.
- `guaranteed` appears only in a warning not to overclaim LEDH improvement or
  raw-flow exactness.
- No P4-local `validated`, `production`, or `first serious` claim appears.

Build/layout caution:

- The build reports DPF-local layout warnings in the edited chapter:
  - overfull lines around the affine-flow prose;
  - underfull table cells in the new exactness/approximation ledger.
- These do not block P4 mathematically, but P10/P12 should revisit table
  readability across all DPF chapters.

## Veto diagnostics

No P4 veto fired.

Specifically:

- EDH no longer remains formula-led.
- Linear-Gaussian recovery includes both covariance and mean.
- LEDH local linearization is derived from the shifted local observation model.
- Stiffness is tied to precision spectra and pseudo-time integration.
- Raw EDH/LEDH cannot be read as a corrected posterior sampler or final HMC
  target from this chapter.

## Exit decision

P4 passes with a layout-readability caution.

Provisional next phase at the time this result was written:

- P5 PF-PF proposal correction and Jacobian audit.

Current governing next gate:

- P3 baseline expansion, followed by P4/P5 impact reconciliation before P6.

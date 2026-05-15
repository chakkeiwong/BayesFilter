# Reset memo: reviewer-grade DPF monograph revision

## Date

2026-05-15

## Scope

This memo tracks the reviewer-grade differentiable particle-filter monograph
revision round.  It is separate from:

- the student DPF baseline workstream;
- controlled-baseline experiment files;
- the older DPF rebuild reset memo;
- unrelated DSGE, SGU, derivative, or filtering implementation work.

## Governing program

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-phase-proposal-2026-05-14.md`
- Phase plans P0--P13 under `docs/plans/`.

## Safety policy

- Do not edit, stage, revert, or use student-baseline files as monograph
  evidence.
- Do not edit, stage, revert, or use controlled-baseline experiment files unless
  a future implementation phase explicitly authorizes that lane.
- Do not push, merge, rebase, reset, delete, or overwrite user work as part of
  this reset memo.
- Treat DPF source support as bibliography-spine plus local derivation unless a
  future ResearchAssistant source-intake/review artifact says otherwise.
- Treat MathDevMCP as diagnostic support only unless a future backend produces a
  real proof certificate.

## Current closeout state

The reviewer-grade writing and audit round completed P0--P13.

Final closeout artifact:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-final-readiness-report-2026-05-15.md`

Audit artifacts:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-derivation-audit-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-hostile-reader-audit-2026-05-15.md`

Final readiness decision:

- Writing/audit complete for rigorous internal review.
- Not empirical validation.
- Not banking, production, or model-risk approval.

## Completed phases

- P0 preflight: passed with cautions.
- P1 claim ledger: passed.
- P2 source grounding: passed with source caveat.
- P4 EDH/LEDH derivation: passed with layout caution.
- P5 PF-PF proposal correction/Jacobian: passed with layout caution.
- P6 resampling/OT/Sinkhorn: passed with layout caution.
- P8 HMC/banking target: passed with layout caution.
- P7 learned OT: passed with layout caution.
- P3 SMC baseline backfill: passed with layout caution.
- P9 debugging verification contract: passed with layout caution.
- P10 notation/claim consolidation: passed with layout caution.
- P11 derivation audit: passed.
- P12 hostile-reader audit: passed with layout caution.
- P13 final readiness: passed.

## Files changed in the DPF monograph lane

Chapter files:

- `docs/chapters/ch19_particle_filters.tex`
- `docs/chapters/ch19b_dpf_literature_survey.tex`
- `docs/chapters/ch19c_dpf_implementation_literature.tex`
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`
- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`

Reviewer-grade plan/result files:

- P0--P13 phase plans and result artifacts.
- Claim ledger, source-grounding register, derivation audit, hostile-reader
  audit, final readiness report, and this reset memo.

## Build status

Final build route:

```text
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Final result:

- Passed.
- `docs/main.pdf` has 218 pages.
- Targeted log scan found no undefined citation/reference, rerun, or multiply
  defined-label warnings.

## Tool status

ResearchAssistant:

- Available in read-only/offline local mode.
- Searches returned no local DPF paper summaries.
- No source was ingested or reviewed in this round.

MathDevMCP:

- Available with SymPy.
- Lean, Sage, LaTeXML, Pandoc, and Lean Dojo unavailable.
- Used for label/typed/scalar diagnostic support.
- No machine-formal proof certification claimed.

## Remaining risks

- Dense longtables and long running headers in the PDF should be improved
  before external distribution.
- Paper-level source provenance remains a future task if the audience requires
  direct primary-source review beyond bibliography-spine support.
- Chapter 26 defines implementation diagnostics, but the diagnostics have not
  been executed in code.
- DPF-HMC for nonlinear DSGE or MacroFinance models remains unvalidated.
- No bank-facing, production, or model-risk approval claim is supported.

## Next recommended program

Create a new implementation-and-evidence program before strengthening any
empirical or banking-facing claims.  Minimum phases:

1. Source-review intake, if primary-paper provenance must be upgraded from
   bibliography-spine to reviewed-source evidence.
2. Chapter 26 diagnostic harness design.
3. Linear-Gaussian recovery tests for PF and EDH.
4. Affine-flow PF-PF density/log-det tests.
5. Soft-resampling and Sinkhorn controlled tests.
6. Learned-OT teacher/student/OOD residual tests.
7. Same-scalar HMC value-gradient and compiled-repeatability tests.
8. Posterior sensitivity and governance-evidence report.

Until that program passes, keep the current readiness label:

```text
Reviewer-grade monograph exposition: complete.
Empirical DPF-HMC validation: not complete.
Banking or production readiness: not claimed.
```

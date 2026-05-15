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

Proposed controlling plan:

- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-master-program-2026-05-16.md`

Proposed subplans:

- IE0 preflight, inventory, and plan audit;
- IE1 source-review intake or explicit deferral;
- IE2 Chapter 26 diagnostic harness design;
- IE3 linear-Gaussian PF/EDH recovery;
- IE4 affine-flow PF-PF density/log-det checks;
- IE5 soft-resampling and Sinkhorn controlled tests;
- IE6 learned-OT teacher/student/OOD residual tests;
- IE7 same-scalar HMC value-gradient checks;
- IE8 posterior sensitivity and research-evidence note.

Continuity caveat:

- The proposed evidence root is clean-room and non-production:
  `experiments/dpf_monograph_evidence/`.
- The proposed program does not authorize production `bayesfilter/` edits or
  imports, student-baseline evidence use, real DPF-HMC/DSGE/MacroFinance target
  chains, or bank-facing/model-risk/production readiness claims.
- Posterior sensitivity remains exploratory unless the relevant row has a
  trusted reference class and all prerequisite actual-target gates pass.  The
  current proposed plan only authorizes analytic or controlled-fixture
  sensitivity unless a later actual-target-instantiation phase is accepted.

IE0 preflight status:

- Result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie0-preflight-result-2026-05-16.md`.
- Exit label: `ie0_plan_accepted`.
- Interpretation: the implementation-and-evidence program is execution-ready
  as a bounded clean-room evidence lane.  IE0 did not create experiment code,
  execute numerical diagnostics, import production `bayesfilter/`, or use
  student-baseline evidence.  IE1 source-review intake is the next justified
  phase and must remain local/offline/read-only unless a separate grant-bound
  source intake workflow is explicitly authorized.

IE1 source-review status:

- Result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie1-source-review-result-2026-05-16.md`.
- Exit label: `ie1_source_review_deferred`.
- Interpretation: no local ResearchAssistant-reviewed DPF source summaries were
  available through read-only/offline checks.  Source support remains
  bibliography-spine plus local derivation or controlled-fixture evidence.
  This deferral does not block IE2, but IE2--IE8 must carry the source-support
  vocabulary and must not describe bibliography-spine support as reviewed-source
  evidence.

IE2 diagnostic-harness status:

- Result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie2-diagnostic-harness-design-result-2026-05-16.md`.
- Exit label: `ie2_harness_ready`.
- Interpretation: the clean-room evidence root
  `experiments/dpf_monograph_evidence/` now has a schema module and validator
  for canonical diagnostic IDs, source-support classes, CPU-only/GPU-hiding
  manifest fields, cap keys, artifact-path containment, coverage status, and
  import-boundary checks.  IE2 validates harness readiness only; no numerical
  DPF, PF-PF, Sinkhorn, learned-OT, HMC, production, banking, or empirical
  claim is supported by IE2 alone.

IE3 linear-Gaussian recovery status:

- Result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie3-linear-gaussian-recovery-result-2026-05-16.md`.
- Exit label: `ie3_linear_gaussian_recovery_passed`.
- Artifacts:
  `experiments/dpf_monograph_evidence/reports/outputs/linear_gaussian_recovery.json`
  and
  `experiments/dpf_monograph_evidence/reports/linear-gaussian-recovery-result.md`.
- Interpretation: the clean-room one-step linear-Gaussian fixture produced
  schema-valid replicated bootstrap PF descriptive diagnostics and deterministic
  EDH special-case recovery against an analytic Kalman reference.  PF summaries
  are descriptive only.  EDH exactness is restricted to this linear-Gaussian
  fixture.  No production, nonlinear, HMC, banking, or empirical validation
  claim is supported.

Until that program passes, keep the current readiness label:

IE4 affine-flow PF-PF status:

- Result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie4-affine-flow-pfpf-result-2026-05-16.md`.
- Exit label: `ie_phase_passed`.
- Local label: `ie4_affine_flow_pfpf_passed`.
- Artifacts:
  `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_synthetic_affine_flow.json`,
  `experiments/dpf_monograph_evidence/reports/outputs/affine_flow_pfpf_algebra_parity.json`,
  and
  `experiments/dpf_monograph_evidence/reports/affine-flow-pfpf-result.md`.
- Interpretation: the clean-room deterministic affine fixture produced
  schema-valid row-level evidence for `synthetic_affine_flow` and
  `pfpf_algebra_parity`.  Codex rejected the first executor artifact as too
  tautological because the analytic proposal-density comparator copied the
  implemented value, patched the diagnostic to use independent closed-form 2D
  Gaussian and determinant formulas, reran the phase, and revalidated both JSON
  rows.  Final residuals were finite and below the `1e-12` threshold.
- Non-implication: IE4 validates only affine closed-form density/log-det and
  corrected-weight parity on clean-room fixtures.  It does not validate
  nonlinear flow integration, solver stability, production `bayesfilter`,
  filtering correctness, DPF-HMC, posterior quality, banking use, model-risk
  use, or production readiness.
- Next justified phase: IE5 remains justified because IE4 passed without
  firing a promotion or continuation veto.  IE5 must independently test
  soft-resampling and Sinkhorn controlled diagnostics and must not treat
  relaxed-target or marginal-residual evidence as posterior validation.

IE5 soft-resampling and Sinkhorn status:

- Result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie5-resampling-sinkhorn-result-2026-05-16.md`.
- Exit label: `ie_phase_passed`.
- Local label: `ie5_resampling_sinkhorn_passed`.
- Artifacts:
  `experiments/dpf_monograph_evidence/reports/outputs/soft_resampling_bias.json`,
  `experiments/dpf_monograph_evidence/reports/outputs/sinkhorn_residual.json`,
  and
  `experiments/dpf_monograph_evidence/reports/resampling-sinkhorn-result.md`.
- Interpretation: the first IE5 executor attempt correctly rejected the phase
  because the original soft-resampling criterion tried to require exact affine
  preservation relative to the categorical reference while also requiring a
  nontrivial relaxed mixture.  Codex and Claude both diagnosed that condition
  as mathematically incoherent for a nontrivial two-particle fixed-support
  mixture toward uniform.  Codex repaired the plan and diagnostics so the
  promotion criterion tests exact relaxed-target arithmetic and records
  categorical-reference deltas as caveated evidence.  After the repair,
  soft-resampling and Sinkhorn rows validated successfully.
- Key values: soft-resampling relaxed-target expectation residuals are zero in
  the repaired fixture; categorical identity and nonlinear deltas are finite
  and recorded as law-change evidence, not preservation evidence.  Sinkhorn
  with epsilon `0.3` and budget ladder `[5, 20, 100]` passed final marginal,
  mass, nonnegativity, finite-plan, and budget-nonincrease checks.
- Non-implication: IE5 does not validate categorical resampling law
  preservation, unbiasedness for nonlinear observables, posterior equivalence,
  exact unregularized OT/EOT equivalence, production `bayesfilter`, banking
  use, model-risk use, or production readiness.
- Next justified phase: IE6 remains justified only as a learned-map residual or
  structured deferral phase.  It must not train networks, import external
  artifacts without provenance, or treat surrogate-map residuals as posterior
  validation.

IE6 learned-OT residual status:

- Result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie6-learned-ot-result-2026-05-16.md`.
- Exit label: `ie_phase_deferred_with_recorded_reason`.
- Local label: `ie6_learned_ot_residual_deferred_no_artifact`.
- Artifacts:
  `experiments/dpf_monograph_evidence/reports/outputs/learned_ot_residual.json`
  and
  `experiments/dpf_monograph_evidence/reports/learned-ot-residual-result.md`.
- Interpretation: IE6 did not execute learned-OT residual tests because no
  approved pre-existing teacher/student artifact with provenance was available.
  Codex tightened the IE6 plan to forbid substituting a newly invented analytic
  teacher/student fixture, then emitted a schema-valid deferred
  `learned_map_residual` row with `blocker_class=source` and
  `continuation_veto_status=fail` for residual execution.
- Non-implication: the deferral does not validate or invalidate learned OT,
  neural OT, surrogate-map quality, posterior quality, production
  `bayesfilter`, banking use, model-risk use, or production readiness.
- Next justified phase: IE7 remains justified because it tests an independent
  fixed-scalar HMC value-gradient contract and does not depend on learned-OT
  residual execution.  IE7 must not reinterpret IE6 as learned-OT validation.

IE7 same-scalar HMC value-gradient status:

- Result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie7-hmc-value-gradient-result-2026-05-16.md`.
- Exit label: `ie_phase_passed`.
- Local label: `ie7_hmc_value_gradient_passed`.
- Artifacts:
  `experiments/dpf_monograph_evidence/reports/outputs/hmc_value_gradient.json`
  and
  `experiments/dpf_monograph_evidence/reports/hmc-value-gradient-result.md`.
- Interpretation: IE7 executed a deterministic fixed scalar target and passed
  same-scalar identity, stable-window finite-difference gradient,
  repeatability, compiled-status reporting, leapfrog reversibility, and bounded
  forward energy-smoke checks.  Codex corrected the plan during execution to
  treat forward energy drift as a bounded smoke threshold (`1e-4`) rather than
  exact conservation, while preserving near-machine-precision reversibility as
  the stricter round-trip check.
- Key values: same-scalar residual `0.0`; finite-difference stable-window
  residual approximately `1.08e-7`; leapfrog position and momentum
  reversibility residuals `0.0` and approximately `5.55e-17`; forward
  energy-drift smoke approximately `3.12e-5`; compiled path explicitly recorded
  as not available in the clean-room CPU-only fixture.
- Non-implication: IE7 does not validate HMC correctness, DPF-HMC correctness,
  posterior/reference agreement, tuning readiness beyond controlled-fixture
  eligibility, production `bayesfilter`, banking use, model-risk use, or
  production readiness.
- Next justified phase: IE8 may summarize controlled-fixture evidence and
  governance status.  IE8 must not run real DPF-HMC, DSGE, MacroFinance, HMC
  chains, posterior validation, or bank/model-risk/production claims.

IE8 aggregate governance status:

- Result:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie8-posterior-sensitivity-governance-result-2026-05-16.md`.
- Final closeout:
  `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-final-closeout-2026-05-16.md`.
- Exit label: `dpf_monograph_evidence_program_complete_with_blockers`.
- Artifacts:
  `experiments/dpf_monograph_evidence/reports/outputs/dpf_monograph_evidence_summary.json`
  and
  `experiments/dpf_monograph_evidence/reports/dpf-monograph-research-evidence-note.md`.
- Interpretation: IE8 ran zero posterior-sensitivity checks and produced the
  aggregate governance ledger.  IE3--IE5 are classified as
  controlled-fixture-supported clean-room evidence; IE6 remains a visible
  deferred learned-OT evidence gap due to missing approved teacher/student
  artifact provenance; IE7 is classified as exploratory-only same-scalar
  fixture evidence; posterior sensitivity remains not tested.
- Source-support ceiling: bibliography-spine support unless a row explicitly
  records a stronger reviewed-source artifact.  IE1 did not identify reviewed
  local DPF source summaries, so the program does not upgrade provenance to
  paper-reviewed support.
- Non-implication: the implementation-evidence program does not validate real
  DPF-HMC targets, DSGE or MacroFinance posterior inference, HMC chains,
  posterior sensitivity, production `bayesfilter`, banking use, model-risk use,
  or production readiness.

Current implementation-evidence readiness label:

```text
DPF monograph implementation evidence: complete with blockers.
Controlled clean-room fixture evidence: available for IE3, IE4, IE5, and IE7.
Learned-OT residual evidence: deferred due to missing approved artifact provenance.
Posterior sensitivity and real DPF-HMC validation: not complete.
Banking, model-risk, or production readiness: not claimed.
```

```text
Reviewer-grade monograph exposition: complete.
Empirical DPF-HMC validation: not complete.
Banking or production readiness: not claimed.
```

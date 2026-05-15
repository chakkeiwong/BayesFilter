# Reviewer-grade DPF monograph adversarial review loop result

## Date

2026-05-15

## Purpose

Run a governed two-agent review loop over the DPF monograph block:

1. Claude Code acts as a read-only hostile academic reviewer.
2. Codex acts as supervisor, audits the review, accepts or rejects findings,
   performs any repairs, and records the evidence.
3. A second read-only worker pass reviews the resulting diff if material repairs
   are made.

The goal is not to create more fluent prose.  The goal is to expose remaining
mathematical, source-support, target-status, implementation-readiness, and
banking-governance gaps before the document is read by skeptical technical
reviewers.

## Scope

In scope:

- `docs/chapters/ch19_particle_filters.tex`
- `docs/chapters/ch19b_dpf_literature_survey.tex`
- `docs/chapters/ch19c_dpf_implementation_literature.tex`
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`
- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-*.md`

Out of scope:

- student-baseline files;
- controlled-baseline experiment files;
- unrelated chapters except where explicitly needed to check a cross-reference;
- commits, pushes, merges, rebases, resets, deletes, or staging.

## Initial branch and worktree classification

- Branch: `main`.
- In-scope DPF chapter edits were already present from the reviewer-grade
  revision and full re-audit.
- In-scope reviewer-grade plan/result artifacts were untracked and remain in
  scope.
- Dirty/untracked student-baseline and controlled-baseline files were present
  but remain out of scope for this loop.

## Governance rules

- Claude worker is read-only in cycle 1.
- Claude findings require exact file and line references.
- Codex supervisor must classify each finding as accepted, partially accepted,
  rejected, or deferred.
- Accepted findings require a repair or an explicit blocker note.
- Rejected findings require a rebuttal grounded in the document text.
- MathDevMCP is used where it can provide useful diagnostic evidence; any
  diagnostic-only status must be recorded.
- No claim may be strengthened about banking use, production readiness,
  HMC validity, novelty, exactness, unbiased gradients, or source support.
- Final gate requires at least:
  - LaTeX build or explicit build blocker;
  - unresolved-reference/citation scan;
  - overclaim scan;
  - branch/status snapshot;
  - list of changed files.

## Cycle 1 worker prompt

Worker:

- name: `dpf-hostile-review-cycle1`;
- tool: `/home/ubuntu/python/claudecodex/scripts/claude_worker.sh`;
- mode: read-only.

Prompt summary:

- audit the whole DPF monograph block and reviewer-grade plan/result files;
- find mathematical, source, implementation-readiness, reproducibility, and
  overclaim gaps;
- classify findings as `BLOCKER`, `MAJOR`, `MINOR`, or `STYLE`;
- include exact file/line references and required repairs;
- identify false-positive candidates where the text is defensible;
- do not edit files.

## Cycle 1 worker report

Claude Code returned a read-only hostile-review report with the following
headline verdict:

- no fatal false mathematics was identified in the core formulas;
- the main risk is that the readiness/audit artifacts sounded more closed than
  the admitted source, formal-proof, and implementation evidence permits;
- several mathematical assumption sites could be made sharper for skeptical
  mathematical reviewers;
- implementation-readiness remains defined by contract, not demonstrated by
  executed tests.

Cycle 1 findings:

| ID | Worker severity | Supervisor decision | Rationale |
|---|---|---|---|
| F1 | BLOCKER | Accepted | Final-readiness language used `writing-complete and audit-complete`; this needed explicit downgrading to writing/manual-audit complete, not evidence-complete. |
| F2 | BLOCKER | Accepted with scope correction | `Passed` is acceptable only if qualified by gate type. Status language was updated to distinguish manual derivation audit from formal/source/implementation closure. |
| F3 | MAJOR | Accepted | PF-PF prioritization should be framed as an experimental hypothesis, not a recommendation with evidence stronger than the current validation record. |
| F4 | MAJOR | Accepted | Bootstrap PF likelihood-unbiasedness proof was strengthened with explicit filtration timing, bounded/integrable test-function scope, and resampling-scheme scope. |
| F5 | MAJOR | Accepted | Homotopy normalizer differentiation now states sufficient dominated-differentiation conditions. |
| F6 | MAJOR | Accepted | Continuity-equation derivation now states the strong/weak admissible setting and boundary assumptions more explicitly. |
| F7 | MAJOR | Accepted | HMC chapter now states PF-PF's advantage is proposal-accounting clarity for a one-step conditional construction, not multi-step posterior fidelity or HMC correctness. |
| F8 | MAJOR | Accepted | Resampling/OT chapter now includes a differentiated-object table for exact EOT, finite unrolled Sinkhorn, implicit fixed point, and actual code scalar. |
| F9 | MAJOR | Accepted | Learned-OT chapter now labels the ladder as an internal bank-facing evidence ladder and explicitly says it is not a formal model-risk framework. |
| F10 | MINOR | Deferred | A diagnostic-type column would enlarge already dense tables. The gap is recorded as an implementation-readiness improvement for the future verification program. |
| F11 | MINOR | Deferred | Table density is a presentation risk, not a mathematical blocker; resolving it is a typography/external-circulation task. |
| F12 | STYLE | Deferred | Filename/chapter-title mismatch is navigational; renaming could disturb includes and is not needed for the mathematical gate. |

## Supervisor adjudication

Accepted findings were repaired in the chapters and plan/result artifacts.
Deferred findings are recorded as future implementation/presentation work, not
as mathematical blockers.

## Repairs made in this loop

Chapter repairs:

- `docs/chapters/ch19_particle_filters.tex`
  - expanded the bootstrap likelihood-unbiasedness proposition and proof scope;
  - added fixed-observation/parameter convention, bounded/integrable
    test-function scope, filtration timing, and resampling-scheme coverage.
- `docs/chapters/ch19b_dpf_literature_survey.tex`
  - added dominated-differentiation conditions for the homotopy normalizer;
  - tightened the continuity-equation derivation to explicit smoothness,
    integrability, weak-form, and boundary settings.
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`
  - recast PF-PF as a bounded experimental hypothesis;
  - clarified that its advantage is one-step proposal-accounting clarity, not
    validated multi-step posterior fidelity or HMC correctness.
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`
  - added a differentiated-object table separating exact regularized EOT,
    finite unrolled Sinkhorn, implicit fixed-point differentiation, and the
    actual HMC/training scalar.
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
  - renamed the banking ladder to an internal bank-facing evidence ladder;
  - stated that it is not a substitute for a formal model-risk or
    production-governance framework.

Plan/result artifact repairs:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-final-readiness-report-2026-05-15.md`
  - downgraded readiness language from `audit-complete` to writing-complete and
    manually audit-reviewed;
  - added an evidence-incomplete classification for missing source review,
    formal proof, code diagnostics, and posterior comparison.
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-derivation-audit-2026-05-15.md`
  - qualified the exit gate as manual derivation audit only.
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-full-reaudit-result-2026-05-15.md`
  - qualified the exit decision as internal-reading/manual-audit status, not
    source-review, formal-proof, implementation-validation, posterior, or
    bank-facing evidence closure.
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-2026-05-15.md`
  - replaced a remaining live ledger statement that characterized PF-PF as the
    "first serious" DPF-HMC candidate with the weaker current claim: the first
    rung in the local ladder with an explicit proposal-corrected value-side
    interpretation, only as an experimental hypothesis;
  - recast the `HMC-ready particle backend` hit as a non-claim boundary item,
    not a current method label.

## Verification

First-pass supervisor checks after the cycle-1 repairs:

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from
  `docs/` completed successfully and produced `docs/main.pdf`.
- The log scan
  `rg -n "undefined|Undefined|multiply defined|Citation.*undefined|Reference.*undefined|LaTeX Warning|Rerun" docs/main.log`
  found no undefined references, undefined citations, multiply-defined labels,
  or rerun warnings.  The only hit was the package banner for
  `rerunfilecheck`.
- The build still emits overfull/underfull box warnings, especially around
  dense DPF tables.  This is a presentation/layout risk, not a mathematical
  proof blocker.
- The overclaim scan over the DPF chapter block and reviewer-grade artifacts
  found risky phrases only in non-claim, warning, gate, plan-instruction, or
  audit-scan contexts after the P1 claim-ledger repair.  Notable surviving
  chapter hits are:
  - `HMC-ready particle backend` in Chapter 20 appears only in a list of items
    the classical particle-filter chapter does not claim;
  - `bank-facing` appears in evidence ladders and in statements that additional
    evidence is required;
  - `exact posterior`, `valid posterior`, `guarantee`, and `guaranteed` appear
    in denied or caveated contexts.
- MathDevMCP audits were run on:
  - `prop:bf-pf-bootstrap-likelihood-status`;
  - `eq:bf-pff-log-homotopy-derivative`;
  - `eq:bf-pff-continuity-equation`;
  - `eq:bf-dr-sinkhorn-residuals`;
  - `eq:bf-dpf-hmc-contract`.
  All five labels were parsed with provenance, but the audits returned
  `unverified` or otherwise diagnostic-only statuses.  They therefore provide
  line-level extraction and obligation diagnostics, not formal proof
  certificates.
- ResearchAssistant status was checked.  The local MCP is read-only/offline, and
  `ra_find_paper` for a DPF/PF-PF/Sinkhorn-resampling query returned no local
  paper summaries.  Source support remains bibliography-spine/manual, not
  ResearchAssistant-reviewed.

## Cycle 2 worker prompt

Worker:

- name: `dpf-hostile-review-cycle2`;
- tool: `/home/ubuntu/python/claudecodex/scripts/claude_worker.sh`;
- mode: read-only.

Prompt summary:

- read this loop artifact first;
- review the current scoped diff only;
- decide whether repairs for F1--F9 are resolved;
- find any new mathematical, status-language, implementation-readiness,
  source-support, overclaim, or LaTeX/build risks;
- do not edit files.

## Cycle 2 worker report and supervisor decisions

Cycle 2 returned no blockers and no major findings.  It reported five minor
precision issues:

| ID | Worker severity | Supervisor decision | Repair |
|---|---|---|---|
| C2-F1 | MINOR | Accepted | `docs/chapters/ch19_particle_filters.tex` now states the finite Feynman--Kac quantity class in the proposition: $Q_t\varphi$ is well defined for bounded measurable induction functions, incremental weights have finite conditional expectation, and displayed conditional expectations are finite. |
| C2-F2 | MINOR | Accepted | `docs/chapters/ch19b_dpf_literature_survey.tex` now says the weak-form identity with compactly supported test functions is the safer default, and the strong-form PDE is used only under the stated differentiability, integrability, and boundary-flux hypotheses. |
| C2-F3 | MINOR | Accepted | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-final-readiness-report-2026-05-15.md` now labels P11 as `Passed for manual audit only`. |
| C2-F4 | MINOR | Accepted | The same report now labels P12 as `Passed for internal reading with layout caution`. |
| C2-F5 | MINOR | Accepted | The same report now labels P13 as `Passed for writing/manual-audit gate` and states that this does not close source-review, formal-proof, implementation-validation, posterior-comparison, or bank-facing evidence gates. |

Cycle 2 classified F1 and F3--F9 as resolved, and F2 as partially resolved
before the additional final-readiness status-label repairs above.  The worker
also explicitly treated several potential findings as false positives because
the current text already denies or gates the relevant claims.

## Final verification

After the cycle-2 repairs:

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from
  `docs/` completed successfully and regenerated `docs/main.pdf`.
- The final log scan again found no undefined references, undefined citations,
  multiply-defined labels, or rerun warnings; the only hit was the
  `rerunfilecheck` package banner.
- The final overclaim scan still found risky phrases only in historical,
  audit-instruction, non-claim, denial, warning, or evidence-gate contexts.
- Remaining build warnings are overfull/underfull table/layout warnings,
  especially in the dense DPF block; they are presentation risks, not
  mathematical or status-language blockers.
- Current branch at final verification: `main`.
- Dirty worktree still contains unrelated student-baseline and
  controlled-baseline files; this loop did not edit, stage, commit, push, merge,
  rebase, reset, or delete those files.

## Exit decision

The two-agent adversarial review loop is complete for the DPF monograph writing
and manual-audit gate.  No blocker remains in the scoped mathematical/status
repairs identified by the two read-only Claude reviews and Codex supervisor
checks.

This is not a formal-proof certificate, not ResearchAssistant-reviewed source
closure, not implementation validation, not posterior-comparison evidence, and
not bank-facing or production-governance approval.  The exact next recommended
phase is an implementation-and-evidence program that executes the Chapter 26
verification contract on code: controlled analytic examples, finite-difference
same-scalar value/gradient checks, variance/repeated-evaluation diagnostics,
posterior comparisons, and versioned artifacts for any bank-facing research
claim.  A separate source-review intake should be run if the goal is to replace
bibliography-spine support with reviewed primary-source evidence.

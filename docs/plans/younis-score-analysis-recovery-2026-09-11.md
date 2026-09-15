# Younis score analysis: recovery checkpoint

Date: 2026-09-11; applicability corrected 2026-09-12.
Status: RECOVERY_COMPLETE_WITH_DEGENERATE_TRANSITION_CORRECTION.

## Active question and outcome

The user asked to recover a stalled session and finish the holistic analysis
of how Younis's work could address biased particle-filter model scores.
The completed analysis is
[Using Younis's mixture gradients to estimate the model score](younis-model-score-analysis-2026-09-11.md).
It includes the derivations, alternatives, recovered results, uncertainty,
and the next discriminating research design.

For regular-transition reference models, model-corrected continuous-mixture
proposals with an ancestor-averaged Fisher-score recursion remain candidates.
This smoothing recursion is not applicable as written to degenerate DSGE:
distinct ancestor supports can collapse backward weights to the original
genealogy. The previous unqualified recommendation was wrong in scope.
The [September 12 correction](younis-degenerate-transition-correction-2026-09-12.md)
places the restriction beside the algorithm and in the proposed study.
DSGE score construction remains unresolved and implementation deferred. Hybrid
pathwise/importance-weight gradients and correctly centered control variates
are variance-reduction candidates. No construction has been newly implemented,
benchmarked, promoted, or declared unbiased at finite particle count.

## Checked findings

- Session `01a06730-c524-7a22-911f-2c6ff57b6a2f` survives. All 20 tools in the
  pasted exchange returned. Their outputs total 366,048 characters; six show
  truncation. Runtime reached 246,141 tokens against a 244,800 compaction
  trigger, then compaction failed during and before continuation. Ordinary
  streams had failed earlier too. The server-side cause remains unknown.
- The latest scientific branch is `kdm-total-score-continuation-20260909`,
  commit `804616e320d940a5ea288ba25cc43bd21d267abb`. It reports Phase 4B score
  losses in two matrix-LGSSM scopes. Five-adapter correctness work does not
  establish utility or complete initial-law/target coverage. Main checkout
  notes are older; use the preserved branch-result snapshots.
- Younis 2023/2024 technical sections and relevant author code were inspected,
  alongside Lai's MPF theorem/weights and author code. Fisher and hybrid
  formulas are derived in the report. Two old Poyiadjis cache files are HTML error
  pages and remain excluded. A valid original author PDF was subsequently
  recovered during manuscript revision and its method/evaluation inspected;
  PaRIS method and expected-work assumptions were also checked. Fresh forward citation
  coverage is limited by an OpenAlex HTTP 429.
- Exact rational checks confirm the log-score bias counterexample, the local
  Gaussian variance calculation, and Fisher recursion versus direct complete
  enumeration in a two-state model with parameter-dependent initialization.
  These are algebra checks, not particle-filter performance evidence.

## Small recovery entry points

All evidence is under `docs/plans/artifacts/younis-score-recovery-20260911/`:
`session-recovery-result.md`, `session-summary.json`, `literature-ledger.md`,
`verification.json`, and `recovered/` source snapshots.
The adjacent scripts reproduce the read-only session summary and exact checks.
Do not reload the 410 MB rollout or dump nested saved tool outputs to recover
this state. The detailed incident history is in the linked evidence directory.

## Scope and exact next action

This task launched no research campaign; compute/attempt budget remaining is
not applicable. The current checkout and unrelated concurrent changes were
preserved. No session database, provider configuration, credential, or global
policy was edited. New session tooling limits were not claimed to repair the
upstream service.

If execution is requested next, read the analysis's final study design,
verify the checkout and current instructions, and prepare the smallest
regular-transition reference experiment with checked support, a fixed target, complete initialization,
independent score oracle, scope-specific tuning, and bounded compute. Do not
resume the old campaign or rerun consumed holdouts automatically. The user's
prior direction defers further DSGE implementation. Keep outputs around 2,000
tokens per response and 4,000 per batch, and update this concise checkpoint
at the next completed stage.

## Preserved September 11 manuscript revision

The September 11 revision contained 34 pages. It includes the Section 3.6 history, implementation, Phase 4
results, reasons for the pivot, full new derivations, alternatives, and the
next comparison. The source directory is ignored in the main checkout, so
final source/PDF copies and a tar archive are preserved under
`docs/plans/artifacts/younis-manuscript-update-20260911/`. Start with its
`completion.md`; do not reconstruct this work from raw session records.

Main-checkout source 5cc59cfa differs from tested research 804616e3: the fused
wrapper forwards reset settings but remains nested scalar row maps, and the
shared score now calls the unified correction. Static wiring was checked;
current runtime parity was not rerun. All 51 old displayed math blocks and
53 labels survive. Exact arithmetic checks pass and the PDF builds cleanly.
The document task is complete; further experiment execution needs its own
bounded plan under the existing scientific direction.

The live source and PDF at
`docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex` and `.pdf`
now include the September 12 degenerate-transition correction (36 pages).
Section 7.1, page 20, states why the ordinary all-ancestor recursion is not a
general DSGE method. The correction note above records the clean build,
reference checks, preserved equations/citations, and new final archive.

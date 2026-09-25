# Younis score session continuation

Date: 2026-09-12. Status: RECOVERED_WITH_DEGENERATE_TRANSITION_CORRECTION.

## Active question and recovered scope

The user requested recovery and continuation of the session in the attached
conversation. Its last substantive directive was to reconsider how Younis's
work could address biased model-score estimation, including alternatives
beyond the existing control-variate and integrated-KDM experiments.

The exact session is `01a06730-c524-7a22-911f-2c6ff57b6a2f`. Its preserved
rollout is under `/home/chakwong/.codex/sessions/2026/09/03/`. The relevant
user messages were checked at records 37367, 37396, and 37609. The existing
session summary reports that all tool calls returned; compaction then failed.
This does not identify an unfinished numerical job or its server-side cause.

The analysis and subsequent manuscript revision were completed on
September 11. The user's September 12 correction identified an overbroad
recommendation: the proposed smoothing recursion does not apply as written to
degenerate DSGE. The active document correction and verification are recorded
in [the correction note](younis-degenerate-transition-correction-2026-09-12.md).
The recovery entry point is
[the recovery checkpoint](younis-score-analysis-recovery-2026-09-11.md).
Do not reconstruct the task by replaying the complete rollout.

## Checked state

- Checkout: `/home/chakwong/BayesFilter`, branch `surrogate-hmc`, commit
  `14a292098f35b6de450ffca29131ea34f4a4d8d7`.
- Before the September 12 correction, all five files in the September 11
  final manifest matched their SHA-256 checksums, and the live TeX,
  bibliography, and 34-page PDF matched that archive. The archive is
  preserved; the correction note records the subsequent source/PDF revision.
- No change from `5cc59cfa` to this checkout affects the four inspected
  canonical value/score and integrated/resampling KDM modules. This is a
  static provenance check, not new runtime parity evidence.
- The saved exact-arithmetic checks support the normalizer/log-score bias
  counterexample and the two-state Fisher recursion. These checks are
  conceptual evidence, not empirical particle-filter validation.
- Younis 2023 equations (14)-(15) were re-inspected: the proposal density is
  held fixed when taking the importance-weight derivative. This local
  expectation identity does not establish unbiased finite-particle model
  log-scores.

## Scientific continuation

The corrected [analysis](younis-model-score-analysis-2026-09-11.md) restricts
model-corrected continuous-mixture proposals and the ancestor-averaged
Fisher-score recursion to regular-transition reference models, including the
initial-law score. This is a smoothing construction, not an established
solution for degenerate DSGE. Its ordinary density/backward-weight formula
can fail or collapse to a single genealogy on ancestor-specific singular
supports. A valid DSGE score construction remains unresolved, and its
implementation remains deferred. Correctly centered
control variates and hybrid pathwise/importance-weight derivatives remain
variance-reduction candidates. The combined construction is untested; no
empirical ranking or finite-particle unbiasedness claim is established.

The skeptical recovery audit found a scope hazard: the original analysis
request must not silently become a rerun of the old KDM campaign or reuse its
consumed holdouts. The first proposed study instead needs fresh data, an exact
Kalman reference, separate proposal/score ablations, scope-specific calibration,
conditional replications, uncertainty estimates, and equal-compute comparisons.
The analysis's final study-design section records the detailed requirements.

## Budget and exact next action

No new research campaign or GPU workload was launched in this recovery.
Compute/attempt budget is not applicable to this document correction.
Unrelated working-tree changes were preserved. The correction note's exact
reference checks, preservation checks, clean 36-page PDF build, and visual
inspection are complete. Deliver the corrected PDF, with the explicit
restriction in Section 7.1 on page 20. No DSGE experiment is scheduled;
do not treat the old campaign as pending.

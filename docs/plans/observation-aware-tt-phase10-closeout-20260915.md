# Phase 10 closeout: SGQF preservation and independent filtering

2026-09-15. The authorized A04–A06 continuation and phase-10 documentation are
complete. Completion records what was learned; the tested TT candidate remains
unpromoted. Human reading feedback on the manuscript is still pending.

## Outcome and evidence

| Stage | Completed work | Scientific outcome | Evidence |
| --- | --- | --- | --- |
| 7 / A04 | Analytical SGQF-to-TT initialization, conversion-error decomposition, controlled starts and validation selector | Useful d4 initialization signal, but conversion/refinement can lose to SGQF; no population safeguard | [A04 result](observation-aware-tt-sgqf-initialization-20260915-result.md) |
| 8 / A05 | Dedicated defensive-mass calibration and actual exact-Gaussian joint consumer | Optional 1e-5 passes stated non-harm/rescue bounds; fitting concerns remain | [A05 result](observation-aware-tt-defense-consumer-20260915-result.md) |
| 9 / A06 | Eight methods on 24 new independent sequences with screened references and paired uncertainty | All references pass; d1 conditional heuristic losses and one d4 guide failure block promotion | [A06 result](observation-aware-tt-independent-filtering-20260915-result.md) |
| 10 | Master, checkpoint, result/inference records, budget and revised manuscript | Terminal interpretation review AGREE; rendered draft checked, human acceptance pending | This note and [preservation record](artifacts/observation-tt-manuscript-20260915-01/preservation-check.json) |

The next scientific questions are distinct: why signed quadrature fails on the
identified large observation, and how to preserve useful SGQF accuracy during
TT conversion/refinement and validation selection. They are unresolved, not
silently answered by the successful-sequence d4 table. The current master has
no further numerical phase active. A later experiment must enter through the
owner-requested amendment and review procedure, with fresh calibration and
confirmation data. The A06 holdout is now exposed.

## Manuscript changes and verification

The [54-page PDF](artifacts/zhao-cui-observation-aware-tt-20260912-01/attempt05_observation_aware_tt_algorithm_note.pdf)
adds Section 16, pages 48–54, and updates the abstract. It derives the SGQF
adjacent Gaussian and both conditioning directions; separates polynomial
truncation from TT rank compression; explains why each new observation requires
a new fit; derives the calibrated defensive fraction at the actual conditional
consumer; and presents the independent filtering outcomes, including failures.
The figure plots the previously declared calibration inequalities and is an
explanation, not a new experiment or tuning procedure.

The old TeX/PDF baseline and hashes are protected under
`artifacts/observation-tt-manuscript-20260915-01/`. All 187 existing displayed
math environments and all existing labels are retained; eight displayed math
environments and a filtering-result table are added. The prior appendix is
identical. The only replaced baseline text is the revision date and an old
two-line administrative campaign-close sentence, now dated as historical.
The full baseline-to-draft difference and current hashes are preserved.

The document was built with `pdflatex -interaction=nonstopmode -halt-on-error`
from `artifacts/zhao-cui-observation-aware-tt-20260912-01/`. Passes 04/05 exposed
two overfull inline confidence intervals; their wording was repaired without
changing any value. Final pass 06 succeeds with no overfull boxes or undefined
references. Existing underfull/path-spacing and math-in-bookmark warnings
remain. Rendered pages 1, 2 and 48–54 were inspected, including Gaussian formulas,
the defense plot, matched filtering table, uncertainty, failure explanation and
appendix transition. These checks do not certify human readability or acceptance.

The new filtering table uses all 12 scalar sequences and the same 11 successful
four-dimensional sequences for every method. It explicitly distinguishes its
normalized squared errors from the earlier single-sequence RMSE table. The
failed d4 sequence remains in the result and blocks promotion. The first
reporting pass incorrectly treated absent optional finite flags as failures;
the corrected reporter separately counts missing flags, explicit failures and
nonfinite stored values. It finds zero stored nonfinite values in completed
particle runs. This reporting repair changed no experiment or selection.

One bounded terminal interpretation review returned AGREE, preserved in
`artifacts/observation-tt-independent-filtering-20260915-01/review-terminal-01.txt`.
It reviewed the scientific interpretation rather than rerunning the experiment.
The manuscript retains the resulting distinctions: seven narrow d1 directional
comparisons are supported under the approximate simultaneous bootstrap, no
overall ranking or improvement over joint SGQF is established, and d4 means
conditional on guide success are descriptive only.

Final source, manuscript-hash, local-link and stopped-experiment checks are preserved in [final verification](artifacts/observation-tt-manuscript-20260915-01/final-verification.json).

## Budget and accounting limits

The owner's total authorization remains 18000 seconds. Active work, including
design, implementation, review and documentation, is charged once in
`artifacts/observation-tt-sgqf-initialization-20260915-01/budget.json`; crash and
user idle intervals are excluded. The ledger is paused on final closure, with
the unused balance retained rather than granting a new five hours.

A06 used one full numerical launch (1611.880 seconds) and one smoke
(19.711 seconds), below its 2700-second numerical cap. The unchanged diagnostic
CPU replay took .230 seconds of numerical work. Numerical success/failure and
the total owner budget are recorded independently of the stage-allocation issue
below.

A06's separate 6000-second active-work suballocation was not timed independently
from concurrent phase-10 drafting. The calendar span from its plan review to
terminal review is longer than 6000 seconds and includes phase-10 work. The
records therefore do **not** establish compliance with that stage suballocation;
do not label every campaign budget gate passed. No additional numerical launch
or new scientific protocol was started. A future amendment should record stage
work intervals separately or use one unambiguous remaining-work ceiling.

Final H6 accounting at 2026-09-15 06:28:16 UTC: **17183.471 seconds charged; 816.529 seconds unused** (about 13 minutes), including a conservative 60-second final-response allowance. The ledger is paused; subsequent idle time is excluded.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close the executed program; retain optional implementations without promotion | A06 advancement fails | d1 observed conditional losses; d4 invalid guide; separate stage-allocation compliance unverified | Finite-sequence inference, reference MC error, guide robustness, selection generalization | A reviewed amendment before another scientific experiment; provisional manuscript can receive human feedback | Reliable fitting, superiority, scalable initialization, source-faithful TT-cross, gradients or HMC readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Current candidate fails the declared heuristic and shared-guide requirements. |
| Statistically supported ranking | Only the seven specified d1 metric/regime contrasts in A06; no global ranking. |
| Descriptive-only differences | TT-variant comparisons, all d4 successful-sequence comparisons, audit counts and timing. |
| Default readiness | Not established. |
| Next evidence needed | Distinct robustness and fitting/selection questions, reviewed calibration, fresh complete confirmation and adequate uncertainty. |

Post-run red-team: deleting the failed guide sequence or substituting an
unreviewed positive/ridged rule would change the question after seeing data.
Neither was done. The current negative result may be overturned by a properly
specified repair on new data; it cannot be overturned by presentation changes.

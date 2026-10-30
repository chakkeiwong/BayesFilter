# Adaptive R replication: terminal review

The frozen ladder finished on 2026-09-23 at 06:08 Hong Kong time
(2026-09-22 22:08 UTC). All 500 batches and 30,000 scheduled records are
present. The score reconstruction failed in 8/1,000 dimension-80 learners:
four rejected nonpositive fitted precisions and four reached the 12-iteration
cap. The QR reconstruction completed all 5,000 learners across five dimensions.
This completes phase 16; it does not complete replication of the original paper.

The reviewed question is conditional likelihood accuracy for the two frozen
R reconstructions, against bootstrap, fully adapted, current-observation and
exact Gaussian controls. No method or parameter was selected after viewing
this result. Terminal inspection follows the existing phase-16 contract.
Skeptical review passes for this bounded interpretation: labels, source hashes,
finite completed results, oracle agreement and budgets are checked; unequal
particle counts, diagnostic-refit timing, one data set per dimension and rare
importance-weight events prohibit broader claims or a speed ranking.

Verification in [verification.json](verification.json) independently checked
the complete label set with no duplicates, all 500 batch-manifest hashes,
three stage-manifest hashes, final statistics/output hashes, nine frozen
source files and successful per-launch source checks. Maximum absolute
oracle/Kalman log-likelihood error is 3.64e-12. This is inspection of saved
evidence using standard-library Python with GPU hidden, not a new numerical
experiment. Every raw RDS payload was not rehashed or rerun.

The four rejected labels are 291, 300, 694 and 722; rejection occurs in their
first fit. Their saved same-cloud negligible-floor diagnostic refits are
valid, which nominates the initial floor transformation for a focused repair
test. This does not establish downstream accuracy after changing that floor.
The capped labels are 844, 891, 947 and 989, all at N=2,000. An extension of
their controller histories can distinguish insufficient iteration allowance
from persistent non-stabilization; the frozen failures must remain failures.

The score arm's dimension-80 whole-arm RMSE is unavailable. The generated
summary lists three heuristic promotion vetoes for that arm, but these arise
from incomplete learners, not an observed MSE loss against those controls.
The summarizer explicitly uses `!complete || difference > 0`. These are
completion vetoes; neither dropping the eight labels nor inventing their
likelihoods supplies the missing comparison.

At dimension 80, QR relative RMSE is 0.1794 with a pointwise bootstrap 95%
interval [0.1707, 0.1883], and mean final N is 1,960. Its largest observed
squared-error contribution is 1.61%. The fully adapted and current-observation
controls have single observations contributing 95.1% and 81.3% of their total
squared errors. Every dimension-80 bootstrap likelihood ratio underflows to
zero despite finite log likelihoods. Those facts limit interpretation of
control intervals. All 27 estimable paired comparisons against cheap controls
have negative pointwise intervals for candidate-minus-control MSE, conditional
on these fixed data. They do not establish simultaneous or across-data
dominance. No direct paired score-versus-QR ranking was specified or computed.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close phase 16 computation and terminal review | Exact labels, source checks and oracle agreement pass | d80 score completion veto; no harness continuation veto | Initial-floor failures, iteration allowance, rare control tails | Bounded replay of four precision rejections and separate extensions of four capped histories, preserving original evidence | Original-paper replication, direction rejection or default readiness |

| Inference item | Status |
|---|---|
| Hard veto screen | Four nonpositive-precision rejections and four iteration caps in d80 score; QR and score d5–d40 remain viable under this screen |
| Statistically supported ranking | Pointwise fixed-data intervals favor completed candidates over controls in 27 estimable comparisons; no broad or direct fitter ranking |
| Descriptive-only differences | QR has smaller observed RMSE than score at d5–d40; particle counts differ; timing includes diagnostic refits |
| Default-readiness | Not evaluated; independent CPU R reconstructions |
| Next evidence needed | Failure repair checks; fresh downstream evaluation for any numerical change; additional data/tail evidence before broad ranking; source identity still unresolved |

Red-team: a larger iteration allowance may repair the four caps, and an initial
floor change may repair fit validity without improving likelihood accuracy.
Conversely, unseen rare importance weights can overturn apparently narrow
bootstrap intervals. The weakest inference is population performance from one
data set per dimension. Nothing here rejects the iAPF research direction.

The campaign charged 26.974 CPU worker hours and no GPU hours. Remaining owner
allocation is 18.252 CPU hours and 47.745 GPU hours, with no active reservations.
Original evidence remains at
`/home/chakwong/BayesFilter/docs/plans/artifacts/iapf-adaptive-replication-ladder-20260922-01/`.
This review and subsequent work use the isolated `ledh-refactor-with-policy-fix`
worktree at commit `89065bc6`. The original supervisor checkpoint intentionally
retains `complete_pending_interpretation`; this dated review closes that task
without rewriting the completed run's evidence.

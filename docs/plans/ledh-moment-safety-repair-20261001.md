# LEDH higher-moment safety repair — 2026-10-01

## Question and authorization

The owner requests a thorough monograph account of the richer fixed residual
design, marginal and pairwise fitting, revised coordinate cap and its safety
problem, followed by a reviewed and executed repair plan. This is a new bounded
local campaign; the completed September campaign and its expired allowance are
not reopened. Baseline: commit 688878c8f on sqmc-development, clean at entry.

Question: can the optional identity-core reset retain useful shape corrections
while rejecting fitting steps that are numerically invalid or increase the
declared moment discrepancy after whitening? This is a numerical safety question,
not a claim of better filtering likelihood, posterior inference or HMC readiness.

## Evidence contract and research intent

Compare (1) affine restoration without moment fitting, (2) the existing marginal
and pairwise fit with its original coordinate cap, (3) the richer-design fit
with the identity-core cap, and (4) that same optional fit with the new safety
guard. An additional small-strength fit is a simple conservative adversary.
These are constructed from the failure mechanisms: no fitting cannot overshoot;
the original cap limits excursions; a smaller step may avoid a local overshoot.
They are falsification comparators, not targets used to select likelihood tuning.

Primary safety criterion: every accepted correction has finite values and
analytical tangents, a well-conditioned covariance before whitening, and no
increase (beyond arithmetic tolerance) of the declared scaled moment loss after
whitening. The complete capped/restored output must also pass a comparison to
the identically protected no-fit cloud. Preserve identical values and tangents
on healthy first-trial acceptances. Rejected corrections must be observable.
The coordinate cap is retained on both the fitted and no-fit alternatives.

Promotion vetoes: a violated invariant, false valid status, missing pairwise
wiring, finite-difference disagreement away from decision boundaries, healthy
no-fire mismatch, or a serious regression relative to a simple safety comparator.
Continuation vetoes: corrupted inputs/artifacts, unfixable implementation or
derivative mismatch, or exhausted compute budget. A failed candidate is a repair
trigger, not a reason to abandon this question. Moment accuracy and cap activity
are explanatory; fitting a teacher does not establish that the teacher is the
true posterior. No stochastic ranking or default promotion is planned.

## Mechanism and assumptions to audit

Use one shared implementation for marginal and pairwise guarded updates.
Try the existing capped displacement first, then a fixed sequence of halvings.
Check covariance before Cholesky; whiten only admissible trials. Accept the
first finite trial whose scaled marginal/pairwise residual sum of squares does
not increase. Otherwise preserve the input and its total tangent and report
rejection. Differentiate the selected finite branch analytically; do not claim
differentiability at acceptance boundaries. The final cap/affine restoration is
part of the complete-map check, rather than an assumed safety theorem.

| Choice | Provenance and justification | Risk / earliest check | Status |
|---|---|---|---|
| Existing LM damping, row caps and identity radius 8 | Frozen September candidate; comparability only | Still insufficient alone; replay complete maps | Comparator, not universal defaults |
| Relative residual scale max(1, absolute target moment) | Dimensionless absolute error near zero and relative error for large teachers | Coupled moments can trade off; report each residual family separately | Explicit safety objective |
| Covariance eigenvalues between one quarter and nine quarters of the input covariance | Whitening gain at most two and inverse gain at most 1.5 in input-whitened coordinates | Very ill-conditioned input; reject before factoring | Derived safety envelope |
| Eight trials, factors 1 through 1/128 | Fixed bounded work; correctness comes from rejection, not eventual acceptance | Stalling; report rejection and minimum step | Work budget, not convergence claim |
| Roundoff tolerance proportional to dtype epsilon and loss scale | Arithmetic tolerance only | FP32 masking meaningful increase; compare FP64 | Numerical hypothesis |
| Scalar, 2D/3D and 10D clouds; normal, skew, mixture, outlier, concentrated-weight and nearly singular regimes | Each exposes a different failure in fitting or whitening | Convenient fixtures may miss historical failures | Diagnostic coverage only |

## Execution and budget

1. Inspect the current shared call chain and eligible saved failing inputs.
   Historical pre-2026-08-21 results cannot be reused as numerical evidence.
   If an original explosive input cannot be recovered, say so and construct
   fresh deterministic stress cases; do not mislabel them as a replay.
2. Preserve an unmodified baseline and record focused failing examples. Implement
   the smallest shared guard and its observability. Keep existing defaults.
3. Test marginal and pairwise behavior, no-fire parity, invalid covariance,
   branch-conditioned total JVPs, shared/batched parity and caller wiring.
4. Execute a bounded deterministic CPU-reference and trusted GPU/XLA comparison,
   including FP64 and the FP32/TF32 execution direction. Use fresh fixed seeds
   for terminal stress checks. No parameter or likelihood tuning, HMC or training.
5. Document the revised algorithm and proofs/limitations in the monograph's
   entropic-OT chapter, with its mirrored manuscript kept consistent. Explain
   the binary-design degeneracy, quartic tail sensitivity, pairwise targets,
   both caps, whitening, full algorithm and derivative branches in teaching order.
   Compile the full monograph and inspect the changed rendered pages.
6. Record results, a decision/inference table, strongest alternative explanation,
   remaining gaps and an active checkpoint.

Total budget: 1,800 GPU seconds and 1,800 CPU experiment/test seconds, at most
three research attempts, with separate bounded build time (20 minutes).
Routine implementation is outside experiment wall accounting. Each attempt has
a fresh directory under docs/plans/artifacts/ledh-moment-safety-20261001/.
Manifest records exact commands, git/diff provenance, seeds, environment,
device/growth policy, JIT/TF32, wall time, plan and result paths. CPU runs hide
GPU devices. GPU probes/runs require trusted execution and verified memory growth.

## Skeptical review before execution

Reviewed by the executing Codex agent; this is not an independent review.
The initial idea of testing the coordinate cap alone is rejected: it cannot
answer the fitting-loop or final-restoration question. Scalar KSC evidence is
also insufficient for pairwise safety. A raw no-fit comparator without the same
final cap would change the protection and is rejected. Merely checking finite
numbers is insufficient; the plan checks conditioning, progress and tangents.
The revised plan addresses these flaws and passes for bounded safety evaluation.
No covariance/step control may be promoted on fourth-moment accuracy alone.

Reader contract: readers know weighted particles, covariance whitening and the
chain rule. They should be able to reconstruct why binary residuals suppress
kurtosis, how mixed moments are fitted, why neither cap proves global stability,
and precisely what the guarded finite algorithm guarantees. Existing source
claims and equations are preserved; new conclusions are locally derived and
distinguished from published GenUT theorems. Human readability review remains
pending; compilation and author review do not certify it.

## Execution completion and terminal review

All six planned steps are complete. The optional guard is implemented once in
`bayesfilter/highdim/moment_safety_tf.py` and called through the marginal/pairwise,
batched and canonical filtering routes. The final regression selection passed
55 tests; eight final GPU/XLA checks covered healthy updates, active tail caps
and analytical tangents. A terminal edge-case audit found that the covariance
comparison's roundoff margin alone could admit a singular trial. The additional
strictly positive trial-eigenvalue check was implemented and regression-tested
before completion.

The broad comparison evaluated 112 GPU cases and 14 CPU reference cases before
that final guard refinement. Its source snapshot is preserved and is not
relabeled as final-source validation. All tested loss/validity checks passed.
Healthy final GPU particle values matched exactly; tangent agreement was at
roundoff (maximum 4.34e-19), so the plan's literal all-output equality condition
was not fully achieved. That condition has not been silently relaxed for a
default claim. The original cap also had lower combined moment error in 26/112
cases, an accuracy-promotion veto. Retain the guard as an explicit option, with
existing defaults unchanged; no statistical ranking or HMC claim follows.

The full monograph compiled to 602 pages with resolved references and labels.
The final revised PDF pages 209–216 were visually inspected, and both manuscript
chapter copies are identical. This review is by the executing agent; human
readability review remains pending. Detailed findings, decision/inference tables,
failed-attempt records, source versions and limitations are in
`docs/benchmarks/ledh-moment-safety-results-20261001.md` and its linked evidence.
Conservative compute charges are 324/1800 GPU seconds and 200/1800 CPU seconds;
all three campaign attempts were used. No execution remains pending.

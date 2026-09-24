# Reassess model-score estimation with degenerate transitions

Date: 2026-09-12. Status: conceptual reassessment complete; implementation deferred.

The user asks for a detailed reconsideration of the score problem after the
ordinary density-based all-ancestor smoothing route was excluded for the
degenerate DSGE target. The current question is how to estimate the original
observed-data likelihood score accurately, while preserving deterministic
transition constraints, initial-law derivatives, and honest finite-sample
claims. The previous regular-transition recommendation is not a solution to
this question.

This is a conceptual and source audit, not authorization for an implementation
campaign, new DSGE model, GPU workload, or default change. Work consists of
inspecting the current score/value definitions and relevant primary sources,
deriving support-valid alternatives, and recording the next discriminating
tests. Tiny exact or independent-reference identities may be checked without
running a stochastic performance comparison. No campaign compute budget is
being consumed or enlarged.

Skeptical audit priorities: distinguish model score, finite-filter derivative,
and HMC proposal force; do not reuse historical pre-August-21 LEDH evidence;
do not hide degeneracy with added process noise or a pseudoinverse; do not
equate a pathwise derivative with the derivative of an expectation over
parameter-dependent resampling; do not assume likelihood unbiasedness implies
log-score unbiasedness; and do not claim a variance control removes bias.
Examine initial-law, observation-support, and differentiability assumptions.

Candidate mechanisms to assess: forward weak derivatives of the filter;
innovation-coordinate value and score estimation with explicit ancestry;
valid corrections for the resampling distribution; analytical conditional
integration and control variates; exact latent-innovation inference as a
separate architectural alternative; and the distinct role of a finite-filter
score as a proposal force. Assess numerical approximation and bias separately
from variance and computational feasibility.

Evidence will be preserved under
`docs/plans/artifacts/degenerate-model-score-reassessment-20260912-01/`, with a
reader-facing analysis, local-source ledger, derivations, reference checks,
and a concise decision/uncertainty record. Existing experiments and manuscript
archives remain unchanged. Web search currently returned HTTP 503; use local
primary sources and direct official full text where accessible, and record
coverage limitations rather than treating uninspected papers as support.

Completed analysis: the disturbance-coordinate score identity, resampling
correction distinction, conditional-integration/control-variate roles, and HMC
architecture consequences are recorded in
`docs/plans/artifacts/degenerate-model-score-reassessment-20260912-01/degenerate-model-score-reassessment.md`.
The exact reference checks are in `verification.json`. The next authorized
engineering step is target-specific DSGE disturbance specification followed by
a small exact rank-deficient oracle; no production or stochastic campaign is
authorized by this note.

# Claude handoff: M13 repair proposal audit

Date: 2026-10-04. Requested by the owner. This is a written review handoff;
Claude has not yet reviewed it. No algorithmic source was changed for this
proposal and no new HMC campaign was launched.

Start with this one-path prompt:

> READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless
> the file itself explicitly asks you to inspect a cited line:
> /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917/docs/plans/surrogate-hmc-m13-marginal-repair-proposal-2026-10-04.tex.
> Do not edit, run commands, launch agents, or review the whole repo.
> Question: Is the proposed M13 marginal importance correction mathematically
> valid under its stated assumptions, with an honest proof boundary and
> adequate requirements to preserve the original LGSSM and KSC-SV values
> and scores?
> End with VERDICT: AGREE or VERDICT: REVISE.

## What the proposal is trying to achieve

The campaign goal remains testing surrogate HMC: useful surrogate forces,
a declared endpoint for Metropolis acceptance, and valid posterior sampling.
This proposal addresses an inaccurate M13 particle likelihood/score that can
make the force poor. Improving an upstream score alone does not complete the
campaign.

The preceding diagnostic found that the analytical derivative agrees with
the implemented LEDH likelihood, but the likelihood can be grossly wrong
relative to an independent T=1 quadrature reference. At low Q, the
pre-observation importance normalizer was .00344047 instead of one,
filter ESS was 6.66/4096, and LEDH's first score coordinate was 3.5768
versus the reference .01150. The failure precedes reset.

The proposed mechanism keeps the UKF/LEDH flow, then uses both the full
predictive mixture and the full flow-proposal mixture in the importance
correction. This is an explicitly new MPF-style correction, not an assertion
that it is the unchanged canonical Li–Coates Algorithm 1.

## Questions requiring skeptical mathematical review

1. **Sampling law and density.** Is the conditional affine-map premise enough
   for the stated q_j? Are the full-rank and invertibility conditions adequate?
   Does the one-child-per-ancestor estimator have the correct leading weights?
   Would an iid-label implementation require the different 1/M factor stated?

2. **Theorem scope.** Verify the conditional integration proof and the
   random-label Rao–Blackwell variance proof. Check the exact two-state
   counterexample: the original stratified estimator has variance zero,
   the marginal estimator variance 555264/28826161. Reject any interpretation
   claiming general non-deterioration, fixed-design unbiasedness, an unbiased
   log score, or an unbiased full likelihood product after deterministic resets.

3. **Total derivative.** Check previous-cloud/weight dependence, child
   dependence, mixture responsibilities, inverse-map differentiation,
   determinant derivative, covariance derivative, noncommuting affine
   composition and the recursion through Contract E and GenUT.
   A partial derivative is wrong relative to the stated total-score target.

4. **Optional defensive component.** Verify that samples must come from the
   actual mixture, that alpha is fixed offline, and that the variance bound
   retains (1/alpha-1) Z_N^2. Do not import the stronger control-variate
   guarantee from Owen–Zhou. No alpha value is yet justified or promoted.

5. **Protection of working models.** Does unchanged execution plus actual
   consumer regression adequately preserve existing LGSSM/KSC scopes?
   Check the stationary initial covariance tangent, chart/prior accounting,
   and KSC's recently repaired 2 d beta / beta observation-score term.
   The 49-component reference is exact for the T=2 KSC mixture, not raw SV.
   Preservation is a conditional implementation theorem, not a completed test.

6. **Feasibility and experimental discrimination.** N-squared mixture
   evaluation must be exact and tiled under the existing chunk policy.
   The bootstrap and qualified quadrature must be included as adversaries.
   T=20 reference-score uncertainty is still open. New scope-specific
   tuning and downstream HMC checks are required; no convenient alpha,
   threshold or transferred setting is a promoted default.

## Existing audit and how to request more evidence

MathDevMCP checked elementary symbolic identities and audited selected
document equations. It did not produce a formal certificate for the whole
document. Its document reports retain formalization and dimension-recognition
findings; the disposition and raw responses are preserved in the separate
audit note. A parser failure, expression-size abstention and inconclusive
high-level counterexample wrapper are recorded honestly.

If mathematical tool evidence is necessary after the first read, request this
single next path:
docs/plans/artifacts/surrogate-hmc-m13-marginal-repair-proposal-20261004-01/audit-result.md

For literature support, request this single next path:
docs/plans/artifacts/surrogate-hmc-m13-marginal-repair-proposal-20261004-01/source-support.md

If the affine-map premise requires code inspection, request the exact
flow-substep region of bayesfilter/highdim/ledh_canonical_score_tf.py
(approximately lines 1167–1403 in the recorded dirty checkout), then request
the consumer endpoint separately if needed. The proposal has no implemented
new route whose call chain can already be certified.

Do not review a broad artifact packet or the whole repository. Identify each
material finding by proposition/equation and explain the failing assumption
or missing term. Distinguish mathematical error, unsupported claim,
implementation obligation and proposed empirical test. A valid proposal can
be worth implementing without having proved empirical superiority.

## Current state and expected follow-through

HEAD: a55c6be7a16e9401242ee3449248fa347eb58d90.
Branch: ledh-graceful-failure-execution-20260917.
The current dirty post-fix source is the baseline; unrelated modifications
must be preserved. The proposal stage did not spend GPU/HMC sampling budget.

The proposed implementation belongs to the master's existing M13 repair
phase. This handoff does not claim that phase is already implemented or that
the all-model campaign is complete. The requested output is an advisory
mathematical verdict with concrete corrections, not an execution approval
token or another governance review chain.

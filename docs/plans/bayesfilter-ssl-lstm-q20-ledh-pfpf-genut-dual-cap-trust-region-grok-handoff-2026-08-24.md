# Grok handoff: LEDH-PFPF-GenUT dual-cap/trust-region solution note

Date: 2026-08-24  
Status: `REVIEW_RECEIVED_ADJUDICATION_RECORDED`  
Reviewer: Grok (response received 2026-08-24)

## Purpose

This is the Grok version of the independent review request. It deliberately
uses a separate result path from the Fable review so the two opinions cannot be
confused or overwrite one another.

Grok must write its completed review to exactly:

`docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-grok-review-2026-08-24.md`

The existing Fable handoff is
`docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-fable-handoff-2026-08-24.md`.
Grok must not edit, overwrite, or append to that file, the plan, or the
mathematical note. Only the new `grok-review-2026-08-24.md` result file may be
written.

## Copy-paste request for Grok

```text
You are performing an independent, skeptical mathematical and scientific
review for the BayesFilter q=20 SSL-LSTM NeuTra investigation.

READ the two paths below. Do not modify either source path, do not modify the
Fable handoff, and do not review the whole repository unless a cited source is
needed to resolve one precise claim. Write your final review only to:
docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-grok-review-2026-08-24.md

PASS A: review exactly this plan:
docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-literature-solution-plan-2026-08-24.md

PASS B: review exactly this mathematical note:
docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.md

For each pass, end with exactly one of:
VERDICT: AGREE
VERDICT: REVISE

Audit questions:

1. Does the plan preserve the research question, exact comparator, evidence
   contract, skeptical audit, promotion criteria, vetoes, stop conditions,
   nonclaims, and default/assumption provenance?
2. Does the note correctly prove that selected GenUT moments and affine
   covariance restoration do not identify a density or IID Gaussian law?
3. Are the radial and coordinate dual caps correctly classified as bounded
   finite maps rather than unconstrained full-support normalizing flows?
4. Is the Li-Coates PF-PF change-of-variables identity paired with the actual
   pre-flow proposal, post-flow transition, observation term, covariance
   lifecycle, and matching determinant?
5. Is the Contract-E/OT/GenUT reset correctly separated from a single-particle
   density map and its determinant?
6. Does the replay theorem require frozen proposal densities, a positive
   deterministic-mixture denominator on the target integrand support, and
   recomputation of all retained proposal components?
7. Does the defensive-mixture result state 0 < epsilon_min <= epsilon <= 1,
   target-support coverage, and the additional second-moment integrability
   condition?
8. Does the tempered SMC proposal correctly use invariant mutation kernels,
   while avoiding any finite-run guarantee of mode discovery or whitening?
9. Does the proposed implementation route remain explicitly conditional, with
   NeuTra/HMC admission blocked until density, replay, mode, and downstream
   posterior gates pass?

Use direct derivations or primary-source anchors, not intuition. Separate every
finding into one of:
source_faithful, project_derivation, extension_or_invention, no_go.
Do not treat whitening loss, ESS, cap residuals, acceptance, or a short chain as
proof of density correctness or HMC readiness.

The result file must begin with:
Reviewer: Grok
Date: 2026-08-24

For every finding, report:
location; severity (blocking/major/minor/editorial); classification; exact
claim checked; derivation or source anchor; and the smallest repair. Order
findings by severity. If there are no findings, list the remaining evidence
gaps and the next artifact required. State explicitly whether the result is a
review of the written claims only or also an implementation-alignment review.
```

## Evidence anchors

If a bounded follow-up is needed, the note itself identifies the primary
technical sources and local copies:

- `.localresources/papers/ebeigbe-et-al-genut-2104.01958.txt`;
- `.localresources/papers/ledh_replay_solution_20260824/` for PF-PF, AMIS,
  defensive-mixture, adaptive-SMC, SQMC, and ensemble-transform papers;
- `bayesfilter/highdim/genut_shape_lm_tf.py:121-167`;
- `bayesfilter/highdim/dual_cap_genut_primal_tf.py:193-245`;
- `bayesfilter/highdim/genut_guided_proposal_tf.py:900-1045`;
- `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py:500-527`.

These are follow-up anchors, not permission to broaden the initial review into
a repository-wide scan.

## Author position to challenge

The current note's position is deliberately qualified:

- dual-cap/trust-region GenUT may improve local finite-cloud conditioning and
  selected empirical moments;
- it does not alone establish density faithfulness, IID Gaussian whitening,
  unbiased normalized replay, or finite mode coverage; and
- a conditional repair may combine invertible PF-PF density bookkeeping,
  defensive support, deterministic-mixture or unnormalized replay, and a
  tempered invariant-mutation lane.

Grok should reject this position if a proof or source anchor contradicts it,
and should not upgrade it to an implementation or HMC claim merely because the
proposed architecture is plausible.

## File-separation rule

This handoff is the request. The expected Grok result is the separate
`...-grok-review-2026-08-24.md` file named above. A missing result file means
that Grok has not yet completed the review; it must not be inferred from the
Fable handoff or from the MathDevMCP audit.

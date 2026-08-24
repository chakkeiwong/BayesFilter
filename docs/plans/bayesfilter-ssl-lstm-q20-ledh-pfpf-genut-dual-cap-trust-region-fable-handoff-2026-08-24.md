# Fable handoff: LEDH-PFPF-GenUT dual-cap/trust-region solution note

Date: 2026-08-24  
Status: `REVIEW_RECEIVED`  
Reviewer: Fable (reply received 2026-08-24; result recorded separately)

## Purpose

This memo asks Fable to independently audit the proposed literature/solution
plan and its proposition-proof mathematical note for the q=20 SSL-LSTM
NeuTra problem. The central question is deliberately narrow:

> Does the note correctly separate finite-cloud moment conditioning from
> density correction, replay validity, and mode exploration, and does its
> conditional solution rely only on assumptions that are stated and justified?

The desired outcome may be a rejection. In particular, a finding that the
dual-cap reset alone cannot provide a density-faithful global whitening map is
expected and should be preserved if correct.

## Bounded review prompts

Use these as two separate read-only passes. Each pass has one exact primary
path, so a review can be reproduced without opening the whole repository.

### Pass A: plan

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-literature-solution-plan-2026-08-24.md
Do not edit, run commands, launch agents, or review the whole repo.
Question: Does this plan have a correct research question, comparator,
evidence contract, skeptical audit, stop conditions, and distinction between
promotion criteria, vetoes, explanations, and nonclaims for the proposed
LEDH-PFPF-GenUT dual-cap/trust-region solution?
End with VERDICT: AGREE or VERDICT: REVISE.
```

### Pass B: mathematical note

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.md
Do not edit, run commands, launch agents, or review the whole repo.
Question: Audit the proposition statements and proofs, the literature/source
boundaries, and the conditional implementation route. Check especially that
(1) selected GenUT moments are not presented as density identification, (2)
the radial and coordinate caps are correctly classified as bounded finite
maps rather than unconstrained normalizing flows, (3) the Li--Coates PF-PF
Jacobian is paired with the pre-flow proposal and post-flow transition terms,
(4) deterministic-mixture replay is conditioned on frozen proposal densities,
(5) replay denominators are positive on the target integrand support and
defensive mixtures state `0 < epsilon_min <= epsilon <= 1` plus their
second-moment assumption, and
(6) tempering/mutation is not advertised as a finite mode-discovery guarantee.
End with VERDICT: AGREE or VERDICT: REVISE.
```

## Required review standard

For every issue, report:

| Field | Required content |
|---|---|
| Location | File and line or proposition heading |
| Severity | `blocking`, `major`, `minor`, or `editorial` |
| Classification | `source_faithful`, `project_derivation`, `extension_or_invention`, or `no_go` |
| Claim checked | The exact mathematical or implementation claim |
| Reason | Derivation or source anchor, not intuition |
| Repair | Exact text/equation/assumption change, if needed |

The review must distinguish:

- what Ebeigbe et al. prove about selected GenUT moments;
- what Li and Coates establish for an invertible particle-flow proposal;
- what Cornuet et al. require for deterministic-mixture AMIS recycling;
- what Hesterberg's defensive-mixture argument does and does not bound;
- what is derived directly in the project note; and
- what is a proposed extension for this q=20 route.

Do not treat a successful cap residual, ESS value, whitening score, or short
simulation as a proof of density correctness. Do not turn the note's
conditional theorem into a claim that the current code satisfies its
hypotheses.

## Anchors available if the bounded review requests them

The note cites the technical primary sources and stores local text copies:

- `.localresources/papers/ledh_replay_solution_20260824/`
  (`li-coates-2017-particle-filtering-invertible-flow.txt`,
  `cornuet-et-al-2009-amis.txt`, `hesterberg-1995-defensive-mixture.txt`,
  `fearnhead-taylor-2013-adaptive-smc.txt`,
  `gerber-chopin-2015-sqmc.txt`, and
  `reich-2013-ensemble-transform.txt`);
- `bayesfilter/highdim/genut_shape_lm_tf.py:121-167`;
- `bayesfilter/highdim/dual_cap_genut_primal_tf.py:193-245`;
- `bayesfilter/highdim/genut_guided_proposal_tf.py:900-1045`;
- `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py:500-527`;
- `docs/fable-rewrite/monograph/chapters/ch19c_dpf_implementation_literature.tex:156-183,230-342`.

These are context requests only. The first review prompt remains bounded to
one plan or note path.

## Current author position (to audit, not to accept automatically)

The proposed answer is conditional rather than affirmative:

1. Dual-cap/trust-region GenUT can improve local finite-cloud conditioning and
   selected empirical moments.
2. It cannot alone create an IID density, a full-support normalizing flow, an
   unbiased replay estimator, or a finite mode-discovery guarantee.
3. A possible repair combines an invertible density-evaluable LEDH proposal, a
   defensive full-support component, deterministic-mixture or unnormalized
   replay, and a tempered invariant-mutation lane.
4. The current artifacts do not establish those assumptions, so NeuTra/HMC
   admission remains blocked.

Fable should challenge this position wherever a proof, source section, or local
implementation contradicts it. Conversely, if the position is sound, the
review should say explicitly that it validates the scoped conditional/no-go
claims, not the implementation or a future empirical result.

## Requested response

Return one verdict per pass, followed by findings ordered by severity. If there
are no findings, state the remaining evidence gaps and the exact next artifact
needed. Do not issue a blanket `AGREE` for source-faithfulness without checking
the cited paper and local source anchors.

This memo is a handoff request only. It is not a Fable verdict, an approval to
change the production/default route, or an HMC admission decision.

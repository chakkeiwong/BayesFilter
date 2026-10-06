# Marginal importance weights: independent merge review

Integration clarification, 2026-10-06: the recommendation below concerns the
reviewed weighting capability. Subsequent inventory found that f5e69d716 bundles
173 files and other work; the full branch differs from main in 434 files. This
method review does not approve merging all those changes. Scoped integration,
shared correction refactoring and PP/SIR canaries are planned in
`docs/plans/ledh-marginal-weight-merge-and-canary-2026-10-06.md`.

Recommendation: merge commit `f5e69d716` as the explicitly selected optional
`importance_weight_policy="marginal_mixture"` implementation. Keep the default
`"ancestor"` setting and the documented applicability guards. The evidence
supports this extension and a further research comparison; it does not support
making it an HMC-ready or general default. This review did not merge, push,
message the other agent or alter its checkout.

## Mathematical and implementation findings

The original ancestor correction is valid. Its large range-bearing score
error does not establish an analytical differentiation bug: a correct derivative
of a poorly integrated finite likelihood can be far from the model score.

For one child X_i per ancestor and normalized incoming weights w_i, the proposed
contribution is w_i g(X_i) p_N(X_i)/q_N(X_i), where p_N=sum_j w_j f_j and
q_N=sum_j w_j q_j. The outer w_i is necessary for this allocation. Conditional
on admissible fixed flow maps and incoming cloud, summing expectations cancels
q_N and integrates g p_N exactly. The implementation at
`ledh_canonical_score_tf.py:645` retains the outer weight. The loop at lines
1196–1370 uses the auxiliary path rather than the sampled innovation to form
its coefficients; it accumulates the actual affine map B and its tangent.
`ledh_marginal_weights_tf.py:99` therefore correctly uses proposal mean F_j(m_j)
and covariance B_j Q B_j^T. Its Gaussian tangent includes query-point, component
mean, covariance and incoming-weight dependence. Batch consumer line 292
forwards the policy to this shared implementation.

The equality is to the predictive mixture conditional on the incoming cloud.
It does not prove unbiased deterministic log likelihoods or scores. The
random-label Rao–Blackwell variance inequality does not automatically extend
to a deterministic one-child-per-ancestor design; the supplied PDF explicitly
provides that qualification and a counterexample. Flow equations are preserved,
but changed weights and reset clouds alter later flow inputs.

Source support: supplied proposal PDF, Sections 2–4; Klaas, de Freitas and
Doucet (2005), Section 3/Figure 3 and Section 3.1. The latter was checked in
full-text local source under the other checkout's
`.localresources/papers/m13-repair-20261004/` and at the author-hosted URL:
https://www.cs.ubc.ca/~arnaud/klass_defreitas_doucet_marginalparticlefilterUAI2005.pdf.
This is a bounded formula/call-chain review, not a comprehensive literature
survey. Current citation metadata and forward-citation coverage were not
assessed; neither is needed to establish the directly checked identity.

## Actual evidence and checks

The saved aggregate agrees with the report. At N=4096, each comparison uses
32 paired designs on its specified dataset; log errors are absolute nats.
Score error is 0.1 times the Euclidean score error divided by sqrt(p).

| Range-bearing scope | Log error ancestor → marginal | Scaled score error ancestor → marginal | Bootstrap log error |
|---|---:|---:|---:|
| T=1 nominal | 1.083 → 0.125 | 0.131 → 0.024 | 0.059 |
| T=1 difficult | 2.428 → 0.497 | 0.171 → 0.074 | 0.045 |
| T=20 nominal | 1.271 → 0.143 | 0.327 → 0.035 | 0.122 |
| T=20 difficult | 21.294 → 0.770 | 6.325 → 0.351 | 0.489 |

All paired improvement intervals exclude zero for those four cells and both
metrics, conditional on the stated reference. In difficult T=20, simultaneous
intervals for marginal-minus-ancestor errors are [-26.171,-14.878] nats and
[-7.646,-4.302] scaled score units. All 15 LGSSM/KSC cells meet the declared
0.05 non-inferiority margin for both errors; this does not prove equality or
universal non-deterioration. The reference has remaining finite-particle bias.

I inspected saved full-filter finite-difference results (14 cases, all pass)
and independently reran the five mixture tests (all pass, 4.86 seconds). An
additional CPU FP64 XLA M13 T=2, N=16, two-row consumer check compared two distinct
parameter rows with separate calls through the batch-size-one consumer. Value
and score differences were exactly zero; a repeated compiled call was also
identical. Both rows were valid. Runtime after import was 14.91 seconds.
Source hashes were rechecked afterward. Full logs, script, input settings and
manifest are under
`docs/plans/artifacts/marginal-weight-independent-review-20261006-01/`.
This executable check establishes only that small CPU mechanics case; it does
not certify broad batch behavior, GPU reproducibility or training suitability.

## Remaining gaps and decision

The bootstrap likelihood errors are descriptively lower in all four rows,
which invokes the existing promotion veto. The repeated saved-versus-current
FP32 comparison reports a 0.035 scaled-score discrepancy; its cause is not
isolated. It compares saved and recompiled evaluations, so it does not by
itself establish nondeterminism of one fixed executable. Our tiny CPU FP64
repeat check does not resolve that GPU/FP32 issue. N=4096 precision and stronger
cross-dataset/rank/path support remain untested. The marginal route costs about
1.65 times the ancestor route at N=4096, T=20, descriptively.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Merge optional method | Valid conditional identity; checked shared call chain and focused regressions | No new blocking mechanics defect found | Test coverage limited to saved scopes plus one tiny batch fixture | Merge scoped commit, resolve conflicts, rerun focused regressions on merged tree | Default promotion |
| Continue scientific evaluation | Paired error reduction on the tested range-bearing data | Lower bootstrap errors veto promotion; FP32 discrepancy unresolved | Reference bias, independent datasets, numerical stability | Localize FP32 discrepancy and test fresh tuned scopes | HMC readiness or universally better filter |

| Inference status | Finding |
|---|---|
| Hard veto screen | Bootstrap underperformance triggers the existing promotion rule; no new N4096 invalid designs |
| Statistically supported ranking | Marginal versus ancestor improvement in four recorded M13 cells, conditional on references/data |
| Descriptive-only differences | Bootstrap comparison and runtime; no supported general ranking |
| Default readiness | Not established; existing default retained |
| Next evidence needed | Reproducible GPU score at claim scope, independent datasets, reference-bias checks and equal-cost comparisons |

Post-review red-team: finite-particle reference bias or one-dataset effects
could limit scientific gains, and altered graph compilation could explain the
FP32 discrepancy. The most consequential missing evidence is stable GPU
scoring at the intended HMC scope. None of those gaps invalidates the checked
conditional importance identity or requires rejecting an optional merge.

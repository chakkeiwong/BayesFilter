# KSC reset repair: literature and proposed next experiment

Date: 2026-09-29. Inspected checkout: `sqmc-development`, `7e687d3a`.
Status: literature assessment and repair proposal; no new numerical experiment.

The question is how to preserve the filtering distribution through the shared
Contract E reset and GenUT-style correction, so that likelihood and all score
coordinates agree with the full seven-component KSC reference. Retain all four
ancestry methods; changing ancestry alone cannot address a shared distortion.

## Local evidence and unresolved mechanism

The [discrepancy report](../benchmarks/sqmc-ksc-discrepancy-results-20260929.md)
covers named FP64 GPU/XLA diagnostic variants. Changed scopes were **UNTUNED**;
these are not production-performance or default-readiness results. Agreement
with finite differences validates the finite program's derivative, not the
score of the true KSC likelihood.

Saved traces establish shape loss across the **combined** reset, correction
and final cap. They do not apportion the loss among those operations. The
campaign endpoint calls the canonical analytical executor, unified correction,
and general `higher_moment_contract_e.py` implementation. Prior executable
checks are in the discrepancy artifacts; no new wiring test was run here.

The scalar residual design repeats positive and negative unit points
(`sqmc_campaign_tf.py:35`). Pairwise state-coordinate corrections have no
off-diagonal coordinates in this scalar model. Marginal correction directions
are `u^2 - 1 - m3*u` and `u^3 - m3 - m4*u`. Both vanish for an exactly
symmetric two-point cloud with `m3=0, m4=1`. This is a possible correction
stall, not proof that every observed cloud equals that limiting case.
The final coordinate cap follows moment correction and precedes affine
mean/covariance restoration. Its validity flag does not require small
higher-moment residuals.

## Checked primary sources

| Source and technical material inspected | Result and implication |
|---|---|
| Ebeigbe et al., *Generalized unscented transformation for forecasting non-Gaussian processes*, Physical Review E 111, 054135 (2025), [published author copy](https://math.gmu.edu/~tsauer/pre/PhysRevE.111.054135.pdf), [DOI](https://doi.org/10.1103/PhysRevE.111.054135). Sections III–V, Theorem 1, Corollary 1 and Algorithms 1–2. | Their weighted sigma-point construction matches specified moments under its conditions. Imposing bounds can preserve mean/covariance while losing kurtosis, or both skewness and kurtosis. The theorem does not certify our damped empirical-cloud update followed by a different cap. |
| Corenflos, Thornton, Deligiannidis and Doucet, *Differentiable Particle Filtering via Entropy-Regularized Optimal Transport*, [ICML/PMLR 139 (2021)](https://proceedings.mlr.press/v139/corenflos21a.html). Sections 3.1 and 4, Proposition 4.2's bound and Proposition 4.3's assumptions; supplement retained. | Entropic transport introduces approximation error. A sufficient schedule in the displayed consistency bound is epsilon_N = o(1/log N), with the other assumptions satisfied. Fixed regularization with more particles does not establish removal of that error. Their compactness, Lipschitz and filtering assumptions do not provide a score-consistency theorem for our unbounded KSC/LEDH program. |
| Acevedo, de Wiljes and Reich, *Second-order accurate ensemble transform particle filters*, [arXiv:1608.08179](https://arxiv.org/abs/1608.08179), revised 2017 manuscript. Section 3, Definition 3.1, equations (28)–(33), Section 7.3 example and final discussion. | A matrix correction restores weighted mean/covariance. Their distribution example also produces transformed samples outside posterior support. Second-order exactness is not distributional accuracy. This is a later alternative to assess, not a guarantee for higher moments or scores. |

The 2025 GenUT publication is the current published source located. The 2021
preprint is retained for version comparison; the published Corollary 1 combines
its constrained-moment cases. The linked author code
[GenUT_Ensemble.m](https://github.com/Schiff-Lab/Generalized-Unscented-Transform/blob/main/Unscented%20Transforms/GenUT_Ensemble.m)
constructs weighted `2d+1` points and recomputes weights when constraints move
them. In one dimension this offers a three-point moment reference, not a
replacement for a particle representation of the full filtering density.

Corenflos author code was inspected at commit
`5d8300ba247c4c17e1a301a22560c24fd0670bfe`, files
`filterflow/resampling/differentiable/regularized_transport/plan.py` and
`sinkhorn.py`. It scales inputs and decreases working epsilon toward a fixed
terminal value. More iterations at that value solve the regularized problem;
they do not remove regularization. Its custom gradient clips upstream gradients
and stops some dependencies, so copying it would not satisfy our analytical
total-derivative claim. No official Acevedo code was located.

## Recommended sequence

1. **Trace the same saved cloud through every stage.** Separate weighted
   source, transport/reset, marginal correction, pairwise correction when
   applicable, final cap and recoloring. Preserve moment residuals, within-cluster
   variance, cap activation/derivatives, and exact next-observation predictions
   and scores. This distinguishes reset shape loss, correction stall and cap
   damage without rerunning the entire campaign.

2. **Evaluate calibrated protection.** Test a candidate that controls correction
   displacement and intervenes on a declared conditioning/domain margin, while
   retaining covariance and finiteness safeguards. Compare it with the current
   final cap. Its safety criterion is non-harm: identical healthy outputs, and
   bounded, flagged behavior when intervention is necessary. Do not silently
   remove the dual-cap mechanism or choose its threshold by lowest KSC score
   error. This is a numerics-changing candidate, not a default change.

3. **Independently test a richer fixed residual design.** Many distinct symmetric
   normal-quantile points, centered and whitened, are a first coverage hypothesis.
   They do not assume a Gaussian posterior. Check ordering relative to the
   transported cloud. Compare current design, richer design alone, protection
   candidate alone, and their combination. Preserve Contract E semantics and
   propagate every new dependence analytically; explicitly identify any change
   to the mathematical reset.

4. **Then vary correction effort and terminal regularization.** A bounded
   multi-step moment solve is worth testing once its directions are usable.
   Record residuals after the final operation, separately from numerical
   validity. Test terminal epsilon relative to the actual cost scale, separately
   from Sinkhorn convergence. Freeze schedules before score checks. Another
   paper's numeric epsilon is not a transferable default.

5. **Evaluate the original scientific quantities.** Compare the full likelihood
   and every score coordinate with the seven-mixture reference, cross-checked
   by the independent density-grid calculation. Keep the Gaussian approximation
   as a cheap heuristic. Use paired designs and uncertainty intervals by regime
   and horizon. Scope-specific calibration/validation precedes untouched claim
   data. Retain all four ancestry methods.

The first discriminating experiment is the stagewise audit followed by isolated
residual-design and protection tests. Additional particles or repetitions of a
stalled correction are not the first repair.

## A local derivation about moment-compatible bounds

If a final standardized scalar Z satisfies E[Z^2]=1 and |Z|<=c, then
Z^4<=c^2 Z^2 pointwise, hence its kurtosis E[Z^4]<=c^2. A requested kurtosis k
therefore requires c>=sqrt(k). This is necessary, not sufficient, and is not by
itself a calibration prescription.

Our current cap is **before** variance restoration; its threshold is not a
bound on the final standardized cloud. Affine re-inflation restores variance
but preserves the capped cloud's standardized skewness/kurtosis. This explains
why restored variance cannot demonstrate recovery of higher moments; it does
not prove that the cap alone caused the measured distortion.

## Assumptions and evidence roles

| Choice / status | Possible failure | Earliest check |
|---|---|---|
| Existing frozen design/cap, not validated KSC defaults | Cluster concentration or shape loss | Same-cloud stage trace |
| Distinct quantiles, proposed coverage hypothesis | Ordering dependence or poor non-Gaussian tails | Distribution and exact predictive functional |
| Conditional protection, safety candidate | Missed instability, branch discontinuity or changed healthy output | Domain margins, non-harm and branch-matched derivative checks |
| Extra moment iterations, solver hypothesis | Flat directions, oscillation, matching an inadequate distribution | Direction norm, final residual, then predictive error |
| Epsilon ladder, literature-motivated hypothesis | Ill conditioning or excessive solve cost | Transport residuals, validity, derivatives and runtime |

Likelihood and all score errors against the full reference are accuracy criteria.
Moment diagnostics explain mechanisms and cannot promote a method. Nonfinite
results, broken covariance identities, wrong analytical derivatives, invalid
references or exhausted budget stop the affected experiment for repair. A finite
but inaccurate candidate fails promotion and can motivate the next repair.
Losing to the Gaussian heuristic in a salient regime vetoes promotion, not
continued investigation. Neither moment matching nor a smoke pass proves score
accuracy or superiority.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Prioritize shared-reset repair; retain four candidates | Repaired likelihood/score accuracy untested | Prior numerical checks passed; reported accuracy remains inadequate | Contributions of reset, correction and final cap | Stagewise audit and isolated candidates | No successful fix, ranking or default promotion |

This proposal is not launch-ready: safety calibration, thresholds, fresh data
partitions, commands and per-stage allocations still need specification.
No GPU time was used here. The aggregate ledger remains 31,543.211557 of
43,200 GPU-seconds, leaving 11,656.788443 seconds; the elapsed deadline remains
2026-09-30T16:15:40.010888+00:00. Neither budget is enlarged.

## Source audit and limitations

Search date: 2026-09-29. Queries covered constrained GenUT, second-order
ensemble transforms, entropic differentiable particle filtering, author code
and later corrections/follow-ups. The forward search found the 2025 GenUT
publication; foundational transport citations were screened for relevance.
This is a focused repair review, not an exhaustive survey. No erratum or
retraction was located in the bounded searches; no comprehensive integrity
check is claimed. Citation counts and venue rankings were not collected.
The Reading repository blocked access, so the revised Acevedo arXiv manuscript
was inspected. Full distribution transports, mixture-adapted proposals and
smoothing/Fisher-identity score estimators are later possibilities, not methods
technically audited or compared here.

Local PDFs/text are under `.localresources/papers/`:
`ebeigbe-et-al-genut-2025-PhysRevE111054135`,
`ebeigbe-et-al-genut-2104.01958`,
`corenflos21a-differentiable-particle-filtering`,
`corenflos21a-supplement`,
`acevedo-dewiljes-reich-second-order-1608.08179`.
Inspected code copies: `genut-author-GenUT_Ensemble.m`,
`corenflos-author-plan.py`, `corenflos-author-sinkhorn.py`.

Self-review: the paper guarantees were not transferred to the local correction;
combined shape loss was not attributed solely to the cap; safeguards were
retained for a dedicated non-harm evaluation; and scientific criteria remain
likelihood and score. The strongest alternative explanation is that other
shared flow/transport errors dominate after shape repair. Staged predictive
and full-filter checks must resolve that. No independent review was performed.

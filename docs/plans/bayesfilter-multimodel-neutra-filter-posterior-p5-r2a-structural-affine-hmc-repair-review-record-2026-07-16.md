# P5 R2A Structural Affine Repair Review Record

Date: 2026-07-16

Reviewed path:
`docs/plans/bayesfilter-multimodel-neutra-filter-posterior-p5-r2a-structural-affine-hmc-repair-subplan-2026-07-16.md`

## Skeptical Review

| Audit question | Finding | Resolution |
| --- | --- | --- |
| Is the baseline described correctly? | Initial framing incorrectly implied a warm-up R-hat cap. The artifact shows one energy-error divergence in the first 1,000-draw chunk. | Reclassified as source-kernel health failure and prohibited an R-hat-cap claim. |
| Is a proxy being promoted? | The short probe and source covariance are tuning diagnostics only. | Admission remains fresh sequential health, modern R-hat, and ESS. |
| Is failed warm-up entering inference? | It could have entered through a covariance estimate. | Restricted it to a pooled mode-start hypothesis; terminal curvature is independently reevaluated from exact source scores. |
| Can regularization hide a saddle? | Eigenvalue clipping alone could produce a plausible chart at a non-mode. | Require terminal score convergence and raw negative-Hessian positive definiteness before regularization. |
| Are defaults transferred silently? | P4 finite-difference and Newton settings are inherited. | Labeled them target-specific hypotheses and added a two-step Hessian stability gate. |
| Are stop conditions and budget bounded? | Yes after making geometry failure a stop before HMC. | One affine repair, unchanged six-hour R2 budget, and existing infrastructure retry ceiling. |
| Do artifacts answer the question? | Yes if geometry and HMC have separate roots and recursive ledgers. | Required geometry admission before HMC and separate manifests/archives. |

Verdict: `AGREE_AFTER_VISIBLE_REPAIR`.

Claude was already attempted for the material P5 structural plan using the
required one-path read-only prompt. The platform denied disclosure of private
workspace content, so no Claude verdict exists. Under the current advisory and
proportional review policy, repeating the same blocked disclosure is not an
execution gate. This local review is not a substitute for the required
numerical checks.


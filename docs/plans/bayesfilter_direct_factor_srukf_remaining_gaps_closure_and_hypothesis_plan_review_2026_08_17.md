# Review: Remaining-Gaps Closure and Hypothesis Plan

Date: 2026-08-17  
Reviewed plan: `bayesfilter_direct_factor_srukf_remaining_gaps_closure_and_hypothesis_plan_2026_08_17.md`

## Verdict

`APPROVED_FOR_BOUNDED_EXECUTION_WITH_BLOCKERS_PRESERVED`

The plan is executable as a local diagnostic campaign and does not silently
promote any of the six blocked registry cells. It correctly separates direct
factor SR-UKF evidence from SGQF, KSC-surrogate, and Zhao–Cui target families.

## Review checks

| Check | Result | Reason |
|---|---|---|
| Mathematical derivation | pass | Prediction and conditional measurement factors are specified as direct residual-stack/block-QR operations; singular support and fixed-chart limits are explicit. |
| Numerical stability | pass | QR is the admitted fixed-chart operation; rectangular factors, pivots, support residuals, rank changes, and branch events have telemetry/gates. |
| Analytical differentiation | pass | Scores are admitted only on fixed rank/pivot/sign/angle charts; rank changes, repeated singular values, and zero circular resultant are value-only/invalid-score cases. |
| Target identity | pass | Each hypothesis has its own target and source-semantics gate; no KSC/SGQF/Zhao–Cui substitution is allowed. |
| Testing coverage | pass | Unit tests cover finiteness, finite differences, batch shape, permutation, eager/XLA parity, mass preservation, source separation, and negative promotion controls. |
| GPU/XLA policy | pass | GPU is required for serious admission scripts; CPU is limited to small reference diagnostics and artifacts must record the exception. |
| Compute bound | pass | One attempt plus one localized retry per hypothesis; no training or HMC launch. |
| Reproducibility | pass | Fresh versioned artifact root, hashes, seeds, environment, command, dtype, device, JIT, and memory policy are required. |
| Scope creep | pass | Non-applicable/owner-excluded rows remain unchanged; no broad registry or default change is authorized by this plan. |

## Required execution corrections

1. The diagnostic runner must report `not_run` when a required GPU/XLA gate is
   unavailable; it must not downgrade the route or claim CPU production
   evidence.
2. Any existing artifact that says `ADMIT_KSC_GAUSSIAN_SUM_UKF` is evidence for
   the KSC Gaussian-sum surrogate only. The new result must label it as a
   hypothesis result and must not rewrite the direct-factor inventory row.
3. The SVX-ZC capability flag must be checked from the repository adapter
   factory. Caller-supplied metadata or a finite manual score cannot promote
   `xla_hmc_ready`.
4. Zhao–Cui tests must cite paper/source anchors or explicitly emit
   `source_anchor_missing`; internal consistency alone is insufficient.

## Stop conditions

Stop a hypothesis on target/hash drift, nonfinite output, a failed score/value
gate, an unverified rank/pivot/branch event, or a missing source anchor. Record
the failure and preserve the blocked row. These are scientific outcomes, not
reasons to relax thresholds or substitute a different target.


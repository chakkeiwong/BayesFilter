# Phase 40 Repair and Refresh Note

| Attempt | Failure class | Repair | Result |
|---|---|---|---|
| root-group preflight | none | deterministic subset-sum whole-root allocation; sign-balanced and complete | pass |
| identity initial launch | harness: integer root IDs sent to floating finiteness assertion | check finiteness only for floating/complex tensors; preserve failed root | repaired; `identity-attempt2` passes |
| identity v2.2 trace | none | fresh GPU/XLA trace with root-group split and checkpoints | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` |
| affine v2.2 trace | none | same split and target with exact train-measure affine oracle | `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED` |
| v2.1 reporter on v2.2 roots | expected schema veto | add version-bound v2.2 reporter; do not weaken historical reporter | repaired; v2.2 report passes |
| v2.2 checkpoint report | none | validation-only selection; audit after selection | `PASS_V2_2_CHECKPOINT_SELECTION_AUDIT_RECEIPT` |
| v2.2 measure report | none | exact source split, root-overlap and signature checks | `PASS_V2_2_THETA_MEASURE_SEPARATION_DIAGNOSTIC` |

The root-group repair removed ancestry overlap but did not make the finite
holdouts representative: validation theta mean[0] was `1.181110` and audit
theta mean[0] was `-1.422781`, versus `0.330821` for training. Affine
validation/audit residuals remained large. This is a support/evidence-boundary
repair trigger, not a continuation veto.

The next subplan must use a fresh independent theta bank as an audit source,
preserve the exact target/proposal signatures, and avoid selecting or tuning
on that bank. No v2.2 result is silently promoted to whitening or HMC status.

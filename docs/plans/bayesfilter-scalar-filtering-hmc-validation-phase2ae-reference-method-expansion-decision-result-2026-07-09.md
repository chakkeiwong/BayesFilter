# Phase 2AE Result: Reference-Method Expansion Decision

Date: 2026-07-09
Status: `CURRENT_SEQUENTIAL_REFERENCE_BRANCH_BLOCKED_EXPANSION_REQUIRED`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-subplan-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Stop the current local fallback-resampling sequential-reference branch and require a materially different reference-method design before further runtime | Passed for decision only: Phase 2AB/2AC/2AD isolate a beta-completion versus ancestor-diversity conflict | No scientific/default/HMC/GPU promotion is supported; Phase 3 remains blocked | A better reference may require stronger rejuvenation, deterministic transport, a different bridge, or another reference family rather than a local fallback-resampling tweak | Draft a new reviewed reference-method master/subplan if continuing; otherwise close the scalar validation program with reference agreement unresolved | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness |

## Evidence Summary

| Phase | Result | Interpretation |
| --- | --- | --- |
| 2AB | Beta stalled at `0.3419540270406287`; no reference nomination. | Baseline sequential pilot could not complete the bridge. |
| 2AC | Beta reached `1.0`, terminal ESS ratio `0.9912539055044092`, max weight `0.010002188339361427`, but unique ancestor fraction fell to `0.21875 < 0.25`. | Forced fallback resampling repaired beta progression but spent too much root-ancestor diversity. |
| 2AD | Unique ancestor fraction stayed `0.4140625`, but beta stalled at `0.9712250668187553`. | Diversity-preserving fallback skipping avoided collapse but lost beta completion. |

## Decision

The current local fallback-resampling sequential-reference branch is blocked for
this scalar validation lane.  The blocker is specific to the attempted
reference construction: the reviewed local repairs trade off beta completion
against ancestor diversity.  This does not invalidate the scalar target, HMC
mechanics, MAP-local geometry, or the broader idea of finding a reference.  It
does block Phase 3 GPU/XLA reproduction and any HMC-reference agreement claim
until a different reviewed reference method passes.

Reasonable next branches, each requiring its own reviewed subplan before
runtime, include:

- stronger rejuvenation or move design inside the sequential bridge;
- a deterministic or learned transport reference route;
- a different reference-family construction informed by the Phase 2Y geometry
  localization;
- a blocker closeout that records reference agreement unresolved and stops the
  scalar HMC-validation program.

## Checks

| Check | Status |
| --- | --- |
| Phase 2AE focused local review | `VERDICT: AGREE_FOR_DECISION_RESULT_ONLY`. |
| Runtime | Not authorized and not run. |
| Phase 3 GPU/XLA | Blocked. |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Current sequential branch blocked for nomination. |
| Reference validity | Not established. |
| HMC-reference agreement | Not assessed. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Phase 2AC versus Phase 2AD exposes the beta/diversity tradeoff. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | A reviewed materially different reference-method design, or program closeout with reference agreement unresolved. |

## Post-Decision Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | A small change in rejuvenation strength, particle count, or schedule might pass, but that would be another policy search unless predeclared as a materially different reviewed branch. |
| What would overturn | A reviewed reference-method design that passes beta, terminal ESS/max-weight, diversity, finiteness, and independent replication gates. |
| Weakest evidence | CPU-hidden one-seed diagnostics; enough to block the current local repair branch, not enough to reject the target or HMC. |

## Final Nonclaims

- No valid independent reference.
- No valid sequential reference.
- No HMC-vs-reference agreement.
- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.

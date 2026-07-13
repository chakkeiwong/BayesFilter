# Phase 2AA Result: Reference Branch Decision

Date: 2026-07-09
Status: `PASSED_DECISION_MOVE_TO_SEQUENTIAL_REFERENCE_PLAN`

Master program:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Subplan:
`docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2aa-reference-branch-decision-subplan-2026-07-09.md`

Review:
`docs/reviews/bayesfilter-scalar-filtering-hmc-validation-phase2aa-claude-review-2026-07-09.md`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not being concluded |
| --- | --- | --- | --- | --- | --- |
| Abandon blind independent SNIS proposal tweaks for this target for now and move to a reviewed sequential/transport reference pilot | Passed: the branch choice is explicit and justified by Phase 2W/2X/2Y/2Z artifacts without upgrading proposal failure into HMC or target failure | No Phase 2AA veto fired; Phase 3 GPU/XLA remains blocked; no unsupported claims were added | Phase 2Y supports proposal-family/global-geometry mismatch only descriptively, and Phase 2Z was a finite pilot, not proof that SNIS or independent references are impossible | Draft and review Phase 2AB transport-or-sequential-reference subplan before any runtime | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness |

## Evidence Contract Status

| Field | Status |
| --- | --- |
| Question | Answered as a decision gate: do not spend another phase on blind independent SNIS tweaks after Phase 2W, 2X, and 2Z failed reference/proposal screens and Phase 2Y localized the observed failure away from affine/proposal replay bugs. |
| Baseline/comparator | Phase 2W fixed standard-normal SNIS, Phase 2X shifted-mixture SNIS, Phase 2Y target-geometry localization, and Phase 2Z four-candidate Student-t proposal pilot. |
| Primary criterion | Passed: the result selects the sequential/transport reference branch and preserves claim boundaries. |
| Veto diagnostics | None fired.  The result does not treat Phase 2Z candidate failure as proof that HMC, the target, SNIS in general, or independent references in general are wrong. |
| Explanatory diagnostics | Prior ESS, ESS ratio, max weight, target/proposal replay, top-anchor norms, and ray residuals are used only to motivate the branch decision. |
| Not concluded | No posterior correctness, HMC readiness/convergence, zero divergences, sampler superiority, statistical ranking, GPU/XLA readiness, default readiness, or source faithfulness. |

## Prior Evidence Summary

| Phase | Diagnostic | Decision role |
| --- | --- | --- |
| 2W | ESS `22.894679726459746`, ESS ratio `0.022358085670370845`, max normalized weight `0.17637097762827186`; finite target and log weights | Rejects the fixed standard-normal independent SNIS proposal as a valid reference for this target. |
| 2X | ESS `33.4215730897076`, ESS ratio `0.01631912748520879`, max normalized weight `0.11446321229118656`; finite target/proposal/log weights | Rejects the shifted-mixture independent SNIS repair as a valid reference for this target. |
| 2Y | Affine orientation replay error `4.440892098500626e-16`; proposal log-density replay delta `0.0`; top-weight anchor norms `3.1676400712527686` to `8.91162729981821`; max ray residual `104.96420467633178` | Makes proposal-family/global-geometry mismatch plausible and does not support an affine/proposal replay bug explanation. |
| 2Z | Best pilot ESS was `26.071556547207543`; no candidate met ESS `>= 256`, ESS ratio `>= 0.05`, and max weight `<= 0.05` | Rejects the tested heavier-tail/anchor/ridge independent proposal pilots as candidates for independent replication. |

## Branch Decision

Phase 2AA chooses a sequential/transport reference route, with Phase 2AB as the
next reviewed subplan.  The decision is intentionally narrow:

- abandon blind independent SNIS tweaks for this scalar target for now;
- do not claim independent references or SNIS are impossible;
- do not claim Phase 2Y proved the full posterior geometry;
- do not proceed to Phase 3 GPU/XLA;
- do not interpret HMC-vs-reference agreement until a reviewed reference
  validity gate passes.

## Review Record

| Review item | Status |
| --- | --- |
| Phase 2AA skeptical plan audit | Passed for decision-only execution. |
| Claude review | `VERDICT: AGREE` on compact bundle. |
| Claude constraints | Read-only; no edits, runtime authorization, or claim authorization. |
| Guardrail carried forward | Phase 2AB must state comparator, promotion status, continuation veto, artifact contract, and nonclaims before runtime. |

## Checks

| Check | Status |
| --- | --- |
| Runtime | Not required for Phase 2AA decision. |
| `git diff --check` | To be run after Phase 2AB subplan review/update. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `52ee244498988e046a6356f926003b581103083b` |
| Git dirty status | Dirty; artifact records planned scalar HMC validation files and unrelated user work. |
| Command | No runtime command; decision phase only. |
| Environment | N/A; no runtime. |
| CPU/GPU status | N/A; no runtime; Phase 3 GPU/XLA remains blocked. |
| JIT/TF32 | N/A; no runtime. |
| Seeds | N/A. |
| Wall time | N/A. |
| Plan/result paths | Master, Phase 2AA subplan, Claude review, this result, and Phase 2AB draft subplan. |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | Passed for decision artifact validity. |
| Reference validity | Still unresolved. |
| HMC-reference agreement | Not assessed. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Prior ESS, ESS ratios, max weights, anchor norms, residuals, and runtimes. |
| Posterior correctness | Not assessed. |
| HMC readiness | Not assessed. |
| GPU/XLA readiness | Blocked. |
| Default readiness | Not assessed. |
| Zero-divergence claim | Not made. |
| Next evidence needed | Reviewed Phase 2AB sequential/transport reference pilot before runtime. |

## Post-Decision Red-Team Note

| Field | Note |
| --- | --- |
| Strongest alternative explanation | A better independent proposal might still work; Phase 2AA rejects only another blind tweak without a new discriminating hypothesis. |
| What would overturn | A reviewed independent proposal design with new artifact-supported geometry insight and fresh validity gates, or a sequential/transport reference that produces a usable replicated reference. |
| Weakest evidence | Phase 2Z used a timeout-repaired sample size and no uncertainty analysis; its role is branch triage, not proof of impossibility. |

## Final Nonclaims

- No valid independent reference.
- No valid sequential/transport reference yet.
- No HMC-vs-reference agreement.
- No posterior correctness.
- No HMC readiness.
- No HMC convergence.
- No zero-divergence claim.
- No sampler superiority or statistical ranking.
- No GPU/XLA production or default readiness.
- No Zhao-Cui source faithfulness.

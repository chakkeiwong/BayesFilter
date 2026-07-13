# Phase 2AA Subplan: Reference Branch Decision

Date: 2026-07-09
Status: `PASSED_WITH_CLAUDE_REVIEW`

## Phase Objective

Decide the next reference-validation route after Phase 2W, 2X, and 2Z all
failed to produce a usable self-normalized importance proposal for the scalar
SSL-LSTM MAP-local `u_new` target.  The phase must choose one of:

- abandon the current SNIS reference-agreement branch for this target;
- move to a transport or sequential reference method with a new reviewed plan;
- run one narrowly justified additional diagnostic only if it answers a
  specific unresolved implementation question.

This is a decision/planning phase.  It is not an HMC run, not a GPU/XLA phase,
not a valid reference claim, not posterior certification, not HMC readiness,
not convergence evidence, not default-readiness evidence, and not Zhao-Cui
source-faithfulness evidence.

## Entry Conditions

- Phase 2W failed ESS and ESS-ratio gates for standard-normal SNIS with finite
  target/proposal/log-weight evaluations.
- Phase 2X failed ESS and ESS-ratio gates for shifted-mixture SNIS with finite
  target/proposal/log-weight evaluations.
- Phase 2Y localized the failure to proposal-family/global-geometry mismatch,
  not affine or proposal-log-density replay bugs.
- Phase 2Z evaluated four heavier-tail or geometry-aware Student-t proposal
  pilots with finite evaluations and no hard vetoes, but nominated no candidate.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2AA result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2aa-reference-branch-decision-result-2026-07-09.md`
- If continuing with transport/sequential reference, draft the next subplan:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-subplan-2026-07-09.md`
- If abandoning the branch, draft a closeout/expansion decision subplan update.
- Optional review bundle if Claude becomes safely available; otherwise local
  Codex substitute review record under `docs/reviews/`.

## Required Checks, Tests, And Reviews

- Review this subplan before writing the result.  Claude may be attempted only
  if the approval layer permits bounded external review of the compact bundle.
  If not, record local Codex substitute review as weaker than Claude.
- No runtime command is required for the decision itself.
- Run `git diff --check` before closing the phase.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Given Phase 2W/2X/2Y/2Z, should we keep trying independent SNIS proposals, switch reference method, or stop the reference branch? |
| Baseline/comparator | Failed Phase 2W, Phase 2X, and Phase 2Z proposal artifacts plus Phase 2Y geometry localization. |
| Primary criterion | The result must select an explicit next branch and justify it using prior artifacts without adding unsupported scientific claims. |
| Veto diagnostics | Treating Phase 2Z candidate failure as proof HMC is wrong, claiming posterior correctness/readiness, proceeding to Phase 3 GPU/XLA, or proposing another blind SNIS tweak without a new discriminating hypothesis. |
| Explanatory diagnostics | ESS, ESS ratio, max weights, Phase 2Y residuals, proposal replay checks, and runtime costs from prior phases. |
| Not concluded | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2AA result, review record, and refreshed runbook/handoff. |

## Forbidden Claims And Actions

- Do not run HMC or GPU/XLA in Phase 2AA.
- Do not change defaults or public API behavior.
- Do not claim the target or HMC chain is invalid from failed SNIS proposals.
- Do not claim a valid reference, posterior correctness, HMC readiness,
  convergence, zero divergences, sampler superiority, statistical ranking,
  default readiness, or Zhao-Cui source faithfulness.
- Do not propose another independent SNIS tweak unless it has a new,
  artifact-supported, discriminating hypothesis not already tested by Phase
  2W/2X/2Z.

## Exact Next-Phase Handoff Conditions

If the decision is to abandon independent SNIS for this target, draft the
program closeout or expansion-decision subplan with Phase 3 GPU/XLA still
blocked.

If the decision is to move to transport or sequential reference, draft a
Phase 2AB subplan that states the target reference method, artifact contract,
validity gates, runtime budget, and nonclaims.  That subplan must be reviewed
before runtime.

If the decision is a final narrow diagnostic, draft that diagnostic subplan
with a single discriminating question and stop before runtime.

## Stop Conditions

Stop for review nonconvergence, missing prior artifacts, unsupported claims,
need to cross HMC-runtime/GPU/default/model-file/source-faithfulness/product
boundaries, or inability to choose a branch from the existing evidence.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the sequence of failed SNIS artifacts plus Phase 2Y diagnostics, not HMC success. |
| Proxy metrics promoted | Failed SNIS screens can reject proposal candidates, not the target or HMC route. |
| Missing stop conditions | Boundary, claim, review, and branch-selection stops are explicit. |
| Unfair comparison | No candidate ranking or sampler comparison is made. |
| Hidden assumptions | The plan requires a new reviewed method if leaving SNIS; it does not smuggle transport/sequential validity. |
| Stale context | Phase 2AA must cite Phase 2W/2X/2Y/2Z result artifacts. |
| Environment mismatch | No runtime evidence is generated, and Phase 3 remains blocked. |
| Artifact mismatch | Result and handoff artifacts are predeclared. |

Audit status: `PASSED_WITH_CLAUDE_REVIEW_FOR_DECISION_ONLY`.

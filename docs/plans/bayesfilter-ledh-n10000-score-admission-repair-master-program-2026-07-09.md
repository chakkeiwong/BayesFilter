# LEDH N10000 Score Admission Repair Master Program

Date: 2026-07-09

Status: `DRAFT_LAUNCH_GATE_RECREATED_AFTER_REVIEW_VISIBILITY_REPAIR`

## Objective

Repair the mechanical blocker left by the July 8 compact-score program: score
computations and raw memory JSONs exist, but leaderboard score rows are not
admitted unless the evidence is serialized as a Phase 1 validated compact
`N=10000` score artifact bound to the same admitted value artifact.

Here `score` means the total derivative, with respect to the row's stated free
parameter vector, of the same realized finite-`N` LEDH
`observed_data_log_likelihood_estimator` reported as `log_likelihood` by the
admitted value route.

## Governing Admission Contract

Full score admission requires:

- `schema_version = bayesfilter.highdim.ledh_score_artifact.v1`;
- `score_admission_status = n10000_same_target_no_tape_score_admitted`;
- compact forward-sensitivity no-tape provenance, never `manual_total_vjp*`;
- matching `row_id`, target scalar, output field, observation policy, theta
  coordinate system, and parameter order with the admitted value artifact;
- correctness status `pass`;
- `memory_diagnostics.n10000_memory_pass = true`;
- `validate_ledh_score_artifact(..., require_admitted=True)` passes against
  the admitted source value artifact.

Raw `primary_pass` memory JSONs, tiny score diagnostics, value-only artifacts,
and historical manual-total-VJP outputs are not full score evidence.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the existing compact score routes be turned into Phase 1 validated `N=10000` score artifacts and admitted into the LEDH leaderboard without changing the target scalar? |
| Baseline/comparator | July 8 Phase 8 integration blocker, shared score validator, admitted value artifacts, LGSSM compact raw memory evidence, and tiny compact per-model diagnostics. |
| Primary pass criterion | Every main LEDH model row has either a validator-admitted compact `N=10000` score artifact or a precise blocker result explaining the smallest fixable reason it cannot yet be admitted. |
| Veto diagnostics | Wrong target scalar; value/score artifact mismatch; historical route admitted; raw legacy JSON treated as admitted; tape/autodiff or stopped partial derivative; missing memory pass; nonfinite score; correctness mismatch; KSC exact-native actual-SV overclaim; generalized-SV target substitution; diagnostic row promotion. |
| Explanatory diagnostics | Runtime, peak memory, FD/exact errors, chunk sizes, device status, model-specific artifact coverage, and candidate leaderboard row status. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, runtime ranking, public benchmark readiness, or all-algorithm leaderboard completion beyond LEDH score admission. |

## Phase Index

| Phase | Name | Purpose | Required result artifact |
| --- | --- | --- | --- |
| 0 | Launch inventory and artifact map | Freeze the inherited blocker, map value/score artifacts, and verify commands can answer the admission question. | `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase0-launch-inventory-result-2026-07-09.md` |
| 1 | Shared admitted artifact emitter | Add or certify one shared way to emit schema-valid score artifacts and validator tests before any full run. | `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase1-shared-emitter-result-2026-07-09.md` |
| 2 | LGSSM N10000 admission | Normalize or rerun LGSSM compact `N=10000` score evidence into an admitted score artifact. | `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-result-2026-07-09.md` |
| 3 | Fixed-SIR N10000 compact rerun | Replace historical fixed-SIR `manual_total_vjp` evidence with compact forward-sensitivity full evidence. | `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase3-fixed-sir-result-2026-07-09.md` |
| 4 | Actual-SV N10000 compact gate | Run exact transformed actual-SV compact score at full row scale after smoke validation. | `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase4-actual-sv-result-2026-07-09.md` |
| 5 | Predator-prey N10000 compact gate | Run predator-prey compact score at full row scale after smoke validation. | `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase5-predator-prey-result-2026-07-09.md` |
| 6 | Generalized-SV N10000 compact gate | Run generalized-SV compact score without source-row substitution. | `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase6-generalized-sv-result-2026-07-09.md` |
| 7 | KSC-SV N10000 compact gate | Run KSC finite-mixture compact score without exact-native actual-SV overclaim. | `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase7-ksc-sv-result-2026-07-09.md` |
| 8 | Inclusive score integration | Wire admitted score artifact paths into the inclusive LEDH leaderboard and regenerate candidate results. | `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase8-integration-result-2026-07-09.md` |

## Subplan Requirements

Each phase subplan must state objective, inherited entry conditions, required
artifacts, required checks/tests/reviews, evidence contract, forbidden
claims/actions, exact next-phase handoff conditions, and stop conditions.

At the end of each phase: run required checks, write a close record, draft or
refresh the next subplan, and review that next subplan for consistency,
correctness, feasibility, artifact coverage, and boundary safety.

## Review And Stop Policy

Claude may be used as read-only reviewer only. If local policy blocks Claude,
use a fresh Codex packet-only review and record the limitation. Stop for target
definition changes, pass/fail criterion changes after results, package
installation, network/data fetches, credentials, destructive git actions,
unrelated dirty-worktree edits, or nonconvergent review after five rounds.

## Nonclaims

This master program does not admit any new score and does not establish HMC
readiness, posterior correctness, scientific superiority, or public benchmark
readiness.

# LEDH Compact Score Default Master Program

Date: 2026-07-08

Status: `DRAFT_LAUNCH_GATE`

## Objective

Demote the memory-inefficient reverse-record/manual-total-VJP LEDH score
routes to historical diagnostic status and make compact forward sensitivity the
default score style for every LEDH leaderboard model.

Here `score` means the total derivative, with respect to the row's stated free
parameter vector, of the same realized finite-`N` LEDH
`observed_data_log_likelihood_estimator` reported as `log_likelihood` by the
admitted value route.

The default admissible score style is:

```text
compact_forward_sensitivity_no_autodiff_same_scalar_ledh_pfpf_ot
```

It carries value state and parameter tangents through a single forward loop:

- particles;
- log weights;
- particle tangents;
- log-weight tangents;
- log likelihood;
- log-likelihood tangents.

It must use streaming transport value+JVP where transport is active and must
not keep per-time reverse records.

## Governance Decision

The old memory-inefficient score style is now classified as:

```text
historical_wrong_for_leaderboard_score_admission
```

This includes any admitted-score candidate that depends on:

- retaining all time-step records for a reverse scan;
- `records.append(...)` plus `reversed(records)`;
- reverse transport pullbacks as the main full-row score path;
- route labels beginning with `manual_total_vjp_no_autodiff_same_scalar_*`
  for future leaderboard admission;
- `GradientTape`, `ForwardAccumulator`, hidden autodiff, stopped partial
  derivatives, or a scalar other than `log_likelihood`.

Historical reverse/manual-total-VJP code may remain only as a diagnostic
comparator during migration. It must not be the default, must not be used for
full `N=10000` score admission, and must not be collected as a leaderboard
score.

## Inherited State

The existing score runbook found:

- LGSSM already has the desired compact forward-sensitivity style in
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`.
- Actual-SV currently uses a reverse-record route in
  `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`; it passes
  tiny same-scalar diagnostics but is not memory-style correct.
- Fixed-SIR and predator-prey currently expose manual-total-VJP score labels
  and must be migrated or blocked from leaderboard score admission.
- Generalized-SV and KSC-SV have admitted value scalars but do not yet have
  compact score routes.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can all LEDH leaderboard score computations be wired to the compact forward-sensitivity no-autodiff style while preserving the exact same finite-`N` `log_likelihood` target as value? |
| Baseline/comparator | LGSSM compact implementation, admitted value artifacts, existing reverse/manual-total diagnostics as historical comparators only, and same-scalar finite differences or exact references. |
| Primary pass criterion | Every model row either has an admitted compact score artifact with same-target value binding, no autodiff, predeclared same-scalar FD or exact-reference agreement at the tested point/coordinates, and `N=10000` memory evidence, or is explicitly blocked from score admission. No reverse-record/manual-total-VJP route can be admitted. |
| Veto diagnostics | Wrong target scalar, value/score route mismatch, reverse-record route used as default, `manual_total_vjp*` route admitted, tape/autodiff, stopped partial derivative, all-time record retention for score, nonfinite score, FD/exact mismatch, memory failure, diagnostic row promotion, or unsupported scientific/HMC claim. |
| Explanatory diagnostics | Runtime, memory, FD error, exact-reference error, per-seed dispersion, component diagnostics, and route parity against historical diagnostics. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, runtime ranking, all-algorithm readiness, or release/public benchmark validity. |
| Artifacts | Master program, visible runbook, ledger, stop handoff, phase subplans/results, review bundles, compact score artifacts, tests, logs, and final leaderboard score integration artifact. |

## Phase Index

| Phase | Name | Purpose | Subplan | Required result artifact |
| --- | --- | --- | --- | --- |
| 0 | Route demotion and policy gate | Demote reverse/manual-total-VJP routes and add/plan admission guards. | `docs/plans/bayesfilter-ledh-compact-score-default-phase0-route-demotion-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase0-route-demotion-result-2026-07-08.md` |
| 1 | Shared compact score contract | Define common compact route IDs, artifact validation, static guards, and no-record tests. | `docs/plans/bayesfilter-ledh-compact-score-default-phase1-contract-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase1-contract-result-2026-07-08.md` |
| 2 | LGSSM reference freeze | Freeze LGSSM compact score as reference default and quarantine historical reverse diagnostics. | `docs/plans/bayesfilter-ledh-compact-score-default-phase2-lgssm-reference-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase2-lgssm-reference-result-2026-07-08.md` |
| 3 | Actual-SV compact port | Replace actual-SV reverse-record score with compact forward sensitivity. | `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-result-2026-07-08.md` |
| 4 | Fixed-SIR compact port | Replace or block fixed-SIR manual-total-VJP admission with compact forward sensitivity. | `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-result-2026-07-08.md` |
| 5 | Predator-prey compact port | Replace predator-prey reverse score with compact forward sensitivity. | `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-result-2026-07-08.md` |
| 6 | Generalized-SV compact port | Build compact generalized-SV score for its admitted value target. | `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-result-2026-07-08.md` |
| 7 | KSC-SV compact port | Build compact KSC finite-mixture score without exact-SV overclaim. | `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-result-2026-07-08.md` |
| 8 | Leaderboard integration | Rebuild LEDH leaderboard score integration using compact scores only. | `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-result-2026-07-08.md` |

## Dependency Order

1. Phase 0 must run first because it changes the admission boundary from
   "manual total VJP may be admissible" to "compact score only".
2. Phase 1 must run before model ports because validators and static tests must
   reject old route labels and reverse-record score implementations.
   Until Phase 1 lands those code-level guards, this boundary is procedural:
   do not run any full-admission command, including any command with
   `--admit-full`, that could validate a `manual_total_vjp*` route.
3. Phase 2 freezes LGSSM as the reference style before porting nonlinear rows.
4. Actual-SV runs before KSC-SV because KSC must preserve SV-family target
   boundaries without inheriting exact transformed actual-SV claims.
5. Fixed-SIR runs after actual-SV only if Phase 1 guard coverage is in place;
   otherwise it may be swapped with Phase 3 after a reviewed subplan update.
6. Phase 8 may run only after every model row is either compact-admitted or
   explicitly blocked with a result artifact.

## Required Subplan Fields

Each phase subplan must state:

- phase objective;
- entry conditions inherited from the previous phase;
- required artifacts;
- required checks/tests/reviews;
- evidence contract;
- forbidden claims/actions;
- exact next-phase handoff conditions;
- stop conditions.

At the end of each phase:

1. run required local checks;
2. write a phase result or blocker record;
3. draft or refresh the next subplan;
4. review the next subplan for consistency, correctness, feasibility, artifact
   coverage, and boundary safety.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Phase 0 freezes admitted value row IDs and current score route inventory. |
| Proxy metrics | Runtime, memory, tiny FD, and finite output cannot admit a row without compact route and predeclared same-scalar FD or exact-reference agreement at the tested point/coordinates. |
| Missing stop conditions | Every phase stops on wrong scalar, old route default, tape/autodiff, partial derivative, memory blowup, or unsupported claim. |
| Unfair comparison | Historical reverse routes are comparators only; they are not promotion baselines. |
| Hidden assumption | Each model phase must name parameters, target observation policy, and compact tangent recurrence before implementation. |
| Stale context | Phase 0 inspects current route IDs and validator allowlists before edits. |
| Environment mismatch | GPU/CUDA commands require trusted execution; CPU-only checks must hide GPU before TensorFlow import. |
| Useless artifacts | Every result must identify the exact target scalar and route style; no vague "same scalar" claims without artifact validation. |

Launch audit status: passed for creating the launch package only. Execution
begins with Phase 0 after local checks and read-only review.

## Review Policy

Claude may be used as a read-only reviewer for material subplans, phase
results, and repair decisions. Claude is not an execution authority and cannot
authorize human, runtime, model-file, funding, product-capability, public
release, or scientific-claim boundaries. Codex remains supervisor and executor.

Review bundles must be bounded and fixed-path. If Claude does not respond, run
a tiny probe. If the probe succeeds, narrow or rewrite the prompt. If Claude is
unavailable or policy-blocked, replace the review with a fresh Codex read-only
review and record the limitation.

## Approvals

Standing approvals already provided in this thread cover:

- trusted GPU/CUDA/TensorFlow/XLA checks, including `N=10000` score-memory
  tests;
- bounded Claude read-only review commands for this repository;
- sending bounded BayesFilter artifacts to Claude for read-only review.

Still stop for approval before package installation, network/data fetches,
destructive git actions, changing row targets, changing pass/fail criteria
after seeing results, or public/release/scientific-claim boundary crossings.

## Nonclaims

This launch package does not admit any new score, does not complete any model
port, and does not establish HMC readiness, posterior correctness, scientific
superiority, or all-algorithm leaderboard readiness.

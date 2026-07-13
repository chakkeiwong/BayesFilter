# LEDH Score Per-Model Master Program

Date: 2026-07-07

Status: `DRAFT_LAUNCH_GATE`

Upstream value layer:

- `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-result-2026-07-07.md`

## Objective

Repair and admit LEDH scores one model row at a time, using the value-layer
artifacts as the target-scalar anchor.

Here `score` means the total derivative of the admitted finite-`N` LEDH
observed-data log likelihood estimator with respect to the row's stated free
parameter vector.

The differentiated scalar is:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

Admitted score computation must be analytical or manual no-tape total
derivative. `GradientTape`, `ForwardAccumulator`, hidden autodiff, stopped
partial derivatives, and score routes that differentiate a different scalar are
forbidden for admitted LEDH score evidence.

## Inherited State

The value-only runbook closed with six main LEDH value rows admitted:

- `benchmark_lgssm_exact_oracle_m3_T50`;
- `zhao_cui_spatial_sir_austria_j9_T20`;
- `zhao_cui_predator_prey_T20`;
- `zhao_cui_sv_actual_nongaussian_T1000`;
- `zhao_cui_generalized_sv_synthetic_from_estimated_values`;
- `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`.

The value integration artifact explicitly blocked score integration:

```text
score_integration_status = blocked_out_of_scope_forward_scalar_only
```

This score program starts from that boundary. It must not infer score
admission from value admission.

Known score history:

- LGSSM has local no-tape same-scalar score machinery and prior tiny/full-score
  diagnostics, but it must be rechecked against the new value row-set and
  score artifact schema before score integration.
- Fixed SIR has local manual no-tape same-scalar tiny evidence for the main row
  and historical/scoped diagnostic evidence for a parameterized SIR row. The
  scoped diagnostic row must not be promoted as the main row.
- Predator-prey, actual-SV, generalized-SV, and KSC-SV have newly admitted
  value scalars but no score admission under this program.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can each of the six admitted LEDH value rows produce an admitted no-tape total derivative of the same finite-`N` `log_likelihood` scalar? |
| Baseline/comparator | Phase 8 value integration artifact, Phase 2-7 value artifacts, existing LGSSM/fixed-SIR no-tape diagnostics, exact derivatives where available, and same-scalar finite differences with fixed randomness otherwise. |
| Primary pass criterion | A row score is admitted only if its row value artifact is admitted, the score route differentiates the same scalar and row parameter vector, no tape/autodiff route is used, tiny checks pass, `N=10000` correctness and memory checks pass, and replay tests validate a score artifact. |
| Veto diagnostics | Score before value; value/score row-set mismatch; diagnostic SIR promotion; `GradientTape`/`ForwardAccumulator`; hidden autodiff; stopped partial derivative; wrong target scalar; wrong parameter vector; nonfinite score; finite-difference/exact mismatch beyond predeclared tolerance; memory failure; runtime-only promotion. |
| Explanatory diagnostics | Runtime, compile time, memory, FD error, exact-reference error, MCSE, per-seed dispersion, component decomposition, and device placement. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, runtime ranking, and all-algorithm comparison. |
| Artifacts | Master program, visible runbook, execution ledger, stop handoff, phase subplans/results, review bundles, score artifacts, tests, logs, and final value-score integration artifact if all gates pass. |

## Phase Index

| Phase | Name | Purpose | Subplan | Required result artifact |
| --- | --- | --- | --- | --- |
| 0 | Baseline and score governance | Freeze value row-set, score meaning, tape ban, and current score blockers. | `docs/plans/bayesfilter-ledh-score-per-model-phase0-baseline-governance-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase0-baseline-governance-result-2026-07-07.md` |
| 1 | Score artifact schema and guards | Define replayable score artifact schema, same-scalar route IDs, memory fields, and no-tape guards. | `docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-result-2026-07-07.md` |
| 2 | LGSSM score | Reconfirm/admit LGSSM no-tape same-scalar score at `N=10000`. | `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-result-2026-07-07.md` |
| 3 | Fixed SIR score | Admit the main fixed SIR row score, not the diagnostic parameterized row. | `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-result-2026-07-07.md` |
| 4 | Predator-prey score | Build/admit additive-Gaussian predator-prey no-tape total score. | `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-result-2026-07-07.md` |
| 5 | Actual-SV score | Build/admit exact transformed actual-SV no-tape total score. | `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-result-2026-07-07.md` |
| 6 | Generalized-SV score | Build/admit source-route generalized-SV no-tape total score. | `docs/plans/bayesfilter-ledh-score-per-model-phase6-generalized-sv-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase6-generalized-sv-result-2026-07-07.md` |
| 7 | KSC-SV score | Build/admit finite-mixture KSC no-tape total score without exact-SV overclaim. | `docs/plans/bayesfilter-ledh-score-per-model-phase7-ksc-sv-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase7-ksc-sv-result-2026-07-07.md` |
| 8 | Value-score integration | Rebuild LEDH value-score artifact only from rows admitted in Phases 2-7. | `docs/plans/bayesfilter-ledh-score-per-model-phase8-integration-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase8-integration-result-2026-07-07.md` |

## Dependency Order

- Phase 0 and Phase 1 are mandatory before model score work.
- LGSSM runs first because it is the linear-Gaussian exact-reference row.
- Fixed SIR runs second because main-row score promotion must be separated from
  the old diagnostic parameterized row.
- Predator-prey runs before SV rows because it exercises a nonlinear Gaussian
  target without transformed heavy-tail likelihoods.
- Actual-SV precedes KSC-SV because KSC must preserve the transformed-SV
  target discipline while using its own finite-mixture target.
- Generalized-SV and KSC-SV must preserve their distinct value target labels.
- Phase 8 may run only if every main value row has either an admitted score or
  an explicit blocker. If any main row remains blocked, Phase 8 must write a
  blocker/result that refuses full value-score integration.

## Phase Discipline

Each phase must have a dedicated subplan before execution. Each subplan must
state:

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
2. write a phase result or blocker result;
3. draft or refresh the next subplan;
4. review the next subplan for consistency, correctness, feasibility, artifact
   coverage, and boundary safety.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Phase 0 must read the Phase 8 value integration artifact and freeze the six main row ids. |
| Proxy metrics | Runtime, memory, finite output, and FD-smoke success do not admit a score without same-scalar total derivative and replay. |
| Missing stop conditions | Every phase must stop on tape/autodiff, partial derivative, wrong scalar, wrong theta, score/value row mismatch, and diagnostic-row promotion. |
| Hidden assumptions | Every model phase must name parameter coordinates and the exact row target before code changes. |
| Stale context | Phase 0 inventories current score artifacts and blockers before implementation. |
| Environment mismatch | GPU/CUDA/XLA score-memory checks require trusted execution; CPU-only diagnostics must hide GPU before TensorFlow import. |
| Useless artifacts | Every score artifact must be replayable from disk and tie back to the admitted value artifact. |

Launch audit status: passed for creating a launch package only. Execution
begins with Phase 0 after local checks and read-only review.

## Review Policy

Claude may be used as read-only reviewer for material subplans, phase results,
and repair decisions. Claude has no edit, execution, approval, policy,
scientific-claim, funding, product-capability, model-file, or boundary
authority. Codex remains supervisor and executor.

Review bundles must be bounded and fixed-path or packet-only. If Claude does
not respond, run a tiny probe. If the probe succeeds, narrow or rewrite the
prompt and retry. If Claude is unavailable or policy-blocked, replace Claude
review with a fresh Codex read-only review and record that limitation.

## Human Approval Boundaries

Stop for human approval before:

- changing a row target definition;
- changing score pass/fail criteria after seeing results;
- using package/network/data fetches;
- destructive git or filesystem actions;
- crossing model-file, funding, product-capability, public-release, or
  scientific-claim boundaries.

Trusted GPU/CUDA/XLA runs and bounded Claude read-only reviews have standing
approval from the user in this thread.

## Nonclaims

- This master program does not admit any score at launch.
- This master program does not establish HMC readiness, posterior correctness,
  runtime ranking, all-algorithm comparison, or scientific superiority.

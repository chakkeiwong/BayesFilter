# Fable R2 Bounded Review Request: Revised Tuning Plan

READ-ONLY BOUNDED REVIEW. Review exactly this file and nothing else unless the
file itself explicitly asks you to inspect the one cited plan path. Do not edit,
run commands, launch agents, or review the whole repository.

## Exact Review Target

Review exactly:

`docs/plans/bayesfilter-tuning-streamline-refactor-plan-2026-08-16.md`

## Question

Fable's prior coverage audit returned `AUDIT_VERDICT: REVISE` and
`PLAN_VERDICT: REVISE`. Codex independently audited F1-F5 and revised the plan
as follows:

1. Added focused run-level baselines: MacroFinance 90 passed / 11 failed and
   dsge_hmc 24 passed / 2 failed, plus a four-entry source-anchored drift ledger.
2. Required adjudicated repairs before Phases 4-6; preserving the failing
   baseline is not a migration pass.
3. Added `tests/test_hmc_mass_matrix.py` and
   `tests/test_hmc_windowed_mass_adaptation.py` as an unfiltered command.
4. Added a deliberately mismatched but valid SPD mass holdout arm, with fresh
   mass-specific epsilon/`L` retuning and independent posterior-validity checks.
5. Classified `bayesfilter/inference/mass_matrix.py` as NumPy migration debt and
   required a TensorFlow/TFP canonical mass path in Phase 2.
6. Required non-Gaussian curved/varying-Hessian evidence or an explicit owner
   waiver before robust numeric-default promotion.
7. Clarified that joint mass/epsilon/`L` tuning is staged and conditional, and
   distinguished one-epsilon-then-`L` from per-`L` fresh epsilon tuning.

Audit whether those repairs are sufficient, operational, and internally
consistent. Also adjudicate two explicit Codex rebuttals:

- F1: the 13 failures are four technical entries, not three uniform
  BayesFilter drift families. The extra L10d redaction failure is a brittle
  substring assertion, and producer/consumer source authority may establish a
  stale test without blanket owner sign-off. Owner direction is reserved for a
  genuinely unresolved intended-contract choice.
- F3: a valid SPD mass can preserve the target and fresh epsilon/`L` tuning may
  compensate. Deterministic mismatch diagnostics must distinguish the bad arm,
  but repair should fire only under a predeclared health, validity, or efficiency
  condition, not solely because the mass differs from analytic covariance.

Do not treat planned fixtures or migration repairs as completed evidence. Do
not infer posterior correctness, convergence, sampler superiority, GPU
behavior, production readiness, scientific validity, or default readiness.

Report findings first with exact plan line anchors. Separate blocking plan
defects from open implementation work already represented as fail-closed gates.

End with exactly two lines:

`AUDIT_VERDICT: AGREE` or `AUDIT_VERDICT: REVISE`

`PLAN_VERDICT: AGREE` or `PLAN_VERDICT: REVISE`

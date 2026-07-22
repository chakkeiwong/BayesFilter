# LGSSM NeuTra Gap Closure Phase 1 - Fresh Seed 1201 Training

Date: 2026-07-15  
Status: `COMPLETE_PASS`  
Parent: `docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-plan-2026-07-15.md`

## Objective

Train `dense_seed1201=(20260713,1201)` from fresh initialization for exactly
5,000 steps using the frozen `wide_2x_lr5e3` recipe and produce one immutable
strict graph-native candidate for downstream frozen-transport validation.

## Entry Conditions

- Phase 0 decision is `PASS_PHASE0_COMPATIBILITY_GATE`.
- Selected recipe file SHA-256 is
  `1984c33142496ecbbd77ecaea17b1d3dc3320caa45a1b08aa947439ca7088c97`.
- Output root
  `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/long-training-attempt-01`
  does not contain an existing seed1201 job.
- Remaining aggregate budgets are 45 compiled minutes and 60 wall minutes.

## Command

Run in trusted/escalated GPU context from the repository root:

```bash
python docs/benchmarks/run_lgssm_neutra_target_specific_protocol_2026_07_14.py \
  train \
  --job-kind final \
  --job-id dense_seed1201 \
  --artifact-root docs/plans/artifacts/neutra-batch-native-training-2026-07-14/long-training-attempt-01 \
  --selected-recipe docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/screen-500/selected_recipe.json
```

## Required Artifacts And Checks

- strict `result.json`, terminal checkpoint, frozen payload, and progress log;
- exact job, seed, target, adapter, recipe, and selected-source identities;
- trusted RTX 4080 SUPER GPU evidence, memory growth, no CPU fallback, XLA JIT,
  float64, one compiled training invocation, and `tf_while_loop`;
- exactly 5,000 completed steps with finite values/gradients and valid target
  status in every emitted record;
- frozen reload, forward, logdet, pullback-score, and logdet-score parity;
- repository import closure without NumPy or host callback;
- wall and compiled time charged to the shared campaign budget.

## Evidence Contract

A pass establishes one fresh engineering-valid 5,000-step frozen transport. It
does not establish training sufficiency, HMC convergence, posterior agreement,
recovery, superiority, or default readiness. Loss and clipping are explanatory.

## Forbidden Claims And Actions

- no screen-weight, checkpoint, or optimizer-state reuse;
- no recipe, seed, target, batch, step, optimizer, or hardware-class change;
- no non-JIT or CPU fallback;
- no overwrite of any existing output;
- no HMC tuning from this result until Phase 3 frozen validation passes.

## Handoff Conditions

On pass, write the seed1201 close record, validate it through
`validate_strict_training_result`, charge observed time, and refresh the
seed1202 subplan with the remaining budget. A scientifically weak loss or
descriptive difference is not a continuation veto.

## Repair And Stop Conditions

A localized infrastructure interruption may be retried once from fresh
initialization in `long-training-attempt-02` after a focused repair check. Do
not resume terminal-only graph-native state. Stop for recipe/target identity
drift, numerical/status/parity failure, trusted GPU/XLA unavailability after a
trusted probe, or shared training-budget exhaustion.

## Suitability Review

The command exactly matches the accepted handoff; Phase 0 has removed the
downstream schema/backend blocker; the root is fresh; memory growth is mandatory
and recorded; the budget is unchanged. Verdict: `SUITABLE_TO_EXECUTE`.

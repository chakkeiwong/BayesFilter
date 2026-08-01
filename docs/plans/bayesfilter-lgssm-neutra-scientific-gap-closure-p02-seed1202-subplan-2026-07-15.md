# LGSSM NeuTra Gap Closure Phase 2 - Fresh Seed 1202 Training

Date: 2026-07-15  
Status: `COMPLETE_PASS`  
Parent: `docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-plan-2026-07-15.md`

## Objective

Train the independent `dense_seed1202=(20260713,1202)` candidate for exactly
5,000 steps using the same immutable selected recipe, with no seed1201 state
reuse, and close the two-seed training phase.

## Entry Conditions

- Phase 1 passed after a localized post-training import-closure repair.
- The repair changed only `bayesfilter.runtime` import timing and passes the
  exact memory-policy/closure regression.
- Seed1202 output root is fresh.
- Remaining aggregate training wall budget is approximately `2781.17 s`; the
  compiled-program ceiling remains sufficient for one measured-rate job.

## Command

```bash
python docs/benchmarks/run_lgssm_neutra_target_specific_protocol_2026_07_14.py \
  train \
  --job-kind final \
  --job-id dense_seed1202 \
  --artifact-root docs/plans/artifacts/neutra-batch-native-training-2026-07-14/long-training-attempt-01 \
  --selected-recipe docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/screen-500/selected_recipe.json
```

## Required Artifacts And Checks

The requirements are identical to Phase 1: exact identity, 5,000 steps, fresh
seed, GPU memory growth, XLA, one compiled invocation, valid finite target
records, terminal checkpoint/payload, frozen reload and score parity, and a
NumPy-free imported execution closure. Record observed time and compare losses
or gradients descriptively only.

## Evidence Contract

Phase 2 establishes a second independent engineering-valid frozen transport.
Together with seed1201 it enables downstream validation; neither loss nor a
between-seed difference establishes transport or sampler superiority.

## Forbidden Claims And Actions

No checkpoint, weights, optimizer state, or random stream may be reused from
seed1201. Do not change recipe, target, batch, optimizer, steps, or hardware
class. Do not use CPU/non-JIT fallback or overwrite evidence. Do not select one
seed using training loss.

## Handoff Conditions

After terminal checks, write the seed1202 and two-seed close record, then draft
Phase 3 with the exact checkpoint/payload hashes and frozen-objective parity
commands. At least one valid candidate is sufficient to continue; a rejected
seed is candidate rejection, not automatic direction rejection.

## Repair And Stop Conditions

One fresh infrastructure retry is allowed only for a localized failure under
the unchanged contract and shared budget. Stop for identity, numerical,
target-status, parity, GPU/XLA, or aggregate-budget veto.

## Suitability Review

The runtime-import defect is fixed and regression tested. Seed1202 is fresh,
the recipe and scientific contract are unchanged, and measured seed1201 timing
leaves ample budget. Verdict: `SUITABLE_TO_EXECUTE`.

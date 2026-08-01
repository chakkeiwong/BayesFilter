# Fresh 5,000-Step LGSSM NeuTra Campaign Handoff

Date: 2026-07-14

## Campaign Objective

Train two fresh `wide_2x_lr5e3` dense-IAF transports for 5,000 optimizer steps
each through the certified exact LGSSM batch-native GPU/XLA route, then preserve
the frozen transports for a separately reviewed downstream validation phase.
This handoff authorizes training only. It does not authorize HMC execution or
scientific/default promotion.

## Entry Conditions

- Master migration phases 0-8 are complete.
- Accepted Phase 7 selected-recipe artifact:
  `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/screen-500/selected_recipe.json`.
- Selected-recipe artifact hash:
  `sha256:00bf189dd8697c33b0378bda92a75d2df74d85ffb0e754f1df7c6dabcb216ac0`.
- Selected-recipe file SHA-256:
  `1984c33142496ecbbd77ecaea17b1d3dc3320caa45a1b08aa947439ca7088c97`.
- The harness requires that exact artifact explicitly and validates its result
  reference before GPU initialization or output creation.

## Evidence Contract

| Item | Contract |
| --- | --- |
| Question | Can the proxy-nominated recipe complete two fresh long trainings with valid exact-target status and reproducible artifacts? |
| Exact target | T=120, 18-parameter deterministic LGSSM, batch binding v2, SVD/eigh graph-status authority |
| Candidate | `wide_2x_lr5e3`, batch 128, 5,000 steps, two predeclared seeds |
| Pass criterion | each job completes exactly 5,000 steps, all recorded values/gradients finite, all target status valid, no floors/fallbacks, GPU/XLA provenance and binding closure preserved, TensorFlow memory growth verified on every physical GPU, frozen reload parity passes |
| Hard veto | wrong selected-recipe/hash chain, wrong device/XLA/batch/binding, unverified GPU memory policy, nonfinite loss/gradient/value, invalid target status, active floor, artifact corruption, or budget exhaustion |
| Explanatory only | training loss, gradient norms, program/wall time, and between-seed differences |
| Nonclaims | successful training does not establish posterior correctness, HMC convergence, recipe superiority, calibration, robustness, generalization, or default readiness |
| Artifact root | `docs/plans/artifacts/neutra-batch-native-training-2026-07-14/long-training-attempt-01` |

## Seeds And Commands

Run sequentially in trusted/escalated GPU context from the repository root.
Do not set `CUDA_VISIBLE_DEVICES=-1`. The harness sets
`TF_FORCE_GPU_ALLOW_GROWTH=true`, applies repository memory growth before
logical-device initialization, fails closed if verification fails, and records
the result in `gpu_manifest.gpu_memory_policy`.

```bash
python docs/benchmarks/run_lgssm_neutra_target_specific_protocol_2026_07_14.py \
  train \
  --job-kind final \
  --job-id dense_seed1201 \
  --artifact-root docs/plans/artifacts/neutra-batch-native-training-2026-07-14/long-training-attempt-01 \
  --selected-recipe docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/screen-500/selected_recipe.json
```

```bash
python docs/benchmarks/run_lgssm_neutra_target_specific_protocol_2026_07_14.py \
  train \
  --job-kind final \
  --job-id dense_seed1202 \
  --artifact-root docs/plans/artifacts/neutra-batch-native-training-2026-07-14/long-training-attempt-01 \
  --selected-recipe docs/plans/artifacts/neutra-batch-native-training-2026-07-14/phase7/screen-500/selected_recipe.json
```

| Job | Seed |
| --- | --- |
| `dense_seed1201` | `(20260713, 1201)` |
| `dense_seed1202` | `(20260713, 1202)` |

## Compute Budget

The nominated 500-step arm used `82.99 s` compiled-program time and `247.53 s`
total wall time. Linear estimates are approximately `13.8 min` compiled and
`16.6 min` wall per 5,000-step seed.

- aggregate compiled-program ceiling: `45 min`;
- aggregate wall ceiling: `60 min`;
- run sequentially on the same RTX 4080 SUPER hardware class;
- use verified TensorFlow memory growth; this prevents eager whole-device
  reservation but is not a hard cap;
- no screen-weight, checkpoint, or optimizer-state reuse;
- no package/environment mutation;
- no HMC or downstream posterior comparison in this campaign.

## Continuation And Repair

After each seed:

1. run the source artifact/hash, binding, GPU/XLA, finite/status, progress, and
   frozen reload checks already emitted by the strict harness;
2. write a seed result/close record with command, device, times, hashes,
   diagnostics, decision, and nonclaims;
3. refresh the next-seed or downstream-validation subplan from the actual result;
4. review next-step suitability and continue when no real target, numerical,
   artifact, device, or budget blocker exists.

A localized infrastructure failure may be retried once into a new root such as
`long-training-attempt-02`, after a focused repair check, while target, recipe,
seed, hardware class, and total budget remain unchanged. The current compiled
trainer writes terminal-only checkpoints, so partial optimizer state is not an
authorized implicit resume source. Preserve the failed attempt and restart the
affected seed fresh.

## Stop Conditions

Stop for a hard veto, corrupted selection/artifact identity, GPU/XLA mismatch,
target invalidity, numerical failure, or aggregate budget exhaustion. A higher
loss or descriptive difference between seeds is not by itself a continuation
veto. After both seeds pass, draft the downstream frozen-transport/posterior-HMC
validation subplan; do not infer that validation from training completion.

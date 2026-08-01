# LGSSM NeuTra Gap Closure Phase 1 Result - Seed 1201

Date: 2026-07-15  
Decision: `PASS_PHASE1_SEED1201_AFTER_LOCALIZED_HARNESS_REPAIR`

## Outcome

`dense_seed1201=(20260713,1201)` completed all 5,000 fresh GPU/XLA
optimization steps under the selected `wide_2x_lr5e3` recipe. Program time was
`813.4168 s`; total strict invocation time was `818.8255 s`. The terminal
checkpoint and frozen transport are valid and immutable.

The original strict result was rejected after training because importing the
focused GPU memory policy first executed `bayesfilter.runtime.__init__`, which
eagerly imported the unrelated NumPy-backed generic runtime runner. The
optimization, target, checkpoint, and frozen payload did not use that runner.
The runtime package was made lazy, the exact import regression now passes, and
a trusted-GPU post-validator restored the preserved terminal checkpoint and
recomputed frozen parity. The original rejected artifact was preserved.

## Evidence

| Field | Result |
| --- | --- |
| Steps / batch | `5000 / 128` |
| Recipe / seed | `wide_2x_lr5e3 / (20260713,1201)` |
| Execution | RTX 4080 SUPER, float64, memory growth, XLA, one compiled `tf.while_loop` invocation |
| Target status | all emitted records valid; zero target floors at terminal step |
| Terminal loss | `43.096647` (explanatory only) |
| Terminal raw gradient norm | `2.458150`; not clipped |
| Frozen parity | forward, logdet, pullback score, and logdet score max absolute difference all `0.0` |
| Checkpoint SHA-256 | `a5519a74e02b259cc0558223384714da7c8ee4a71148b70eb4436ce3083a8384` |
| Frozen payload SHA-256 | `6429977ba1754ce5f36248104c82fa18639311a0727298bc3ed436b4a670a745` |
| Post-validation artifact hash | `sha256:ba66acef9fb18b915957e86b5ed89d894d7ae3d8ccd1ce25efc46f0d89cf7305` |
| Focused repair tests | `38 passed, 2 warnings` |

Primary evidence:
`docs/plans/artifacts/lgssm-neutra-gap-closure-2026-07-15/phase1/seed1201_post_validation.json`.

## Failure Classification And Repair

The first terminal classification was
`post_training_unrelated_eager_import_false_positive`. It did not invalidate
the harness mathematics, target, optimization, training data, checkpoint, or
frozen transport. Repair changed only runtime-package import timing. No training
state was resumed, replayed, or changed, and the one permitted infrastructure
retry was not consumed.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve seed1201 as a Phase 3 candidate | exact 5,000-step completion plus restored frozen parity passes | no identity, finite, target-status, GPU/XLA, memory, closure, or parity veto | downstream HMC geometry and posterior validity not tested | train independent seed1202 | no HMC, posterior, recovery, superiority, or default claim |

## Handoff

Shared training wall budget used: `818.8255 s`; approximately `2781.17 s`
remains. Continue with the refreshed seed1202 subplan.

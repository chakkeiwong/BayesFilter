# SSL-LSTM q=20 Single-Seed NeuTra Training Diagnostic Result

Date: 2026-07-20  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-single-seed-neutra-training-diagnostic-plan-2026-07-20.md`  
Decision: `MECHANISM_PATH_VALID_FIXED_SMOKE_STREAM_VETOED_BY_SATURATION`

## Scope

One fresh `seed-a` q=20 stream was run to test the revised 2,000-step NeuTra
training path end to end. This was deliberately not a Phase 3 admission run:
q=20 has no Optuna nomination receipt, the parameters were the existing fixed
smoke values, and one seed cannot establish robustness.

Parameters:

```text
learning_rate=0.0004
initialization_scale=0.01
gradient_clip_norm=10.0
batch_size=480
validation_batch_size=64
validation_check_every=250
patience_steps=250
post_repair_no_improvement_cycles=2
maximum_steps=2000
learning_rate_factor=0.5
```

## Engineering Launch Record

The first two fresh launches were invalid before training and are retained only
as debugging evidence:

1. `run-01` failed because the runner called `set_memory_growth` after project
   imports had initialized TensorFlow.
2. `run-02` failed because spawned workers re-imported the benchmark and the
   top-level GPU selector overwrote their inherited `CUDA_VISIBLE_DEVICES=-1`.

The runner was repaired by enabling memory growth immediately after the
TensorFlow import and honoring the `BAYESFILTER_CPU_VALUE_SCORE_WORKER=1`
marker before parent GPU selection. The repaired boundary passed:

- `8` q-general runner tests;
- `6` process-parallel worker visibility/startup/native-external parity tests;
- Python compilation and `git diff --check`.

Only `run-03` is scientific/engineering execution evidence.

## Valid Run

Artifact root:
`docs/plans/artifacts/ssl-lstm-q20-single-seed-neutra-diagnostic-2026-07-20/run-03/`.

| Field | Value |
| --- | --- |
| Status | `COMPLETED` summary, `DIAGNOSTIC_VETOED` stream |
| q | `20` |
| Stream | `seed-a` |
| Selected physical GPU | `1` |
| Parent logical GPU | `/device:GPU:0` after visibility remap |
| Worker topology | `16` configured CPU workers, GPU hidden in worker receipts |
| Tensor dtype | `float64` |
| Parent JIT | XLA enabled |
| TF32 | enabled |
| Cumulative charged time | `4705.69897` seconds (`78.43` minutes) |
| Declared cap | `28800` seconds (`8` hours) |
| Parent high-water RSS | `1.12` GB |
| Worker active RSS sum at terminal support probe | `13.58` GB |
| GPU allocator process memory during run | about `0.83` GB |
| Terminal program step | `750` |
| Best step | `500` |
| Learning-rate reductions | `0` |

## Validation History

| Step | Heldout mean | Paired mean delta vs best | One-sided upper bound | Saturation | Action | Best |
| ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 0 | 79.75985549 | N/A | N/A | finite/eligible | `initialize_best` | 0 |
| 250 | 43.49455328 | -36.26530221 | -24.73490445 | 0.00390625 | `improved` | 250 |
| 500 | 42.16940131 | -1.32515198 | -1.00152806 | below cap | `improved` | 500 |
| 750 | 41.82412910 | not evaluated after veto | not evaluated after veto | 0.05989583 | `stop` / `scale_saturation_above_cap` | 500 |

The step-500 paired mean delta versus the step-250 best was
`-1.3251519765986692`; its recorded one-sided upper bound was `-1.00152806`. At step 250
and step 500 the upper bounds were strictly negative, establishing meaningful
heldout improvement under the predeclared controller rule. At step 750
saturation exceeded the hard cap, so the controller stopped before comparing
loss to the best checkpoint or halving the learning rate. All terminal support
probes were finite with round-trip residual below `3.6e-15` and moderate-shell
inverse radius about `4.0`.

## Interpretation

The full q=20 mechanism path is operational through target construction,
CPU-hidden process workers, external value/score training updates, validation,
support probing, best-state checkpointing, frozen transport export, and hard
stop handling. The fixed-smoke stream is rejected because its learned scales
crossed the declared saturation cap at step 750. This weakens the convenience
hyperparameter hypothesis only.

The run does not establish that 2,000 steps is sufficient, that the 250-step
cadence is optimal, that the LR-repair branch is effective at q=20, or that a
q=20 NeuTra transport is suitable for HMC. It also does not establish seed
robustness, posterior correctness, convergence, or scientific validity.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep the adaptive runner path | End-to-end checkpoints and finite worker/target/support artifacts | Launch bugs repaired; candidate saturation veto at step 750 | Fixed-smoke LR/initialization may be unsuitable; live LR repair not reached | Run a separately authorized q=20 hyperparameter repair or lower-LR diagnostic designed to avoid immediate saturation | No q=20 admission, HMC, posterior, or scientific claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Mechanism artifacts passed; stream vetoed by scale saturation |
| Statistically supported ranking | Not applicable; one seed and no candidate comparison |
| Descriptive-only differences | Heldout loss trajectory and runtime are descriptive for this stream |
| Default-readiness | Not established |
| Next evidence needed | q=20 target-specific tuning/repair, then independent seed and downstream HMC gates |

## Post-Run Red Team

The strongest alternative explanation is that the fixed-smoke learning rate or
initialization scale drives the dense IAF scales into saturation, rather than
the q=20 geometry being intrinsically untrainable. A lower-LR or tuned q=20
repair that remains support-eligible through multiple boundaries would weaken
the candidate-failure explanation. Conversely, repeated saturation across
target-specific tuned settings would be evidence about the q=20 transport
candidate, not about the correctness of the controller implementation.

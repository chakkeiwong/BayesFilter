# LGSSM NeuTra Target-Specific Protocol Phase A Result

Date: 2026-07-14  
Status: `PASS_IMPLEMENTATION_AND_REVIEW`  
Plan: `docs/plans/bayesfilter-lgssm-neutra-target-specific-training-protocol-amendment-2026-07-14.md`

## Decision

Phase A implementation and local verification pass. The target-specific GPU
screen may begin after bounded implementation review converges. The expanded
long-budget campaign still requires the owner's plain approval of the compute
budget stated in the amendment.

## Changes

- Added four explicit target-specific training recipes and a deterministic
  screen/final seed ledger.
- Added a versioned v1 campaign contract and output root.
- Added common held-out batches with paired-difference MCSE nomination.
- Added a terminal zero-survivor result.
- Added exact trainable-versus-frozen forward, logdet, value, and explicit-score
  parity on GPU/XLA.
- Added same-configuration infrastructure resume into a fresh output directory,
  with exact parent-checkpoint hash and lineage.
- Reused the existing modern Phase 5 tuning and Phase 6 serious-sampling code
  through an explicit execution context; no second sampler was implemented.
- Preserved the interrupted 1,000-step artifacts as historical diagnostic
  evidence and did not resume or overwrite them.

## Local Checks

Command:

```text
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tf-gpu/bin/python3.11 -m pytest -q tests/test_lgssm_neutra_target_specific_protocol.py tests/test_lgssm_neutra_serious_validation.py tests/test_neutra_training.py tests/test_fixed_transport_hmc_tuning.py
```

Result: `55 passed`, with two TensorFlow Probability third-party deprecation
warnings. `py_compile` and `git diff --check` also passed for all changed
protocol, trainer, CLI, test, and plan files.

## Implementation Review

Claude Code reviewed exactly
`bayesfilter/testing/lgssm_neutra_target_specific_protocol_tf.py` with
read-only tools at Opus/max effort. The initial verdict was `REVISE`. Material
findings were patched visibly and covered by focused regressions:

- source-anchor-failed selection now retains the actual lowest held-out mean;
- every consumed training-job result is bound to recipe, seed, steps, target,
  and adapter identity;
- amended Phase 4 cannot pass on the reused affine control alone;
- the reused affine result and payload are bound to exact historical hashes
  and identities;
- each Phase 5/6 candidate is individually checked against the learned survivor
  set or the separately rebound affine control;
- learned payload, terminal `checkpoint_step_005000.json`, and progress
  references are rebound to exact candidate/attempt paths and rechecked by
  SHA-256 and byte count.

The final one-finding convergence review returned `VERDICT: AGREE` with no
remaining material defect in the canonical learned-artifact path binding.

## Contract

Path:

`docs/benchmarks/artifacts/lgssm_neutra_target_specific_protocol_2026_07_14/campaign_contract.json`

- Embedded contract hash:
  `4bf98c1b5714563bf7f48115304027c8a41fb25c0c09e561e499d02aebe0dbe3`
- File SHA-256:
  `6815261d1c66fe85160b2aa76b5bd97bfc9872efe21da3da31e94b36d539e2d9`
- Recipes: four explicit complete recipes.
- Final training: 5,000 steps, batch 128, immutable checkpoint every 50
  steps, two fresh training seeds.

## Trusted GPU/XLA Preflight

Trusted `nvidia-smi` passed on an NVIDIA GeForce RTX 4080 SUPER. A corrected
TensorFlow framework probe compiled and executed a float64 matrix product under
XLA on `/GPU:0`.

| Field | Value |
| --- | --- |
| TensorFlow | `2.19.1` |
| Physical/logical GPU | one GPU, `/GPU:0` |
| XLA | compiled cluster confirmed |
| Output device | `/job:localhost/replica:0/task:0/device:GPU:0` |
| dtype | `float64` |
| Soft placement | `false` |
| TF32 | `true` |

The first attempted framework probe had a Python one-line decorator syntax
error and performed no framework computation. It was corrected immediately;
the trusted corrected probe above is the decision-bearing result.

## Crash Classification

The VS Code crash terminated both original convenience-configuration jobs.
They had valid step-50 checkpoints, finite heartbeats through steps 80 and 60,
valid target status, and zero target floors. This was an infrastructure
interruption, not evidence against the target, transport math, or NeuTra.

The jobs are not resumed because the source/default audit established that the
1,000-step, batch-256, linearly decayed `0.001` recipe was not a valid serious
training protocol. Preserving but superseding those artifacts avoids answering
the wrong scientific question.

## Evidence Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Local tests and trusted GPU/XLA preflight pass. |
| Numerical/sampler validity | Frozen score mechanics are covered locally; no new training screen or HMC result exists yet. |
| Scientific interpretation | No training recipe, transport, or sampler is promoted by Phase A. |

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Proceed to bounded GPU smokes after implementation review. | Implementation/local gate passed. | No implementation or trusted-device veto observed. | Real target-specific recipes have not yet trained. | Complete bounded implementation review, then run four five-step smokes. | No recipe nomination, HMC readiness, posterior correctness, superiority, or default readiness. |

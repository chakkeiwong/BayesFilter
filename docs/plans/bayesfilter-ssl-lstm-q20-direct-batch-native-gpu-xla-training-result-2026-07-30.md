# q=20 Direct Batch-Native GPU/XLA NeuTra Training Result

Date: 2026-07-30  
Status: `STOPPED_UNDER_BUDGETED_AND_TARGET_IDENTITY_INVALID`

## Result

The direct q=20 GPU/XLA mechanics gate passed for the target realized by that
GPU process. It does not support a claim-bearing training result. The campaign
stopped before tuning selection or final training for two independent reasons:

1. The pre-repair target identity was hardware-dependent. CPU-hidden contract
   construction issued target signature `e920ec...`, while trusted GPU
   mechanics issued `302d50...`. The frozen fixture was identical, but the 30
   synthetic observations differed at about `1e-16` because they were generated
   with TensorFlow on the selected device. This is wrong relative to the claim
   that both artifacts bind one hardware-invariant target.
2. The first `(32,32), lr=2e-4` tuning arm timed out after 3,600 seconds before
   writing its step-50 progress receipt. Four 100-update tuning arms followed by
   two plateau-controlled final streams cannot fit the remaining campaign
   budget. The campaign is under-budgeted; the arm is not scientifically
   rejected.

No tuning selection, final training stream, or HMC run was launched.

## Mechanics Evidence

| Diagnostic | Result | Role |
| --- | ---: | --- |
| Direct batch size | 100 | reviewed warm-start mechanics setting |
| XLA | compiled; HLO SHA-256 `5496caa...` | engineering correctness |
| Trainable-variable device | GPU 1 exposed as logical `/device:GPU:0` | device provenance |
| Bound status | all 100 rows hard-valid | hard veto screen |
| Active floors | 0 | hard veto screen |
| Minimum innovation eigenvalue | `0.3647075164` | hard veto screen |
| One-step loss | `67.0205582665` | explanatory only |
| Gradient norm | `63.9768519511` | explanatory only |
| Clipped gradient norm | `17.3207979520` | explanatory only |
| Frozen round trip | `8.881784197e-16` | support veto screen |
| Peak TensorFlow allocator bytes | `2,047,227,392` | resource provenance |
| Process wall time | `1,214.109 s` | budget diagnostic |

The mechanics path used TensorFlow 2.20.0 in `tfgpu`, `float64`, XLA JIT,
compiled custom principal square root, TF32 enabled as platform metadata,
verified memory growth, soft placement disabled, no scalar fallback, no
row-mapped target, and the managed-session trusted GPU basis. Its result SHA-256
is `e1bde2f0372950f34aaa8cdc98f2e664bd208ff51649c64642e9a72872ed1554`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `882679796e8ee684b6b020b7cd84e3cfc1d92d58` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-direct-batch-native-gpu-xla-training-plan-2026-07-30.md` |
| Runner | `docs/benchmarks/run_ssl_lstm_q20_direct_batch_native_gpu_xla_training_2026_07_30.py` |
| Artifact root | `docs/plans/artifacts/ssl-lstm-q20-direct-batch-native-gpu-xla-training-2026-07-30/r1/` |
| Mechanics environment | Python 3.13.13; TensorFlow 2.20.0; `tfgpu` |
| Hardware | physical GPU 1, NVIDIA GeForce RTX 4080 SUPER |
| GPU policy | `TF_FORCE_GPU_ALLOW_GROWTH=true`; repository memory-growth helper verified before logical initialization |
| Seeds | mechanics and all planned tuning/final seeds recorded in the plan; only mechanics and the incomplete tuning seed were used |
| Material cap | `18,000 s` |
| Charged material time | `4,815.309 s` |
| Remaining unused time | `13,184.691 s` |
| Data version | pre-repair process-generated q=20 synthetic fixture/observations, separately hashed by target signature |

The material charge includes a conservative 3,600-second timeout charge and a
1.2-second sandbox visibility failure. The timed-out process released GPU 1.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Stop r1 before tuning selection | Not evaluated | continuation veto: target identity mismatch and insufficient campaign budget | exact warm-update count before timeout was not persisted | preserve r1 as engineering evidence only | no candidate rejection |
| Retain mechanics as engineering evidence | Passed for GPU-realized pre-repair target | numerical/status/support screens passed | not the same byte-identical target as CPU contract | use only to estimate feasibility and prove direct GPU/XLA mechanics | no claim-bearing training pass |
| Withhold final/HMC execution | No eligible tuning artifact | required tuning and target-identity gates absent | repaired target GPU parity not checked | require fresh r2 preflight and budget review | no HMC or posterior claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | `FAIL_TARGET_IDENTITY_AND_BUDGET` for the campaign; mechanics numerical screen passed locally |
| Statistically supported ranking | none; no complete tuning comparator |
| Descriptive-only differences | mechanics loss, gradient, runtime, support, memory, and incomplete-arm timing |
| Viable candidates | not assessed; `(32,32), lr=2e-4` is incomplete, not rejected |
| Default readiness | not established |
| Next evidence needed | trusted CPU/GPU signature parity for the repaired v2 target, then a newly budgeted scope-specific training protocol |

## Negative-Result Classification

| Failure class | Verdict |
| --- | --- |
| Implementation failure | Pre-repair target identity construction was wrong relative to hardware invariance; repaired after detection |
| Tuning failure | not evaluated; no arm reached a declared validation endpoint |
| Diagnostic failure | timeout path lacked a step receipt, so optimizer progress is not claim evidence |
| Evidence against NeuTra | none |
| Evidence against this campaign design | strong: the planned ladder is under-budgeted at measured q=20 direct runtime |

## Repair State

Static fixture/observation construction is now explicitly placed on `/CPU:0`
and bound by `explicit_cpu_device_hardware_invariant_target_identity_v1` in the
v2 target payload. Focused CPU tests pass and the repaired CPU signature is
`2f7e29d32e45dc309533859c994583db94d82e90ed0a5b8318adef5b9f5f476e`.
Trusted GPU parity is `not checked`: two approval-review attempts timed out
before launching the diagnostic. The repair therefore cannot yet support r2.

## Post-Run Red Team

The strongest alternative explanation for the timeout is compile and validation
overhead rather than slow steady updates. That does not rescue this plan: the
predeclared artifact boundary was step 50, and even that boundary did not fit a
3,600-second arm cap. A runner with per-update receipts could measure the split,
but it would be a new diagnostic, not completion of r1.

The result that would overturn the target-identity veto is byte-identical CPU
and trusted-GPU v2 signature payloads under the repaired policy. The result that
would overturn the budget veto is a predeclared smaller protocol with adequate
statistical/downstream evidence and measured headroom, or a materially larger
authorized compute budget. The weakest r1 evidence is the incomplete arm: it
supports only a performance stop, not a loss or quality conclusion.

## Nonclaims

This result does not establish convergence, posterior correctness, HMC
readiness, transport promotion, architecture or optimizer ranking, production
or default readiness, predictive validity, or scientific validity. It does not
reject NeuTra or the `(32,32), lr=2e-4` candidate.

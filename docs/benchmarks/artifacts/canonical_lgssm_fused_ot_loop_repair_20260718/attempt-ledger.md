# Canonical LGSSM Fused OT Loop Repair Attempt Ledger

Campaign: `canonical-lgssm-fused-ot-loop-repair-20260718`
Plan: `docs/plans/bayesfilter-canonical-lgssm-fused-ot-loop-performance-repair-plan-2026-07-18.md`

| Attempt | Purpose | Status | Failure classification / repair | Wall time | Remaining GPU launches |
| --- | --- | --- | --- | ---: | ---: |
| preflight | trusted GPU and TensorFlow device visibility | passed | RTX 4080 SUPER; TensorFlow 2.19.1 sees GPU 0; CUDA build 12.4, `sm_89` | N/A | 6 |
| 01 | `T=2,N=128`, one-seed fused XLA compile/resource smoke | candidate failed as expected; harness valid | Performance/graph/work gates passed. `balance_steps=1` failed `TV_col=1.17846e-4` and `E_row=1.00477e-2`; proceed to marginal-only ladder | 20.47 s | 5 |
| approval timeout | Attempt 02 permission review | not launched; no budget consumed | Platform approval review timed out before process creation; no artifact/process/GPU allocation. Same command retried once unchanged | N/A | 5 |
| 02 | `T=2,N=128`, 8-seed design ladder plus disjoint 8-seed audit | passed; selected `balance_steps=2` | Design: `TV_col=5.516e-5`, `E_row=5.645e-3`; audit: `3.663e-5`, `6.900e-4`. Selection was marginal-only | 57.52 s | 4 |
| approval timeout | Attempt 03 permission reviews | not launched; no budget consumed | Two platform approval reviews timed out before process creation; exact script-scoped approval then succeeded | N/A | 4 |
| 03 | `T=2,N=1024`, 16-seed float64 all-active correctness/performance gate | passed | Hard-valid; `TV_col=1.785e-5`, `E_row=3.925e-4`; one shared solve per step; zero diagnostic solver/sweep; warm 3.890 s; peak 1.647 GB | 29.15 s | 3 |
| 04 | `T=2,N=1024`, 16-seed float32/TF32 precision/performance gate | infrastructure failure before compile | Harness used `tf.convert_to_tensor(existing_float64_tensor, float32)` instead of explicit cast. Structured artifact preserved; repair is target-neutral | 1.52 s | 2 |
| 04b | same TF32 gate after explicit harness cast | infrastructure failure before compile | Preparation owns raw-float64-to-final-dtype casting and rejected the already-float32 Tensor. Preserve raw dataset Tensor through preparation | 1.60 s | 1 |
| 05 | combined repaired T=2 TF32 gate and conditional T=10/T=50 ladder | stopped on declared T=2 veto | XLA/TF32 compiled and warm-ran in 0.892 s with correct one-solve work counts and 0.890 GB peak. `E_row=7.601e-4` passed but `TV_col=1.772e-4` failed the `1e-4` gate; T=10/T=50 were not launched | 27.55 s | 0 |

Budget: at most six trusted-GPU launches and four total GPU hours. Every launch
uses a fresh output path and the fixed 8192 MiB TensorFlow memory cap.

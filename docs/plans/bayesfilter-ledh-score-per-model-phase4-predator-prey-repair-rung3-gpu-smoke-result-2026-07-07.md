# Phase 4 Repair Rung 3 Result: Predator-Prey GPU Smoke Gate

metadata_date: 2026-07-07
status: `RUNG3_GPU_SMOKE_MIXED_BLOCKED_FULL_RUN_NOT_LAUNCHED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 4-repair-rung3

## Objective

Before any full `N=10000,T=20` predator-prey score/memory run, test the new
no-tape total-score CLI on a bounded GPU rung.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Full predator-prey score admission is still blocked; do not launch full row yet. |
| Primary criterion status | Mixed: FP64 GPU diagnostic passes tightly, but float32/TF32 coordinate FD smoke fails strict correctness tolerances. |
| Veto diagnostic status | Float32/TF32 FD mismatch vetoes using that smoke as admission evidence. |
| Main uncertainty | Need a reviewed full-row correctness policy: FP64 reference, calibrated finite-difference tolerance/step ladder, or another all-parameter correctness bridge that is not raw noisy TF32 FD. |
| Next justified action | Write a next repair subplan for full-row correctness calibration before full `N=10000,T=20` score admission. |
| What is not concluded | No predator-prey full score admission; no evidence against the FP64 mathematics; no full-row memory pass; no HMC, posterior, runtime, source-faithfulness, exact-likelihood, or scientific claim. |

## Commands And Outcomes

Initial GPU smoke invocation used the wrong abbreviated route string:

```text
--transport-gradient-mode manual_streaming_finite
```

It failed at argparse. This was an invocation defect only. The valid current
CLI choice is:

```text
manual_streaming_finite_sinkhorn_stopped_scale_keys
```

with:

```text
--transport-ad-mode full
```

selecting the total-VJP route.

The second GPU smoke exposed a dtype bug in finite-difference perturbations.
The diagnostic now configures precision before constructing theta and casts
the FD step to the active dtype.

### Float32/TF32 GPU Smoke

Command shape:

```text
T=2, N=64, batch_seeds=[81120], dtype=float32, tf32=enabled,
sinkhorn_iterations=2, fd_step=1e-3
```

Result artifact:

- `/tmp/predator-prey-score-gpu-smoke.json`

Result:

- `score_admission_status = blocked_score_not_run`;
- `score_correctness.status = fail`;
- `max_abs_error = 0.2005905956029892`;
- `max_rel_error = 0.9131697416305542`;
- peak memory: `16.5986328125 MiB`.

Interpretation:

- This is not admission evidence.
- It likely reflects finite-difference/TF32 numerical sensitivity at this
  smoke rung, but the runbook treats it as a veto until a reviewed correctness
  policy distinguishes numerical FD noise from implementation error.

### FP64 GPU Diagnostic

Command shape:

```text
T=2, N=64, batch_seeds=[81120], dtype=float64, tf32=disabled,
sinkhorn_iterations=2, fd_step=1e-4
```

Result artifact:

- `/tmp/predator-prey-score-gpu-smoke-fp64.json`

Result:

- `score_admission_status = tiny_score_diagnostic_not_admitted`;
- `score_correctness.status = pass`;
- `max_abs_error = 3.948576079437771e-06`;
- `max_rel_error = 1.5018796503683548e-08`;
- peak memory: `33.15185546875 MiB`.

Interpretation:

- The no-tape total-score route is strongly supported at bounded FP64 GPU
  scale.
- This is still tiny diagnostic evidence only and does not admit the full
  `N=10000,T=20` row.

## Local Checks

Focused contract tests:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py -q
```

Result:

```text
9 passed, 2 warnings
```

Combined Phase 4 checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
29 passed, 2 warnings
```

Tiny CLI smoke:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  --source-value-artifact docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json \
  --output /tmp/predator-prey-score-tiny.json \
  --markdown-output /tmp/predator-prey-score-tiny.md \
  --batch-seeds 81120 --time-steps 1 --num-particles 2 \
  --sinkhorn-iterations 1 --row-chunk-size 2 --col-chunk-size 2 \
  --particle-chunk-size 2 --dtype float64 --tf32-mode disabled
```

Result:

- diagnostic-only pass;
- `max_abs_error = 1.4451367746914912e-06`;
- `max_rel_error = 7.776625608382123e-09`.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Is the predator-prey no-tape score route ready for full `N=10000,T=20` score admission? |
| Answer | Not yet. Bounded FP64 evidence is strong, but float32/TF32 FD correctness failed and no full-row memory/correctness artifact exists. |
| Baseline/comparator | Same-scalar coordinate finite differences at tiny CPU, GPU float32/TF32 smoke, and GPU FP64 smoke. |
| Primary criterion | Failed/not met for full admission. |
| Veto diagnostics | Float32/TF32 FD mismatch and absent full-row memory/correctness artifact. |
| Artifact | This result plus `/tmp` smoke artifacts. |

## Next Repair Subplan Needed

Draft a Phase 4 full-row correctness calibration subplan that decides, before
launching the full row:

- whether correctness should be checked in FP64 reference mode, FP32/TF32 with
  a calibrated step/tolerance ladder, or another reviewed bridge;
- how to avoid using noisy FD as a false veto or false pass;
- whether full-row admission should be `float32/tf32` production memory plus
  FP64 bounded correctness, or must include full-row all-coordinate FD;
- what artifact fields separate production runtime evidence from correctness
  evidence.

## Nonclaims

- Predator-prey score is not admitted.
- FP64 tiny correctness is not full-row score admission.
- Float32/TF32 failure is not evidence against the mathematics by itself.
- No exact nonlinear likelihood correctness, Zhao-Cui source-faithfulness, HMC
  readiness, posterior correctness, scientific superiority, runtime ranking,
  or all-algorithm comparison is claimed.

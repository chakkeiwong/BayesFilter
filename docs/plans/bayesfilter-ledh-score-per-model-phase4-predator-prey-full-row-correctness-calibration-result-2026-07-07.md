# Phase 4 Calibration Result: Predator-Prey Score

metadata_date: 2026-07-07
status: `CLOSED_BLOCKED_FULL_SCORE_NOT_ADMITTED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 4-repair-calibration

## Phase Objective

Calibrate whether the predator-prey LEDH score route can be admitted after
bounded FP64 correctness passed but float32/TF32 finite-difference diagnostics
failed.

The target score remains the no-tape total derivative of the same realized
finite-`N` LEDH scalar:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

for row:

```text
zhao_cui_predator_prey_T20
```

with physical parameter order:

```text
r, K, a, s, u, v
```

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Predator-prey score remains blocked, not admitted. |
| Primary criterion status | Not met: no validating full `N=10000,T=20` score artifact exists, and float32/TF32 same-scalar FD diagnostics failed at bounded scale. |
| Veto diagnostic status | Full admission is vetoed by absent full-row correctness/memory evidence and failed FP32/TF32 FD diagnostics. |
| Main uncertainty | The no-tape mathematics is strongly supported in FP64 bounded mode, but a reviewed bridge from bounded FP64 correctness to production FP32/TF32 full-row admission has not been established. |
| Next justified action | Proceed to Phase 5 actual-SV score with predator-prey recorded as blocked. Do not include predator-prey in any admitted value-score integration. |
| What is not concluded | No rejection of the FP64 manual total-VJP mathematics; no HMC readiness, posterior correctness, exact nonlinear likelihood, source-faithfulness, runtime ranking, or scientific superiority claim. |

## Calibration Evidence

### FP64 Bounded Correctness

The FP64 bounded route passed the same-scalar all-coordinate finite-difference
diagnostics.

Rung 3 `T=2,N=64` FP64 diagnostic:

- artifact: `/tmp/predator-prey-score-gpu-smoke-fp64.json`;
- `score_admission_status = tiny_score_diagnostic_not_admitted`;
- `score_correctness.status = pass`;
- `max_abs_error = 3.948576079437771e-06`;
- `max_rel_error = 1.5018796503683548e-08`;
- peak memory: `33.15185546875 MiB`.

Calibration `T=5,N=256` FP64 diagnostic:

- artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-calibration-fp64-t5-n256-2026-07-07.json`;
- summary:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-calibration-fp64-t5-n256-2026-07-07.md`;
- `score_admission_status = tiny_score_diagnostic_not_admitted`;
- `score_correctness.status = pass`;
- `max_abs_error = 5.208683106161516e-05`;
- `max_rel_error = 2.5908625190534283e-07`;
- peak memory: `134.421875 MiB`.

Interpretation:

- Correct for the bounded FP64 diagnostic target.
- Unsupported as full-row admission because it is not `N=10000,T=20` and
  `memory_diagnostics.n10000_memory_pass = false`.

### Float32/TF32 Production-Lane Diagnostics

The float32/TF32 diagnostics failed the strict same-scalar FD gate.

Rung 3 `T=2,N=64` float32/TF32 diagnostic:

- artifact: `/tmp/predator-prey-score-gpu-smoke.json`;
- `score_admission_status = blocked_score_not_run`;
- `score_correctness.status = fail`;
- `max_abs_error = 0.2005905956029892`;
- `max_rel_error = 0.9131697416305542`;
- peak memory: `16.5986328125 MiB`.

Calibration `T=5,N=256` float32/TF32 diagnostic:

- artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-calibration-fp32-tf32-t5-n256-2026-07-07.json`;
- summary:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-calibration-fp32-tf32-t5-n256-2026-07-07.md`;
- `score_admission_status = blocked_score_not_run`;
- `score_correctness.status = fail`;
- `max_abs_error = 0.106048583984375`;
- `max_rel_error = 0.08375504612922668`;
- peak memory: `67.261962890625 MiB`.

Interpretation:

- Wrong relative to the predeclared strict FD correctness gate.
- The failure may be numerical FD/TF32 sensitivity, but that explanation is
  not an admission bridge.
- Runtime and bounded memory remain explanatory only.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | What correctness evidence is required before predator-prey full-row score admission after FP64 passes but FP32/TF32 FD is noisy? |
| Answer | Current evidence is insufficient for admission. FP64 bounded correctness is diagnostic; FP32/TF32 bounded correctness failed; no full validating score artifact exists. |
| Baseline/comparator | Same-scalar coordinate FD in FP64 bounded GPU mode, FP32/TF32 production-lane diagnostics, and admitted predator-prey value artifact. |
| Primary criterion | Failed/not met because no score artifact validates with `require_admitted=True`. |
| Veto diagnostics | Failed FP32/TF32 FD, absent `N=10000,T=20` correctness/memory artifact, and no reviewed bridge from bounded FP64 to production full-row admission. |
| Explanatory diagnostics | Per-coordinate FD errors, runtime, memory, dtype, TF32 mode, and device placement. |
| Artifact | This calibration result plus the FP64/FP32 JSON and Markdown diagnostics. |

## Phase 5 Handoff

Phase 5 actual-SV may start with these inherited conditions:

- LGSSM score is blocked/not admitted.
- Fixed-SIR score is blocked/not admitted.
- Predator-prey score is blocked/not admitted.
- Bounded FP64 score correctness remains diagnostic only.
- FP32/TF32 memory/runtime diagnostics remain explanatory only.
- Any future predator-prey admission requires a new reviewed bridge or a full
  validating score artifact.

Required next subplan:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-subplan-2026-07-07.md`

## Nonclaims

- Predator-prey score is not admitted.
- FP64 bounded correctness is not full-row score admission.
- FP32/TF32 failure is not evidence against the mathematics by itself.
- Runtime and memory are not correctness evidence.
- No exact nonlinear likelihood correctness, Zhao-Cui source-faithfulness, HMC
  readiness, posterior correctness, scientific superiority, runtime ranking,
  or all-algorithm comparison is claimed.

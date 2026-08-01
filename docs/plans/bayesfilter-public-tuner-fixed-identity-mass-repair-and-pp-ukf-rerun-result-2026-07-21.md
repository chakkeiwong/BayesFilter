# Fixed-Identity Tuner Repair and PP-UKF Rerun Result

Date: 2026-07-21  
Plan: `docs/plans/bayesfilter-public-tuner-fixed-identity-mass-repair-and-pp-ukf-rerun-plan-2026-07-21.md`

## Decision

`FIXED_IDENTITY_REPAIRED_PP_UKF_RERUN_RESOURCE_STOP_BEFORE_TERMINAL_TUNING`

The shared bug was repaired and focused regression coverage passed. A fresh
trusted GPU/XLA PP-UKF tuning-only attempt then entered the repaired operational
warmup, completed 100 of 1,000 Phase-4 transitions, and was stopped after a
prospective wall-time review. It did not produce a terminal tuning result.

The attempt is incomplete infrastructure/resource evidence, not a tuning pass,
candidate rejection, convergence result, or scientific PP-UKF result.

## Repair Implemented

The public `mass_policy` now propagates through the missing boundary:

`HMCKernelTuningConfig.mass_policy` -> `HMCWindowedMassStageConfig.mass_policy`
-> `_windowed_mass_stage_internal_config(..., mass_policy=...)`
-> `WindowedMassAdaptationConfig.mass_policy`.

The warmup schedule now marks slow windows as mass-update windows only for
`windowed_adaptive`. Fixed identity therefore has no covariance assessment,
metric candidate, coordinate replacement, or metric update. The operational
result validator also fails closed if a fixed-identity result contains any such
update or changes coordinate/metric signatures.

## Verification

| Check | Result |
| --- | --- |
| Windowed-mass adaptation and warmup suite | `87 passed, 1 skipped` |
| Public/outer-loop/fixed-mass tuner suites | `178 passed` |
| Python compilation | Passed |
| `git diff --check` | Passed |
| Fixed-identity schedule regression | Slow-window `update_mass=False` for every window |
| Fixed-identity operational regression | Covariance assessor not called; zero metric decisions; unchanged coordinate/metric; adaptation generation zero |

## Fresh PP-UKF Attempt

Output root:
`docs/plans/artifacts/bayesfilter-pp-ukf-fixed-identity-repair-rerun-20260721-01/`

The attempt used the existing frozen transport without retraining:

- frozen transport SHA-256: `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221`;
- target signature: `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5`;
- target dimension: 6;
- TensorFlow/TFP GPU/XLA route with memory growth;
- `--tuning-only`, so no sequential HMC sampling launched;
- fresh seed offset `0` and fresh output root.

Bootstrap completed with finite execution but exhausted its acceptance-repair
budget. The repaired Phase-4 operational warmup then ran on identity mass. Its
progress artifact records 100/1,000 transitions across 2/19 segments.

The private event log contains only the initial identity mass event. There is no
post-bootstrap covariance, mass replacement, or coordinate-change event. This
is runtime evidence that the fixed-identity repair held during the executed
portion.

The process was stopped with `KeyboardInterrupt` during retained-target health
evaluation after approximately 47 minutes. No `hmc_kernel_tuning_result.json`,
run manifest, or terminal public result was written. The stale `run_state.json`
is explicitly repaired below to classify the attempt as interrupted.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Repair shared fixed-identity tuner | Focused tests and runtime guard passed | No repair veto | Other callers may have historical artifacts from the old route | Reclassify old artifacts as historical and use repaired source for new runs | No sampler or scientific claim |
| Admit PP-UKF tuning kernel | No terminal tuning artifact | `RESOURCE_PROJECTION_STOP_BEFORE_TERMINAL_TUNING` | Whether a faster retained-target health path or larger explicit cap is appropriate | Add prospective wall-time/rate preflight; then review a new bounded continuation | No candidate pass/fail, convergence, posterior correctness, or default readiness |
| Launch sequential HMC | No admitted tuning kernel | Correctly closed | None | Do not launch until terminal tuning verification exists | No posterior result |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Engineering fixed-identity guard passed; PP-UKF tuning screen unevaluated to terminal completion |
| Statistically supported ranking | None |
| Descriptive-only differences | Bootstrap acceptance repairs, elapsed time, transition count, and memory |
| Default readiness | Not assessed |
| Next evidence needed | Prospective rate/wall-time projection, a bounded continuation, terminal fixed-identity tuning verification, then independent retained-chain diagnostics |

## Research Question Guardian

| Question | Verdict |
| --- | --- |
| Harness invalidated? | No. Focused tests, GPU/XLA initialization, frozen transport binding, and 100 operational transitions completed. |
| Implementation invalidated? | No. The fixed-identity mass invariant held for the executed transitions. |
| Target or mathematics invalidated? | No. The stop occurred during expensive retained-target health evaluation, not a numerical target veto. |
| Candidate failed? | No candidate reached terminal fixed-mass screening. |
| Research direction rejected? | No. The result triggers a resource/observability redesign only. |

## Post-Run Red Team

The strongest alternative explanation is that retained-target health evaluation,
rather than HMC transition execution, dominates the wall time. The traceback
supports that explanation: interruption occurred in `_evaluate_retained_target_health`
while replaying the PP-UKF value/score path. A faster batched health evaluator or
an explicit wall-time preflight could overturn the resource stop. The weakest
part of the current evidence is that only 100 transitions completed and the
process was intentionally interrupted before terminal serialization.

## Required Next Change Before Continuation

Do not rerun this exact serious budget blindly. Add a prospective wall-time/rate
preflight that measures the actual repaired source after warmup and projects the
complete Phase-4 plus downstream tuning cost against an explicit cap. A
continuation then needs a fresh output root and either a materially cheaper
validated retained-target health path or an explicit owner-approved larger
compute cap.

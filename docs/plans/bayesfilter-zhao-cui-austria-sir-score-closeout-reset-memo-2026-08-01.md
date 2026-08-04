# Zhao-Cui Austria SIR Score Closeout Reset Memo

Date: 2026-08-01T22:32:10+08:00

Status: `FRESH_T1_REPAIR_PLAN_REQUIRED`

## Read First

1. `AGENTS.md` supplied in the active session.
2. `docs/plans/bayesfilter-zhao-cui-austria-sir-score-closeout-plan-2026-08-01.md`.
3. `docs/plans/bayesfilter-zhao-cui-austria-sir-score-closeout-result-2026-08-01.md`.
4. `bayesfilter/highdim/zhao_cui_austria_sir_lane_b_training_jvp_tf.py`.
5. `tests/highdim/test_zhao_cui_austria_sir_lane_b_training_jvp_tf.py`.

The old closeout plan is stopped. Do not issue a third T1 launch or run T2
under it.

## Current Verdict

Trusted GPU access works. The RTX 4080 SUPER is visible, TensorFlow creates one
CUDA logical device with the reviewed 6,144 MiB hard cap, and the production
cluster compiles with XLA.

The current T1 issuer candidate is nevertheless rejected. Its final budgeted
launch produced maximum parent-core residual `2.73558953e-13` against the
required exact threshold `0`. No T1 score artifact exists. T2 remains unrun
and correctly closed.

This is candidate rejection, not research-direction rejection.

## Attempts Preserved

- `docs/plans/artifacts/zhao-cui-austria-sir-score-closeout-20260801/t1-training-jvp-01`
  exists and is empty. It failed during XLA tracing because an eager model
  validator called `.numpy()` on a symbolic tensor.
- `docs/plans/artifacts/zhao-cui-austria-sir-score-closeout-20260801/t1-training-jvp-02`
  exists and is empty. The repaired CUDA/XLA route reached and failed the exact
  replay gate at residual `2.73558953e-13`.
- `docs/plans/artifacts/zhao-cui-austria-sir-score-closeout-20260801/t2-training-jvp-01`
  is absent.

Do not remove, overwrite, or reuse either T1 directory.

## Implemented Repair

The T1 module now has a claim-local graph-native FP64 row target using the
existing Austria CUDA/XLA transition and isotropic-Gaussian primitives from
`bayesfilter/highdim/models.py`. That dependency is bound into the strict T1
issuer source closure.

Focused evidence passed:

- graph-native/eager row-value parity;
- all-three-coordinate graph-native/analytical-score parity;
- real XLA reverse-gradient plus outer forward-JVP execution;
- strict T1/T2 issuer and tamper checks; and
- fixed hard-cap policy tests.

Final repaired suite: `17 passed, 2 warnings in 77.70s`.

These tests establish engineering viability. They do not override the failed
exact parent replay.

## Leading Repair Hypothesis

`prepare_t1_replay_inputs` materializes the zero-parameter graph-native log
target outside the CUDA/XLA optimizer graph. `_training_loss` evaluates the
active graph-native log target inside CUDA/XLA and subtracts the materialized
origin value. At `theta=0` the expressions are mathematically equal, but the
two execution contexts can differ in last-bit arithmetic. That can perturb
normalized weights and produce the observed small nonzero core residual.

A fresh plan may test computing both terms inside the same compiled graph:

```text
active_log = graph_value(theta, points, y1)
origin_log = graph_value(zeros_like(theta), points, y1)
log_ratio = active_log - origin_log
```

This is an untested repair hypothesis. Before any full replay, the new plan
should require the smallest CUDA/XLA diagnostic showing that the origin
`log_ratio` is elementwise bitwise zero and that a bounded one-step/full-primal
replay follows the admitted operation order. If that diagnostic fails, do not
launch another issuer.

## Immutable Gates

- Parent-core residual must remain exactly `0`.
- Do not replace or regenerate the admitted clouds under a different backend.
- Do not change optimizer order, seeds, parent identities, shifts, `tau`,
  ranks, bases, schedules, or prepared inputs.
- Do not use CPU replay as claim evidence.
- Do not run T2 before a passed T1 issuer and strict T1 reload.
- Do not run HMC.
- The assembled parameter-score route remains `extension_or_invention`, not
  source-faithful.

## Fresh Plan Requirements

A new plan needs a new versioned output root and new explicit attempt/time
budget. Its evidence contract must retain exact parent replay, manual/JVP/FD
parity, strict loader admission, trusted GPU/XLA, and the 6,144 MiB hard cap.
It should treat same-graph origin evaluation as a repair hypothesis, not as a
new default or established solution.

Only after exact T1 admission may T2 be reconsidered. Later horizons and HMC
remain closed.

## Worktree Warning

The repository is heavily dirty with unrelated user work. All active score
files are untracked. Do not run destructive Git cleanup or broad restore
commands. Current underlying commit:
`fb9a0679adb7c731ff2ac42551f39bdcc15222a1`.

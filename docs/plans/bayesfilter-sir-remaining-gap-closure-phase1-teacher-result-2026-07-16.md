# SIR Remaining-Gap Closure Phase 1 Result

Date: 2026-07-16

Plan: `docs/plans/bayesfilter-sir-remaining-gap-closure-master-plan-2026-07-16.md`

Status: `PASS_TEACHER_IMPLEMENTATION_FOCUSED_CHECKS`

## Result

The independent TensorFlow float64 online `O(N^2)` SIR filtering-score teacher
is implemented in
`bayesfilter/highdim/sir_online_score_teacher_tf.py`. It uses a bootstrap
particle filter, stateless systematic resampling, normalized all-pairs backward
kernels, explicit initial/transition/observation complete-data score terms,
progressive score marks, and `tf.while_loop` for horizon and RK4 recursion. It
does not call LEDH or Contract E code.

The first test run failed because `static_spec_from_model` passed the bound
method `base._rk4_step` to `tf.convert_to_tensor`. The repaired implementation
uses the model's actual internal step,
`base.delta / tf.cast(base._rk4_substeps, tf.float64)`, matching the existing
latent-SIR Contract E adapter.

## Checks

Command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-sir-gap \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q \
tests/highdim/test_sir_online_score_teacher_tf.py
```

Latest result after the explicit initial-score check and numeric-only compiled
return repair: `5 passed, 2 warnings in 10.95s`.

The checks establish:

- local initial, transition, and observation scores match TensorFlow autodiff;
- fixed seeds reproduce identical value and score outputs;
- backward-kernel rows are finite and normalized;
- the one-observation score reduces to normalized initial local-score averaging;
- previous score marks and local transition scores are both required at two
  observations; and
- the factory defaults to XLA JIT while the algorithm has no Python
  horizon-scaling or particle-scaling loop.

A deliberate CPU-only XLA smoke at `T=2,N=8,R=1` logged
`Compiled cluster using XLA!` and returned finite value and score. This is graph
compatibility evidence only, not GPU or target-accuracy evidence.

## Evidence Boundary

This phase establishes implementation wiring and local mathematical terms. It
does not establish target-level teacher accuracy, unbiasedness, convergence,
LEDH agreement, GPU/XLA execution, canonical identity, HMC readiness, or
leaderboard readiness. Phase 2 performs the independent `J=1` dense-reference
mismatch screen under the amended observation-count convention.

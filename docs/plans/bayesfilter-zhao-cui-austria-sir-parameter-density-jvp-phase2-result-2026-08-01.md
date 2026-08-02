# Zhao-Cui Austria SIR Parameter Density/JVP Phase 2 Result

Date: 2026-08-01

Status: `PASS_PHASE2_MECHANICS_AFTER_FROZEN_BASIS_REPAIR_ONLY`

Parent plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-parameter-conditioned-density-jvp-plan-2026-07-31.md`.

## Outcome

The T1 parameterized target, analytical complete-data score, absolute-scale
I-divergence, auxiliary origin point-score loss, fixed-rank residual
initialization, heldout diagnostics, independent likelihood/prefix score
estimators, and GPU/XLA optimizer mechanics are implemented and pass focused
checks.

This does not establish a target-trained Austria likelihood score. No serious
T1 candidate has been trained or selected, no untouched stream has been
consumed, and T2/HMC remain closed.

## Frozen-Basis Backend Repair

The first campaign pilot exposed an implementation defect, not evidence about
the centered-density candidate.  The local interpolation nodes were generated
inside the evaluation path with `tf.linalg.eigvalsh`.  CPU and GPU then used
slightly different nodes: a one-point basis discrepancy of about `1.7e-4`
accumulated across 36 axes, produced about a two-percent parent contraction
difference, and could make the GPU prefix contraction non-finite.  The
resulting live-GPU prefix score near `-5.1e10` was spurious; CPU reload of the
same pre-repair child gave about `0.00164`.

The repaired route freezes local nodes and barycentric weights on CPU before
TensorFlow tracing and evaluates only those fixed FP64 constants on GPU/XLA.
Its policy identifier is
`setup_static_cpu_nodes_barycentric_weights_v1`.  The policy identifier and
node/weight hashes are now part of the child identity.

Post-repair reconstructed-child parity is:

- CPU versus GPU/XLA value maximum difference: `1.78e-15`;
- CPU versus GPU/XLA score maximum difference: `3.04e-18`;
- CPU versus GPU basis-value difference: exactly zero; and
- trusted GPU allocator peak: `65,591,040` bytes.

Every centered child and capacity artifact created before this repair is stale
and ineligible.  In particular, the old Stage A `a01` artifact, its step-eight
diagnostic, and the old capacity result must not be upgraded or selected.  A
fresh capacity result and fresh child identities are required.

## Claimed And Computed Objects

The implemented absolute objective is

\[
  J(\theta)=Z_{\rm child}(\theta)
  -E_{p_\theta(z_0)f_\theta(z_1\mid z_0)}
    [e^{c_1}g_\theta(y_1\mid z_1)\log\rho_{\rm child}(\theta,z_1,z_0)].
\]

The importance weight is unnormalized. The origin auxiliary loss compares the
normalized point score of the same finite child with the analytical
complete-data score centered by an independent likelihood-weighted Fisher
normalizer estimate. The derivative loss is disabled only when its configured
weight is exactly zero; a nonzero weight without all explicit origin arrays
fails closed.

The first theta row is required to be exactly zero. This prevents a caller from
silently applying the derivative target to a non-origin slice.

## Implemented Surface

- `bayesfilter/highdim/zhao_cui_austria_sir_parameter_density_training_tf.py`
- `tests/highdim/test_zhao_cui_austria_sir_parameter_density_training_tf.py`

The code remains TensorFlow-only and batch-native across theta and proposal
samples. It does not use `map_fn`, `vectorized_map`, a theta-state retained
grid, a block-sum TT, or full time history. The parent is immutable.

## Verification

Intentional CPU-only reference command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q \
tests/highdim/test_zhao_cui_austria_sir_parameter_density_training_tf.py \
-k 'not train_step and not compiles and not compiled'
```

Result: `10 passed, 3 deselected` in `38.56 s`.

The checks include:

- batch-native target/value and analytical score parity with the scalar model;
- maximum earlier parity residuals of `1.14e-13` for value and `2.84e-14`
  for score;
- exact absolute-weight identity and physical/local/reference coordinate ledger;
- known continuous target-family objective ordering;
- actual optimization reducing the known-family objective by more than 50%;
- fail-closed derivative inputs and origin ordering;
- deterministic, distinct fixed-rank residual initialization;
- coordinatewise origin point-score metrics;
- heldout absolute log-mass uncertainty, normalized density RMS, and ESS; and
- finite independent likelihood and retained-prefix score estimates with MCSE.

Trusted GPU prerequisites:

- NVIDIA GeForce RTX 4080 SUPER, driver `591.86`;
- TensorFlow physical device `/physical_device:GPU:0`;
- `TF_FORCE_GPU_ALLOW_GROWTH=true` before import; and
- verified memory growth `True`.

Trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q \
tests/highdim/test_zhao_cui_austria_sir_parameter_density_training_tf.py \
-k 'train_step or compiles or compiled'
```

Result: `3 passed, 10 deselected` in `148.18 s`.

The compiled checks cover the batch-native target, a zero-derivative-weight
optimizer update, and a nonzero derivative-weight optimizer update consuming
the real analytical origin score arrays.

## Repair Ledger

| Attempt | Classification | Finding | Repair |
|---|---|---|---|
| Unfinished compiled derivative path | Implementation defect | The compiled step supplied zero scores and hardcoded derivative weight zero. | Made all origin arrays explicit inputs and used the configured weight. |
| Synthetic measure check | Test validity defect | Points were sampled from `[-0.98,0.98]` while the exact objective used the uniform `[-1,1]` reference law. | Sampled exactly from `Uniform[-1,1]`; the upper endpoint is excluded by stateless uniform. |
| Derivative convenience API | Silent-assumption defect | The first theta row was assumed to be the origin. | Added an exact fail-closed origin-row assertion. |
| Missing derivative arrays | Silent-disable defect | A nonzero derivative weight could produce a zero derivative loss when arrays were absent. | Nonzero weight now requires all three explicit origin arrays. |
| Rank ladder | Mislabeled-default defect | Default residuals copied the rank-4 parent, so a future rank ladder would not vary rank. | Added deterministic fixed-rank residual initialization with explicit rank and seed. |
| Synthetic optimizer, LR `0.02` | Tuning failure | Full-core Adam left the known optimum and stalled. | A bounded learning-rate diagnostic found `3e-4` reduced the fixed known-family loss from about `0.0400` to `0.00253` in 20 steps. The test claims only objective progress, not global parameter recovery. |
| Exact origin assertion | Test/API semantic defect | TensorFlow `assert_near` with zero tolerances rejects equality because it uses a strict inequality. | Replaced it with `assert_equal`. |
| Runtime interpolation eigenproblem | Backend implementation defect | CPU and GPU generated different interpolation nodes, corrupting 36-axis contractions and GPU prefix scores. | Freeze setup-static CPU nodes and barycentric weights, bind their hashes into identity, and reject every pre-repair child artifact. |

## Decision Table

| Field | Decision |
|---|---|
| Decision | Admit Phase 2 mechanics and open the separately audited T1 training campaign. |
| Primary criterion | Passed for mechanics only; no scientific score criterion has run. |
| Veto diagnostics | Target parity, measure identity, derivative enablement, rank identity, finite gradients, origin equality, and XLA passed. |
| Main uncertainty | Whether a low-rank centered residual family trained off-origin can pass untouched likelihood and retained-prefix score gates. |
| Next justified action | Replace the unjustified near-constant residual initializer with a training-only target-informed score prefit, rerun the corrected capacity probe, then refresh the T1 ladder. |
| Not concluded | No target-trained T1 score, T2/T20 score, exact nonlinear likelihood, HMC readiness, source-faithful assembled score route, superiority, or production readiness. |

## Ledgers And Red Team

| Ledger | Status |
|---|---|
| Engineering correctness | Phase 2 focused CPU and GPU/XLA checks pass. |
| Numerical validity | Target and analytical score tie out to scalar authority; known-family objective and optimizer respond correctly under the tested bounded arm. |
| Scientific interpretation | Unsupported beyond mechanics because no disjoint validation or untouched claim has run. |

The strongest alternative explanation is that the synthetic target is much
easier than the Austria target and starts close to its known optimum. That is
why it is only a mechanics gate. A successful real training command could
still be misleading if importance ESS collapses, only normalized shape fits,
the origin score overfits the auxiliary loss, prefix scores fail, or XLA memory
exceeds the cap. The Phase 3 evidence contract tests those risks separately.

The repaired mechanics also expose an unresolved scientific default.  The
campaign initializer used first-core amplitude `1e-3`; training-only audits
showed an origin child score near zero and normalized point-score RMS near one,
even when the raw amplitude scale was increased.  The independent training
Fisher score was approximately `[-5.7843, 2.9614, -4.9495]`, so the initializer
does not seed the required score scale or shape.  The old Stage A ladder is
therefore paused.  Its output would answer optimizer sensitivity around an
uninformative initializer, not whether the finite child can represent the
required value and score.

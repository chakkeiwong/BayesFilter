# Zhao-Cui Austria SIR Parameter-Conditioned Density/JVP Phase 1 Result

Date: 2026-07-31

Status: `PASS_PHASE1_CENTERED_DENSITY_MECHANICS_ONLY`

Plan:
`docs/plans/bayesfilter-zhao-cui-austria-sir-parameter-conditioned-density-jvp-plan-2026-07-31.md`.

## Outcome

The centered residual TT mechanics are implemented and pass focused CPU and
trusted GPU/XLA checks. The admitted T1 and T2 parents are embedded exactly at
theta zero; manual increment, point-density, and retained-prefix scores match
diagnostic autodiff for the implemented finite child; serialization rejects
tensor and identity-manifest tampering; and the storage estimator is linear in
the number of independent residual components.

This is an engineering mechanics result. No residual TT has been trained on the
parameterized Austria target. Therefore there is still no admitted Austria
Zhao-Cui likelihood score.

## Implemented Surface

- `bayesfilter/highdim/zhao_cui_austria_sir_centered_density_tf.py`
- `tests/highdim/test_zhao_cui_austria_sir_centered_density_tf.py`

The implementation provides:

1. centered linear, pure-quadratic, and interaction parameter features;
2. immutable admitted parent plus independent residual state TTs;
3. point amplitude, defensive density, and analytical parameter score;
4. exact cross-component Gram normalizer and score;
5. retained-prefix cross contractions and quotient score;
6. repository-issued identity, save/reload, and tamper rejection; and
7. explicit stored, point-workspace, normalizer-pair, and prefix-pair memory
   estimates.

It does not materialize a theta-state tensor grid or a block-rank sum TT.

## Verification

Intentional CPU-only reference command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q \
tests/highdim/test_zhao_cui_austria_sir_centered_density_tf.py -k 'not xla'
```

Result: `8 passed, 1 deselected`.

Trusted GPU prerequisites:

- `nvidia-smi`: NVIDIA GeForce RTX 4080, driver `591.86`, CUDA reported `13.1`;
- TensorFlow physical device: `/physical_device:GPU:0`;
- verified memory growth: `True`;
- `TF_FORCE_GPU_ALLOW_GROWTH=true` set before TensorFlow import.

Trusted GPU/XLA command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q \
tests/highdim/test_zhao_cui_austria_sir_centered_density_tf.py -k xla
```

Result: `1 passed, 7 deselected` before the identity-manifest-only CPU test was
added. The compiled test covers increment, point-density, and retained-prefix
value/score parity on the GPU XLA path.

Adjacent historical/parent regression command:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp/matplotlib-cache \
/home/chakwong/anaconda3/envs/tf-gpu/bin/python -m pytest -q \
tests/highdim/test_zhao_cui_austria_sir_parameter_child_tf.py \
tests/highdim/test_zhao_cui_austria_sir_lane_b_tf.py
```

Result: passed. The command output was compact (`.....`) and returned exit code
zero.

Syntax compilation and `git diff --check` also passed.

## Repair Ledger

| Attempt | Classification | Finding | Repair |
|---|---|---|---|
| CPU reference 1 | Test tolerance only | Feature Jacobian differed from diagnostic autodiff at floating-point operation-order scale under an exact-bit assertion. | Replaced the exact-bit diagnostic assertion with machine-scale FP64 tolerance. No algorithm changed. |
| GPU/XLA 1 | Graph-construction implementation defect | Product-basis construction occurred inside `tf.function`; the legacy constructor calls `.numpy()` and cannot trace. | Froze basis, settings, and shift in child initialization. Compiled kernels now contain tensor operations only. |
| Terminal resource review | Documentation/estimator defect | The initial plan stated `O(R^4)` pair workspace but omitted the prefix batch factor. | Corrected prefix-pair workspace to `O(BR^4)` and added a separate conservative reverse-mode training-memory bound. |
| Terminal authority review | Evidence-contract gap | T1 retained-prefix score authority was required but not explicitly constructed. | Added the analytical conditional-ratio definition and ESS/MCSE validity requirement to the plan. |

## Decision Table

| Field | Decision |
|---|---|
| Decision | Admit Phase 1 mechanics and proceed to Phase 2 target/absolute-loss harness only. |
| Primary criterion | Passed for mechanics: exact origin values plus manual/diagnostic derivative tie-outs. |
| Veto status | No target, identity, graph, XLA, or mechanics veto remains in Phase 1. |
| Main uncertainty | Whether a target-specific residual family can learn the nonzero-theta absolute density and retained-prefix score in 36 dimensions. |
| Next justified action | Implement batch-native parameterized T1 target/proposal generation, absolute I-divergence, independent prefix-score diagnostic, and a synthetic recovery test. |
| Not concluded | No correct target-trained T1 score, T2/T20 score, exact likelihood, source-faithful assembled route, HMC readiness, superiority, or production readiness. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Status |
|---|---|
| Engineering correctness | Phase 1 focused tests pass on CPU and GPU/XLA. |
| Numerical validity | Manual derivatives agree with diagnostic autodiff for synthetic residual components; origin parent values and marginals agree within FP64 contraction tolerances. |
| Scientific interpretation | Unsupported beyond mechanics because no parameterized target fit or untouched score claim has run. |

## Post-Run Red Team

The strongest alternative explanation for this success is that the tests use
residual components proportional to the parent, a deliberately easy algebraic
case. That is sufficient to test centering, cross contractions, quotient rules,
identity, and XLA mechanics, but it provides no evidence that independent
low-rank residual TTs can represent the Austria off-origin density.

The result would be overturned by an independent mechanics test showing an
incorrect cross-component contraction, failed origin equality, or backend
mismatch. The weakest current evidence is training feasibility: Phase 1 did not
measure reverse-mode activation/optimizer memory and cannot support a serious
training launch.


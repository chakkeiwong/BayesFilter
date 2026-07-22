# Phase 1 Result: Trainable Dense-IAF And Checkpoint Controller

Date: 2026-07-13  
Status: `PASS_PHASE1_TRAINING_CONTROLLER`  
Plan: `docs/plans/bayesfilter-lgssm-neutra-knowledge-transfer-and-serious-validation-plan-2026-07-13.md`

## Outcome

BayesFilter now has a focused TensorFlow implementation of the mature plain
NeuTra pattern:

```text
T_phi(z) = fixed_full_affine(IAF_2(P(IAF_1(P(IAF_0(z))))))
```

where `P` is the fixed reverse permutation. Only dense-IAF parameters are
trained. The objective is exact-target reverse KL through the reviewed
value/score custom-gradient bridge.

The controller provides stateless per-step base draws, XLA-compiled training,
manual Adam state, linear learning-rate decay, global-norm clipping, immutable
step checkpoints, a latest checkpoint pointer, append-only progress events,
resume validation, no-overwrite frozen payloads, and direct emission into the
existing dense-IAF frozen schema.

## Evidence

| Check | Result |
| --- | --- |
| Frozen reload forward parity | Passed to `1e-12` |
| Frozen reload logdet parity | Passed to `1e-12` |
| Uninterrupted versus resumed trainable state | Bitwise equal |
| Uninterrupted versus resumed Adam moments | Bitwise equal |
| Resume config mismatch | Rejected |
| No-overwrite checkpoint/freeze | Rejected duplicate write |
| Reverse-KL gradient | Matched central finite difference |
| Focused Phase 0/1/loader suite | `17 passed` |

## Historical Knowledge-Transfer Parity

A matched-weight check against
`/home/chakwong/python/src/dsge_hmc/estimation/_transports.py` used three dense
IAF stages, two reverse permutations, the same full affine map, and seven
deterministic input rows.

| Quantity | Maximum absolute difference |
| --- | ---: |
| Forward transport | `0.0` |
| Log absolute determinant | `3.000266701747023e-12` |

The logdet difference is a checked constant: the historical XLA mixing layer
computes a Cholesky logdet after adding a `1e-12` Gram-matrix nugget, while the
reviewed frozen schema uses exact `slogdet(matrix)`. With two fixed permutation
matrices the difference is approximately `3e-12`, has zero parameter/position
gradient, and cannot change reverse-KL gradients or HMC forces. The
implementations are not claimed bitwise equal in absolute log density.

The new implementation restricts `s_max` to `1.0`. At this value the historical
and frozen-schema bounded-scale formulas coincide. No broader `s_max`
equivalence is claimed.

## Files

- `bayesfilter/inference/neutra_training.py`
- `tests/test_neutra_training.py`

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Proceed to frozen score and modern tuning gates | Composition, objective gradient, checkpoint/resume, no-overwrite, and freeze/reload checks pass | No Phase 1 veto fired | Exact 18D GPU runtime not yet checked; frozen dense-IAF lacks the explicit HMC score methods before Phase 2 | Implement and verify manual frozen score pullbacks and serious modern tuning mode | Transport quality, posterior convergence, HMC readiness, or scientific superiority |


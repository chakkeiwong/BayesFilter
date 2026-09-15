"""Parity check: is the primal canonical value independent of K directions?

The dual-parameter HMC target computes the exact value with K=1 direction
instead of K=P, on the grounds that the fused engine asserts the primal value
is direction-invariant. That is a correctness claim about the value consumed by
the HMC acceptance ratio, so it is checked executably here rather than assumed.

Compares canonical_batch_fused_value_score value output at K=1 vs K=P on the
same frozen fixture. Diagnostic only.
"""

import os
import sys

import tensorflow as tf

_GPUS = tf.config.list_physical_devices("GPU")
for _g in _GPUS:
    tf.config.experimental.set_memory_growth(_g, True)
    if not tf.config.experimental.get_memory_growth(_g):
        raise RuntimeError(f"set_memory_growth failed on {_g.name}")

from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    canonical_batch_fused_value_score,
)
from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
    _diagonal_lgssm_fused_model,
    _lgssm_frozen_observations,
)


def main():
    observations = _lgssm_frozen_observations()
    model = _diagonal_lgssm_fused_model()

    d = 3
    N = int(os.environ.get("PARITY_N", 24))
    T = int(os.environ.get("PARITY_T", 5))  # slice: claim is T-independent
    substeps = int(os.environ.get("PARITY_SUBSTEPS", 2))
    sinkhorn = int(os.environ.get("PARITY_SINKHORN", 2))
    dtype = observations.dtype

    observations = observations[:T]

    generator = tf.random.Generator.from_seed(81100)
    initial_states = generator.normal([N, d], dtype=dtype) * 0.1
    initial_covariances = (
        tf.tile(tf.eye(d, dtype=dtype)[None, :, :], [N, 1, 1]) * 0.01
    )
    noises = generator.normal([T, N, d], dtype=dtype) * 0.1

    reset_basis = tf.concat(
        [tf.eye(d, dtype=dtype), -tf.eye(d, dtype=dtype)], axis=0
    )
    reset_repeats = (N + 2 * d - 1) // (2 * d)
    reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:N]

    params = dict(
        substeps=substeps,
        reset_policy="contract_e",
        reset_design=reset_design,
        reset_epsilon=2.0,
        reset_sinkhorn_steps=sinkhorn,
        reset_balance_steps=sinkhorn,
        reset_ridge=1e-5,
        correction_steps=4,
        correction_strength=0.2,
        correction_lm_damping=1e-2,
        correction_lm_scale_floor=1e-4,
        correction_trust_radius=0.5,
        pairwise_steps=4,
        pairwise_strength=0.02,
        pairwise_rms_cap=2.0,
        coordinate_cap=0.0,
        annealed_stages=1,
        annealed_seed=0,
    )

    theta = tf.constant([[1.0, 1.0, 1.0, 0.5, 0.3]], dtype=dtype)  # [1, P]
    P = int(theta.shape[1])
    full_directions = tf.eye(P, dtype=dtype)[None, :, :]  # [1, P, P]
    one_direction = full_directions[:, :1, :]  # [1, 1, P]

    print("Value direction-invariance parity check (diagnostic only)")
    print(f"  gpus={[g.name for g in _GPUS]} "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}")
    print(f"  N={N} T={T} d={d} P={P} substeps={substeps} sinkhorn={sinkhorn}")
    print(f"  K=P costs P={P} filter passes; K=1 costs 1\n")

    print("  running K=P ...")
    value_full, score_full, _ = canonical_batch_fused_value_score(
        model, theta, full_directions, initial_states, initial_covariances,
        noises, observations, **params,
    )
    print("  running K=1 ...")
    value_one, score_one, _ = canonical_batch_fused_value_score(
        model, theta, one_direction, initial_states, initial_covariances,
        noises, observations, **params,
    )

    vf = float(value_full[0])
    vo = float(value_one[0])
    abs_diff = abs(vf - vo)
    rel_diff = abs_diff / max(abs(vf), 1e-300)

    print(f"\n  value @ K=P : {vf!r}")
    print(f"  value @ K=1 : {vo!r}")
    print(f"  abs diff    : {abs_diff:.3e}")
    print(f"  rel diff    : {rel_diff:.3e}")

    # The K=1 score must equal the first coordinate of the K=P score, since
    # both are the directional derivative along e_0.
    s_full_0 = float(tf.reshape(score_full, [-1])[0])
    s_one_0 = float(tf.reshape(score_one, [-1])[0])
    score_abs_diff = abs(s_full_0 - s_one_0)
    print(f"\n  score[0] @ K=P : {s_full_0!r}")
    print(f"  score[0] @ K=1 : {s_one_0!r}")
    print(f"  abs diff       : {score_abs_diff:.3e}")

    # Bitwise-identical is the expected outcome: same code path, same inputs,
    # same order of operations. Accept a tight relative tolerance.
    tol = 1e-12
    ok_value = rel_diff <= tol
    ok_score = score_abs_diff <= tol * max(abs(s_full_0), 1.0)

    print(f"\n  value parity (rel <= {tol:g}): {'PASS' if ok_value else 'FAIL'}")
    print(f"  score[0] parity            : {'PASS' if ok_score else 'FAIL'}")
    print(f"  bitwise identical value   : {vf == vo}")

    if not (ok_value and ok_score):
        print("\n  RESULT: FAIL - K=1 substitution is NOT value-preserving.")
        return 1
    print("\n  RESULT: PASS - K=1 returns the same value; substitution is safe.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

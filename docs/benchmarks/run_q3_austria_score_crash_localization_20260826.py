"""Localize the Austria production-score crash: S6 reset vs S7 dual-cap.

The production Austria score cell crashed on 6/8 seeds with a GPU
`SelfAdjointEigV2` info=1 (non-converged eigendecomposition). The score
lane reaches an eigendecomposition only through S7's dual-cap spectral
machinery (`higher_moment_shape_jvp`); S6's own factorizations are
Cholesky. This run re-executes the crashing seeds with S7 disabled to
attribute the crash, and reports the S6 gap-matrix conditioning.

Classification: repair trigger, debugging-only. No score value from this
run enters a leaderboard cell. Fixtures replicate `score_cells` exactly
(rng 9000+seed, flow_substeps=8, annealed_stages=4, annealed_seed=17).
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "docs", "benchmarks"))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

DTYPE = tf.float64
CRASH_SEEDS = [1, 2, 3, 5, 6, 7]
N, DIM = 1008, 18


def main() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    started = time.time()

    from run_q3_leaderboard_20260824 import production_score_kwargs
    from bayesfilter.highdim.ledh_canonical_models_tf import (
        austria_sir_canonical_model,
    )
    from bayesfilter.highdim.ledh_canonical_neutra_targets_tf import (
        make_canonical_neutra_target,
    )
    from bayesfilter.highdim.ledh_canonical_score_tf import (
        canonical_value_and_analytical_score,
    )
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model

    with tf.device("/CPU:0"):
        target = make_canonical_neutra_target("austria_sir",
                                              particle_count=N)
    observations = tf.cast(target.observations, DTYPE)
    theta0 = tf.constant([0.0, 0.0, 0.0], DTYPE)
    model, set_direction = austria_sir_canonical_model(theta0)
    set_direction(tf.constant([1.0, 0.0, 0.0], DTYPE))
    mean_np = tf.cast(
        zhao_cui_sir_austria_model().initial_mean, DTYPE
    ).numpy()
    horizon = int(observations.shape[0])
    base_kwargs = production_score_kwargs(DIM)

    def fixture(seed):
        rng = np.random.default_rng(9000 + seed)
        initial = tf.constant(
            mean_np[None, :] + rng.standard_normal((N, DIM)), DTYPE
        )
        covs = tf.constant(np.stack([np.eye(DIM)] * N), DTYPE)
        noises = tf.constant(
            rng.standard_normal((horizon, N, DIM)), DTYPE
        )
        return initial, covs, noises

    arms = {
        "S6+S7 (production)": dict(base_kwargs),
        "S6 only (S7 off)": {
            **base_kwargs,
            "correction_steps": 0,
            "pairwise_steps": 0,
            "coordinate_cap": 0.0,
        },
    }
    report = {}
    for arm_name, kwargs in arms.items():
        outcomes = []
        for seed in CRASH_SEEDS:
            initial, covs, noises = fixture(seed)
            try:
                _v, score = canonical_value_and_analytical_score(
                    model, theta0, initial, covs, noises, observations,
                    flow_substeps=8, with_score=True,
                    annealed_stages=4, annealed_seed=17, **kwargs,
                )
                value = float(score[0].numpy())
                outcomes.append(
                    f"{value:.3g}" if np.isfinite(value) else "nonfinite"
                )
            except Exception as error:
                text = str(error)
                outcomes.append(
                    "eigh-crash" if "SelfAdjointEig" in text
                    else "cholesky-crash" if "Cholesky" in text
                    else f"other:{type(error).__name__}"
                )
        report[arm_name] = outcomes
        print(f"[{arm_name}] seeds={CRASH_SEEDS}", flush=True)
        print(f"    outcomes={outcomes}", flush=True)

    prod = report["S6+S7 (production)"]
    s6 = report["S6 only (S7 off)"]
    def _failed(outcome):
        # ANY non-numeric outcome is a failure (the earlier version
        # counted only "crash" substrings and reported 0/6 while three
        # seeds were raising other:InvalidArgumentError).
        try:
            float(outcome)
            return False
        except ValueError:
            return True

    crashed_prod = sum(1 for o in prod if _failed(o))
    crashed_s6 = sum(1 for o in s6 if _failed(o))
    print(
        f"\nATTRIBUTION: production crashes {crashed_prod}/"
        f"{len(CRASH_SEEDS)}; S7-off crashes {crashed_s6}/"
        f"{len(CRASH_SEEDS)} -> "
        + (
            "crash is IN S7 (dual-cap spectral path)"
            if crashed_prod > 0 and crashed_s6 == 0
            else "crash NOT attributable to S7 alone"
        ),
        flush=True,
    )
    print(f"wall={time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

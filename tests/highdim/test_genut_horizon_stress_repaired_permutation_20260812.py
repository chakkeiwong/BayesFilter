from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim import ledh_pfpf_genut_initial_rqmc_tf as core
from docs.benchmarks import run_genut_horizon_stress_repaired_permutation_20260812 as runner


def test_horizon_scope_and_partitions_are_fresh() -> None:
    assert runner.MODEL_HORIZONS == {
        "ksc_sv_T100": 100,
        "exact_sv_T100": 100,
        "generalized_sv_T100": 100,
        "predator_prey_T50": 50,
        "austria_sir_T40": 40,
        "austria_sir_T50": 50,
    }
    runner._validate_partitions()


def test_horizon_models_have_static_shapes_and_event_orders() -> None:
    models = runner.build_horizon_stress_models(include_references=False)
    assert [m.row_id for m in models] == list(runner.MODEL_HORIZONS)
    for model in models:
        assert tuple(model.observations.shape[:1]) == (runner.MODEL_HORIZONS[model.row_id],)


def test_repaired_ancestry_is_exact_permutation() -> None:
    selected = core._transition_ancestors(  # noqa: SLF001
        tf.random.stateless_normal([runner.PARTICLE_COUNT, 3], [811, 812]),
        tf.fill([runner.PARTICLE_COUNT], tf.cast(1.0 / runner.PARTICLE_COUNT, tf.float32)),
        tf.sort(tf.random.stateless_uniform([runner.PARTICLE_COUNT], [813, 814])),
        ancestry_policy="hilbert_permutation_one_to_one",
        state_map_location=tf.zeros([3], tf.float32),
        state_map_scale=tf.ones([3], tf.float32),
        hilbert_bits=12,
        state_map_policy="fixed_supplied",
    )
    assert int(selected["ancestry_unique_count"].numpy()) == runner.PARTICLE_COUNT
    assert bool(selected["ancestry_permutation_valid"].numpy())


def test_runner_is_gpu_xla_and_scope_bound() -> None:
    source = inspect.getsource(runner)
    assert "configure_tensorflow_gpu_memory_growth" in source
    assert 'with tf.device("/GPU:0")' in source
    assert "jit_compile=True" in source
    assert "repository_standard_pairwise_backward_filtering_score" in source
    for forbidden in ("GradientTape", "ForwardAccumulator", "finite_difference_score"):
        assert forbidden not in source

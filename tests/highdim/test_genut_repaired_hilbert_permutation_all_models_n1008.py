from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.highdim import ledh_pfpf_genut_initial_rqmc_tf as core
from docs.benchmarks import (
    run_genut_repaired_hilbert_permutation_all_models_n1008 as runner,
)


def test_campaign_scope_is_full_horizon_n1008_and_fresh_partitions() -> None:
    assert runner.PARTICLE_COUNT == 1008
    assert runner.MODEL_HORIZONS == {
        "lgssm_T50": 50,
        "ksc_sv_T10": 10,
        "exact_sv_T10": 10,
        "generalized_sv_T10": 10,
        "predator_prey_T20": 20,
        "austria_sir_T20": 20,
    }
    runner._validate_partitions()


def test_repaired_ancestry_is_exact_n1008_permutation() -> None:
    count = runner.PARTICLE_COUNT
    particles = tf.random.stateless_normal([count, 3], [811, 812])
    weights = tf.fill([count], tf.cast(1.0 / count, tf.float32))
    selected = core._transition_ancestors(  # noqa: SLF001
        particles,
        weights,
        tf.sort(tf.random.stateless_uniform([count], [813, 814])),
        ancestry_policy="hilbert_permutation_one_to_one",
        state_map_location=tf.zeros([3], tf.float32),
        state_map_scale=tf.ones([3], tf.float32),
        hilbert_bits=12,
        state_map_policy="fixed_supplied",
    )
    assert int(selected["ancestry_unique_count"].numpy()) == count
    assert bool(selected["ancestry_permutation_valid"].numpy())
    tf.debugging.assert_equal(
        tf.sort(selected["selected_row_identities"]), tf.range(count)
    )


def test_runner_uses_standard_score_cpu_xla_and_no_local_derivative() -> None:
    source = inspect.getsource(runner)
    assert "jit_compile=True" in source
    assert 'with tf.device("/CPU:0")' in source
    assert "repository_standard_pairwise_backward_filtering_score" in source
    assert '"hilbert_inverse_cdf"' in source
    assert '"hilbert_permutation_one_to_one"' in source
    for forbidden in ("GradientTape", "ForwardAccumulator", "finite_difference_score"):
        assert forbidden not in source


def test_chapter18b_is_explicitly_excluded_for_support_mismatch() -> None:
    source = inspect.getsource(runner._manifest)
    assert "not_applicable_support_correct_route_absent" in source
    assert "rank_one_transition_support" in source

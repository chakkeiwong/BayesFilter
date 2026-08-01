from __future__ import annotations

import ast
from pathlib import Path

import tensorflow as tf

from docs.benchmarks import emit_contract_e_canonical_lgssm_phase8_rung0b as rung0b


ROOT = Path(__file__).resolve().parents[2]


def test_hmc_chain_factors_match_declared_coordinates() -> None:
    theta = tf.constant([0.5, 0.25, -0.25, 0.5, 0.75], tf.float64)
    tf.debugging.assert_equal(
        rung0b._hmc_chain_factors(theta),
        tf.constant([0.75, 0.9375, 0.9375, 0.5, 0.75], tf.float64),
    )


def test_kalman_oracle_is_transition_first_repository_route() -> None:
    source = Path(rung0b.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "tf_kalman_log_likelihood" in names
    assert "make_canonical_value_and_score_tf" not in source.replace(
        "canonical.make_canonical_value_and_score_tf", ""
    )
    assert "historical_raw" not in source
    assert "compact-sensitivity" not in source


def test_direct_joint_oracle_is_finite_and_differentiable() -> None:
    theta = tf.constant([0.5, 0.25, -0.25, 0.5, 0.75], tf.float64)
    observations = tf.constant([[0.2, -0.1, 0.05], [0.1, 0.15, -0.2]], tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        value = rung0b._direct_joint_gaussian_value(theta, observations)
    score = tape.gradient(value, theta)
    tf.debugging.assert_all_finite(value, "direct joint value")
    tf.debugging.assert_all_finite(score, "direct joint score")
    assert score.shape == (5,)


def test_rung0b_default_fixture_and_upstream_result_exist() -> None:
    assert rung0b.DEFAULT_FIXTURE.is_file()
    assert rung0b.UPSTREAM_DTYPE_RESULT.is_file()
    assert rung0b.DEFAULT_FIXTURE.is_relative_to(ROOT)

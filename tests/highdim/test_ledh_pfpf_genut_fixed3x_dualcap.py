from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.dual_cap_genut_primal_tf import dual_cap_genut_primal
from bayesfilter.highdim.genut_guided_proposal_tf import _restore_cloud_primal
from docs.benchmarks.run_ledh_pfpf_genut_initial_rqmc_all_models import _design
from docs.benchmarks.run_ledh_pfpf_genut_full_sqmc_full_horizons import campaign_inputs
from docs.benchmarks.run_ledh_pfpf_genut_full_sqmc_full_horizons import build_full_horizon_models
from docs.benchmarks import run_ledh_pfpf_genut_fixed3x_dualcap_all_models as runner


def _weighted_moments(points: tf.Tensor, weights: tf.Tensor):
    mean = tf.reduce_sum(weights[:, None] * points, axis=0)
    centered = points - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    return mean, covariance


def _uniform_moments(points: tf.Tensor):
    mean = tf.reduce_mean(points, axis=0)
    centered = points - mean[None, :]
    covariance = tf.einsum("ni,nj->ij", centered, centered) / tf.cast(
        tf.shape(points)[0], points.dtype
    )
    return mean, covariance


def test_dual_cap_primal_is_finite_and_restores_weighted_affine_moments():
    source = tf.random.stateless_normal([72, 3], [101, 102], dtype=tf.float32)
    weights = tf.nn.softmax(
        0.3 * tf.random.stateless_normal([72], [103, 104], dtype=tf.float32)
    )
    reset = tf.random.stateless_normal([72, 3], [105, 106], dtype=tf.float32)
    result = dual_cap_genut_primal(source, weights, reset)
    target_mean, target_covariance = _weighted_moments(source, weights)
    output_mean, output_covariance = _uniform_moments(result["particles"])
    tf.debugging.assert_near(output_mean, target_mean, atol=2.0e-5, rtol=2.0e-5)
    tf.debugging.assert_near(
        output_covariance, target_covariance, atol=2.0e-4, rtol=2.0e-4
    )
    assert bool(result["valid"].numpy())
    assert float(result["maximum_coordinatewise_post_cap_absolute"].numpy()) < 0.98
    assert float(result["fraction_coordinatewise_cap_active"].numpy()) > 0.0


def test_pairwise_and_radial_controls_are_structural_noop_for_scalar_state():
    source = tf.random.stateless_normal([72, 1], [111, 112], dtype=tf.float32)
    weights = tf.nn.softmax(
        tf.random.stateless_normal([72], [113, 114], dtype=tf.float32)
    )
    reset = tf.random.stateless_normal([72, 1], [115, 116], dtype=tf.float32)
    baseline = dual_cap_genut_primal(
        source,
        weights,
        reset,
        pairwise_steps=0,
        pairwise_particle_rms_cap=0.0,
    )
    candidate = dual_cap_genut_primal(
        source,
        weights,
        reset,
        pairwise_steps=4,
        pairwise_particle_rms_cap=2.0,
    )
    tf.debugging.assert_equal(candidate["particles"], baseline["particles"])
    tf.debugging.assert_equal(
        candidate["minimum_pairwise_particle_cap_scale"], tf.ones([], tf.float32)
    )


def test_restore_cloud_dual_cap_disabled_has_exact_historical_parity():
    particles = tf.random.stateless_normal([72, 3], [121, 122], dtype=tf.float32)
    weights = tf.nn.softmax(
        tf.random.stateless_normal([72], [123, 124], dtype=tf.float32)
    )
    kwargs = dict(
        epsilon=2.0,
        sinkhorn_steps=8,
        balance_steps=8,
        ridge=1.0e-5,
        reset_policy="contract_e",
    )
    historical = _restore_cloud_primal(particles, weights, _design(3), **kwargs)
    explicit_off = _restore_cloud_primal(
        particles, weights, _design(3), dual_cap_enabled=False, **kwargs
    )
    tf.debugging.assert_equal(historical["particles"], explicit_off["particles"])
    tf.debugging.assert_equal(
        historical["reset_valid"], explicit_off["reset_valid"]
    )


def test_restore_cloud_dual_cap_is_finite_and_marks_validity():
    particles = tf.random.stateless_normal([72, 3], [131, 132], dtype=tf.float32)
    weights = tf.nn.softmax(
        tf.random.stateless_normal([72], [133, 134], dtype=tf.float32)
    )
    result = _restore_cloud_primal(
        particles,
        weights,
        _design(3),
        epsilon=2.0,
        sinkhorn_steps=8,
        balance_steps=8,
        ridge=1.0e-5,
        dual_cap_enabled=True,
    )
    assert bool(result["reset_valid"].numpy())
    assert bool(result["dual_cap_valid"].numpy())
    tf.debugging.assert_all_finite(result["particles"], "dual-cap reset")


def test_combined_core_uses_standard_score_module_and_no_autodiff():
    source = (
        __import__(
            "bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf",
            fromlist=["dummy"],
        )
        .__file__
    )
    text = open(source, encoding="utf-8").read()
    assert "standard_pairwise_backward_marks(" in text
    assert "ForwardAccumulator" not in text
    assert "GradientTape" not in text


def test_n1008_inputs_and_designs_are_fully_sized():
    particle_count = 1008
    for model in build_full_horizon_models(include_references=False):
        inputs = campaign_inputs(model, 98201, particle_count)
        for arm in ("iid_existing", "initial_rqmc", "full_sqmc_halton"):
            payload = inputs[arm]
            assert payload["initial"].shape[0] == particle_count
            assert payload["process"].shape[1] == particle_count
            assert payload["ancestors"].shape[1] == particle_count
        assert _design(model.callbacks.state_dimension, particle_count).shape == (
            particle_count,
            model.callbacks.state_dimension,
        )


def test_checkpoint_write_is_atomic_and_row_key_is_stable(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    payload = {"rows": [{"model_id": "lgssm_T50", "seed": 7, "arm": "iid"}]}
    runner._write_json(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not (tmp_path / ".checkpoint.json.tmp").exists()
    assert runner._row_key(payload["rows"][0]) == ("lgssm_T50", 7, "iid")


def test_detached_launcher_preserves_cpu_checkpoint_and_status_contract():
    path = Path("scripts/launch_ledh_pfpf_genut_n1008_cpu_pilot_detached.sh")
    source = path.read_text(encoding="utf-8")
    for required in (
        "setsid bash -c",
        "CUDA_VISIBLE_DEVICES=-1",
        "--resume-checkpoint",
        "launcher.pid",
        "run.log",
        "exit-status.txt",
        "status.txt",
        "trusted/escalated host context",
    ):
        assert required in source

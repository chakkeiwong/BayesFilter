"""Static and exact-law checks for the generic five-stage model harness."""

import importlib.util
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("generic_five_stage_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_runner_uses_generic_controller_and_separates_model_validation() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "train_neutra_five_stage(" in source
    assert "dense_iaf_five_stage_variable_groups(transport)" in source
    assert 'choices=("funnel", "gaussian", "banana", "mixture")' in source
    assert 'choices=("staged", "cold")' in source
    assert "known_law_gate_passed" in source
    assert 'os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"' in source
    assert "configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)" in source
    assert "jit_compile=True" in source
    assert "import numpy" not in source
    assert "tf.map_fn" not in source
    assert "tf.vectorized_map" not in source


def test_runner_has_disjoint_selection_and_audit_seeds() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "seed=(20260815, 42001)" in source
    assert "44001 + int(args.seed_index)" in source
    assert "43000 + 10000 * phase_index[phase] + update" in source


class _ExactTransport:
    def __init__(self, forward):
        self._forward = forward

    def forward_and_logdet(self, latent):
        return self._forward(latent)


def test_exact_funnel_gaussian_and_banana_pass_known_law_audits() -> None:
    runner = _load_runner()
    for name in ("funnel", "gaussian", "banana"):
        model = runner._model(tf, name)
        if name == "funnel":
            transform = lambda z: (
                tf.concat((z[:, :1], z[:, 1:] * tf.exp(z[:, :1])), axis=1),
                tf.cast(model["dimension"] - 1, tf.float64) * z[:, 0],
            )
        elif name == "gaussian":
            mean = tf.constant(model["manifest"]["mean"], tf.float64)
            factor = tf.constant(model["manifest"]["factor"], tf.float64)
            logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)))
            transform = lambda z: (
                mean + tf.linalg.matvec(factor[tf.newaxis, :, :], z),
                tf.fill((tf.shape(z)[0],), logdet),
            )
        else:
            curvature = tf.constant(model["manifest"]["curvature"], tf.float64)
            transform = lambda z: (
                tf.concat(
                    (
                        z[:, :1],
                        z[:, 1:2] + curvature * (tf.square(z[:, :1]) - 1.0),
                        z[:, 2:],
                    ),
                    axis=1,
                ),
                tf.zeros((tf.shape(z)[0],), tf.float64),
            )
        result = runner._proposal_audit(
            tf,
            _ExactTransport(transform),
            model,
            sample_count=131072,
            seed=(20260815, 45001),
        )
        assert bool(result["passed"]), name


def test_exact_mixture_draws_pass_component_and_moment_screens() -> None:
    runner = _load_runner()
    model = runner._model(tf, "mixture")
    from bayesfilter.testing.importance_sampling_tf import (
        gaussian_mixture_log_prob_responsibilities_score,
        sample_gaussian_mixture,
    )
    from bayesfilter.testing.gaussian_mixture_diagnostics_tf import gaussian_mixture_moments

    target = model["mixture"]
    rows, _labels = sample_gaussian_mixture(
        131072,
        target["probabilities"],
        target["means"],
        target["covariances"],
        seed=(20260815, 45002),
    )
    _value, responsibilities, _score = gaussian_mixture_log_prob_responsibilities_score(
        rows,
        target["probabilities"],
        target["means"],
        target["covariances"],
    )
    moments = gaussian_mixture_moments(
        target["probabilities"], target["means"], target["covariances"]
    )
    second = tf.linalg.diag_part(moments["covariance"]) + tf.square(moments["mean"])
    screens = (
        runner._mean_interval(tf, responsibilities, target["probabilities"]),
        runner._mean_interval(tf, rows, moments["mean"]),
        runner._mean_interval(tf, tf.square(rows), second),
    )
    assert all(bool(screen["all_passed"]) for screen in screens)

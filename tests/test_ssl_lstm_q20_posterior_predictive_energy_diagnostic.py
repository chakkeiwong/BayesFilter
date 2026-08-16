from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_posterior_predictive_energy_diagnostic_2026_08_09.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("ssl_posterior_predictive", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(tmp_path: Path, module, *, mode_status: str = "RESOLVED") -> Path:
    draws = tf.constant(
        [
            [0.35, -0.08, 0.65, 0.05],
            [0.36, -0.07, 0.64, 0.04],
            [0.34, -0.09, 0.66, 0.06],
        ],
        tf.float64,
    )
    draw_path = tmp_path / "draws.tftensor"
    draw_path.write_bytes(bytes(tf.io.serialize_tensor(draws).numpy()))
    diagnostic_path = tmp_path / "sampler-result.json"
    diagnostic_path.write_text('{"passed":true}\n', encoding="ascii")
    artifact_path = tmp_path / "posterior.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": module.POSTERIOR_SCHEMA,
                "status": module.EXPECTED_STATUS,
                "target_signature": module.TARGET_SIGNATURE,
                "parameter_names": list(module.PARAMETER_NAMES),
                "warmup_excluded": True,
                "draw_weight_semantics": module.EXPECTED_WEIGHT_SEMANTICS,
                "relative_mode_weights_status": mode_status,
                "mode_weight_authority": "test_fixture",
                "sampler_diagnostics": {
                    "passed": True,
                    "result": {
                        "path": diagnostic_path.as_posix(),
                        "sha256": _sha(diagnostic_path),
                    },
                },
                "physical_draws": {
                    "path": draw_path.as_posix(),
                    "sha256": _sha(draw_path),
                    "dtype": "float64",
                    "shape": list(draws.shape),
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    return artifact_path


def test_authorized_posterior_loader_checks_draws_and_provenance(tmp_path: Path) -> None:
    module = _module()
    artifact = _artifact(tmp_path, module)
    draws, receipt = module.load_authorized_posterior_draws(artifact, tf)
    assert draws.shape == (3, 4)
    assert receipt["draw_count"] == 3
    assert receipt["relative_mode_weights_status"] == "RESOLVED"
    assert receipt["warmup_excluded"] is True
    assert receipt["artifact_sha256"] == _sha(artifact)


@pytest.mark.parametrize("mode_status", ("UNRESOLVED", None, "VISITED_ONLY"))
def test_loader_fails_closed_on_unresolved_multimodal_weights(
    tmp_path: Path, mode_status: str | None
) -> None:
    module = _module()
    artifact = _artifact(tmp_path, module, mode_status=mode_status)
    with pytest.raises(module.SSLLSTMPosteriorPredictiveError, match="weights"):
        module.load_authorized_posterior_draws(artifact, tf)


def test_loader_rejects_warmup_target_and_hash_drift(tmp_path: Path) -> None:
    module = _module()
    artifact = _artifact(tmp_path, module)
    payload = json.loads(artifact.read_text(encoding="ascii"))
    for field, value, match in (
        ("warmup_excluded", False, "warm-up"),
        ("target_signature", "wrong", "target signature"),
    ):
        changed = dict(payload)
        changed[field] = value
        artifact.write_text(json.dumps(changed) + "\n", encoding="ascii")
        with pytest.raises(module.SSLLSTMPosteriorPredictiveError, match=match):
            module.load_authorized_posterior_draws(artifact, tf)
    artifact = _artifact(tmp_path, module)
    payload = json.loads(artifact.read_text(encoding="ascii"))
    Path(payload["physical_draws"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(module.SSLLSTMPosteriorPredictiveError, match="SHA-256"):
        module.load_authorized_posterior_draws(artifact, tf)


def test_ssl_adapter_enforces_one_path_per_parameter_and_replication_one() -> None:
    module = _module()
    calls = []

    def forecast(parameters, **kwargs):
        calls.append((tf.identity(parameters), dict(kwargs)))
        count = int(parameters.shape[0])
        horizon = int(kwargs["horizon"])
        paths = tf.broadcast_to(parameters[:, :1, tf.newaxis], [count, 1, horizon])
        return SimpleNamespace(
            observations=paths,
            status=tf.ones([count], tf.bool),
        )

    parameters = tf.constant(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], tf.float64
    )
    simulator = module.ssl_lstm_batch_conditional_simulator(
        horizon=20, forecast=forecast
    )
    paths = simulator(parameters, tf.constant((1, 2), tf.int32))
    assert paths.shape == (2, 20)
    tf.debugging.assert_equal(paths[:, 0], parameters[:, 0])
    assert len(calls) == 1
    assert calls[0][1]["replication_count"] == 1
    assert calls[0][1]["q"] == 20


def test_ssl_adapter_rejects_multiple_paths_per_parameter() -> None:
    module = _module()

    def wrong_forecast(parameters, **kwargs):
        return SimpleNamespace(
            observations=tf.zeros(
                [int(parameters.shape[0]), 2, int(kwargs["horizon"])], tf.float64
            ),
            status=tf.ones([int(parameters.shape[0])], tf.bool),
        )

    simulator = module.ssl_lstm_batch_conditional_simulator(
        horizon=10, forecast=wrong_forecast
    )
    with pytest.raises(module.SSLLSTMPosteriorPredictiveError, match="one path"):
        simulator(tf.zeros([3, 4], tf.float64), tf.constant((1, 2), tf.int32))


def test_runner_source_forbids_fixed_summary_fallback_and_joint_decision() -> None:
    module = _module()
    source = RUNNER.read_text(encoding="utf-8")
    assert module.HORIZONS == (10, 20, 30, 50, 100)
    assert module.PATH_COUNT == 1000
    assert module.PERMUTATION_COUNT == 9999
    assert "replication_count=1" in source
    assert '"one_posterior_row_per_path": True' in source
    assert '"posterior_mean_used": False' in source
    assert '"posterior_median_used": False' in source
    assert '"posterior_map_used": False' in source
    assert '"joint_test_computed": False' in source
    assert "tf.reduce_mean(draws" not in source


def test_horizon_seed_domains_are_disjoint() -> None:
    module = _module()
    seeds = [seed for horizon in module.HORIZONS for seed in module._seeds(horizon).values()]
    assert len(seeds) == len(set(seeds)) == 20


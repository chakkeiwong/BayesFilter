"""CPU-only worker wiring test; tuning/device work is mocked, traces are real TF.

These fixtures exercise the inference consumer's decision boundary. They are
not numerical, GPU, tuning, or posterior evidence.
"""
import hashlib
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("scenario", ["healthy", "original_force_failure", "generic_failure"])
def test_worker_checks_original_force_before_posterior_functionals(tmp_path, monkeypatch, scenario):
    import tensorflow as tf
    import bayesfilter.inference as api
    from bayesfilter.inference import surrogate_hmc_equivalence as equivalence
    from bayesfilter.runtime import gpu_memory_policy
    from bayesfilter.testing import surrogate_hmc_sv_inference_consumer as consumers
    from bayesfilter.testing import surrogate_hmc_sv_inference_metrics as metrics
    from scripts import run_surrogate_hmc_sv as script

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    inputs = {}
    for name, value in (("claim", {}), ("selection", {"controls": {}}),
                        ("posterior_report", {"numerical_refinement": {
                            "expectations": [[0.]*13], "allowance": [0.]*13}, "margins": [1.]*13})):
        path = tmp_path/f"{name}.json"
        save(path, value)
        inputs[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save(tmp_path/"run-manifest.json", dict(source_hashes={}, inputs=inputs))
    save(tmp_path/"report.json", dict(status="RUNNING"))
    # No physical device enumeration or initialization occurs in this test.
    fake_device = SimpleNamespace(name="mock_device_for_cpu_wiring_test")
    monkeypatch.setattr(gpu_memory_policy, "configure_tensorflow_gpu_memory_growth", lambda *a, **k: {})
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda *a: [fake_device])
    monkeypatch.setattr(tf.config, "list_logical_devices", lambda *a: [fake_device])
    monkeypatch.setattr(tf.config.experimental, "enable_tensor_float_32_execution", lambda *a: None)
    monkeypatch.setattr(tf.config.experimental, "get_memory_info", lambda *a: dict(current=0, peak=0))
    monkeypatch.setattr(consumers, "make_sv_inference_consumer", lambda *a, **k:
        SimpleNamespace(manifest={}, force=object(), endpoint=object()))
    binding = SimpleNamespace(binding_hash="test", payload=lambda: {})
    monkeypatch.setattr(api, "bind_neural_force_hmc_tuning_runner", lambda **k: binding)
    monkeypatch.setattr(script, "tuning_config", lambda scope: object())
    result = SimpleNamespace(passed=True, checks={"test": True}, artifact_manifest_path="mock-tuning",
        nomination=SimpleNamespace(step_size=.1, num_leapfrog_steps=1, covariance=tf.eye(2, dtype=tf.float64)))
    monkeypatch.setattr(api, "tune_hmc_kernel", lambda **k: result)
    monkeypatch.setattr(api, "load_verified_tensorflow_hmc_result", lambda *a, **k: result)
    calls = dict(chunks=0, functionals=0, looks=0)

    class Runner:
        def run(self, config):
            calls["chunks"] += 1
            directory = config.output_dir
            directory.mkdir(parents=True)
            flags = tf.zeros([250, 4], tf.bool)
            if scenario == "original_force_failure":
                flags = tf.tensor_scatter_nd_update(flags, [[17, 2]], [True])
            def tensor(name, value):
                raw = tf.io.serialize_tensor(value).numpy()
                (directory/name).write_bytes(raw)
                return dict(path=name, sha256=hashlib.sha256(raw).hexdigest())
            healthy = scenario != "generic_failure"
            path = directory/"manifest.json"
            save(path, dict(health_passed=healthy, checks=dict(finite=healthy),
                raw_samples=tensor("samples.tfbin", tf.zeros([250, 4, 2], tf.float64)),
                chunk=dict(force_fallback=tensor("flags.tfbin", flags))))
            return dict(archive_manifest_path=str(path), health_passed=healthy)

    monkeypatch.setattr(api, "build_retained_bound_hmc_archive_runner_from_tuning_result", lambda **k: Runner())
    def functionals(samples):
        calls["functionals"] += 1
        return samples, tf.zeros([250, 4, 13], tf.float64), tf.ones([250, 4], tf.bool)
    monkeypatch.setattr(metrics, "make_functionals", lambda *a: functionals)
    def look(*a, **k):
        calls["looks"] += 1
        return dict(status="EQUIVALENT", diagnostic_valid=True, draws_per_chain=1000)
    monkeypatch.setattr(equivalence, "assess_equivalence", look)

    assert script.worker(tmp_path) == 0
    report = json.loads((tmp_path/"report.json").read_text())
    row = report["arms"]["B"]
    if scenario == "healthy":
        assert calls == dict(chunks=4, functionals=4, looks=1)
        assert row["status"] == "EQUIVALENT"
    else:
        assert calls == dict(chunks=1, functionals=0, looks=0)
        assert row["status"] == "HEALTH_VETO"
    assert row["chunks"][0]["health_passed"] is (scenario != "generic_failure")
    assert row["chunks"][0]["original_force_health"]["passed"] is (scenario != "original_force_failure")

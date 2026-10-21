"""CPU-only controller regressions; no posterior-convergence claims."""
from dataclasses import replace
from types import SimpleNamespace
import copy
import json

import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_tensorflow_resume as resume
from bayesfilter.inference import hmc_tensorflow_verified as verified
from bayesfilter.inference import hmc_tensorflow_tuning as stage
from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config


def chunk(error):
    states = tf.ones([2, 4, 2], tf.float64)
    flags = tf.zeros([2, 4], tf.bool)
    status = tf.concat([tf.ones([2, 4, 1], tf.int32),
                        tf.zeros([2, 4, 10], tf.int32)], axis=-1)
    return stage._TensorChunk(states, tf.minimum(0., -tf.fill([2, 4], tf.constant(error, tf.float64))),
        flags, flags, tf.fill([2, 4], tf.constant(error, tf.float64)), ~flags,
        states[-1], tf.constant(.1, tf.float64), status)


@pytest.mark.parametrize("error", [0., 999., 1001., 1e9, -1e9])
def test_finite_energy_is_warning_only(error):
    c = chunk(error)
    checks = verified._health(c, tf.zeros([4, 2], tf.float64), 1000.)
    assert "energy_valid" not in checks
    assert all(bool(value) for value in checks.values())
    d = verified._chunk_diagnostics(c, None, 1000.)
    assert d["energy"]["max_abs_error"] == abs(error)
    assert d["energy"]["threshold_exceedance_count"] == (8 if abs(error) > 1000 else 0)
    assert ("large_finite_energy_error" in d["warnings"]) == (abs(error) > 1000)
    old = verified._health(c, tf.zeros([4, 2], tf.float64), 1000., legacy_energy_gate=True)
    assert bool(old["energy_valid"]) == (abs(error) <= 1000)


def test_energy_policy_does_not_hide_nonfinite_or_endpoint_failures():
    c = chunk(float("nan"))
    assert not bool(verified._health(c, tf.zeros([4, 2], tf.float64), 1000.)["finite"])
    c = chunk(1e9)
    c = c._replace(divergence=tf.ones([2, 4], tf.bool),
        endpoint_status=tf.tensor_scatter_nd_update(c.endpoint_status, [[0, 0, 8]], [1]))
    checks = verified._health(c, tf.zeros([4, 2], tf.float64), 1000.)
    assert not bool(checks["no_divergences"])
    assert not bool(checks["endpoint_valid"])


def payload(monkeypatch):
    cfg = tuning_config("energy-migration-test")
    binding = SimpleNamespace(payload=lambda: {"identity": "same"})
    adapter = SimpleNamespace(adapter_signature=lambda: "same")
    current = {"hmc_tensorflow_verified.py": "new-verified",
               "hmc_tensorflow_resume.py": "new-resume", "neural_force_hmc.py": "same-kernel"}
    monkeypatch.setattr(verified, "_source_hashes", lambda: current)
    old = dict(current, **{"hmc_tensorflow_verified.py": resume._PRE_ENERGY_WARNING_VERIFIED_SHA256,
        "hmc_tensorflow_resume.py": resume._PRE_ENERGY_WARNING_RESUME_SHA256})
    data = dict(schema=verified.VERIFIED_TENSORFLOW_HMC_SCHEMA, artifact_authority=True,
        passed=False, handoff_eligible=False, rhat_threshold=1.01, config=cfg.payload(),
        adapter_signature="same", runner_binding=binding.payload(), source_hashes=old,
        endpoint_status_fields=list(stage.ENDPOINT_STATUS_TRACE_FIELDS[:-3]),
        sequence=dict(status="HEALTH_VETO", warmup_ready=False, chunks=[{}],
            events=[dict(checks=dict(finite=True, endpoint_valid=True, all_chains_moved=True,
                final_state_matches=True, no_divergences=True, energy_valid=False))],
            counts=dict(warmup=1500, verification=0)))
    return data, replace(cfg, resume_manifest="saved/manifest.json"), adapter, binding


def test_resume_allows_only_retired_energy_veto(monkeypatch):
    data, cfg, adapter, binding = payload(monkeypatch)
    previous, migration = resume.validate_resume_payload(data, cfg, adapter, binding)
    assert migration["numerical_kernel_changed"] is False
    for key in ("finite", "endpoint_valid", "all_chains_moved", "final_state_matches", "no_divergences"):
        bad = copy.deepcopy(data)
        bad["sequence"]["events"][0]["checks"][key] = False
        with pytest.raises(ValueError, match="healthy budget"):
            resume.validate_resume_payload(bad, cfg, adapter, binding)
    data["source_hashes"]["neural_force_hmc.py"] = "changed-kernel"
    with pytest.raises(ValueError, match="numerical source drift"):
        resume.validate_resume_payload(data, cfg, adapter, binding)


def test_legacy_gate_cannot_run_new_transitions(tmp_path):
    with pytest.raises(ValueError, match="archive-validation only"):
        resume.replay_then_continue(config=None, nomination=None, binding=None,
            source_root=tmp_path, records=[], output_root=tmp_path/"new", legacy_energy_gate=True)


@pytest.mark.parametrize("saved_empty", [{}, None])
def test_prepare_resume_accepts_writer_empty_phase(tmp_path, monkeypatch, saved_empty):
    cfg = tuning_config("empty-phase-test")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(dict(nomination_manifest_sha256="same", sequence={},
                                    warmup={}, verification=saved_empty)))
    cfg = replace(cfg, resume_manifest=str(path))
    nomination = SimpleNamespace(config=cfg.nomination)
    monkeypatch.setattr(resume, "validate_resume_payload",
        lambda *args: (cfg, {"kind": "unchanged"}))
    monkeypatch.setattr(stage, "_sha256", lambda *args: "same")
    monkeypatch.setattr(stage, "load_tensorflow_hmc_tuning_result", lambda *args, **kw: nomination)
    monkeypatch.setattr(verified, "_validate_nomination", lambda *args: None)
    monkeypatch.setattr(verified, "_check_sequence_replay", lambda *args: None)
    payload = json.loads(path.read_text())
    payload["sequence"]["chunks"] = []
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(resume, "replay_then_continue",
        lambda **kw: {"phases": {"warmup": [], "verification": []}})
    assert resume.prepare_resume(adapter=None, config=cfg, runner_binding=None)[2] is nomination

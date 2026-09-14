"""P18 backend parity/oracle tests; the no-MCMC guard forbids real transitions."""

import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import hmc_kernel_selection as selection
from bayesfilter.inference import hmc_kernel_tuning as tuning
from bayesfilter.inference import hmc_tuning


@pytest.mark.parametrize("actual,expected", [
    (1.0, 1.0), (1.0 + 1e-12, 1.0), (1.1, 1.0),
    (float("nan"), 1.0), (float("inf"), float("inf")), (1.0, float("inf")),
    (0.0, 1e-12), (-1.0 - 1e-12, -1.0),
])
def test_asymmetric_scalar_tolerance_matches_reference(actual, expected):
    for relative, absolute in ((1e-12, 0.0), (1e-5, 1e-8), (0.1, 0.0)):
        assert selection._scalar_close(actual, expected, rtol=relative, atol=absolute) == bool(
            np.isclose(actual, expected, rtol=relative, atol=absolute))


def test_scalar_and_json_contract_preserves_numpy_caller_inputs():
    assert selection._strict_integer(np.int64(7), name="count") == 7
    for invalid in (True, np.bool_(False), 1.0, np.float64(1.0)):
        with pytest.raises(ValueError):
            selection._strict_integer(invalid, name="count")
    assert selection._strict_bool(np.bool_(True), name="flag") is True
    for invalid in (1, 1.0, [True]):
        with pytest.raises(ValueError):
            selection._strict_bool(invalid, name="flag")
    observed = selection._json_ready({"integer": np.int64(7), "flag": np.bool_(True), "number": np.float64(0.12345678912345678)})
    assert json.loads(json.dumps(observed)) == {"integer": 7, "flag": True, "number": 0.12345678912345678}
    assert selection._optional_int(np.array([7])) == 7
    assert selection._optional_int(np.array([7.0])) is None
    assert selection._optional_int(np.array([True])) is None
    assert selection._optional_float([0.12345678912345678]) == 0.12345678912345678


@pytest.mark.parametrize("shape", [(), (0,), (9,), (3, 3), (2, 3, 4)])
def test_mass_npz_is_readable_exact_float64(shape, tmp_path):
    count = math.prod(shape)
    values = (np.arange(count, dtype=np.float64) + 0.12345678912345678).reshape(shape)
    path = tmp_path / "mass.npz"
    tuning._write_float64_mass_npz(path, {"position": tf.constant(values, tf.float64)})
    with np.load(path, allow_pickle=False) as archive:
        assert archive["position"].shape == shape
        assert archive["position"].dtype == np.dtype("float64")
        np.testing.assert_array_equal(archive["position"], values)


def test_mass_archive_preserves_debug_nonfinites_and_signed_zero(tmp_path):
    values = np.array([0.0, -0.0, np.nan, np.inf, -np.inf], dtype=np.float64)
    path = tmp_path / "debug.npz"
    tuning._write_float64_mass_npz(path, {"factor": values})
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["factor"], values)
        np.testing.assert_array_equal(np.signbit(archive["factor"]), np.signbit(values))


def test_windowed_capture_keeps_tensor_values_and_decision_counts():
    samples = np.arange(8, dtype=np.float64).reshape((4, 2)) / 7
    run = SimpleNamespace(samples=tf.constant(samples),
                          trace={"is_accepted": tf.constant([True, False, True, True]),
                                 "log_accept_ratio": tf.constant([-0.1, -3.0, 0.0, -0.1], tf.float64),
                                 "target_log_prob": tf.constant([-1.0, -2.0, -1.0, -1.0], tf.float64)},
                          diagnostics={"finite_sample_count": 4, "nonfinite_sample_count": 0},
                          metadata={"fixture_only": True})
    capture = tuning._windowed_stage_capture_payload(run, expected_steps=4, target_dimension=2)
    assert tf.is_tensor(capture["warmup_draws"])
    np.testing.assert_array_equal(capture["warmup_draws"], samples)
    np.testing.assert_array_equal(capture["acceptance_trace"], [1.0, 0.0, 1.0, 1.0])
    assert capture["raw_diagnostics"]["accepted_decision_count"] == 3
    assert capture["raw_diagnostics"]["acceptance_decision_count"] == 4


def test_numpy_import_is_confined_to_explicit_gaussian_diagnostics():
    from bayesfilter.runtime import runner

    for module in (selection, tuning, runner):
        tree = ast.parse(Path(module.__file__).read_text())
        assert not any(isinstance(node, ast.Import) and any(alias.name == "numpy" for alias in node.names)
                       for node in ast.walk(tree))
    tree = ast.parse(Path(hmc_tuning.__file__).read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)
               and any(alias.name == "numpy" for alias in node.names)]
    containing = {function.name for function in tree.body if isinstance(function, ast.FunctionDef)
                  for node in imports if function.lineno <= node.lineno <= function.end_lineno}
    assert containing == {"run_gaussian_dual_averaging_diagnostic", "_run_fixed_trajectory_candidate"}
    assert len(imports) == 2


@pytest.mark.parametrize("as_tensor", [False, True])
def test_runtime_json_preserves_exact_existing_hashes(as_tensor):
    from bayesfilter.runtime.runner import stable_config_hash

    values = np.array([[0.12345678912345678, -0.0], [2.0, 3.0]], dtype=np.float64)
    integer = 2**60 + 1
    actual = {"values": tf.constant(values) if as_tensor else values,
              "integer": tf.constant(integer, tf.int64) if as_tensor else np.int64(integer),
              "flag": tf.constant(True) if as_tensor else np.bool_(True),
              "empty": tf.constant([], tf.float64) if as_tensor else np.array([])}
    expected = {"values": values.tolist(), "integer": integer, "flag": True, "empty": []}
    serialized = json.dumps(expected, sort_keys=True, separators=(",", ":"))
    assert stable_config_hash(actual) == hashlib.sha256(serialized.encode()).hexdigest()


def test_repaired_backend_binds_both_policy_descriptions():
    config = tuning.HMCKernelTuningConfig.standard()
    resolved = tuning._public_resolved_policy_payload(config, output_path_enabled=True)
    payload = {"config": config.payload(), "resolved_policy": resolved}
    assert resolved["runtime_backend_policy"] == "ordinary_tf_tfp_runtime_v1"
    assert resolved["claim_bearing_blockers"] == ()
    assert tuning._require_claim_bearing_tuning_policy(payload) == resolved
    assert resolved["scientific_promotion_authority"] is False


@pytest.mark.parametrize("field", ["config", "resolved_policy"])
@pytest.mark.parametrize("historical", [None, "unknown_backend"])
def test_missing_or_mismatched_backend_cannot_upgrade_old_artifact(field, historical):
    config = tuning.HMCKernelTuningConfig.standard()
    payload = {"config": dict(config.payload()),
               "resolved_policy": dict(tuning._public_resolved_policy_payload(config, output_path_enabled=True))}
    payload[field]["runtime_backend_policy"] = historical
    with pytest.raises(ValueError, match="ordinary_runtime_numpy_policy_pending"):
        tuning._require_claim_bearing_tuning_policy(payload)

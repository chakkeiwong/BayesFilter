"""A finite fallback can pass generic map health but fail a stricter consumer."""
import hashlib
import json

import pytest

from bayesfilter.inference.surrogate_hmc_force_health import original_force_archive_health


def archive(tmp_path, *, fallback=False, shape=(250, 4)):
    import tensorflow as tf
    flags = tf.zeros(shape, tf.bool)
    if fallback:
        flags = tf.tensor_scatter_nd_update(flags, [[17, 2]], [True])
    raw = tf.io.serialize_tensor(flags).numpy()
    (tmp_path/"flags.tfbin").write_bytes(raw)
    path = tmp_path/"manifest.json"
    path.write_text(json.dumps(dict(health_passed=True, checks=dict(finite=True, endpoint_valid=True),
        chunk=dict(force_fallback=dict(path="flags.tfbin", sha256=hashlib.sha256(raw).hexdigest())))))
    return path


@pytest.mark.parametrize("fallback", [False, True])
def test_consumer_guard_checks_original_status_even_if_generic_health_passes(tmp_path, fallback):
    path = archive(tmp_path, fallback=fallback)
    result = original_force_archive_health(path, expected_transitions=250)
    assert result["passed"] is not fallback
    assert result["force_fallback_count_by_chain"] == [0, 0, int(fallback), 0]
    assert json.loads(path.read_text())["health_passed"] is True


def test_missing_or_corrupt_force_telemetry_cannot_pass(tmp_path):
    path = archive(tmp_path)
    (tmp_path/"flags.tfbin").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        original_force_archive_health(path, expected_transitions=250)
    path.write_text(json.dumps(dict(chunk={})))
    with pytest.raises(KeyError):
        original_force_archive_health(path, expected_transitions=250)


def test_chain_time_transposition_cannot_pass(tmp_path):
    path = archive(tmp_path, shape=(4, 250))
    with pytest.raises(ValueError, match="shape"):
        original_force_archive_health(path, expected_transitions=250)

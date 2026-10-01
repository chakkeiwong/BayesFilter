"""Identity compatibility for the repository's bounded XLA dispatcher."""

from functools import wraps

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_contract_e_identity import _CallableRoleSpec, _PRODUCTION_FACTORY, _wrapper_payload
from bayesfilter.highdim.moment_teacher_native_tf import freeze_scale_shift_core
from bayesfilter.ops.fixed_signature_tf import fixed_signature_function, fixed_signature_metadata


def test_dispatch_identity_binds_settings_and_repository_source():
    @fixed_signature_function(floating_dtype=tf.float64, max_specializations=2)
    def evaluate(value):
        return value * 2.

    role = _CallableRoleSpec("value", "fixture:evaluate", wrapper_kind="tensorflow_function", jit_compile=True)
    payload = _wrapper_payload(evaluate, role)
    assert payload["wrapper_kind"] == "repository_fixed_signature_function"
    assert payload["jit_compile"] is True
    assert payload["max_specializations"] == 2
    assert payload["floating_dtype"] == "float64"
    assert payload["factory_source_file"] == "bayesfilter/ops/fixed_signature_tf.py"
    assert len(payload["factory_loaded_code_sha256"]) == 64
    assert len(payload["factory_source_sha256"]) == 64
    copied = fixed_signature_metadata(evaluate)
    copied["jit_compile"] = False
    assert fixed_signature_metadata(evaluate)["jit_compile"] is True

    @wraps(evaluate)
    def copied_attributes(value):
        return evaluate(value)

    assert fixed_signature_metadata(copied_attributes) is None
    with pytest.raises(ValueError, match="non-TensorFlow"):
        _wrapper_payload(copied_attributes, role)
    evaluate._jit_compile = False
    with pytest.raises(ValueError, match="modified"):
        _wrapper_payload(evaluate, role)


def test_dispatch_identity_rejects_wrong_jit_requirement():
    @fixed_signature_function()
    def evaluate(value):
        return value

    role = _CallableRoleSpec("value", "fixture:evaluate", wrapper_kind="tensorflow_function", jit_compile=False)
    with pytest.raises(ValueError, match="jit_compile"):
        _wrapper_payload(evaluate, role)


def test_dependency_code_check_uses_its_own_future_flags():
    symbol = "bayesfilter.highdim.moment_teacher_native_tf:freeze_scale_shift_core"
    record = _PRODUCTION_FACTORY._binding_record("dependency", symbol, freeze_scale_shift_core)
    assert record["symbol"] == symbol
    assert record["loaded_code_sha256"]

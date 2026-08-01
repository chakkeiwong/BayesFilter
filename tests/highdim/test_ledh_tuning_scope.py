from __future__ import annotations

import pytest

from bayesfilter.highdim.ledh_tuning_scope import (
    LEDHTuningScope,
    require_scope_match,
    scope_from_mapping,
)


def _scope(**overrides):
    values = {
        "model_id": "model_a",
        "target_id": "target_a",
        "route_id": "route_a",
        "reset_contract_id": "contract_e_chol_v1",
        "horizon": 10,
        "prepared_data_id": "data_a",
        "particle_count": 1024,
        "state_dimension": 3,
        "parameter_count": 5,
        "dtype": "float32",
        "tf32_enabled": True,
        "jit_compile": True,
        "chunk_policy_id": "dpf_transport_exact_divisor_cap3000_v1",
        "row_chunk_size": 1024,
        "col_chunk_size": 1024,
        "row_blocks": 1,
        "col_blocks": 1,
        "control_family_id": "streaming_ot_sinkhorn_balance_v1",
    }
    values.update(overrides)
    return LEDHTuningScope(**values)


def test_scope_round_trip_and_digest_are_deterministic() -> None:
    scope = _scope()
    assert scope_from_mapping(scope.as_dict()) == scope
    assert scope_from_mapping(scope.as_dict()).scope_sha256 == scope.scope_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("model_id", "model_b"),
        ("route_id", "route_b"),
        ("horizon", 50),
        ("prepared_data_id", "data_b"),
        ("particle_count", 2048),
        ("row_chunk_size", 512),
        ("dtype", "float64"),
        ("control_family_id", "contract_e_tp_feature_chart_v1"),
    ),
)
def test_any_bound_scope_change_requires_new_tuning(field: str, value) -> None:
    expected = _scope()
    actual = expected.as_dict()
    actual[field] = value
    if field == "dtype":
        actual["tf32_enabled"] = False
    if field == "particle_count":
        actual["row_blocks"] = 2
        actual["col_blocks"] = 2
    if field == "row_chunk_size":
        actual["col_chunk_size"] = value
        actual["row_blocks"] = 2
        actual["col_blocks"] = 2
    with pytest.raises(ValueError, match="does not match"):
        require_scope_match(expected, actual, label=field)

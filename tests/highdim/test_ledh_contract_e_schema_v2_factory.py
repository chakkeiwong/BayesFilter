from __future__ import annotations

import copy
import functools
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pytest

from bayesfilter.highdim import _contract_e_phase2_test_fixture as fixture
from bayesfilter.highdim.ledh_contract_e_identity import (
    CONTRACT_E_DERIVATIVE_COMPOSITION_ID,
    CONTRACT_E_PHASE2_CANDIDATE_STATUS,
    CONTRACT_E_RESET_CONTRACT_ID,
    CONTRACT_E_ROUTE_FACTORY_ID,
    CONTRACT_E_ROW_NORMALIZATION_POLICY_ID,
    _CallableRoleSpec,
    _ExternalPrimitiveSpec,
    _PRODUCTION_FACTORY,
    _PreparedFieldSpec,
    _RouteSpecification,
    _make_test_candidate_factory,
    _require_factory_identity,
    issue_contract_e_route_identity,
)
from bayesfilter.highdim.ledh_forward_contract_v2 import (
    LEDH_FORWARD_SCALAR_ARTIFACT_V2_SCHEMA_VERSION,
    LEDH_FORWARD_V2_NONCLAIMS,
    build_ledh_forward_scalar_artifact_v2,
    ledh_forward_v2_evidence_gates,
    ledh_forward_v2_artifact_sha256,
    validate_ledh_forward_scalar_artifact_v2,
)
from bayesfilter.highdim.ledh_score_contract_v2 import (
    LEDH_SCORE_ARTIFACT_V2_SCHEMA_VERSION,
    LEDH_SCORE_V2_NONCLAIMS,
    build_ledh_score_artifact_v2,
    ledh_score_v2_correctness,
    validate_ledh_score_artifact_v2,
)
from docs.benchmarks.benchmark_two_lane_highdim_ledh_inclusive_results import (
    _score_payload,
)


MODULE = fixture.__name__
ROUTE_ID = "phase2_test_contract_e_route_v1"
ROW_ID = "phase2_contract_e_schema_test_row"


def _symbol(name: str) -> str:
    return f"{MODULE}:{name}"


def _specification(
    *,
    reset_symbol: str | None = None,
    value_symbol: str | None = None,
    gradient_symbol: str | None = None,
    owned_dependencies: tuple[str, ...] | None = None,
    allowed_external_roots: tuple[str, ...] = ("math",),
) -> _RouteSpecification:
    return _RouteSpecification(
        route_specification_id=ROUTE_ID,
        row_id=ROW_ID,
        target_scalar="observed_data_log_likelihood_estimator",
        target_output_tensor_field="log_likelihood",
        theta_coordinate_system="phase2_test_physical_coordinates",
        parameter_names=("alpha", "beta"),
        residual_design_id="fixed_centered_residual_design_v1",
        ridge_policy_id="prepared_parameter_independent_ridge_v1",
        prepared_fields=(
            _PreparedFieldSpec(
                name="residual_design",
                semantic_role="fixed realized residual design",
                allowed_dtype_names=("float64",),
                shape=("N", "D"),
            ),
            _PreparedFieldSpec(
                name="source_particles",
                semantic_role="fixed source particle preparation",
                allowed_dtype_names=("float64",),
                shape=("N", "D"),
            ),
            _PreparedFieldSpec(
                name="ridge",
                semantic_role="prepared parameter-independent ridge",
                allowed_dtype_names=("float64",),
                shape=("B",),
                strictly_positive=True,
            ),
        ),
        callable_roles=(
            _CallableRoleSpec("reset", reset_symbol or _symbol("fixture_reset")),
            _CallableRoleSpec("value", value_symbol or _symbol("fixture_value")),
            _CallableRoleSpec(
                "gradient", gradient_symbol or _symbol("fixture_gradient")
            ),
        ),
        owned_dependency_symbols=(
            owned_dependencies
            if owned_dependencies is not None
            else (_symbol("fixture_dependency"), _symbol("fixture_reset"))
        ),
        allowed_external_roots=allowed_external_roots,
    )


def _factory(specification: _RouteSpecification | None = None, *, external=()):
    return _make_test_candidate_factory(
        route_specifications=(specification or _specification(),),
        owned_module_roots=("bayesfilter.highdim.",),
        external_primitive_specs=external,
    )


def _callables(**overrides: Any) -> dict[str, Any]:
    output = {
        "reset": fixture.fixture_reset,
        "value": fixture.fixture_value,
        "gradient": fixture.fixture_gradient,
    }
    output.update(overrides)
    return output


def _prepared(*, order: str = "normal", **overrides: Any) -> dict[str, Any]:
    output = {
        "residual_design": np.arange(6, dtype=np.float64).reshape(3, 2),
        "source_particles": np.linspace(-0.5, 0.5, 6, dtype=np.float64).reshape(3, 2),
        "ridge": np.array([1.0e-3], dtype=np.float64),
    }
    output.update(overrides)
    if order == "reverse":
        return {name: output[name] for name in reversed(tuple(output))}
    return output


def _identity(*, specification: _RouteSpecification | None = None, **prepared_overrides):
    return _factory(specification).issue(
        route_specification_id=ROUTE_ID,
        callables=_callables(),
        prepared_inputs=_prepared(**prepared_overrides),
    )


def _forward(identity=None) -> dict[str, Any]:
    identity = identity or _identity()
    return {
        "schema_version": LEDH_FORWARD_SCALAR_ARTIFACT_V2_SCHEMA_VERSION,
        "route_identity": identity.to_dict(),
        "route_identity_sha256": identity.identity_sha256,
        "row_id": identity.row_id,
        "target_scalar": identity.target_scalar,
        "target_output_tensor_field": identity.target_output_tensor_field,
        "theta_coordinate_system": identity.theta_coordinate_system,
        "parameter_names": list(identity.parameter_names),
        "theta_values": [0.1, -0.2],
        "batch_seeds": [81120, 81121],
        "time_steps": 2,
        "num_particles": 8,
        "log_likelihood_by_seed": [-4.0, -4.1],
        "average_log_likelihood_by_seed": [-2.0, -2.05],
        "finite_output": True,
        "evidence_gates": ledh_forward_v2_evidence_gates(),
        "admission_status": CONTRACT_E_PHASE2_CANDIDATE_STATUS,
        "admitted": False,
        "canonical_admission_eligible": False,
        "nonclaims": list(LEDH_FORWARD_V2_NONCLAIMS),
    }


def _score(identity, forward) -> dict[str, Any]:
    return {
        "schema_version": LEDH_SCORE_ARTIFACT_V2_SCHEMA_VERSION,
        "route_identity": identity.to_dict(),
        "route_identity_sha256": identity.identity_sha256,
        "source_forward_artifact_sha256": ledh_forward_v2_artifact_sha256(forward),
        "row_id": identity.row_id,
        "target_scalar": identity.target_scalar,
        "target_output_tensor_field": identity.target_output_tensor_field,
        "theta_coordinate_system": identity.theta_coordinate_system,
        "score_parameter_names": list(identity.parameter_names),
        "score": [0.25, -0.75],
        "value_score_route_status": "same_factory_identity_candidate",
        "score_correctness": ledh_score_v2_correctness(),
        "evidence_gates": ledh_forward_v2_evidence_gates(),
        "score_admission_status": CONTRACT_E_PHASE2_CANDIDATE_STATUS,
        "admitted": False,
        "canonical_admission_eligible": False,
        "nonclaims": list(LEDH_SCORE_V2_NONCLAIMS),
    }


def test_identity_is_deterministic_across_mapping_order_and_reissuance() -> None:
    factory = _factory()
    first = factory.issue(
        route_specification_id=ROUTE_ID,
        callables=_callables(),
        prepared_inputs=_prepared(order="normal"),
    )
    second = factory.issue(
        route_specification_id=ROUTE_ID,
        callables={name: _callables()[name] for name in ("value", "reset", "gradient")},
        prepared_inputs=_prepared(order="reverse"),
    )

    assert first.identity_sha256 == second.identity_sha256
    assert first.to_dict() == second.to_dict()
    assert first.to_dict()["reset_contract_id"] == CONTRACT_E_RESET_CONTRACT_ID
    assert (
        first.to_dict()["derivative_composition_id"]
        == CONTRACT_E_DERIVATIVE_COMPOSITION_ID
    )
    assert first.to_dict()["route_factory_id"] == CONTRACT_E_ROUTE_FACTORY_ID
    assert (
        first.to_dict()["row_normalization_policy_id"]
        == CONTRACT_E_ROW_NORMALIZATION_POLICY_ID
    )
    assert [item["global_name"] for item in first.global_value_records] == [
        "FIXTURE_SCALE"
    ]
    assert [item["module_root"] for item in first.external_provenance] == ["math"]


@pytest.mark.parametrize(
    ("prepared", "message"),
    [
        ({"residual_design": np.ones((3, 2)), "ridge": np.array([1e-3])}, "missing"),
        ({**_prepared(), "extra": np.array([1.0])}, "extra"),
        (_prepared(ridge=np.array([0.0], dtype=np.float64)), "strictly positive"),
        (_prepared(ridge=np.array([np.nan], dtype=np.float64)), "finite"),
        (_prepared(ridge=np.array([1e-3], dtype=np.float32)), "forbidden dtype"),
        (
            _prepared(source_particles=np.ones((4, 2), dtype=np.float64)),
            "shape symbol",
        ),
    ],
)
def test_route_specification_owns_exhaustive_prepared_inputs(prepared, message) -> None:
    with pytest.raises(ValueError, match=message):
        _factory().issue(
            route_specification_id=ROUTE_ID,
            callables=_callables(),
            prepared_inputs=prepared,
        )


def test_tensor_dtype_shape_and_bytes_all_change_identity() -> None:
    base = _identity()
    changed_value = _identity(
        residual_design=np.arange(6, dtype=np.float64).reshape(3, 2) + 1.0
    )
    changed_shape = _identity(
        residual_design=np.arange(6, dtype=np.float64).reshape(2, 3),
        source_particles=np.linspace(-0.5, 0.5, 6, dtype=np.float64).reshape(2, 3),
    )
    assert len(
        {base.identity_sha256, changed_value.identity_sha256, changed_shape.identity_sha256}
    ) == 3


def test_identity_is_frozen_and_nested_mutation_invalidates_issuance() -> None:
    identity = _identity()
    with pytest.raises(FrozenInstanceError):
        identity.row_id = "forged"  # type: ignore[misc]
    forged = replace(identity, row_id="forged")
    with pytest.raises(ValueError, match="digest"):
        _require_factory_identity(forged)

    nested = _identity()
    with pytest.raises(TypeError):
        nested.prepared_input_records[0]["shape"] = (999, 2)
    _require_factory_identity(nested)


def test_factory_registries_cannot_be_mutated_after_construction() -> None:
    candidate = _factory()
    forged_external = _ExternalPrimitiveSpec(
        module_root="forged",
        distribution="forged",
        allowed_roles=("reset",),
    )
    with pytest.raises(TypeError):
        candidate._route_specifications[ROUTE_ID] = _specification()  # type: ignore[index]
    with pytest.raises(TypeError):
        candidate._external["forged"] = forged_external  # type: ignore[index]
    with pytest.raises(TypeError):
        _PRODUCTION_FACTORY._route_specifications[ROUTE_ID] = _specification()  # type: ignore[index]
    with pytest.raises(TypeError):
        _PRODUCTION_FACTORY._external["forged"] = forged_external  # type: ignore[index]
    with pytest.raises(AttributeError, match="immutable"):
        candidate._route_specifications = {}  # type: ignore[assignment]
    with pytest.raises(AttributeError, match="immutable"):
        candidate._external = {}  # type: ignore[assignment]
    with pytest.raises(AttributeError, match="immutable"):
        _PRODUCTION_FACTORY._route_specifications = {  # type: ignore[assignment]
            ROUTE_ID: _specification()
        }
    with pytest.raises(AttributeError, match="immutable"):
        _PRODUCTION_FACTORY._external = {"forged": forged_external}  # type: ignore[assignment]

    identity = candidate.issue(
        route_specification_id=ROUTE_ID,
        callables=_callables(),
        prepared_inputs=_prepared(),
    )
    _require_factory_identity(identity)
    with pytest.raises(ValueError, match="unregistered"):
        issue_contract_e_route_identity(
            route_specification_id=ROUTE_ID,
            callables=_callables(),
            prepared_inputs=_prepared(),
        )


def test_public_factory_is_inert_and_public_builders_reject_test_scope() -> None:
    with pytest.raises(ValueError, match="unregistered"):
        issue_contract_e_route_identity(
            route_specification_id=ROUTE_ID,
            callables=_callables(),
            prepared_inputs=_prepared(),
        )
    identity = _identity()
    with pytest.raises(ValueError, match="production factory instance"):
        build_ledh_forward_scalar_artifact_v2(
            route_identity=identity,
            theta_values=[0.1, -0.2],
            batch_seeds=[81120],
            time_steps=2,
            num_particles=8,
            log_likelihood_by_seed=[-4.0],
            average_log_likelihood_by_seed=[-2.0],
        )
    with pytest.raises(ValueError, match="production factory instance"):
        build_ledh_score_artifact_v2(
            route_identity=identity,
            source_forward_artifact=_forward(identity),
            score=[0.25, -0.75],
        )


def test_raw_wrapper_lambda_and_monkeypatched_symbols_fail_closed(monkeypatch) -> None:
    factory = _factory()
    with pytest.raises(ValueError, match="exact registered symbol"):
        factory.issue(
            route_specification_id=ROUTE_ID,
            callables=_callables(reset=fixture.fixture_raw_route),
            prepared_inputs=_prepared(),
        )
    with pytest.raises(ValueError, match="exact registered symbol"):
        factory.issue(
            route_specification_id=ROUTE_ID,
            callables=_callables(reset=functools.partial(fixture.fixture_reset)),
            prepared_inputs=_prepared(),
        )
    with pytest.raises(ValueError, match="exact registered symbol"):
        factory.issue(
            route_specification_id=ROUTE_ID,
            callables=_callables(reset=lambda value: value),
            prepared_inputs=_prepared(),
        )
    monkeypatch.setattr(fixture, "fixture_reset", fixture.fixture_raw_route)
    monkeypatched = _factory()
    with pytest.raises(ValueError, match="monkeypatched"):
        monkeypatched.issue(
            route_specification_id=ROUTE_ID,
            callables=_callables(reset=fixture.fixture_raw_route),
            prepared_inputs=_prepared(),
        )


def test_stale_loaded_code_and_incomplete_owned_closure_fail_closed(monkeypatch) -> None:
    original_code = fixture.fixture_reset.__code__
    monkeypatch.setattr(fixture.fixture_reset, "__code__", fixture.fixture_raw_route.__code__)
    with pytest.raises(ValueError, match="does not match current inspected source"):
        _factory().issue(
            route_specification_id=ROUTE_ID,
            callables=_callables(),
            prepared_inputs=_prepared(),
        )
    monkeypatch.setattr(fixture.fixture_reset, "__code__", original_code)
    incomplete = _specification(owned_dependencies=(_symbol("fixture_dependency"),))
    with pytest.raises(ValueError, match="owned dependency closure mismatch"):
        _factory(incomplete).issue(
            route_specification_id=ROUTE_ID,
            callables=_callables(),
            prepared_inputs=_prepared(),
        )


def test_external_primitive_requires_allowlist_and_version_provenance() -> None:
    spec = _specification(
        gradient_symbol=_symbol("fixture_external"),
        owned_dependencies=(_symbol("fixture_dependency"), _symbol("fixture_reset")),
        allowed_external_roots=("math", "packaging"),
    )
    callables = _callables(gradient=fixture.fixture_external)
    with pytest.raises(ValueError, match="not allowlisted"):
        _factory(spec).issue(
            route_specification_id=ROUTE_ID,
            callables=callables,
            prepared_inputs=_prepared(),
        )
    ambiguous = _ExternalPrimitiveSpec(
        module_root="packaging",
        distribution=None,
        allowed_roles=("gradient",),
    )
    with pytest.raises(ValueError, match="ambiguous provenance"):
        _factory(spec, external=(ambiguous,)).issue(
            route_specification_id=ROUTE_ID,
            callables=callables,
            prepared_inputs=_prepared(),
        )
    reviewed = _ExternalPrimitiveSpec(
        module_root="packaging",
        distribution="packaging",
        allowed_roles=("gradient",),
    )
    identity = _factory(spec, external=(reviewed,)).issue(
        route_specification_id=ROUTE_ID,
        callables=callables,
        prepared_inputs=_prepared(),
    )
    packaging_record = next(
        item for item in identity.external_provenance if item["module_root"] == "packaging"
    )
    assert packaging_record["distribution"] == "packaging"
    assert packaging_record["version"]


def test_tensorflow_wrapper_jit_and_version_are_factory_bound() -> None:
    specification = _specification(
        reset_symbol=_symbol("fixture_tf_reset"),
        value_symbol=_symbol("fixture_tf_value"),
        gradient_symbol=_symbol("fixture_tf_gradient"),
        owned_dependencies=(
            _symbol("fixture_dependency"),
            _symbol("fixture_gradient"),
            _symbol("fixture_reset"),
            _symbol("fixture_value"),
        ),
        allowed_external_roots=("math", "tensorflow"),
    )
    specification = replace(
        specification,
        callable_roles=tuple(
            _CallableRoleSpec(
                item.role,
                item.symbol,
                wrapper_kind="tensorflow_function",
                jit_compile=True,
            )
            for item in specification.callable_roles
        ),
    )
    tensorflow_external = _ExternalPrimitiveSpec(
        module_root="tensorflow",
        distribution="tensorflow",
        allowed_roles=("reset", "value", "gradient"),
    )
    factory = _factory(specification, external=(tensorflow_external,))
    callables = {
        "reset": fixture.fixture_tf_reset,
        "value": fixture.fixture_tf_value,
        "gradient": fixture.fixture_tf_gradient,
    }
    identity = factory.issue(
        route_specification_id=ROUTE_ID,
        callables=callables,
        prepared_inputs=_prepared(),
    )
    wrappers = {item["role"]: item["wrapper"] for item in identity.callable_records}
    assert all(item["jit_compile"] is True for item in wrappers.values())
    tensorflow_record = next(
        item
        for item in identity.external_provenance
        if item["module_root"] == "tensorflow"
    )
    assert tensorflow_record["version"]

    wrong_jit_spec = replace(
        specification,
        callable_roles=tuple(
            replace(item, jit_compile=False) for item in specification.callable_roles
        ),
    )
    with pytest.raises(ValueError, match="jit_compile"):
        _factory(wrong_jit_spec, external=(tensorflow_external,)).issue(
            route_specification_id=ROUTE_ID,
            callables=callables,
            prepared_inputs=_prepared(),
        )


def test_forward_candidate_validates_but_can_never_be_admitted() -> None:
    identity = _identity()
    artifact = _forward(identity)
    normalized = validate_ledh_forward_scalar_artifact_v2(
        artifact,
        expected_route_identity=identity,
        expected_row_id=ROW_ID,
    )
    assert normalized["admitted"] is False
    assert normalized["canonical_admission_eligible"] is False
    with pytest.raises(ValueError, match="Phase 2 identity candidate"):
        validate_ledh_forward_scalar_artifact_v2(
            artifact,
            expected_route_identity=identity,
            require_admitted=True,
        )


def test_blocker_factories_return_independent_canonical_copies() -> None:
    forward_first = ledh_forward_v2_evidence_gates()
    forward_first["production_reset_implementation"] = "pass"
    assert (
        ledh_forward_v2_evidence_gates()["production_reset_implementation"]
        == "blocked_phase3_not_run"
    )
    score_first = ledh_score_v2_correctness()
    score_first["status"] = "pass"
    assert ledh_score_v2_correctness()["status"] == "blocked_phase5_not_run"


def test_validators_return_canonical_state_without_input_aliases() -> None:
    identity = _identity()
    forward_input = _forward(identity)
    forward_normalized = validate_ledh_forward_scalar_artifact_v2(
        forward_input,
        expected_route_identity=identity,
    )
    forward_input["evidence_gates"]["production_reset_implementation"] = "pass"
    forward_input["route_identity"]["row_id"] = "forged"
    forward_normalized["nonclaims"].append("caller mutation")
    assert (
        forward_normalized["evidence_gates"]["production_reset_implementation"]
        == "blocked_phase3_not_run"
    )
    assert forward_normalized["route_identity"]["row_id"] == ROW_ID
    assert forward_input["nonclaims"] == list(LEDH_FORWARD_V2_NONCLAIMS)

    canonical_forward = _forward(identity)
    score_input = _score(identity, canonical_forward)
    score_normalized = validate_ledh_score_artifact_v2(
        score_input,
        source_forward_artifact=canonical_forward,
        expected_route_identity=identity,
    )
    score_input["score_correctness"]["status"] = "pass"
    score_input["evidence_gates"]["same_scalar_value_gradient_graph"] = "pass"
    score_normalized["route_identity"]["row_id"] = "mutated normalized copy"
    assert score_normalized["score_correctness"]["status"] == "blocked_phase5_not_run"
    assert (
        score_normalized["evidence_gates"]["same_scalar_value_gradient_graph"]
        == "blocked_phase5_not_run"
    )
    assert score_input["route_identity"]["row_id"] == ROW_ID


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update(row_id="self_labeled_row"),
        lambda payload: payload["route_identity"].update(row_id="forged"),
        lambda payload: payload.update(route_identity_sha256="0" * 64),
        lambda payload: payload["evidence_gates"].update(
            production_reset_implementation="pass"
        ),
        lambda payload: payload.update(admitted=True),
        lambda payload: payload.update(canonical_admission_eligible=True),
        lambda payload: payload.update(extra_field="forged"),
    ],
)
def test_forward_forgery_and_missing_gate_paths_fail(mutator) -> None:
    identity = _identity()
    artifact = _forward(identity)
    mutator(artifact)
    with pytest.raises(ValueError):
        validate_ledh_forward_scalar_artifact_v2(
            artifact,
            expected_route_identity=identity,
        )


def test_forged_visible_identity_mapping_is_not_factory_authority() -> None:
    identity = _identity()
    forged_expected = copy.deepcopy(identity.to_dict())
    with pytest.raises(TypeError, match="issued"):
        validate_ledh_forward_scalar_artifact_v2(
            _forward(identity),
            expected_route_identity=forged_expected,  # type: ignore[arg-type]
        )


def test_score_binds_exact_forward_and_same_factory_identity() -> None:
    identity = _identity()
    forward = _forward(identity)
    score = _score(identity, forward)
    normalized = validate_ledh_score_artifact_v2(
        score,
        source_forward_artifact=forward,
        expected_route_identity=identity,
        expected_row_id=ROW_ID,
    )
    assert normalized["score_admission_status"] == CONTRACT_E_PHASE2_CANDIDATE_STATUS
    with pytest.raises(ValueError, match="Phase 2 identity candidate"):
        validate_ledh_score_artifact_v2(
            score,
            source_forward_artifact=forward,
            expected_route_identity=identity,
            require_admitted=True,
        )


def test_score_rejects_forward_identity_digest_and_semantic_mismatch() -> None:
    identity = _identity()
    forward = _forward(identity)
    score = _score(identity, forward)
    changed_forward = copy.deepcopy(forward)
    changed_forward["log_likelihood_by_seed"][0] += 0.5
    with pytest.raises(ValueError, match="exact source forward"):
        validate_ledh_score_artifact_v2(
            score,
            source_forward_artifact=changed_forward,
            expected_route_identity=identity,
        )
    score["score_parameter_names"] = ["beta", "alpha"]
    with pytest.raises(ValueError, match="score_parameter_names"):
        validate_ledh_score_artifact_v2(
            score,
            source_forward_artifact=forward,
            expected_route_identity=identity,
        )


def test_existing_inclusive_leaderboard_consumer_rejects_v2_candidate() -> None:
    identity = _identity()
    forward = _forward(identity)
    score = _score(identity, forward)
    admitted, candidate, reason = _score_payload(
        score,
        expected_row_id=ROW_ID,
        source_value_artifact={},
    )
    assert admitted is None
    assert candidate["score_admission_status"] == "legacy_raw_score_memory_not_admitted"
    assert "not admitted" in reason

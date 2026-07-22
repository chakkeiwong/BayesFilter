from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from bayesfilter.inference.hmc_identity import (
    CANONICAL_ARRAY_IDENTITY_SCHEMA_V1,
    DETERMINISTIC_LGSSM_TARGET_ROUTE_V1,
    TFP_HMC_INTEGRATOR_ROUTE_V1,
    TFP_HMC_KERNEL_FAMILY_V1,
    CanonicalArrayIdentityV1,
    CanonicalFloat64V1,
    DeterministicLGSSMTargetIdentityV1,
    FrozenHMCExecutionContractV1,
    FrozenHMCTransformIdentityV1,
    FrozenHMCTransitionIdentityV1,
    SelectionProvenanceIdentityV1,
    SelectionStageIdentityV1,
    artifact_file_sha256,
    canonical_artifact_payload_hash,
)


TARGET_SCOPE = "test_lgssm_scope"
PARAMETER_NAMES = (
    "a11_raw",
    "a22_raw",
    "a33_raw",
    "a44_raw",
    "a21_raw",
    "a31_raw",
    "a32_raw",
    "a41_raw",
    "a42_raw",
    "a43_raw",
    "log_q1",
    "log_q2",
    "log_q3",
    "log_q4",
    "log_r1",
    "log_r2",
    "log_r3",
    "log_r4",
)
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
BARE_C = "c" * 64
BARE_D = "d" * 64


def _source_contract() -> dict:
    return {
        "schema": "unused.reporting.schema",
        "contract_id": "unused-contract-id",
        "static_shape": {
            "horizon": 999,
            "state_dim": 4,
            "observation_dim": 4,
            "innovation_dim": 999,
            "parameter_dim": 18,
        },
        "transform": {
            "rho_max": 0.85,
            "lower_scale": 0.35,
            "diagonal_transform": "unused description",
            "lower_transform": "unused description",
        },
        "truth_template": {
            "diag_A": [0.62, 0.48, 0.30, 0.16],
            "lower_A": {
                "a21": 0.18,
                "a31": -0.10,
                "a32": 0.14,
                "a41": 0.06,
                "a42": -0.08,
                "a43": 0.11,
            },
            "process_std": [0.30, 0.26, 0.22, 0.18],
            "observation_std": [0.12, 0.11, 0.10, 0.09],
        },
        "filter_program": {"score_authority": "unused reporting field"},
        "nonclaims": ["unused reporting field"],
    }


def _observations() -> np.ndarray:
    return np.arange(32, dtype=np.float64).reshape(8, 4) / 10.0


class _Capability:
    runtime_backend = "tensorflow_manual_lgssm_svd_graph_status_score"
    target_scope = TARGET_SCOPE


class _BaseAdapter:
    parameter_dim = 18

    def __init__(
        self,
        *,
        observations: np.ndarray | None = None,
        contract: dict | None = None,
        parameter_names=PARAMETER_NAMES,
    ) -> None:
        self._observations = _observations() if observations is None else observations
        self._contract = _source_contract() if contract is None else contract
        self._parameter_names = tuple(parameter_names)

    def value_score_capability(self):
        return _Capability()


class _TransformAdapter:
    parameter_dim = 18
    target_scope = TARGET_SCOPE

    def __init__(self, base_adapter, *, route: str, center, factor) -> None:
        self.base_adapter = base_adapter
        self.runtime_backend = route
        self.transform = SimpleNamespace(
            center=np.asarray(center, dtype=np.float64),
            factor=np.asarray(factor, dtype=np.float64),
            factor_orientation="row_right_transpose",
            log_jacobian_convention="constant_omitted",
        )


class _Replay:
    def __init__(self, base: _BaseAdapter | None = None) -> None:
        base_adapter = _BaseAdapter() if base is None else base
        phase4 = _TransformAdapter(
            base_adapter,
            route="test.phase4_affine.v1",
            center=np.linspace(-0.2, 0.2, 18),
            factor=np.eye(18),
        )
        final_factor = np.eye(18)
        final_factor[1, 0] = 0.1
        self.adapter = _TransformAdapter(
            phase4,
            route="test.final_affine.v1",
            center=np.linspace(0.3, 0.6, 18),
            factor=final_factor,
        )
        self.final_kernel_payload = {
            "step_size": 0.125,
            "num_leapfrog_steps": 5,
            "selection_provenance_not_consumed": "ignored",
        }
        self.contract = {
            "base_adapter_signature": "reconstruction-only-signature",
            "target_scope": TARGET_SCOPE,
            "target_dimension": 18,
            "final_hmc_adapter_signature": "reconstruction-only-final-signature",
        }


def _transition(replay: _Replay | None = None) -> FrozenHMCTransitionIdentityV1:
    return FrozenHMCTransitionIdentityV1.from_replay(
        _Replay() if replay is None else replay
    )


def _phase7_config() -> SimpleNamespace:
    return SimpleNamespace(
        payload={
            "execution": {
                "worker_count": 2,
                "chains_per_worker": 2,
                "root_seed": [20260711, 701],
                "cuda_visible_devices": "-1",
                "jit_compile": True,
                "use_xla": True,
                "chain_execution_mode": "tf_function",
                "compile_workers_sequentially": True,
                "wall_time_cap_seconds": 28800,
                "thread_environment": {
                    "TF_NUM_INTRAOP_THREADS": "8",
                    "TF_NUM_INTEROP_THREADS": "1",
                    "OMP_NUM_THREADS": "8",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                },
            },
            "burnin": {
                "initial_results_per_chain": 2000,
                "extension_results_per_chain": 1000,
                "check_window_results_per_chain": 1000,
                "max_results_per_chain": 16000,
            },
            "retained": {
                "initial_results_per_chain": 4000,
                "extension_results_per_chain": 2000,
                "check_interval_results_per_chain": 2000,
                "max_results_per_chain": 40000,
            },
            "diagnostics": {
                "rhat_max": 1.01,
                "bulk_ess_min": 1000.0,
                "tail_ess_min": 400.0,
                "all_parameters_required": True,
                "coordinate_system": (
                    "raw_lgssm_parameters_after_two_mass_transforms"
                ),
            },
        }
    )


def _execution(*, smoke: bool = False) -> FrozenHMCExecutionContractV1:
    return FrozenHMCExecutionContractV1.from_phase7_config(
        transition=_transition(),
        config=_phase7_config(),
        smoke=smoke,
        tensorflow_version="2.test",
        tfp_version="0.test",
        python_version="3.test",
    )


def _stage(stage_id: str = "bootstrap") -> SelectionStageIdentityV1:
    return SelectionStageIdentityV1(
        stage_id=stage_id,
        source_schema=f"test.{stage_id}.v1",
        canonical_payload_hash=SHA_A,
        selected_index=0,
    )


def _provenance(source_payload: dict | None = None) -> SelectionProvenanceIdentityV1:
    payload = (
        {
            "schema": "test.selection.v1",
            "final_status": "passed",
            "handoff_screen_policy": "current",
            "diagnostics": {"acceptance": 0.7},
        }
        if source_payload is None
        else source_payload
    )
    return SelectionProvenanceIdentityV1.from_source_payload(
        source_selection_payload=payload,
        tuning_config_hash=SHA_B,
        stage_lineage=(_stage(),),
        selected_step_hash=BARE_C,
        selected_trajectory_hash=BARE_D,
    )


def test_canonical_array_identity_is_layout_and_endian_independent() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    c_order = CanonicalArrayIdentityV1.from_array(np.array(base, order="C"))
    f_order = CanonicalArrayIdentityV1.from_array(np.array(base, order="F"))
    big_endian = CanonicalArrayIdentityV1.from_array(base.astype(">f8"))

    assert c_order == f_order == big_endian
    assert c_order.identity_hash == f_order.identity_hash == big_endian.identity_hash
    assert c_order.semantic_dtype == "float64"
    assert c_order.shape == (2, 2)


def test_canonical_array_identity_changes_for_dtype_shape_and_value() -> None:
    base = np.asarray([1.0, 2.0], dtype=np.float64)
    identity = CanonicalArrayIdentityV1.from_array(base)
    changed = base.copy()
    changed[1] = np.nextafter(changed[1], np.inf)

    assert CanonicalArrayIdentityV1.from_array(base.astype(np.float32)).identity_hash != identity.identity_hash
    assert CanonicalArrayIdentityV1.from_array(base.reshape(1, 2)).identity_hash != identity.identity_hash
    assert CanonicalArrayIdentityV1.from_array(changed).identity_hash != identity.identity_hash


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_canonical_array_identity_rejects_nonfinite(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        CanonicalArrayIdentityV1.from_array([value])


def test_canonical_float64_uses_exact_bits_and_rejects_bool() -> None:
    base = CanonicalFloat64V1.from_value(0.125)
    adjacent = CanonicalFloat64V1.from_value(np.nextafter(0.125, np.inf))

    assert base.ieee754_hex == "3fc0000000000000"
    assert adjacent.ieee754_hex != base.ieee754_hex
    assert base.value == 0.125
    with pytest.raises(ValueError, match="numeric"):
        CanonicalFloat64V1.from_value(True)


def test_target_round_trip_and_every_consumed_input_is_bound() -> None:
    target = _transition().target
    restored = DeterministicLGSSMTargetIdentityV1.from_payload(target.payload())
    changed_observations = _observations()
    changed_observations[0, 0] = np.nextafter(changed_observations[0, 0], np.inf)
    swapped_names = list(PARAMETER_NAMES)
    swapped_names[0], swapped_names[1] = swapped_names[1], swapped_names[0]
    array_mutations = (
        ("truth_diag_a", [0.61, 0.48, 0.30, 0.16]),
        ("truth_lower_a", [0.17, -0.10, 0.14, 0.06, -0.08, 0.11]),
        ("truth_process_std", [0.31, 0.26, 0.22, 0.18]),
        ("truth_observation_std", [0.13, 0.11, 0.10, 0.09]),
        ("prior_scales", [0.51] + [0.50] * 3 + [0.60] * 6 + [0.35] * 8),
    )
    mutations = [
        replace(target, observations=CanonicalArrayIdentityV1.from_array(changed_observations)),
        replace(target, parameter_names=tuple(swapped_names)),
        replace(target, rho_max=CanonicalFloat64V1.from_value(0.84)),
        replace(target, lower_scale=CanonicalFloat64V1.from_value(0.34)),
        replace(target, kalman_jitter=CanonicalFloat64V1.from_value(2.0e-9)),
        replace(target, singular_floor=CanonicalFloat64V1.from_value(2.0e-12)),
    ]
    mutations.extend(
        replace(
            target,
            **{name: CanonicalArrayIdentityV1.from_array(np.asarray(values, dtype=np.float64))},
        )
        for name, values in array_mutations
    )

    assert restored == target
    assert restored.identity_hash == target.identity_hash
    assert all(item.identity_hash != target.identity_hash for item in mutations)


def test_unused_contract_and_fixture_reporting_fields_do_not_change_target() -> None:
    first = _source_contract()
    second = copy.deepcopy(first)
    second["schema"] = "changed.reporting.schema"
    second["contract_id"] = "changed-reporting-id"
    second["static_shape"]["horizon"] = 12345
    second["static_shape"]["innovation_dim"] = 12345
    second["transform"]["diagonal_transform"] = "changed description"
    second["filter_program"] = {"changed": True}
    second["nonclaims"] = ["changed"]

    assert _transition(_Replay(_BaseAdapter(contract=first))).target == _transition(
        _Replay(_BaseAdapter(contract=second))
    ).target


@pytest.mark.parametrize(
    "field,value",
    [
        ("state_dim", 5),
        ("observation_dim", 5),
        ("parameter_dim", 19),
    ],
)
def test_target_rejects_wrong_lane_dimensions(field: str, value: int) -> None:
    target = _transition().target
    with pytest.raises(ValueError, match="requires"):
        replace(target, **{field: value})


def test_transition_round_trip_and_mechanical_mutations() -> None:
    transition = _transition()
    restored = FrozenHMCTransitionIdentityV1.from_payload(transition.payload())
    first, second = transition.transforms
    changed_center = replace(
        first,
        center=CanonicalArrayIdentityV1.from_array(
            np.asarray(first.center.shape and np.linspace(-0.3, 0.2, 18), dtype=np.float64)
        ),
    )
    changed_factor_array = np.eye(18)
    changed_factor_array[0, 0] = 1.1
    changed_factor = replace(
        second,
        factor=CanonicalArrayIdentityV1.from_array(changed_factor_array),
    )

    assert restored == transition
    assert restored.identity_hash == transition.identity_hash
    assert replace(transition, step_size=CanonicalFloat64V1.from_value(0.25)).identity_hash != transition.identity_hash
    assert replace(transition, num_leapfrog_steps=6).identity_hash != transition.identity_hash
    assert replace(transition, transforms=(changed_center, second)).identity_hash != transition.identity_hash
    assert replace(transition, transforms=(first, changed_factor)).identity_hash != transition.identity_hash
    with pytest.raises(ValueError, match="order"):
        replace(transition, transforms=(second, first))


def test_transition_routes_are_runner_constants_not_caller_overrides() -> None:
    transition = _transition()

    assert transition.kernel_family == TFP_HMC_KERNEL_FAMILY_V1
    assert transition.integrator_route == TFP_HMC_INTEGRATOR_ROUTE_V1
    assert transition.target.target_route == DETERMINISTIC_LGSSM_TARGET_ROUTE_V1
    with pytest.raises(TypeError, match="unexpected keyword"):
        FrozenHMCTransitionIdentityV1.from_replay(
            _Replay(),
            kernel_family="caller.override",
        )
    with pytest.raises(ValueError, match="kernel family"):
        replace(transition, kernel_family="caller.override")
    with pytest.raises(ValueError, match="float64"):
        replace(transition, state_dtype=None)


def test_full_replay_signatures_and_fixture_provenance_are_not_transition_payload() -> None:
    payload = json.dumps(_transition().payload(), sort_keys=True)

    assert "base_adapter_signature" not in payload
    assert "final_hmc_adapter_signature" not in payload
    assert "fixture_hash" not in payload
    assert "mass_artifact_signature" not in payload
    assert "nonclaims" not in payload


def test_transition_builder_checks_live_target_scope_backend_and_dimension() -> None:
    replay = _Replay()
    replay.contract["target_dimension"] = 17
    with pytest.raises(ValueError, match="dimension"):
        _transition(replay)

    replay = _Replay()
    replay.contract["target_scope"] = "wrong"
    with pytest.raises(ValueError, match="scope"):
        _transition(replay)

    replay = _Replay()
    replay.adapter.base_adapter.base_adapter.value_score_capability = lambda: SimpleNamespace(
        runtime_backend="wrong",
        target_scope=TARGET_SCOPE,
    )
    with pytest.raises(ValueError, match="backend"):
        _transition(replay)

    replay = _Replay()
    replay.adapter.base_adapter.base_adapter.value_score_capability = lambda: SimpleNamespace(
        runtime_backend="tensorflow_manual_lgssm_svd_graph_status_score",
        target_scope=None,
    )
    with pytest.raises(ValueError, match="capability scope"):
        _transition(replay)


def test_execution_round_trip_binds_serious_controller_semantics() -> None:
    execution = _execution()
    restored = FrozenHMCExecutionContractV1.from_payload(execution.payload())

    assert restored == execution
    assert restored.identity_hash == execution.identity_hash
    assert execution.run_mode == "serious"
    assert execution.global_initial_state.shape == (4, 18)
    assert execution.burnin_initial == 2000
    assert execution.retained_maximum == 40000
    assert execution.max_chunk_results == 4000
    assert execution.seed_dtype == "int32"
    assert execution.burnin_stage_index == 1
    assert execution.retained_stage_index == 2
    assert execution.compile_probe_stage_index == 1
    assert execution.compile_probe_check_index == 9999
    assert execution.compile_probe_advances_state is False
    assert execution.trace_policy == "reduced"
    assert execution.target_status_trace_policy == "none"
    assert dict(execution.thread_environment)["CUDA_VISIBLE_DEVICES"] == "-1"


def test_smoke_and_serious_execution_identities_are_distinct() -> None:
    serious = _execution(smoke=False)
    smoke = _execution(smoke=True)

    assert smoke.identity_hash != serious.identity_hash
    assert smoke.burnin_initial == 4
    assert smoke.retained_initial == 8
    assert smoke.max_chunk_results == 8
    assert smoke.bulk_ess_min.value == 1.0
    assert serious.bulk_ess_min.value == 1000.0
    assert smoke.diagnostic_gate_policy == "finite_diagnostics_only_non_promoting.v1"
    assert serious.diagnostic_gate_policy == "all_rank_normalized_thresholds_pass.v1"


@pytest.mark.parametrize(
    "field,value",
    [
        ("root_seed", (20260711, 702)),
        ("burnin_initial", 2001),
        ("retained_extension", 2001),
        ("wall_time_cap_seconds", 28799),
        ("tensorflow_version", "2.changed"),
        ("tfp_version", "0.changed"),
        ("python_version", "3.changed"),
    ],
)
def test_variable_execution_mutations_change_only_execution_identity(field, value) -> None:
    transition = _transition()
    execution = _execution()
    changed = replace(execution, **{field: value})

    assert changed.identity_hash != execution.identity_hash
    assert transition.identity_hash == _transition().identity_hash


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("compile_probe_advances_state", True, "requires"),
        ("jit_compile", False, "requires"),
        ("use_xla", False, "requires"),
        ("manual_thinning_allowed", True, "requires"),
        ("compile_probe_check_index", True, "integer"),
        ("burnin_stage_index", 2, "unsupported"),
        ("diagnostic_gate_policy", "wrong", "run_mode"),
        ("max_chunk_results", 4001, "controller chunks"),
        ("worker_count", 2.5, "integer"),
    ],
)
def test_execution_rejects_route_weakening_and_lossy_integer_coercion(
    field,
    value,
    match,
) -> None:
    with pytest.raises(ValueError, match=match):
        replace(_execution(), **{field: value})


def test_selection_provenance_is_typed_round_trippable_and_separate() -> None:
    transition = _transition()
    provenance = _provenance()
    restored = SelectionProvenanceIdentityV1.from_payload(provenance.payload())
    changed_source = {
        "schema": "test.selection.v1",
        "final_status": "passed",
        "handoff_screen_policy": "changed",
        "diagnostics": {"acceptance": 0.7},
    }
    changed = _provenance(changed_source)

    assert restored == provenance
    assert restored.identity_hash == provenance.identity_hash
    assert changed.identity_hash != provenance.identity_hash
    assert transition.identity_hash == _transition().identity_hash
    assert not hasattr(provenance, "provenance_payload")


def test_selection_stage_rejects_unknown_stage_and_invalid_index() -> None:
    with pytest.raises(ValueError, match="stage_id"):
        replace(_stage(), stage_id="unknown")
    with pytest.raises(ValueError, match="integer"):
        replace(_stage(), selected_index=True)


@pytest.mark.parametrize(
    "value",
    [
        "sha256:short",
        "sha256:" + "A" * 64,
        "a" * 64,
        None,
        "   ",
    ],
)
def test_tagged_sha256_validation_fails_closed(value) -> None:
    with pytest.raises(ValueError, match="SHA-256|sha256|nonblank"):
        replace(_execution(), transition_identity_hash=value)


@pytest.mark.parametrize(
    "builder,payload",
    [
        (
            CanonicalArrayIdentityV1.from_payload,
            {
                "schema": CANONICAL_ARRAY_IDENTITY_SCHEMA_V1,
                "semantic_dtype": "float64",
                "shape": [1],
                "canonical_byte_order": "big_endian",
                "canonical_memory_order": "C",
                "byte_sha256": "0" * 64,
            },
        ),
        (
            CanonicalFloat64V1.from_payload,
            {
                "schema": "bayesfilter.canonical_float64.v1",
                "ieee754_hex": "3ff0000000000000",
            },
        ),
        (FrozenHMCTransitionIdentityV1.from_payload, _transition().payload()),
        (FrozenHMCExecutionContractV1.from_payload, _execution().payload()),
        (SelectionProvenanceIdentityV1.from_payload, _provenance().payload()),
    ],
)
def test_payload_parsers_reject_unknown_fields(builder, payload) -> None:
    bad = copy.deepcopy(payload)
    bad["unknown"] = True
    with pytest.raises(ValueError, match="fields mismatch"):
        builder(bad)


def test_payload_parsers_reject_missing_fields_unknown_schema_and_nonstring_keys() -> None:
    missing = dict(_transition().payload())
    missing.pop("step_size")
    with pytest.raises(ValueError, match="fields mismatch"):
        FrozenHMCTransitionIdentityV1.from_payload(missing)

    unknown_schema = dict(_transition().payload())
    unknown_schema["schema"] = "bayesfilter.frozen_hmc_transition_identity.v999"
    with pytest.raises(ValueError, match="unsupported"):
        FrozenHMCTransitionIdentityV1.from_payload(unknown_schema)

    nonstring_key = dict(_transition().payload())
    nonstring_key[1] = "collision"
    with pytest.raises(ValueError, match="keys must be strings"):
        FrozenHMCTransitionIdentityV1.from_payload(nonstring_key)


def test_constructed_identities_do_not_retain_mutable_source_payloads() -> None:
    source = {
        "schema": "test.selection.v1",
        "final_status": "passed",
        "nested": {"items": [1, 2]},
    }
    provenance = _provenance(source)
    before = provenance.identity_hash
    source["nested"]["items"].append(3)

    config = _phase7_config()
    execution = FrozenHMCExecutionContractV1.from_phase7_config(
        transition=_transition(),
        config=config,
        smoke=False,
        tensorflow_version="2.test",
        tfp_version="0.test",
        python_version="3.test",
    )
    execution_before = execution.identity_hash
    config.payload["execution"]["root_seed"][0] += 1

    returned = dict(_transition().payload())
    returned["target"] = dict(returned["target"])
    returned["target"]["parameter_names"] = ("mutated",)

    assert provenance.identity_hash == before
    assert execution.identity_hash == execution_before
    assert _transition().target.parameter_names == PARAMETER_NAMES


def test_artifact_integrity_helpers_cover_exact_bytes_and_strict_full_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(b'{"a":1}\n')
    first = artifact_file_sha256(path)
    path.write_bytes(b'{"a":1} \n')
    second = artifact_file_sha256(path)

    assert first != second
    assert canonical_artifact_payload_hash({"a": 1}) != canonical_artifact_payload_hash(
        {"a": 1, "provenance": "changed"}
    )
    assert canonical_artifact_payload_hash({"value": []}) != canonical_artifact_payload_hash(
        {"value": {}}
    )
    assert canonical_artifact_payload_hash({"value": 1}) != canonical_artifact_payload_hash(
        {"value": 1.0}
    )
    assert canonical_artifact_payload_hash({"value": float("inf")}) == (
        canonical_artifact_payload_hash({"value": float("inf")})
    )
    assert canonical_artifact_payload_hash({"value": float("inf")}) != (
        canonical_artifact_payload_hash({"value": float("-inf")})
    )
    with pytest.raises(ValueError, match="non-string"):
        canonical_artifact_payload_hash({1: "bad"})


def test_transform_requires_float64_and_rejects_missing_field() -> None:
    transform = _transition().transforms[0]
    payload = dict(transform.payload())
    payload.pop("runtime_route")
    with pytest.raises(ValueError, match="fields mismatch"):
        FrozenHMCTransformIdentityV1.from_payload(payload)

    with pytest.raises(ValueError, match="float64"):
        replace(
            transform,
            center=CanonicalArrayIdentityV1.from_array(
                np.zeros(18, dtype=np.float32)
            ),
        )

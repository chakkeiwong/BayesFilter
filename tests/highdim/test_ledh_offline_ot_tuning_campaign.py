from __future__ import annotations

import inspect
import copy
import pytest

from docs.benchmarks import run_ledh_offline_ot_tuning_campaign as campaign
from bayesfilter.highdim.ledh_tuning_scope import require_scope_match


def _valid_result() -> dict[str, object]:
    return {
        "time_steps": 10,
        "num_particles": 1024,
        "sinkhorn_steps": 20,
        "balance_steps": 5,
        "finite": True,
        "bitwise_replay": True,
            "chart_valid": True,
            "marginal_valid": True,
            "reset_valid": True,
        "maximum_tv_column_error": 9.0e-5,
        "maximum_row_error": 9.0e-3,
        "preparation_identity": {
            "num_particles": 1024,
            "sinkhorn_steps": 20,
            "balance_steps": 5,
            "transport_chunk_policy_id": "dpf_transport_exact_divisor_cap3000_v1",
            "transport_block_grid": [1, 1],
            "row_chunk_size": 1024,
            "col_chunk_size": 1024,
        },
        "device": {
            "dtype": "float32",
            "tf32_enabled": True,
            "jit_compile": True,
        },
        "work": {
            "sinkhorn_state_constructions": 10,
            "terminal_balance_state_constructions": 10,
            "transport_tile_sweeps": 10,
            "marginal_tile_sweeps": 0,
            "diagnostic_solver_reconstructions": 0,
        },
        "graph": {
            "python_horizon_unroll": False,
            "while_operation_types": ["StatelessWhile"],
        },
    }


def test_candidate_grids_are_explicit_and_validated() -> None:
    assert campaign._parse_positive_candidates("20,25,30", label="sinkhorn") == (
        20,
        25,
        30,
    )
    with pytest.raises(ValueError, match="positive and nonempty"):
        campaign._parse_positive_candidates("0,2", label="balance")
    with pytest.raises(ValueError, match="unique"):
        campaign._parse_positive_candidates("2,2", label="balance")


def test_active_defaults_name_the_per_scope_program() -> None:
    assert campaign.CAMPAIGN_ID == "ledh-per-model-scope-tuning-20260719"
    assert campaign.PLAN_PATH.endswith(
        "bayesfilter-ledh-per-model-scope-tuning-master-program-2026-07-19.md"
    )


def test_candidate_identity_uses_the_calling_scope_program() -> None:
    args = campaign._candidate_args(
        sinkhorn_steps=20,
        balance_steps=8,
        seeds=(1, 2),
        horizon=50,
        num_particles=2000,
        attempt_id="identity-test",
        replay_diagnostic=False,
        campaign_id="scope-campaign",
        plan_path="docs/plans/scope-plan.md",
    )
    assert args.campaign_id == "scope-campaign"
    assert args.plan_path == "docs/plans/scope-plan.md"
    assert args.num_particles == 2000


def test_gpu_configuration_precedes_scope_package_import() -> None:
    source = inspect.getsource(campaign.main)
    configure_index = source.index("runner._configure_gpu")
    scope_import_index = source.index(
        "from bayesfilter.highdim.ledh_tuning_scope import require_scope_match"
    )
    scope_creation_index = source.index("scope = _scope")
    assert configure_index < scope_import_index < scope_creation_index


def test_node_status_separates_numerical_and_resource_vetoes() -> None:
    assert campaign._node_status(direct_gate_passed=True, elapsed_seconds=1.0) == "PASS"
    assert (
        campaign._node_status(direct_gate_passed=False, elapsed_seconds=1.0)
        == "FAIL_DIRECT_GATE"
    )
    assert (
        campaign._node_status(
            direct_gate_passed=True,
            elapsed_seconds=campaign.NODE_CAP_SECONDS + 1.0,
        )
        == "FAIL_NODE_CAP"
    )
    assert (
        campaign._node_status(
            direct_gate_passed=True,
            elapsed_seconds=11.0,
            node_cap_seconds=10.0,
        )
        == "FAIL_NODE_CAP"
    )


def test_direct_gate_checks_probability_errors_and_program_identity() -> None:
    result = _valid_result()
    assert campaign._direct_gate(result)

    result["maximum_row_error"] = 0.0100001
    assert not campaign._direct_gate(result)

    result = _valid_result()
    result["preparation_identity"]["sinkhorn_steps"] = 25
    assert not campaign._direct_gate(result)

    result = _valid_result()
    result["preparation_identity"]["row_chunk_size"] = 512
    assert not campaign._direct_gate(result)


def test_horizon_change_creates_a_new_tuning_scope() -> None:
    t10 = campaign._scope(horizon=10, num_particles=1024)
    t50 = campaign._scope(horizon=50, num_particles=1024)
    assert t10 != t50
    assert t10.scope_sha256 != t50.scope_sha256
    with pytest.raises(ValueError, match="does not match"):
        require_scope_match(t50, t10.as_dict(), label="T=10 selection")


def test_tuning_partition_summaries_remain_separate() -> None:
    result = _valid_result()
    result.update(
        {
            "per_seed_value": [-1.0, -2.0],
            "per_seed_physical_score": [[1.0], [2.0]],
            "tv_column_error_by_seed_time": [[9.0e-5], [2.0e-4]],
            "maximum_row_error_by_seed_time": [[9.0e-3], [9.0e-3]],
            "reset_valid_by_seed_time": [[True], [True]],
            "chart_valid_by_seed": [True, True],
        }
    )
    node = {"result": result}
    assert campaign._partition_summary(node, 0, 1)["pass"] is True
    assert campaign._partition_summary(node, 1, 2)["pass"] is False


def test_scope_identity_binds_control_family_and_particle_count() -> None:
    scope = campaign._scope(horizon=10, num_particles=1024)
    changed_control = scope.as_dict()
    changed_control["control_family_id"] = "contract_e_tp_feature_chart_v1"
    with pytest.raises(ValueError, match="does not match"):
        require_scope_match(scope, changed_control, label="wrong route controls")

    changed_particles = scope.as_dict()
    changed_particles.update(
        particle_count=2000,
        row_chunk_size=2000,
        col_chunk_size=2000,
    )
    with pytest.raises(ValueError, match="does not match"):
        require_scope_match(scope, changed_particles, label="wrong particle count")


def _microbatch_result(seed: int, value: float) -> dict[str, object]:
    result = _valid_result()
    result.update(
        {
            "git_commit": "abc",
            "source_sha256": {"source.py": "digest"},
            "campaign_id": "campaign",
            "plan_path": "docs/plans/plan.md",
            "arm": "all_active_contract_e",
            "estimator_seeds": [seed],
            "cache_same_cloud_geometry": False,
            "warm_repetitions": 1,
            "theta": [0.72, 0.55, 0.35, 0.35, 0.45],
            "per_seed_value": [value],
            "per_seed_physical_score": [[value] * 5],
            "aggregate_value": value,
            "aggregate_physical_score": [value] * 5,
            "kalman_value": None,
            "kalman_physical_score": None,
            "value_difference_to_kalman": None,
            "physical_score_difference_to_kalman": None,
            "replay_checked": True,
            "tv_column_error_by_seed_time": [[9.0e-5] * 10],
            "maximum_row_error_by_seed_time": [[9.0e-3] * 10],
            "marginal_valid_by_seed_time": [[True] * 10],
            "reset_valid_by_seed_time": [[True] * 10],
            "chart_valid_by_seed": [True],
            "work_valid": True,
            "hard_valid": True,
            "timing_seconds": {
                "trace": 1.0,
                "compile_plus_first_execution": 2.0,
                "warm_execution": 3.0,
            },
            "gpu_allocator_bytes": {"current": 10, "peak": 20},
        }
    )
    result["preparation_identity"] = copy.deepcopy(result["preparation_identity"])
    result["preparation_identity"].update(
        time_steps=10,
        root_seeds_in_order=[seed],
        tensor_sha256={"initial_noise": str(seed)},
    )
    return result


def test_seed_groups_are_ordered_and_bounded() -> None:
    assert campaign._seed_groups((1, 2, 3, 4, 5), seed_microbatch_size=2) == (
        (1, 2),
        (3, 4),
        (5,),
    )
    with pytest.raises(ValueError, match="must be positive"):
        campaign._seed_groups((1,), seed_microbatch_size=0)


def test_microbatch_merge_preserves_seed_order_and_recomputes_means() -> None:
    first = _microbatch_result(1, -1.0)
    second = _microbatch_result(2, -3.0)
    merged = campaign._merge_microbatch_results(
        [first, second],
        seeds=(1, 2),
        seed_groups=((1,), (2,)),
    )
    assert merged["estimator_seeds"] == [1, 2]
    assert merged["per_seed_value"] == [-1.0, -3.0]
    assert merged["aggregate_value"] == -2.0
    assert merged["aggregate_physical_score"] == [-2.0] * 5
    assert merged["preparation_identity"]["root_seeds_in_order"] == [1, 2]
    assert merged["microbatching"]["microbatch_count"] == 2


def test_microbatch_merge_rejects_cross_scope_drift() -> None:
    first = _microbatch_result(1, -1.0)
    second = _microbatch_result(2, -3.0)
    second["num_particles"] = 2000
    with pytest.raises(ValueError, match="invariant mismatch"):
        campaign._merge_microbatch_results(
            [first, second],
            seeds=(1, 2),
            seed_groups=((1,), (2,)),
        )

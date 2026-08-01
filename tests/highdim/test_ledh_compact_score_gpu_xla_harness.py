from __future__ import annotations

import copy
import inspect
import json
import statistics
from argparse import Namespace
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_forward_contract import validate_ledh_forward_scalar_artifact
from bayesfilter.highdim.ledh_score_contract import (
    LEDH_SCORE_ADMISSION_STATUS_FULL,
    LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW,
)
from bayesfilter.ledh_fd_policy import evaluate_ledh_fd_policy
from docs.benchmarks import benchmark_ledh_compact_score_gpu_xla as harness
from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p
from docs.benchmarks import (
    build_complete_highdim_ledh_phase2_phase3_command_manifest as complete_commands,
)
from docs.benchmarks import build_ledh_phase9_gpu_command_manifest as command_manifest
from experiments.dpf_implementation.tf_tfp.filters import (
    experimental_batched_ledh_pfpf_ot_tf as core_tf,
)


def _tiny_args(row: str) -> Namespace:
    particles = 4
    args = Namespace(
        batch_seeds=[81120],
        time_steps=1,
        num_particles=particles,
        theta_values=[0.0, 0.0, 0.0],
        fd_step=1.0e-3,
        transport_policy="active-all",
        sinkhorn_iterations=1,
        sinkhorn_epsilon=1.0,
        annealed_scaling=0.9,
        annealed_convergence_threshold=1.0e-3,
        transport_plan_mode="streaming",
        transport_gradient_mode=core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE,
        transport_ad_mode="full",
        row_chunk_size=particles,
        col_chunk_size=particles,
        particle_chunk_size=particles,
        dtype="float64",
        tf32_mode="disabled",
        flow_observation_variance=None,
        device="/CPU:0",
        device_scope="cpu",
        cuda_visible_devices=None,
        expect_device_kind="cpu",
        historical_raw_diagnostic=True,
    )
    if row == "lgssm":
        args.sinkhorn_epsilon = harness.ROW_SPECS[row].sinkhorn_epsilon
    elif row == "fixed-sir":
        args.theta_values = [0.0, 0.0, 0.0]
    elif row == "predator-prey":
        args.num_particles = 2
        args.row_chunk_size = 2
        args.col_chunk_size = 2
        args.particle_chunk_size = 2
    elif row == "actual-sv":
        args.flow_observation_variance = 3.141592653589793**2 / 2.0
    elif row == "generalized-sv":
        args.flow_observation_variance = 2.0
    elif row == "ksc-sv":
        args.flow_observation_variance = 3.141592653589793**2 / 2.0
    else:
        raise AssertionError(row)
    return args


def _parse_aggregate_args(
    row: str,
    *,
    score_paths: list[str] | None = None,
    fd_paths: list[str] | None = None,
) -> Namespace:
    spec = harness.ROW_SPECS[row]
    return harness._parse_args(  # noqa: SLF001
        [
            "--row",
            row,
            "--stage",
            "aggregate",
            "--batch-seeds",
            ",".join(str(seed) for seed in harness.FULL_ROW_BATCH_SEEDS),
            "--score-shards",
            ",".join(score_paths or ["score.json"]),
            "--fd-shards",
            ",".join(fd_paths or ["fd.json"]),
            "--output",
            "/tmp/aggregate.json",
            "--time-steps",
            str(spec.full_time_steps),
            "--num-particles",
            str(spec.full_num_particles),
            "--historical-raw-diagnostic",
        ]
    )


def _source_identity(spec: harness.RowSpec) -> tuple[dict, dict]:
    path = harness.ROOT / spec.source_value_artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalized = validate_ledh_forward_scalar_artifact(
        payload,
        expected_row_id=spec.row_id,
        require_admitted=False,
    )
    return payload, normalized


def _manifest(args: Namespace, spec: harness.RowSpec, seed: int, stage: str) -> dict:
    output = "/tmp/shard.json"
    argv = [
        "python",
        "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py",
        "--row",
        spec.name,
        "--stage",
        stage,
        "--batch-seeds",
        str(seed),
        "--historical-raw-diagnostic",
        "--output",
        output,
    ]
    manifest = {
        "command": harness.shlex.join(argv),
        "command_argv": argv,
        "runner_path": "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py",
        "output": output,
        "markdown_output": None,
        "working_directory": str(harness.ROOT),
        "git_commit": harness._git_output(("git", "rev-parse", "HEAD")),  # noqa: SLF001
        "git_status_short": " M scoped-file.py",
        "code_source_sha256": harness._code_source_sha256(spec),  # noqa: SLF001
        "governance_artifact_sha256": harness._governance_artifact_sha256(),  # noqa: SLF001
        "python_executable": "/home/chakwong/anaconda3/envs/tf-gpu/bin/python",
        "python_version": "3.11",
        "tensorflow_version": "2.19.1",
        "host": "test-host",
        "platform": "test-platform",
        "gpu_trust_basis": harness.GPU_TRUST_BASIS,
        "device_scope": "visible",
        "cuda_visible_devices": "0",
        "device": "/GPU:0",
        "expect_device_kind": "gpu",
        "jit_compile": True,
        "dtype": "float32",
        "tf32_mode": "enabled",
        "row": spec.name,
        "row_id": spec.row_id,
        "canonical_target_artifact": harness.CANONICAL_TARGETS_PATH,
        "canonical_target_artifact_sha256": harness.CANONICAL_TARGETS_SHA256,
        "canonical_target_sha256": harness._canonical_target_sha256(spec),  # noqa: SLF001
        "fd_endpoint_contract": harness._fd_endpoint_contract(spec),  # noqa: SLF001
        "phase1_gate_artifact_sha256": harness._phase1_gate_bindings(),  # noqa: SLF001
        "configuration_identity": harness._configuration_identity(args, spec),  # noqa: SLF001
        "route_identity": harness._route_identity(args, spec),  # noqa: SLF001
        "source_value_artifact": spec.source_value_artifact,
        "source_value_artifact_sha256": harness._source_value_sha256(spec),  # noqa: SLF001
        "score_parameter_names": list(spec.parameter_names),
        "truth_theta": list(spec.truth_theta),
        "stage": stage,
        "score_reference_json": None,
        "time_steps": int(args.time_steps),
        "num_particles": int(args.num_particles),
        "batch_seeds": [seed],
        "transport_policy": args.transport_policy,
        "sinkhorn_iterations": int(args.sinkhorn_iterations),
        "sinkhorn_epsilon": float(args.sinkhorn_epsilon),
        "annealed_scaling": float(args.annealed_scaling),
        "annealed_convergence_threshold": float(args.annealed_convergence_threshold),
        "row_chunk_size": int(args.row_chunk_size),
        "col_chunk_size": int(args.col_chunk_size),
        "particle_chunk_size": int(args.particle_chunk_size),
        "transport_plan_mode": args.transport_plan_mode,
        "transport_ad_mode": args.transport_ad_mode,
        "transport_gradient_mode": args.transport_gradient_mode,
        "flow_observation_variance": args.flow_observation_variance,
        "memory_budget_mib": float(args.memory_budget_mib),
        "command_timeout_seconds": args.command_timeout_seconds,
        "legacy_module_fd_step_not_used_by_gpu_fd": spec.legacy_module_fd_step,
        "fd_step_policy": harness.ledh_fd_step_policy_metadata(),
        "fd_policy_id": harness.LEDH_FD_POLICY_ID,
        "fd_diagnostic_scope": harness.LEDH_FD_DIAGNOSTIC_SCOPE,
        "fd_base_relative_tolerance": harness.LEDH_FD_BASE_RELATIVE_TOLERANCE,
        "fd_coordinate_relative_error_denominator": harness.LEDH_FD_DENOMINATOR,
        "fd_pass_rule": harness.LEDH_FD_PASS_RULE,
        "fd_statistical_interpretation": harness.LEDH_FD_STATISTICAL_STATUS,
        "gpu_execution_authorized": harness.GPU_EXECUTION_AUTHORIZED,
        "root_cause_repair_gpu_execution_authorized": (
            harness.ROOT_CAUSE_REPAIR_GPU_EXECUTION_AUTHORIZED
        ),
        "plan_path": harness.PLAN_PATH,
        "fd_policy_correction_plan_path": harness.FD_POLICY_CORRECTION_PLAN_PATH,
        "historical_result_path": harness.HISTORICAL_RESULT_PATH,
        "result_path": harness.RESULT_PATH,
        "execution_manifest_path": harness.EXECUTION_MANIFEST_PATH,
        "exact_commands_path": harness.EXACT_COMMANDS_PATH,
        "gate_b_review_path": harness.GATE_B_REVIEW_PATH,
        "gate_b_repair_review_path": harness.GATE_B_REPAIR_REVIEW_PATH,
        "gate_b_result_path": harness.GATE_B_RESULT_PATH,
        "gate_b_result_review_path": harness.GATE_B_RESULT_REVIEW_PATH,
        "root_cause_repair_plan_path": harness.ROOT_CAUSE_REPAIR_PLAN_PATH,
        "root_cause_repair_commands_path": harness.ROOT_CAUSE_REPAIR_COMMANDS_PATH,
        "complete_highdim_exact_commands_path": harness.COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH,
        "complete_highdim_exact_commands_sha256": harness._sha256(  # noqa: SLF001
            harness.ROOT / harness.COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH
        ),
    }
    manifest["command_identity"] = harness._command_identity_from_manifest(manifest)  # noqa: SLF001
    return manifest


def _common_shard(args: Namespace, spec: harness.RowSpec, seed: int, stage: str) -> dict:
    _payload, source = _source_identity(spec)
    prepared_input_fingerprint = harness._prepared_input_fingerprint(  # noqa: SLF001
        {"tensors": {"fixture": tf.constant([float(seed)], dtype=tf.float32)}}
    )
    configuration_identity = harness._configuration_identity(args, spec)  # noqa: SLF001
    route_identity = harness._route_identity(args, spec)  # noqa: SLF001
    return {
        "schema_version": harness.SCHEMA_VERSION,
        "artifact_status": "completed",
        "terminal_artifact": True,
        "timestamp_utc": "2026-07-10T00:00:00+00:00",
        "elapsed_seconds": 1.0,
        "row_id": spec.row_id,
        "score_route": spec.score_route,
        "score_admission_status": LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW,
        "evidence_class": harness.GPU_TRUST_BASIS,
        "source_value_artifact": spec.source_value_artifact,
        "source_value_artifact_sha256": harness._sha256(  # noqa: SLF001
            harness.ROOT / spec.source_value_artifact
        ),
        "target_observation_policy": source["target_observation_policy"],
        "theta_coordinate_system": source["theta_coordinate_system"],
        "score_parameter_names": list(spec.parameter_names),
        "truth_theta": list(spec.truth_theta),
        "score_evaluation_theta": harness._float32_theta(spec),  # noqa: SLF001
        "canonical_target_sha256": harness._canonical_target_sha256(spec),  # noqa: SLF001
        "configuration_identity": configuration_identity,
        "route_identity": route_identity,
        "randomness_identity": harness._randomness_identity(  # noqa: SLF001
            seed=seed,
            prepared_input_fingerprint=prepared_input_fingerprint,
            configuration_identity=configuration_identity,
            route_identity=route_identity,
        ),
        "prepared_input_fingerprint": prepared_input_fingerprint,
        "physical_gpus": ["PhysicalDevice('/physical_device:GPU:0')"],
        "logical_gpus": ["LogicalDevice('/device:GPU:0')"],
        "precision": {
            "dtype": "float32",
            "active_dtype": "float32",
            "tf_dtype": "float32",
            "tf32_mode": "enabled",
            "tf32_execution_enabled": True,
        },
        "run_manifest": _manifest(args, spec, seed, stage),
    }


def _score_shard(args: Namespace, spec: harness.RowSpec, seed: int) -> dict:
    payload = _common_shard(args, spec, seed, "score-only")
    score = [float(index + 1) for index in range(len(spec.parameter_names))]
    payload.update(
        {
            "score": score,
            "objective": 1.0,
            "total_log_likelihood": 1.0,
            "log_likelihood_by_seed": [1.0],
            "per_seed_score": [list(score)],
            "score_derivative_provenance": spec.score_route,
            "value_score_route_status": "same_route_value_score",
            "value_score_same_transport_algorithm": True,
            "score_finite": True,
            "score_output_devices": ["/job:localhost/device:GPU:0"],
            "score_gpu_memory_stats_reset": True,
            "score_gpu_memory_info_before": {"current": 0, "peak": 0},
            "score_gpu_memory_info_after": {"current": 0, "peak": 256 * 1024 * 1024},
            "memory_diagnostics": {
                "score_memory_budget_pass": True,
                "full_row_memory_gate_applicable": True,
                "n10000_memory_pass": True,
                "peak_mib": 256.0,
                "budget_mib": harness.MEMORY_BUDGET_MIB,
                "source": "score_gpu_memory_info_after",
            },
            "no_autodiff_score_route": True,
            "uses_gradient_tape": False,
            "uses_forward_accumulator": False,
            "uses_stopped_partial_derivative": False,
        }
    )
    return payload


def _fd_shard(
    args: Namespace,
    spec: harness.RowSpec,
    seed: int,
    score: list[float],
    score_hash: str,
) -> dict:
    payload = _common_shard(args, spec, seed, "fd-only")
    diagnostics = []
    finite_differences = []
    for index, name in enumerate(spec.parameter_names):
        theta = float(tf.constant(spec.truth_theta[index], tf.float32).numpy())
        nominal_step = harness.coordinate_central_difference_step(theta)
        step_tensor = tf.constant(nominal_step, tf.float32)
        theta_tensor = tf.constant(theta, tf.float32)
        minus_parameter = float((theta_tensor - step_tensor).numpy())
        plus_parameter = float((theta_tensor + step_tensor).numpy())
        denominator = float(
            (tf.constant(plus_parameter, tf.float32) - tf.constant(minus_parameter, tf.float32)).numpy()
        )
        effective_step = float((tf.constant(denominator, tf.float32) / 2.0).numpy())
        minus_objective = 0.0
        plus_objective = float(
            (tf.constant(score[index], tf.float32) * tf.constant(denominator, tf.float32)).numpy()
        )
        numerator = float(
            (tf.constant(plus_objective, tf.float32) - tf.constant(minus_objective, tf.float32)).numpy()
        )
        finite_difference = float(
            (tf.constant(numerator, tf.float32) / tf.constant(denominator, tf.float32)).numpy()
        )
        finite_differences.append(finite_difference)
        diagnostics.append(
            {
                "parameter": name,
                "direction_index": index,
                "theta": theta,
                "nominal_step": nominal_step,
                "minus_parameter": minus_parameter,
                "plus_parameter": plus_parameter,
                "effective_step": effective_step,
                "effective_denominator": denominator,
                "center_theta": harness._float32_theta(spec),  # noqa: SLF001
                "minus_endpoint": {
                    "role": "minus",
                    "theta": [
                        minus_parameter if j == index else center
                        for j, center in enumerate(harness._float32_theta(spec))  # noqa: SLF001
                    ],
                    "total_log_likelihood": minus_objective,
                },
                "plus_endpoint": {
                    "role": "plus",
                    "theta": [
                        plus_parameter if j == index else center
                        for j, center in enumerate(harness._float32_theta(spec))  # noqa: SLF001
                    ],
                    "total_log_likelihood": plus_objective,
                },
                "minus_objective": minus_objective,
                "plus_objective": plus_objective,
                "objective_numerator": numerator,
                "endpoint_objectives_equal": plus_objective == minus_objective,
                "finite_difference": finite_difference,
            }
        )
    payload.update(
        {
            "score": list(score),
            "score_derivative_provenance": spec.score_route,
            "value_score_route_status": "same_route_value_score",
            "score_reference_sha256": score_hash,
            "value_output_devices": ["/job:localhost/device:GPU:0"],
            "score_correctness": {
                "kind": "same_scalar_finite_difference",
                "status": "pass",
                "step_policy": harness.ledh_fd_step_policy_metadata(),
                "finite_difference_diagnostics": diagnostics,
                "fd_policy": evaluate_ledh_fd_policy(
                    score,
                    finite_differences,
                    spec.parameter_names,
                ),
                "uses_value_only_scalar_route": True,
            },
        }
    )
    return payload


def _rewrite_shard(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _refresh_command_identity(payload: dict) -> None:
    manifest = payload["run_manifest"]
    manifest["command"] = harness.shlex.join(manifest["command_argv"])
    manifest["command_identity"] = harness._command_identity_from_manifest(  # noqa: SLF001
        manifest
    )


def _set_fd_direction(payload: dict, index: int, requested_fd: float) -> float:
    diagnostic = payload["score_correctness"]["finite_difference_diagnostics"][index]
    denominator = tf.constant(diagnostic["effective_denominator"], tf.float32)
    minus_objective = tf.constant(0.0, tf.float32)
    plus_objective = tf.constant(requested_fd, tf.float32) * denominator
    numerator = plus_objective - minus_objective
    finite_difference = numerator / denominator
    minus_value = float(minus_objective.numpy())
    plus_value = float(plus_objective.numpy())
    numerator_value = float(numerator.numpy())
    fd_value = float(finite_difference.numpy())
    diagnostic.update(
        {
            "minus_objective": minus_value,
            "plus_objective": plus_value,
            "objective_numerator": numerator_value,
            "endpoint_objectives_equal": plus_value == minus_value,
            "finite_difference": fd_value,
        }
    )
    diagnostic["minus_endpoint"]["total_log_likelihood"] = minus_value
    diagnostic["plus_endpoint"]["total_log_likelihood"] = plus_value
    correctness = payload["score_correctness"]
    correctness["fd_policy"]["parameters"][index]["finite_difference"] = fd_value
    finite_differences = [
        float(entry["finite_difference"])
        for entry in correctness["fd_policy"]["parameters"]
    ]
    correctness["fd_policy"] = evaluate_ledh_fd_policy(
        payload["score"],
        finite_differences,
        payload["score_parameter_names"],
    )
    return fd_value


def _increment(values: list[float], index: int, amount: float) -> None:
    values[index] += amount


def test_gate_a_registry_freezes_row_identity_transport_and_fd_policy() -> None:
    expected = {
        "lgssm": (50, 10000, (512, 512, 256), 1.0e-3, 0.5),
        "fixed-sir": (20, 10000, (1024, 1024, 512), 1.0e-3, 1.0),
        "predator-prey": (20, 10000, (512, 512, 512), 1.0e-4, 1.0),
        "actual-sv": (1000, 10000, (512, 512, 512), 1.0e-4, 1.0),
        "generalized-sv": (1008, 10000, (512, 512, 512), 1.0e-4, 1.0),
        "ksc-sv": (1000, 10000, (512, 512, 512), 1.0e-4, 1.0),
    }
    assert set(harness.ROW_SPECS) == set(expected)
    for row, spec in harness.ROW_SPECS.items():
        full_time, particles, chunks, fd, epsilon = expected[row]
        assert (spec.full_time_steps, spec.full_num_particles) == (full_time, particles)
        assert (spec.row_chunk_size, spec.col_chunk_size, spec.particle_chunk_size) == chunks
        assert spec.legacy_module_fd_step == fd
        assert spec.sinkhorn_epsilon == epsilon
        _payload, source = _source_identity(spec)
        assert source["row_id"] == spec.row_id
        assert tuple(source["forward_contract"]["theta_contract"]["parameter_order"]) == spec.parameter_names


def test_gate_a_parser_requires_opt_in_then_preserves_historical_runtime_defaults() -> None:
    argv = [
        "--row",
        "predator-prey",
        "--stage",
        "score-only",
        "--output",
        "/tmp/score.json",
    ]
    with pytest.raises(ValueError, match="historical raw-barycentric diagnostic"):
        harness._parse_args(argv)  # noqa: SLF001
    args = harness._parse_args(  # noqa: SLF001
        [*argv, "--historical-raw-diagnostic"]
    )
    spec = harness.ROW_SPECS["predator-prey"]
    assert args.jit_compile is True
    assert args.dtype == "float32"
    assert args.tf32_mode == "enabled"
    assert args.batch_seeds == [81120]
    assert args.time_steps == spec.full_time_steps
    assert args.num_particles == spec.full_num_particles
    assert args.memory_budget_mib == harness.MEMORY_BUDGET_MIB
    assert args.source_value_artifact == spec.source_value_artifact


def test_phase9_historical_exact_command_manifest_is_current_but_superseded() -> None:
    path = harness.ROOT / harness.EXACT_COMMANDS_PATH
    frozen = json.loads(path.read_text(encoding="utf-8"))
    generated = command_manifest.build_manifest()
    assert frozen == generated
    assert len(frozen["gate_b_commands"]) == 10
    assert len(frozen["gate_c_commands"]) == 36
    assert len(frozen["gate_d_commands"]) == 40
    assert len(frozen["aggregate_commands"]) == 5

    for command in frozen["gate_b_commands"] + frozen["gate_c_commands"] + frozen["gate_d_commands"]:
        argv = command["argv"]
        assert argv[:2] == [command_manifest.PYTHON, command_manifest.RUNNER]
        with pytest.raises(ValueError, match="historical raw-barycentric diagnostic"):
            harness._parse_args(argv[2:])  # noqa: SLF001
        parsed = harness._parse_args(argv[2:], validate=False)  # noqa: SLF001
        assert parsed.stage == command["stage"]
        assert parsed.batch_seeds == [command["seed"]]
        assert parsed.time_steps == command["time_steps"]
        assert parsed.num_particles == command["num_particles"]
        assert parsed.output == command["output"]
        if parsed.stage == "fd-only":
            assert parsed.score_reference_json == command["score_reference_json"]
        else:
            assert command["score_reference_json"] is None
        with pytest.raises(ValueError, match="historical Phase 9 exact-command manifest is superseded"):
            harness._validate_exact_execution_command(parsed, harness.ROW_SPECS[parsed.row])  # noqa: SLF001

    for command in frozen["aggregate_commands"]:
        argv = command["argv"]
        with pytest.raises(ValueError, match="historical raw-barycentric diagnostic"):
            harness._parse_args(argv[2:])  # noqa: SLF001
        parsed = harness._parse_args(argv[2:], validate=False)  # noqa: SLF001
        assert parsed.stage == "aggregate"
        assert parsed.device_scope == "cpu"
        assert parsed.expect_device_kind == "cpu"
        assert parsed.batch_seeds == list(harness.FULL_ROW_BATCH_SEEDS)
        assert parsed.score_shards == command["score_shards"]
        assert parsed.fd_shards == command["fd_shards"]
        with pytest.raises(ValueError, match="historical Phase 9 exact-command manifest is superseded"):
            harness._validate_exact_execution_command(parsed, harness.ROW_SPECS[parsed.row])  # noqa: SLF001


def test_complete_highdim_exact_command_manifest_is_preserved_and_superseded(
) -> None:
    path = harness.ROOT / harness.COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH
    frozen = json.loads(path.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="historical raw-barycentric diagnostic"):
        complete_commands.build_manifest()
    assert frozen["status"] == "frozen_not_execution_authority"
    assert frozen["command_count"] == 96
    assert len(frozen["phase2_commands"]) == 42
    assert len(frozen["phase3_commands"]) == 48
    assert len(frozen["aggregate_commands"]) == 6
    assert frozen["execution_authority_status_at_freeze"] == (
        "absent_required_before_each_phase"
    )

    commands = [
        *frozen["phase2_commands"],
        *frozen["phase3_commands"],
        *frozen["aggregate_commands"],
    ]
    assert len({command["output"] for command in commands}) == len(commands)
    assert len({command["exact_command_sha256"] for command in commands}) == len(
        commands
    )
    for command in commands:
        parsed = harness._parse_args(command["argv"][2:], validate=False)  # noqa: SLF001
        assert parsed.row == command["row"]
        assert parsed.stage == command["stage"]
        assert parsed.time_steps == command["time_steps"]
        assert parsed.num_particles == command["num_particles"]
        assert parsed.output == command["output"]
        assert parsed.command_timeout_seconds == command["command_timeout_seconds"]
        with pytest.raises(ValueError, match="historical raw-barycentric diagnostic"):
            harness._validate_args(parsed, harness.ROW_SPECS[parsed.row])  # noqa: SLF001


@pytest.mark.parametrize("field", ("output", "timeout", "argv_order"))
def test_complete_highdim_exact_command_gate_rejects_drift(field: str) -> None:
    frozen = json.loads(
        (harness.ROOT / harness.COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH).read_text(
            encoding="utf-8"
        )
    )
    command = copy.deepcopy(frozen["phase2_commands"][0])
    argv = list(command["argv"][1:])
    if field == "output":
        argv[argv.index("--output") + 1] = "/tmp/unreviewed-output.json"
    elif field == "timeout":
        argv[argv.index("--command-timeout-seconds") + 1] = "999"
    else:
        row_index = argv.index("--row")
        row_pair = argv[row_index : row_index + 2]
        del argv[row_index : row_index + 2]
        argv.extend(row_pair)
    parsed = harness._parse_args(argv[1:], validate=False)  # noqa: SLF001

    with pytest.raises(
        ValueError,
        match="not an exact reviewed Phase 9 command|complete-highdim command freeze",
    ):
        harness._validate_exact_execution_command(  # noqa: SLF001
            parsed,
            harness.ROW_SPECS[parsed.row],
            argv,
        )


def test_root_cause_repair_exact_commands_are_now_superseded_without_opt_in() -> None:
    commands = harness._root_cause_repair_execution_commands()  # noqa: SLF001
    assert len(commands) == 4
    assert {command["row"] for command in commands} == {"predator-prey", "generalized-sv"}
    for command in commands:
        argv = list(command["argv"])
        with pytest.raises(ValueError, match="historical raw-barycentric diagnostic"):
            harness._parse_args(argv[2:])  # noqa: SLF001


def test_root_cause_repair_gate_rejects_shape_or_argv_drift() -> None:
    command = harness._root_cause_repair_execution_commands()[0]  # noqa: SLF001
    argv = list(command["argv"])
    drifted = list(argv)
    drifted[drifted.index("--output") + 1] = "/tmp/unreviewed-repair.json"
    parsed = harness._parse_args(drifted[2:], validate=False)  # noqa: SLF001
    with pytest.raises(ValueError, match="not an exact reviewed"):
        harness._validate_exact_execution_command(  # noqa: SLF001
            parsed,
            harness.ROW_SPECS[parsed.row],
            drifted[1:],
        )

    reordered = list(argv[1:])
    row_index = reordered.index("--row")
    row_pair = reordered[row_index : row_index + 2]
    del reordered[row_index : row_index + 2]
    reordered.extend(row_pair)
    parsed = harness._parse_args(reordered[1:], validate=False)  # noqa: SLF001
    with pytest.raises(ValueError, match="reviewed repair command"):
        harness._validate_exact_execution_command(  # noqa: SLF001
            parsed,
            harness.ROW_SPECS[parsed.row],
            reordered,
        )


def test_phase9_exact_command_gate_rejects_unreviewed_gpu_output() -> None:
    command = command_manifest.build_manifest()["gate_b_commands"][0]
    argv = list(command["argv"][2:])
    argv[argv.index("--output") + 1] = "/tmp/unreviewed-score.json"
    args = harness._parse_args(argv, validate=False)  # noqa: SLF001

    with pytest.raises(ValueError, match="not an exact reviewed Phase 9 command"):
        harness._validate_exact_execution_command(args, harness.ROW_SPECS[args.row])  # noqa: SLF001


def test_phase9_exact_command_gate_rejects_unreviewed_cuda_visibility() -> None:
    command = command_manifest.build_manifest()["gate_b_commands"][0]
    argv = list(command["argv"][2:])
    argv[argv.index("--cuda-visible-devices") + 1] = "1"
    args = harness._parse_args(argv, validate=False)  # noqa: SLF001

    with pytest.raises(ValueError, match="not an exact reviewed Phase 9 command"):
        harness._validate_exact_execution_command(args, harness.ROW_SPECS[args.row])  # noqa: SLF001


def test_phase9_exact_command_gate_rejects_parser_equivalent_reordered_argv() -> None:
    command = command_manifest.build_manifest()["gate_b_commands"][0]
    argv = list(command["argv"][1:])
    row_index = argv.index("--row")
    row_pair = argv[row_index : row_index + 2]
    del argv[row_index : row_index + 2]
    argv.extend(row_pair)
    args = harness._parse_args(argv[1:], validate=False)  # noqa: SLF001

    with pytest.raises(ValueError, match="argv does not exactly match"):
        harness._validate_exact_execution_command(  # noqa: SLF001
            args,
            harness.ROW_SPECS[args.row],
            argv,
        )


def test_gate_a_governance_and_code_hashes_are_frozen_for_process_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = harness.ROW_SPECS["predator-prey"]
    governance = harness._governance_artifact_sha256()  # noqa: SLF001
    code = harness._code_source_sha256(spec)  # noqa: SLF001

    monkeypatch.setattr(
        harness,
        "_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("hash unexpectedly recomputed")),
    )

    assert harness._governance_artifact_sha256() == governance  # noqa: SLF001
    assert harness._code_source_sha256(spec) == code  # noqa: SLF001


@pytest.mark.parametrize(
    ("extra", "match"),
    (
        (["--batch-seeds", "81120,81121"], "exactly one seed"),
        (["--batch-seeds", "999"], "frozen full-row seed set"),
        (["--memory-budget-mib", "15000"], "frozen 14000 MiB"),
        (["--sinkhorn-iterations", "9"], "frozen admitted transport"),
        (["--row-chunk-size", "256"], "frozen admitted transport"),
        (["--source-value-artifact", "wrong.json"], "frozen admitted source"),
        (["--device-scope", "cpu"], "CPU scope cannot claim GPU"),
    ),
)
def test_gate_a_parser_rejects_unreviewed_runtime_evidence_policy(extra, match) -> None:
    with pytest.raises(ValueError, match=match):
        harness._parse_args(  # noqa: SLF001
            [
                "--row",
                "predator-prey",
                "--stage",
                "score-only",
                "--output",
                "/tmp/score.json",
                *extra,
            ]
        )


def test_gate_a_compiled_entry_points_hard_code_xla_without_autodiff() -> None:
    source = "\n".join(
        (
            inspect.getsource(harness._compiled_score),  # noqa: SLF001
            inspect.getsource(harness._compiled_value),  # noqa: SLF001
        )
    )
    parser_source = inspect.getsource(harness._parse_args)  # noqa: SLF001

    assert source.count("jit_compile=True") == 2
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source
    assert "--no-jit-compile" not in parser_source


def test_gate_b_fixed_sir_parameterized_callbacks_are_tensor_only() -> None:
    source = inspect.getsource(p8p._make_sir_callbacks_from_scaled_parameters)  # noqa: SLF001

    assert "_dpf_sir_callbacks" not in source
    assert ".numpy" not in source
    assert 'tensors["transition_covariance"]' in source


def test_gate_b_fixed_sir_value_adapter_xla_matches_eager_objective() -> None:
    spec = harness.ROW_SPECS["fixed-sir"]
    args = _tiny_args("fixed-sir")
    theta = tf.constant(spec.truth_theta, dtype=tf.float64)
    prepared = spec.module._prepare_compact_xla_inputs(args)  # noqa: SLF001
    compiled = harness._compiled_value(spec, args, prepared)  # noqa: SLF001

    compiled_objective, compiled_log_likelihood = compiled(theta)
    eager_objective = spec.module._value_objective_from_components(  # noqa: SLF001
        args,
        theta,
        prepared_tensors=prepared["tensors"],
    )

    tf.debugging.assert_near(compiled_objective, eager_objective, atol=1.0e-10, rtol=1.0e-10)
    tf.debugging.assert_near(
        compiled_objective,
        tf.reduce_mean(compiled_log_likelihood),
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_gate_b_predator_prey_transition_helpers_are_graph_safe() -> None:
    module = harness.ROW_SPECS["predator-prey"].module
    source = "\n".join(
        (
            inspect.getsource(module._predator_prey_transition_mean_with_aux_tf),  # noqa: SLF001
            inspect.getsource(module._predator_prey_transition_mean_vjp_tf),  # noqa: SLF001
            inspect.getsource(module._predator_prey_transition_mean_jvp_tf),  # noqa: SLF001
        )
    )

    assert "p30_predator_prey_fixture_model" not in source
    assert ".numpy" not in source
    assert "_PREDATOR_PREY_RK4_SUBSTEPS" in source
    assert "_PREDATOR_PREY_DELTA" in source


@pytest.mark.parametrize("row", tuple(harness.ROW_SPECS))
def test_gate_b_all_tensor_adapters_compile_with_xla(row: str) -> None:
    spec = harness.ROW_SPECS[row]
    args = _tiny_args(row)
    theta = tf.constant(spec.truth_theta, dtype=tf.float64)
    prepared = spec.module._prepare_compact_xla_inputs(args)  # noqa: SLF001
    eager_score = spec.module._compact_score_tensor_outputs(args, theta, prepared)  # noqa: SLF001
    eager_value = spec.module._value_tensor_outputs(args, theta, prepared)  # noqa: SLF001

    compiled_score = harness._compiled_score(spec, args, prepared)(theta)  # noqa: SLF001
    compiled_value = harness._compiled_value(spec, args, prepared)(theta)  # noqa: SLF001

    for actual, expected in zip(compiled_score, eager_score, strict=True):
        tf.debugging.assert_near(actual, expected, atol=1.0e-10, rtol=1.0e-10)
    for actual, expected in zip(compiled_value, eager_value, strict=True):
        tf.debugging.assert_near(actual, expected, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize("row", tuple(harness.ROW_SPECS))
def test_gate_a_prepared_tensor_entry_points_match_existing_eager_routes(row: str) -> None:
    spec = harness.ROW_SPECS[row]
    args = _tiny_args(row)
    theta = tf.constant(spec.truth_theta, dtype=tf.float64)

    legacy_score = spec.module._compact_value_and_score_from_components(  # noqa: SLF001
        args,
        list(spec.truth_theta),
    )
    prepared = spec.module._prepare_compact_xla_inputs(args)  # noqa: SLF001
    score_outputs = spec.module._compact_score_tensor_outputs(args, theta, prepared)  # noqa: SLF001

    tf.debugging.assert_near(score_outputs[0], legacy_score["objective"], atol=1.0e-10, rtol=1.0e-10)
    tf.debugging.assert_near(score_outputs[1], legacy_score["log_likelihood"], atol=1.0e-10, rtol=1.0e-10)
    tf.debugging.assert_near(score_outputs[2], legacy_score["gradient_tensor"], atol=1.0e-10, rtol=1.0e-10)
    tf.debugging.assert_near(score_outputs[3], legacy_score["per_seed_gradient"], atol=1.0e-10, rtol=1.0e-10)

    value_outputs = spec.module._value_tensor_outputs(args, theta, prepared)  # noqa: SLF001
    if row == "fixed-sir":
        legacy_objective = spec.module._value_objective_from_components(  # noqa: SLF001
            args,
            list(spec.truth_theta),
        )
        tf.debugging.assert_near(value_outputs[0], legacy_objective, atol=1.0e-10, rtol=1.0e-10)
    else:
        legacy_value = spec.module._manual_value_only_from_components(  # noqa: SLF001
            args,
            list(spec.truth_theta),
        )
        tf.debugging.assert_near(value_outputs[0], legacy_value["objective"], atol=1.0e-10, rtol=1.0e-10)
        tf.debugging.assert_near(value_outputs[1], legacy_value["log_likelihood"], atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize("row", tuple(harness.ROW_SPECS))
def test_gate_a_row_preparation_rejects_multi_seed_runtime_shards(row: str) -> None:
    args = _tiny_args(row)
    args.batch_seeds = [81120, 81121]
    with pytest.raises(ValueError, match="exactly one seed"):
        harness.ROW_SPECS[row].module._prepare_compact_xla_inputs(args)  # noqa: SLF001


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda payload: payload["run_manifest"].__setitem__("jit_compile", False), "jit_compile"),
        (lambda payload: payload.__setitem__("evidence_class", "forged"), "trust basis"),
        (lambda payload: payload.__setitem__("score_route", "historical_manual"), "row or compact"),
        (lambda payload: payload.__setitem__("source_value_artifact", "wrong.json"), "source value"),
        (lambda payload: payload.__setitem__("target_observation_policy", "wrong"), "target observation"),
        (lambda payload: payload["precision"].__setitem__("tf32_execution_enabled", False), "precision"),
        (lambda payload: payload.__setitem__("score_output_devices", ["/CPU:0"]), "GPU outputs"),
        (lambda payload: payload.__setitem__("score_gpu_memory_stats_reset", False), "reset"),
        (lambda payload: payload.__setitem__("score_admission_status", LEDH_SCORE_ADMISSION_STATUS_FULL), "explicitly historical"),
        (
            lambda payload: payload["run_manifest"].__setitem__(
                "batch_seeds", [81120, 81121]
            ),
            "command identity|exactly one seed",
        ),
        (lambda payload: payload.__setitem__("source_value_artifact_sha256", "0" * 64), "artifact hash"),
        (lambda payload: payload.__setitem__("physical_gpus", []), "physical and logical"),
        (lambda payload: payload.__setitem__("logical_gpus", []), "physical and logical"),
        (lambda payload: payload["score_gpu_memory_info_after"].pop("peak"), "peak memory"),
        (lambda payload: payload["score_gpu_memory_info_after"].__setitem__("peak", float("inf")), "peak memory"),
        (lambda payload: payload["run_manifest"].__setitem__("transport_plan_mode", "dense"), "transport_plan_mode"),
        (lambda payload: payload["run_manifest"].__setitem__("transport_ad_mode", "stabilized"), "transport_ad_mode"),
        (lambda payload: payload["run_manifest"].__setitem__("transport_gradient_mode", "historical"), "transport_gradient_mode"),
        (lambda payload: payload["run_manifest"].__setitem__("annealed_scaling", 0.8), "annealed_scaling"),
        (
            lambda payload: payload["run_manifest"].__setitem__("annealed_convergence_threshold", 2.0e-3),
            "annealed_convergence_threshold",
        ),
        (lambda payload: payload["run_manifest"].__setitem__("memory_budget_mib", 15000.0), "memory_budget_mib"),
        (
            lambda payload: payload["run_manifest"].__setitem__("device_scope", "cpu"),
            "command identity|device provenance",
        ),
        (lambda payload: payload["run_manifest"].__setitem__("git_commit", "short"), "git_commit"),
        (
            lambda payload: payload["run_manifest"].__setitem__("git_commit", "0" * 40),
            "current HEAD",
        ),
        (
            lambda payload: payload["run_manifest"]["code_source_sha256"].__setitem__(
                "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py",
                "0" * 64,
            ),
            "code source hashes",
        ),
        (
            lambda payload: payload["run_manifest"]["governance_artifact_sha256"].__setitem__(
                harness.GATE_B_RESULT_PATH,
                "0" * 64,
            ),
            "governance artifact hashes",
        ),
        (
            lambda payload: payload["run_manifest"]["governance_artifact_sha256"].__setitem__(
                harness.GATE_B_RESULT_REVIEW_PATH,
                "0" * 64,
            ),
            "governance artifact hashes",
        ),
        (
            lambda payload: payload["run_manifest"].__setitem__(
                "gate_b_repair_review_path",
                "docs/reviews/unreviewed-repair.md",
            ),
            "gate_b_repair_review_path",
        ),
        (
            lambda payload: payload["run_manifest"].__setitem__(
                "gate_b_result_path",
                "docs/plans/unreviewed-gate-b-result.md",
            ),
            "gate_b_result_path",
        ),
        (
            lambda payload: payload["run_manifest"].__setitem__(
                "gate_b_result_review_path",
                "docs/reviews/unreviewed-gate-b-result.md",
            ),
            "gate_b_result_review_path",
        ),
        (lambda payload: payload["run_manifest"].pop("command"), "command"),
        (lambda payload: payload.__setitem__("truth_theta", [99.0, 99.0, 99.0]), "truth theta"),
        (
            lambda payload: payload.__setitem__("canonical_target_sha256", "0" * 64),
            "canonical target signature",
        ),
        (
            lambda payload: _increment(payload["score_evaluation_theta"], 0, 1.0),
            "score evaluation theta",
        ),
        (
            lambda payload: payload["configuration_identity"].__setitem__("sha256", "0" * 64),
            "configuration identity",
        ),
        (
            lambda payload: payload["route_identity"].__setitem__("sha256", "0" * 64),
            "route identity",
        ),
        (
            lambda payload: payload["randomness_identity"].__setitem__("sha256", "0" * 64),
            "randomness identity",
        ),
        (
            lambda payload: payload["run_manifest"].__setitem__("command", "forged command"),
            "command string/argv",
        ),
        (lambda payload: payload.__setitem__("objective", 2.0), "objective must equal"),
        (
            lambda payload: payload.__setitem__("total_log_likelihood", 2.0),
            "paired total value",
        ),
        (
            lambda payload: payload["prepared_input_fingerprint"].__setitem__(
                "aggregate_sha256",
                "0" * 64,
            ),
            "aggregate SHA-256 mismatch",
        ),
    ),
)
def test_gate_a_score_shard_validator_rejects_forged_evidence(mutation, match) -> None:
    args = _parse_aggregate_args("predator-prey")
    spec = harness.ROW_SPECS["predator-prey"]
    payload = _score_shard(args, spec, 81120)
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        harness._validate_raw_score_shard(  # noqa: SLF001
            payload,
            args,
            spec,
            require_gpu=True,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda payload: payload["score_correctness"].__setitem__("status", "fail"), "must pass"),
        (
            lambda payload: payload["score_correctness"]["step_policy"].__setitem__(
                "coefficient",
                2.0e-4,
            ),
            "step policy mismatch",
        ),
        (
            lambda payload: payload["score_correctness"]["finite_difference_diagnostics"][0].__setitem__(
                "nominal_step",
                2.0e-4,
            ),
            "nominal_step mismatch",
        ),
        (
            lambda payload: payload["score_correctness"]["finite_difference_diagnostics"][0].__setitem__(
                "plus_objective",
                99.0,
            ),
            "total log likelihood mismatch|objective numerator mismatch",
        ),
        (
            lambda payload: payload["score_correctness"]["fd_policy"].__setitem__(
                "max_coordinate_relative_error",
                0.01,
            ),
            "does not match recomputed",
        ),
        (lambda payload: payload["score_correctness"].__setitem__("uses_value_only_scalar_route", False), "value-only"),
        (lambda payload: payload.__setitem__("score_derivative_provenance", "historical"), "provenance"),
        (lambda payload: payload.__setitem__("value_output_devices", ["/CPU:0"]), "GPU value"),
        (
            lambda payload: _increment(
                payload["score_correctness"]["finite_difference_diagnostics"][0]["center_theta"],
                0,
                1.0,
            ),
            "center theta mismatch",
        ),
        (
            lambda payload: payload["score_correctness"]["finite_difference_diagnostics"][0][
                "minus_endpoint"
            ].__setitem__("role", "plus"),
            "minus endpoint role mismatch",
        ),
        (
            lambda payload: payload["score_correctness"]["finite_difference_diagnostics"][0][
                "plus_endpoint"
            ].__setitem__("role", "minus"),
            "plus endpoint role mismatch",
        ),
        (
            lambda payload: payload["score_correctness"]["finite_difference_diagnostics"][0].__setitem__(
                "direction_index", 1
            ),
            "direction index mismatch",
        ),
        (
            lambda payload: _increment(
                payload["score_correctness"]["finite_difference_diagnostics"][0]["plus_endpoint"]["theta"],
                0,
                1.0e-2,
            ),
            "plus endpoint theta mismatch",
        ),
        (
            lambda payload: _increment(
                payload["score_correctness"]["finite_difference_diagnostics"][0]["minus_endpoint"]["theta"],
                1,
                1.0e-2,
            ),
            "minus endpoint theta mismatch",
        ),
        (
            lambda payload: payload["score_correctness"]["finite_difference_diagnostics"][0][
                "plus_endpoint"
            ].__setitem__("theta", list(payload["score_correctness"]["finite_difference_diagnostics"][0]["minus_endpoint"]["theta"])),
            "plus endpoint theta mismatch",
        ),
        (
            lambda payload: payload["score_correctness"]["finite_difference_diagnostics"][0][
                "plus_endpoint"
            ].__setitem__("total_log_likelihood", float("inf")),
            "must be finite",
        ),
        (
            lambda payload: payload["configuration_identity"].__setitem__("sha256", "0" * 64),
            "configuration identity",
        ),
        (
            lambda payload: payload["route_identity"].__setitem__("sha256", "0" * 64),
            "route identity",
        ),
        (
            lambda payload: payload["randomness_identity"].__setitem__("sha256", "0" * 64),
            "randomness identity",
        ),
    ),
)
def test_gate_a_fd_shard_validator_rejects_wrong_comparator_evidence(mutation, match) -> None:
    args = _parse_aggregate_args("predator-prey")
    spec = harness.ROW_SPECS["predator-prey"]
    score = [float(index + 1) for index in range(len(spec.parameter_names))]
    payload = _fd_shard(args, spec, 81120, score, "score-hash")
    mutation(payload)
    with pytest.raises(ValueError, match=match):
        harness._validate_raw_fd_shard(  # noqa: SLF001
            payload,
            args,
            spec,
            require_gpu=True,
        )


def test_gate_a_fd_validator_recomputes_pass_instead_of_trusting_label() -> None:
    args = _parse_aggregate_args("predator-prey")
    spec = harness.ROW_SPECS["predator-prey"]
    score = [float(index + 1) for index in range(len(spec.parameter_names))]
    payload = _fd_shard(args, spec, 81120, score, "score-hash")
    policy = payload["score_correctness"]["fd_policy"]
    entry = policy["parameters"][0]
    entry["finite_difference"] = entry["score"] + 1.0

    with pytest.raises(ValueError, match="endpoint diagnostic finite difference mismatch"):
        harness._validate_raw_fd_shard(  # noqa: SLF001
            payload,
            args,
            spec,
            require_gpu=True,
        )


def test_score_reference_uses_its_own_stage_specific_timeout(
    tmp_path: Path,
) -> None:
    spec = harness.ROW_SPECS["fixed-sir"]
    score_path = tmp_path / "score.json"
    fd_path = tmp_path / "fd.json"
    score_argv = [
        "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py",
        "--row",
        "fixed-sir",
        "--stage",
        "score-only",
        "--batch-seeds",
        "81120",
        "--time-steps",
        "1",
        "--num-particles",
        "10000",
        "--device-scope",
        "visible",
        "--cuda-visible-devices",
        "0",
        "--device",
        "/GPU:0",
        "--expect-device-kind",
        "gpu",
        "--command-timeout-seconds",
        "900",
        "--historical-raw-diagnostic",
        "--output",
        str(score_path),
    ]
    score_args = harness._parse_args(score_argv[1:])  # noqa: SLF001
    score_payload = _score_shard(score_args, spec, 81120)
    score_payload["memory_diagnostics"]["full_row_memory_gate_applicable"] = False
    score_payload["run_manifest"]["command_argv"] = score_argv
    score_payload["run_manifest"]["output"] = str(score_path)
    _refresh_command_identity(score_payload)
    _rewrite_shard(score_path, score_payload)
    fd_args = harness._parse_args(  # noqa: SLF001
        [
            "--row",
            "fixed-sir",
            "--stage",
            "fd-only",
            "--batch-seeds",
            "81120",
            "--time-steps",
            "1",
            "--num-particles",
            "10000",
            "--device-scope",
            "visible",
            "--cuda-visible-devices",
            "0",
            "--device",
            "/GPU:0",
            "--expect-device-kind",
            "gpu",
            "--command-timeout-seconds",
            "1200",
            "--historical-raw-diagnostic",
            "--score-reference-json",
            str(score_path),
            "--output",
            str(fd_path),
        ]
    )

    assert harness._load_score_reference(str(score_path), fd_args, spec) == score_payload  # noqa: SLF001

    fd_args.time_steps = 5
    with pytest.raises(ValueError, match="time_steps does not match"):
        harness._load_score_reference(str(score_path), fd_args, spec)  # noqa: SLF001


@pytest.mark.parametrize("row", tuple(harness.ROW_SPECS))
def test_gate_a_prepared_input_fingerprint_is_deterministic(row: str) -> None:
    spec = harness.ROW_SPECS[row]
    args = _tiny_args(row)

    first = spec.module._prepare_compact_xla_inputs(args)  # noqa: SLF001
    second = spec.module._prepare_compact_xla_inputs(args)  # noqa: SLF001

    assert harness._prepared_input_fingerprint(first) == harness._prepared_input_fingerprint(second)  # noqa: SLF001


def _write_complete_shards(tmp_path: Path, row: str) -> tuple[Namespace, list[str], list[str]]:
    spec = harness.ROW_SPECS[row]
    score_paths = []
    fd_paths = []
    placeholder_args = _parse_aggregate_args(row)
    for seed in harness.FULL_ROW_BATCH_SEEDS:
        score_path = tmp_path / f"score-{seed}.json"
        score_payload = _score_shard(placeholder_args, spec, seed)
        score_payload["run_manifest"]["output"] = str(score_path)
        score_payload["run_manifest"]["command_argv"][-1] = str(score_path)
        score_payload["run_manifest"]["command"] = harness.shlex.join(
            score_payload["run_manifest"]["command_argv"]
        )
        score_payload["run_manifest"]["command_identity"] = (
            harness._command_identity_from_manifest(score_payload["run_manifest"])  # noqa: SLF001
        )
        score_path.write_text(json.dumps(score_payload), encoding="utf-8")
        score_hash = harness._sha256(score_path)  # noqa: SLF001
        fd_path = tmp_path / f"fd-{seed}.json"
        fd_payload = _fd_shard(
            placeholder_args,
            spec,
            seed,
            list(score_payload["score"]),
            score_hash,
        )
        fd_payload["run_manifest"]["output"] = str(fd_path)
        fd_payload["run_manifest"]["command_argv"][-1] = str(fd_path)
        fd_payload["run_manifest"]["command"] = harness.shlex.join(
            fd_payload["run_manifest"]["command_argv"]
        )
        fd_payload["run_manifest"]["command_identity"] = (
            harness._command_identity_from_manifest(fd_payload["run_manifest"])  # noqa: SLF001
        )
        fd_path.write_text(json.dumps(fd_payload), encoding="utf-8")
        score_paths.append(str(score_path))
        fd_paths.append(str(fd_path))
    return _parse_aggregate_args(row, score_paths=score_paths, fd_paths=fd_paths), score_paths, fd_paths


@pytest.mark.parametrize("row", tuple(harness.ROW_SPECS))
def test_gate_a_offline_aggregate_preserves_historical_status_for_valid_shards(
    tmp_path: Path,
    row: str,
) -> None:
    args, _score_paths, _fd_paths = _write_complete_shards(tmp_path, row)
    spec = harness.ROW_SPECS[row]

    result = harness._aggregate(args, spec, 0.0)  # noqa: SLF001

    assert result["artifact_status"] == "completed"
    assert result["evidence_class"] == "offline_aggregate_of_validated_trusted_gpu_shards"
    assert result["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW
    assert (
        result["score_artifact"]["score_admission_status"]
        == LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW
    )
    assert result["memory_diagnostics"]["n10000_memory_pass"] is True
    assert result["execution_strategy"]["monolithic_batch_memory_claim"] is False
    assert result["execution_strategy"]["monolithic_batch_runtime_claim"] is False
    assert [record["seed"] for record in result["per_seed_records"]] == list(
        harness.FULL_ROW_BATCH_SEEDS
    )
    assert all(
        len(record["finite_difference_diagnostics"]) == len(spec.parameter_names)
        for record in result["per_seed_records"]
    )


def test_gate_a_aggregate_rejects_fd_bound_to_different_score_shard(tmp_path: Path) -> None:
    args, _score_paths, fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    first_fd = Path(fd_paths[0])
    payload = json.loads(first_fd.read_text(encoding="utf-8"))
    payload["score_reference_sha256"] = "0" * 64
    first_fd.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="score-reference hash"):
        harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001


def test_gate_a_aggregate_rejects_fd_score_content_mismatch(tmp_path: Path) -> None:
    args, _score_paths, fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    first_fd = Path(fd_paths[0])
    payload = json.loads(first_fd.read_text(encoding="utf-8"))
    payload["score"][0] += 1.0
    first_fd.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="score does not match"):
        harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001


def test_gate_a_aggregate_rejects_fd_prepared_input_mismatch(tmp_path: Path) -> None:
    args, _score_paths, fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    first_fd = Path(fd_paths[0])
    payload = json.loads(first_fd.read_text(encoding="utf-8"))
    payload["prepared_input_fingerprint"] = harness._prepared_input_fingerprint(  # noqa: SLF001
        {"tensors": {"fixture": tf.constant([-1.0], dtype=tf.float32)}}
    )
    first_fd.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="randomness identity mismatch|prepared inputs do not match",
    ):
        harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001


@pytest.mark.parametrize("kind", ("missing", "duplicate"))
def test_gate_a_aggregate_rejects_incomplete_or_duplicate_seed_shards(tmp_path: Path, kind: str) -> None:
    args, score_paths, _fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    args.score_shards = score_paths[:-1] if kind == "missing" else [*score_paths[:-1], score_paths[0]]

    with pytest.raises(ValueError, match="missing shard seeds" if kind == "missing" else "duplicate shard seed"):
        harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001


@pytest.mark.parametrize("kind", ("extra", "substituted"))
def test_gate_a_aggregate_rejects_extra_or_substituted_seed_set(
    tmp_path: Path,
    kind: str,
) -> None:
    args, score_paths, _fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    source_path = Path(score_paths[-1])
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["run_manifest"]["batch_seeds"] = [81125]
    seed_index = payload["run_manifest"]["command_argv"].index("--batch-seeds") + 1
    payload["run_manifest"]["command_argv"][seed_index] = "81125"
    payload["randomness_identity"] = harness._randomness_identity(  # noqa: SLF001
        seed=81125,
        prepared_input_fingerprint=payload["prepared_input_fingerprint"],
        configuration_identity=payload["configuration_identity"],
        route_identity=payload["route_identity"],
    )
    replacement_path = tmp_path / "score-81125.json"
    payload["run_manifest"]["output"] = str(replacement_path)
    payload["run_manifest"]["command_argv"][-1] = str(replacement_path)
    _refresh_command_identity(payload)
    _rewrite_shard(replacement_path, payload)
    args.score_shards = (
        [*score_paths, str(replacement_path)]
        if kind == "extra"
        else [*score_paths[:-1], str(replacement_path)]
    )

    with pytest.raises(
        ValueError,
        match="outside the frozen full-row seed set|unexpected shard seed",
    ):
        harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001


def test_gate_a_aggregate_rejects_cross_row_score_substitution(tmp_path: Path) -> None:
    predator_path = tmp_path / "predator"
    fixed_path = tmp_path / "fixed"
    predator_path.mkdir()
    fixed_path.mkdir()
    args, score_paths, _fd_paths = _write_complete_shards(predator_path, "predator-prey")
    _, fixed_score_paths, _ = _write_complete_shards(fixed_path, "fixed-sir")
    args.score_shards[0] = fixed_score_paths[0]

    with pytest.raises(ValueError, match="row or compact provenance"):
        harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001


def test_gate_a_aggregate_rejects_cross_seed_command_template_drift(tmp_path: Path) -> None:
    args, score_paths, fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    path = Path(score_paths[0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_manifest"]["python_executable"] = "/tmp/alternate-python"
    _refresh_command_identity(payload)
    _rewrite_shard(path, payload)
    fd_payload = json.loads(Path(fd_paths[0]).read_text(encoding="utf-8"))
    fd_payload["score_reference_sha256"] = harness._sha256(path)  # noqa: SLF001
    _rewrite_shard(fd_paths[0], fd_payload)

    with pytest.raises(ValueError, match="command template family mismatch"):
        harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001


def test_gate_a_aggregate_rejects_output_path_collision(tmp_path: Path) -> None:
    args, score_paths, fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    path = Path(fd_paths[0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_manifest"]["output"] = score_paths[0]
    payload["run_manifest"]["command_argv"][-1] = score_paths[0]
    _refresh_command_identity(payload)
    _rewrite_shard(path, payload)

    with pytest.raises(ValueError, match="output path mismatch|output paths must be unique"):
        harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001


def test_gate_a_aggregate_rejects_failed_seed_even_when_mean_fd_passes(tmp_path: Path) -> None:
    args, _score_paths, fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    path = Path(fd_paths[0])
    payload = json.loads(path.read_text(encoding="utf-8"))
    failed_fd = _set_fd_direction(payload, 0, 0.5)
    _rewrite_shard(path, payload)
    aggregate_fd = statistics.fmean([failed_fd, 1.0, 1.0, 1.0, 1.0])
    aggregate_fds = [
        aggregate_fd,
        *[float(value) for value in payload["score"][1:]],
    ]
    assert evaluate_ledh_fd_policy(
        payload["score"],
        aggregate_fds,
        payload["score_parameter_names"],
    )["status"] == "pass"
    assert payload["score_correctness"]["fd_policy"]["status"] == "fail"

    with pytest.raises(ValueError, match="every FD shard must pass"):
        harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001


def test_gate_a_aggregate_preserves_total_not_average_value_scale(tmp_path: Path) -> None:
    args, score_paths, fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    totals = [20.0, 40.0, 60.0, 80.0, 100.0]
    for total, score_path, fd_path in zip(totals, score_paths, fd_paths, strict=True):
        score_payload = json.loads(Path(score_path).read_text(encoding="utf-8"))
        score_payload["objective"] = total
        score_payload["total_log_likelihood"] = total
        score_payload["log_likelihood_by_seed"] = [total]
        _rewrite_shard(score_path, score_payload)
        fd_payload = json.loads(Path(fd_path).read_text(encoding="utf-8"))
        fd_payload["score_reference_sha256"] = harness._sha256(Path(score_path))  # noqa: SLF001
        _rewrite_shard(fd_path, fd_payload)

    result = harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001

    assert result["total_log_likelihood"] == statistics.fmean(totals)
    assert result["average_log_likelihood"] == statistics.fmean(totals) / 20.0
    assert result["total_log_likelihood"] != result["average_log_likelihood"]
    assert [record["total_log_likelihood"] for record in result["per_seed_records"]] == totals


def test_gate_a_aggregate_blocks_score_peak_above_memory_budget(tmp_path: Path) -> None:
    args, score_paths, fd_paths = _write_complete_shards(tmp_path, "predator-prey")
    score_path = Path(score_paths[0])
    score_payload = json.loads(score_path.read_text(encoding="utf-8"))
    score_payload["score_gpu_memory_info_after"]["peak"] = int(
        (harness.MEMORY_BUDGET_MIB + 1.0) * 1024 * 1024
    )
    score_payload["memory_diagnostics"].update(
        {
            "score_memory_budget_pass": False,
            "n10000_memory_pass": False,
            "peak_mib": harness.MEMORY_BUDGET_MIB + 1.0,
        }
    )
    score_path.write_text(json.dumps(score_payload), encoding="utf-8")
    fd_path = Path(fd_paths[0])
    fd_payload = json.loads(fd_path.read_text(encoding="utf-8"))
    fd_payload["score_reference_sha256"] = harness._sha256(score_path)  # noqa: SLF001
    fd_path.write_text(json.dumps(fd_payload), encoding="utf-8")

    result = harness._aggregate(args, harness.ROW_SPECS["predator-prey"], 0.0)  # noqa: SLF001

    assert result["artifact_status"] == "blocked_memory_budget"
    assert result["score_admission_status"] == LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW
    assert result["memory_diagnostics"]["n10000_memory_pass"] is False


def test_gate_a_main_preserves_terminal_artifact_on_semantic_failure(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"
    with pytest.raises(ValueError, match="exactly one seed"):
        harness.main(
            [
                "--row",
                "predator-prey",
                "--stage",
                "score-only",
                "--batch-seeds",
                "81120,81121",
                "--output",
                str(output),
            ]
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_status"] == "failed"
    assert payload["terminal_artifact"] is True
    assert payload["error_type"] == "ValueError"


def test_gate_a_main_preserves_terminal_artifact_on_supervisor_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "interrupt.json"

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("bounded supervisor timeout")

    monkeypatch.setattr(harness, "_score_only", interrupt)
    monkeypatch.setattr(harness, "_validate_exact_execution_command", lambda *_args: None)
    with pytest.raises(KeyboardInterrupt, match="bounded supervisor timeout"):
        harness.main(
            [
                "--row",
                "predator-prey",
                "--stage",
                "score-only",
                "--historical-raw-diagnostic",
                "--output",
                str(output),
            ]
        )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_status"] == "failed"
    assert payload["terminal_artifact"] is True
    assert payload["error_type"] == "KeyboardInterrupt"


def test_gate_a_runner_uses_no_numpy_algorithm_backend() -> None:
    source = Path(harness.__file__).read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "np." not in source
    assert "@tf.function(jit_compile=True" in source
    assert "GradientTape" not in source
    assert "ForwardAccumulator" not in source
    assert p8p.core_tf.MANUAL_STREAMING_FINITE_TRANSPORT_GRADIENT_MODE == harness.COMPACT_GRADIENT_MODE

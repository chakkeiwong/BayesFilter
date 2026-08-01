"""Build the exact six-row LEDH Phase 2/3 command freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

from docs.benchmarks import benchmark_ledh_compact_score_gpu_xla as harness


PYTHON = "/home/chakwong/anaconda3/envs/tf-gpu/bin/python"
RUNNER = "docs/benchmarks/benchmark_ledh_compact_score_gpu_xla.py"
OUTPUT_PATH = harness.COMPLETE_HIGHDIM_EXACT_COMMANDS_PATH
RUN_ID = "complete-highdim-leaderboard-local-20260712-134906"
ARTIFACT_ROOT = "docs/plans/artifacts/complete-highdim-leaderboard"
LOG_ROOT = "docs/plans/logs/complete-highdim-leaderboard"
REVISION = "repair1"
SUPERSEDED_MANIFEST_PATH = (
    "docs/plans/complete-highdim-leaderboard-ledh-phase2-phase3-"
    "exact-commands-2026-07-11.json"
)
SEEDS = harness.FULL_ROW_BATCH_SEEDS
PHASE2_SEED = SEEDS[0]
PHASE3_SEEDS = SEEDS[1:]

PREFIXES = {
    "lgssm": (1, 10, 50),
    "fixed-sir": (1, 5, 20),
    "predator-prey": (1, 5, 20),
    "actual-sv": (4, 50, 250, 1000),
    "generalized-sv": (4, 50, 252, 1008),
    "ksc-sv": (4, 50, 250, 1000),
}

ENVIRONMENT_GPU = {
    "CONDA_DEFAULT_ENV": "tf-gpu",
    "CONDA_PREFIX": "/home/chakwong/anaconda3/envs/tf-gpu",
    "CUDA_VISIBLE_DEVICES": "0",
    "MPLCONFIGDIR": "/tmp",
    "PYTHONNOUSERSITE": "1",
}
ENVIRONMENT_CPU = {**ENVIRONMENT_GPU, "CUDA_VISIBLE_DEVICES": "-1"}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _timeout_seconds(row: str, time_steps: int, stage: str) -> int:
    full_time = harness.ROW_SPECS[row].full_time_steps
    if stage == "aggregate":
        return 300
    if time_steps == full_time:
        return 3600 if stage == "fd-only" else 2400
    return 1200 if stage == "fd-only" else 900


def _paths(
    phase: str,
    row: str,
    time_steps: int,
    seed: int,
    stage: str,
) -> tuple[str, str, str]:
    stem = f"{row}-t{time_steps}-n10000-seed{seed}-{stage}"
    directory = f"{ARTIFACT_ROOT}/{phase}-ledh-{REVISION}/{row}"
    log_directory = f"{LOG_ROOT}/{phase}-ledh-{REVISION}/{row}"
    return (
        f"{directory}/{stem}.json",
        f"{directory}/{stem}.md",
        f"{log_directory}/{stem}.log",
    )


def _args_for(
    *,
    row: str,
    stage: str,
    seed: int,
    time_steps: int,
    output: str,
    markdown: str,
    timeout_seconds: int,
    score_reference: str | None,
) -> list[str]:
    argv = [
        RUNNER,
        "--row",
        row,
        "--stage",
        stage,
        "--batch-seeds",
        str(seed),
        "--time-steps",
        str(time_steps),
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
        str(timeout_seconds),
    ]
    if score_reference is not None:
        argv.extend(("--score-reference-json", score_reference))
    argv.extend(("--output", output, "--markdown-output", markdown))
    return argv


def _command_identity(
    args: argparse.Namespace,
    spec: harness.RowSpec,
    invoked_argv: Sequence[str],
) -> dict[str, Any]:
    pseudo_manifest = {
        "command_argv": list(invoked_argv),
        "runner_path": RUNNER,
        "working_directory": str(ROOT),
        "python_executable": PYTHON,
        "row": spec.name,
        "row_id": spec.row_id,
        "stage": args.stage,
        "configuration_identity": harness._configuration_identity(args, spec),  # noqa: SLF001
        "route_identity": harness._route_identity(args, spec),  # noqa: SLF001
        "canonical_target_sha256": harness._canonical_target_sha256(spec),  # noqa: SLF001
        "fd_endpoint_contract": harness._fd_endpoint_contract(spec),  # noqa: SLF001
        "output": args.output,
        "markdown_output": args.markdown_output,
        "batch_seeds": list(args.batch_seeds),
        "score_reference_json": args.score_reference_json,
        "device_scope": args.device_scope,
        "cuda_visible_devices": (
            "-1" if args.device_scope == "cpu" else args.cuda_visible_devices
        ),
        "device": args.device,
        "expect_device_kind": args.expect_device_kind,
        "command_timeout_seconds": args.command_timeout_seconds,
    }
    return harness._command_identity_from_manifest(pseudo_manifest)  # noqa: SLF001


def _runtime_command(
    *,
    phase: str,
    row: str,
    seed: int,
    time_steps: int,
    stage: str,
) -> dict[str, Any]:
    output, markdown, log = _paths(phase, row, time_steps, seed, stage.removesuffix("-only"))
    timeout_seconds = _timeout_seconds(row, time_steps, stage)
    score_reference = None
    if stage == "fd-only":
        score_reference = _paths(phase, row, time_steps, seed, "score")[0]
    invoked_argv = _args_for(
        row=row,
        stage=stage,
        seed=seed,
        time_steps=time_steps,
        output=output,
        markdown=markdown,
        timeout_seconds=timeout_seconds,
        score_reference=score_reference,
    )
    args = harness._parse_args(invoked_argv[1:])  # noqa: SLF001
    spec = harness.ROW_SPECS[row]
    identity = _command_identity(args, spec, invoked_argv)
    argv = [PYTHON, *invoked_argv]
    supervised_argv = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30",
        str(timeout_seconds),
        "env",
        *[f"{key}={value}" for key, value in sorted(ENVIRONMENT_GPU.items())],
        *argv,
    ]
    return {
        "phase": phase,
        "row": row,
        "row_id": spec.row_id,
        "stage": stage,
        "seed": seed,
        "time_steps": time_steps,
        "num_particles": spec.full_num_particles,
        "score_reference_json": score_reference,
        "output": output,
        "markdown_output": markdown,
        "log": log,
        "working_directory": str(ROOT),
        "environment": dict(ENVIRONMENT_GPU),
        "python_executable": PYTHON,
        "conda_environment": "tf-gpu",
        "command_timeout_seconds": timeout_seconds,
        "timeout_enforcement": "external_supervisor_timeout_argv",
        "argv": argv,
        "invoked_argv": invoked_argv,
        "supervised_argv": supervised_argv,
        "shell_display": shlex.join(supervised_argv),
        "shell_command": (
            f"{shlex.join(supervised_argv)} > {shlex.quote(log)} 2>&1"
        ),
        "canonical_target_artifact": harness.CANONICAL_TARGETS_PATH,
        "canonical_target_artifact_sha256": harness.CANONICAL_TARGETS_SHA256,
        "canonical_target_sha256": harness._canonical_target_sha256(spec),  # noqa: SLF001
        "source_value_artifact": spec.source_value_artifact,
        "source_value_artifact_sha256": harness._source_value_sha256(spec),  # noqa: SLF001
        "configuration_identity": harness._configuration_identity(args, spec),  # noqa: SLF001
        "route_identity": harness._route_identity(args, spec),  # noqa: SLF001
        "fd_endpoint_contract": harness._fd_endpoint_contract(spec),  # noqa: SLF001
        "fd_policy": {
            "policy_id": harness.LEDH_FD_POLICY_ID,
            "diagnostic_scope": harness.LEDH_FD_DIAGNOSTIC_SCOPE,
            "pass_rule": harness.LEDH_FD_PASS_RULE,
            "step_policy": harness.ledh_fd_step_policy_metadata(),
        },
        "gpu_policy": {
            "trust_basis": harness.GPU_TRUST_BASIS,
            "device_scope": "visible",
            "cuda_visible_devices": "0",
            "device": "/GPU:0",
            "expect_device_kind": "gpu",
            "jit_compile": True,
            "dtype": "float32",
            "tf32_mode": "enabled",
        },
        "command_identity": identity,
        "template_family_sha256": identity["template_family_sha256"],
        "exact_command_sha256": identity["exact_command_sha256"],
    }


def _aggregate_command(row: str) -> dict[str, Any]:
    spec = harness.ROW_SPECS[row]
    timeout_seconds = _timeout_seconds(row, spec.full_time_steps, "aggregate")
    score_paths = [
        _paths(
            "phase2" if seed == PHASE2_SEED else "phase3",
            row,
            spec.full_time_steps,
            seed,
            "score",
        )[0]
        for seed in SEEDS
    ]
    fd_paths = [
        _paths(
            "phase2" if seed == PHASE2_SEED else "phase3",
            row,
            spec.full_time_steps,
            seed,
            "fd",
        )[0]
        for seed in SEEDS
    ]
    directory = f"{ARTIFACT_ROOT}/phase3-ledh-{REVISION}/{row}"
    output = f"{directory}/{row}-full-five-seed-aggregate.json"
    markdown = f"{directory}/{row}-full-five-seed-aggregate.md"
    log = (
        f"{LOG_ROOT}/phase3-ledh-{REVISION}/{row}/"
        f"{row}-full-five-seed-aggregate.log"
    )
    invoked_argv = [
        RUNNER,
        "--row",
        row,
        "--stage",
        "aggregate",
        "--batch-seeds",
        ",".join(str(seed) for seed in SEEDS),
        "--time-steps",
        str(spec.full_time_steps),
        "--num-particles",
        str(spec.full_num_particles),
        "--device-scope",
        "cpu",
        "--expect-device-kind",
        "cpu",
        "--command-timeout-seconds",
        str(timeout_seconds),
        "--score-shards",
        ",".join(score_paths),
        "--fd-shards",
        ",".join(fd_paths),
        "--output",
        output,
        "--markdown-output",
        markdown,
    ]
    args = harness._parse_args(invoked_argv[1:])  # noqa: SLF001
    identity = _command_identity(args, spec, invoked_argv)
    argv = [PYTHON, *invoked_argv]
    supervised_argv = [
        "timeout",
        "--signal=TERM",
        "--kill-after=30",
        str(timeout_seconds),
        "env",
        *[f"{key}={value}" for key, value in sorted(ENVIRONMENT_CPU.items())],
        *argv,
    ]
    return {
        "phase": "phase3",
        "row": row,
        "row_id": spec.row_id,
        "stage": "aggregate",
        "seeds": list(SEEDS),
        "time_steps": spec.full_time_steps,
        "num_particles": spec.full_num_particles,
        "score_reference_json": None,
        "score_shards": score_paths,
        "fd_shards": fd_paths,
        "output": output,
        "markdown_output": markdown,
        "log": log,
        "working_directory": str(ROOT),
        "environment": dict(ENVIRONMENT_CPU),
        "python_executable": PYTHON,
        "conda_environment": "tf-gpu",
        "command_timeout_seconds": timeout_seconds,
        "timeout_enforcement": "external_supervisor_timeout_argv",
        "argv": argv,
        "invoked_argv": invoked_argv,
        "supervised_argv": supervised_argv,
        "shell_display": shlex.join(supervised_argv),
        "shell_command": (
            f"{shlex.join(supervised_argv)} > {shlex.quote(log)} 2>&1"
        ),
        "canonical_target_artifact": harness.CANONICAL_TARGETS_PATH,
        "canonical_target_artifact_sha256": harness.CANONICAL_TARGETS_SHA256,
        "canonical_target_sha256": harness._canonical_target_sha256(spec),  # noqa: SLF001
        "source_value_artifact": spec.source_value_artifact,
        "source_value_artifact_sha256": harness._source_value_sha256(spec),  # noqa: SLF001
        "configuration_identity": harness._configuration_identity(args, spec),  # noqa: SLF001
        "route_identity": harness._route_identity(args, spec),  # noqa: SLF001
        "fd_endpoint_contract": harness._fd_endpoint_contract(spec),  # noqa: SLF001
        "fd_policy": {
            "policy_id": harness.LEDH_FD_POLICY_ID,
            "diagnostic_scope": harness.LEDH_FD_DIAGNOSTIC_SCOPE,
            "pass_rule": harness.LEDH_FD_PASS_RULE,
            "step_policy": harness.ledh_fd_step_policy_metadata(),
        },
        "gpu_policy": {
            "trust_basis": "offline_aggregate_of_validated_trusted_gpu_shards",
            "device_scope": "cpu",
            "cuda_visible_devices": "-1",
            "expect_device_kind": "cpu",
            "jit_compile": True,
            "dtype": "float32",
            "tf32_mode": "enabled_for_source_gpu_shards",
        },
        "command_identity": identity,
        "template_family_sha256": identity["template_family_sha256"],
        "exact_command_sha256": identity["exact_command_sha256"],
    }


def _validate_commands(commands: Sequence[Mapping[str, Any]]) -> None:
    expected_rows = set(harness.ROW_SPECS)
    if {str(command["row"]) for command in commands} != expected_rows:
        raise ValueError("command manifest does not cover exactly the six frozen rows")
    outputs = [str(command["output"]) for command in commands]
    markdown = [str(command["markdown_output"]) for command in commands]
    logs = [str(command["log"]) for command in commands]
    if len(set(outputs)) != len(outputs):
        raise ValueError("command manifest contains an output collision")
    if len(set(markdown)) != len(markdown):
        raise ValueError("command manifest contains a Markdown output collision")
    if len(set(logs)) != len(logs):
        raise ValueError("command manifest contains a log collision")
    invoked = [tuple(str(item) for item in command["invoked_argv"]) for command in commands]
    if len(set(invoked)) != len(invoked):
        raise ValueError("command manifest contains duplicate argv")
    exact_hashes = [str(command["exact_command_sha256"]) for command in commands]
    if len(set(exact_hashes)) != len(exact_hashes):
        raise ValueError("command manifest contains duplicate exact-command hashes")
    for command in commands:
        parsed = harness._parse_args(command["invoked_argv"][1:])  # noqa: SLF001
        spec = harness.ROW_SPECS[parsed.row]
        identity = _command_identity(parsed, spec, command["invoked_argv"])
        if command["command_identity"] != identity:
            raise ValueError("command identity does not match reparsed argv")
        if command["exact_command_sha256"] != identity["exact_command_sha256"]:
            raise ValueError("exact-command SHA-256 does not match reparsed argv")
        if command["template_family_sha256"] != identity["template_family_sha256"]:
            raise ValueError("template-family SHA-256 does not match reparsed argv")


def build_manifest() -> dict[str, Any]:
    phase2 = []
    phase3 = []
    for row in harness.ROW_SPECS:
        for time_steps in PREFIXES[row]:
            phase2.append(
                _runtime_command(
                    phase="phase2",
                    row=row,
                    seed=PHASE2_SEED,
                    time_steps=time_steps,
                    stage="score-only",
                )
            )
            phase2.append(
                _runtime_command(
                    phase="phase2",
                    row=row,
                    seed=PHASE2_SEED,
                    time_steps=time_steps,
                    stage="fd-only",
                )
            )
        full_time = harness.ROW_SPECS[row].full_time_steps
        for seed in PHASE3_SEEDS:
            phase3.append(
                _runtime_command(
                    phase="phase3",
                    row=row,
                    seed=seed,
                    time_steps=full_time,
                    stage="score-only",
                )
            )
            phase3.append(
                _runtime_command(
                    phase="phase3",
                    row=row,
                    seed=seed,
                    time_steps=full_time,
                    stage="fd-only",
                )
            )
    aggregates = [_aggregate_command(row) for row in harness.ROW_SPECS]
    commands = [*phase2, *phase3, *aggregates]
    _validate_commands(commands)
    required_directories = sorted(
        {
            str(Path(str(command[field])).parent)
            for command in commands
            for field in ("output", "markdown_output", "log")
        }
    )
    payload = {
        "schema_version": "bayesfilter.complete_highdim.ledh_phase2_phase3_exact_commands.v1",
        "run_id": RUN_ID,
        "status": "frozen_not_execution_authority",
        "revision": REVISION,
        "superseded_manifest": {
            "path": SUPERSEDED_MANIFEST_PATH,
            "sha256": harness._sha256(ROOT / SUPERSEDED_MANIFEST_PATH),  # noqa: SLF001
            "reason": "score_reference_validation_used_fd_timeout_for_score_shard",
        },
        "working_directory": str(ROOT),
        "python_executable": PYTHON,
        "runner": RUNNER,
        "builder": str(Path(__file__).resolve().relative_to(ROOT)),
        "output_path": OUTPUT_PATH,
        "artifact_root": ARTIFACT_ROOT,
        "log_root": LOG_ROOT,
        "canonical_target_artifact": harness.CANONICAL_TARGETS_PATH,
        "canonical_target_artifact_sha256": harness.CANONICAL_TARGETS_SHA256,
        "p1a_receipt_path": harness.P1A_RECEIPT_PATH,
        "p1a_receipt_sha256": harness.P1A_RECEIPT_SHA256,
        "p1b_receipt_path": harness.P1B_RECEIPT_PATH,
        "p1b_receipt_sha256": harness.P1B_RECEIPT_SHA256,
        "execution_authority_paths": {
            "phase2": harness.COMPLETE_HIGHDIM_PHASE2_EXECUTION_AUTHORITY_PATH,
            "phase3": harness.COMPLETE_HIGHDIM_PHASE3_EXECUTION_AUTHORITY_PATH,
        },
        "execution_authority_status_at_freeze": "absent_required_before_each_phase",
        "seeds": list(SEEDS),
        "phase2_seed": PHASE2_SEED,
        "phase3_seeds": list(PHASE3_SEEDS),
        "prefixes": {row: list(values) for row, values in PREFIXES.items()},
        "rows": {
            row: {
                "row_id": spec.row_id,
                "time_steps": spec.full_time_steps,
                "num_particles": spec.full_num_particles,
                "parameter_names": list(spec.parameter_names),
                "score_evaluation_theta": harness._float32_theta(spec),  # noqa: SLF001
                "canonical_target_sha256": harness._canonical_target_sha256(spec),  # noqa: SLF001
                "source_value_artifact": spec.source_value_artifact,
                "source_value_artifact_sha256": harness._source_value_sha256(spec),  # noqa: SLF001
                "fd_endpoint_contract": harness._fd_endpoint_contract(spec),  # noqa: SLF001
            }
            for row, spec in harness.ROW_SPECS.items()
        },
        "phase2_commands": phase2,
        "phase3_commands": phase3,
        "aggregate_commands": aggregates,
        "command_count": len(commands),
        "required_directories": required_directories,
        "command_set_sha256": _canonical_sha256(
            [command["exact_command_sha256"] for command in commands]
        ),
        "nonclaims": [
            "this manifest freezes commands but does not authorize execution",
            "prefix commands are explanatory diagnostics and cannot admit a row",
            "finite difference validates but is never the admitted score",
            "no ranking, HMC, posterior, source-faithfulness, or release claim",
        ],
    }
    return payload


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    payload = build_manifest()
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != serialized:
            raise SystemExit(f"exact command manifest is stale: {output}")
        print(f"PASS_COMPLETE_HIGHDIM_LED_H_COMMAND_MANIFEST_CHECK {output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialized, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

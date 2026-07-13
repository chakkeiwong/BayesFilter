#!/usr/bin/env python3
"""Generate strict A2 CPU-reference and trusted GPU/XLA artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

import bayesfilter.nonlinear.ssl_lstm_predictive_tf as predictive
from bayesfilter.nonlinear.ssl_lstm_sgqf_ukf_adapters import (
    make_ssl_lstm_svd_ukf_components,
    ssl_lstm_observation,
    ssl_lstm_transition,
)


PLAN_PATH = "docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md"
RESULT_PATH = "docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-result-2026-07-11.md"
BOUNDARY_PATH = "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/pre-run-boundary.json"
SUBPLAN_REVIEW_PATH = "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-round6-2026-07-12.md"

POINT_NAMES = [
    "truth_free",
    "phase2s_center",
    "shell_0_minus",
    "shell_0_plus",
    "shell_1_minus",
    "shell_1_plus",
    "shell_2_minus",
    "shell_2_plus",
    "shell_3_minus",
    "shell_3_plus",
]
POINTS_HEX = [
    ["0x1.6666666666666p-2", "-0x1.47ae147ae147bp-4", "0x1.4cccccccccccdp-1", "0x1.999999999999ap-5"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.ee87ac2b0ee48p-2", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.50dd6faf210bep-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.b19cbccaf903cp-3", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.2cd959924a756p-5", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.25964cacd3b9fp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.7f2fe6466d539p-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.8891e0688b5c0p-5"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.c88ade80893d6p-3"],
]
POINT_MATRIX_SHA256 = "d6ba48e5a64897f87caeece4de776c139d8fc62d00fc118d89b4d88da468829a"
EPSILON = 2.0**-52
HEAD_SHA256 = "a644d29c5c2fd09a0deb3a7b5212799ff1fcb163"
EXPECTED_ENTRY_HASHES = {
    "bayesfilter/nonlinear/ssl_lstm_posterior_tf.py": "6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667",
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json": "b6dc26637d584dbf6d62575a999af5cf43bb7bab35a5cf9eb6984d1cfaf6a068",
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json": "1538032c6e0c9ea664ea92ce9ea334c92c916c13831fd08af69865435c822f6e",
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md": "78f269a53fb0536017d32bd12c2b36967cd013a85dcb1102936ed79ae95e34b5",
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-result-codex-substitute-review-2026-07-11.md": "a3a7ca4f6396f352fe29e7da24e164ae4cb0d1144ea492b4b7270fe4f3f0decf",
    PLAN_PATH: "6b6b9799782be3304ecbd2dee465c52285688b5e2d1b3087d911ccad1279bbb0",
    SUBPLAN_REVIEW_PATH: "846574f1d6140efd5ff8e10f772f0d886be916585f30ccdac6960bd1eacfeaa1",
}
EXPECTED_A1_EVIDENCE_SIGNATURES = {
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json": "c208b513e2fbf74d654b3b349695a7fcb811b2a6c36f5c2fa76a30dd5e9c922d",
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json": "077abbd5d5d8dc1068d99aba90fc8b6dd5b74001cda1dd1fe4428d13a0b4631c",
}

ENTRY_PATHS = [
    "bayesfilter/nonlinear/ssl_lstm_posterior_tf.py",
    "bayesfilter/nonlinear/__init__.py",
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json",
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json",
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md",
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-result-codex-substitute-review-2026-07-11.md",
    PLAN_PATH,
    SUBPLAN_REVIEW_PATH,
]
SOURCE_PATHS = [
    "bayesfilter/nonlinear/ssl_lstm_predictive_tf.py",
    "bayesfilter/nonlinear/__init__.py",
    "tests/test_ssl_lstm_predictive_tf.py",
    "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py",
    "docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py",
]
CHECK_NAMES = [
    "a1_entry_hashes",
    "bank_hashes",
    "batch_parity",
    "compiler_hlo",
    "covariance_validity",
    "device_placement",
    "eager_xla_parity",
    "filter_parity",
    "forecast_replay",
    "no_cache_writes",
    "observation_timing",
    "process_noise_placement",
    "status_admission",
    "total_target_parity",
    "write_boundary",
]


class ContractError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def evidence_signature(payload: dict[str, Any]) -> str:
    projected = copy.deepcopy(payload)
    projected.pop("evidence_signature", None)
    projected.pop("created_at_utc", None)
    manifest = projected.get("run_manifest")
    if isinstance(manifest, dict):
        manifest.pop("started_at_utc", None)
        manifest.pop("completed_at_utc", None)
        manifest.pop("wall_time_seconds", None)
    return hashlib.sha256(canonical_bytes(projected)).hexdigest()


def sha256(path: str | Path) -> str:
    return hashlib.sha256((ROOT / Path(path)).read_bytes()).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(canonical_bytes(payload) + b"\n")


def load_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ContractError(f"nonfinite JSON constant {value!r}")

    payload = json.loads(
        (ROOT / path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ContractError(f"{path} must contain an object")
    return payload


def points() -> tf.Tensor:
    return tf.constant(
        [[float.fromhex(value) for value in row] for row in POINTS_HEX],
        tf.float64,
    )


def raw_bytes(tensor: tf.Tensor) -> bytes:
    return b"".join(
        struct.pack("<d", float(value))
        for value in tf.unstack(tf.reshape(tensor, [-1]))
    )


def tensor_row(name: str, tensor: tf.Tensor) -> dict[str, Any]:
    values = [float(value) for value in tf.unstack(tf.reshape(tensor, [-1]))]
    if not all(math.isfinite(value) for value in values):
        raise ContractError(f"nonfinite output tensor {name}")
    raw = b"".join(struct.pack("<d", value) for value in values)
    return {
        "name": name,
        "dtype": "float64",
        "shape": list(tensor.shape),
        "values_hex": [value.hex() for value in values],
        "raw_little_endian_sha256": hashlib.sha256(raw).hexdigest(),
    }


def device_rows(devices: list[Any]) -> list[dict[str, str]]:
    return [
        {"name": str(device.name), "device_type": str(device.device_type)}
        for device in devices
    ]


def git(*args: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not_installed"


def a2_named_cache_paths() -> list[str]:
    needles = (
        "ssl_lstm_predictive_tf",
        "test_ssl_lstm_predictive_tf",
        "benchmark_ssl_lstm_completion_phase_a2_terminal_forecast",
        "verify_ssl_lstm_completion_phase_a2_terminal_forecast",
    )
    result = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        is_cache = (
            path.name == ".pytest_cache"
            or "__pycache__" in relative.parts
            or path.suffix == ".pyc"
        )
        if is_cache and any(needle in relative.as_posix() for needle in needles):
            result.append(relative.as_posix())
    return sorted(result)


def verify_entry_state() -> None:
    if git("rev-parse", "HEAD").strip() != HEAD_SHA256:
        raise ContractError("HEAD drifted from the accepted A1 anchor")
    for path, expected in EXPECTED_ENTRY_HASHES.items():
        if sha256(path) != expected:
            raise ContractError(f"entry hash mismatch: {path}")
    for path, expected in EXPECTED_A1_EVIDENCE_SIGNATURES.items():
        if load_json(Path(path)).get("evidence_signature") != expected:
            raise ContractError(f"A1 evidence signature mismatch: {path}")
    boundary = load_json(Path(BOUNDARY_PATH))
    if (
        boundary.get("schema_version")
        != "bayesfilter.ssl_lstm_completion.phase_a2_scoped_boundary.v1"
        or boundary.get("status") != "A2_SCOPED_BOUNDARY_FROZEN"
        or boundary.get("evidence_signature") != evidence_signature(boundary)
        or not boundary.get("cache_scan", {}).get("passed")
    ):
        raise ContractError("A2 pre-run boundary is invalid")
    boundary_rows = {
        row["path"]: row for row in boundary.get("literal_a2_rows", [])
    }
    for path in SOURCE_PATHS:
        row = boundary_rows.get(path)
        if row is None or row.get("sha256") != sha256(path):
            raise ContractError(f"A2 source drift from boundary: {path}")
    if a2_named_cache_paths():
        raise ContractError("A2-named repository cache exists")
    required_environment = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "/tmp/bayesfilter-a2-pycache",
        "TMPDIR": "/tmp/bayesfilter-a2-tmp",
        "CUDA_CACHE_PATH": "/tmp/bayesfilter-a2-tmp/cuda-cache",
        "XLA_FLAGS": "--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla",
    }
    for name, expected in required_environment.items():
        if os.environ.get(name) != expected:
            raise ContractError(f"required runtime environment mismatch: {name}")


def file_row(path: str, role: str) -> dict[str, Any]:
    relative = Path(path)
    exists = (ROOT / relative).is_file()
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative.as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    ).returncode == 0
    return {
        "path": relative.as_posix(),
        "exists": exists,
        "tracked": tracked,
        "sha256": sha256(relative) if exists else None,
        "role": role,
    }


def binding_row(path: str, role: str) -> dict[str, Any]:
    relative = Path(path)
    if not (ROOT / relative).is_file():
        raise ContractError(f"missing entry binding {path}")
    signature = None
    if relative.suffix == ".json":
        signature = load_json(relative).get("evidence_signature")
    return {
        "path": relative.as_posix(),
        "file_sha256": sha256(relative),
        "evidence_signature": signature,
        "role": role,
    }


def bank_payload(bank: predictive.SSLLSTMInnovationBank) -> dict[str, Any]:
    derived_names = ("role", "arm", "terminal", "process", "observation")
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a2_innovation_bank.v1",
        "status": "A2_INNOVATION_BANK_FROZEN",
        "created_at_utc": utc_now(),
        "tensorflow_version": tf.__version__,
        "algorithm": bank.algorithm,
        "root_seed": [int(value) for value in tf.unstack(bank.root_seed)],
        "role": bank.role,
        "role_code": bank.role_code,
        "arm_id": bank.arm_id,
        "family_codes": predictive.FAMILY_CODES,
        "derived_seeds": {
            name: [int(value) for value in tf.unstack(seed)]
            for name, seed in zip(derived_names, bank.derived_seeds, strict=True)
        },
        "draw_count": bank.draw_count,
        "replication_count": bank.replication_count,
        "horizon": predictive.FORECAST_HORIZON,
        "tensors": [
            tensor_row("terminal", bank.terminal_standard_normal),
            tensor_row("process", bank.process_standard_normal),
            tensor_row("observation", bank.observation_standard_normal),
        ],
        "evidence_signature": "",
        "nonclaims": list(predictive.NONCLAIMS),
    }
    payload["evidence_signature"] = evidence_signature(payload)
    return payload


def load_bank(path: Path, config: predictive.SSLLSTMForecastConfig):
    import dataclasses

    payload = load_json(path)
    rows = {row["name"]: row for row in payload["tensors"]}
    if payload["evidence_signature"] != evidence_signature(payload):
        raise ContractError("persisted bank signature mismatch")
    materialized = {}
    for name in ("terminal", "process", "observation"):
        row = rows[name]
        tensor = tf.reshape(
            tf.constant(
                [float.fromhex(value) for value in row["values_hex"]],
                tf.float64,
            ),
            row["shape"],
        )
        if tensor_row(name, tensor)["raw_little_endian_sha256"] != row["raw_little_endian_sha256"]:
            raise ContractError(f"persisted bank tensor hash mismatch: {name}")
        materialized[name] = tensor
    provisional = predictive.SSLLSTMInnovationBank(
        terminal_standard_normal=materialized["terminal"],
        process_standard_normal=materialized["process"],
        observation_standard_normal=materialized["observation"],
        root_seed=tf.constant(payload["root_seed"], tf.int32),
        algorithm=payload["algorithm"],
        role=payload["role"],
        role_code=payload["role_code"],
        arm_id=payload["arm_id"],
        derived_seeds=tuple(
            tf.constant(payload["derived_seeds"][name], tf.int32)
            for name in ("role", "arm", "terminal", "process", "observation")
        ),
        content_signature="",
    )
    bank = dataclasses.replace(
        provisional,
        content_signature=predictive._innovation_bank_signature(provisional),
    )
    predictive._validate_innovation_bank(bank, draw_count=2, config=config)
    return payload, bank


def tolerance(multiplier: int, left: tf.Tensor, right: tf.Tensor) -> float:
    return multiplier * EPSILON * max(
        1.0,
        float(tf.reduce_max(tf.abs(left))),
        float(tf.reduce_max(tf.abs(right))),
    )


def max_residual(left: tf.Tensor, right: tf.Tensor) -> float:
    return float(tf.reduce_max(tf.abs(left - right)))


def path_tensors(paths: predictive.SSLLSTMForecastPaths):
    return [
        ("terminal_states", paths.terminal_states),
        ("states", paths.states),
        ("deterministic_transition_means", paths.deterministic_transition_means),
        ("process_innovations", paths.process_innovations),
        ("observation_means", paths.observation_means),
        ("observation_innovations", paths.observation_innovations),
        ("observations", paths.observations),
    ]


def zero_bank(bank: predictive.SSLLSTMInnovationBank):
    import dataclasses

    provisional = dataclasses.replace(
        bank,
        terminal_standard_normal=tf.zeros_like(bank.terminal_standard_normal),
        process_standard_normal=tf.zeros_like(bank.process_standard_normal),
        observation_standard_normal=tf.zeros_like(bank.observation_standard_normal),
        content_signature="",
    )
    return dataclasses.replace(
        provisional,
        content_signature=predictive._innovation_bank_signature(provisional),
    )


def compiler_row(name: str, program: Any, inputs: tuple[tf.Tensor, ...], outputs: tuple[tf.Tensor, ...]):
    hlo = program.experimental_get_compiler_ir(*inputs)(stage="hlo")
    if not isinstance(hlo, str) or "ENTRY" not in hlo:
        raise ContractError(f"missing HLO ENTRY for {name}")
    encoded = hlo.encode("utf-8")
    trace_count = int(program.experimental_get_tracing_count())
    if trace_count != 1:
        raise ContractError(f"trace count for {name} is {trace_count}, expected 1")
    return {
        "callable_name": name,
        "static_input_shapes": [list(tensor.shape) for tensor in inputs],
        "hlo_sha256": hashlib.sha256(encoded).hexdigest(),
        "hlo_byte_count": len(encoded),
        "hlo_text": hlo,
        "hlo_entry_present": True,
        "concrete_trace_count": trace_count,
        "output_devices": [str(tensor.device) for tensor in outputs],
    }


def covariance_rows(terminal: predictive.SSLLSTMTerminalState):
    return [
        {
            "name": name,
            "minimum_eigenvalue_hex": float(terminal.minimum_eigenvalue[index]).hex(),
            "psd_tolerance_hex": float(terminal.psd_tolerance[index]).hex(),
            "symmetry_residual": float(terminal.symmetry_residual[index]),
            "projection_residual": float(terminal.projection_residual[index]),
            "factor_reconstruction_residual": float(
                terminal.factor_reconstruction_residual[index]
            ),
            "status": int(terminal.status[index]),
        }
        for index, name in enumerate(POINT_NAMES)
    ]


def terminal_rows(terminal: predictive.SSLLSTMTerminalState):
    rows = []
    for index, name in enumerate(POINT_NAMES):
        status = int(terminal.status[index])
        filter_residual = float(terminal.filter_parity_residual[index])
        filter_threshold = float(terminal.filter_parity_tolerance[index])
        total_residual = float(terminal.total_parity_residual[index])
        total_threshold = float(terminal.total_parity_tolerance[index])
        passed = (
            status == 0
            and filter_residual <= filter_threshold
            and total_residual <= total_threshold
        )
        rows.append(
            {
                "name": name,
                "status": status,
                "filter_log_likelihood_hex": float(terminal.filter_log_likelihood[index]).hex(),
                "a1_filter_log_likelihood_hex": float(terminal.a1_filter_log_likelihood[index]).hex(),
                "filter_residual": filter_residual,
                "filter_threshold": filter_threshold,
                "total_value_hex": float(terminal.total_value[index]).hex(),
                "a1_target_value_hex": float(terminal.target_value[index]).hex(),
                "total_residual": total_residual,
                "total_threshold": total_threshold,
                "minimum_eigenvalue_hex": float(terminal.minimum_eigenvalue[index]).hex(),
                "psd_tolerance_hex": float(terminal.psd_tolerance[index]).hex(),
                "symmetry_residual": float(terminal.symmetry_residual[index]),
                "projection_residual": float(terminal.projection_residual[index]),
                "factor_reconstruction_residual": float(
                    terminal.factor_reconstruction_residual[index]
                ),
                "passed": passed,
            }
        )
    return rows


def direct_recursion_residual(
    config: predictive.SSLLSTMForecastConfig,
    free_points: tf.Tensor,
    paths: predictive.SSLLSTMForecastPaths,
) -> float:
    residual = 0.0
    for draw_index in range(2):
        components = make_ssl_lstm_svd_ukf_components(
            config.posterior_config.parameter_mask.embed(free_points[draw_index]),
            config.posterior_config.static_config,
            evidence_path=predictive.A2_RESULT_PATH,
            std_floor=config.posterior_config.std_floor,
        )
        previous = paths.terminal_states[draw_index]
        for horizon in range(10):
            state = ssl_lstm_transition(components.parameters, previous)
            observation = ssl_lstm_observation(components.parameters, state)
            residual = max(
                residual,
                max_residual(paths.states[draw_index, :, horizon], state),
                max_residual(paths.observations[draw_index, :, horizon], observation),
            )
            previous = state
    return residual


def stochastic_recursion_residuals(
    config: predictive.SSLLSTMForecastConfig,
    free_points: tf.Tensor,
    bank: predictive.SSLLSTMInnovationBank,
    paths: predictive.SSLLSTMForecastPaths,
) -> tuple[float, float, float, float]:
    transition_residual = 0.0
    process_residual = 0.0
    observation_mean_residual = 0.0
    observation_noise_residual = 0.0
    for draw_index in range(2):
        components = make_ssl_lstm_svd_ukf_components(
            config.posterior_config.parameter_mask.embed(free_points[draw_index]),
            config.posterior_config.static_config,
            evidence_path=predictive.A2_RESULT_PATH,
            std_floor=config.posterior_config.std_floor,
        )
        params = components.parameters
        previous = paths.terminal_states[draw_index]
        for horizon in range(10):
            expected_transition = ssl_lstm_transition(params, previous)
            expected_process = (
                bank.process_standard_normal[draw_index, :, horizon, :]
                * params.process_std[tf.newaxis, :]
            )
            expected_observation_mean = ssl_lstm_observation(
                params, paths.states[draw_index, :, horizon]
            )
            expected_observation_noise = (
                bank.observation_standard_normal[draw_index, :, horizon, :]
                * params.observation_std[tf.newaxis, :]
            )
            transition_residual = max(
                transition_residual,
                max_residual(
                    paths.deterministic_transition_means[draw_index, :, horizon],
                    expected_transition,
                ),
            )
            process_residual = max(
                process_residual,
                max_residual(
                    paths.process_innovations[draw_index, :, horizon],
                    expected_process,
                ),
            )
            observation_mean_residual = max(
                observation_mean_residual,
                max_residual(
                    paths.observation_means[draw_index, :, horizon],
                    expected_observation_mean,
                ),
            )
            observation_noise_residual = max(
                observation_noise_residual,
                max_residual(
                    paths.observation_innovations[draw_index, :, horizon],
                    expected_observation_noise,
                ),
            )
            previous = paths.states[draw_index, :, horizon]
    return (
        transition_residual,
        process_residual,
        observation_mean_residual,
        observation_noise_residual,
    )


def check_row(name: str, role: str, passed: bool, residual=None, threshold=None):
    if residual is not None and not math.isfinite(float(residual)):
        raise ContractError(f"nonfinite residual for {name}")
    if threshold is not None and not math.isfinite(float(threshold)):
        raise ContractError(f"nonfinite threshold for {name}")
    return {
        "name": name,
        "role": role,
        "passed": bool(passed),
        "residual": None if residual is None else float(residual),
        "threshold": None if threshold is None else float(threshold),
    }


def provenance_payload(
    config: predictive.SSLLSTMForecastConfig,
    terminal: predictive.SSLLSTMTerminalState,
    paths: predictive.SSLLSTMForecastPaths,
    bank_file: Path,
    bank_json: dict[str, Any],
    compiler_evidence: list[dict[str, Any]],
):
    posterior = config.posterior_config
    runtime = paths.provenance
    return {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a2_provenance.v1",
        "a0_target_semantic_sha256": runtime.target_semantic_sha256,
        "a1_adapter_signature": runtime.a1_adapter_signature,
        "a1_parameter_mask_sha256": runtime.parameter_mask_sha256,
        "a1_observation_raw_sha256": runtime.observation_raw_sha256,
        "a1_full_fixture_raw_sha256": runtime.full_fixture_raw_sha256,
        "a1_prior_center_raw_sha256": runtime.prior_center_raw_sha256,
        "a1_result_file_sha256": runtime.a1_result_file_sha256,
        "a2_subplan_file_sha256": runtime.a2_subplan_file_sha256,
        "a2_contract_signature": runtime.a2_contract_signature,
        "forecast_config_signature": runtime.forecast_config_signature,
        "innovation_bank_file_sha256": sha256(bank_file),
        "innovation_bank_evidence_signature": bank_json["evidence_signature"],
        "free_draw_matrix_raw_sha256": runtime.free_draw_matrix_raw_sha256,
        "embedded_full_parameter_matrix_raw_sha256": runtime.embedded_full_parameter_matrix_raw_sha256,
        "filter": {
            "backend": "tf_svd_ukf",
            "std_floor_hex": float(posterior.std_floor).hex(),
            "alpha_hex": float(posterior.alpha).hex(),
            "beta_hex": float(posterior.beta).hex(),
            "kappa_hex": float(posterior.kappa).hex(),
            "placement_floor_hex": float(posterior.placement_floor).hex(),
            "innovation_floor_hex": float(posterior.innovation_floor).hex(),
            "rank_tolerance_hex": float(posterior.rank_tolerance).hex(),
            "jitter_hex": float(posterior.jitter).hex(),
            "return_filtered": True,
        },
        "terminal_covariances": covariance_rows(terminal),
        "runtime": {
            "tensorflow_version": tf.__version__,
            "tensorflow_probability_version": package_version("tensorflow-probability"),
            "dtype": "float64",
            "jit_compile": True,
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "execution_role": runtime.execution_role,
            "physical_devices": [
                {"name": name, "device_type": kind}
                for name, kind in runtime.physical_devices
            ],
            "logical_devices": [
                {"name": name, "device_type": kind}
                for name, kind in runtime.logical_devices
            ],
            "trust_basis": runtime.trust_basis,
            "compiler_evidence": compiler_evidence,
        },
        "innovations": {
            "algorithm": "philox",
            "role": bank_json["role"],
            "role_code": bank_json["role_code"],
            "arm_id": bank_json["arm_id"],
            "root_seed": bank_json["root_seed"],
            "family_codes": bank_json["family_codes"],
            "tensor_hashes": {
                row["name"]: row["raw_little_endian_sha256"]
                for row in bank_json["tensors"]
            },
            "horizon": 10,
            "draw_count": 2,
            "replication_count": 2,
        },
        "horizon_convention": "state_and_observation_after_transition_t_plus_1",
        "cluster_unit": "complete_ten_step_path_per_draw_replication",
        "approximation_qualification": "conditional_on_approximate_historical_svd_ukf_not_exact_nonlinear_filter",
        "nonclaims": list(predictive.NONCLAIMS),
    }


def environment_manifest() -> dict[str, str | None]:
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "PYTHONPYCACHEPREFIX": os.environ.get("PYTHONPYCACHEPREFIX"),
        "TMPDIR": os.environ.get("TMPDIR"),
        "CUDA_CACHE_PATH": os.environ.get("CUDA_CACHE_PATH"),
        "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
    }


def exact_command(
    mode: str,
    bank_path: Path,
    output: Path,
    log_path: Path,
    bank_log_path: Path | None,
    cpu_reference_path: Path | None,
) -> str:
    prefix = (
        "CUDA_VISIBLE_DEVICES=-1 " if mode == "cpu-reference" else ""
    ) + (
        "PYTHONDONTWRITEBYTECODE=1 "
        "PYTHONPYCACHEPREFIX=/tmp/bayesfilter-a2-pycache "
        "TMPDIR=/tmp/bayesfilter-a2-tmp "
        "CUDA_CACHE_PATH=/tmp/bayesfilter-a2-tmp/cuda-cache "
        "XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla' "
        "/usr/bin/strace -f -qq -e trace=%file -o "
    )
    trace = (
        "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2/"
        + ("cpu-generation-write-trace.log" if mode == "cpu-reference" else "gpu-generation-write-trace.log")
    )
    command = (
        prefix
        + trace
        + " /home/ubuntu/anaconda3/envs/tfgpu/bin/python "
        + "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py"
        + f" --mode {mode} --bank {bank_path.as_posix()}"
    )
    if bank_log_path is not None:
        command += f" --bank-log {bank_log_path.as_posix()}"
    if cpu_reference_path is not None:
        command += f" --cpu-reference {cpu_reference_path.as_posix()}"
    command += f" --output {output.as_posix()} --log-path {log_path.as_posix()}"
    return command


def make_artifact(
    mode: str,
    bank_path: Path,
    output: Path,
    log_path: Path,
    bank_log_path: Path | None,
    cpu_reference_path: Path | None,
) -> dict[str, Any]:
    started = utc_now()
    start = time.perf_counter()
    config = predictive.SSLLSTMForecastConfig()
    config.assert_evidence_config()
    verify_entry_state()
    all_points = points()
    if hashlib.sha256(raw_bytes(all_points)).hexdigest() != POINT_MATRIX_SHA256:
        raise ContractError("frozen point matrix hash mismatch")

    if mode == "cpu-reference":
        generated = predictive.make_ssl_lstm_innovation_bank(
            config,
            2,
            tf.constant([20260712, 1202], tf.int32),
            "paired_diagnostic_shared",
            0,
        )
        if (ROOT / bank_path).exists():
            existing = load_json(bank_path)
            if existing.get("evidence_signature") != evidence_signature(existing):
                raise ContractError("existing bank has invalid signature")
            expected = bank_payload(generated)
            if [row["raw_little_endian_sha256"] for row in existing["tensors"]] != [
                row["raw_little_endian_sha256"] for row in expected["tensors"]
            ]:
                raise ContractError("existing bank content differs from frozen bank")
            bank_json, bank = load_bank(bank_path, config)
        else:
            bank_json = bank_payload(generated)
            write_json(bank_path, bank_json)
            bank_json, bank = load_bank(bank_path, config)
    else:
        if not (ROOT / bank_path).is_file():
            raise ContractError("GPU canary requires persisted CPU bank")
        bank_json, bank = load_bank(bank_path, config)

    runtime_role = (
        "cpu_hidden_xla_reference"
        if mode == "cpu-reference"
        else "trusted_gpu_xla_canary"
    )
    trust_basis = (
        "cpu_hidden_reference_exception_not_gpu_evidence"
        if mode == "cpu-reference"
        else "owner_designated_managed_session_visible_gpu_trusted"
    )
    terminal = predictive.extract_ssl_lstm_terminal_states(all_points, config)
    paths = predictive.forecast_ssl_lstm_paths(
        all_points[:2],
        bank,
        config,
        runtime_execution_role=runtime_role,
        trust_basis=trust_basis,
    )
    replay = predictive.forecast_ssl_lstm_paths(
        all_points[:2],
        bank,
        config,
        runtime_execution_role=runtime_role,
        trust_basis=trust_basis,
    )
    eager = predictive.eager_debug_ssl_lstm_forecast_paths(
        all_points[:2], bank, config
    )
    deterministic = predictive.forecast_ssl_lstm_paths(
        all_points[:2], zero_bank(bank), config
    )

    terminal_program = predictive.ssl_lstm_terminal_compiled_program(config, 10)
    forecast_program = predictive.ssl_lstm_forecast_compiled_program(config, 2)
    terminal_inputs = (all_points,)
    terminal_outputs = tuple(terminal_program(*terminal_inputs))
    forecast_inputs = (
        all_points[:2],
        paths.terminal.mean,
        paths.terminal.factor,
        bank.terminal_standard_normal,
        bank.process_standard_normal,
        bank.observation_standard_normal,
    )
    forecast_outputs = tuple(forecast_program(*forecast_inputs))
    compiler_evidence = sorted(
        [
            compiler_row(
                "forecast_ssl_lstm_paths_core",
                forecast_program,
                forecast_inputs,
                forecast_outputs,
            ),
            compiler_row(
                "extract_ssl_lstm_terminal_states_core",
                terminal_program,
                terminal_inputs,
                terminal_outputs,
            ),
        ],
        key=lambda row: row["callable_name"],
    )

    replay_residual = max(
        max_residual(left, right)
        for (_name, left), (_other_name, right) in zip(
            path_tensors(paths), path_tensors(replay), strict=True
        )
    )
    eager_residual = max(
        max_residual(left, right)
        for (_name, left), (_other_name, right) in zip(
            path_tensors(paths), path_tensors(eager), strict=True
        )
    )
    batch_residual = 0.0
    for index in range(2):
        # Scalar/batch parity is exercised by slicing the canonical tensors, not regenerating seeds.
        import dataclasses

        provisional = dataclasses.replace(
            bank,
            terminal_standard_normal=bank.terminal_standard_normal[index : index + 1],
            process_standard_normal=bank.process_standard_normal[index : index + 1],
            observation_standard_normal=bank.observation_standard_normal[index : index + 1],
            content_signature="",
        )
        one_bank = dataclasses.replace(
            provisional,
            content_signature=predictive._innovation_bank_signature(provisional),
        )
        scalar = predictive.forecast_ssl_lstm_path(all_points[index], one_bank, config)
        for (_name, scalar_tensor), (_batch_name, batch_tensor) in zip(
            path_tensors(scalar), path_tensors(paths), strict=True
        ):
            batch_residual = max(
                batch_residual, max_residual(scalar_tensor, batch_tensor[index])
            )

    process_coordinate_residual = max_residual(
        paths.states[..., :1] - paths.deterministic_transition_means[..., :1],
        paths.process_innovations,
    )
    deterministic_coordinate_residual = max_residual(
        paths.states[..., 1:], paths.deterministic_transition_means[..., 1:]
    )
    observation_residual = max_residual(
        paths.observations - paths.observation_means,
        paths.observation_innovations,
    )
    (
        transition_mean_residual,
        process_scale_residual,
        observation_mean_residual,
        observation_scale_residual,
    ) = stochastic_recursion_residuals(config, all_points[:2], bank, paths)
    recursion_residual = direct_recursion_residual(
        config, all_points[:2], deterministic
    )
    filter_residual = float(tf.reduce_max(terminal.filter_parity_residual))
    filter_threshold = float(tf.reduce_max(terminal.filter_parity_tolerance))
    total_residual = float(tf.reduce_max(terminal.total_parity_residual))
    total_threshold = float(tf.reduce_max(terminal.total_parity_tolerance))
    eager_threshold = max(
        tolerance(512, left, right)
        for (_name, left), (_other_name, right) in zip(
            path_tensors(paths), path_tensors(eager), strict=True
        )
    )
    batch_threshold = max(
        128 * EPSILON * max(1.0, float(tf.reduce_max(tf.abs(tensor))))
        for _name, tensor in path_tensors(paths)
    )
    recursion_threshold = max(
        128 * EPSILON * max(1.0, float(tf.reduce_max(tf.abs(tensor))))
        for _name, tensor in path_tensors(deterministic)
    )

    logical = tf.config.list_logical_devices()
    logical_gpu = [device for device in logical if device.device_type == "GPU"]
    output_devices = [str(tensor.device) for _name, tensor in path_tensors(paths)]
    device_pass = (
        not logical_gpu and all("CPU" in device.upper() for device in output_devices)
        if mode == "cpu-reference"
        else bool(logical_gpu) and all("GPU" in device.upper() for device in output_devices)
    )
    if not device_pass:
        raise ContractError("runtime output device placement failed")

    cpu_reference_binding = None
    cpu_gpu_residual = None
    cpu_gpu_threshold = None
    cpu_reference_crosslink = False
    if mode == "gpu-xla-canary":
        if cpu_reference_path is None or not (ROOT / cpu_reference_path).is_file():
            raise ContractError("GPU canary requires CPU reference")
        cpu = load_json(cpu_reference_path)
        if cpu.get("evidence_signature") != evidence_signature(cpu):
            raise ContractError("CPU crosslink signature mismatch")
        cpu_reference_binding = binding_row(
            cpu_reference_path.as_posix(), "accepted_phase_a2_cpu_reference"
        )
        cpu_reference_crosslink = True
        residuals = []
        thresholds = []
        for stored, (_name, current) in zip(
            cpu["forecast_tensors"], path_tensors(paths), strict=True
        ):
            reference = tf.reshape(
                tf.constant(
                    [float.fromhex(value) for value in stored["values_hex"]],
                    tf.float64,
                ),
                stored["shape"],
            )
            residuals.append(max_residual(reference, current))
            thresholds.append(tolerance(4096, reference, current))
        cpu_gpu_residual = max(residuals)
        cpu_gpu_threshold = max(thresholds)

    checks = {
        "a1_entry_hashes": check_row("a1_entry_hashes", "promotion_veto", True),
        "bank_hashes": check_row("bank_hashes", "promotion_veto", True),
        "batch_parity": check_row(
            "batch_parity", "promotion_veto", batch_residual <= batch_threshold, batch_residual, batch_threshold
        ),
        "compiler_hlo": check_row("compiler_hlo", "promotion_veto", True),
        "covariance_validity": check_row(
            "covariance_validity", "promotion_veto", bool(tf.reduce_all(terminal.status == 0))
        ),
        "device_placement": check_row("device_placement", "promotion_veto", device_pass),
        "eager_xla_parity": check_row(
            "eager_xla_parity", "promotion_veto", eager_residual <= eager_threshold, eager_residual, eager_threshold
        ),
        "filter_parity": check_row(
            "filter_parity", "promotion_veto", filter_residual <= filter_threshold, filter_residual, filter_threshold
        ),
        "forecast_replay": check_row(
            "forecast_replay", "promotion_veto", replay_residual == 0.0, replay_residual, 0.0
        ),
        "no_cache_writes": check_row(
            "no_cache_writes", "promotion_veto", not a2_named_cache_paths()
        ),
        "observation_timing": check_row(
            "observation_timing",
            "promotion_veto",
            max(
                observation_residual,
                observation_mean_residual,
                observation_scale_residual,
                recursion_residual,
            )
            <= recursion_threshold,
            max(
                observation_residual,
                observation_mean_residual,
                observation_scale_residual,
                recursion_residual,
            ),
            recursion_threshold,
        ),
        "process_noise_placement": check_row(
            "process_noise_placement",
            "promotion_veto",
            max(
                process_coordinate_residual,
                deterministic_coordinate_residual,
                transition_mean_residual,
                process_scale_residual,
                recursion_residual,
            )
            <= recursion_threshold,
            max(
                process_coordinate_residual,
                deterministic_coordinate_residual,
                transition_mean_residual,
                process_scale_residual,
                recursion_residual,
            ),
            recursion_threshold,
        ),
        "status_admission": check_row(
            "status_admission", "promotion_veto", all(row["passed"] for row in terminal_rows(terminal))
        ),
        "total_target_parity": check_row(
            "total_target_parity", "promotion_veto", total_residual <= total_threshold, total_residual, total_threshold
        ),
        "write_boundary": check_row("write_boundary", "promotion_veto", True),
    }
    if mode == "gpu-xla-canary":
        checks["cpu_gpu_parity"] = check_row(
            "cpu_gpu_parity",
            "promotion_veto",
            bool(cpu_gpu_residual <= cpu_gpu_threshold),
            cpu_gpu_residual,
            cpu_gpu_threshold,
        )
        checks["cpu_reference_crosslink"] = check_row(
            "cpu_reference_crosslink",
            "promotion_veto",
            cpu_reference_crosslink,
        )
    if recursion_residual > recursion_threshold:
        raise ContractError("zero-bank direct recursion failed")
    if not all(row["passed"] for row in checks.values()):
        failed = sorted(name for name, row in checks.items() if not row["passed"])
        raise ContractError(f"runtime contract checks failed: {failed}")

    completed = utc_now()
    command = exact_command(
        mode,
        bank_path,
        output,
        log_path,
        bank_log_path,
        cpu_reference_path,
    )
    output_paths = [output.as_posix(), log_path.as_posix()]
    if mode == "cpu-reference":
        output_paths.append(bank_path.as_posix())
        if bank_log_path is not None:
            output_paths.append(bank_log_path.as_posix())
    manifest = {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(git("status", "--porcelain=v1", "-uall").strip()),
        "command": command,
        "cwd": str(ROOT),
        "interpreter": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python_version": platform.python_version(),
        "packages": {
            "tensorflow": tf.__version__,
            "tensorflow_probability": package_version("tensorflow-probability"),
        },
        "environment": environment_manifest(),
        "physical_devices": device_rows(tf.config.list_physical_devices()),
        "logical_devices": device_rows(tf.config.list_logical_devices()),
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "jit_compile": True,
        "dtype": "float64",
        "random_seeds": [20260712, 1202],
        "started_at_utc": started,
        "completed_at_utc": completed,
        "wall_time_seconds": time.perf_counter() - start,
        "output_paths": sorted(output_paths),
        "plan_path": PLAN_PATH,
        "result_path": RESULT_PATH,
        "trust_basis": trust_basis,
    }
    bank_hashes = {
        row["name"]: row["raw_little_endian_sha256"] for row in bank_json["tensors"]
    }
    artifact = {
        "schema_version": (
            "bayesfilter.ssl_lstm_completion.phase_a2_cpu_reference.v1"
            if mode == "cpu-reference"
            else "bayesfilter.ssl_lstm_completion.phase_a2_gpu_xla_canary.v1"
        ),
        "artifact_role": (
            "phase_a2_cpu_hidden_reference"
            if mode == "cpu-reference"
            else "phase_a2_trusted_gpu_xla_canary"
        ),
        "status": (
            "CPU_REFERENCE_CONTRACT_PASSED"
            if mode == "cpu-reference"
            else "GPU_XLA_CANARY_PASSED"
        ),
        "created_at_utc": completed,
        "run_manifest": manifest,
        "entry_bindings": sorted(
            [binding_row(path, "accepted_a1_or_a2_entry") for path in ENTRY_PATHS],
            key=lambda row: row["path"],
        ),
        "source_files": sorted(
            [file_row(path, "a2_source_test_harness_verifier") for path in SOURCE_PATHS],
            key=lambda row: row["path"],
        ),
        "frozen_design": {
            "point_names": POINT_NAMES,
            "points": tensor_row("points", all_points),
            "point_matrix_sha256": POINT_MATRIX_SHA256,
            "forecast_point_names": POINT_NAMES[:2],
            "replication_count": 2,
            "horizon": 10,
            "tolerance_multipliers": {
                "filter": 64,
                "total": 64,
                "recursion": 128,
                "batch": 128,
                "eager_xla": 512,
                "cpu_gpu": 4096,
            },
        },
        "bank_binding": {
            "path": bank_path.as_posix(),
            "file_sha256": sha256(bank_path),
            "evidence_signature": bank_json["evidence_signature"],
            "role": bank_json["role"],
            "tensor_hashes": bank_hashes,
        },
        "cpu_reference_binding": cpu_reference_binding,
        "terminal_results": terminal_rows(terminal),
        "forecast_tensors": [tensor_row(name, tensor) for name, tensor in path_tensors(paths)],
        "compiler_evidence": compiler_evidence,
        "provenance": provenance_payload(
            config, terminal, paths, bank_path, bank_json, compiler_evidence
        ),
        "contract_checks": [checks[name] for name in sorted(checks)],
        "evidence_signature": "",
        "nonclaims": list(predictive.NONCLAIMS),
    }
    artifact["evidence_signature"] = evidence_signature(artifact)
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("cpu-reference", "gpu-xla-canary"), required=True
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-log", type=Path)
    parser.add_argument("--cpu-reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-path", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    artifact = make_artifact(
        args.mode,
        args.bank,
        args.output,
        args.log_path,
        args.bank_log,
        args.cpu_reference,
    )
    write_json(args.output, artifact)
    (ROOT / args.log_path).write_text(
        json.dumps(
            {
                "status": artifact["status"],
                "evidence_signature": artifact["evidence_signature"],
                "output": args.output.as_posix(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.bank_log is not None:
        bank = load_json(args.bank)
        (ROOT / args.bank_log).write_text(
            json.dumps(
                {
                    "status": bank["status"],
                    "evidence_signature": bank["evidence_signature"],
                    "output": args.bank.as_posix(),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    print(artifact["status"])
    print(artifact["evidence_signature"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"A2_CONTRACT_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

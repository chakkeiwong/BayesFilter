#!/usr/bin/env python3
"""Independent Tier 2 verifier for A3 predictive-validation artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import struct
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE_DIR = Path("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3")
LIVE_PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-predictive-validation-live-plan-2026-07-13.md"
)
FIXTURE_PATH = PHASE_DIR / "fixture-contract.json"
CPU_REFERENCE_PATH = PHASE_DIR / "oracle-cpu-reference.json"
CPU_VERIFICATION_PATH = PHASE_DIR / "oracle-cpu-reference-verify.json"
GPU_REFERENCE_PATH = PHASE_DIR / "oracle-gpu-xla-canary.json"
RUNNER_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_predictive_validation_a3_2026_07_13.py"
)
VERIFIER_PATH = Path(
    "docs/benchmarks/verify_ssl_lstm_predictive_validation_a3_2026_07_13.py"
)
GENERATION_CORE_PATH = Path(
    "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a3_forecast_oracle_2026_07_13.py"
)
VERIFICATION_CORE_PATH = Path(
    "docs/benchmarks/verify_ssl_lstm_completion_phase_a3_forecast_oracle_2026_07_13.py"
)
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
ORACLE_SOURCE = Path("bayesfilter/testing/scalar_lgssm_forecast_oracle.py")
PREDICTIVE_TEST = Path("tests/test_predictive_equivalence.py")
ORACLE_TEST = Path("tests/test_scalar_lgssm_forecast_oracle.py")

CPU_SCHEMA = "bayesfilter.ssl_lstm_predictive_validation.a3_cpu_reference.v2"
GPU_SCHEMA = "bayesfilter.ssl_lstm_predictive_validation.a3_gpu_xla.v2"
CPU_STATUS = "A3_CPU_REFERENCE_PASSED"
GPU_STATUS = "A3_GPU_XLA_PARITY_PASSED"
VERIFICATION_SCHEMA = "bayesfilter.ssl_lstm_predictive_validation.a3_verification.v1"
CPU_GPU_TOLERANCE_MULTIPLIER = 8192
NUMERIC_FIXTURE_FIELDS = (
    "controlled_alternatives",
    "fixture_constants",
    "lgssm",
    "numeric_provenance",
    "quantile_contract",
)


class VerificationError(RuntimeError):
    """Raised when an A3 artifact fails independent verification."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes((ROOT / path).read_bytes())


def _signature(payload: dict[str, Any]) -> str:
    projection = copy.deepcopy(payload)
    projection.pop("evidence_signature", None)
    projection.pop("created_at_utc", None)
    manifest = projection.get("run_manifest")
    if isinstance(manifest, dict):
        for field in ("started_at_utc", "completed_at_utc", "wall_time_seconds"):
            manifest.pop(field, None)
    return _sha256_bytes(_canonical_bytes(projection))


def _strict_load(path: Path) -> dict[str, Any]:
    def pairs_hook(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise VerificationError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise VerificationError(f"nonfinite JSON constant {value!r} in {path}")

    value = json.loads(
        (ROOT / path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(value, dict):
        raise VerificationError(f"{path} must contain a JSON object")
    if (ROOT / path).read_bytes() != _canonical_bytes(value) + b"\n":
        raise VerificationError(f"artifact is not canonical JSON: {path}")
    return value


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    if spec is None or spec.loader is None:
        raise VerificationError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _numeric_fixture_projection(fixture: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in NUMERIC_FIXTURE_FIELDS if field not in fixture]
    if missing:
        raise VerificationError(f"numeric fixture fields missing: {missing}")
    return {field: copy.deepcopy(fixture[field]) for field in NUMERIC_FIXTURE_FIELDS}


def _source_rows() -> list[dict[str, Any]]:
    rows = (
        (ORACLE_SOURCE, "oracle_implementation"),
        (PREDICTIVE_SOURCE, "predictive_statistics_implementation"),
        (ORACLE_TEST, "oracle_focused_tests"),
        (PREDICTIVE_TEST, "predictive_statistics_focused_tests"),
        (GENERATION_CORE_PATH, "reviewed_numerical_generation_core"),
        (RUNNER_PATH, "tier2_generation_adapter"),
    )
    return [
        {"path": path.as_posix(), "sha256": _sha256(path), "role": role}
        for path, role in rows
    ]


def _verification_source_rows() -> list[dict[str, Any]]:
    rows = (
        (VERIFICATION_CORE_PATH, "independent_numerical_replay_core"),
        (VERIFIER_PATH, "tier2_independent_verifier"),
    )
    return [
        {"path": path.as_posix(), "sha256": _sha256(path), "role": role}
        for path, role in rows
    ]


def _configuration_binding(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "fixture_path": FIXTURE_PATH.as_posix(),
        "numeric_configuration_sha256": _sha256_bytes(
            _canonical_bytes(_numeric_fixture_projection(fixture))
        ),
        "live_plan_path": LIVE_PLAN_PATH.as_posix(),
        "classification": "A3_TEST_FIXTURE_ONLY_NOT_A4_FROZEN",
    }


def _verify_tensor_row(row: dict[str, Any]) -> None:
    dtype = row.get("dtype")
    shape = row.get("shape")
    if not isinstance(shape, list) or not all(type(item) is int and item >= 0 for item in shape):
        raise VerificationError(f"invalid tensor shape for {row.get('name')}")
    if dtype == "float64":
        values = [float.fromhex(item) for item in row.get("values_hex", [])]
        if not all(math.isfinite(item) for item in values):
            raise VerificationError(f"nonfinite tensor row {row.get('name')}")
        raw = b"".join(struct.pack("<d", item) for item in values)
    elif dtype == "int32":
        values = row.get("values", [])
        if not all(type(item) is int and -(2**31) <= item < 2**31 for item in values):
            raise VerificationError(f"invalid int32 row {row.get('name')}")
        raw = b"".join(struct.pack("<i", item) for item in values)
    elif dtype == "bool":
        values = row.get("values", [])
        if not all(type(item) is bool for item in values):
            raise VerificationError(f"invalid bool row {row.get('name')}")
        raw = bytes(int(item) for item in values)
    else:
        raise VerificationError(f"unknown tensor dtype {dtype!r}")
    if math.prod(shape) != len(values):
        raise VerificationError(f"tensor shape/value mismatch: {row.get('name')}")
    if row.get("raw_little_endian_sha256") != _sha256_bytes(raw):
        raise VerificationError(f"tensor raw hash mismatch: {row.get('name')}")


def _section(payload: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("tensor_sections", {}).get(name)
    if not isinstance(rows, list):
        raise VerificationError(f"missing tensor section {name!r}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise VerificationError(f"malformed tensor row in {name!r}")
        _verify_tensor_row(row)
        if row["name"] in result:
            raise VerificationError(f"duplicate tensor row {name}/{row['name']}")
        result[row["name"]] = row
    return result


def _materialized_indices(
    core: ModuleType,
    payload: dict[str, Any],
    section: str,
    constants: dict[str, Any],
    stats: Any,
    tf: Any,
) -> Any:
    rows = core._section(payload, section)
    expected = {
        "chain_indices",
        "draw_indices",
        "forecast_replication_indices",
        "seed",
    }
    if set(rows) != expected:
        raise VerificationError(f"index section {section!r} differs")
    seed = core._decode_tensor_row(rows["seed"], tf)
    if tuple(seed.shape) != (2,):
        raise VerificationError(f"index seed shape differs: {section}")
    return stats.HierarchicalBootstrapIndices(
        chain_indices=core._decode_tensor_row(rows["chain_indices"], tf),
        draw_indices=core._decode_tensor_row(rows["draw_indices"], tf),
        forecast_replication_indices=core._decode_tensor_row(
            rows["forecast_replication_indices"], tf
        ),
        block_length=int(constants["block_length"]),
        block_mode="moving",
        chain_mode="stratified_fixed_chains",
        seed=seed,
        status=tf.constant("VALID"),
    )


def _near_rows(
    observed: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    label: str,
) -> None:
    if [row["name"] for row in observed] != [row["name"] for row in expected]:
        raise VerificationError(f"tensor row names differ: {label}")
    for left, right in zip(observed, expected):
        _verify_tensor_row(left)
        _verify_tensor_row(right)
        if left["dtype"] != right["dtype"] or left["shape"] != right["shape"]:
            raise VerificationError(f"tensor metadata differs: {label}/{left['name']}")
        if left["dtype"] != "float64":
            if left != right:
                raise VerificationError(f"exact tensor mismatch: {label}/{left['name']}")
            continue
        left_values = [float.fromhex(item) for item in left["values_hex"]]
        right_values = [float.fromhex(item) for item in right["values_hex"]]
        scale = max(1.0, *(abs(item) for item in left_values + right_values))
        tolerance = CPU_GPU_TOLERANCE_MULTIPLIER * 2.0**-52 * scale
        residual = max(
            (abs(a - b) for a, b in zip(left_values, right_values)),
            default=0.0,
        )
        if residual > tolerance:
            raise VerificationError(
                f"scale-aware tensor mismatch: {label}/{left['name']}"
            )


def _near_scalar(left: float, right: float, label: str) -> None:
    if not math.isfinite(left) or not math.isfinite(right):
        raise VerificationError(f"nonfinite scalar comparison: {label}")
    scale = max(1.0, abs(left), abs(right))
    tolerance = CPU_GPU_TOLERANCE_MULTIPLIER * 2.0**-52 * scale
    if abs(left - right) > tolerance:
        raise VerificationError(f"scale-aware scalar mismatch: {label}")


def _normalize_hlo(text: str) -> str:
    text = re.sub(
        r"(a_inference__[A-Za-z0-9_]+)_\d+__",
        r"\1_<process_id>__",
        text,
    )
    return re.sub(
        r"(__inference__[A-Za-z0-9_]+)_\d+",
        r"\1_<process_id>",
        text,
    )


def _verify_compiler_evidence(
    payload: dict[str, Any],
    recomputed: dict[str, Any],
    schema: str,
) -> None:
    oracle = recomputed["oracle"]
    parameters = recomputed["parameters"]
    tf = recomputed["tf"]
    probabilities = recomputed["analytic"].quantile_probabilities
    bank = recomputed["banks"][0]
    programs = (
        (
            "scalar_lgssm_analytic",
            oracle.scalar_lgssm_analytic_compiled_program(
                int(probabilities.shape[0])
            ),
            (parameters.as_tensor(), probabilities),
        ),
        (
            "scalar_lgssm_simulation",
            oracle.scalar_lgssm_simulation_compiled_program(
                *bank.terminal_standard_normal.shape
            ),
            (
                parameters.as_tensor(),
                bank.terminal_standard_normal,
                bank.process_standard_normal,
                bank.observation_standard_normal,
            ),
        ),
    )
    rows = payload.get("compiler_evidence")
    if not isinstance(rows, list) or len(rows) != len(programs):
        raise VerificationError("compiler evidence row count differs")
    expected_device = "CPU:" if schema == CPU_SCHEMA else "GPU:"
    for row, (name, program, inputs) in zip(rows, programs):
        if set(row) != {
            "callable_name",
            "hlo_text",
            "hlo_sha256",
            "hlo_byte_count",
            "hlo_entry_present",
            "concrete_trace_count",
            "output_devices",
        }:
            raise VerificationError(f"compiler evidence fields differ for {name}")
        stored_hlo = row["hlo_text"]
        if (
            row["callable_name"] != name
            or not isinstance(stored_hlo, str)
            or row["hlo_sha256"] != _sha256_bytes(stored_hlo.encode("utf-8"))
            or row["hlo_byte_count"] != len(stored_hlo.encode("utf-8"))
            or row["hlo_entry_present"] is not True
            or "ENTRY" not in stored_hlo
            or row["concrete_trace_count"] != 1
        ):
            raise VerificationError(f"stored compiler evidence is invalid for {name}")
        fresh_hlo = str(program.experimental_get_compiler_ir(*inputs)(stage="hlo"))
        outputs = program(*inputs)
        fresh_devices = sorted(
            {
                str(item.device)
                for item in tf.nest.flatten(outputs)
                if hasattr(item, "device") and item.device
            }
        )
        trace_count = len(program._list_all_concrete_functions_for_serialization())
        if (
            not fresh_hlo
            or "ENTRY" not in fresh_hlo
            or trace_count != 1
            or row["output_devices"] != fresh_devices
            or not fresh_devices
            or not all(expected_device in device for device in fresh_devices)
            or _normalize_hlo(stored_hlo) != _normalize_hlo(fresh_hlo)
        ):
            raise VerificationError(f"compiler/HLO/device structure differs for {name}")


def _verify_controlled_alternatives(
    payload: dict[str, Any],
    recomputed: dict[str, Any],
    fixture: dict[str, Any],
    core: ModuleType,
) -> None:
    tf = recomputed["tf"]
    stats = recomputed["stats"]
    constants = fixture["fixture_constants"]
    horizon = int(constants["horizon"])
    feature_alpha = tf.constant(core._hex_float(constants["feature_alpha_hex"]), tf.float64)
    mmd_alpha = tf.constant(core._hex_float(constants["mmd_alpha_hex"]), tf.float64)
    total_alpha = tf.constant(core._hex_float(constants["total_alpha_hex"]), tf.float64)
    bandwidths = tf.constant(
        [core._hex_float(item) for item in constants["bandwidths_hex"]],
        tf.float64,
    )
    weights = tf.constant(
        [core._hex_float(item) for item in constants["mixture_weights_hex"]],
        tf.float64,
    )
    schedule = tf.constant(constants["chain_pair_schedule"], tf.int32)
    alternatives = fixture["controlled_alternatives"]
    mean_shift = tf.constant(core._hex_float(alternatives["mean_shift_hex"]), tf.float64)
    variance_increment = tf.constant(
        core._hex_float(alternatives["variance_increment_hex"]), tf.float64
    )
    skew_coefficient = tf.constant(
        core._hex_float(alternatives["skew_coefficient_hex"]), tf.float64
    )
    dependence_correlation = tf.constant(
        core._hex_float(alternatives["dependence_correlation_hex"]), tf.float64
    )
    analytic = recomputed["analytic"]
    right = recomputed["simulations"][1].observations
    centered = right - analytic.observation_mean
    standardized_base = centered / tf.sqrt(analytic.observation_variance)
    common = standardized_base[..., :1]
    correlations = analytic.observation_covariance[:, 0] / tf.sqrt(
        analytic.observation_variance * analytic.observation_variance[0]
    )
    independent_weight = tf.sqrt(1.0 - tf.square(dependence_correlation))
    normalization = tf.sqrt(
        1.0
        + 2.0
        * independent_weight
        * dependence_correlation
        * correlations
    )
    reconstructed_paths = {
        "mean": right + mean_shift,
        "variance": analytic.observation_mean
        + centered
        * tf.sqrt(
            (analytic.observation_variance + variance_increment)
            / analytic.observation_variance
        ),
        "skew": right
        + skew_coefficient
        * (tf.square(centered) - analytic.observation_variance),
        "dependence": analytic.observation_mean
        + tf.sqrt(analytic.observation_variance)
        * (
            independent_weight * standardized_base
            + dependence_correlation * common
        )
        / normalization,
    }
    config = stats.PredictiveStatisticsConfig(
        horizon=horizon,
        quantile_probabilities=tuple(
            core._hex_float(item)
            for item in fixture["quantile_contract"]["probabilities_hex"]
        ),
        jit_compile=True,
    )
    bootstrap_left = core._resample(
        tf, recomputed["simulations"][0].observations, recomputed["indices_left"]
    )
    right_summary = recomputed["summaries"][1]
    mechanics = {
        "mean_shift_mean_residual": float(
            tf.reduce_max(
                tf.abs(
                    tf.reduce_mean(
                        reconstructed_paths["mean"] - right, axis=[0, 1, 2]
                    )
                    - tf.fill([horizon], mean_shift)
                )
            )
        ),
        "variance_log_variance_direction": float(
            tf.reduce_max(
                stats.summarize_forecast_paths(
                    reconstructed_paths["variance"], config
                ).log_variances
                - right_summary.log_variances
            )
        ),
        "skew_third_moment_change": float(
            tf.reduce_max(
                tf.abs(
                    stats.summarize_forecast_paths(
                        reconstructed_paths["skew"], config
                    ).central_moments[0]
                    - right_summary.central_moments[0]
                )
            )
        ),
        "dependence_covariance_change": float(
            tf.reduce_max(
                tf.abs(
                    stats.summarize_forecast_paths(
                        reconstructed_paths["dependence"], config
                    ).cross_horizon_covariance
                    - right_summary.cross_horizon_covariance
                )
            )
        ),
    }
    expected_records: list[dict[str, Any]] = []
    for name, paths in reconstructed_paths.items():
        summary = stats.summarize_forecast_paths(paths, config)
        estimate = tf.concat(
            [
                recomputed["summaries"][0].means - summary.means,
                recomputed["summaries"][0].log_variances - summary.log_variances,
            ],
            axis=0,
        )
        bootstrap_right = core._resample(
            tf, paths, recomputed["indices_right"]
        )
        bootstrap = core._bootstrap_features(tf, bootstrap_left, bootstrap_right)
        expected_inputs = [
            core._tensor_row("paths", paths),
            core._tensor_row("feature_estimate", estimate),
            core._tensor_row("bootstrap_feature_estimates", bootstrap),
        ]
        observed_inputs = _section(payload, f"alternative_{name}_inputs")
        _near_rows(
            [observed_inputs[row["name"]] for row in expected_inputs],
            expected_inputs,
            f"alternative {name} inputs",
        )
        feature = stats.simultaneous_feature_intervals(
            estimate,
            feature_alpha=feature_alpha,
            method="bootstrap_max_statistic",
            bootstrap_estimates=bootstrap,
            minimum_bootstrap_count=20,
            jit_compile=True,
        )
        standardized = stats.standardize_forecast_paths(
            paths,
            analytic.observation_mean,
            tf.sqrt(analytic.observation_variance),
            scale_floor=tf.constant(2.0**-40, tf.float64),
            jit_compile=True,
            allow_floor_use=False,
        )
        cross = stats.cross_chain_linear_mmd(
            recomputed["standardized"][0],
            standardized,
            bandwidths=bandwidths,
            mixture_weights=weights,
            chain_pair_schedule=schedule,
            independent_arm_banks_verified=recomputed["global_domain_nonreuse"],
            stationarity_verified=True,
            mixing_verified=True,
            jit_compile=True,
        )
        interval = stats.cross_chain_mmd_upper_interval(
            cross,
            mmd_alpha=mmd_alpha,
            block_length=int(constants["block_length"]),
            jit_compile=True,
        )
        decision = stats.classify_predictive_evidence(
            feature,
            interval,
            margins=recomputed["margins"],
            mmd_tolerance=recomputed["mmd_tolerance"],
            total_alpha=total_alpha,
            feature_alpha=feature_alpha,
            mmd_alpha=mmd_alpha,
        )
        valid = (
            core._status_text(summary.status) == "VALID"
            and feature.inference_admissible
            and core._status_text(feature.status) == "VALID"
            and interval.inference_admissible
            and core._status_text(interval.status) == "VALID"
            and decision.status != "INVALID_HARD_VETO"
        )
        expected_record = {
            "name": name,
            "valid": bool(valid),
            "decision": core._decision_row(name, decision),
            "repair_trigger": bool(valid and decision.status != "MATERIAL_DIFFERENCE"),
            "feature_max_abs": float(tf.reduce_max(tf.abs(estimate))),
            "mmd_estimate": float(cross.squared_mmd_linear),
            "mmd_lower": float(interval.lower),
            "mmd_upper": float(interval.upper),
        }
        expected_records.append(expected_record)
        observed_record = next(
            row
            for row in payload["alternative_diagnostics"]["records"]
            if row.get("name") == name
        )
        for field in ("name", "valid", "decision", "repair_trigger"):
            if observed_record.get(field) != expected_record[field]:
                raise VerificationError(
                    f"alternative {name} categorical record differs: {field}"
                )
        for field in ("feature_max_abs", "mmd_estimate", "mmd_lower", "mmd_upper"):
            _near_scalar(
                float(observed_record[field]),
                float(expected_record[field]),
                f"alternative {name} {field}",
            )
        feature_rows = [
            core._tensor_row(field, value)
            for field, value in core._dataclass_dict(feature).items()
            if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)
        ]
        observed_feature = _section(payload, f"alternative_{name}_feature_interval")
        _near_rows(
            [observed_feature[row["name"]] for row in feature_rows],
            feature_rows,
            f"alternative {name} feature interval",
        )
        cross_rows = [
            core._tensor_row("squared_mmd_linear", cross.squared_mmd_linear),
            core._tensor_row("kernel_contrast_sequence", cross.kernel_contrast_sequence),
            core._int_tensor_row("chain_pair_schedule", cross.chain_pair_schedule),
        ]
        observed_cross = _section(payload, f"alternative_{name}_cross_chain")
        _near_rows(
            [observed_cross[row["name"]] for row in cross_rows],
            cross_rows,
            f"alternative {name} cross-chain",
        )
        interval_rows = [
            (
                core._int_tensor_row(field, value)
                if getattr(value.dtype, "is_integer", False)
                else core._tensor_row(field, value)
            )
            for field, value in core._dataclass_dict(interval).items()
            if hasattr(value, "dtype")
            and (
                getattr(value.dtype, "is_floating", False)
                or getattr(value.dtype, "is_integer", False)
            )
        ]
        observed_interval = _section(payload, f"alternative_{name}_mmd_interval")
        _near_rows(
            [observed_interval[row["name"]] for row in interval_rows],
            interval_rows,
            f"alternative {name} MMD interval",
        )
    diagnostics = payload["alternative_diagnostics"]
    if diagnostics.get("policy") != "valid_underpowered_is_repair_trigger_not_hard_veto":
        raise VerificationError("controlled-alternative policy differs")
    if set(diagnostics.get("mechanics", {})) != set(mechanics):
        raise VerificationError("controlled-alternative mechanics fields differ")
    for name, value in mechanics.items():
        _near_scalar(
            float(diagnostics["mechanics"][name]),
            float(value),
            f"controlled-alternative mechanics {name}",
        )
    if not (
        mechanics["mean_shift_mean_residual"] <= 512.0 * 2.0**-52
        and mechanics["variance_log_variance_direction"] > 0.0
        and mechanics["skew_third_moment_change"] > 0.0
        and mechanics["dependence_covariance_change"] > 0.0
        and all(record["valid"] for record in expected_records)
    ):
        raise VerificationError(
            "controlled-alternative mechanics or inferential validity failed"
        )


def _verify_manifest(
    payload: dict[str, Any], path: Path, schema: str, fixture: dict[str, Any]
) -> None:
    manifest = payload.get("run_manifest")
    if not isinstance(manifest, dict):
        raise VerificationError("run manifest is missing")
    required = {
        "git_commit", "git_dirty", "command", "cwd", "interpreter",
        "conda_env", "python_version", "packages", "environment",
        "physical_devices", "logical_devices", "tf32_enabled", "jit_compile",
        "dtype", "random_seeds", "started_at_utc", "completed_at_utc",
        "wall_time_seconds", "output_paths", "plan_path", "result_path",
        "fixture_path", "execution_role", "trust_basis",
    }
    if set(manifest) != required:
        raise VerificationError(
            f"run manifest fields differ: {sorted(set(manifest) ^ required)}"
        )
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"]):
        raise VerificationError("manifest Git commit is invalid")
    if type(manifest["git_dirty"]) is not bool:
        raise VerificationError("manifest dirty state is invalid")
    if manifest["cwd"] != str(ROOT) or manifest["interpreter"] != sys.executable:
        raise VerificationError("manifest execution location differs")
    try:
        command = shlex.split(manifest["command"])
    except ValueError as exc:
        raise VerificationError("manifest command cannot be parsed") from exc
    expected_mode = "cpu-reference" if schema == CPU_SCHEMA else "gpu-xla"
    required_command = [
        sys.executable,
        RUNNER_PATH.as_posix(),
        "--mode",
        expected_mode,
    ]
    if command[:4] != required_command:
        raise VerificationError("manifest command prefix differs")
    if manifest["packages"] != {
        "tensorflow": str(__import__("tensorflow").__version__),
        "tensorflow_probability": str(
            __import__("tensorflow_probability").__version__
        ),
    }:
        raise VerificationError("manifest package versions differ")
    if manifest["jit_compile"] is not True or manifest["dtype"] != "float64":
        raise VerificationError("manifest JIT/dtype contract differs")
    if manifest["random_seeds"] != fixture["fixture_constants"]["root_seed"]:
        raise VerificationError("manifest random seed differs")
    if manifest["output_paths"] != [path.as_posix()]:
        raise VerificationError("manifest output path differs")
    if manifest["plan_path"] != LIVE_PLAN_PATH.as_posix():
        raise VerificationError("manifest live-plan path differs")
    if manifest["fixture_path"] != FIXTURE_PATH.as_posix():
        raise VerificationError("manifest fixture path differs")
    try:
        started = datetime.fromisoformat(manifest["started_at_utc"])
        completed = datetime.fromisoformat(manifest["completed_at_utc"])
    except (TypeError, ValueError) as exc:
        raise VerificationError("manifest timestamps are invalid") from exc
    if (
        started.tzinfo is None
        or completed.tzinfo is None
        or completed < started
        or payload["created_at_utc"] != manifest["completed_at_utc"]
    ):
        raise VerificationError("manifest timestamp ordering differs")
    wall_time = manifest["wall_time_seconds"]
    if type(wall_time) not in (int, float) or not math.isfinite(wall_time) or wall_time < 0:
        raise VerificationError("manifest wall time is invalid")
    device_types = {
        row.get("device_type")
        for key in ("physical_devices", "logical_devices")
        for row in manifest[key]
        if isinstance(row, dict)
    }
    cuda_visibility = manifest["environment"].get("CUDA_VISIBLE_DEVICES")
    if schema == CPU_SCHEMA:
        if (
            cuda_visibility != "-1"
            or "GPU" in device_types
            or manifest["execution_role"] != "cpu_hidden_xla_reference"
            or manifest["trust_basis"]
            != "cpu_hidden_reference_exception_not_gpu_evidence"
        ):
            raise VerificationError("CPU-hidden provenance differs")
    else:
        if (
            cuda_visibility == "-1"
            or "GPU" not in device_types
            or manifest["execution_role"] != "trusted_gpu_xla_oracle"
            or manifest["trust_basis"]
            != "owner_designated_managed_session_visible_gpu_trusted"
        ):
            raise VerificationError("trusted GPU provenance differs")


def _load_cpu_reference_for_gpu(binding: dict[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "evidence_signature"}:
        raise VerificationError("GPU CPU-reference binding fields differ")
    if binding["path"] != CPU_REFERENCE_PATH.as_posix():
        raise VerificationError("GPU CPU-reference path differs")
    cpu = _strict_load(CPU_REFERENCE_PATH)
    if (
        cpu.get("schema_version") != CPU_SCHEMA
        or cpu.get("status") != CPU_STATUS
        or cpu.get("evidence_signature") != _signature(cpu)
        or binding["file_sha256"] != _sha256(CPU_REFERENCE_PATH)
        or binding["evidence_signature"] != cpu["evidence_signature"]
    ):
        raise VerificationError("GPU CPU-reference binding is invalid")
    receipt = _strict_load(CPU_VERIFICATION_PATH)
    if (
        receipt.get("status") != "A3_CPU_REFERENCE_VERIFIED"
        or receipt.get("artifact_sha256") != _sha256(CPU_REFERENCE_PATH)
        or receipt.get("evidence_signature") != cpu["evidence_signature"]
        or receipt.get("verifier_sources") != _verification_source_rows()
    ):
        raise VerificationError("GPU CPU reference lacks valid independent receipt")
    return cpu


def verify(path: Path) -> dict[str, Any]:
    payload = _strict_load(path)
    fixture = _strict_load(FIXTURE_PATH)
    schema = payload.get("schema_version")
    expected_status = (
        CPU_STATUS if schema == CPU_SCHEMA else GPU_STATUS if schema == GPU_SCHEMA else None
    )
    if expected_status is None or payload.get("status") != expected_status:
        raise VerificationError("artifact identity/status differs")
    expected_keys = {
        "schema_version", "artifact_role", "status", "created_at_utc",
        "run_manifest", "configuration_binding", "source_files",
        "cpu_reference_binding", "cpu_gpu_parity", "tensor_sections",
        "compiler_evidence", "deterministic_residuals",
        "monte_carlo_diagnostics", "alternative_diagnostics", "role_ledger",
        "decision_rows", "uncertainty", "contract_checks", "bank_provenance",
        "resampling_provenance", "statistical_metadata", "evidence_signature",
        "nonclaims", "governance_tier",
    }
    if set(payload) != expected_keys:
        raise VerificationError(
            f"artifact fields differ: {sorted(set(payload) ^ expected_keys)}"
        )
    if payload["evidence_signature"] != _signature(payload):
        raise VerificationError("artifact evidence signature differs")
    if payload["configuration_binding"] != _configuration_binding(fixture):
        raise VerificationError("artifact configuration binding differs")
    if payload["source_files"] != _source_rows():
        raise VerificationError("artifact source bindings differ")
    if payload["governance_tier"] != "TIER2_MATERIAL_RESEARCH_ENGINEERING":
        raise VerificationError("artifact governance classification differs")
    expected_role = (
        "tier2_a3_cpu_hidden_oracle_reference"
        if schema == CPU_SCHEMA
        else "tier2_a3_trusted_gpu_xla_parity"
    )
    if payload["artifact_role"] != expected_role:
        raise VerificationError("artifact execution role differs")

    core = _load_module(VERIFICATION_CORE_PATH, "bayesfilter_a3_verification_core")
    core.CPU_SCHEMA = CPU_SCHEMA
    core.GPU_SCHEMA = GPU_SCHEMA
    core._indices = lambda payload, section, constants, stats, tf: _materialized_indices(
        core, payload, section, constants, stats, tf
    )
    check_names = [
        "configuration_binding" if name == "fixture_binding" else name
        for name in core.CHECK_NAMES
    ]
    checks = payload["contract_checks"]
    if [row.get("name") for row in checks] != check_names:
        raise VerificationError("artifact check names/order differ")
    if not all(row.get("passed") is True for row in checks):
        raise VerificationError("artifact contains a failed contract check")
    for rows in payload["tensor_sections"].values():
        for row in rows:
            _verify_tensor_row(row)
    if payload["role_ledger"].get("quadratic_mmd_u") != "descriptive_only_even_iid_fixture":
        raise VerificationError("quadratic MMD role was promoted")
    bank = payload["bank_provenance"]
    resampling = payload["resampling_provenance"]
    if (
        bank.get("authority") != "materialized_float64_tensor_rows_and_raw_hashes"
        or bank.get("seed_metadata_is_replay_authority") is not False
    ):
        raise VerificationError("bank replay authority differs")
    expected_index_hashes = {
        arm: {
            name: row["raw_little_endian_sha256"]
            for name, row in _section(payload, section).items()
        }
        for arm, section in (
            ("left", "resampling_indices_left"),
            ("right", "resampling_indices_right"),
        )
    }
    if (
        resampling.get("authority") != "materialized_indices_replay_authority"
        or resampling.get("seed_metadata_is_replay_authority") is not False
        or resampling.get("arm_tensor_hashes") != expected_index_hashes
    ):
        raise VerificationError("resampling replay authority differs")

    recomputed = core._recompute_core(payload, fixture)
    for section_name, expected_rows in recomputed["sections"].items():
        observed = _section(payload, section_name)
        selected = [observed[row["name"]] for row in expected_rows]
        _near_rows(selected, expected_rows, section_name)
    coverage_payload = dict(payload)
    if schema == CPU_SCHEMA:
        # The reused seed-attestation core checks only this historical label.
        # Tier 2 provenance is validated above and in the run manifest.
        coverage_payload["artifact_role"] = "phase_a3_cpu_hidden_oracle_reference"
    core._verify_coverage(coverage_payload, recomputed, fixture)
    core._verify_diagnostics(payload, recomputed, fixture)
    core._verify_decisions(payload, recomputed, fixture)
    _verify_controlled_alternatives(payload, recomputed, fixture, core)
    _verify_compiler_evidence(payload, recomputed, schema)

    if schema == GPU_SCHEMA:
        cpu = _load_cpu_reference_for_gpu(payload["cpu_reference_binding"])
        parity = payload["cpu_gpu_parity"]
        if not isinstance(parity, dict) or parity.get("passed") is not True:
            raise VerificationError("GPU parity did not pass")
        if bank.get("generation_mode") != "gpu_reconstruction_from_cpu_artifact_values":
            raise VerificationError("GPU did not consume persisted CPU inputs")
        maximum_residual = 0.0
        maximum_threshold = 0.0
        if set(payload["tensor_sections"]) != set(cpu["tensor_sections"]):
            raise VerificationError("GPU/CPU tensor section sets differ")
        for section_name, gpu_rows in payload["tensor_sections"].items():
            cpu_rows = cpu["tensor_sections"][section_name]
            if [row["name"] for row in gpu_rows] != [row["name"] for row in cpu_rows]:
                raise VerificationError(f"GPU/CPU row names differ: {section_name}")
            for gpu_row, cpu_row in zip(gpu_rows, cpu_rows):
                if gpu_row["dtype"] != cpu_row["dtype"] or gpu_row["shape"] != cpu_row["shape"]:
                    raise VerificationError(f"GPU/CPU metadata differs: {section_name}")
                if gpu_row["dtype"] != "float64":
                    if gpu_row != cpu_row:
                        raise VerificationError(f"GPU/CPU exact mismatch: {section_name}")
                    continue
                gpu_values = [float.fromhex(item) for item in gpu_row["values_hex"]]
                cpu_values = [float.fromhex(item) for item in cpu_row["values_hex"]]
                scale = max(1.0, *(abs(item) for item in gpu_values + cpu_values))
                threshold = CPU_GPU_TOLERANCE_MULTIPLIER * 2.0**-52 * scale
                residual = max(
                    (abs(a - b) for a, b in zip(gpu_values, cpu_values)),
                    default=0.0,
                )
                maximum_residual = max(maximum_residual, residual)
                maximum_threshold = max(maximum_threshold, threshold)
                if residual > threshold:
                    raise VerificationError(
                        f"GPU/CPU parity fails: {section_name}/{gpu_row['name']}"
                    )
        if (
            parity.get("maximum_absolute_residual") != maximum_residual
            or parity.get("maximum_scale_aware_threshold") != maximum_threshold
            or parity.get("tolerance_multiplier") != CPU_GPU_TOLERANCE_MULTIPLIER
        ):
            raise VerificationError("GPU parity summary differs")
    elif payload["cpu_reference_binding"] is not None or payload["cpu_gpu_parity"] is not None:
        raise VerificationError("CPU artifact must not self-crosslink")

    _verify_manifest(payload, path, schema, fixture)
    return {
        "schema_version": VERIFICATION_SCHEMA,
        "status": (
            "A3_CPU_REFERENCE_VERIFIED"
            if schema == CPU_SCHEMA
            else "A3_GPU_XLA_PARITY_VERIFIED"
        ),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_path": path.as_posix(),
        "artifact_sha256": _sha256(path),
        "evidence_signature": payload["evidence_signature"],
        "contract_check_count": len(checks),
        "all_contract_checks_passed": True,
        "independent_numerical_replay_passed": True,
        "verifier_sources": _verification_source_rows(),
        "nonclaims": payload["nonclaims"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    receipt = verify(args.artifact)
    absolute = ROOT / args.output
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(_canonical_bytes(receipt) + b"\n")
    print(_canonical_bytes(receipt).decode("ascii"))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (VerificationError, Exception) as exc:
        # Preserve TensorFlow/core exception context while keeping a stable CLI status.
        print(f"A3_VERIFICATION_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

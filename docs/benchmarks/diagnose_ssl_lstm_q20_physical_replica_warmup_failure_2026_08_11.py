#!/usr/bin/env python3
"""Diagnose occupancy versus within-region disagreement in failed warm-up."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-physical-replica-travel-repair-plan-2026-08-10.md"
)
R8_RESULT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r8-material-24x1-eight-hour/material.json"
)
R11_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r11-material-24x1-resumed"
)
R11_RESULT = R11_ROOT / "material.json"
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r12-warmup-failure-decomposition"
)
FINAL = OUTPUT_ROOT / "diagnosis.json"
GEOMETRY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/"
    "r1/geometry.json"
)
CHECKPOINT_RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_checkpoint_2026_08_10.py"
)
RESUME_RUNNER = Path(
    "docs/benchmarks/"
    "resume_ssl_lstm_q20_physical_distributed_replica_material_2026_08_11.py"
)

R8_SHA256 = "9e6771652842b6f96e304509a042949dc2513923ef8279021a7783b4fd82b9d9"
R11_SHA256 = "0fbec0c372008d406953908a30b6aa66a27d843781a93dba5ae52cd98235c66b"
WINDOW_ENDPOINTS = (600, 700, 800, 900, 1000)
WINDOW_DRAWS = 300
CHAINS = 4
PARAMETER_DIM = 4
RHAT_THRESHOLD = 1.05


class WarmupDiagnosisError(RuntimeError):
    """Raised when bound trace evidence cannot support the diagnosis."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise WarmupDiagnosisError(f"refusing to overwrite {path}")
    encoded = json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, _abs(path))
    if spec is None or spec.loader is None:
        raise WarmupDiagnosisError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read_receipt(tf: Any, receipt: Mapping[str, Any]) -> Any:
    path = Path(str(receipt["path"]))
    raw = _abs(path).read_bytes()
    if len(raw) != int(receipt["bytes"]):
        raise WarmupDiagnosisError(f"receipt byte count mismatch: {path}")
    if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
        raise WarmupDiagnosisError(f"receipt hash mismatch: {path}")
    tensor = tf.io.parse_tensor(raw, out_type=tf.dtypes.as_dtype(receipt["dtype"]))
    if list(tensor.shape) != list(receipt["shape"]):
        raise WarmupDiagnosisError(f"receipt shape mismatch: {path}")
    return tensor


def _longest_runs(tf: Any, signs: Any) -> Mapping[str, Any]:
    values = tf.convert_to_tensor(signs, tf.bool)
    draws = int(values.shape[0])
    current_negative = tf.zeros((CHAINS,), tf.int32)
    current_positive = tf.zeros((CHAINS,), tf.int32)
    longest_negative = tf.zeros((CHAINS,), tf.int32)
    longest_positive = tf.zeros((CHAINS,), tf.int32)
    for index in range(draws):
        negative = values[index]
        current_negative = tf.where(negative, current_negative + 1, 0)
        current_positive = tf.where(negative, 0, current_positive + 1)
        longest_negative = tf.maximum(longest_negative, current_negative)
        longest_positive = tf.maximum(longest_positive, current_positive)
    return {
        "longest_negative_run_by_chain": longest_negative,
        "longest_positive_run_by_chain": longest_positive,
    }


def _between_chain_decomposition(tf: Any, values: Any, signs: Any, centers: Any) -> Mapping[str, Any]:
    physical = tf.convert_to_tensor(values, tf.float64)
    negative = tf.convert_to_tensor(signs, tf.bool)
    plus = tf.convert_to_tensor(centers, tf.float64)[0]
    minus = tf.convert_to_tensor(centers, tf.float64)[1]
    selected_center = tf.where(negative[..., tf.newaxis], minus, plus)
    residual = physical - selected_center
    raw_means = tf.reduce_mean(physical, axis=0)
    composition_means = tf.reduce_mean(selected_center, axis=0)
    residual_means = tf.reduce_mean(residual, axis=0)
    raw_centered = raw_means - tf.reduce_mean(raw_means, axis=0)
    composition_centered = composition_means - tf.reduce_mean(composition_means, axis=0)
    residual_centered = residual_means - tf.reduce_mean(residual_means, axis=0)
    raw_ss = tf.reduce_sum(tf.square(raw_centered))
    composition_ss = tf.reduce_sum(tf.square(composition_centered))
    residual_ss = tf.reduce_sum(tf.square(residual_centered))
    cross_twice = 2.0 * tf.reduce_sum(composition_centered * residual_centered)
    identity_error = tf.abs(raw_ss - composition_ss - residual_ss - cross_twice)
    return {
        "raw_chain_means": raw_means,
        "occupancy_implied_chain_means": composition_means,
        "source_center_residual_chain_means": residual_means,
        "raw_between_chain_sum_squares": raw_ss,
        "occupancy_between_chain_sum_squares": composition_ss,
        "residual_between_chain_sum_squares": residual_ss,
        "twice_cross_term": cross_twice,
        "sum_squares_identity_error": identity_error,
        "residual_to_raw_norm_ratio": tf.sqrt(residual_ss) / tf.maximum(
            tf.sqrt(raw_ss), tf.constant(1.0e-300, tf.float64)
        ),
        "residual_draws": residual,
    }


def _classify(sign_report: Mapping[str, Any], residual_report: Mapping[str, Any]) -> str:
    sign_finite = int(sign_report["nonfinite_rhat_count"]) == 0
    residual_finite = int(residual_report["nonfinite_rhat_count"]) == 0
    if not sign_finite or not residual_finite:
        return "INDETERMINATE_NONFINITE_EXPLANATORY_DIAGNOSTIC"
    sign_pass = bool(sign_report["passed"])
    residual_pass = bool(residual_report["passed"])
    if not sign_pass and residual_pass:
        return "OCCUPANCY_DISAGREEMENT_DOMINANT_UNDER_SOURCE_CENTER_DECOMPOSITION"
    if sign_pass and not residual_pass:
        return "WITHIN_REGION_OR_LOCAL_GEOMETRY_DISAGREEMENT_DOMINANT"
    if not sign_pass and not residual_pass:
        return "MIXED_OCCUPANCY_AND_WITHIN_REGION_DISAGREEMENT"
    return "NEITHER_DECOMPOSITION_COMPONENT_FAILS_RHAT_SCREEN"


def run() -> Mapping[str, Any]:
    if _abs(FINAL).exists():
        raise WarmupDiagnosisError("refusing to overwrite diagnosis")
    bindings = {"r8_sha256": _sha(R8_RESULT), "r11_sha256": _sha(R11_RESULT)}
    if bindings != {"r8_sha256": R8_SHA256, "r11_sha256": R11_SHA256}:
        raise WarmupDiagnosisError("bound material result identity mismatch")

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise WarmupDiagnosisError("CPU diagnostic found visible GPU")
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary

    resume = _load("warmup_diagnosis_resume_support", RESUME_RUNNER)
    checkpoint = _load("warmup_diagnosis_checkpoint_support", CHECKPOINT_RUNNER)
    r8 = resume._load_r8_history(tf)
    r11 = json.loads(_abs(R11_RESULT).read_text(encoding="utf-8"))
    if r11.get("status") != "RESUMED_MATERIAL_WARMUP_NOT_READY":
        raise WarmupDiagnosisError("r11 terminal status is not the reviewed warm-up failure")
    if r11["hard_failures"] or int(r11["counts"]["retained_draws_per_chain"]) != 0:
        raise WarmupDiagnosisError("r11 is not an engineering-clean zero-retained failure")

    resumed_states = []
    verified_r11_receipts = 0
    for index, bound in enumerate(r11["chunk_manifests"]):
        manifest_path = Path(str(bound["path"]))
        raw = _abs(manifest_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != bound["sha256"]:
            raise WarmupDiagnosisError(f"r11 manifest hash mismatch: {index}")
        manifest = json.loads(raw)
        if manifest["chunk_index"] != index:
            raise WarmupDiagnosisError("r11 chunk order mismatch")
        verified = {
            name: _read_receipt(tf, receipt)
            for name, receipt in manifest["receipts"].items()
        }
        verified_r11_receipts += len(verified)
        resumed_states.append(verified["state"])
    if len(resumed_states) != 50 or verified_r11_receipts != 550:
        raise WarmupDiagnosisError("r11 receipt inventory mismatch")

    latent = tf.concat((r8["state"], tf.concat(resumed_states, axis=0)), axis=0)
    if latent.shape != (1000, 6, CHAINS, PARAMETER_DIM):
        raise WarmupDiagnosisError("combined trace shape mismatch")
    geometry = json.loads(_abs(GEOMETRY).read_text(encoding="utf-8"))
    chart = checkpoint._chart(tf, geometry)
    physical = chart["center"] + tf.matmul(
        tf.reshape(latent, (-1, PARAMETER_DIM)),
        chart["factor"],
        transpose_b=True,
    )
    physical = tf.reshape(physical, (1000, 6, CHAINS, PARAMETER_DIM))
    cold = physical[:, 0]
    source_centers = chart["source_centers"]
    windows = []
    for endpoint in WINDOW_ENDPOINTS:
        values = cold[endpoint - WINDOW_DRAWS : endpoint]
        signs = values[..., 2] < 0.0
        sign_changes = tf.reduce_sum(
            tf.cast(signs[1:] != signs[:-1], tf.int32), axis=0
        )
        sign_report = rank_normalized_split_rhat_summary(
            tf.cast(signs[..., tf.newaxis], tf.float64),
            rhat_max=RHAT_THRESHOLD,
        )
        decomposition = _between_chain_decomposition(
            tf, values, signs, source_centers
        )
        residual_report = rank_normalized_split_rhat_summary(
            decomposition.pop("residual_draws"),
            rhat_max=RHAT_THRESHOLD,
        )
        windows.append(
            {
                "endpoint": endpoint,
                "start": endpoint - WINDOW_DRAWS,
                "negative_fraction_by_chain": tf.reduce_mean(
                    tf.cast(signs, tf.float64), axis=0
                ),
                "cold_sign_changes_by_chain": sign_changes,
                "run_lengths": _longest_runs(tf, signs),
                "sign_indicator_rhat": sign_report,
                "source_center_residual_rhat": residual_report,
                "between_chain_decomposition": decomposition,
                "classification": _classify(sign_report, residual_report),
            }
        )

    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_replica_warmup_failure_diagnosis.v1",
        "status": "WARMUP_FAILURE_DECOMPOSITION_COMPLETE",
        "bindings": bindings,
        "configuration": {
            "window_endpoints": WINDOW_ENDPOINTS,
            "window_draws": WINDOW_DRAWS,
            "rhat_threshold": RHAT_THRESHOLD,
            "mode_boundary": "physical_parameter_index_2_less_than_zero",
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
            "target_evaluations": 0,
        },
        "evidence_contract": {
            "question": "is failed cold R-hat explained by mode occupancy or source-center-residual disagreement",
            "primary": "binary sign R-hat versus source-center-residual R-hat classification",
            "veto": "any bound manifest or tensor receipt mismatch, nonfinite trace, or wrong terminal status",
            "explanatory": "occupancy fractions, sign changes, dwell lengths, chain means, and sum-square decomposition",
            "nonclaim": "diagnosis cannot admit samples, validate posterior mass, prove an exact symmetry, or rank sampler repairs",
        },
        "source_centers": source_centers,
        "windows": windows,
        "terminal_classification": windows[-1]["classification"],
        "verified_inventory": {
            "r8_manifests": 50,
            "r8_tensor_receipts": 650,
            "r11_manifests": len(resumed_states),
            "r11_tensor_receipts": verified_r11_receipts,
        },
        "nonclaims": (
            "source-center subtraction is an explanatory decomposition, not an exact mode-folding map",
            "raw sign occupancy is not posterior mass authority",
            "no posterior, predictive, exhaustive-mode, superiority, or default-readiness claim",
            "nonfinite explanatory R-hat scalars serialize as null with nonfinite counts preserved",
        ),
    }
    if not bool(tf.reduce_all(tf.math.is_finite(cold)).numpy()):
        raise WarmupDiagnosisError("combined cold trace is nonfinite")
    _write_json(FINAL, payload)
    return payload


def main() -> None:
    payload = run()
    print(json.dumps({"status": payload["status"], "terminal_classification": payload["terminal_classification"]}, sort_keys=True))


if __name__ == "__main__":
    main()

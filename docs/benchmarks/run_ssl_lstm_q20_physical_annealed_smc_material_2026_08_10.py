#!/usr/bin/env python3
"""Run and aggregate the independent material annealed-SMC campaign."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-physical-annealed-smc-material-plan-2026-08-10.md")
RESULT = Path("docs/plans/bayesfilter-ssl-lstm-q20-physical-annealed-smc-material-result-2026-08-10.md")
RUNNER = Path("docs/benchmarks/run_ssl_lstm_q20_physical_annealed_smc_material_2026_08_10.py")
CHILD_RUNNER = Path("docs/benchmarks/run_ssl_lstm_q20_physical_annealed_smc_canary_2026_08_10.py")
SMC_HELPER = Path("bayesfilter/testing/annealed_smc_tf.py")
CANARY = Path("docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r1/canary.json")
AIS_MATERIAL = Path("docs/plans/artifacts/ssl-lstm-q20-physical-ais-repair-2026-08-10/r3/material.json")
OUTPUT_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2")
PROGRESS = OUTPUT_ROOT / "progress.json"
FINAL = OUTPUT_ROOT / "material.json"

CENTRAL_BATCHES = 8
SENSITIVITY_BATCHES = 2
CENTRAL_CESS = 0.80
SENSITIVITY_CESS = 0.70
CHILD_TIMEOUT_SECONDS = 900.0
RUNNER_CAP_SECONDS = 4100.0
CANARY_SHA256 = "df3ed264797e4bc6a374cd09b20a759b51ba341f7f50da1b9aa2f054918d6631"
AIS_MATERIAL_SHA256 = "1c95aa6712dd08567a7cd2b51ada5755a3de14f5ea7f50a054de5a25abac79ff"


class SMCMaterialError(RuntimeError):
    """Raised when material SMC cannot produce admissible evidence."""


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
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any], *, overwrite: bool = False) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists() and not overwrite:
        raise SMCMaterialError(f"refusing to overwrite artifact: {path}")
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(absolute)


def _write_tensor(path: Path, value: Any, tf: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise SMCMaterialError(f"refusing to overwrite artifact: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def batch_specs() -> list[Mapping[str, Any]]:
    central = [
        {
            "family": "central",
            "index": index,
            "target_ess_fraction": CENTRAL_CESS,
            "seed_domain_offset": 100000 + 10000 * index,
        }
        for index in range(CENTRAL_BATCHES)
    ]
    sensitivity = [
        {
            "family": "sensitivity",
            "index": index,
            "target_ess_fraction": SENSITIVITY_CESS,
            "seed_domain_offset": 300000 + 10000 * index,
        }
        for index in range(SENSITIVITY_BATCHES)
    ]
    return central + sensitivity


def _child_output(spec: Mapping[str, Any]) -> Path:
    return OUTPUT_ROOT / f"{spec['family']}-{int(spec['index']):02d}"


def _child_command(spec: Mapping[str, Any]) -> list[str]:
    return [
        "/home/ubuntu/anaconda3/envs/tfgpu/bin/python",
        CHILD_RUNNER.as_posix(),
        "--output-root",
        _child_output(spec).as_posix(),
        "--target-ess-fraction",
        str(float(spec["target_ess_fraction"])),
        "--seed-domain-offset",
        str(int(spec["seed_domain_offset"])),
        "--plan-file",
        PLAN.as_posix(),
        "--result-file",
        RESULT.as_posix(),
    ]


def _run_child(spec: Mapping[str, Any]) -> float:
    started = time.perf_counter()
    process = subprocess.Popen(
        _child_command(spec),
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=CHILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise SMCMaterialError(
            f"SMC child timed out: {spec['family']}-{int(spec['index']):02d}"
        )
    if return_code != 0:
        raise SMCMaterialError(
            f"SMC child failed with code {return_code}: "
            f"{spec['family']}-{int(spec['index']):02d}"
        )
    return time.perf_counter() - started


def _verify_receipt(receipt: Mapping[str, Any]) -> None:
    path = Path(str(receipt["path"]))
    if _sha(path) != str(receipt["sha256"]):
        raise SMCMaterialError(f"receipt hash mismatch: {path}")


def _stage_receipt_groups(stage: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    """Normalize current nested and historical flat stage receipt schemas."""

    receipts = stage["receipts"]
    if "pre" in receipts or "post" in receipts:
        return {
            "pre": dict(receipts.get("pre", {})),
            "post": dict(receipts.get("post", {})),
        }
    # Historical v1 files may omit same-named pre-resampling receipts because
    # post-resampling entries overwrote them in the JSON mapping.
    return {"pre": dict(receipts), "post": {}}


def _load_child(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    root = _child_output(spec)
    final_path = root / "canary.json"
    payload = json.loads(_abs(final_path).read_text(encoding="utf-8"))
    if payload.get("status") != "SMC_CANARY_PASSED":
        raise SMCMaterialError(f"child canary did not pass: {final_path}")
    if abs(
        float(payload["configuration"]["target_ess_fraction"])
        - float(spec["target_ess_fraction"])
    ) > 1.0e-12:
        raise SMCMaterialError(f"child cESS mismatch: {final_path}")
    for receipt in payload["initial_receipts"].values():
        _verify_receipt(receipt)
    stage_count = int(payload["stage_count"])
    stages = []
    receipt_count = len(payload["initial_receipts"])
    for stage_index in range(stage_count):
        stage_path = root / f"stage-{stage_index:02d}.json"
        stage = json.loads(_abs(stage_path).read_text(encoding="utf-8"))
        if int(stage["stage_index"]) != stage_index:
            raise SMCMaterialError(f"child stage index mismatch: {stage_path}")
        for group in _stage_receipt_groups(stage).values():
            for receipt in group.values():
                _verify_receipt(receipt)
                receipt_count += 1
        stages.append(stage)
    terminal = stages[-1]
    if not terminal.get("terminal_pre_resampling") or terminal.get("resampled"):
        raise SMCMaterialError(f"invalid terminal resampling policy: {final_path}")
    return {
        "spec": dict(spec),
        "payload": payload,
        "stages": stages,
        "terminal": terminal,
        "receipt_count": receipt_count,
        "final_sha256": _sha(final_path),
    }


def _load_terminal_tensor(child: Mapping[str, Any], name: str, tf: Any, dtype: Any) -> Any:
    receipt = _stage_receipt_groups(child["terminal"])["pre"][name]
    return tf.io.parse_tensor(tf.io.read_file(receipt["path"]), out_type=dtype)


def _terminal_summary(child: Mapping[str, Any], tf: Any) -> Mapping[str, Any]:
    signs = _load_terminal_tensor(child, "sign", tf, tf.bool)
    weights = _load_terminal_tensor(child, "normalized_weights", tf, tf.float64)
    roots = _load_terminal_tensor(child, "roots", tf, tf.int32)
    root_signs = _load_terminal_tensor(child, "root_signs", tf, tf.bool)
    tf.debugging.assert_near(
        tf.reduce_sum(weights), tf.constant(1.0, tf.float64), atol=1.0e-10
    )
    mass = tf.reduce_sum(weights * tf.cast(signs, tf.float64))
    unique_roots = tf.unique(roots).y
    positive_roots = tf.unique(
        tf.boolean_mask(roots, tf.logical_not(root_signs))
    ).y
    negative_roots = tf.unique(tf.boolean_mask(roots, root_signs)).y
    return {
        "negative_region_probability": mass,
        "terminal_ess_fraction": tf.convert_to_tensor(
            child["payload"]["terminal_pre_resampling_ess_fraction"], tf.float64
        ),
        "terminal_maximum_weight": tf.convert_to_tensor(
            child["payload"]["terminal_pre_resampling_maximum_weight"], tf.float64
        ),
        "unique_root_count": tf.size(unique_roots),
        "positive_root_count": tf.size(positive_roots),
        "negative_root_count": tf.size(negative_roots),
        "stage_count": tf.convert_to_tensor(child["payload"]["stage_count"], tf.int32),
        "hmc_sign_changes": tf.convert_to_tensor(
            child["payload"]["total_hmc_sign_changes"], tf.int32
        ),
    }


def _terminal_failure(
    started: float,
    completed: list[str],
    error: BaseException,
) -> None:
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_annealed_smc.material.v1",
        "status": "SMC_MATERIAL_HARNESS_FAILED",
        "decision": "STOP_FOR_HARNESS_REPAIR",
        "completed_batches": completed,
        "error_type": type(error).__name__,
        "error": str(error),
        "wall_time_seconds": time.perf_counter() - started,
    }
    _write_json(FINAL, payload)
    _write_json(
        PROGRESS,
        {**payload, "result": FINAL.as_posix()},
        overwrite=True,
    )


def run_material() -> Mapping[str, Any]:
    started = time.perf_counter()
    if _abs(FINAL).exists():
        raise SMCMaterialError("refusing to overwrite material SMC result")
    if _sha(CANARY) != CANARY_SHA256:
        raise SMCMaterialError("SMC canary identity mismatch")
    if _sha(AIS_MATERIAL) != AIS_MATERIAL_SHA256:
        raise SMCMaterialError("AIS comparator identity mismatch")
    specs = batch_specs()
    completed: list[str] = []
    runtimes: dict[str, float] = {}
    _write_json(
        PROGRESS,
        {
            "status": "SMC_MATERIAL_RUNNING",
            "completed_batches": completed,
            "total_batches": len(specs),
        },
        overwrite=True,
    )
    try:
        for spec in specs:
            if time.perf_counter() - started >= RUNNER_CAP_SECONDS:
                raise SMCMaterialError("SMC material runner reached wall cap")
            name = f"{spec['family']}-{int(spec['index']):02d}"
            runtimes[name] = _run_child(spec)
            child = _load_child(spec)
            completed.append(name)
            _write_json(
                PROGRESS,
                {
                    "status": "SMC_MATERIAL_RUNNING",
                    "completed_batches": completed,
                    "total_batches": len(specs),
                    "elapsed_seconds": time.perf_counter() - started,
                    "last_child_sha256": child["final_sha256"],
                },
                overwrite=True,
            )
    except BaseException as error:
        _terminal_failure(started, completed, error)
        raise

    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.testing.importance_sampling_tf import independent_batch_interval

    children = [_load_child(spec) for spec in specs]
    summaries = [_terminal_summary(child, tf) for child in children]
    central = summaries[:CENTRAL_BATCHES]
    sensitivity = summaries[CENTRAL_BATCHES:]
    central_estimates = tf.stack(
        [summary["negative_region_probability"] for summary in central]
    )
    sensitivity_estimates = tf.stack(
        [summary["negative_region_probability"] for summary in sensitivity]
    )
    interval = independent_batch_interval(central_estimates)
    central_mean = tf.reduce_mean(central_estimates)
    sensitivity_mean = tf.reduce_mean(sensitivity_estimates)
    cess_difference = tf.abs(central_mean - sensitivity_mean)
    per_central_gates = [
        {
            "terminal_ess_fraction_at_least_0.50": bool(
                (summary["terminal_ess_fraction"] >= 0.50).numpy()
            ),
            "terminal_maximum_weight_at_most_0.05": bool(
                (summary["terminal_maximum_weight"] <= 0.05).numpy()
            ),
            "unique_root_fraction_at_least_0.30": bool(
                (tf.cast(summary["unique_root_count"], tf.float64) / 100.0 >= 0.30).numpy()
            ),
            "at_least_10_positive_roots": int(summary["positive_root_count"].numpy()) >= 10,
            "at_least_10_negative_roots": int(summary["negative_root_count"].numpy()) >= 10,
        }
        for summary in central
    ]
    gates = {
        "all_children_passed_mechanics": all(
            child["payload"]["status"] == "SMC_CANARY_PASSED" for child in children
        ),
        "all_central_terminal_weight_and_ancestry_gates": all(
            all(batch.values()) for batch in per_central_gates
        ),
        "central_interval_half_width_at_most_0.08": bool(
            (interval["half_width"] <= 0.08).numpy()
        ),
        "cess_mass_difference_at_most_0.08": bool((cess_difference <= 0.08).numpy()),
        "wall_time_within_4200_seconds": time.perf_counter() - started <= 4200.0,
    }
    passed = all(gates.values())
    receipt_count = sum(int(child["receipt_count"]) for child in children)
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_annealed_smc.material.v1",
        "status": "SMC_WEIGHT_EVIDENCE_PASSED" if passed else "SMC_WEIGHT_EVIDENCE_FAILED",
        "decision": (
            "TWO_KNOWN_REGION_SMC_WEIGHT_AUTHORITY_VIABLE"
            if passed
            else "STOP_SMC_WEIGHT_PROMOTION_AND_REPAIR"
        ),
        "configuration": {
            "central_batches": CENTRAL_BATCHES,
            "sensitivity_batches": SENSITIVITY_BATCHES,
            "particles_per_batch": 100,
            "central_target_ess_fraction": CENTRAL_CESS,
            "sensitivity_target_ess_fraction": SENSITIVITY_CESS,
            "hmc_step_size": 0.03,
            "hmc_leapfrog_steps": 4,
            "terminal_policy": "beta_one_pre_resampling",
        },
        "gates": gates,
        "per_central_gates": per_central_gates,
        "central": {
            "batch_estimates": central_estimates,
            "mean_negative_region_probability": central_mean,
            "independent_batch_interval": interval,
            "terminal_ess_fractions": tf.stack(
                [summary["terminal_ess_fraction"] for summary in central]
            ),
            "terminal_maximum_weights": tf.stack(
                [summary["terminal_maximum_weight"] for summary in central]
            ),
            "unique_root_counts": tf.stack(
                [summary["unique_root_count"] for summary in central]
            ),
            "positive_root_counts": tf.stack(
                [summary["positive_root_count"] for summary in central]
            ),
            "negative_root_counts": tf.stack(
                [summary["negative_root_count"] for summary in central]
            ),
            "stage_counts": tf.stack([summary["stage_count"] for summary in central]),
            "hmc_sign_changes": tf.stack(
                [summary["hmc_sign_changes"] for summary in central]
            ),
        },
        "sensitivity": {
            "batch_estimates": sensitivity_estimates,
            "mean_negative_region_probability": sensitivity_mean,
            "difference_from_central": cess_difference,
            "terminal_ess_fractions": tf.stack(
                [summary["terminal_ess_fraction"] for summary in sensitivity]
            ),
            "unique_root_counts": tf.stack(
                [summary["unique_root_count"] for summary in sensitivity]
            ),
        },
        "child_runs": [
            {
                "family": child["spec"]["family"],
                "index": child["spec"]["index"],
                "target_ess_fraction": child["spec"]["target_ess_fraction"],
                "seed_domain_offset": child["spec"]["seed_domain_offset"],
                "runtime_seconds": runtimes[
                    f"{child['spec']['family']}-{int(child['spec']['index']):02d}"
                ],
                "terminal_sha256": child["final_sha256"],
                "receipt_count": child["receipt_count"],
            }
            for child in children
        ],
        "aggregate_receipts": {
            "central_estimates": _write_tensor(
                OUTPUT_ROOT / "central-estimates.tftensor", central_estimates, tf
            ),
            "sensitivity_estimates": _write_tensor(
                OUTPUT_ROOT / "sensitivity-estimates.tftensor", sensitivity_estimates, tf
            ),
        },
        "verified_child_receipt_count": receipt_count,
        "run_manifest": {
            "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "git_dirty": bool(subprocess.run(("git", "status", "--porcelain"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
            "jit_compile": True,
            "cpu_ids": list(range(100)),
            "wall_time_seconds": time.perf_counter() - started,
            "artifact_root": OUTPUT_ROOT.as_posix(),
            "plan_file": PLAN.as_posix(),
            "result_file": RESULT.as_posix(),
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "child_runner": _sha(CHILD_RUNNER),
                "smc_helper": _sha(SMC_HELPER),
                "canary": _sha(CANARY),
                "ais_material": _sha(AIS_MATERIAL),
            },
        },
        "nonclaims": (
            "the authority is limited to the two known proposal-supported sign regions",
            "passing does not prove exhaustive mode discovery or full-posterior correctness",
            "global HMC stationarity and predictive validity remain separate gates",
        ),
    }
    _write_json(FINAL, payload)
    _write_json(
        PROGRESS,
        {
            "status": payload["status"],
            "completed_batches": completed,
            "total_batches": len(specs),
            "elapsed_seconds": time.perf_counter() - started,
            "result": FINAL.as_posix(),
        },
        overwrite=True,
    )
    return payload


if __name__ == "__main__":
    material = run_material()
    print(json.dumps({"status": material["status"]}, sort_keys=True))

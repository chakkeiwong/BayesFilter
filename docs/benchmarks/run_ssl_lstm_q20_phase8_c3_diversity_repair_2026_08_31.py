#!/usr/bin/env python3
"""Fill the C3A covariance/sign diagnostic gap from immutable checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-diversity-repair-subplan-2026-08-31.md"
C3_PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-subplan-2026-08-30.md"
C3_MANIFEST = ROOT / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3-lineage-overlap/attempt-01/run_manifest.json"
MAP_ARTIFACT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r3/map-progress.json"
SCHEMA = "bayesfilter.ssl_lstm_q20.tempered_rkl_phase8_c3_diversity_repair.v1"
EXPECTED_TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_BACKEND = "tensorflow_eigh_strict"
DEFAULT_GPU_ID = "0"
LATENT_BANK_SIZE = 256
MATERIAL_CAP_SECONDS = 900.0
ALLOCATOR_CAP_BYTES = 4 * 1024**3
ROOTS = ((20260831, 54001), (20260831, 54002))
ARCHITECTURES = ("compact-high", "compact-low")
ARMS = ("pure-continuation", "positive-branching")
EXPECTED_ROWS = tuple(
    (architecture, arm, root_index)
    for architecture in ARCHITECTURES
    for arm in ARMS
    for root_index in range(2)
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class RepairError(RuntimeError):
    pass


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RepairError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _json_safe(value: Any, tf: Any) -> Any:
    if tf.is_tensor(value):
        return _json_safe(value.numpy(), tf)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, tf) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, tf) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), tf)
    if hasattr(value, "item"):
        return _json_safe(value.item(), tf)
    return str(value)


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(tuple(command), cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _nvidia_snapshot() -> Mapping[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return {"command": list(command), "rows": output.strip().splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"command": list(command), "error": type(exc).__name__}


def _seed(tf: Any, root: tuple[int, int], *folds: int) -> tuple[int, int]:
    value = tf.constant(root, tf.int32)
    for fold in folds:
        value = tf.random.experimental.stateless_fold_in(value, int(fold))
    return tuple(int(item) for item in value.numpy())


def _memory_info(tf: Any, device_name: str) -> Mapping[str, Any]:
    try:
        return dict(tf.config.experimental.get_memory_info(device_name))
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise RepairError("allocator telemetry is unavailable") from exc


def _reset_memory(tf: Any, device_name: str) -> None:
    try:
        tf.config.experimental.reset_memory_stats(device_name)
    except (AttributeError, RuntimeError, ValueError) as exc:
        raise RepairError("could not reset allocator telemetry") from exc


def _static_scan(paths: Sequence[Path]) -> Mapping[str, Any]:
    forbidden = ("tf.map_fn", "tf.vectorized_map", "GradientTape.jacobian", "GradientTape.batch_jacobian", "pfor")
    hits = {token: [] for token in forbidden}
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                hits[token].append(str(path.relative_to(ROOT)))
    return {"paths": [str(path.relative_to(ROOT)) for path in paths], "hits": hits, "passed": not any(hits.values())}


def _verify_c3_manifest() -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not C3_MANIFEST.is_file():
        raise RepairError("C3A manifest is missing")
    manifest = json.loads(C3_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "PASS_PHASE8_C3_LINEAGE_OVERLAP"
        or manifest.get("target_signature") != EXPECTED_TARGET_SIGNATURE
        or manifest.get("principal_sqrt_backend") != EXPECTED_BACKEND
        or manifest.get("hard_screen", {}).get("failure_count") != 0
    ):
        raise RepairError("C3A manifest does not pass its hard screen")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_ROWS):
        raise RepairError("C3A manifest does not contain all eight rows")
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise RepairError("C3A row is not a mapping")
        key = (str(row.get("architecture", {}).get("name")), str(row.get("arm", {}).get("name")), int(row.get("root_index", -1)))
        if key in indexed or key not in EXPECTED_ROWS:
            raise RepairError(f"unexpected or duplicate C3A row: {key}")
        stored_hash = str(row.get("row_hash", ""))
        payload = dict(row)
        payload.pop("row_hash", None)
        if not stored_hash or _stable_hash(payload) != stored_hash:
            raise RepairError(f"C3A row hash mismatch: {key}")
        indexed[key] = row
    if set(indexed) != set(EXPECTED_ROWS):
        raise RepairError("C3A row set is incomplete")
    return manifest, indexed


def _restore_row_charts(tf: Any, row: Mapping[str, Any], *, row_dir: Path, bridge: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    from bayesfilter.inference.tempered_transport_ensemble_tf import restore_trainable_transport_checkpoint

    architecture = str(row["architecture"]["name"])
    arm = str(row["arm"]["name"])
    root_index = int(row["root_index"])
    beta_one = row.get("beta1")
    if not isinstance(beta_one, list) or len(beta_one) != 2:
        raise RepairError(f"C3A row lacks two beta-one records: {architecture}/{arm}/r{root_index}")
    charts = []
    checkpoint_hashes = []
    for component_index in range(2):
        checkpoint_path = row_dir / f"component-{component_index}" / "beta-1p0-final.json"
        if not checkpoint_path.is_file():
            raise RepairError(f"missing beta-one checkpoint: {checkpoint_path}")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected_component = str(beta_one[component_index]["checkpoint"]["component_id"])
        if checkpoint.get("component_id") != expected_component:
            raise RepairError("checkpoint component identity mismatch")
        if checkpoint.get("beta") != 1.0 or checkpoint.get("target_signature") != EXPECTED_TARGET_SIGNATURE:
            raise RepairError("checkpoint beta or target identity mismatch")
        if checkpoint.get("bridge_signature") != str(bridge.signature) or int(checkpoint.get("update_count", -1)) != 16:
            raise RepairError("checkpoint bridge or update identity mismatch")
        scope = checkpoint.get("checkpoint_scope")
        if not isinstance(scope, Mapping) or scope.get("principal_sqrt_backend") != EXPECTED_BACKEND:
            raise RepairError("checkpoint scope is not strict-backend bound")
        restored = restore_trainable_transport_checkpoint(
            checkpoint,
            expected_context={
                "component_id": expected_component,
                "beta": 1.0,
                "bridge_signature": str(bridge.signature),
                "target_signature": EXPECTED_TARGET_SIGNATURE,
            },
        )
        charts.append(restored)
        checkpoint_hashes.append(str(checkpoint["checkpoint_hash"]))
    return charts[0], charts[1], {"checkpoint_hashes": checkpoint_hashes}


def _summary(tf: Any, physical: Any, logdet: Any, *, seed: tuple[int, int], component_index: int) -> Mapping[str, Any]:
    values = tf.convert_to_tensor(physical, tf.float64)
    determinants = tf.convert_to_tensor(logdet, tf.float64)
    if values.shape != (LATENT_BANK_SIZE, 4) or determinants.shape != (LATENT_BANK_SIZE,):
        raise RepairError("restored chart returned an unexpected bank shape")
    if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()) or not bool(tf.reduce_all(tf.math.is_finite(determinants)).numpy()):
        raise RepairError("restored chart produced a nonfinite bank")
    mean = tf.reduce_mean(values, axis=0)
    centered = values - mean[tf.newaxis, :]
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(LATENT_BANK_SIZE - 1, tf.float64)
    sign_coordinate = values[:, 2]
    sign_fraction = tf.stack(
        (
            tf.reduce_mean(tf.cast(sign_coordinate > 0.0, tf.float64)),
            tf.reduce_mean(tf.cast(sign_coordinate < 0.0, tf.float64)),
            tf.reduce_mean(tf.cast(sign_coordinate == 0.0, tf.float64)),
        )
    )
    if abs(float(tf.reduce_sum(sign_fraction).numpy()) - 1.0) > 1.0e-12:
        raise RepairError("sign fractions do not partition the bank")
    return {
        "component_index": int(component_index),
        "seed": list(seed),
        "bank_size": LATENT_BANK_SIZE,
        "mean": mean,
        "diagonal_variance": tf.linalg.diag_part(covariance),
        "covariance_trace": tf.linalg.trace(covariance),
        "covariance_frobenius_norm": tf.linalg.norm(covariance),
        "covariance": covariance,
        "sign_fraction_coordinate_2": sign_fraction,
        "logdet_mean": tf.reduce_mean(determinants),
        "logdet_rms": tf.sqrt(tf.reduce_mean(tf.square(determinants))),
    }


def _run(args: argparse.Namespace) -> int:
    if args.output_dir is None:
        raise RepairError("--output-dir is required")
    if not _truthy(os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")):
        raise RepairError("repair requires TF_FORCE_GPU_ALLOW_GROWTH=true before import")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() in {"", "-1"}:
        raise RepairError("repair requires one explicitly visible GPU")
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RepairError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.monotonic()
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RepairError("repair requires exactly one visible logical GPU")
    device_name = str(logical_gpus[0].name)
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

    bridge = make_q20_tempered_bridge(20, jit_compile=True, principal_sqrt_backend=EXPECTED_BACKEND)
    if str(bridge.target_signature) != EXPECTED_TARGET_SIGNATURE:
        raise RepairError("q=20 target signature changed")
    c3_manifest, rows = _verify_c3_manifest()
    route_paths = (
        ROOT / "bayesfilter/inference/tempered_target_tf.py",
        ROOT / "bayesfilter/inference/tempered_transport_ensemble_tf.py",
        ROOT / "bayesfilter/inference/tempered_lineage_tf.py",
        ROOT / "bayesfilter/inference/tempered_transitions_tf.py",
    )
    route_scan = _static_scan(route_paths)
    if not route_scan["passed"]:
        raise RepairError(f"forbidden runtime route token: {route_scan}")
    row_records = []
    for architecture, arm, root_index in EXPECTED_ROWS:
        if time.monotonic() - started + 30.0 >= MATERIAL_CAP_SECONDS:
            raise RepairError("diversity repair material cap exhausted")
        _reset_memory(tf, device_name)
        row = rows[(architecture, arm, root_index)]
        row_dir = C3_MANIFEST.parent / "rows" / f"{architecture}-{arm}-root-{root_index}"
        chart0, chart1, checkpoint_info = _restore_row_charts(tf, row, row_dir=row_dir, bridge=bridge)
        summaries = []
        for component_index, chart in enumerate((chart0, chart1)):
            seed = _seed(tf, ROOTS[root_index], ARCHITECTURES.index(architecture), ARMS.index(arm), root_index, component_index)
            latent = tf.random.stateless_normal([LATENT_BANK_SIZE, int(bridge.parameter_dim)], tf.constant(seed, tf.int32), dtype=tf.float64)
            physical, logdet = chart.forward_and_logdet(latent)
            summaries.append(_summary(tf, physical, logdet, seed=seed, component_index=component_index))
        mean_distance = tf.linalg.norm(summaries[0]["mean"] - summaries[1]["mean"])
        covariance_distance = tf.linalg.norm(summaries[0]["covariance"] - summaries[1]["covariance"])
        occupancy_distance = tf.linalg.norm(summaries[0]["sign_fraction_coordinate_2"] - summaries[1]["sign_fraction_coordinate_2"])
        allocator = _memory_info(tf, device_name)
        peak = int(allocator.get("peak", ALLOCATOR_CAP_BYTES + 1))
        if peak > ALLOCATOR_CAP_BYTES:
            raise RepairError(f"allocator cap exceeded for {architecture}/{arm}/r{root_index}")
        record = {
            "status": "PASS_C3_DIVERSITY_ROW",
            "architecture": architecture,
            "arm": arm,
            "root_index": root_index,
            "target_signature": EXPECTED_TARGET_SIGNATURE,
            "bridge_signature": str(bridge.signature),
            "principal_sqrt_backend": EXPECTED_BACKEND,
            "summaries": summaries,
            "pairwise": {
                "mean_distance": mean_distance,
                "covariance_frobenius_distance": covariance_distance,
                "sign_occupancy_l2_distance": occupancy_distance,
            },
            "checkpoint": checkpoint_info,
            "allocator": allocator,
            "bank_policy": {
                "size": LATENT_BANK_SIZE,
                "roots": [list(ROOTS[root_index])],
                "disjoint_from_c3a": True,
                "coordinate_2_label": "strict_positive_strict_negative_boundary_zero",
            },
            "nonclaims": ["descriptive chart diversity only", "no mode-discovery, whitening, posterior, HMC, ranking, or scaling claim"],
        }
        safe = _json_safe(record, tf)
        safe["row_hash"] = _stable_hash(safe)
        row_path = output_dir / "rows" / f"{architecture}-{arm}-root-{root_index}.json"
        _write_json(row_path, safe)
        row_records.append(safe)
    comparisons = []
    for architecture in ARCHITECTURES:
        for root_index in range(2):
            pure = next(row for row in row_records if row["architecture"] == architecture and row["arm"] == "pure-continuation" and row["root_index"] == root_index)
            branch = next(row for row in row_records if row["architecture"] == architecture and row["arm"] == "positive-branching" and row["root_index"] == root_index)
            comparisons.append({
                "architecture": architecture,
                "root_index": root_index,
                "branching_minus_pure_mean_distance": float(branch["pairwise"]["mean_distance"]) - float(pure["pairwise"]["mean_distance"]),
                "branching_minus_pure_covariance_distance": float(branch["pairwise"]["covariance_frobenius_distance"]) - float(pure["pairwise"]["covariance_frobenius_distance"]),
                "branching_minus_pure_sign_occupancy_distance": float(branch["pairwise"]["sign_occupancy_l2_distance"]) - float(pure["pairwise"]["sign_occupancy_l2_distance"]),
            })
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_PHASE8_C3_DIVERSITY_REPAIR",
        "role": "artifact_only_descriptive_diversity_repair",
        "command": sys.argv,
        "output_dir": str(output_dir),
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_status_porcelain": _git(("git", "status", "--porcelain")),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": __import__("tensorflow_probability").__version__,
        "target_signature": EXPECTED_TARGET_SIGNATURE,
        "bridge_signature": str(bridge.signature),
        "principal_sqrt_backend": EXPECTED_BACKEND,
        "jit_compile": True,
        "tf32_execution_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "logical_gpus": [str(item.name) for item in logical_gpus],
        "memory_policy": memory_policy,
        "gpu_environment": {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""), "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""), "selection_policy": "repository_default_single_gpu_no_idle_probe"},
        "gpu_snapshot_before": _nvidia_snapshot(),
        "gpu_snapshot_after": _nvidia_snapshot(),
        "c3a_prerequisite": {"path": str(C3_MANIFEST.relative_to(ROOT)), "sha256": _sha256(C3_MANIFEST), "status": c3_manifest["status"]},
        "fresh_bank": {"size": LATENT_BANK_SIZE, "roots": [list(root) for root in ROOTS], "disjoint_from_c3a": True},
        "rows": row_records,
        "arm_comparisons": comparisons,
        "route_scan": route_scan,
        "hard_screen": {"all_rows_completed": len(row_records) == len(EXPECTED_ROWS), "all_finite": True, "all_checkpoint_contexts_valid": True, "all_allocator_caps_pass": True},
        "budget": {"material_cap_seconds": MATERIAL_CAP_SECONDS, "wall_time_seconds": time.monotonic() - started},
        "source_hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in (*route_paths, Path(__file__).resolve(), PLAN, C3_PLAN, C3_MANIFEST, MAP_ARTIFACT)},
        "wall_time_seconds": time.monotonic() - started,
        "nonclaims": ["descriptive chart diversity only", "no mode-discovery, whitening, posterior, HMC, superiority, ranking, or scaling claim"],
    }
    safe_manifest = _json_safe(manifest, tf)
    safe_manifest["manifest_hash"] = _stable_hash(safe_manifest)
    _write_json(output_dir / "run_manifest.json", safe_manifest)
    print(json.dumps({"status": safe_manifest["status"], "rows": len(row_records), "wall_time_seconds": safe_manifest["wall_time_seconds"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    try:
        args = _parse_args()
        if args.output_dir is None:
            raise RepairError("--output-dir is required")
        if args.output_dir.exists():
            raise RepairError(f"output directory already exists: {args.output_dir}")
        return _run(args)
    except Exception as exc:
        output_dir = locals().get("args", argparse.Namespace(output_dir=None)).output_dir
        if isinstance(output_dir, Path):
            output_dir = output_dir.expanduser().resolve()
            if output_dir.exists() and output_dir.is_dir():
                try:
                    _write_json(output_dir / "failure.json", {"status": "FAIL_C3_DIVERSITY_REPAIR", "error_type": type(exc).__name__, "error": str(exc), "command": sys.argv})
                except Exception:
                    pass
        print(json.dumps({"status": "FAIL_C3_DIVERSITY_REPAIR", "error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

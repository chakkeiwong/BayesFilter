#!/usr/bin/env python3
"""Build an untouched q=20 target reference and compare seed-B draws."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-seed-b-posterior-reference-plan-2026-08-07.md"
SCRIPT = Path(__file__).resolve()
ARCHIVE_ROOT = ROOT / "docs/plans/artifacts/ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r2/sequential/archive"
ARCHIVE_MANIFEST = ARCHIVE_ROOT / "seed-b-terminal-manifest.json"
SUMMARY = ARCHIVE_ROOT.parent / "summary.json"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
SCHEMA = "bayesfilter.ssl_lstm.q20_seed_b_posterior_reference.v1"
REFERENCE_SCHEMA = "bayesfilter.ssl_lstm.q20_seed_b_posterior_reference_quadrature.v1"
WORKERS = 25
ROWS_PER_WORKER = 4
ORDERS = (7, 9, 11, 13)
SCALES = (1.0, 1.5)
MAX_REFERENCE_SECONDS = 18_000.0
MARGINS = {"mean": 0.10, "sd": 0.10, "covariance": 0.15, "quantile": 0.15}
STABILITY = {"mean": 0.02, "sd": 0.02, "covariance": 0.03, "quantile": 0.03}
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_BLOCK = 32
BOOTSTRAP_SEED = (20260807, 87001)
PRIOR_CENTER_VALUES = (0.35, -0.08, 0.65, 0.05)
PRIOR_SD = 4.0


class CampaignError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).rstrip(b"\n")).hexdigest()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise CampaignError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe(v) for v in value]
    if hasattr(value, "numpy"):
        return safe(value.numpy().tolist())
    if isinstance(value, (float, int, bool, str)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value)!r}")


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, Mapping):
        raise CampaignError(f"expected JSON object: {path}")
    return payload


def _check_cpu_tf(tf: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise CampaignError("reference requires CUDA_VISIBLE_DEVICES=-1")
    if tf.config.list_physical_devices("GPU"):
        raise CampaignError("reference process can see a GPU")
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)


def _verify_archive(*, parse_tensor_values: bool = False) -> Mapping[str, Any]:
    summary = _load_json(SUMMARY)
    if summary.get("schema") != "bayesfilter.ssl_lstm.q20_seed_b_terminal_sequential_hmc.v1":
        raise CampaignError("sequential summary schema mismatch")
    diagnostics = summary.get("diagnostics", {})
    if diagnostics.get("hard_vetoes") != []:
        raise CampaignError("seed-B sequential summary has hard vetoes")
    provenance = summary.get("provenance", {})
    if provenance.get("target_signature") != TARGET_SIGNATURE:
        raise CampaignError("seed-B summary target signature mismatch")
    if provenance.get("target_adapter_signature") != ADAPTER_SIGNATURE:
        raise CampaignError("seed-B summary adapter signature mismatch")
    manifest = _load_json(ARCHIVE_MANIFEST)
    if manifest.get("schema") != "bayesfilter.neutra.sequential_hmc_result.v1":
        raise CampaignError("archive manifest schema mismatch")
    chunks = manifest.get("retained_chunks")
    if not isinstance(chunks, list) or len(chunks) != 2:
        raise CampaignError("expected exactly two retained chunks")
    if manifest.get("warmup_excluded_from_posterior") is not True:
        raise CampaignError("warm-up exclusion is not recorded")
    for chunk in chunks:
        receipt = chunk.get("sample_receipt")
        if not isinstance(receipt, Mapping):
            raise CampaignError("retained sample receipt missing")
        path = Path(str(receipt["path"]))
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists() or sha256(path) != str(receipt.get("sha256")):
            raise CampaignError(f"retained sample hash mismatch: {path}")
        if list(receipt.get("shape", ())) != [500, 4, 4] or receipt.get("dtype") != "float64":
            raise CampaignError("retained sample shape/dtype mismatch")
        if parse_tensor_values:
            import tensorflow as tf
            tensor = tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64)
            tf.debugging.assert_all_finite(tensor, "retained seed-B draw")
    return {"summary": summary, "manifest": manifest}


def _gh_nodes_weights(tf: Any, order: int) -> tuple[Any, Any]:
    if order < 2:
        raise CampaignError("Gauss-Hermite order must be at least two")
    n = int(order)
    diagonal = tf.zeros([n], tf.float64)
    off = tf.sqrt(tf.cast(tf.range(1, n), tf.float64) / 2.0)
    jacobi = tf.linalg.diag(diagonal)
    jacobi = jacobi + tf.linalg.diag(off, k=1) + tf.linalg.diag(off, k=-1)
    eigenvalues, eigenvectors = tf.linalg.eigh(jacobi)
    # E[f(Z)] for Z~N(0,1): nodes=sqrt(2)*roots(H_n), weights=v_0^2.
    return tf.sqrt(tf.constant(2.0, tf.float64)) * eigenvalues, tf.square(eigenvectors[0, :])


def _mesh_quadrature(tf: Any, center: Any, factor: Any, order: int, scale: float) -> tuple[Any, Any, Any]:
    nodes, weights = _gh_nodes_weights(tf, order)
    axes = tf.meshgrid(nodes, nodes, nodes, nodes, indexing="ij")
    wmesh = tf.meshgrid(weights, weights, weights, weights, indexing="ij")
    standard = tf.stack([tf.reshape(axis, [-1]) for axis in axes], axis=1)
    product_weights = tf.ones([tf.size(wmesh[0])], tf.float64)
    for weight_axis in wmesh:
        product_weights = product_weights * tf.reshape(weight_axis, [-1])
    points = center[tf.newaxis, :] + tf.cast(scale, tf.float64) * tf.matmul(standard, factor, transpose_b=True)
    sign, log_abs_det = tf.linalg.slogdet(factor)
    tf.debugging.assert_equal(tf.abs(sign), tf.constant(1.0, tf.float64))
    log_proposal = (
        -0.5 * tf.reduce_sum(tf.square(standard), axis=1)
        - 2.0 * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
        - log_abs_det
        - 4.0 * tf.math.log(tf.cast(scale, tf.float64))
    )
    return points, product_weights, log_proposal


def _target_value_score(target: Any, theta: Any) -> tuple[Any, Any]:
    import tensorflow as tf
    value, score = target.log_prob_and_grad(theta)
    return tf.convert_to_tensor(value, tf.float64), tf.convert_to_tensor(score, tf.float64)


def _map_search(tf: Any, target: Any, *, progress_path: Path | None = None) -> Mapping[str, Any]:
    import tensorflow_probability as tfp
    center = tf.constant(PRIOR_CENTER_VALUES, tf.float64)
    starts = [center]
    for axis in range(4):
        for sign in (-1.0, 1.0):
            direction = tf.one_hot(axis, 4, dtype=tf.float64)
            starts.append(center + tf.cast(sign * PRIOR_SD, tf.float64) * direction)
    rows = []
    def objective_and_grad(position: Any) -> tuple[Any, Any]:
        value, score = _target_value_score(target, tf.reshape(position, [4]))
        return -value, -score
    compiled = tf.function(objective_and_grad, jit_compile=True, reduce_retracing=False)
    for index, start in enumerate(starts):
        result = tfp.optimizer.lbfgs_minimize(
            compiled,
            initial_position=start,
            max_iterations=200,
            tolerance=tf.constant(1.0e-10, tf.float64),
            parallel_iterations=1,
        )
        position = tf.reshape(tf.convert_to_tensor(result.position, tf.float64), [4])
        value, score = _target_value_score(target, position)
        rows.append({
            "start_index": index,
            "start": safe(start),
            "position": safe(position),
            "log_prob": float(value.numpy()),
            "score_inf_norm": float(tf.reduce_max(tf.abs(score)).numpy()),
            "converged": bool(result.converged.numpy()),
            "failed": bool(result.failed.numpy()),
            "iterations": int(result.num_iterations.numpy()),
        })
        if progress_path is not None:
            write_json(progress_path, {"schema": SCHEMA, "status": "MAP_DIAGNOSTIC_RUNNING", "completed_starts": len(rows), "starts": rows}, replace=True)
    stationary = [row for row in rows if not row["failed"] and math.isfinite(float(row["log_prob"])) and math.isfinite(float(row["score_inf_norm"])) and float(row["score_inf_norm"]) <= 1.0e-5]
    optimizer_clean = [row for row in stationary if row["converged"]]
    if len(stationary) < 5:
        return {"valid": False, "vetoes": ["fewer_than_five_stationary_map_starts"], "starts": rows, "stationary_count": len(stationary), "optimizer_clean_stationary_count": len(optimizer_clean)}
    best = max(stationary, key=lambda row: float(row["log_prob"]))
    best_value = float(best["log_prob"])
    best_position = tf.constant(best["position"], tf.float64)
    for row in stationary:
        row["distance_to_best_inf"] = float(tf.reduce_max(tf.abs(tf.constant(row["position"], tf.float64) - best_position)).numpy())
        row["log_prob_gap_to_best"] = best_value - float(row["log_prob"])
        row["matches_best_basin"] = bool(row["distance_to_best_inf"] <= 1.0e-4 and row["log_prob_gap_to_best"] <= 1.0e-6)
    matches = [row for row in stationary if row["matches_best_basin"]]
    clean_matches = [row for row in matches if row["converged"]]
    covered_axes = {
        (int(row["start_index"]) - 1) // 2
        for row in matches
        if int(row["start_index"]) > 0
    }
    competing = [row for row in stationary if not row["matches_best_basin"] and float(row["log_prob_gap_to_best"]) < 20.0]
    vetoes = []
    if len(clean_matches) < 5:
        vetoes.append("fewer_than_five_optimizer_clean_map_starts")
    if not any(int(row["start_index"]) == 0 for row in clean_matches):
        vetoes.append("prior_center_start_not_optimizer_clean_in_best_basin")
    clean_covered_axes = {(int(row["start_index"]) - 1) // 2 for row in clean_matches if int(row["start_index"]) > 0}
    if clean_covered_axes != set(range(4)):
        vetoes.append("optimizer_clean_axial_coverage_incomplete")
    if competing:
        vetoes.append("competing_stationary_mode_within_20_log_units")
    return {"valid": not vetoes, "vetoes": vetoes, "starts": rows, "stationary_count": len(stationary), "optimizer_clean_stationary_count": len(optimizer_clean), "best_basin_count": len(matches), "optimizer_clean_best_basin_count": len(clean_matches), "covered_axes": sorted(covered_axes), "optimizer_clean_covered_axes": sorted(clean_covered_axes), "map": best}


def _hessian(tf: Any, target: Any, center: Any) -> Mapping[str, Any]:
    values = []
    matrices = []
    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        columns = []
        for axis in range(4):
            direction = tf.one_hot(axis, 4, dtype=tf.float64) * tf.cast(step, tf.float64)
            _vp, sp = _target_value_score(target, center + direction)
            _vm, sm = _target_value_score(target, center - direction)
            columns.append((sp - sm) / tf.cast(2.0 * step, tf.float64))
        hessian = tf.stack(columns, axis=1)
        precision = -0.5 * (hessian + tf.transpose(hessian))
        eigenvalues, eigenvectors = tf.linalg.eigh(precision)
        if bool(tf.reduce_any(eigenvalues <= 0.0).numpy()) or not bool(tf.reduce_all(tf.math.is_finite(precision)).numpy()):
            raise CampaignError("negative Hessian is not finite SPD")
        values.append({"step": step, "eigenvalues": safe(eigenvalues), "precision": safe(precision)})
        matrices.append(precision)
    delta = tf.linalg.norm(matrices[-1] - matrices[-2], ord="euclidean") / tf.linalg.norm(matrices[-1], ord="euclidean")
    if float(delta.numpy()) > 1.0e-3:
        raise CampaignError("negative Hessian is not stable across differencing steps")
    eigenvalues, eigenvectors = tf.linalg.eigh(matrices[-1])
    factor = tf.matmul(eigenvectors, tf.linalg.diag(tf.math.rsqrt(eigenvalues)))
    covariance = tf.matmul(factor, factor, transpose_b=True)
    return {"steps": values, "relative_last_step_delta": float(delta.numpy()), "factor": safe(factor), "covariance": safe(covariance)}


def _stats(tf: Any, points: Any, log_weights: Any) -> Mapping[str, Any]:
    normalized = tf.nn.softmax(log_weights)
    mean = tf.reduce_sum(points * normalized[:, tf.newaxis], axis=0)
    centered = points - mean[tf.newaxis, :]
    covariance = tf.einsum("n,ni,nj->ij", normalized, centered, centered)
    sd = tf.sqrt(tf.linalg.diag_part(covariance))
    # Weighted quantiles use the sorted cumulative normalized weights.
    transposed = tf.transpose(points)
    order = tf.argsort(transposed, axis=1)
    sorted_points = tf.gather(transposed, order, axis=1, batch_dims=1)
    sorted_weights = tf.gather(tf.broadcast_to(normalized[tf.newaxis, :], tf.shape(transposed)), order, axis=1, batch_dims=1)
    cumulative = tf.cumsum(sorted_weights, axis=1)
    quantiles = []
    for probability in (0.05, 0.50, 0.95):
        mask = cumulative >= tf.cast(probability, tf.float64)
        first = tf.argmax(tf.cast(mask, tf.int32), axis=1, output_type=tf.int32)
        quantiles.append(tf.gather(sorted_points, first, axis=1, batch_dims=1, name="weighted_quantile"))
    return {"mean": safe(mean), "sd": safe(sd), "covariance": safe(covariance), "quantiles": safe(tf.stack(quantiles, axis=0)), "weight_ess": float((1.0 / tf.reduce_sum(tf.square(normalized))).numpy()), "max_weight": float(tf.reduce_max(normalized).numpy()), "log_normalizer_relative": float(tf.reduce_logsumexp(log_weights).numpy())}


def _stability(tf: Any, left: Mapping[str, Any], right: Mapping[str, Any]) -> Mapping[str, float]:
    left_sd = tf.constant(left["sd"], tf.float64)
    right_sd = tf.constant(right["sd"], tf.float64)
    scale = tf.maximum(0.5 * (left_sd + right_sd), tf.constant(1.0e-12, tf.float64))
    left_cov = tf.constant(left["covariance"], tf.float64)
    right_cov = tf.constant(right["covariance"], tf.float64)
    denominator = tf.maximum(0.5 * (tf.linalg.norm(left_cov) + tf.linalg.norm(right_cov)), tf.constant(1.0e-12, tf.float64))
    return {
        "mean": float(tf.reduce_max(tf.abs(tf.constant(left["mean"], tf.float64) - tf.constant(right["mean"], tf.float64)) / scale).numpy()),
        "sd": float(tf.reduce_max(tf.abs(left_sd - right_sd) / scale).numpy()),
        "covariance": float((tf.linalg.norm(left_cov - right_cov) / denominator).numpy()),
        "quantile": float(tf.reduce_max(tf.abs(tf.constant(left["quantiles"], tf.float64) - tf.constant(right["quantiles"], tf.float64)) / scale[tf.newaxis, :]).numpy()),
    }


def _stability_passed(metrics: Mapping[str, float]) -> bool:
    return all(float(metrics[name]) <= STABILITY[name] for name in STABILITY)


def _metadata_summary(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    startup = metadata.get("startup_worker_metadata", ())
    return {
        "backend": metadata.get("backend"),
        "evaluation_mode": metadata.get("evaluation_mode"),
        "configured_worker_count": metadata.get("configured_worker_count"),
        "batch_per_worker": metadata.get("batch_per_worker"),
        "task_count": metadata.get("task_count"),
        "worker_runtime_max_seconds": metadata.get("worker_runtime_max_seconds"),
        "worker_result_pid_count": len(set(metadata.get("worker_result_pids", ()))),
        "status_jit_compile_all": bool(startup) and all(row.get("status_jit_compile") is True for row in startup),
        "cuda_hidden_all": bool(startup) and all(row.get("cuda_visible_devices") == "-1" and row.get("tensorflow_gpu_devices") == [] for row in startup),
        "target_signature_all": bool(startup) and all(row.get("target_signature") == TARGET_SIGNATURE for row in startup),
        "adapter_signature_all": bool(startup) and all(row.get("adapter_signature") == ADAPTER_SIGNATURE for row in startup),
    }


def _reference(args: argparse.Namespace) -> int:
    import tensorflow as tf
    _check_cpu_tf(tf)
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import complexity_posterior_target
    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool, TFBatchValueScorePoolConfig
    target = complexity_posterior_target(20, jit_compile=True)
    if target.target_signature() != TARGET_SIGNATURE:
        raise CampaignError("target signature mismatch")
    started = time.monotonic()
    map_result = _map_search(tf, target, progress_path=args.output_root / "map-progress.json")
    if map_result.get("valid") is not True:
        write_json(args.output_root / "reference.json", {"schema": REFERENCE_SCHEMA, "status": "REFERENCE_INVALID", "error": "MAP validity veto", "map": map_result, "rows": [], "source_sha256": {"script": sha256(SCRIPT), "plan": sha256(PLAN)}}, replace=True)
        raise CampaignError("MAP validity veto: " + ", ".join(map_result.get("vetoes", ())))
    center = tf.constant(map_result["map"]["position"], tf.float64)
    hessian = _hessian(tf, target, center)
    config = TFBatchValueScorePoolConfig(
        factory_path="bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:batch_native_complexity_target_worker_factory",
        factory_config={"q": 20, "principal_sqrt_backend": "tensorflow_eigh", "jit_compile": True},
        dimension=4, worker_count=WORKERS, cores_per_worker=1,
        batch_sizes=(ROWS_PER_WORKER,), batch_per_worker=ROWS_PER_WORKER,
        worker_cpu_ids=tuple(range(WORKERS)), timeout_seconds=900.0,
    )
    rows = []
    selected = None
    try:
        with TFBatchValueScorePool(config) as pool:
            pool._ensure_started()
            for order in ORDERS:
                for scale in SCALES:
                    if time.monotonic() - started >= float(args.cap_seconds):
                        raise CampaignError("reference wall-time cap exhausted")
                    points, gh_weights, log_proposal = _mesh_quadrature(tf, center, tf.constant(hessian["factor"], tf.float64), order, scale)
                    point_count = int(points.shape[0])
                    padding = (-point_count) % ROWS_PER_WORKER
                    evaluated_points = tf.concat((points, tf.repeat(points[-1:, :], padding, axis=0)), axis=0) if padding else points
                    values, _scores, status, metadata = pool.evaluate_with_status(evaluated_points, request_id=f"q20-reference-{scale}-{order}")
                    values = values[:point_count]
                    status = {key: value[:point_count] for key, value in status.items()}
                    valid = tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool)
                    invalid_count = int(tf.reduce_sum(tf.cast(tf.logical_not(valid), tf.int32)).numpy())
                    finite = bool(tf.reduce_all(tf.math.is_finite(values)).numpy())
                    if invalid_count or not finite:
                        raise CampaignError(f"quadrature target invalid rows: scale={scale} order={order}")
                    log_weights = tf.math.log(gh_weights) + values - log_proposal
                    stats = _stats(tf, points, log_weights)
                    if float(stats["weight_ess"]) < 50.0 or float(stats["max_weight"]) > 0.05:
                        raise CampaignError(f"quadrature weights are concentrated: scale={scale} order={order}")
                    rows.append({"scale": scale, "order": order, "point_count": point_count, "evaluated_point_count": int(evaluated_points.shape[0]), "padding_count": padding, "target_invalid_count": invalid_count, "target_all_finite": finite, "stats": stats, "worker_metadata": _metadata_summary(metadata), "max_log_prob": float(tf.reduce_max(values).numpy())})
                    write_json(args.output_root / "progress.json", {"schema": SCHEMA, "status": "RUNNING", "completed": len(rows), "rows": rows}, replace=True)
                    if order >= 9:
                        current = rows[-1]
                        previous = next(row for row in rows if row["scale"] == scale and row["order"] == order - 2)
                        adjacent = _stability(tf, previous["stats"], current["stats"])
                        cross = None
                        if scale == SCALES[-1]:
                            other = next(row for row in rows if row["scale"] == SCALES[0] and row["order"] == order)
                            cross = _stability(tf, other["stats"], current["stats"])
                            if _stability_passed(adjacent) and _stability_passed(cross):
                                other_previous = next(row for row in rows if row["scale"] == SCALES[0] and row["order"] == order - 2)
                                other_adjacent = _stability(tf, other_previous["stats"], other["stats"])
                                if _stability_passed(other_adjacent):
                                    selected = {"order": order, "scale": scale, "row": current, "adjacent": adjacent, "other_scale_adjacent": other_adjacent, "cross_scale": cross, "numerical_error_bound": {name: max(adjacent[name], other_adjacent[name], cross[name]) for name in STABILITY}}
                if selected is not None:
                    break
    except BaseException as exc:
        write_json(args.output_root / "reference.json", {"schema": REFERENCE_SCHEMA, "status": "REFERENCE_INVALID", "error": f"{type(exc).__name__}: {exc}", "map": map_result, "hessian": hessian, "rows": rows}, replace=True)
        raise
    if selected is None:
        raise CampaignError("quadrature did not pass adjacent-order and cross-scale stability")
    reference = {"schema": REFERENCE_SCHEMA, "status": "REFERENCE_COMPLETED", "target_signature": TARGET_SIGNATURE, "adapter_signature": ADAPTER_SIGNATURE, "jit_compile": True, "dtype": "float64", "cpu_only": True, "map": map_result, "hessian": hessian, "orders": list(ORDERS), "scales": list(SCALES), "selected": selected, "rows": rows, "source_sha256": {"script": sha256(SCRIPT), "plan": sha256(PLAN)}, "wall_seconds": time.monotonic() - started, "run_manifest": {"command": " ".join(sys.argv), "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"), "git_commit": os.popen("git rev-parse HEAD").read().strip(), "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import", "tensorflow": tf.__version__, "worker_count": WORKERS, "rows_per_worker": ROWS_PER_WORKER, "cap_seconds": float(args.cap_seconds)}, "nonclaims": ["not an analytic posterior", "no model adequacy claim", "no sampler superiority claim", "no default readiness claim"]}
    reference["reference_hash"] = stable_hash(reference)
    write_json(args.output_root / "reference.json", reference, replace=True)
    return 0


def _compare(args: argparse.Namespace) -> int:
    import tensorflow as tf
    _check_cpu_tf(tf)
    _verify_archive(parse_tensor_values=True)
    reference = _load_json(args.output_root / "reference.json")
    if reference.get("status") != "REFERENCE_COMPLETED" or reference.get("target_signature") != TARGET_SIGNATURE:
        raise CampaignError("reference is absent, invalid, or target-mismatched")
    reference_payload = dict(reference)
    supplied_reference_hash = str(reference_payload.pop("reference_hash", ""))
    if supplied_reference_hash != stable_hash(reference_payload):
        raise CampaignError("reference artifact hash mismatch")
    source_hashes = reference.get("source_sha256", {})
    if source_hashes.get("script") != sha256(SCRIPT) or source_hashes.get("plan") != sha256(PLAN):
        raise CampaignError("reference source or plan hash is stale")
    retained = []
    manifest = _load_json(ARCHIVE_MANIFEST)
    for chunk in manifest["retained_chunks"]:
        receipt = chunk["sample_receipt"]
        path = Path(str(receipt["path"]))
        tensor = tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64)
        retained.append(tensor)
    draws = tf.concat(retained, axis=0)  # [1000, chains, dim]
    draws = tf.transpose(draws, [1, 0, 2])
    flat = tf.reshape(draws, [-1, 4])
    candidate_stats = _unweighted_stats(tf, flat)
    selected = reference.get("selected")
    if not isinstance(selected, Mapping) or not isinstance(selected.get("row"), Mapping):
        raise CampaignError("reference lacks a frozen selected quadrature row")
    ref_stats = selected["row"]["stats"]
    bootstrap = _bootstrap(tf, draws, ref_stats, selected["numerical_error_bound"])
    result = {"schema": SCHEMA, "status": "COMPARISON_COMPLETED", "reference_hash": reference["reference_hash"], "archive_manifest_sha256": sha256(ARCHIVE_MANIFEST), "candidate_stats": candidate_stats, "reference_stats": ref_stats, "bootstrap": bootstrap, "decision": _decision(bootstrap), "run_manifest": {"command": " ".join(sys.argv), "git_commit": os.popen("git rev-parse HEAD").read().strip(), "tensorflow": tf.__version__, "cpu_gpu_status": "CPU-only comparison; CUDA hidden", "bootstrap_seed": list(BOOTSTRAP_SEED), "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_block_length": BOOTSTRAP_BLOCK}, "nonclaims": ["agreement is scoped to this q20 target and archived candidate", "no model adequacy, superiority, robustness, or default claim"]}
    result["comparison_hash"] = stable_hash(result)
    write_json(args.output_root / "comparison.json", result, replace=True)
    return 0


def _unweighted_stats(tf: Any, values: Any) -> Mapping[str, Any]:
    import tensorflow_probability as tfp
    mean = tf.reduce_mean(values, axis=0)
    centered = values - mean[tf.newaxis, :]
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.cast(tf.shape(values)[0], tf.float64)
    sd = tf.sqrt(tf.linalg.diag_part(covariance))
    quantiles = tfp.stats.percentile(values, [5.0, 50.0, 95.0], axis=0, interpolation="linear")
    return {"mean": safe(mean), "sd": safe(sd), "covariance": safe(covariance), "quantiles": safe(quantiles)}


def _bootstrap(tf: Any, draws: Any, reference: Mapping[str, Any], numerical_error: Mapping[str, Any]) -> Mapping[str, Any]:
    import tensorflow_probability as tfp
    chain_count = tf.shape(draws)[0]
    sample_count = tf.shape(draws)[1]
    block_count = tf.cast(tf.math.ceil(tf.cast(sample_count, tf.float64) / BOOTSTRAP_BLOCK), tf.int32)
    starts = tf.random.stateless_uniform([BOOTSTRAP_REPLICATES, chain_count, block_count], seed=BOOTSTRAP_SEED, minval=0, maxval=sample_count, dtype=tf.int32)
    offsets = tf.range(BOOTSTRAP_BLOCK, dtype=tf.int32)[tf.newaxis, tf.newaxis, tf.newaxis, :]
    indices = tf.math.mod(starts[:, :, :, tf.newaxis] + offsets, sample_count)
    indices = tf.reshape(indices[:, :, :, :], [BOOTSTRAP_REPLICATES, chain_count, -1])[:, :, :sample_count]
    chain_ids = tf.broadcast_to(tf.range(chain_count)[tf.newaxis, :, tf.newaxis], tf.shape(indices))
    sampled = tf.gather_nd(draws, tf.stack([chain_ids, indices], axis=-1))
    sampled = tf.reshape(sampled, [BOOTSTRAP_REPLICATES, -1, 4])
    means = tf.reduce_mean(sampled, axis=1)
    centered = sampled - means[:, tf.newaxis, :]
    covariances = tf.einsum("bni,bnj->bij", centered, centered) / tf.cast(tf.shape(sampled)[1], tf.float64)
    sds = tf.sqrt(tf.linalg.diag_part(covariances))
    quantiles = tfp.stats.percentile(sampled, [5.0, 50.0, 95.0], axis=1, interpolation="linear")
    quantiles = tf.transpose(quantiles, [1, 0, 2])
    ref_mean = tf.constant(reference["mean"], tf.float64)
    ref_sd = tf.constant(reference["sd"], tf.float64)
    ref_cov = tf.constant(reference["covariance"], tf.float64)
    ref_q = tf.constant(reference["quantiles"], tf.float64)
    mean_error = tf.reduce_max(tf.abs(means - ref_mean) / ref_sd, axis=1)
    sd_error = tf.reduce_max(tf.abs(sds - ref_sd) / ref_sd, axis=1)
    cov_error = tf.linalg.norm(covariances - ref_cov, axis=[1, 2]) / tf.linalg.norm(ref_cov)
    quantile_error = tf.reduce_max(tf.abs(quantiles - ref_q[tf.newaxis, :, :]) / ref_sd[tf.newaxis, tf.newaxis, :], axis=[1, 2])
    maxima = tf.stack([mean_error, sd_error, cov_error, quantile_error], axis=1)
    stochastic_upper = tfp.stats.percentile(maxima, 99.0, axis=0, interpolation="linear")
    quadrature_error = tf.constant([numerical_error[name] for name in ("mean", "sd", "covariance", "quantile")], tf.float64)
    upper = stochastic_upper + quadrature_error
    return {"upper_99": safe(upper), "stochastic_upper_99": safe(stochastic_upper), "quadrature_numerical_error_bound": safe(quadrature_error), "max_discrepancy_upper_99": safe(tf.reduce_max(upper)), "margin": MARGINS, "replicates": BOOTSTRAP_REPLICATES, "block_length": BOOTSTRAP_BLOCK, "maxima_mean": safe(tf.reduce_mean(maxima, axis=0)), "maxima_q95": safe(tfp.stats.percentile(maxima, 95.0, axis=0, interpolation="linear"))}


def _decision(bootstrap: Mapping[str, Any]) -> Mapping[str, Any]:
    upper = bootstrap["upper_99"]
    names = ("mean", "sd", "covariance", "quantile")
    passed = all(float(value) <= MARGINS[name] for name, value in zip(names, upper))
    return {"posterior_agreement": passed, "status": "AGREEMENT_PASS" if passed else "CANDIDATE_DISAGREEMENT_OR_INCONCLUSIVE", "criterion": "simultaneous_99_percent_bootstrap_upper_bounds_within_frozen_margins", "hard_vetoes": [] if passed else ["posterior_equivalence_margin_failed"], "ranking_supported": False}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "reference", "compare"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=MAX_REFERENCE_SECONDS)
    args = parser.parse_args()
    if not 0.0 < args.cap_seconds <= MAX_REFERENCE_SECONDS:
        parser.error("--cap-seconds must be in (0,18000]")
    args.output_root = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root.resolve()
    if not args.output_root.is_relative_to(ROOT):
        parser.error("output-root must be inside repository")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        import tensorflow as tf
        _check_cpu_tf(tf)
        _verify_archive()
        from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import complexity_posterior_target
        target = complexity_posterior_target(20, jit_compile=True)
        value, score = target.log_prob_and_grad(tf.constant(PRIOR_CENTER_VALUES, tf.float64))
        if not bool(tf.reduce_all(tf.math.is_finite([value, tf.reduce_max(tf.abs(score))])).numpy()):
            raise CampaignError("target preflight is nonfinite")
        write_json(args.output_root / "preflight.json", {"schema": SCHEMA, "status": "PREFLIGHT_PASSED", "target_signature": target.target_signature(), "adapter_signature": target.adapter_signature(), "value_at_prior_center": float(value.numpy()), "score_at_prior_center": safe(score), "jit_compile": True, "cpu_only": True, "source_sha256": {"script": sha256(SCRIPT), "plan": sha256(PLAN)}}, replace=True)
        return 0
    if args.mode == "reference":
        return _reference(args)
    return _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())

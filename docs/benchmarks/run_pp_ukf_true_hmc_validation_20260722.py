#!/usr/bin/env python3
"""Run the claim-bearing PP-UKF frozen-kernel sequential HMC campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path(
    "docs/plans/bayesfilter-pp-ukf-true-hmc-continuation-repair-plan-2026-07-30.md"
)
PREVIOUS_PREFLIGHT = Path(
    "docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-02/preflight.json"
)
TRANSPORT_PATH = Path(
    "docs/plans/artifacts/bayesfilter-public-tuner-fixed-identity-mass-phase5-20260720/"
    "campaign-01/PP-UKF/final/segments/steps-004001-005000/frozen_transport.json"
)
TRANSPORT_SHA256 = "b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221"
TARGET_SIGNATURE = "d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5"
CAMPAIGN_CAP_SECONDS = 86_400.0
CHAIN_COUNT = 4
WARMUP_MIN = 2_000
RETAINED_MIN = 1_000
MAX_RESULTS = 10_000
BULK_ESS_MIN = 1_000.0
TAIL_ESS_MIN = 400.0
WARMUP_RHAT_MAX = 1.05
RETAINED_RHAT_MAX = 1.01
INITIAL_STATE_SEED = (20260722, 8300)
WARMUP_SEED_ROOT = (20260722, 9200)
RETAINED_SEED_ROOT = (20260722, 10200)
CONTINUATION_REPLACEMENT_INDICES = (1, 2, 5)
CONTINUATION_PREFIX_RESULTS = 3_000
ATTEMPT_09_PROGRESS_SHA256 = (
    "acd34ab3d4bd1ecf0907c193cb87a4aeed1fa95c6a2d637ece8b6bb8fdd4eec8"
)
CONTINUATION_CHUNK_BUDGET_RESERVE_SECONDS = 900.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"artifact must be fresh: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    temporary.replace(path)


def _write_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="ascii")
    tmp.replace(path)


def _progress_payload(
    *,
    prior_rows: list[Mapping[str, Any]],
    rows: list[Mapping[str, Any]],
    planned_candidate_count: int,
    elapsed_seconds: float,
    terminal: bool,
    prior_elapsed_seconds: float = 0.0,
    prior_progress_sha256: str | None = None,
) -> Mapping[str, Any]:
    all_rows = tuple(prior_rows) + tuple(rows)
    elapsed = float(elapsed_seconds)
    prior = float(prior_elapsed_seconds)
    return {
        "schema": "bayesfilter.pp_ukf.true_hmc_validation.progress.v1",
        "completed_candidate_count": sum(
            int(row.get("terminal_candidate_result", True)) for row in all_rows
        ),
        "planned_candidate_count": int(planned_candidate_count),
        "candidate_rows": all_rows,
        "elapsed_seconds": elapsed,
        "prior_elapsed_seconds": prior,
        "aggregate_elapsed_seconds": prior + elapsed,
        "prior_progress_sha256": prior_progress_sha256,
        "terminal": bool(terminal),
    }


def _campaign_budget_exhausted(
    *,
    prior_elapsed_seconds: float,
    current_elapsed_seconds: float,
    reserve_seconds: float = 0.0,
) -> bool:
    """Return whether another chunk would violate the aggregate campaign cap."""
    return (
        float(prior_elapsed_seconds)
        + float(current_elapsed_seconds)
        + float(reserve_seconds)
        >= CAMPAIGN_CAP_SECONDS
    )


def _validate_run_request(
    *,
    candidate_indices: tuple[int, ...] | None,
    replacement_indices: tuple[int, ...] | None,
    prior_elapsed_seconds: float,
    resume_progress: Path | None,
) -> None:
    prior = float(prior_elapsed_seconds)
    if not math.isfinite(prior) or prior < 0.0 or prior >= CAMPAIGN_CAP_SECONDS:
        raise ValueError("prior_elapsed_seconds must be finite and within campaign cap")
    if candidate_indices is not None and replacement_indices is not None:
        raise ValueError("candidate and replacement selection modes are exclusive")
    if replacement_indices is not None and resume_progress is None:
        raise ValueError("replacement continuation requires --resume-progress")
    if resume_progress is not None and not resume_progress.is_file():
        raise FileNotFoundError(f"resume progress does not exist: {resume_progress}")


def _archive_path(value: Any) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"archive path escapes repository: {path}") from exc
    return resolved


def _verify_continuation_prefix(
    row: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    candidate_index: int,
) -> Mapping[str, Any]:
    if row.get("candidate", {}).get("candidate_id") != candidate.get("candidate_id"):
        raise ValueError("continuation candidate identity mismatch")
    if int(row["candidate"]["num_leapfrog_steps"]) != int(
        candidate["num_leapfrog_steps"]
    ):
        raise ValueError("continuation leapfrog count mismatch")
    if not math.isclose(
        float(row["candidate"]["step_size"]),
        float(candidate["step_size"]),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ValueError("continuation step size mismatch")
    if row.get("warmup_passed") is not True or row.get("retained_passed") is not False:
        raise ValueError("continuation requires passed warmup and censored retained row")
    if row.get("passed") is not False or tuple(row.get("hard_vetoes", ())) != ():
        raise ValueError("continuation row must be a no-veto retained-cap failure")
    if int(row.get("retained_results_per_chain", -1)) != CONTINUATION_PREFIX_RESULTS:
        raise ValueError("continuation prefix must contain exactly 3000 retained draws")
    if int(row.get("config", {}).get("retained_max_results", -1)) != CONTINUATION_PREFIX_RESULTS:
        raise ValueError("continuation source row was not censored at the 3000 cap")

    retained_rows = [
        item for item in row.get("archive", ()) if item.get("stage") == "retained"
    ]
    chunk_rows = sorted(
        (item for item in retained_rows if item.get("cumulative") is False),
        key=lambda item: int(item["chunk_index"]),
    )
    expected_chunks = CONTINUATION_PREFIX_RESULTS // 500
    if tuple(int(item["chunk_index"]) for item in chunk_rows) != tuple(
        range(expected_chunks)
    ):
        raise ValueError("continuation retained chunk sequence is incomplete")
    expected_warmup_seed = (
        WARMUP_SEED_ROOT[0],
        WARMUP_SEED_ROOT[1] + int(candidate_index) * 1009,
    )
    expected_retained_seed = (
        RETAINED_SEED_ROOT[0],
        RETAINED_SEED_ROOT[1] + int(candidate_index) * 1009,
    )
    if tuple(int(item) for item in row["config"]["warmup_seed"]) != expected_warmup_seed:
        raise ValueError("continuation warmup seed root mismatch")
    retained_seed = tuple(int(item) for item in row["config"]["retained_seed"])
    if retained_seed != expected_retained_seed:
        raise ValueError("continuation retained seed root mismatch")
    from bayesfilter.inference.neutra_hmc import sequential_chunk_seed

    for item in chunk_rows:
        index = int(item["chunk_index"])
        if tuple(int(value) for value in item["seed"]) != sequential_chunk_seed(
            retained_seed, index
        ):
            raise ValueError("continuation retained chunk seed mismatch")
        if tuple(int(value) for value in item["shape"]) != (500, CHAIN_COUNT, 6):
            raise ValueError("continuation retained chunk shape mismatch")
        for role in ("latent", "raw"):
            path = _archive_path(item[f"{role}_path"])
            if not path.is_file() or _sha256(path) != item[f"{role}_sha256"]:
                raise ValueError(f"continuation {role} chunk hash mismatch: {path}")

    cumulative_rows = [
        item for item in retained_rows if item.get("cumulative") is True
    ]
    if len(cumulative_rows) != 1:
        raise ValueError("continuation requires one cumulative retained archive")
    cumulative = cumulative_rows[0]
    if tuple(int(value) for value in cumulative["shape"]) != (
        CONTINUATION_PREFIX_RESULTS,
        CHAIN_COUNT,
        6,
    ):
        raise ValueError("continuation cumulative retained shape mismatch")
    paths = {}
    for role in ("latent", "raw"):
        path = _archive_path(cumulative[f"{role}_path"])
        if not path.is_file() or _sha256(path) != cumulative[f"{role}_sha256"]:
            raise ValueError(f"continuation cumulative {role} hash mismatch: {path}")
        paths[role] = path
    return {
        "candidate_id": candidate["candidate_id"],
        "prefix_results_per_chain": CONTINUATION_PREFIX_RESULTS,
        "next_retained_chunk_index": expected_chunks,
        "latent_path": paths["latent"],
        "raw_path": paths["raw"],
        "latent_sha256": cumulative["latent_sha256"],
        "raw_sha256": cumulative["raw_sha256"],
        "source_row_sha256": _stable_hash(row),
        "source_row": row,
        "chunk_latent_paths": tuple(
            _archive_path(item["latent_path"]) for item in chunk_rows
        ),
        "chunk_raw_paths": tuple(
            _archive_path(item["raw_path"]) for item in chunk_rows
        ),
    }


def _validate_replacement_contract(
    resume_progress: Path,
    replacement_indices: tuple[int, ...],
) -> tuple[Mapping[str, Any], Mapping[int, Mapping[str, Any]]]:
    indices = tuple(sorted(set(int(item) for item in replacement_indices)))
    if indices != CONTINUATION_REPLACEMENT_INDICES:
        raise ValueError(
            "continuation replacement indices must be exactly (1, 2, 5)"
        )
    if _sha256(resume_progress) != ATTEMPT_09_PROGRESS_SHA256:
        raise ValueError("attempt-09 continuation progress hash mismatch")
    payload = json.loads(resume_progress.read_text(encoding="utf-8"))
    rows = list(payload.get("candidate_rows", ()))
    candidates = list(_candidate_manifest()["candidates"])
    if len(rows) != len(candidates) or len(rows) != 10:
        raise ValueError("continuation progress must contain all ten candidate rows")
    if tuple(row["candidate"]["candidate_id"] for row in rows) != tuple(
        candidate["candidate_id"] for candidate in candidates
    ):
        raise ValueError("continuation progress candidate order or identity drifted")
    prefixes = {
        index: _verify_continuation_prefix(
            rows[index], candidate=candidates[index], candidate_index=index
        )
        for index in indices
    }
    return payload, prefixes


def _merge_replacement_rows(
    prior_rows: list[Mapping[str, Any]],
    replacement_rows: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    replacements = {
        str(row["candidate"]["candidate_id"]): row for row in replacement_rows
    }
    if len(replacements) != len(replacement_rows):
        raise ValueError("replacement candidate IDs must be unique")
    prior_ids = [str(row["candidate"]["candidate_id"]) for row in prior_rows]
    if not set(replacements).issubset(prior_ids):
        raise ValueError("replacement candidate is absent from prior rows")
    merged = [replacements.get(candidate_id, row) for candidate_id, row in zip(prior_ids, prior_rows)]
    if len(merged) != len(prior_rows) or len({row["candidate"]["candidate_id"] for row in merged}) != len(merged):
        raise ValueError("replacement merge changed row count or uniqueness")
    return merged


def _load_operational_module() -> Any:
    path = ROOT / "docs/benchmarks/run_pp_ukf_operational_broad_grid_20260721.py"
    spec = importlib.util.spec_from_file_location("pp_ukf_operational_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load PP-UKF operational base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate_manifest() -> Mapping[str, Any]:
    payload = json.loads((ROOT / PREVIOUS_PREFLIGHT).read_text(encoding="utf-8"))
    if payload.get("status") != "blocked_before_sampling_missing_fresh_partition_and_budget":
        raise ValueError("unexpected preflight status")
    manifest = payload["candidate_manifest"]
    if tuple(int(c["num_leapfrog_steps"]) for c in manifest["candidates"]) != (5, 9, 12, 13, 14, 17, 18, 19, 24, 25):
        raise ValueError("candidate set drifted")
    if manifest["target_signature"] != TARGET_SIGNATURE or manifest["transport_sha256"] != TRANSPORT_SHA256:
        raise ValueError("preflight identity drifted")
    return manifest


def _fresh_partition(tf: Any, parameter_dim: int) -> Mapping[str, Any]:
    initial = tf.random.stateless_normal(
        (CHAIN_COUNT, parameter_dim), seed=tf.constant(INITIAL_STATE_SEED, tf.int32), dtype=tf.float64
    ) * tf.constant(0.25, tf.float64)
    serialized = bytes(tf.io.serialize_tensor(initial).numpy())
    return {
        "schema": "bayesfilter.pp_ukf.true_hmc_validation.execution_partition.v1",
        "kind": "fresh_execution_partition_on_immutable_observation_sequence",
        "observation_data_signature": "sha256:dc63294b6e77913aef0c92796dd2d3c7a1721a766f976fcc392cd02a70754387",
        "initial_state_seed": INITIAL_STATE_SEED,
        "warmup_seed_root": WARMUP_SEED_ROOT,
        "retained_seed_root": RETAINED_SEED_ROOT,
        "initial_state_sha256": hashlib.sha256(serialized).hexdigest(),
        "tuning_seed_disjoint": True,
        "tuning_draws_reused": False,
        "partition_signature": _stable_hash(
            {
                "observation_data_signature": "sha256:dc63294b6e77913aef0c92796dd2d3c7a1721a766f976fcc392cd02a70754387",
                "initial_state_seed": INITIAL_STATE_SEED,
                "warmup_seed_root": WARMUP_SEED_ROOT,
                "retained_seed_root": RETAINED_SEED_ROOT,
                "initial_state_sha256": hashlib.sha256(serialized).hexdigest(),
            }
        ),
    }


def _transform_batch(transport: Any, values: Any) -> Any:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(values, tf.float64)
    shape = tf.shape(tensor)
    flat = tf.reshape(tensor, (-1, tensor.shape[-1]))
    transformed = transport.forward_batch(flat)
    return tf.reshape(
        transformed,
        tf.concat(
            [shape[:-1], tf.constant([int(tensor.shape[-1])], tf.int32)], axis=0
        ),
    )


def _archive_callback(root: Path, candidate_id: str):
    import tensorflow as tf

    candidate_root = root / "private" / candidate_id
    metadata: list[Mapping[str, Any]] = []

    def archive(**kwargs: Any) -> Mapping[str, Any]:
        stage = str(kwargs["stage"])
        chunk_index = kwargs.get("chunk_index")
        suffix = "cumulative" if kwargs.get("cumulative") else f"chunk-{int(chunk_index):04d}"
        folder = candidate_root / stage
        folder.mkdir(parents=True, exist_ok=True)
        latent_path = folder / f"{suffix}-latent.tftensor"
        raw_path = folder / f"{suffix}-raw.tftensor"
        latent = tf.convert_to_tensor(kwargs["latent_samples"], tf.float64)
        raw = tf.convert_to_tensor(kwargs["model_samples"], tf.float64)
        for path, tensor in ((latent_path, latent), (raw_path, raw)):
            serialized = bytes(tf.io.serialize_tensor(tensor).numpy())
            temporary = path.with_name("." + path.name + ".tmp")
            temporary.write_bytes(serialized)
            temporary.replace(path)
        row = {
            "stage": stage,
            "chunk_index": chunk_index,
            "cumulative": bool(kwargs.get("cumulative")),
            "seed": kwargs.get("seed"),
            "latent_path": str(latent_path),
            "raw_path": str(raw_path),
            "latent_sha256": _sha256(latent_path),
            "raw_sha256": _sha256(raw_path),
            "shape": tuple(int(x) for x in latent.shape),
        }
        metadata.append(row)
        return row

    archive.metadata = metadata
    return archive


def _parse_tensor(path: Path, tf: Any) -> Any:
    return tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64)


def _load_verified_continuation_tensors(
    prefix: Mapping[str, Any], tf: Any
) -> tuple[Any, Any]:
    prefix_latent = _parse_tensor(prefix["latent_path"], tf)
    prefix_raw = _parse_tensor(prefix["raw_path"], tf)
    expected_shape = (CONTINUATION_PREFIX_RESULTS, CHAIN_COUNT, 6)
    if tuple(int(value) for value in prefix_latent.shape) != expected_shape or tuple(
        int(value) for value in prefix_raw.shape
    ) != expected_shape:
        raise ValueError("continuation prefix tensor shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(prefix_latent)).numpy()) or not bool(
        tf.reduce_all(tf.math.is_finite(prefix_raw)).numpy()
    ):
        raise ValueError("continuation prefix tensors must be finite")
    chunk_latent = tf.concat(
        tuple(_parse_tensor(path, tf) for path in prefix["chunk_latent_paths"]),
        axis=0,
    )
    chunk_raw = tf.concat(
        tuple(_parse_tensor(path, tf) for path in prefix["chunk_raw_paths"]),
        axis=0,
    )
    if tuple(int(value) for value in chunk_latent.shape) != expected_shape or tuple(
        int(value) for value in chunk_raw.shape
    ) != expected_shape:
        raise ValueError("continuation chunk concatenation shape mismatch")
    if not bool(tf.reduce_all(tf.equal(prefix_latent, chunk_latent)).numpy()) or not bool(
        tf.reduce_all(tf.equal(prefix_raw, chunk_raw)).numpy()
    ):
        raise ValueError("continuation cumulative tensor differs from chunk concatenation")
    return prefix_latent, prefix_raw


def _continuation_prefix_payload(prefix: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        key: value
        for key, value in prefix.items()
        if key not in {"source_row", "chunk_latent_paths", "chunk_raw_paths"}
    }


def _continuation_row(
    *,
    candidate: Mapping[str, Any],
    config: Any,
    prefix: Mapping[str, Any],
    continuation: Mapping[str, Any],
    new_archive: tuple[Mapping[str, Any], ...],
) -> Mapping[str, Any]:
    source = prefix["source_row"]
    source_archive = tuple(
        {**item, "continuation_archive_origin": "attempt09_prefix"}
        for item in source.get("archive", ())
    )
    continuation_archive = tuple(
        {**item, "continuation_archive_origin": "current_attempt"}
        for item in new_archive
    )
    completion_status = str(
        continuation.get("completion_status", "continuation_in_progress")
    )
    terminal = completion_status in {"passed", "hard_veto", "retained_cap_reached"}
    return {
        "candidate": candidate,
        "config": config.payload(chain_count=CHAIN_COUNT),
        "passed": bool(continuation.get("passed", False)),
        "decision": continuation.get(
            "decision", "INCOMPLETE_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
        ),
        "completion_status": completion_status,
        "terminal_candidate_result": terminal,
        "warmup_passed": True,
        "retained_passed": bool(continuation.get("retained_passed", False)),
        "retained_cap_hit": bool(continuation.get("retained_cap_hit", False)),
        "warmup_results_per_chain": int(source["warmup_results_per_chain"]),
        "retained_results_per_chain": int(
            continuation["retained_results_per_chain"]
        ),
        "warmup_checks": tuple(source.get("warmup_checks", ())),
        "retained_checks": tuple(source.get("retained_checks", ()))
        + tuple(continuation.get("retained_checks", ())),
        "hard_vetoes": tuple(continuation.get("hard_vetoes", ())),
        "elapsed_seconds": float(continuation.get("elapsed_seconds", 0.0)),
        "archive": source_archive + continuation_archive,
        "continuation_prefix": _continuation_prefix_payload(prefix),
        "continuation_replacement": True,
        "continuation_prefix_results_per_chain": CONTINUATION_PREFIX_RESULTS,
        "continuation_source_config": source.get("config"),
        "continuation_new_check_count": len(
            tuple(continuation.get("retained_checks", ()))
        ),
    }


def _run_retained_continuation(
    *,
    adapter: Any,
    transport: Any,
    prefix: Mapping[str, Any],
    base_adapter: Any,
    thresholds: Any,
    archive: Any,
    tf: Any,
    rank_normalized_hmc_diagnostics: Any,
    config: Any,
    stop_requested_fn: Any,
    checkpoint_callback: Any,
) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc import (
        run_retained_neutra_hmc_continuation,
    )

    prefix_latent, prefix_raw = _load_verified_continuation_tensors(prefix, tf)
    return run_retained_neutra_hmc_continuation(
        adapter=adapter,
        prefix_latent=prefix_latent,
        prefix_model=prefix_raw,
        model_transform=lambda values: _transform_batch(transport, values),
        parameter_names=base_adapter.parameter_names,
        config=config,
        next_chunk_index=int(prefix["next_retained_chunk_index"]),
        retained_diagnostic_fn=lambda draws: rank_normalized_hmc_diagnostics(
            draws,
            parameter_names=base_adapter.parameter_names,
            thresholds=thresholds,
        ),
        archive_callback=archive,
        checkpoint_callback=checkpoint_callback,
        stop_requested_fn=stop_requested_fn,
    )


def run_campaign(
    output_root: Path,
    *,
    candidate_indices: tuple[int, ...] | None = None,
    replacement_indices: tuple[int, ...] | None = None,
    prior_elapsed_seconds: float = 0.0,
    resume_progress: Path | None = None,
) -> Mapping[str, Any]:
    _validate_run_request(
        candidate_indices=candidate_indices,
        replacement_indices=replacement_indices,
        prior_elapsed_seconds=prior_elapsed_seconds,
        resume_progress=resume_progress,
    )
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc)

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    # This must run before importing any project module that may create a
    # TensorFlow logical device or tensor.
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

    from bayesfilter.inference.hmc_convergence import RankNormalizedHMCThresholds, rank_normalized_hmc_diagnostics
    from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig, run_sequential_neutra_hmc
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        PredatorPreyUKFLikelihoodRecomposer,
        generate_frozen_predator_prey_dataset_tf,
        make_predator_prey_ukf_neutra_adapter,
        source_six_probit_jacobian_value_score,
        source_uniform_prior_value_score,
    )

    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    transport_path = ROOT / TRANSPORT_PATH
    if _sha256(transport_path) != TRANSPORT_SHA256:
        raise ValueError("frozen transport hash mismatch")
    loaded = load_frozen_neutra_artifact(
        json.loads(transport_path.read_text(encoding="utf-8")),
        expected_target_signature=TARGET_SIGNATURE,
    )
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    base_adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)
    if base_adapter.contract is None:
        raise ValueError("PP-UKF contract unavailable")
    bound = _load_operational_module().PPUKFBatchNativeBoundAdapter(
        base_adapter, target_signature=TARGET_SIGNATURE
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bound,
        transport=loaded.transport,
        target_scope="PP-UKF:true_hmc_claim_validation_20260722",
        evidence_path=str(PLAN),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    partition = _fresh_partition(tf, adapter.parameter_dim)
    initial_state = tf.random.stateless_normal(
        (CHAIN_COUNT, adapter.parameter_dim), seed=tf.constant(INITIAL_STATE_SEED, tf.int32), dtype=tf.float64
    ) * tf.constant(0.25, tf.float64)

    # Independent component recomposition on fresh validation points.
    points = tf.random.stateless_normal((4, adapter.parameter_dim), seed=tf.constant((20260722, 8400), tf.int32), dtype=tf.float64) * 0.2
    production_value, production_score = bound.log_prob_and_grad(points)
    recomposer = PredatorPreyUKFLikelihoodRecomposer(base_adapter)
    likelihood_value, likelihood_score = recomposer(points)
    prior_value, prior_score = source_uniform_prior_value_score(points)
    jac_value, jac_score = source_six_probit_jacobian_value_score(points)
    recomposition = {
        "point_signature": hashlib.sha256(bytes(tf.io.serialize_tensor(points).numpy())).hexdigest(),
        "maximum_value_error": float(tf.reduce_max(tf.abs(production_value - (likelihood_value + prior_value + jac_value))).numpy()),
        "maximum_score_error": float(tf.reduce_max(tf.abs(production_score - (likelihood_score + prior_score + jac_score))).numpy()),
    }
    recomposition["passed"] = bool(recomposition["maximum_value_error"] <= 1.0e-8 and recomposition["maximum_score_error"] <= 1.0e-7)
    if not recomposition["passed"]:
        raise ValueError("independent PP-UKF recomposition failed")

    all_candidates = _candidate_manifest()["candidates"]
    selected_indices = tuple(range(len(all_candidates))) if candidate_indices is None else tuple(sorted(set(candidate_indices)))
    prior_rows: list[Mapping[str, Any]] = []
    prior_progress_sha256 = None
    continuation_prefixes: Mapping[int, Mapping[str, Any]] = {}
    replacement_mode = replacement_indices is not None
    if resume_progress is not None:
        prior_progress_sha256 = _sha256(resume_progress)
        prior_rows = list(json.loads(resume_progress.read_text(encoding="utf-8")).get("candidate_rows", ()))
        if replacement_mode:
            _, continuation_prefixes = _validate_replacement_contract(
                resume_progress, tuple(replacement_indices or ())
            )
            selected_indices = tuple(sorted(set(int(item) for item in replacement_indices or ())))
            prior_rows = [
                {
                    **row,
                    "completion_status": "awaiting_retained_continuation",
                    "terminal_candidate_result": False,
                }
                if row_index in selected_indices
                else row
                for row_index, row in enumerate(prior_rows)
            ]
        else:
            completed_ids = {str(row["candidate"]["candidate_id"]) for row in prior_rows}
            selected_indices = tuple(index for index in selected_indices if str(all_candidates[index]["candidate_id"]) not in completed_ids)
    if not selected_indices or any(index < 0 or index >= len(all_candidates) for index in selected_indices):
        raise ValueError("candidate_indices must select valid manifest rows")
    candidates = tuple(all_candidates[index] for index in selected_indices)
    thresholds = RankNormalizedHMCThresholds(rhat_max=RETAINED_RHAT_MAX, bulk_ess_min=BULK_ESS_MIN, tail_ess_min=TAIL_ESS_MIN)
    rows: list[Mapping[str, Any]] = []
    hard_vetoes: list[str] = []
    campaign_incomplete = False
    for index, candidate in zip(selected_indices, candidates):
        candidate_id = str(candidate["candidate_id"])
        archive = _archive_callback(output_root, candidate_id)
        config = SequentialNeuTraHMCConfig(
            step_size=float(candidate["step_size"]),
            num_leapfrog_steps=int(candidate["num_leapfrog_steps"]),
            warmup_seed=(WARMUP_SEED_ROOT[0], WARMUP_SEED_ROOT[1] + index * 1009),
            retained_seed=(RETAINED_SEED_ROOT[0], RETAINED_SEED_ROOT[1] + index * 1009),
            warmup_chunk_results=250,
            warmup_min_results=WARMUP_MIN,
            warmup_check_window_results=1000,
            warmup_max_results=MAX_RESULTS,
            warmup_rhat_max=WARMUP_RHAT_MAX,
            retained_chunk_results=500,
            retained_min_results=RETAINED_MIN,
            retained_max_results=MAX_RESULTS,
            retained_rhat_max=RETAINED_RHAT_MAX,
            minimum_chain_count=CHAIN_COUNT,
            jit_compile=True,
        )
        if replacement_mode:
            prefix = continuation_prefixes[index]

            def stop_requested() -> bool:
                return _campaign_budget_exhausted(
                    prior_elapsed_seconds=prior_elapsed_seconds,
                    current_elapsed_seconds=time.monotonic() - started,
                    reserve_seconds=CONTINUATION_CHUNK_BUDGET_RESERVE_SECONDS,
                )

            def checkpoint(continuation: Mapping[str, Any]) -> None:
                current = _continuation_row(
                    candidate=candidate,
                    config=config,
                    prefix=prefix,
                    continuation=continuation,
                    new_archive=tuple(archive.metadata),
                )
                checkpoint_rows = _merge_replacement_rows(
                    prior_rows, list(tuple(rows) + (current,))
                )
                _write_progress(
                    output_root / "progress.json",
                    _progress_payload(
                        prior_rows=checkpoint_rows,
                        rows=[],
                        planned_candidate_count=len(all_candidates),
                        elapsed_seconds=time.monotonic() - started,
                        terminal=False,
                        prior_elapsed_seconds=prior_elapsed_seconds,
                        prior_progress_sha256=prior_progress_sha256,
                    ),
                )

            run = _run_retained_continuation(
                adapter=adapter,
                transport=loaded.transport,
                prefix=prefix,
                base_adapter=base_adapter,
                thresholds=thresholds,
                archive=archive,
                tf=tf,
                rank_normalized_hmc_diagnostics=rank_normalized_hmc_diagnostics,
                config=config,
                stop_requested_fn=stop_requested,
                checkpoint_callback=checkpoint,
            )
            row = _continuation_row(
                candidate=candidate,
                config=config,
                prefix=prefix,
                continuation=run,
                new_archive=tuple(archive.metadata),
            )
        else:
            if _campaign_budget_exhausted(
                prior_elapsed_seconds=prior_elapsed_seconds,
                current_elapsed_seconds=time.monotonic() - started,
            ):
                hard_vetoes.append("campaign_budget_exhausted")
                campaign_incomplete = True
                break
            run = run_sequential_neutra_hmc(
                adapter=adapter,
                initial_state=initial_state,
                model_transform=lambda values, transport=loaded.transport: _transform_batch(transport, values),
                parameter_names=base_adapter.parameter_names,
                config=config,
                retained_diagnostic_fn=lambda draws: rank_normalized_hmc_diagnostics(draws, parameter_names=base_adapter.parameter_names, thresholds=thresholds),
                archive_callback=archive,
            )
            row = {
                "candidate": candidate,
                "config": config.payload(chain_count=CHAIN_COUNT),
                "passed": bool(run["passed"]),
                "decision": run["decision"],
                "completion_status": "passed" if run["passed"] else "terminal_rejection",
                "terminal_candidate_result": True,
                "warmup_passed": run["warmup_passed"],
                "retained_passed": run["retained_passed"],
                "warmup_results_per_chain": run["warmup_results_per_chain"],
                "retained_results_per_chain": run["retained_results_per_chain"],
                "warmup_checks": run["warmup_checks"],
                "retained_checks": run["retained_checks"],
                "hard_vetoes": run["hard_vetoes"],
                "elapsed_seconds": run["elapsed_seconds"],
                "archive": archive.metadata,
            }
        rows.append(row)
        hard_vetoes.extend(f"{candidate_id}:{veto}" for veto in run["hard_vetoes"])
        checkpoint_rows = (
            _merge_replacement_rows(prior_rows, rows)
            if replacement_mode
            else list(tuple(prior_rows) + tuple(rows))
        )
        _write_progress(
            output_root / "progress.json",
            _progress_payload(
                prior_rows=checkpoint_rows,
                rows=[],
                planned_candidate_count=len(all_candidates),
                elapsed_seconds=time.monotonic() - started,
                terminal=False,
                prior_elapsed_seconds=prior_elapsed_seconds,
                prior_progress_sha256=prior_progress_sha256,
            ),
        )
        if row.get("terminal_candidate_result") is not True:
            hard_vetoes.append("campaign_budget_exhausted")
            campaign_incomplete = True
            break

    wall = time.monotonic() - started
    aggregate_elapsed = float(prior_elapsed_seconds) + wall
    final_rows = (
        _merge_replacement_rows(prior_rows, rows)
        if replacement_mode
        else list(tuple(prior_rows) + tuple(rows))
    )
    selected_terminal_count = sum(
        int(row.get("terminal_candidate_result", True)) for row in rows
    )
    campaign_completed = bool(
        len(rows) == len(candidates)
        and selected_terminal_count == len(candidates)
        and not campaign_incomplete
    )
    result = {
        "schema": "bayesfilter.pp_ukf.true_hmc_validation.result.v1",
        "status": (
            "completed_selected_candidates"
            if campaign_completed
            else "budget_stopped_incomplete"
        ),
        "candidate_rows": tuple(final_rows),
        "candidate_count_completed": sum(
            int(row.get("terminal_candidate_result", True)) for row in final_rows
        ),
        "candidate_count_planned": len(all_candidates),
        "selected_candidate_count_completed": selected_terminal_count,
        "selected_candidate_count_planned": len(candidates),
        "selected_candidate_indices": selected_indices,
        "prior_elapsed_seconds": float(prior_elapsed_seconds),
        "prior_progress_sha256": prior_progress_sha256,
        "viable_candidates": tuple(row["candidate"]["candidate_id"] for row in final_rows if row["passed"]),
        "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "wall_seconds": wall,
        "aggregate_elapsed_seconds": aggregate_elapsed,
        "target_signature": TARGET_SIGNATURE,
        "transport_sha256": TRANSPORT_SHA256,
        "metric_policy": "fixed_identity",
        "partition": partition,
        "recomposition": recomposition,
        "thresholds": thresholds.payload(),
        "memory_policy": memory_policy,
        "devices": tuple(str(d) for d in tf.config.list_logical_devices()),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "plan": str(PLAN),
        "native_divergence_status": "not_exposed_by_tfp_hamiltonian_monte_carlo",
        "ranking_performed": False,
        "continuation_mode": replacement_mode,
        "replacement_candidate_indices": selected_indices if replacement_mode else (),
        "continuation_prefix_results_per_chain": (
            CONTINUATION_PREFIX_RESULTS if replacement_mode else None
        ),
        "continuation_prefixes": (
            tuple(
                _continuation_prefix_payload(continuation_prefixes[index])
                for index in selected_indices
            )
            if replacement_mode
            else ()
        ),
        "continuation_chunk_budget_reserve_seconds": (
            CONTINUATION_CHUNK_BUDGET_RESERVE_SECONDS
            if replacement_mode
            else None
        ),
        "nonclaims": ("no sampler superiority", "no default readiness", "no claim beyond declared PP-UKF validation scope"),
    }
    _write_new_json(output_root / "public_result.json", result)
    _write_new_json(output_root / "run_manifest.json", {
        "schema": "bayesfilter.pp_ukf.true_hmc_validation.manifest.v1",
        "command": sys.argv,
        "started_utc": started_utc.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": result["git_commit"],
        "python": sys.version,
        "python_executable": sys.executable,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "platform": platform.platform(),
        "campaign_cap_seconds": CAMPAIGN_CAP_SECONDS,
        "wall_seconds": wall,
        "aggregate_elapsed_seconds": aggregate_elapsed,
        "target_signature": TARGET_SIGNATURE,
        "transport_path": str(TRANSPORT_PATH),
        "transport_sha256": TRANSPORT_SHA256,
        "partition": partition,
        "thresholds": thresholds.payload(),
        "memory_policy": memory_policy,
        "devices": tuple(str(d) for d in tf.config.list_logical_devices()),
        "dtype": "float64",
        "jit_compile": True,
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "selected_candidate_configs": tuple(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_index": index,
                "config": SequentialNeuTraHMCConfig(
                    step_size=float(candidate["step_size"]),
                    num_leapfrog_steps=int(candidate["num_leapfrog_steps"]),
                    warmup_seed=(
                        WARMUP_SEED_ROOT[0],
                        WARMUP_SEED_ROOT[1] + index * 1009,
                    ),
                    retained_seed=(
                        RETAINED_SEED_ROOT[0],
                        RETAINED_SEED_ROOT[1] + index * 1009,
                    ),
                    warmup_chunk_results=250,
                    warmup_min_results=WARMUP_MIN,
                    warmup_check_window_results=1000,
                    warmup_max_results=MAX_RESULTS,
                    warmup_rhat_max=WARMUP_RHAT_MAX,
                    retained_chunk_results=500,
                    retained_min_results=RETAINED_MIN,
                    retained_max_results=MAX_RESULTS,
                    retained_rhat_max=RETAINED_RHAT_MAX,
                    minimum_chain_count=CHAIN_COUNT,
                    jit_compile=True,
                ).payload(chain_count=CHAIN_COUNT),
            }
            for index, candidate in zip(selected_indices, candidates)
        ),
        "output_root": str(output_root),
        "progress_path": str(output_root / "progress.json"),
        "public_result_path": str(output_root / "public_result.json"),
        "run_manifest_path": str(output_root / "run_manifest.json"),
        "plan": str(PLAN),
        "sampling_launched": True,
        "warmup_excluded_from_posterior": True,
        "native_divergence_status": "not_exposed_by_tfp_hamiltonian_monte_carlo",
        "continuation_mode": replacement_mode,
        "replacement_candidate_indices": selected_indices if replacement_mode else (),
        "continuation_prefix_results_per_chain": (
            CONTINUATION_PREFIX_RESULTS if replacement_mode else None
        ),
        "prior_progress_sha256": prior_progress_sha256,
        "continuation_chunk_budget_reserve_seconds": (
            CONTINUATION_CHUNK_BUDGET_RESERVE_SECONDS
            if replacement_mode
            else None
        ),
    })
    _write_progress(
        output_root / "progress.json",
        _progress_payload(
            prior_rows=final_rows if replacement_mode else prior_rows,
            rows=[] if replacement_mode else rows,
            planned_candidate_count=len(all_candidates),
            elapsed_seconds=wall,
            terminal=True,
            prior_elapsed_seconds=prior_elapsed_seconds,
            prior_progress_sha256=prior_progress_sha256,
        ),
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-index", type=int, action="append")
    parser.add_argument("--replace-candidate-index", type=int, action="append")
    parser.add_argument("--prior-elapsed-seconds", type=float, default=0.0)
    parser.add_argument("--resume-progress", type=Path)
    args = parser.parse_args()
    try:
        result = run_campaign(
            args.output_root,
            candidate_indices=None if args.candidate_index is None else tuple(args.candidate_index),
            replacement_indices=None if args.replace_candidate_index is None else tuple(args.replace_candidate_index),
            prior_elapsed_seconds=args.prior_elapsed_seconds,
            resume_progress=args.resume_progress,
        )
    except Exception as exc:
        if args.output_root.is_dir():
            failure_path = args.output_root / "failure.json"
            if not failure_path.exists():
                _write_new_json(
                    failure_path,
                    {
                        "schema": "bayesfilter.pp_ukf.true_hmc_validation.failure.v1",
                        "status": "harness_exception",
                        "failed_utc": datetime.now(timezone.utc).isoformat(),
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                        "command": sys.argv,
                        "git_commit": subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], text=True
                        ).strip(),
                        "plan": str(PLAN),
                        "prior_elapsed_seconds": args.prior_elapsed_seconds,
                    },
                )
        raise
    print(json.dumps({"status": result["status"], "completed": result["candidate_count_completed"], "planned": result["candidate_count_planned"], "wall_seconds": result["wall_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = Path("docs/plans/bayesfilter-pp-ukf-true-hmc-validation-plan-2026-07-22.md")
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
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="ascii")


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
) -> Mapping[str, Any]:
    all_rows = tuple(prior_rows) + tuple(rows)
    return {
        "schema": "bayesfilter.pp_ukf.true_hmc_validation.progress.v1",
        "completed_candidate_count": len(all_rows),
        "planned_candidate_count": int(planned_candidate_count),
        "candidate_rows": all_rows,
        "elapsed_seconds": float(elapsed_seconds),
        "terminal": bool(terminal),
    }


def _campaign_budget_exhausted(
    *, prior_elapsed_seconds: float, current_elapsed_seconds: float
) -> bool:
    """Return whether another chunk would violate the aggregate campaign cap."""
    return (
        float(prior_elapsed_seconds) + float(current_elapsed_seconds)
        >= CAMPAIGN_CAP_SECONDS
    )


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
    retained_seed = tuple(int(item) for item in row["config"]["retained_seed"])
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
        index: _verify_continuation_prefix(rows[index], candidate=candidates[index])
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
        latent_path.write_bytes(bytes(tf.io.serialize_tensor(latent).numpy()))
        raw_path.write_bytes(bytes(tf.io.serialize_tensor(raw).numpy()))
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


def _run_retained_continuation(
    *,
    adapter: Any,
    transport: Any,
    candidate: Mapping[str, Any],
    prefix: Mapping[str, Any],
    base_adapter: Any,
    output_root: Path,
    thresholds: Any,
    archive: Any,
    tf: Any,
    rank_normalized_hmc_diagnostics: Any,
    config: Any,
    campaign_started: float,
    prior_elapsed_seconds: float,
) -> Mapping[str, Any]:
    from bayesfilter.inference.neutra_hmc import (
        _build_batched_hmc_program,
        _summarize_batched_hmc_output,
        sequential_chunk_seed,
    )

    prefix_latent = _parse_tensor(prefix["latent_path"], tf)
    prefix_raw = _parse_tensor(prefix["raw_path"], tf)
    if tuple(int(value) for value in prefix_latent.shape) != (
        CONTINUATION_PREFIX_RESULTS,
        CHAIN_COUNT,
        6,
    ) or tuple(int(value) for value in prefix_raw.shape) != tuple(
        int(value) for value in prefix_latent.shape
    ):
        raise ValueError("continuation prefix tensor shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(prefix_latent)).numpy()) or not bool(
        tf.reduce_all(tf.math.is_finite(prefix_raw)).numpy()
    ):
        raise ValueError("continuation prefix tensors must be finite")
    state = tf.convert_to_tensor(prefix_latent[-1], tf.float64)
    compiled: dict[int, Any] = {}
    retained_latent_chunks = [prefix_latent]
    retained_raw_chunks = [prefix_raw]
    retained_checks: list[Mapping[str, Any]] = []
    hard_vetoes: list[str] = []
    retained_count = CONTINUATION_PREFIX_RESULTS
    retained_passed = False
    started = time.monotonic()
    retained_seed = tuple(int(item) for item in config.retained_seed)
    chunk_index = int(prefix["next_retained_chunk_index"])
    while retained_count < config.retained_max_results:
        if _campaign_budget_exhausted(
            prior_elapsed_seconds=prior_elapsed_seconds,
            current_elapsed_seconds=time.monotonic() - campaign_started,
        ):
            hard_vetoes.append("campaign_budget_exhausted")
            break
        active = min(config.retained_chunk_results, config.retained_max_results - retained_count)
        if active not in compiled:
            compiled[active] = _build_batched_hmc_program(
                adapter=adapter,
                num_results=active,
                num_burnin_steps=0,
                step_size=config.step_size,
                num_leapfrog_steps=config.num_leapfrog_steps,
                jit_compile=config.jit_compile,
            )
        seed = sequential_chunk_seed(retained_seed, chunk_index)
        samples, trace = compiled[active](state, tf.constant(seed, tf.int32))
        chunk = _summarize_batched_hmc_output(
            initial_state=state,
            samples=samples,
            trace=trace,
            config=type("ContinuationChunkConfig", (), {
                "energy_error_log_accept_threshold": config.energy_error_log_accept_threshold,
                "jit_compile": config.jit_compile,
            })(),
            chain_count=CHAIN_COUNT,
            elapsed_seconds=0.0,
        )
        latent_samples = tf.convert_to_tensor(chunk["samples"], tf.float64)
        model_samples = tf.convert_to_tensor(
            transport.forward_batch(tf.reshape(latent_samples, (-1, 6))), tf.float64
        )
        model_samples = tf.reshape(model_samples, tf.shape(latent_samples))
        retained_latent_chunks.append(latent_samples)
        retained_raw_chunks.append(model_samples)
        state = latent_samples[-1]
        retained_count += active
        if archive is not None:
            _call_archive = archive
            _call_archive(
                stage="retained",
                chunk_index=chunk_index,
                latent_samples=latent_samples,
                model_samples=model_samples,
                seed=seed,
                cumulative=False,
            )
        cumulative_raw = tf.concat(retained_raw_chunks, axis=0)
        diagnostic = rank_normalized_hmc_diagnostics(
            cumulative_raw,
            parameter_names=base_adapter.parameter_names,
            thresholds=thresholds,
        )
        check = {
            "chunk_index": chunk_index,
            "completed_results_per_chain": retained_count,
            "seed": seed,
            "health": chunk["diagnostics"],
            "diagnostic_role": "full_convergence",
            "full_convergence": diagnostic,
            "passed": bool(retained_count >= config.retained_min_results and diagnostic["passed"]),
            "continuation_prefix_results_per_chain": CONTINUATION_PREFIX_RESULTS,
        }
        retained_checks.append(check)
        if chunk["diagnostics"]["health_passed"] is not True:
            hard_vetoes.append("retained_continuation_chunk_health_failed")
            break
        retained_passed = bool(check["passed"])
        if retained_passed:
            break
        chunk_index += 1

    cumulative_latent = tf.concat(retained_latent_chunks, axis=0)
    cumulative_raw = tf.concat(retained_raw_chunks, axis=0)
    cumulative_archive = None
    if archive is not None:
        cumulative_archive = archive(
            stage="retained",
            chunk_index=None,
            latent_samples=cumulative_latent,
            model_samples=cumulative_raw,
            seed=None,
            cumulative=True,
        )
    final_diagnostic = rank_normalized_hmc_diagnostics(
        cumulative_raw,
        parameter_names=base_adapter.parameter_names,
        thresholds=thresholds,
    )
    return {
        "passed": bool(retained_passed and not hard_vetoes),
        "decision": "ADMIT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL" if retained_passed and not hard_vetoes else "REJECT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL",
        "warmup_passed": True,
        "retained_passed": retained_passed,
        "warmup_results_per_chain": WARMUP_MIN,
        "retained_results_per_chain": retained_count,
        "warmup_checks": (),
        "retained_checks": tuple(retained_checks),
        "hard_vetoes": tuple(dict.fromkeys(hard_vetoes)),
        "elapsed_seconds": time.monotonic() - started,
        "archive": getattr(archive, "metadata", ()) if archive is not None else (),
        "continuation_prefix": prefix,
        "final_diagnostic": final_diagnostic,
        "cumulative_archive": cumulative_archive,
    }


def run_campaign(
    output_root: Path,
    *,
    candidate_indices: tuple[int, ...] | None = None,
    replacement_indices: tuple[int, ...] | None = None,
    prior_elapsed_seconds: float = 0.0,
    resume_progress: Path | None = None,
) -> Mapping[str, Any]:
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
        else:
            completed_ids = {str(row["candidate"]["candidate_id"]) for row in prior_rows}
            selected_indices = tuple(index for index in selected_indices if str(all_candidates[index]["candidate_id"]) not in completed_ids)
    elif replacement_mode:
        raise ValueError("replacement continuation requires --resume-progress")
    if not selected_indices or any(index < 0 or index >= len(all_candidates) for index in selected_indices):
        raise ValueError("candidate_indices must select valid manifest rows")
    candidates = tuple(all_candidates[index] for index in selected_indices)
    thresholds = RankNormalizedHMCThresholds(rhat_max=RETAINED_RHAT_MAX, bulk_ess_min=BULK_ESS_MIN, tail_ess_min=TAIL_ESS_MIN)
    rows: list[Mapping[str, Any]] = []
    hard_vetoes: list[str] = []
    for index, candidate in zip(selected_indices, candidates):
        elapsed = time.monotonic() - started
        if _campaign_budget_exhausted(
            prior_elapsed_seconds=prior_elapsed_seconds,
            current_elapsed_seconds=elapsed,
        ):
            hard_vetoes.append("campaign_budget_exhausted")
            break
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
            run = _run_retained_continuation(
                adapter=adapter,
                transport=loaded.transport,
                candidate=candidate,
                prefix=continuation_prefixes[index],
                base_adapter=base_adapter,
                output_root=output_root,
                thresholds=thresholds,
                archive=archive,
                tf=tf,
                rank_normalized_hmc_diagnostics=rank_normalized_hmc_diagnostics,
                config=config,
                campaign_started=started,
                prior_elapsed_seconds=prior_elapsed_seconds,
            )
        else:
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
        if replacement_mode:
            row["continuation_prefix"] = run["continuation_prefix"]
            row["continuation_replacement"] = True
            row["continuation_prefix_results_per_chain"] = CONTINUATION_PREFIX_RESULTS
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
            ),
        )

    wall = time.monotonic() - started
    aggregate_elapsed = float(prior_elapsed_seconds) + wall
    final_rows = (
        _merge_replacement_rows(prior_rows, rows)
        if replacement_mode
        else list(tuple(prior_rows) + tuple(rows))
    )
    result = {
        "schema": "bayesfilter.pp_ukf.true_hmc_validation.result.v1",
        "status": "completed_selected_candidates" if len(rows) == len(candidates) else "budget_or_veto_stopped",
        "candidate_rows": tuple(final_rows),
        "candidate_count_completed": len(final_rows),
        "candidate_count_planned": len(all_candidates),
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
            tuple(continuation_prefixes[index] for index in selected_indices)
            if replacement_mode
            else ()
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
        "sampling_launched": True,
        "warmup_excluded_from_posterior": True,
        "native_divergence_status": "not_exposed_by_tfp_hamiltonian_monte_carlo",
        "continuation_mode": replacement_mode,
        "replacement_candidate_indices": selected_indices if replacement_mode else (),
        "continuation_prefix_results_per_chain": (
            CONTINUATION_PREFIX_RESULTS if replacement_mode else None
        ),
        "prior_progress_sha256": prior_progress_sha256,
    })
    _write_progress(
        output_root / "progress.json",
        _progress_payload(
            prior_rows=final_rows if replacement_mode else prior_rows,
            rows=[] if replacement_mode else rows,
            planned_candidate_count=len(all_candidates),
            elapsed_seconds=wall,
            terminal=True,
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
    result = run_campaign(
        args.output_root,
        candidate_indices=None if args.candidate_index is None else tuple(args.candidate_index),
        replacement_indices=None if args.replace_candidate_index is None else tuple(args.replace_candidate_index),
        prior_elapsed_seconds=args.prior_elapsed_seconds,
        resume_progress=args.resume_progress,
    )
    print(json.dumps({"status": result["status"], "completed": result["candidate_count_completed"], "planned": result["candidate_count_planned"], "wall_seconds": result["wall_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

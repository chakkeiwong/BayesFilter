"""CPU multicore external sample-generation boundary for frozen NeuTra payloads.

This Phase 12 fixture demonstrates a narrow post-training sample-generation
boundary: read a frozen affine payload, split a tiny deterministic diagnostic
draw workload across CPU workers, and write provenance-rich JSON.

It does not train NeuTra, run HMC sampling or tuning, use GPU sample generation,
repair XLA, or establish posterior correctness.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPECTED_PHASE10_TARGET_SIGNATURE = (
    "290a91d2a8f90d5b29243965b258b1ec6fd965aa46ffca69dcb78f7fa1ecabcb"
)
EXPECTED_PHASE10_ADAPTER_SIGNATURE = (
    "0a48b43d2750cad5b452708f7619a1119a259231d8955769809460f256575a97"
)
DEFAULT_PHASE11_ARTIFACT_DIR = Path(
    "docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07"
)
DEFAULT_PHASE11_PAYLOAD_PATH = DEFAULT_PHASE11_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_gpu_frozen_payload_seed20260707.json"
)
DEFAULT_PHASE12_OUTPUT_PATH = DEFAULT_PHASE11_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_cpu_multicore_sample_boundary_seed20260707.json"
)
DEFAULT_SEED = 20260707
DEFAULT_SAMPLE_COUNT = 12
DEFAULT_WORKER_COUNT = 2

NEUTRA_CPU_SAMPLE_BOUNDARY_NONCLAIMS = (
    "Phase 12 CPU multicore external sample-generation boundary only",
    "generated draws are diagnostic base/transport samples, not posterior samples",
    "no NeuTra training claim",
    "no CPU NeuTra training claim",
    "no GPU sample generation claim",
    "no HMC tuning or sampling claim",
    "no XLA readiness claim",
    "no posterior convergence claim",
    "no route ranking claim",
    "no production readiness claim",
    "no default execution readiness claim",
    "no scientific validity claim",
)


class NeuTraCPUSampleBoundaryError(RuntimeError):
    """Raised when the Phase 12 CPU sample boundary is violated."""


@dataclass(frozen=True)
class NeuTraCPUSampleBoundaryConfig:
    """Configuration for Phase 12 CPU multicore sample-boundary smoke."""

    payload_path: Path = DEFAULT_PHASE11_PAYLOAD_PATH
    output_path: Path = DEFAULT_PHASE12_OUTPUT_PATH
    seed: int = DEFAULT_SEED
    sample_count: int = DEFAULT_SAMPLE_COUNT
    worker_count: int = DEFAULT_WORKER_COUNT
    require_cpu_hidden: bool = True
    allow_neutra_training: bool = False
    allow_cpu_neutra_training: bool = False
    allow_hmc_sampling_or_tuning: bool = False
    allow_gpu_sample_generation: bool = False
    allow_xla: bool = False
    target_signature: str = EXPECTED_PHASE10_TARGET_SIGNATURE
    adapter_signature: str = EXPECTED_PHASE10_ADAPTER_SIGNATURE

    def normalized(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.cpu_sample_boundary_config.v1",
            "phase": "phase12_cpu_multicore_external_sample_boundary",
            "payload_path": str(self.payload_path),
            "output_path": str(self.output_path),
            "seed": int(self.seed),
            "sample_count": int(self.sample_count),
            "worker_count": int(self.worker_count),
            "require_cpu_hidden": bool(self.require_cpu_hidden),
            "execution_target": "cpu_multicore_external_sample_generation_boundary",
            "sample_semantics": (
                "diagnostic_standard_normal_base_draws_and_frozen_affine_forward"
            ),
            "allow_neutra_training": bool(self.allow_neutra_training),
            "allow_cpu_neutra_training": bool(self.allow_cpu_neutra_training),
            "allow_hmc_sampling_or_tuning": bool(self.allow_hmc_sampling_or_tuning),
            "allow_gpu_sample_generation": bool(self.allow_gpu_sample_generation),
            "allow_xla": bool(self.allow_xla),
            "expected_target_signature": str(self.target_signature),
            "expected_adapter_signature": str(self.adapter_signature),
            "nonclaims": NEUTRA_CPU_SAMPLE_BOUNDARY_NONCLAIMS,
        }


@dataclass(frozen=True)
class NeuTraCPUSampleBoundaryResult:
    """Materialized Phase 12 sample-boundary artifact."""

    config: NeuTraCPUSampleBoundaryConfig
    payload: Mapping[str, Any]
    artifact: Mapping[str, Any]
    output_path: Path

    @property
    def passed(self) -> bool:
        return bool(self.artifact["passed"])


def generate_cpu_multicore_external_sample_boundary(
    config: NeuTraCPUSampleBoundaryConfig | None = None,
) -> NeuTraCPUSampleBoundaryResult:
    """Generate tiny diagnostic base/affine samples through a CPU worker pool."""

    cfg = NeuTraCPUSampleBoundaryConfig() if config is None else config
    start = time.monotonic()
    _validate_config(cfg)
    payload = _read_json_mapping(cfg.payload_path, "Phase 11 payload")
    _validate_payload(
        payload,
        expected_target_signature=str(cfg.target_signature),
        expected_adapter_signature=str(cfg.adapter_signature),
    )
    chunks = _chunk_work(int(cfg.sample_count), int(cfg.worker_count))
    worker_inputs = [
        {
            "chunk_index": chunk_index,
            "start": start_index,
            "count": count,
            "seed": int(cfg.seed),
            "dimension": int(payload["dimension"]),
            "shift": list(_float_tuple(payload.get("shift"), "shift")),
            "raw_scale": list(_float_tuple(payload.get("raw_scale"), "raw_scale")),
        }
        for chunk_index, (start_index, count) in enumerate(chunks)
        if count > 0
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(cfg.worker_count),
    ) as executor:
        worker_outputs = list(executor.map(_generate_worker_chunk, worker_inputs))
    worker_outputs = sorted(worker_outputs, key=lambda row: int(row["chunk_index"]))
    base_samples = [
        sample
        for worker_output in worker_outputs
        for sample in worker_output["base_samples"]
    ]
    transported_samples = [
        sample
        for worker_output in worker_outputs
        for sample in worker_output["transported_samples"]
    ]
    if len(base_samples) != int(cfg.sample_count):
        raise NeuTraCPUSampleBoundaryError("generated sample count mismatch")
    artifact = _artifact_payload(
        config=cfg,
        payload=payload,
        worker_outputs=worker_outputs,
        base_samples=base_samples,
        transported_samples=transported_samples,
        elapsed_seconds=time.monotonic() - start,
    )
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.output_path, artifact)
    return NeuTraCPUSampleBoundaryResult(
        config=cfg,
        payload=payload,
        artifact=artifact,
        output_path=cfg.output_path,
    )


def _validate_config(config: NeuTraCPUSampleBoundaryConfig) -> None:
    if bool(config.require_cpu_hidden) and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise NeuTraCPUSampleBoundaryError(
            "Phase 12 sample-boundary checks must run with CUDA_VISIBLE_DEVICES=-1"
        )
    if int(config.seed) < 0:
        raise NeuTraCPUSampleBoundaryError("seed must be nonnegative")
    if int(config.sample_count) <= 0 or int(config.sample_count) > 10000:
        raise NeuTraCPUSampleBoundaryError("sample_count must be in 1..10000")
    if int(config.worker_count) < 2 or int(config.worker_count) > 64:
        raise NeuTraCPUSampleBoundaryError("worker_count must be in 2..64")
    forbidden_flags = {
        "allow_neutra_training": bool(config.allow_neutra_training),
        "allow_cpu_neutra_training": bool(config.allow_cpu_neutra_training),
        "allow_hmc_sampling_or_tuning": bool(config.allow_hmc_sampling_or_tuning),
        "allow_gpu_sample_generation": bool(config.allow_gpu_sample_generation),
        "allow_xla": bool(config.allow_xla),
    }
    enabled = [name for name, value in forbidden_flags.items() if value]
    if enabled:
        raise NeuTraCPUSampleBoundaryError(
            "Phase 12 forbids these capabilities: " + ", ".join(enabled)
        )
    _require_sha256_hex(str(config.target_signature), "target_signature")
    _require_sha256_hex(str(config.adapter_signature), "adapter_signature")


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    expected_target_signature: str,
    expected_adapter_signature: str,
) -> None:
    if payload.get("schema") != "bayesfilter.neutra.frozen_affine_diag.v1":
        raise NeuTraCPUSampleBoundaryError("Phase 11 payload schema mismatch")
    if payload.get("phase") != "phase11_frozen_gpu_trained_affine_payload":
        raise NeuTraCPUSampleBoundaryError("Phase 11 payload phase mismatch")
    if payload.get("target_signature") != expected_target_signature:
        raise NeuTraCPUSampleBoundaryError("target_signature mismatch")
    if payload.get("source_adapter_signature") != expected_adapter_signature:
        raise NeuTraCPUSampleBoundaryError("adapter_signature mismatch")
    if payload.get("source_training_execution_target") != "gpu_required":
        raise NeuTraCPUSampleBoundaryError("source payload must come from GPU training")
    if bool(payload.get("training_executed")):
        raise NeuTraCPUSampleBoundaryError("Phase 12 must not consume training payload")
    if bool(payload.get("hmc_executed")):
        raise NeuTraCPUSampleBoundaryError("Phase 12 must not consume HMC payload")
    if bool(payload.get("external_sample_generation_executed")):
        raise NeuTraCPUSampleBoundaryError(
            "Phase 12 must start from a payload with no prior sample generation"
        )
    if bool(payload.get("jit_compile")):
        raise NeuTraCPUSampleBoundaryError("Phase 12 must not require JIT/XLA")
    dimension = int(payload.get("dimension", 0))
    if dimension <= 0:
        raise NeuTraCPUSampleBoundaryError("payload dimension must be positive")
    if len(_float_tuple(payload.get("shift"), "shift")) != dimension:
        raise NeuTraCPUSampleBoundaryError("payload shift length mismatch")
    if len(_float_tuple(payload.get("raw_scale"), "raw_scale")) != dimension:
        raise NeuTraCPUSampleBoundaryError("payload raw_scale length mismatch")


def _artifact_payload(
    *,
    config: NeuTraCPUSampleBoundaryConfig,
    payload: Mapping[str, Any],
    worker_outputs: Sequence[Mapping[str, Any]],
    base_samples: Sequence[Sequence[float]],
    transported_samples: Sequence[Sequence[float]],
    elapsed_seconds: float,
) -> Mapping[str, Any]:
    base_finite = _nested_sequence_is_finite(base_samples)
    transported_finite = _nested_sequence_is_finite(transported_samples)
    worker_ids = [int(output["process_id"]) for output in worker_outputs]
    distinct_worker_processes = len(set(worker_ids))
    finite_checks = {
        "base_samples_finite": base_finite,
        "transported_samples_finite": transported_finite,
        "sample_count_match": len(base_samples) == int(config.sample_count),
        "sample_dimension_match": all(
            len(sample) == int(payload["dimension"]) for sample in base_samples
        )
        and all(len(sample) == int(payload["dimension"]) for sample in transported_samples),
    }
    boundary_checks = {
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "worker_count_recorded": int(config.worker_count) >= 2,
        "multicore_worker_pool_requested": True,
        "distinct_worker_processes": distinct_worker_processes,
        "source_gpu_training_artifact": (
            payload.get("source_training_execution_target") == "gpu_required"
        ),
        "new_neutra_training_executed": False,
        "cpu_neutra_training_executed": False,
        "gpu_sample_generation_executed": False,
        "hmc_sampling_or_tuning_executed": False,
        "xla_repair_executed": False,
    }
    passed = (
        all(bool(value) for value in finite_checks.values())
        and bool(boundary_checks["cpu_hidden"])
        and bool(boundary_checks["worker_count_recorded"])
        and bool(boundary_checks["multicore_worker_pool_requested"])
        and int(boundary_checks["distinct_worker_processes"]) >= 1
        and bool(boundary_checks["source_gpu_training_artifact"])
        and not bool(boundary_checks["new_neutra_training_executed"])
        and not bool(boundary_checks["cpu_neutra_training_executed"])
        and not bool(boundary_checks["gpu_sample_generation_executed"])
        and not bool(boundary_checks["hmc_sampling_or_tuning_executed"])
        and not bool(boundary_checks["xla_repair_executed"])
    )
    artifact_without_hash = {
        "schema": "bayesfilter.neutra.cpu_multicore_sample_boundary.v1",
        "phase": "phase12_cpu_multicore_external_sample_boundary",
        "passed": bool(passed),
        "decision": (
            "PASS_PHASE12_CPU_MULTICORE_SAMPLE_BOUNDARY"
            if passed
            else "BLOCK_PHASE12_CPU_MULTICORE_SAMPLE_BOUNDARY"
        ),
        "config": config.normalized(),
        "target_signature": str(config.target_signature),
        "adapter_signature": str(config.adapter_signature),
        "source_payload_path": str(config.payload_path),
        "source_payload_file_sha256": _file_sha256(config.payload_path),
        "source_payload_stable_hash": f"sha256:{_stable_json_hash(payload)}",
        "source_transport_id": str(payload["transport_id"]),
        "source_transport_hash": str(payload.get("transport_hash", "not_in_payload")),
        "source_payload_signature": str(payload.get("source_training_state_artifact_hash", "")),
        "sample_semantics": (
            "diagnostic_standard_normal_base_draws_and_frozen_affine_forward"
        ),
        "sample_interpretation": (
            "not_posterior_samples_not_hmc_samples_not_transport_quality_evidence"
        ),
        "sample_count": int(config.sample_count),
        "dimension": int(payload["dimension"]),
        "worker_count": int(config.worker_count),
        "worker_process_ids": worker_ids,
        "worker_chunks": [
            {
                "chunk_index": int(output["chunk_index"]),
                "start": int(output["start"]),
                "count": int(output["count"]),
                "process_id": int(output["process_id"]),
            }
            for output in worker_outputs
        ],
        "base_samples": _json_safe(base_samples),
        "transported_samples": _json_safe(transported_samples),
        "sample_summary": _sample_summary(transported_samples),
        "finite_checks": finite_checks,
        "boundary_checks": boundary_checks,
        "elapsed_seconds": float(elapsed_seconds),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "training_executed": False,
        "cpu_neutra_training_executed": False,
        "hmc_executed": False,
        "external_sample_generation_executed": True,
        "gpu_sample_generation_executed": False,
        "jit_compile": False,
        "xla_readiness_claimed": False,
        "nonclaims": NEUTRA_CPU_SAMPLE_BOUNDARY_NONCLAIMS,
    }
    return {
        **artifact_without_hash,
        "artifact_stable_hash": f"sha256:{_stable_json_hash(artifact_without_hash)}",
        "artifact_hash_semantics": "stable_json_sha256_excluding_self_hash_field",
    }


def _generate_worker_chunk(spec: Mapping[str, Any]) -> Mapping[str, Any]:
    start = int(spec["start"])
    count = int(spec["count"])
    seed = int(spec["seed"])
    dimension = int(spec["dimension"])
    shift = [float(value) for value in spec["shift"]]
    raw_scale = [float(value) for value in spec["raw_scale"]]
    scale = [math.exp(value) for value in raw_scale]
    base_samples = []
    transported_samples = []
    for sample_index in range(start, start + count):
        sample = [
            _deterministic_standard_normal(
                seed=seed,
                sample_index=sample_index,
                dimension_index=dimension_index,
            )
            for dimension_index in range(dimension)
        ]
        transported = [
            shift[dimension_index] + scale[dimension_index] * sample[dimension_index]
            for dimension_index in range(dimension)
        ]
        base_samples.append(sample)
        transported_samples.append(transported)
    return {
        "chunk_index": int(spec["chunk_index"]),
        "start": start,
        "count": count,
        "process_id": os.getpid(),
        "base_samples": base_samples,
        "transported_samples": transported_samples,
    }


def _deterministic_standard_normal(
    *,
    seed: int,
    sample_index: int,
    dimension_index: int,
) -> float:
    random_seed = (
        (int(seed) + 1) * 1_000_003
        + (int(sample_index) + 1) * 9_176
        + (int(dimension_index) + 1) * 611_953
    )
    generator = random.Random(random_seed)
    return float(generator.gauss(0.0, 1.0))


def _chunk_work(sample_count: int, worker_count: int) -> tuple[tuple[int, int], ...]:
    base = sample_count // worker_count
    remainder = sample_count % worker_count
    chunks = []
    start = 0
    for worker_index in range(worker_count):
        count = base + (1 if worker_index < remainder else 0)
        chunks.append((start, count))
        start += count
    return tuple(chunks)


def _sample_summary(samples: Sequence[Sequence[float]]) -> Mapping[str, Any]:
    if not samples:
        return {"count": 0, "mean": [], "min": [], "max": []}
    dimension = len(samples[0])
    means = []
    mins = []
    maxs = []
    for dimension_index in range(dimension):
        values = [float(sample[dimension_index]) for sample in samples]
        means.append(sum(values) / len(values))
        mins.append(min(values))
        maxs.append(max(values))
    return {
        "count": len(samples),
        "mean": means,
        "min": mins,
        "max": maxs,
        "summary_interpretation": "descriptive_only_not_posterior_diagnostic",
    }


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise NeuTraCPUSampleBoundaryError(f"{label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise NeuTraCPUSampleBoundaryError(f"{label} must be a JSON object")
    return payload


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if value is None:
        raise NeuTraCPUSampleBoundaryError(f"{name} is required")
    try:
        result = tuple(float(item) for item in value)
    except TypeError as exc:
        raise NeuTraCPUSampleBoundaryError(f"{name} must be a numeric sequence") from exc
    if not result:
        raise NeuTraCPUSampleBoundaryError(f"{name} must be nonempty")
    if not _nested_sequence_is_finite([result]):
        raise NeuTraCPUSampleBoundaryError(f"{name} must be finite")
    return result


def _nested_sequence_is_finite(values: Sequence[Sequence[float]]) -> bool:
    return all(
        _is_finite_number(float(item))
        for row in values
        for item in row
    )


def _is_finite_number(value: float) -> bool:
    return value == value and abs(value) < float("inf")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stable_json_hash(payload: Any) -> str:
    blob = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256_hex(value: str, name: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise NeuTraCPUSampleBoundaryError(f"{name} must be a lowercase sha256 hex")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for Phase 12 sample-boundary smoke."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-path", type=Path, default=DEFAULT_PHASE11_PAYLOAD_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PHASE12_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--worker-count", type=int, default=DEFAULT_WORKER_COUNT)
    args = parser.parse_args(argv)
    result = generate_cpu_multicore_external_sample_boundary(
        NeuTraCPUSampleBoundaryConfig(
            payload_path=args.payload_path,
            output_path=args.output_path,
            seed=args.seed,
            sample_count=args.sample_count,
            worker_count=args.worker_count,
        )
    )
    print(
        json.dumps(
            {
                "passed": bool(result.passed),
                "output_path": str(result.output_path),
                "sample_count": result.artifact["sample_count"],
                "worker_count": result.artifact["worker_count"],
                "target_signature": result.artifact["target_signature"],
                "artifact_stable_hash": result.artifact["artifact_stable_hash"],
                "nonclaims": NEUTRA_CPU_SAMPLE_BOUNDARY_NONCLAIMS,
            },
            sort_keys=True,
        )
    )
    if not result.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

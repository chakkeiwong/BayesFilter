"""Runtime helpers with lazy public exports.

Importing a focused device or GPU-memory policy module must not eagerly load
the unrelated NumPy-backed generic runner into a TensorFlow-only process.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "CandidateResult",
    "EVIDENCE_MANIFEST_SCOPES",
    "EvidenceManifest",
    "GPUSelection",
    "PartialResultSnapshot",
    "ReducerRowStatus",
    "RunManifest",
    "StageEvent",
    "TIMING_BUCKET_NAMES",
    "TF_GPU_MEMORY_POLICY_SCHEMA",
    "TensorFlowGPUMemoryPolicyError",
    "TimingBucket",
    "TimeoutRecord",
    "VALID_REDUCER_STATUSES",
    "WorkerManifest",
    "WorkerRecord",
    "append_heartbeat",
    "append_jsonl",
    "append_stage_event",
    "assert_cpu_only_env",
    "assert_gpu_memory_growth_env",
    "assert_gpu_memory_growth_env",
    "atomic_write_json",
    "build_worker_manifest",
    "build_trusted_gpu_snapshot",
    "canonical_candidate_order",
    "configs_match_exact",
    "configure_tensorflow_gpu_memory_growth",
    "ensure_cpu_only_env",
    "ensure_gpu_memory_growth_env",
    "ensure_gpu_memory_growth_env",
    "make_timing_bucket",
    "record_timeout",
    "record_worker_result",
    "reduce_worker_artifacts",
    "select_first_tie_candidate",
    "select_preferred_gpu",
    "stable_config_hash",
    "stale_artifacts_match_exact",
    "stale_match_payload",
    "write_evidence_manifest",
    "write_partial_result_snapshot",
    "write_worker_manifest",
]

_EXPORTS = {
    "GPUSelection": "bayesfilter.runtime.device_policy",
    "assert_cpu_only_env": "bayesfilter.runtime.device_policy",
    "assert_gpu_memory_growth_env": "bayesfilter.runtime.device_policy",
    "build_trusted_gpu_snapshot": "bayesfilter.runtime.device_policy",
    "ensure_cpu_only_env": "bayesfilter.runtime.device_policy",
    "ensure_gpu_memory_growth_env": "bayesfilter.runtime.device_policy",
    "select_preferred_gpu": "bayesfilter.runtime.device_policy",
    "TF_GPU_MEMORY_POLICY_SCHEMA": "bayesfilter.runtime.gpu_memory_policy",
    "TensorFlowGPUMemoryPolicyError": "bayesfilter.runtime.gpu_memory_policy",
    "configure_tensorflow_gpu_memory_growth": "bayesfilter.runtime.gpu_memory_policy",
    "CandidateResult": "bayesfilter.runtime.selection",
    "canonical_candidate_order": "bayesfilter.runtime.selection",
    "select_first_tie_candidate": "bayesfilter.runtime.selection",
}

for _name in (
    "EVIDENCE_MANIFEST_SCOPES",
    "EvidenceManifest",
    "PartialResultSnapshot",
    "ReducerRowStatus",
    "RunManifest",
    "StageEvent",
    "TIMING_BUCKET_NAMES",
    "TimingBucket",
    "TimeoutRecord",
    "VALID_REDUCER_STATUSES",
    "WorkerManifest",
    "WorkerRecord",
    "append_heartbeat",
    "append_jsonl",
    "append_stage_event",
    "atomic_write_json",
    "build_worker_manifest",
    "configs_match_exact",
    "make_timing_bucket",
    "record_timeout",
    "record_worker_result",
    "reduce_worker_artifacts",
    "stable_config_hash",
    "stale_artifacts_match_exact",
    "stale_match_payload",
    "write_evidence_manifest",
    "write_partial_result_snapshot",
    "write_worker_manifest",
):
    _EXPORTS[_name] = "bayesfilter.runtime.runner"


def __getattr__(name: str):
    try:
        module_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

"""Phase 19 CPU multicore fixed-transport HMC chain harness boundary.

This helper validates the CPU-hidden worker boundary for later fixed-transport
LGSSM NeuTra HMC chains.  Each worker loads the Phase 17 frozen payload, binds
the fixed transport to the current LGSSM target, and compiles a tiny static
chain value/score function with ``jit_compile=True``.

It does not train NeuTra, run full HMC sampling or tuning, validate posterior
agreement, generate GPU samples, run a non-JIT fallback, or establish readiness
claims.  Phase 20 owns LGSSM reference posterior validation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bayesfilter.runtime.device_policy import assert_cpu_only_env, ensure_cpu_only_env


PHASE19_ROUTE = "phase19_cpu_multicore_hmc_chain_harness"
DEFAULT_PHASE16_ARTIFACT_DIR = Path(
    "docs/plans/artifacts/lgssm-first-neutra-gpu-training-2026-07-07"
)
DEFAULT_SEED = 20260707
DEFAULT_PHASE17_PAYLOAD_PATH = DEFAULT_PHASE16_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_"
    "gpu_xla_frozen_payload_seed20260707.json"
)
DEFAULT_PHASE18_OUTPUT_PATH = DEFAULT_PHASE16_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_"
    "phase18_fixed_transport_hmc_mechanics_xla_compile_seed20260707.json"
)
DEFAULT_PHASE19_OUTPUT_PATH = DEFAULT_PHASE16_ARTIFACT_DIR / (
    "lgssm_static_qr_exact_kalman_affine_neutra_"
    "phase19_cpu_multicore_hmc_chain_harness_seed20260707.json"
)
EXPECTED_PHASE19_TARGET_SIGNATURE = (
    "275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038"
)
EXPECTED_PHASE19_ADAPTER_SIGNATURE = (
    "d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900"
)
EXPECTED_PHASE19_FIXED_TRANSPORT_ADAPTER_SIGNATURE = (
    "db6b58a7adc8190f5ed2e48e42482956d32faf02bdf10a7104659a2bd86722c9"
)

NEUTRA_CPU_MULTICORE_HMC_CHAIN_HARNESS_NONCLAIMS = (
    "Phase 19 CPU multicore HMC chain harness boundary only",
    "tiny worker value/score compile smoke only",
    "no full-chain HMC sampling or tuning claim",
    "no HMC transition claim",
    "no posterior reference validation claim",
    "no NeuTra training claim",
    "no GPU sample generation claim",
    "no sampler convergence claim",
    "no production readiness claim",
    "no default execution readiness claim",
    "no scientific validity claim",
)


class NeuTraCPUMulticoreHMCChainHarnessError(RuntimeError):
    """Raised when the Phase 19 harness boundary is violated."""


@dataclass(frozen=True)
class NeuTraCPUMulticoreHMCChainHarnessConfig:
    """Configuration for the Phase 19 CPU multicore harness smoke."""

    payload_path: Path = DEFAULT_PHASE17_PAYLOAD_PATH
    phase18_diagnostic_path: Path = DEFAULT_PHASE18_OUTPUT_PATH
    output_path: Path = DEFAULT_PHASE19_OUTPUT_PATH
    seed: int = DEFAULT_SEED
    worker_count: int = 2
    chain_count: int = 2
    num_results: int = 1
    num_burnin_steps: int = 0
    num_leapfrog_steps: int = 2
    step_size: float = 0.1
    jit_compile: bool = True
    require_cpu_hidden: bool = True
    allow_hmc_transition: bool = False
    target_signature: str = EXPECTED_PHASE19_TARGET_SIGNATURE
    adapter_signature: str = EXPECTED_PHASE19_ADAPTER_SIGNATURE
    fixed_transport_adapter_signature: str = (
        EXPECTED_PHASE19_FIXED_TRANSPORT_ADAPTER_SIGNATURE
    )

    def normalized(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.cpu_multicore_hmc_chain_harness_config.v1",
            "phase": PHASE19_ROUTE,
            "payload_path": str(self.payload_path),
            "phase18_diagnostic_path": str(self.phase18_diagnostic_path),
            "output_path": str(self.output_path),
            "seed": int(self.seed),
            "worker_count": int(self.worker_count),
            "chain_count": int(self.chain_count),
            "num_results": int(self.num_results),
            "num_burnin_steps": int(self.num_burnin_steps),
            "num_leapfrog_steps": int(self.num_leapfrog_steps),
            "step_size": float(self.step_size),
            "jit_compile": bool(self.jit_compile),
            "jit_compile_false_runtime_allowed": False,
            "require_cpu_hidden": bool(self.require_cpu_hidden),
            "allow_hmc_transition": bool(self.allow_hmc_transition),
            "execution_target": "cpu_hidden_multicore_worker_harness",
            "hmc_policy": "worker_value_score_compile_smoke_no_transition",
            "posterior_validation_policy": "not_run_deferred_to_phase20",
            "training_execution_target": "not_run",
            "gpu_sample_generation_policy": "forbidden",
            "expected_target_signature": str(self.target_signature),
            "expected_adapter_signature": str(self.adapter_signature),
            "expected_fixed_transport_adapter_signature": str(
                self.fixed_transport_adapter_signature
            ),
            "nonclaims": NEUTRA_CPU_MULTICORE_HMC_CHAIN_HARNESS_NONCLAIMS,
        }


def run_cpu_multicore_hmc_chain_harness(
    config: NeuTraCPUMulticoreHMCChainHarnessConfig | None = None,
) -> Mapping[str, Any]:
    """Run the Phase 19 CPU-hidden multicore worker harness."""

    cfg = NeuTraCPUMulticoreHMCChainHarnessConfig() if config is None else config
    start = time.monotonic()
    _validate_config(cfg)
    payload = _read_json_mapping(cfg.payload_path, "Phase 17 payload")
    phase18 = _read_json_mapping(cfg.phase18_diagnostic_path, "Phase 18 diagnostic")
    payload_hash = _file_sha256(cfg.payload_path)
    phase18_hash = _file_sha256(cfg.phase18_diagnostic_path)
    _validate_payload_and_phase18(cfg, payload=payload, phase18=phase18)
    worker_inputs = [
        {
            "worker_index": index,
            "payload_path": str(cfg.payload_path),
            "seed": int(cfg.seed) + 1009 * (index + 1),
            "chain_count": int(cfg.chain_count),
            "jit_compile": bool(cfg.jit_compile),
            "target_signature": str(cfg.target_signature),
            "adapter_signature": str(cfg.adapter_signature),
        }
        for index in range(int(cfg.worker_count))
    ]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(cfg.worker_count),
    ) as executor:
        workers = list(executor.map(_worker_compile_smoke, worker_inputs))
    workers = sorted(workers, key=lambda row: int(row["worker_index"]))
    worker_success = all(bool(row.get("passed")) for row in workers)
    worker_return_codes = [int(row.get("return_code", 1)) for row in workers]
    fixed_signatures = sorted(
        {
            str(row.get("fixed_transport_adapter_signature"))
            for row in workers
            if row.get("fixed_transport_adapter_signature")
        }
    )
    boundary_checks = {
        "cpu_hidden": os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
        "worker_count_match": len(workers) == int(cfg.worker_count),
        "worker_return_codes_zero": all(code == 0 for code in worker_return_codes),
        "worker_seeds_distinct": len({int(row["seed"]) for row in workers})
        == len(workers),
        "jit_compile_true": bool(cfg.jit_compile) is True,
        "jit_compile_false_runtime_not_executed": True,
        "training_not_executed": True,
        "gpu_sample_generation_not_executed": True,
        "posterior_validation_not_executed": True,
        "hmc_transition_not_executed": True,
        "fixed_transport_adapter_signature_match": fixed_signatures
        == [str(cfg.fixed_transport_adapter_signature)],
    }
    passed = bool(worker_success and all(boundary_checks.values()))
    artifact = {
        "schema": "bayesfilter.neutra.cpu_multicore_hmc_chain_harness_result.v1",
        "phase": PHASE19_ROUTE,
        "passed": passed,
        "decision": (
            "PASS_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS"
            if passed
            else "BLOCK_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS"
        ),
        "config": cfg.normalized(),
        "payload_path": str(cfg.payload_path),
        "payload_file_sha256": payload_hash,
        "phase18_diagnostic_path": str(cfg.phase18_diagnostic_path),
        "phase18_diagnostic_file_sha256": phase18_hash,
        "target_signature": str(cfg.target_signature),
        "adapter_signature": str(cfg.adapter_signature),
        "fixed_transport_adapter_signature": (
            fixed_signatures[0] if len(fixed_signatures) == 1 else None
        ),
        "transport_hash": (
            workers[0].get("transport_hash")
            if workers and workers[0].get("transport_hash")
            else None
        ),
        "artifact_signature": phase18.get("artifact_signature"),
        "worker_count": int(cfg.worker_count),
        "chain_count": int(cfg.chain_count),
        "num_results": int(cfg.num_results),
        "num_burnin_steps": int(cfg.num_burnin_steps),
        "num_leapfrog_steps": int(cfg.num_leapfrog_steps),
        "step_size": float(cfg.step_size),
        "workers": workers,
        "boundary_checks": boundary_checks,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "tensorflow_imported_in_parent": _tensorflow_imported(),
        "jit_compile": True,
        "jit_compile_false_runtime_executed": False,
        "training_executed": False,
        "hmc_transition_executed": False,
        "hmc_sampling_or_tuning_executed": False,
        "posterior_validation_executed": False,
        "gpu_sample_generation_executed": False,
        "elapsed_seconds": time.monotonic() - start,
        "nonclaims": NEUTRA_CPU_MULTICORE_HMC_CHAIN_HARNESS_NONCLAIMS,
    }
    artifact = {
        **artifact,
        "artifact_hash": f"sha256:{_stable_payload_sha256(artifact)}",
        "artifact_hash_semantics": "stable_json_sha256_excluding_artifact_hash_fields",
    }
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(cfg.output_path, artifact)
    return artifact


def phase19_error_payload(
    error: BaseException,
    *,
    config: NeuTraCPUMulticoreHMCChainHarnessConfig,
) -> Mapping[str, Any]:
    """Build a Phase 19 blocker payload without fallback execution."""

    return {
        "schema": "bayesfilter.neutra.cpu_multicore_hmc_chain_harness_result.v1",
        "phase": PHASE19_ROUTE,
        "passed": False,
        "decision": "BLOCK_PHASE19_CPU_MULTICORE_HMC_CHAIN_HARNESS",
        "config": config.normalized(),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "jit_compile": bool(config.jit_compile),
        "jit_compile_false_runtime_executed": False,
        "training_executed": False,
        "hmc_transition_executed": False,
        "hmc_sampling_or_tuning_executed": False,
        "posterior_validation_executed": False,
        "gpu_sample_generation_executed": False,
        "nonclaims": NEUTRA_CPU_MULTICORE_HMC_CHAIN_HARNESS_NONCLAIMS,
    }


def _worker_compile_smoke(worker_input: Mapping[str, Any]) -> Mapping[str, Any]:
    ensure_cpu_only_env()
    worker_start = time.monotonic()
    try:
        assert_cpu_only_env()
        import tensorflow as tf

        from bayesfilter.inference import (
            FixedTransportValueScoreAdapter,
            load_frozen_neutra_artifact,
            stable_frozen_neutra_artifact_signature,
        )
        from bayesfilter.ssm import (
            stable_ssm_posterior_adapter_signature,
            stable_ssm_target_signature,
        )
        from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
            make_lgssm_generic_target_fixture,
        )

        payload_path = Path(str(worker_input["payload_path"]))
        payload = _read_json_mapping(payload_path, "Phase 17 payload")
        loaded = load_frozen_neutra_artifact(
            payload,
            expected_target_signature=str(worker_input["target_signature"]),
        )
        fixture = make_lgssm_generic_target_fixture()
        target_signature = stable_ssm_target_signature(fixture.contract)
        adapter_signature = stable_ssm_posterior_adapter_signature(fixture.adapter)
        adapter = FixedTransportValueScoreAdapter(
            base_adapter=fixture.adapter,
            transport=loaded.transport,
            target_scope="lgssm-neutra-phase18-fixed-transport-hmc-mechanics",
            evidence_path=(
                "bayesfilter/testing/"
                "neutra_cpu_multicore_hmc_chain_harness_tf.py"
            ),
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=False,
        )
        capability = adapter.value_score_capability()
        chain_count = int(worker_input["chain_count"])
        dimension = int(loaded.manifest.dimension)
        seed = int(worker_input["seed"])
        base = tf.cast(tf.range(chain_count * dimension), tf.float64)
        chain_state = tf.reshape(base, (chain_count, dimension))
        chain_state = chain_state * tf.constant(0.01, dtype=tf.float64)
        chain_state = chain_state + tf.cast(seed % 17, tf.float64) * 0.001
        objective = tf.function(
            lambda z_arg: adapter.log_prob_and_grad_batch(z_arg),
            jit_compile=True,
        )
        compile_start = time.perf_counter()
        value, score = objective(chain_state)
        first_wall = time.perf_counter() - compile_start
        second_start = time.perf_counter()
        value2, score2 = objective(chain_state)
        second_wall = time.perf_counter() - second_start
        finite = bool(
            tf.reduce_all(tf.math.is_finite(value)).numpy()
            and tf.reduce_all(tf.math.is_finite(score)).numpy()
            and tf.reduce_all(tf.math.is_finite(value2)).numpy()
            and tf.reduce_all(tf.math.is_finite(score2)).numpy()
        )
        passed = bool(
            finite
            and target_signature == str(worker_input["target_signature"])
            and adapter_signature == str(worker_input["adapter_signature"])
            and capability.is_accepted_xla_hmc_authority
            and not capability.is_accepted_full_chain_xla_diagnostic_authority
        )
        return {
            "worker_index": int(worker_input["worker_index"]),
            "pid": os.getpid(),
            "return_code": 0 if passed else 1,
            "passed": passed,
            "seed": seed,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tensorflow_version": tf.__version__,
            "target_signature": target_signature,
            "adapter_signature": adapter_signature,
            "fixed_transport_adapter_signature": adapter.adapter_signature(),
            "artifact_signature": stable_frozen_neutra_artifact_signature(loaded),
            "transport_hash": loaded.manifest.transport_hash,
            "value_score_capability": {
                "value_score_authority": capability.value_score_authority,
                "xla_hmc_ready": bool(capability.xla_hmc_ready),
                "accepted_xla_hmc_authority": bool(
                    capability.is_accepted_xla_hmc_authority
                ),
                "full_chain_xla_diagnostic_ready": bool(
                    capability.full_chain_xla_diagnostic_ready
                ),
                "accepted_full_chain_xla_diagnostic_authority": bool(
                    capability.is_accepted_full_chain_xla_diagnostic_authority
                ),
                "target_scope": capability.target_scope,
                "evidence_path": capability.evidence_path,
            },
            "chain_state_shape": list(chain_state.shape.as_list()),
            "value_shape": list(value.shape.as_list()),
            "score_shape": list(score.shape.as_list()),
            "finite_checks": {
                "first_value_finite": bool(tf.reduce_all(tf.math.is_finite(value)).numpy()),
                "first_score_finite": bool(tf.reduce_all(tf.math.is_finite(score)).numpy()),
                "second_value_finite": bool(tf.reduce_all(tf.math.is_finite(value2)).numpy()),
                "second_score_finite": bool(tf.reduce_all(tf.math.is_finite(score2)).numpy()),
            },
            "first_call_wall_seconds": float(first_wall),
            "second_call_wall_seconds": float(second_wall),
            "compile_time_proxy_seconds": max(0.0, float(first_wall - second_wall)),
            "jit_compile": True,
            "jit_compile_false_runtime_executed": False,
            "training_executed": False,
            "hmc_transition_executed": False,
            "posterior_validation_executed": False,
            "gpu_sample_generation_executed": False,
            "elapsed_seconds": time.monotonic() - worker_start,
        }
    except Exception as exc:
        return {
            "worker_index": int(worker_input.get("worker_index", -1)),
            "pid": os.getpid(),
            "return_code": 1,
            "passed": False,
            "seed": int(worker_input.get("seed", -1)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "jit_compile": bool(worker_input.get("jit_compile", True)),
            "jit_compile_false_runtime_executed": False,
            "training_executed": False,
            "hmc_transition_executed": False,
            "posterior_validation_executed": False,
            "gpu_sample_generation_executed": False,
            "elapsed_seconds": time.monotonic() - worker_start,
        }


def _validate_config(config: NeuTraCPUMulticoreHMCChainHarnessConfig) -> None:
    if bool(config.require_cpu_hidden):
        try:
            assert_cpu_only_env()
        except RuntimeError as exc:
            raise NeuTraCPUMulticoreHMCChainHarnessError(str(exc)) from exc
    if not bool(config.jit_compile):
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "Phase 19 forbids jit_compile=false"
        )
    if bool(config.allow_hmc_transition):
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "Phase 19 forbids HMC transitions; Phase 20 owns validation"
        )
    if int(config.seed) < 0:
        raise NeuTraCPUMulticoreHMCChainHarnessError("seed must be nonnegative")
    if int(config.worker_count) < 1 or int(config.worker_count) > 8:
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "worker_count must be in 1..8 for Phase 19"
        )
    if int(config.chain_count) < 1 or int(config.chain_count) > 8:
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "chain_count must be in 1..8 for Phase 19"
        )
    if int(config.num_results) != 1:
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "Phase 19 smoke records num_results=1 but does not run HMC"
        )
    if int(config.num_burnin_steps) != 0:
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "Phase 19 smoke records num_burnin_steps=0"
        )
    if int(config.num_leapfrog_steps) <= 0:
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "num_leapfrog_steps must be positive"
        )
    if float(config.step_size) <= 0.0:
        raise NeuTraCPUMulticoreHMCChainHarnessError("step_size must be positive")


def _validate_payload_and_phase18(
    config: NeuTraCPUMulticoreHMCChainHarnessConfig,
    *,
    payload: Mapping[str, Any],
    phase18: Mapping[str, Any],
) -> None:
    if payload.get("schema") != "bayesfilter.neutra.frozen_affine_diag.v1":
        raise NeuTraCPUMulticoreHMCChainHarnessError("Phase 17 payload schema mismatch")
    if payload.get("target_signature") != str(config.target_signature):
        raise NeuTraCPUMulticoreHMCChainHarnessError("payload target signature mismatch")
    if payload.get("source_adapter_signature") != str(config.adapter_signature):
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "payload adapter signature mismatch"
        )
    if bool(payload.get("training_executed")):
        raise NeuTraCPUMulticoreHMCChainHarnessError("payload records hidden training")
    if bool(payload.get("hmc_executed")):
        raise NeuTraCPUMulticoreHMCChainHarnessError("payload records HMC execution")
    if phase18.get("decision") != "PASS_PHASE18_FIXED_TRANSPORT_HMC_MECHANICS_XLA_COMPILE":
        raise NeuTraCPUMulticoreHMCChainHarnessError("Phase 18 diagnostic did not pass")
    if phase18.get("target_signature") != str(config.target_signature):
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "Phase 18 target signature mismatch"
        )
    if phase18.get("adapter_signature") != str(config.adapter_signature):
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "Phase 18 adapter signature mismatch"
        )
    if phase18.get("fixed_transport_adapter_signature") != str(
        config.fixed_transport_adapter_signature
    ):
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "Phase 18 fixed-transport adapter signature mismatch"
        )
    if phase18.get("jit_compile") is not True:
        raise NeuTraCPUMulticoreHMCChainHarnessError("Phase 18 was not jit_compile=True")
    if phase18.get("jit_compile_false_runtime_executed") is not False:
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "Phase 18 records jit_compile=false runtime"
        )
    if phase18.get("hmc_sampling_or_tuning_executed") is not False:
        raise NeuTraCPUMulticoreHMCChainHarnessError(
            "Phase 18 records HMC sampling/tuning"
        )


def _read_json_mapping(path: Path, label: str) -> Mapping[str, Any]:
    if not path.exists():
        raise NeuTraCPUMulticoreHMCChainHarnessError(f"{label} is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise NeuTraCPUMulticoreHMCChainHarnessError(f"{label} must be a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_payload_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_hash", None)
    normalized.pop("artifact_hash_semantics", None)
    blob = json.dumps(_json_safe(normalized), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _tensorflow_imported() -> bool:
    import sys

    return "tensorflow" in sys.modules


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the Phase 19 CPU multicore harness."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-path", type=Path, default=DEFAULT_PHASE17_PAYLOAD_PATH)
    parser.add_argument(
        "--phase18-diagnostic-path",
        type=Path,
        default=DEFAULT_PHASE18_OUTPUT_PATH,
    )
    parser.add_argument("--output-path", type=Path, default=DEFAULT_PHASE19_OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--worker-count", type=int, default=2)
    parser.add_argument("--chain-count", type=int, default=2)
    parser.add_argument("--num-results", type=int, default=1)
    parser.add_argument("--num-burnin-steps", type=int, default=0)
    parser.add_argument("--num-leapfrog-steps", type=int, default=2)
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--jit-compile", choices=("true", "false"), default="true")
    args = parser.parse_args(argv)
    config = NeuTraCPUMulticoreHMCChainHarnessConfig(
        payload_path=args.payload_path,
        phase18_diagnostic_path=args.phase18_diagnostic_path,
        output_path=args.output_path,
        seed=args.seed,
        worker_count=args.worker_count,
        chain_count=args.chain_count,
        num_results=args.num_results,
        num_burnin_steps=args.num_burnin_steps,
        num_leapfrog_steps=args.num_leapfrog_steps,
        step_size=args.step_size,
        jit_compile=(args.jit_compile == "true"),
    )
    try:
        result = run_cpu_multicore_hmc_chain_harness(config)
    except Exception as exc:
        result = phase19_error_payload(exc, config=config)
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(config.output_path, result)
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "output_path": str(config.output_path),
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "passed": bool(result["passed"]),
                "output_path": str(config.output_path),
                "worker_count": result["worker_count"],
                "jit_compile": result["jit_compile"],
                "posterior_validation_executed": result[
                    "posterior_validation_executed"
                ],
                "nonclaims": NEUTRA_CPU_MULTICORE_HMC_CHAIN_HARNESS_NONCLAIMS,
            },
            sort_keys=True,
        )
    )
    return 0 if bool(result["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

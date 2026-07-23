#!/usr/bin/env python3
"""Sequential q=20 NeuTra-HMC for the two frozen (32, 32) charts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def _requested_mode() -> str | None:
    if "--mode" not in sys.argv:
        return None
    index = sys.argv.index("--mode")
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def _configure_visibility() -> str:
    mode = _requested_mode()
    if mode not in {"mechanics-smoke", "acquire"}:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        return os.environ["CUDA_VISIBLE_DEVICES"]
    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    if existing is not None:
        return existing
    probe = subprocess.run(
        ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    available = {
        int(line.strip())
        for line in probe.stdout.splitlines()
        if line.strip().isdigit()
    }
    selected = "1" if 1 in available else ("0" if 0 in available else "")
    if not selected:
        raise RuntimeError("no physical GPU 1 or GPU 0 is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


SELECTED_PHYSICAL_GPU = _configure_visibility()

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.gpu_memory_policy import (  # noqa: E402
    configure_tensorflow_gpu_memory_growth,
)


GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
    tf,
    require_gpu=_requested_mode() in {"mechanics-smoke", "acquire"},
)
tf.config.experimental.enable_tensor_float_32_execution(True)
tf.config.set_soft_device_placement(False)

from bayesfilter.inference.batched_value_score import (  # noqa: E402
    FixedTransportValueScoreAdapter,
)
from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_hmc import (  # noqa: E402
    NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
    SequentialNeuTraHMCConfig,
    run_sequential_neutra_hmc,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    complexity_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm.q20_neutra_sequential_real.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-sequential-real-run-plan-2026-07-22.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
TUNING_SUMMARY = Path(
    "docs/plans/artifacts/ssl-lstm-q20-32x32-hmc-tuning-2026-07-21/"
    "tuning-lock-20260722/summary.json"
)
CHART_INPUTS = {
    "chart-a": Path(
        "docs/plans/artifacts/ssl-lstm-q20-two-architecture-loss-gate-2026-07-21/"
        "arch-32x32/seed-a/result.json"
    ),
    "chart-b": Path(
        "docs/plans/artifacts/ssl-lstm-q20-two-architecture-loss-gate-2026-07-21/"
        "arch-32x32/seed-b/result.json"
    ),
}
INITIAL_Z = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
KERNEL_HYPOTHESIS = {
    "mass_matrix": "identity",
    "step_size": 0.5656854249492381,
    "num_leapfrog_steps": 4,
    "status": "canary_nominated_hypothesis_not_admitted_kernel",
}
ROOT_SEEDS = {
    "chart-a": (20260722, 41001),
    "chart-b": (20260722, 42001),
}
PROPOSED_CAP_SECONDS = 77700.0
MEASURED_SECONDS_PER_TRANSITION_LEAPFROG = 2.158400593840876
PROSPECTIVE_COST_MARGIN = 1.5
HOST_RAM_CAP_BYTES = 64 * 1024**3


class Q20SequentialRunError(RuntimeError):
    """Raised when a q=20 sequential-run binding fails."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Q20SequentialRunError(f"expected JSON object: {path}")
    return payload


def _repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise Q20SequentialRunError(f"{label} must remain inside the repository")
    return resolved


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise Q20SequentialRunError(f"output already exists: {path}")
    path.write_bytes(_canonical(payload))


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


class TargetBridge:
    """Grant sequential HMC authority to the exact q=20 target only."""

    supports_retained_draw_batch = False
    supports_retained_flat_batch = True
    supports_retained_value_score_status = True

    def __init__(self, target: Any, *, evidence_path: str) -> None:
        self.target = target
        self.parameter_dim = int(target.parameter_dim)
        self.parameter_names = tuple(target.parameter_names)
        self.target_scope = f"{target.target_scope}:sequential_real_hmc"
        self.evidence_path = str(evidence_path)

    def adapter_signature(self) -> str:
        return _payload_sha256(
            {
                "base_target_signature": self.target.target_signature(),
                "scope": self.target_scope,
                "status_telemetry": True,
            }
        )

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="ssl_lstm_q20_sequential_real_target_bridge",
            evidence_path=self.evidence_path,
            target_scope=self.target_scope,
            nonclaims=(
                "q=20 controlled synthetic target only",
                "finite-sample sequential screen only",
                "no posterior oracle or stationarity proof",
            ),
        )

    def log_prob_and_grad(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(values, tf.float64)
        if tensor.shape.rank == 1:
            return self.target.value_and_score(tensor)
        if tensor.shape.rank == 2:
            return self.target.batch_value_and_score(tensor)
        raise ValueError("target bridge requires rank-one or rank-two positions")

    def log_prob_and_grad_status(
        self, values: Any
    ) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, tf.Tensor]]:
        return self.target.log_prob_and_grad_status(values)

    def target_status_telemetry(self, values: Any) -> Mapping[str, tf.Tensor]:
        return self.target.target_status_telemetry(values)


def _load_input_binding(label: str) -> dict[str, Any]:
    receipt_path = _repo_path(CHART_INPUTS[label], label=f"{label} training receipt")
    receipt = _strict_json(receipt_path)
    if receipt.get("status") != "ADMITTED" or int(receipt.get("q", -1)) != 20:
        raise Q20SequentialRunError(f"{label} training receipt is not admitted q=20")
    payload_relative = receipt.get("best_frozen_payload_path")
    expected_hash = receipt.get("best_frozen_payload_sha256")
    if not isinstance(payload_relative, str) or not isinstance(expected_hash, str):
        raise Q20SequentialRunError(f"{label} frozen payload binding is missing")
    payload_path = _repo_path(Path(payload_relative), label=f"{label} frozen payload")
    if _sha256(payload_path) != expected_hash:
        raise Q20SequentialRunError(f"{label} frozen payload hash mismatch")
    return {
        "label": label,
        "training_receipt_path": receipt_path.relative_to(ROOT).as_posix(),
        "training_receipt_sha256": _sha256(receipt_path),
        "training_stream": receipt.get("stream"),
        "payload_path": payload_path.relative_to(ROOT).as_posix(),
        "payload_sha256": expected_hash,
    }


def _validate_campaign_bindings(bindings: Mapping[str, Mapping[str, Any]]) -> None:
    if set(bindings) != set(CHART_INPUTS):
        raise Q20SequentialRunError("exactly chart-a and chart-b are required")
    if len({row["payload_sha256"] for row in bindings.values()}) != 2:
        raise Q20SequentialRunError("the two chart payloads are not distinct")
    tuning_path = _repo_path(TUNING_SUMMARY, label="tuning summary")
    tuning = _strict_json(tuning_path)
    if int(tuning.get("q", -1)) != 20:
        raise Q20SequentialRunError("tuning summary q mismatch")
    tuning_bindings = tuning.get("bindings")
    if not isinstance(tuning_bindings, Mapping):
        raise Q20SequentialRunError("tuning summary bindings are missing")
    for label, row in bindings.items():
        if tuning_bindings.get(label, {}).get("payload_sha256") != row["payload_sha256"]:
            raise Q20SequentialRunError(f"{label} tuning/payload binding mismatch")


def _build_adapter(binding: Mapping[str, Any]) -> tuple[Any, Mapping[str, Any]]:
    payload_path = _repo_path(Path(str(binding["payload_path"])), label="frozen payload")
    target = complexity_posterior_target(20, jit_compile=True)
    payload = _strict_json(payload_path)
    artifact = load_frozen_neutra_artifact(
        payload,
        expected_target_signature=target.target_signature(),
    )
    bridge = TargetBridge(target, evidence_path=PLAN.as_posix())
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=artifact.transport,
        target_scope=f"{bridge.target_scope}:{binding['label']}",
        runtime_backend="ssl_lstm_q20_sequential_fixed_transport_hmc",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "fixed independently trained (32,32) dense-IAF chart",
            "identity-mass canary-nominated kernel hypothesis",
            "finite-sample sequential screen only",
        ),
    )
    if not adapter.supports_retained_value_score_status:
        raise Q20SequentialRunError("transformed adapter lacks target-status telemetry")
    if not adapter.value_score_capability().is_accepted_full_chain_xla_diagnostic_authority:
        raise Q20SequentialRunError("transformed adapter lacks full-chain XLA authority")
    return adapter, {
        **dict(binding),
        "target_signature": target.target_signature(),
        "artifact_signature": artifact.artifact_signature,
        "transport_hash": artifact.manifest.transport_hash,
        "topology_hash": artifact.manifest.topology_hash,
        "tensor_hash": artifact.manifest.tensor_hash,
        "adapter_signature": adapter.adapter_signature(),
        "target_scope": adapter.target_scope,
    }


class CampaignBudget:
    def __init__(self, cap_seconds: float) -> None:
        self.cap_seconds = float(cap_seconds)
        self.started = time.perf_counter()
        self.refusals: list[Mapping[str, Any]] = []

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def allow(self, transition_leapfrogs: int) -> bool:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if rss > HOST_RAM_CAP_BYTES:
            raise Q20SequentialRunError("host RSS exceeded the 64 GiB campaign cap")
        reserve = (
            int(transition_leapfrogs)
            * MEASURED_SECONDS_PER_TRANSITION_LEAPFROG
            * PROSPECTIVE_COST_MARGIN
        )
        allowed = self.elapsed + reserve <= self.cap_seconds
        if not allowed:
            self.refusals.append(
                {
                    "elapsed_seconds": self.elapsed,
                    "requested_transition_leapfrogs": int(transition_leapfrogs),
                    "prospective_reserve_seconds": reserve,
                    "cap_seconds": self.cap_seconds,
                }
            )
        return allowed


def _source_bindings() -> Mapping[str, Any]:
    paths = {
        "launcher": SCRIPT,
        "plan": PLAN,
        "controller": Path("bayesfilter/inference/neutra_hmc.py"),
        "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        "derivatives": Path("bayesfilter/nonlinear/svd_sigma_point_derivatives_tf.py"),
        "adapter": Path("bayesfilter/inference/batched_value_score.py"),
        "artifact_loader": Path("bayesfilter/inference/neutra_artifacts.py"),
        "diagnostics": Path("bayesfilter/inference/hmc_posterior_diagnostics.py"),
        "tuning_summary": TUNING_SUMMARY,
    }
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "paths": {key: path.as_posix() for key, path in paths.items()},
        "sha256": {key: _sha256(ROOT / path) for key, path in paths.items()},
    }


def _config(label: str, *, mechanics_smoke: bool) -> SequentialNeuTraHMCConfig:
    common = {
        "step_size": KERNEL_HYPOTHESIS["step_size"],
        "num_leapfrog_steps": KERNEL_HYPOTHESIS["num_leapfrog_steps"],
        "seed": ROOT_SEEDS[label],
        "chain_count": 4,
        "use_xla": True,
        "target_status_required": True,
    }
    if not mechanics_smoke:
        return SequentialNeuTraHMCConfig(**common)
    return SequentialNeuTraHMCConfig(
        **common,
        warmup_chunk_size=4,
        warmup_min_results=4,
        warmup_window_results=4,
        warmup_max_results=4,
        retained_chunk_size=4,
        retained_min_results=4,
        retained_max_results=4,
        bulk_ess_min=1.0,
        tail_ess_min=1.0,
    )


def contract_payload() -> Mapping[str, Any]:
    config = _config("chart-a", mechanics_smoke=False)
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "status": "PASSED",
        "q": 20,
        "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        "charts": list(CHART_INPUTS),
        "kernel_hypothesis": KERNEL_HYPOTHESIS,
        "initial_z": [list(row) for row in INITIAL_Z],
        "sequential_policy": config.payload(),
        "diagnostic_coordinate_systems": ["hmc_coordinates", "model_parameters"],
        "energy_error_identity": "delta_h_equals_negative_log_accept_ratio",
        "native_divergence_status": "not_exposed_by_kernel",
        "proposed_cap_seconds": PROPOSED_CAP_SECONDS,
        "material_execution_authorized": False,
        "gpu_memory_policy": GPU_MEMORY_POLICY,
        "source_bindings": _source_bindings(),
        "nonclaims": [
            "contract/import smoke only",
            "no target evaluation or HMC execution",
            "canary kernel is a hypothesis, not an admitted tuning result",
            "native divergence unavailability is not zero divergences",
        ],
    }


def _allocator_memory() -> Mapping[str, Any]:
    try:
        return _json_safe(tf.config.experimental.get_memory_info("GPU:0"))
    except (RuntimeError, ValueError):
        return {"status": "unavailable"}


def _run_manifest(
    *,
    args: argparse.Namespace,
    budget: CampaignBudget,
    started_utc: str,
    bindings: Mapping[str, Any],
    chart_results: Mapping[str, Any],
) -> Mapping[str, Any]:
    return {
        "schema": "bayesfilter.serious_run_manifest.v1",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "command": shlex.join([sys.executable, *sys.argv]),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": budget.elapsed,
        "cap_seconds": budget.cap_seconds,
        "cpu_gpu_status": {
            "selected_physical_gpu": SELECTED_PHYSICAL_GPU,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_gpus": [item.name for item in tf.config.list_logical_devices("GPU")],
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "xla_jit": True,
            "tensorflow": tf.__version__,
            "gpu_allocator_bytes": _allocator_memory(),
            "gpu_memory_policy": GPU_MEMORY_POLICY,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "data_version": {
            label: {
                "target_signature": row.get("target_signature"),
                "payload_sha256": row["payload_sha256"],
            }
            for label, row in bindings.items()
        },
        "random_seeds": {label: list(seed) for label, seed in ROOT_SEEDS.items()},
        "host_ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
        "output_artifacts": {
            label: row.get("archive", {}) for label, row in chart_results.items()
        },
        "plan_file": PLAN.as_posix(),
        "result_file": (args.output_root / "summary.json").as_posix(),
        "source_bindings": _source_bindings(),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = _repo_path(args.output_root, label="output root")
    if output.exists() and any(output.iterdir()):
        raise Q20SequentialRunError("output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    budget = CampaignBudget(args.cap_seconds)
    bindings = {label: _load_input_binding(label) for label in CHART_INPUTS}
    _validate_campaign_bindings(bindings)
    preflight = {
        "schema": SCHEMA,
        "mode": args.mode,
        "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        "kernel_hypothesis": KERNEL_HYPOTHESIS,
        "bindings": bindings,
        "gpu_memory_policy": GPU_MEMORY_POLICY,
        "cap_seconds": budget.cap_seconds,
        "started_at_utc": started_utc,
    }
    _write_json(output / "preflight.json", preflight)
    chart_results: dict[str, Any] = {}
    enriched_bindings: dict[str, Any] = {}
    chart_errors: dict[str, str] = {}
    labels: Sequence[str] = ("chart-a",) if args.mode == "mechanics-smoke" else tuple(CHART_INPUTS)
    for label in labels:
        if budget.refusals:
            break
        try:
            adapter, enriched = _build_adapter(bindings[label])
            enriched_bindings[label] = enriched
            result = run_sequential_neutra_hmc(
                adapter,
                tf.constant(INITIAL_Z, tf.float64),
                _config(label, mechanics_smoke=args.mode == "mechanics-smoke"),
                archive_root=output / "charts" / label,
                archive_label=label,
                budget_check=budget.allow,
            )
            chart_results[label] = result.payload()
            del result, adapter
            gc.collect()
        except Exception as exc:  # Preserve a terminal campaign record.
            chart_errors[label] = f"{type(exc).__name__}: {exc}"
            if "campaign cap" in str(exc).lower() or "host rss" in str(exc).lower():
                break
    merged_bindings = {**bindings, **enriched_bindings}
    statuses = {
        label: row.get("stop_reason") for label, row in chart_results.items()
    }
    if args.mode == "mechanics-smoke":
        status = (
            "MECHANICS_SMOKE_COMPLETED"
            if chart_results and not chart_errors
            else "MECHANICS_SMOKE_FAILED"
        )
    elif chart_errors:
        status = "CAMPAIGN_EXECUTION_ERROR"
    elif budget.refusals or any(
        "campaign_resource_cap" in row.get("diagnostics", {}).get("hard_vetoes", [])
        for row in chart_results.values()
    ):
        status = "CAMPAIGN_RESOURCE_STOP"
    elif len(chart_results) == 2 and all(row.get("passed") for row in chart_results.values()):
        status = "BOTH_CHARTS_PASSED_SEQUENTIAL_SCREEN"
    else:
        status = "ONE_OR_MORE_CHARTS_FAILED_SEQUENTIAL_SCREEN"
    manifest = _run_manifest(
        args=args,
        budget=budget,
        started_utc=started_utc,
        bindings=merged_bindings,
        chart_results=chart_results,
    )
    summary = {
        "schema": SCHEMA,
        "mode": args.mode,
        "status": status,
        "policy_id": NEUTRA_SEQUENTIAL_HMC_POLICY_ID,
        "q": 20,
        "kernel_hypothesis": KERNEL_HYPOTHESIS,
        "bindings": merged_bindings,
        "chart_results": chart_results,
        "chart_stop_reasons": statuses,
        "chart_errors": chart_errors,
        "budget_refusals": budget.refusals,
        "run_manifest": manifest,
        "inference_status": {
            "hard_veto_screen": {
                label: row.get("diagnostics", {}).get("hard_vetoes", [])
                for label, row in chart_results.items()
            },
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": [
                "acceptance probability",
                "runtime",
                "continuous R-hat and ESS before threshold passage",
            ],
            "default_readiness": "not_assessed",
            "next_evidence_needed": (
                "downstream predictive validation after both charts pass"
            ),
        },
        "nonclaims": [
            "no posterior oracle, stationarity proof, or model adequacy claim",
            "no chart, transport, or architecture ranking",
            "acceptance is explanatory only",
            "native divergence unavailability is not zero divergences",
            "a failed chart rejects that fixed candidate, not the NeuTra direction",
            "mechanics-smoke output is never posterior evidence",
        ],
    }
    _write_json(output / "run-manifest.json", manifest)
    _write_json(output / "summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("contract-smoke", "mechanics-smoke", "acquire"),
        required=True,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "docs/plans/artifacts/ssl-lstm-q20-neutra-sequential-real-2026-07-22/"
            "unlaunched"
        ),
    )
    parser.add_argument("--cap-seconds", type=float, default=PROPOSED_CAP_SECONDS)
    parser.add_argument("--authorize-material-run", action="store_true")
    args = parser.parse_args(argv)
    if args.mode != "contract-smoke" and not args.authorize_material_run:
        parser.error("material modes require --authorize-material-run")
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0:
        parser.error("--cap-seconds must be positive and finite")
    if args.mode == "acquire" and args.cap_seconds > PROPOSED_CAP_SECONDS:
        parser.error("--cap-seconds exceeds the reviewed proposed campaign cap")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = contract_payload() if args.mode == "contract-smoke" else run(args)
    print(json.dumps(_json_safe(payload), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the trusted GPU/XLA synthetic canary for the multi-model P1 harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
P0_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p0/attempt-04-20260715T1658"
)
P0_REGISTRY_SHA256 = (
    "eba02073b8b2f4a2b648128ace5163356cf5971a5c66724bd82048e97d522a3d"
)
PLAN_PATH = Path(
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p1-shared-harness-subplan-2026-07-15.md"
)
NONCLAIMS = (
    "synthetic Gaussian P1 harness canary only",
    "training loss and HMC acceptance are explanatory only",
    "no declared model/filter cell executed",
    "no HMC convergence, posterior recovery, or transport-quality claim",
    "no production, default, or scientific readiness claim",
)


def run_canary(output_root: Path) -> Mapping[str, Any]:
    """Execute one fresh bounded canary and return its terminal result."""

    if output_root.exists():
        raise FileExistsError(f"P1 canary output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    import tensorflow_probability as tfp

    from bayesfilter.inference.neutra_campaign import (
        CampaignCellLedger,
        admit_independent_posterior_recomposition,
        campaign_fixed_transport_adapter,
        issue_typed_neutra_target_identity,
        load_campaign_neutra_transport,
        load_validated_p0_registry,
        train_campaign_neutra,
    )
    from bayesfilter.inference.neutra_hmc import BatchedHMCConfig, run_batched_hmc
    from bayesfilter.inference.neutra_training import PlainDenseIAFTrainingConfig
    from bayesfilter.testing.multimodel_neutra_p1_canary_tf import (
        SYNTHETIC_CANARY_SCOPE,
        SyntheticGaussianCampaignAdapter,
        synthetic_exponential_chart_jacobian_value_score,
        synthetic_gaussian_likelihood_value_score,
        synthetic_gaussian_prior_value_score,
    )

    registry = load_validated_p0_registry(
        P0_ROOT / "target_registry.json",
        expected_file_sha256=P0_REGISTRY_SHA256,
    )
    ledger = CampaignCellLedger(registry)
    adapter = SyntheticGaussianCampaignAdapter()
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=tf.constant(
            [[-0.75, 0.50], [0.0, 0.0], [0.60, -0.40]], tf.float64
        ),
        prior_value_score_fn=synthetic_gaussian_prior_value_score,
        likelihood_value_score_fn=synthetic_gaussian_likelihood_value_score,
        jacobian_value_score_fn=synthetic_exponential_chart_jacobian_value_score,
    )
    identity = issue_typed_neutra_target_identity(
        program_id=PROGRAM_ID,
        scope_kind="synthetic_canary",
        scope_id=SYNTHETIC_CANARY_SCOPE,
        adapter=adapter,
        recomposition=recomposition,
    )

    probe = tf.function(
        adapter.neutra_batch_log_prob_and_grad_status,
        input_signature=[tf.TensorSpec([None, 2], tf.float64)],
        jit_compile=True,
        reduce_retracing=True,
    )
    probe_input = tf.constant(
        [[-0.4, 0.2], [-0.2, -0.1], [0.2, 0.1], [0.4, -0.2]], tf.float64
    )
    compile_started = time.monotonic()
    probe_first = probe(probe_input)
    compile_and_first_seconds = time.monotonic() - compile_started
    warm_started = time.monotonic()
    probe_second = probe(tf.reverse(probe_input, axis=(0,)))
    warm_seconds = time.monotonic() - warm_started
    del probe_second
    tf.debugging.assert_all_finite(probe_first[0], "canary XLA values must be finite")
    tf.debugging.assert_all_finite(probe_first[1], "canary XLA scores must be finite")

    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if not logical_gpus:
        raise RuntimeError("trusted canary requires a logical TensorFlow GPU")
    memory_device = logical_gpus[0].name
    try:
        tf.config.experimental.reset_memory_stats(memory_device)
    except (ValueError, RuntimeError):
        pass

    training_dir = output_root / "training"
    training_config = PlainDenseIAFTrainingConfig(
        target_signature=identity.target_signature,
        dimension=2,
        affine_center=(0.0, 0.0),
        affine_factor=((1.0, 0.0), (0.0, 1.0)),
        output_dir=training_dir,
        seed=(20260715, 501),
        hidden_layers=(8,),
        stage_count=1,
        steps=64,
        batch_size=64,
        learning_rate=1.0e-3,
        checkpoint_every=64,
        heartbeat_every=8,
        device="/GPU:0",
        require_gpu=True,
    )
    training_started = time.monotonic()
    trained = train_campaign_neutra(
        identity=identity,
        adapter=adapter,
        config=training_config,
        freeze_transport_id="p1-synthetic-gaussian-dense-iaf",
        gpu_memory_policy=memory_policy,
    )
    training_seconds = time.monotonic() - training_started
    frozen_payload = json.loads(
        trained.frozen_payload_path.read_text(encoding="utf-8")
    )
    loaded = load_campaign_neutra_transport(
        identity=identity,
        adapter=adapter,
        payload=frozen_payload,
    )
    transformed = campaign_fixed_transport_adapter(
        identity=identity,
        adapter=adapter,
        loaded_artifact=loaded,
    )
    hmc_started = time.monotonic()
    hmc = run_batched_hmc(
        adapter=transformed,
        initial_state=tf.constant(
            [[-0.3, 0.2], [-0.1, -0.2], [0.1, 0.2], [0.3, -0.2]], tf.float64
        ),
        config=BatchedHMCConfig(
            num_results=16,
            num_burnin_steps=8,
            step_size=0.20,
            num_leapfrog_steps=2,
            seed=(20260715, 502),
        ),
    )
    hmc_seconds = time.monotonic() - hmc_started

    memory_info: Mapping[str, Any]
    try:
        memory_info = dict(tf.config.experimental.get_memory_info(memory_device))
    except (ValueError, RuntimeError) as exc:
        memory_info = {"unavailable": f"{type(exc).__name__}: {exc}"}
    blocked_states = ledger.payload()["states"]
    training_runtime = dict(trained.runtime_metadata)
    training_devices = tuple(training_runtime["trainable_variable_devices"])
    hmc_health = bool(hmc["diagnostics"]["health_passed"])
    checks = {
        "registry_has_eleven_cells": len(blocked_states) == 11,
        "all_model_cells_remain_target_blocked": set(blocked_states.values())
        == {"TARGET_BLOCKED"},
        "model_target_signatures_issued": 0,
        "recomposition_passed": recomposition.passed,
        "typed_identity_is_synthetic_only": identity.scope_kind
        == "synthetic_canary",
        "typed_identity_binds_status_telemetry": identity.status_execution_surface[
            "method_name"
        ]
        == "target_status_telemetry",
        "batch_xla_probe_finite": bool(
            tf.reduce_all(tf.math.is_finite(probe_first[0])).numpy()
            and tf.reduce_all(tf.math.is_finite(probe_first[1])).numpy()
        ),
        "training_jit_compile": training_runtime["jit_compile"] is True,
        "training_single_compiled_program": training_runtime[
            "compiled_training_program_invocations"
        ]
        == 1,
        "training_uses_tf_while_loop": training_runtime[
            "compiled_training_control_flow"
        ]
        == "tf_while_loop",
        "training_variables_on_gpu": bool(training_devices)
        and all("GPU" in device.upper() for device in training_devices),
        "frozen_transport_target_matches": loaded.manifest.target_signature
        == identity.target_signature,
        "transformed_hmc_health_passed": hmc_health,
        "memory_growth_verified": memory_policy[
            "all_physical_devices_memory_growth"
        ]
        is True,
        "xla_requested": True,
    }
    passed = all(
        bool(value)
        for key, value in checks.items()
        if key != "model_target_signatures_issued"
    ) and checks["model_target_signatures_issued"] == 0
    terminal = {
        "schema": "bayesfilter.multimodel_neutra_p1_canary_result.v1",
        "program_id": PROGRAM_ID,
        "phase": "P1",
        "scope": SYNTHETIC_CANARY_SCOPE,
        "passed": bool(passed),
        "decision": (
            "PASS_P1_SYNTHETIC_GPU_XLA_CANARY"
            if passed
            else "FAIL_P1_SYNTHETIC_GPU_XLA_CANARY"
        ),
        "checks": checks,
        "target_identity": identity.payload(),
        "recomposition": recomposition.payload(),
        "model_cell_states": blocked_states,
        "training": {
            "config": training_config.payload(),
            "completed_steps": trained.completed_steps,
            "state_hash": trained.state_hash,
            "frozen_transport_path": str(trained.frozen_payload_path),
            "frozen_transport_signature": loaded.artifact_signature,
            "runtime_metadata": training_runtime,
            "elapsed_seconds": training_seconds,
            "initial_loss": trained.records[0]["loss"],
            "final_loss": trained.records[-1]["loss"],
            "loss_role": "explanatory_only",
        },
        "hmc_health_smoke": {
            "config": hmc["config"],
            "diagnostics": hmc["diagnostics"],
            "elapsed_seconds": hmc_seconds,
            "diagnostic_role": "engineering_health_smoke_only",
        },
        "timing": {
            "batch_target_compile_and_first_run_seconds": compile_and_first_seconds,
            "batch_target_warm_run_seconds": warm_seconds,
            "training_compile_plus_run_seconds": training_seconds,
            "transformed_hmc_compile_plus_run_seconds": hmc_seconds,
            "total_seconds": time.monotonic() - started,
        },
        "gpu_memory": {
            "logical_device": memory_device,
            "allocator_bytes": memory_info,
            "memory_growth_policy": memory_policy,
        },
        "nonclaims": NONCLAIMS,
    }
    _write_new_json(output_root / "result.json", terminal)
    _write_new_json(output_root / "target_identity.json", identity.payload())
    _write_new_json(output_root / "recomposition.json", recomposition.payload())
    _write_new_json(output_root / "cell_ledger.json", ledger.payload())
    _write_new_json(
        output_root / "run_manifest.json",
        _run_manifest(
            output_root=output_root,
            started_at=started_at,
            tensorflow_version=tf.__version__,
            tfp_version=tfp.__version__,
            memory_policy=memory_policy,
            physical_gpus=tuple(str(item) for item in tf.config.list_physical_devices("GPU")),
            logical_gpus=tuple(str(item) for item in logical_gpus),
            target_signature=identity.target_signature,
            total_seconds=time.monotonic() - started,
        ),
    )
    artifact_hashes = {
        str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_new_json(
        output_root / "artifact_hashes.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p1_artifact_hashes.v1",
            "program_id": PROGRAM_ID,
            "artifacts": artifact_hashes,
        },
    )
    return terminal


def _run_manifest(
    *,
    output_root: Path,
    started_at: datetime,
    tensorflow_version: str,
    tfp_version: str,
    memory_policy: Mapping[str, Any],
    physical_gpus: Sequence[str],
    logical_gpus: Sequence[str],
    target_signature: str,
    total_seconds: float,
) -> Mapping[str, Any]:
    git_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty_count = len(
        subprocess.run(
            ("git", "status", "--porcelain"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    return {
        "schema": "bayesfilter.multimodel_neutra_p1_run_manifest.v1",
        "program_id": PROGRAM_ID,
        "phase": "P1",
        "git_commit": git_commit,
        "dirty_worktree_entry_count": dirty_count,
        "dirty_worktree_disclosure": (
            "shared dirty worktree with concurrent lanes; P1 used scoped paths only"
        ),
        "command": (
            "TF_FORCE_GPU_ALLOW_GROWTH=true python "
            "docs/benchmarks/run_multimodel_neutra_p1_canary.py "
            f"--output-root {output_root}"
        ),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_executable": os.sys.executable,
        "tensorflow_version": tensorflow_version,
        "tensorflow_probability_version": tfp_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "tf_force_gpu_allow_growth": os.environ.get(
            "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
        ),
        "physical_gpus": physical_gpus,
        "logical_gpus": logical_gpus,
        "gpu_memory_policy": memory_policy,
        "jit_compile": True,
        "tf32_execution_enabled": True,
        "dtype": "float64",
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "seeds": {
            "training": (20260715, 501),
            "transformed_hmc": (20260715, 502),
        },
        "target_signature": target_signature,
        "data_version": "p1-synthetic-gaussian-observation-v1",
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_seconds": float(total_seconds),
        "output_root": str(output_root),
        "plan_file": str(PLAN_PATH),
        "result_file": str(output_root / "result.json"),
        "nonclaims": NONCLAIMS,
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_canary(args.output_root)
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "decision": result["decision"],
                "target_signature": result["target_identity"]["target_signature"],
                "output_root": str(args.output_root),
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

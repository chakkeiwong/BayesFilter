#!/usr/bin/env python3
"""Check that a strict tuning handoff cannot be consumed by the factor route."""

from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    started = time.perf_counter()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    import tensorflow as tf

    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FixedTransportHMCCandidateResult,
        FixedTransportHMCKernelTuningConfig,
        FixedTransportHMCKernelTuningResult,
        build_verified_fixed_transport_hmc_handoff_from_tuning_result,
    )
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        AffineDiagonalTransport,
    )

    strict = make_q20_tempered_bridge(
        20, jit_compile=False, principal_sqrt_backend="tensorflow_eigh_strict"
    )
    factor = make_q20_tempered_bridge(
        20,
        jit_compile=False,
        principal_sqrt_backend="tensorflow_eigh_strict_factor_cached",
    )
    strict_adapter = strict.fixed_beta_adapter(1.0)
    factor_adapter = factor.fixed_beta_adapter(1.0)
    if strict_adapter.adapter_signature() == factor_adapter.adapter_signature():
        raise AssertionError("strict and factor adapter identities unexpectedly collide")

    config = FixedTransportHMCKernelTuningConfig(
        initial_step_size=1.0e-3,
        step_size_candidates=(1.0e-3, 2.0e-3),
        leapfrog_grid=(2, 3),
        selection_num_results=4,
        selection_num_burnin_steps=1,
        chain_execution_mode="eager",
        use_xla=False,
        target_scope=factor_adapter.target_scope,
    )
    candidate = FixedTransportHMCCandidateResult(
        candidate_index=0,
        num_leapfrog_steps=2,
        ladder_result=None,
        verification_config_payload=None,
        verification_diagnostics={"acceptance_rate": 0.70},
        final_status="passed",
        diagnostic_role="identity_fixture",
        fixed_kernel_step_size=1.0e-3,
    )
    stale_strict_result = FixedTransportHMCKernelTuningResult(
        config=config,
        # This is the deliberate stale payload: it was produced for strict.
        transformed_adapter_signature=strict_adapter.adapter_signature(),
        base_adapter_signature=strict_adapter.adapter_signature(),
        fixed_transport_manifest_hash="stale-transport-hash",
        target_dimension=4,
        identity_z_mass_artifact_payload={},
        identity_z_mass_artifact_signature="stale-mass-hash",
        candidates=(candidate,),
        selected_candidate_index=0,
        final_status="passed",
        final_kernel_payload={},
        tuning_scope_payload={},
        route_record_payload={},
        coordinate_payload={},
        source_dependency_closure={},
        candidate_selection_payload={},
    )
    transport = AffineDiagonalTransport(
        tf.zeros([4], tf.float64), tf.ones([4], tf.float64), component_id="identity-fixture"
    )
    rejected = False
    error = ""
    try:
        build_verified_fixed_transport_hmc_handoff_from_tuning_result(
            tuning_result=stale_strict_result,
            base_adapter=factor_adapter,
            fixed_transport=transport,
        )
    except ValueError as exc:
        rejected = True
        error = str(exc)
    payload = {
        "schema": "bayesfilter.ssl_lstm_q20.factor_route_identity_admission.v1",
        "status": "PASS_IDENTITY_REJECTION" if rejected else "FAIL_IDENTITY_REJECTION",
        "python": sys.executable,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", ""),
        "execution_policy": {
            "device_class": "cpu_hidden_identity_fixture",
            "jit_compile": False,
            "pfor": False,
        },
        "plan_path": "docs/plans/bayesfilter-ssl-lstm-q20-factor-route-promotion-test-plan-2026-09-04.md",
        "command": list(sys.argv),
        "git": {
            "commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "status_sha256": hashlib.sha256(
                subprocess.check_output(
                    ("git", "status", "--porcelain"), cwd=ROOT, text=True
                ).encode("utf-8")
            ).hexdigest(),
        },
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "source_provenance": {
            "identity_benchmark_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "bridge_module_sha256": hashlib.sha256(
                (ROOT / "bayesfilter/inference/tempered_target_tf.py").read_bytes()
            ).hexdigest(),
            "tuning_module_sha256": hashlib.sha256(
                (ROOT / "bayesfilter/inference/fixed_transport_hmc_tuning_tf.py").read_bytes()
            ).hexdigest(),
        },
        "strict_target_signature": strict.target_signature,
        "factor_target_signature": factor.target_signature,
        "strict_component_adapter_signature": strict.component_target_adapter_signature,
        "factor_component_adapter_signature": factor.component_target_adapter_signature,
        "strict_fixed_beta_adapter_signature": strict_adapter.adapter_signature(),
        "factor_fixed_beta_adapter_signature": factor_adapter.adapter_signature(),
        "stale_payload_adapter_signature": strict_adapter.adapter_signature(),
        "factor_expected_adapter_signature": factor_adapter.adapter_signature(),
        "stale_strict_payload_rejected_by_factor": rejected,
        "rejection_error": error,
        "elapsed_seconds": time.perf_counter() - started,
        "nonclaims": [
            "no tuning-quality claim",
            "no HMC convergence claim",
            "no posterior or default-readiness claim",
        ],
    }
    output = Path(sys.argv[1]) if len(sys.argv) == 2 else None
    if output is None:
        raise SystemExit("usage: benchmark... OUTPUT.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}, sort_keys=True))
    return 0 if rejected else 1


if __name__ == "__main__":
    raise SystemExit(main())

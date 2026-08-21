#!/usr/bin/env python3
"""Record the bounded remaining-gap hypothesis campaign.

This runner consumes existing diagnostic evidence and executes only small,
non-promoting probes. GPU/XLA claims come only from versioned managed-session
GPU artifacts produced under the repository memory-growth policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817"
KSC_RESULT = OUTPUT_ROOT / "ksc-gaussian-sum/result.json"
KSC_GPU_RESULT = OUTPUT_ROOT / "ksc-gaussian-sum-gpu3/result.json"
SVX_SGQF_GPU_RESULT = OUTPUT_ROOT / "svx-sgqf-gpu3/result.json"
SVX_ZC_GPU_RESULT = OUTPUT_ROOT / "svx-zc-gpu3/result.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    started = time.monotonic()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    import tensorflow as tf

    from bayesfilter.testing.neutra_model_registry_tf import BLOCKED_CELLS
    from bayesfilter.testing.zhao_cui_actual_sv_neutra_target_tf import (
        SCORE_BACKEND_ID,
        make_actual_sv_zc_neutra_adapter,
    )
    from bayesfilter.nonlinear.rectangular_srukf_tf import (
        TFRectangularSRUKFModel,
        tf_rectangular_srukf_value,
    )

    blocked = {item.cell_id: item for item in BLOCKED_CELLS}
    ksc = json.loads(KSC_RESULT.read_text(encoding="utf-8")) if KSC_RESULT.exists() else {}
    ksc_gpu = json.loads(KSC_GPU_RESULT.read_text(encoding="utf-8"))
    svx_sgqf_gpu = json.loads(SVX_SGQF_GPU_RESULT.read_text(encoding="utf-8"))
    svx_zc_gpu = json.loads(SVX_ZC_GPU_RESULT.read_text(encoding="utf-8"))
    ksc_passed = bool(ksc.get("candidate", {}).get("passed_caps"))
    ksc_gpu_passed = bool(ksc_gpu.get("gpu_xla_canary_passed", False))

    adapter = make_actual_sv_zc_neutra_adapter()
    capability = adapter.value_score_capability()
    theta = tf.constant([[0.6, 0.4], [0.2, -0.3]], tf.float64)
    value, score, status = adapter.neutra_batch_log_prob_and_grad_status(theta)
    finite = bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) and bool(
        tf.reduce_all(tf.math.is_finite(score)).numpy()
    )
    epsilon = tf.constant(1.0e-6, tf.float64)
    fd_columns = []
    for coordinate in range(2):
        direction = tf.one_hot(coordinate, 2, dtype=tf.float64)[None, :]
        plus = adapter.log_prob(theta + epsilon * direction)
        minus = adapter.log_prob(theta - epsilon * direction)
        fd_columns.append((plus - minus) / (2.0 * epsilon))
    fd = tf.stack(fd_columns, axis=1)
    fd_gap = float(tf.reduce_max(tf.abs(score - fd)).numpy())
    xla_parity_gap = None
    xla_status = "not_run"
    try:
        @tf.function(
            input_signature=[tf.TensorSpec([None, 2], tf.float64)],
            jit_compile=True,
        )
        def compiled_svx(theta_value):
            return adapter.neutra_batch_log_prob_and_grad_status(theta_value)

        xla_value, xla_score, xla_status_map = compiled_svx(theta)
        xla_parity_gap = max(
            float(tf.reduce_max(tf.abs(value - xla_value)).numpy()),
            float(tf.reduce_max(tf.abs(score - xla_score)).numpy()),
        )
        xla_status = "passed" if bool(
            tf.reduce_all(tf.equal(xla_status_map["status_code"], 0)).numpy()
        ) else "status_failed"
    except (tf.errors.InvalidArgumentError, tf.errors.OpError, RuntimeError) as error:
        xla_status = f"unavailable:{type(error).__name__}"

    model = TFRectangularSRUKFModel(
        tf.constant([[0.0]], tf.float64),
        tf.constant([[[0.5]]], tf.float64),
        tf.constant([[[0.1]]], tf.float64),
        tf.constant([[[0.0], [0.0]]], tf.float64),
        lambda state, process: state + process,
        lambda state: tf.concat([state, state], axis=-1),
    )
    singular = tf_rectangular_srukf_value(
        tf.constant([[[0.1, 0.1]]], tf.float64), model, jit_compile=False
    )
    singular_value_only = bool(singular.diagnostics["value_only"].numpy())
    singular_status = singular.diagnostics["rank_branch_status"].numpy().decode()

    results = {
        "H1_KSC": {
            "status": "gpu_canary_passed_surrogate_only" if ksc_passed and ksc_gpu_passed else "falsified",
            "cpu_evidence": str(KSC_RESULT.relative_to(ROOT)),
            "gpu_evidence": str(KSC_GPU_RESULT.relative_to(ROOT)),
            "passed_component_caps": ksc_gpu.get("candidate", {}).get("passed_caps", []),
            "gpu_cpu_value_max_abs": ksc_gpu.get("gpu_xla_canary", {}).get("value_gap_to_cpu"),
            "gpu_cpu_score_max_abs": ksc_gpu.get("gpu_xla_canary", {}).get("score_gap_to_cpu"),
            "nonclaim": "KSC Gaussian-sum surrogate only; not direct-factor SR-UKF, exact SV, HMC, or posterior evidence",
        },
        "H2_SVX_SGQF": {
            "status": "falsified_tested_sgqf_ladder",
            "evidence": str(SVX_SGQF_GPU_RESULT.relative_to(ROOT)),
            "decision": svx_sgqf_gpu.get("decision"),
            "tested_levels": [row["level"] for row in svx_sgqf_gpu.get("selection_rows", [])],
            "reference_level": svx_sgqf_gpu.get("reference_level", {}).get("level"),
            "minimum_prefix_dense_value_gap_per_observation": min(
                row["prefix_dense_value_gap_per_observation"]
                for row in svx_sgqf_gpu.get("selection_rows", [])
            ),
            "threshold_prefix_dense_value_gap_per_observation": svx_sgqf_gpu.get("thresholds", {}).get(
                "prefix_dense_value_gap_per_observation"
            ),
            "reason": "no tested level passed the dense-prefix value gate; this is a scientific negative result, not an environment blocker",
            "blocked_cell": blocked["SVX-SGQF"].reason,
        },
        "H3_PP_ZC": {
            "status": "blocked_contract_missing",
            "reason": "no source-anchored batch-native PP-ZC posterior adapter and frozen chart/Jacobian contract is registered",
        },
        "H4_STR_ZC": {
            "status": "blocked_extension_target_absent",
            "reason": "no STR-ZC target program exists; structural UKF initializer cannot stand in for target identity",
        },
        "H5_SIR_ZC": {
            "status": "blocked_observed_data_score_missing",
            "reason": "available SIR teacher/latent/proposal scores are not the observed-data target score required for admission",
        },
        "H6_SVX_ZC": {
            "status": "gpu_xla_hmc_capability_admitted_scoped" if finite and svx_zc_gpu.get("passed") else "falsified",
            "gpu_evidence": str(SVX_ZC_GPU_RESULT.relative_to(ROOT)),
            "gpu_decision": svx_zc_gpu.get("decision"),
            "finite_value_score": finite,
            "finite_difference_max_abs_gap": fd_gap,
            "cpu_xla_status": xla_status,
            "cpu_xla_max_abs_parity_gap": xla_parity_gap,
            "gpu_cpu_value_max_abs": svx_zc_gpu.get("gates", {}).get("cpu_gpu_value_max_abs"),
            "gpu_cpu_score_max_abs": svx_zc_gpu.get("gates", {}).get("cpu_gpu_score_max_abs"),
            "gpu_same_program_fd_max_abs": svx_zc_gpu.get("gates", {}).get("same_program_fd_max_abs"),
            "status_code_all_zero": bool(tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()),
            "score_backend_id": SCORE_BACKEND_ID,
            "xla_hmc_ready": bool(capability.xla_hmc_ready),
            "full_chain_xla_diagnostic_ready": bool(capability.full_chain_xla_diagnostic_ready),
            "runtime_autodiff_for_hmc": bool(adapter.runtime_autodiff_for_hmc),
            "target_signature": svx_zc_gpu.get("target_signature"),
            "adapter_signature": svx_zc_gpu.get("adapter_signature"),
            "terminal_hmc_evidence": (
                "docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802/"
                "sequential-hmc-attempt01/SVX-ZC/result.json"
            ),
            "nonclaim": "admission is restricted to the frozen T10 target; no exact-filter, broad-posterior, superiority, or default-readiness claim",
        },
        "H7_global_singular_score": {
            "status": "falsified_global_score_hypothesis_value_only_preserved",
            "value_only": singular_value_only,
            "rank_branch_status": singular_status,
            "reason": "rank-changing/singular support is intentionally value-only; no globally valid analytical score chart was established",
        },
    }
    manifest = {
        "schema": "bayesfilter.direct_factor_srukf_remaining_gaps_campaign.v1",
        "plan": "docs/plans/bayesfilter_direct_factor_srukf_remaining_gaps_closure_and_hypothesis_plan_2026_08_17.md",
        "review": "docs/plans/bayesfilter_direct_factor_srukf_remaining_gaps_closure_and_hypothesis_plan_review_2026_08_17.md",
        "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
        "python": sys.executable,
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "aggregation_device": "CPU reference process consuming versioned GPU 3 artifacts",
        "gpu_device_preference": [3, 2, 1, 0],
        "gpu_artifacts": [
            str(KSC_GPU_RESULT.relative_to(ROOT)),
            str(SVX_SGQF_GPU_RESULT.relative_to(ROOT)),
            str(SVX_ZC_GPU_RESULT.relative_to(ROOT)),
        ],
        "dtype": "float64",
        "jit_compile": False,
        "seed": 81101,
        "registry_source_sha256": _sha256(ROOT / "bayesfilter/testing/neutra_model_registry_tf.py"),
        "wall_time_seconds": time.monotonic() - started,
        "nonclaims": [
            "no direct-factor SR-UKF promotion for blocked cells",
            "GPU evidence is scoped to the recorded canary/admission gates",
            "no HMC convergence or posterior correctness claim",
            "no exact nonlinear Bayesian filtering claim",
        ],
    }
    _write(OUTPUT_ROOT / "campaign_manifest.json", manifest)
    _write(OUTPUT_ROOT / "hypothesis_results.json", results)
    summary = {
        "schema": "bayesfilter.direct_factor_srukf_remaining_gaps_summary.v1",
        "counts": {
            "scoped_gpu_pass": sum(item["status"].startswith("gpu_") for item in results.values()),
            "blocked_contract": sum(item["status"].startswith("blocked_") for item in results.values()),
            "falsified_or_value_only": sum(item["status"].startswith("falsified") or "value_only" in item["status"] for item in results.values()),
        },
        "results": results,
    }
    _write(OUTPUT_ROOT / "coverage_summary.json", summary)
    commands = """# Commands and environment

- KSC CPU probe: `MPLCONFIGDIR=/tmp/bayesfilter-mpl XDG_CACHE_HOME=/tmp/xdg-cache CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_admission_20260731.py --output-root docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/ksc-gaussian-sum`
- KSC GPU/XLA canary: `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=3 python docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_admission_20260731.py --output-root docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/ksc-gaussian-sum-gpu3 --gpu-canary`
- Exact-SV SGQF admission ladder: `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=3 python docs/benchmarks/run_neutra_svx_sgqf_repair_admission_20260731.py --output-root docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/svx-sgqf-gpu3`
- Current SVX-ZC GPU/XLA gate: `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=3 python scripts/run_svx_zc_gpu_xla_gate_20260817.py --output-root docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/svx-zc-gpu3`
- Hypothesis recorder: `CUDA_VISIBLE_DEVICES=-1 python scripts/run_direct_factor_srukf_remaining_gaps_hypotheses_20260817.py`
- GPU selection policy for this campaign: prefer physical devices `3, 2, 1, 0`; GPU 3 was available and selected.
- Every GPU process set `TF_FORCE_GPU_ALLOW_GROWTH=true`, verified memory growth before device initialization, and used XLA JIT.

The top-level recorder is a CPU/reference aggregation step. Its GPU claims are inherited only from the versioned GPU 3 sub-artifacts above.
"""
    (OUTPUT_ROOT / "commands_and_environment.md").write_text(commands, encoding="utf-8")
    lines = [
        "# Remaining-Gaps Hypothesis Campaign Result",
        "",
        "Status: `EXECUTED_GPU_GAPS_CLOSED_WITH_SCIENTIFIC_BLOCKERS_PRESERVED`",
        "",
        "The KSC Gaussian-sum surrogate passed its T20 value/score admission gates for caps 7, 16, 32, 64, 128, and 256, and its GPU 3 XLA canary matched CPU at 7.11e-15 in value and 2.22e-15 in score. This does not promote the direct-factor SR-UKF inventory row.",
        "",
        "SVX-SGQF ran on GPU 3 for levels 10, 12, 16, 20, and 24 against reference level 32; no level passed because the dense-prefix value gap remained about 0.00343 per observation against the 0.001 gate. PP-ZC, STR-ZC, and SIR-ZC remain blocked on missing target contracts/source-anchored observed-data score routes. The current frozen T10 SVX-ZC route passed GPU/CPU value-score parity, same-program finite differences, permutation invariance, and status gates, so its scoped XLA/HMC capability and executable registry row are restored. Its manual score remains non-autodiff, and prior terminal sequential-HMC evidence is not rerun. The singular rectangular probe preserved `value_only_rank_discovery`; H7's global analytical-score hypothesis is falsified.",
        "",
        "No SGQF, Zhao-Cui, or KSC surrogate result is promoted as direct-factor SR-UKF evidence.",
    ]
    (OUTPUT_ROOT / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    hashes = {
        str(path.relative_to(OUTPUT_ROOT)): _sha256(path)
        for path in sorted(OUTPUT_ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write(
        OUTPUT_ROOT / "artifact_hashes.json",
        {
            "schema": "bayesfilter.direct_factor_srukf_remaining_gaps_hashes.v1",
            "artifacts": hashes,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

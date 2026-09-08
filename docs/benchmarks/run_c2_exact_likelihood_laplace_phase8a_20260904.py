"""Run the C2 Phase 8A exact-likelihood Laplace GPU/XLA mechanics smoke."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Mapping


_DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH = os.environ.pop(
    "TF_FORCE_GPU_ALLOW_GROWTH", None
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tensorflow as tf


DTYPE = tf.float64
PARTICLE_COUNT = 32
HORIZON = 20
FIXTURE_PATH = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
PLAN_PATH = ROOT / "docs/plans/c2-exact-likelihood-laplace-mixture-apf-phase8-20260904.md"
DRIVER_PATH = ROOT / "docs/benchmarks/run_c2_exact_likelihood_laplace_phase8a_20260904.py"
KERNEL_PATH = ROOT / "bayesfilter/highdim/exact_likelihood_laplace_apf_tf.py"
ADAPTER_PATH = ROOT / "bayesfilter/highdim/c2_exact_likelihood_laplace_adapter.py"
MODEL_PATH = ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py"
EXACT_PATH = ROOT / "bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py"
PHASE_ID = "c2_exact_likelihood_laplace_phase8a_gpu_smoke_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def _jsonable(value: object) -> object:
    if isinstance(value, tf.Tensor):
        return _jsonable(value.numpy().tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact contains a non-finite float")
        return value
    raise TypeError(f"unsupported artifact value: {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _max_abs(value: tf.Tensor) -> float:
    return float(tf.reduce_max(tf.abs(tf.convert_to_tensor(value, DTYPE))).numpy())


def _finite_difference(program, theta: tf.Tensor, index: int) -> tf.Tensor:
    step = tf.constant(1.0e-5, DTYPE)
    direction = tf.one_hot(index, 2, dtype=DTYPE)
    return (
        program.evaluate(theta + step * direction)["log_likelihood"]
        - program.evaluate(theta - step * direction)["log_likelihood"]
    ) / (2.0 * step)


def _fresh_output(value: str) -> Path:
    candidate = Path(value)
    output = (ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=False)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--jit-compile", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    if not bool(args.jit_compile):
        raise ValueError("Phase 8A GPU smoke requires XLA")
    output = _fresh_output(args.output_root)
    (output / "plan-at-launch.md").write_text(
        PLAN_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    started = time.perf_counter()

    if _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH is not None:
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = _DEFERRED_TF_FORCE_GPU_ALLOW_GROWTH
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    physical_gpus = tuple(tf.config.list_physical_devices("GPU"))
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if not logical_gpus:
        raise RuntimeError("Phase 8A requires one logical TensorFlow GPU")
    with tf.device("/GPU:0"):
        placement_probe = tf.reduce_sum(tf.ones([32], DTYPE))
    if "GPU" not in placement_probe.device.upper():
        raise RuntimeError("placement probe did not execute on the GPU")

    from bayesfilter.highdim.c2_exact_likelihood_laplace_adapter import (
        compile_c2_exact_likelihood_laplace_apf_k1,
    )
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
        C2StochasticVolatilityFrozenAPFModel,
    )
    from bayesfilter.highdim.exact_likelihood_laplace_apf_tf import (
        FixedLaplaceConfig,
    )
    from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
        prepare_frozen_proposal_apf_program,
    )

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    theta = tf.constant([fixture["gamma"], math.log(fixture["beta"])], DTYPE)
    transition = tf.constant(fixture["transition_matrix"], DTYPE)
    coupling = transition - theta[0] * tf.eye(4, dtype=DTYPE)
    model = C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=coupling, sigma=float(fixture["sigma"])
    )
    observations = tf.constant(fixture["observations"], DTYPE)
    config = FixedLaplaceConfig(
        tempering_schedule=(0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0),
        step_fractions=(1.0,) * 8,
        start_scale=0.0,
        stationarity_relative_tolerance=2.0e-10,
    )

    xla_compilation = compile_c2_exact_likelihood_laplace_apf_k1(
        model=model,
        observations=observations,
        theta_reference=theta,
        particle_count=PARTICLE_COUNT,
        seed=20260904,
        laplace_config=config,
        jit_compile=True,
    )
    reference_compilation = compile_c2_exact_likelihood_laplace_apf_k1(
        model=model,
        observations=observations,
        theta_reference=theta,
        particle_count=PARTICLE_COUNT,
        seed=20260904,
        laplace_config=config,
        jit_compile=False,
    )
    xla_program = prepare_frozen_proposal_apf_program(
        model, xla_compilation.branch
    )
    eager_result = xla_program.evaluate(theta)
    compiled_result = xla_program.compiled(jit_compile=True)(theta)
    finite_difference = tf.stack(
        [_finite_difference(xla_program, theta, index) for index in range(2)]
    )
    diagnostics = xla_compilation.proposal_diagnostics
    max_stationarity = max(
        _max_abs(row["relative_stationarity_residual"]) for row in diagnostics
    )
    minimum_precision = min(
        float(tf.reduce_min(row["minimum_precision_eigenvalue"]).numpy())
        for row in diagnostics
    )
    minimum_objective_improvement = min(
        float(
            tf.reduce_min(
                row["iteration_objective_after_step"]
                - row["iteration_objective_before_step"]
            ).numpy()
        )
        for row in diagnostics
    )
    xla_reference_state_error = _max_abs(
        xla_compilation.branch.states - reference_compilation.branch.states
    )
    xla_reference_q_error = _max_abs(
        xla_compilation.branch.transition_log_proposal_density
        - reference_compilation.branch.transition_log_proposal_density
    )
    ancestor_equal = bool(
        tf.reduce_all(
            tf.equal(
                xla_compilation.branch.ancestors,
                reference_compilation.branch.ancestors,
            )
        ).numpy()
    )
    value_parity = abs(
        float(eager_result["log_likelihood"].numpy())
        - float(compiled_result["log_likelihood"].numpy())
    )
    score_parity = _max_abs(eager_result["score"] - compiled_result["score"])
    score_fd_error = _max_abs(eager_result["score"] - finite_difference)
    output_device = str(compiled_result["log_likelihood"].device)
    gpu_output = "GPU" in output_device.upper()
    checks = {
        "all_laplace_steps_valid": all(
            bool(row["laplace_valid"].numpy()) for row in diagnostics
        ),
        "all_proposal_steps_finite": all(
            bool(row["proposal_finite"].numpy()) for row in diagnostics
        ),
        "all_exact_prefixes_finite": all(
            bool(row["exact_prefix_finite"].numpy()) for row in diagnostics
        ),
        "all_newton_steps_nondecreasing": all(
            bool(
                tf.reduce_all(
                    row["iteration_objective_after_step"]
                    >= row["iteration_objective_before_step"]
                    - 2.0e-13
                    * (1.0 + tf.abs(row["iteration_objective_before_step"]))
                ).numpy()
            )
            for row in diagnostics
        ),
        "laplace_trace_count_one": xla_compilation.manifest[
            "laplace_kernel_trace_count"
        ]
        == 1,
        "sampler_trace_count_one": xla_compilation.manifest["sampler_trace_count"]
        == 1,
        "xla_non_xla_ancestors_equal": ancestor_equal,
        "xla_non_xla_states_close": xla_reference_state_error <= 5.0e-11,
        "xla_non_xla_proposal_close": xla_reference_q_error <= 5.0e-11,
        "compiled_value_parity": value_parity <= 5.0e-11,
        "compiled_score_parity": score_parity <= 5.0e-11,
        "analytical_score_central_fd": score_fd_error <= 5.0e-7,
        "finite_exact_value_and_score": bool(eager_result["finite"].numpy()),
        "gpu_output": gpu_output,
        "memory_growth_verified": bool(
            memory_policy["all_physical_devices_memory_growth"]
        ),
    }
    status = "PASS_PHASE8A_GPU_XLA_MECHANICS" if all(checks.values()) else "FAIL_PHASE8A_GPU_XLA_MECHANICS"
    elapsed = time.perf_counter() - started
    workspace_status = _git("status", "--porcelain=v1")
    memory_info = tf.config.experimental.get_memory_info("GPU:0")
    result = {
        "schema_version": "c2_phase8a_gpu_smoke_result_v1",
        "phase": PHASE_ID,
        "status": status,
        "route_classification": ROUTE_CLASSIFICATION,
        "checks": checks,
        "particle_count": PARTICLE_COUNT,
        "horizon": HORIZON,
        "observation_seed": fixture["observation_seed"],
        "model_seed": fixture["model_seed"],
        "includes_phase7_failure_time_14": True,
        "score": eager_result["score"],
        "finite_difference_score": finite_difference,
        "score_max_abs_error": score_fd_error,
        "xla_non_xla_state_max_abs_error": xla_reference_state_error,
        "xla_non_xla_proposal_max_abs_error": xla_reference_q_error,
        "xla_compiled_value_max_abs_error": value_parity,
        "xla_compiled_score_max_abs_error": score_parity,
        "maximum_relative_stationarity_residual": max_stationarity,
        "minimum_precision_eigenvalue": minimum_precision,
        "minimum_newton_objective_improvement": minimum_objective_improvement,
        "laplace_kernel_trace_count": xla_compilation.manifest[
            "laplace_kernel_trace_count"
        ],
        "sampler_trace_count": xla_compilation.manifest["sampler_trace_count"],
        "output_device": output_device,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER", "unset"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get(
                "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
            ),
            "physical_gpus": [str(device.name) for device in physical_gpus],
            "logical_gpus": [str(device.name) for device in logical_gpus],
            "placement_probe_device": str(placement_probe.device),
            "memory_policy": memory_policy,
            "allocator_current_bytes": int(memory_info["current"]),
            "allocator_peak_bytes": int(memory_info["peak"]),
            "jit_compile": True,
        },
        "elapsed_seconds": elapsed,
        "manifest": xla_compilation.manifest,
        "nonclaims": [
            "tiny mechanics smoke only",
            "no proposal-efficiency or ESS conclusion",
            "no posterior-correctness or unbiased-likelihood conclusion",
            "no adaptive-total-gradient conclusion",
            "no default, production, or general-model conclusion",
        ],
        "workspace": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("branch", "--show-current"),
            "git_status": workspace_status,
            "git_status_sha256": hashlib.sha256(
                workspace_status.encode("utf-8")
            ).hexdigest(),
        },
        "sources": {
            str(path.relative_to(ROOT)): _sha256_file(path)
            for path in (
                PLAN_PATH,
                DRIVER_PATH,
                FIXTURE_PATH,
                KERNEL_PATH,
                ADAPTER_PATH,
                MODEL_PATH,
                EXACT_PATH,
            )
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output / "result.json", result)
    _write_json(
        output / "manifest.json",
        {
            "schema_version": "c2_phase8a_gpu_smoke_manifest_v1",
            "phase": PHASE_ID,
            "command": list(sys.argv),
            "plan_sha256": _sha256_file(PLAN_PATH),
            "result_file": str((output / "result.json").relative_to(ROOT)),
            "environment": result["environment"],
            "workspace": result["workspace"],
            "sources": result["sources"],
            "elapsed_seconds": elapsed,
        },
    )
    (output / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    (output / "result.md").write_text(
        "\n".join(
            [
                "# C2 Phase 8A GPU/XLA mechanics smoke",
                "",
                f"Status: `{status}`",
                "",
                "| Check | Result |",
                "| --- | :---: |",
                *[
                    f"| `{name}` | {value} |"
                    for name, value in sorted(checks.items())
                ],
                "",
                f"Analytical-score central-difference max error: `{score_fd_error:.6g}`.",
                f"XLA/non-XLA state max error: `{xla_reference_state_error:.6g}`.",
                f"XLA/non-XLA proposal-density max error: `{xla_reference_q_error:.6g}`.",
                f"Maximum relative stationarity residual: `{max_stationarity:.6g}`.",
                f"Minimum proposal precision eigenvalue: `{minimum_precision:.6g}`.",
                f"Minimum Newton log-objective improvement: `{minimum_objective_improvement:.6g}`.",
                f"Elapsed seconds: `{elapsed:.3f}`.",
                "",
                "This is a tiny mechanics smoke. It does not test or promote proposal efficiency.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if status != "PASS_PHASE8A_GPU_XLA_MECHANICS":
        raise RuntimeError(status)
    return output


def main() -> None:
    print(run(_parse_args()))


if __name__ == "__main__":
    main()

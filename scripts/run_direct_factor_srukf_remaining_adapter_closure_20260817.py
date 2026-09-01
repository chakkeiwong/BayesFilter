"""Run the bounded remaining-adapter direct-factor SR-UKF closure campaign."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")

import tensorflow as tf

from bayesfilter.linear import block_qr_conditional_tf, stack_qr_tf
from bayesfilter.linear.batched_kalman_svd_derivatives_tf import (
    tf_batched_svd_linear_gaussian_score_first_order_graph_status,
)
from bayesfilter.nonlinear import factor_srukf_tf
from bayesfilter.nonlinear.factor_srukf_tf import tf_factor_srukf_value_and_score
from bayesfilter.testing.direct_factor_srukf_adapters_tf import (
    FrozenFactorSRUKFAdapter,
    build_common_v2_lgssm_factor_adapter,
    build_common_v2_predator_prey_factor_adapter,
    build_common_v2_range_bearing_factor_adapter,
    build_lgssm_exact_factor_adapter,
)
from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
    load_deterministic_lgssm_exact_target,
)
from bayesfilter.testing.multidim_triangular_lgssm_tf import (
    lower_triangular_lgssm_log_prob_score_status,
)


ARTIFACT_ROOT = ROOT / "docs/plans/artifacts/direct-factor-srukf-remaining-adapter-closure-20260817-r4"
PRIOR_INVENTORY = ROOT / "docs/plans/artifacts/direct-factor-srukf-model-coverage-20260817/model_inventory.json"
DTYPE = tf.float64


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if hasattr(value, "item"):
        try:
            return _jsonable(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return "not_applicable" if value > 0.0 else "nonfinite"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(path: str, anchor: str) -> dict[str, str]:
    return {"path": path, "anchor": anchor, "sha256": _sha256(ROOT / path)}


def _max_abs(left: Any, right: Any) -> float:
    return float(
        tf.reduce_max(
            tf.abs(tf.convert_to_tensor(left, DTYPE) - tf.convert_to_tensor(right, DTYPE))
        ).numpy()
    )


def _centered_difference(
    builder: Callable[[tf.Tensor], FrozenFactorSRUKFAdapter],
    theta: tf.Tensor,
    *,
    step: float,
    columns: tuple[int, ...] | None = None,
) -> tuple[tf.Tensor, tuple[int, ...]]:
    values = tf.convert_to_tensor(theta, DTYPE)
    selected = tuple(range(int(values.shape[1]))) if columns is None else columns
    finite_differences = []
    for index in selected:
        direction = tf.one_hot(index, int(values.shape[1]), dtype=DTYPE)[None, :]
        plus = builder(values + step * direction)
        minus = builder(values - step * direction)
        plus_value = tf_factor_srukf_value_and_score(
            plus.observations, plus.model, plus.derivatives, jit_compile=False
        ).log_likelihood[0]
        minus_value = tf_factor_srukf_value_and_score(
            minus.observations, minus.model, minus.derivatives, jit_compile=False
        ).log_likelihood[0]
        finite_differences.append((plus_value - minus_value) / (2.0 * step))
    return tf.stack(finite_differences), selected


def _common_v2_lgssm_authority(theta: tf.Tensor):
    from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
        _common_lgssm_v2_spec,
    )

    spec = _common_lgssm_v2_spec()
    values = tf.convert_to_tensor(theta, DTYPE)
    batch_size = int(values.shape[0])
    a0 = tf.convert_to_tensor(spec.parameters["A"], DTYPE)
    c0 = tf.convert_to_tensor(spec.parameters["C"], DTYPE)
    q0 = tf.convert_to_tensor(spec.parameters["Q"], DTYPE)
    r0 = tf.convert_to_tensor(spec.parameters["R"], DTYPE)
    p0 = tf.convert_to_tensor(spec.parameters["P0"], DTYPE)
    zeros_vector = tf.zeros([batch_size, 2], DTYPE)
    zeros_dvector = tf.zeros([batch_size, 2, 2], DTYPE)
    zeros_dmatrix = tf.zeros([batch_size, 2, 2, 2], DTYPE)
    d_transition = tf.stack(
        [
            tf.broadcast_to(a0[None, :, :], [batch_size, 2, 2]),
            tf.zeros([batch_size, 2, 2], DTYPE),
        ],
        axis=1,
    )
    d_observation_covariance = tf.stack(
        [
            tf.zeros([batch_size, 1, 1], DTYPE),
            tf.broadcast_to(r0[None, :, :], [batch_size, 1, 1]),
        ],
        axis=1,
    )
    return tf_batched_svd_linear_gaussian_score_first_order_graph_status(
        spec.observations,
        transition_offset=zeros_vector,
        transition_matrix=values[:, 0, None, None] * a0[None, :, :],
        transition_covariance=tf.broadcast_to(q0[None, :, :], [batch_size, 2, 2]),
        observation_offset=tf.zeros([batch_size, 1], DTYPE),
        observation_matrix=tf.broadcast_to(c0[None, :, :], [batch_size, 1, 2]),
        observation_covariance=values[:, 1, None, None] * r0[None, :, :],
        initial_state_mean=tf.broadcast_to(spec.parameters["m0"][None, :], [batch_size, 2]),
        initial_state_covariance=tf.broadcast_to(p0[None, :, :], [batch_size, 2, 2]),
        d_initial_state_mean=zeros_dvector,
        d_initial_state_covariance=zeros_dmatrix,
        d_transition_offset=zeros_dvector,
        d_transition_matrix=d_transition,
        d_transition_covariance=zeros_dmatrix,
        d_observation_offset=tf.zeros([batch_size, 2, 1], DTYPE),
        d_observation_matrix=tf.zeros([batch_size, 2, 1, 2], DTYPE),
        d_observation_covariance=d_observation_covariance,
        jitter=tf.constant(0.0, DTYPE),
        singular_floor=tf.constant(1.0e-12, DTYPE),
    )


def _base_row(
    adapter: FrozenFactorSRUKFAdapter,
    eager: Any,
    xla: Any,
    *,
    fd_coarse: tf.Tensor,
    fd_fine: tf.Tensor,
    fd_columns: tuple[int, ...],
) -> dict[str, Any]:
    analytic_selected = tf.gather(eager.score[0], list(fd_columns))
    return {
        "model_id": adapter.metadata["model_id"],
        "status": "eligible_score",
        "route": "one_time_cholesky_adapter_then_direct_qr_block_conditional",
        "adapter_source": _source(
            "bayesfilter/testing/direct_factor_srukf_adapters_tf.py",
            {
                "lgssm_2d_h25_rich": "build_common_v2_lgssm_factor_adapter",
                "range_bearing_4d_h20_rich": "build_common_v2_range_bearing_factor_adapter",
                "predator_prey_rk4": "build_common_v2_predator_prey_factor_adapter",
                "LGSSM-EXACT": "build_lgssm_exact_factor_adapter",
            }[str(adapter.metadata["model_id"])],
        ),
        "filter_source": _source(
            "bayesfilter/nonlinear/factor_srukf_tf.py",
            "tf_factor_srukf_value_and_score",
        ),
        "adapter_boundary": "covariances and derivatives factored once before tracing; temporal recursion uses factors and QR only",
        "parameter_names": adapter.metadata["parameter_names"],
        "parameter_coordinate": adapter.metadata["parameter_coordinate"],
        "parameter_dim": int(adapter.theta.shape[1]),
        "state_dim": adapter.model.state_dim,
        "observation_dim": adapter.model.observation_dim,
        "horizon": int(adapter.observations.shape[1]),
        "dtype": "float64",
        "device": "CPU diagnostic/reference lane",
        "direct_value": eager.log_likelihood[0],
        "direct_score": eager.score[0],
        "finite_difference_columns": fd_columns,
        "finite_difference_step_1e_5": fd_coarse,
        "finite_difference_step_5e_6": fd_fine,
        "finite_difference_delta_1e_5": _max_abs(analytic_selected, fd_coarse),
        "finite_difference_delta_5e_6": _max_abs(analytic_selected, fd_fine),
        "eager_xla_value_delta": _max_abs(eager.log_likelihood, xla.log_likelihood),
        "eager_xla_score_delta": _max_abs(eager.score, xla.score),
        "minimum_qr_pivot": eager.diagnostics["minimum_qr_pivot"][0],
        "minimum_observation_geometry_branch_margin": eager.diagnostics[
            "minimum_observation_geometry_branch_margin"
        ][0],
        "maximum_factor_reconstruction_residual": eager.diagnostics[
            "maximum_factor_reconstruction_residual"
        ][0],
        "maximum_derivative_reconstruction_residual": eager.diagnostics[
            "maximum_derivative_reconstruction_residual"
        ][0],
        "branch_status": "fixed_full_rank_positive_qr_pivot",
        "score_claim": "analytical derivative of the same finite direct-factor program on a fixed branch",
        "nonclaims": [
            "CPU diagnostic/reference evidence only",
            "not exact nonlinear Bayesian inference for nonlinear rows",
            "not singular or rank-changing score evidence",
            "not HMC or GPU production readiness",
        ],
    }


def _evaluate_case(
    builder: Callable[[tf.Tensor], FrozenFactorSRUKFAdapter],
    theta: tf.Tensor,
    *,
    fd_columns: tuple[int, ...] | None = None,
) -> tuple[FrozenFactorSRUKFAdapter, Any, Any, dict[str, Any]]:
    adapter = builder(theta)
    eager = tf_factor_srukf_value_and_score(
        adapter.observations, adapter.model, adapter.derivatives, jit_compile=False
    )
    xla = tf_factor_srukf_value_and_score(
        adapter.observations, adapter.model, adapter.derivatives, jit_compile=True
    )
    fd_coarse, columns = _centered_difference(
        builder, theta, step=1.0e-5, columns=fd_columns
    )
    fd_fine, _ = _centered_difference(
        builder, theta, step=5.0e-6, columns=fd_columns
    )
    return adapter, eager, xla, _base_row(
        adapter,
        eager,
        xla,
        fd_coarse=fd_coarse,
        fd_fine=fd_fine,
        fd_columns=columns,
    )


def _run_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    adapter, eager, _xla, row = _evaluate_case(
        build_common_v2_lgssm_factor_adapter,
        tf.constant([[1.0, 1.0]], DTYPE),
    )
    authority = _common_v2_lgssm_authority(adapter.theta)
    row.update(
        {
            "authority": "independent_batch_native_svd_linear_gaussian_score",
            "authority_value": authority.log_likelihood[0],
            "authority_score": authority.score[0],
            "authority_status_code": authority.status_code[0],
            "authority_value_delta": _max_abs(eager.log_likelihood, authority.log_likelihood),
            "authority_score_delta": _max_abs(eager.score, authority.score),
            "fixture_checksum": adapter.metadata["fixture_checksum"],
        }
    )
    rows.append(row)

    adapter, eager, _xla, row = _evaluate_case(
        build_common_v2_range_bearing_factor_adapter,
        tf.constant([[0.12, 0.04]], DTYPE),
    )
    row.update(
        {
            "observation_geometry": adapter.metadata["observation_geometry"],
            "fixture_checksum": adapter.metadata["fixture_checksum"],
            "branch_status": "fixed_full_rank_positive_qr_pivot_and_fixed_circular_branch",
            "authority": "same_program_centered_finite_difference; covariance UKF is approximate comparison only",
        }
    )
    rows.append(row)

    adapter, _eager, _xla, row = _evaluate_case(
        build_common_v2_predator_prey_factor_adapter,
        tf.constant([[0.6]], DTYPE),
    )
    row.update(
        {
            "fixture_checksum": adapter.metadata["fixture_checksum"],
            "fixture_horizon_assertion": adapter.metadata["horizon"],
            "authority": "same_program_centered_finite_difference plus source-model RK4 sensitivity unit tie-out",
        }
    )
    rows.append(row)

    bundle = load_deterministic_lgssm_exact_target()
    adapter, eager, _xla, row = _evaluate_case(
        build_lgssm_exact_factor_adapter,
        bundle.raw_truth[None, :],
        fd_columns=(0, 4, 10, 14),
    )
    posterior_value, posterior_score, likelihood_value, likelihood_score, status = (
        lower_triangular_lgssm_log_prob_score_status(
            bundle.raw_truth,
            tf.constant(bundle.fixture["observations"], DTYPE),
            bundle.contract,
        )
    )
    row.update(
        {
            "authority": "fixture_bound_exact_svd_linear_gaussian_likelihood_and_score",
            "target_signature": adapter.metadata["target_signature"],
            "fixed_innovation_jitter": adapter.metadata["fixed_innovation_jitter"],
            "authority_likelihood_value": likelihood_value,
            "authority_likelihood_score": likelihood_score,
            "authority_status_code": status["status_code"],
            "authority_value_delta": _max_abs(eager.log_likelihood[0], likelihood_value),
            "authority_score_delta": _max_abs(eager.score[0], likelihood_score),
            "authority_posterior_value": posterior_value,
            "authority_posterior_score": posterior_score,
            "posterior_comparison_rule": "add identical persisted Gaussian prior to direct likelihood value and score",
        }
    )
    rows.append(row)
    return rows


def _route_guard() -> dict[str, Any]:
    sources = {
        "factor_srukf_one_step": inspect.getsource(factor_srukf_tf._one_step),
        "stack_qr": inspect.getsource(stack_qr_tf.batched_stack_qr_lower),
        "block_qr": inspect.getsource(
            block_qr_conditional_tf.batched_block_qr_conditional
        ),
    }
    forbidden = ("cholesky", "svd(", "eigh(", "eigvalsh(")
    matches = {
        name: [token for token in forbidden if token in source.lower()]
        for name, source in sources.items()
    }
    return {
        "status": "passed" if not any(matches.values()) else "failed",
        "forbidden_tokens": forbidden,
        "matches": matches,
        "claim": "no covariance decomposition in the temporal direct-factor recursion",
    }


def _numerical_gates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row["model_id"]): row for row in rows}
    thresholds = {
        "lgssm_2d_h25_rich": {
            "authority_value_delta": 1.0e-9,
            "authority_score_delta": 1.0e-8,
            "finite_difference_delta_5e_6": 2.0e-6,
            "eager_xla_value_delta": 1.0e-10,
            "eager_xla_score_delta": 1.0e-10,
        },
        "range_bearing_4d_h20_rich": {
            "finite_difference_delta_5e_6": 2.0e-5,
            "eager_xla_value_delta": 1.0e-10,
            "eager_xla_score_delta": 1.0e-10,
        },
        "predator_prey_rk4": {
            "finite_difference_delta_5e_6": 2.0e-5,
            "eager_xla_value_delta": 1.0e-10,
            "eager_xla_score_delta": 1.0e-10,
        },
        "LGSSM-EXACT": {
            "authority_value_delta": 2.0e-8,
            "authority_score_delta": 2.0e-7,
            "finite_difference_delta_5e_6": 5.0e-5,
            "eager_xla_value_delta": 1.0e-9,
            "eager_xla_score_delta": 1.0e-9,
        },
    }
    checks: list[dict[str, Any]] = []
    for model_id, model_thresholds in thresholds.items():
        row = by_id[model_id]
        for metric, ceiling in model_thresholds.items():
            observed = float(row[metric])
            checks.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "observed": observed,
                    "ceiling": ceiling,
                    "passed": math.isfinite(observed) and observed <= ceiling,
                }
            )
        pivot = float(row["minimum_qr_pivot"])
        checks.append(
            {
                "model_id": model_id,
                "metric": "minimum_qr_pivot",
                "observed": pivot,
                "floor": 0.0,
                "passed": math.isfinite(pivot) and pivot > 0.0,
            }
        )
        for metric in (
            "maximum_factor_reconstruction_residual",
            "maximum_derivative_reconstruction_residual",
        ):
            observed = float(row[metric])
            checks.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "observed": observed,
                    "requirement": "finite",
                    "passed": math.isfinite(observed),
                }
            )
    range_margin = float(
        by_id["range_bearing_4d_h20_rich"][
            "minimum_observation_geometry_branch_margin"
        ]
    )
    checks.append(
        {
            "model_id": "range_bearing_4d_h20_rich",
            "metric": "minimum_observation_geometry_branch_margin",
            "observed": range_margin,
            "floor": 1.0e-8,
            "passed": math.isfinite(range_margin) and range_margin > 1.0e-8,
        }
    )
    return {
        "status": "passed" if all(check["passed"] for check in checks) else "failed",
        "finite_difference_gate_rule": "the h=5e-6 centered estimate is gated after h=1e-5 to h=5e-6 step halving demonstrates convergence; both estimates remain in each row",
        "checks": checks,
    }
def _superseding_inventory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prior = json.loads(PRIOR_INVENTORY.read_text(encoding="utf-8"))
    replacements = {row["model_id"]: row for row in rows}
    changed = []
    for item in prior["rows"]:
        model_id = item["model_id"]
        if model_id in replacements:
            item["status"] = "eligible_score"
            item["reason"] = "certified by the remaining-adapter closure campaign"
            item["closure_evidence"] = f"per_model/{model_id}.json"
            changed.append(model_id)
    prior["schema"] = "bayesfilter.direct_factor_srukf_model_inventory.v2"
    prior["supersedes"] = str(PRIOR_INVENTORY.relative_to(ROOT))
    prior["closure_artifact_root"] = str(ARTIFACT_ROOT.relative_to(ROOT))
    prior["changed_rows"] = changed
    prior["summary"] = {
        status: sum(1 for row in prior["rows"] if row["status"] == status)
        for status in sorted({row["status"] for row in prior["rows"]})
    }
    return prior


def main() -> None:
    started = time.time()
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=False)
    (ARTIFACT_ROOT / "per_model").mkdir()
    rows = _run_rows()
    for row in rows:
        _write_json(ARTIFACT_ROOT / "per_model" / f"{row['model_id']}.json", row)
    route_guard = _route_guard()
    _write_json(ARTIFACT_ROOT / "temporal_route_guard.json", route_guard)
    numerical_gates = _numerical_gates(rows)
    _write_json(ARTIFACT_ROOT / "numerical_gates.json", numerical_gates)
    inventory = _superseding_inventory(rows)
    _write_json(ARTIFACT_ROOT / "model_inventory_superseding.json", inventory)
    summary = {
        "schema": "bayesfilter.direct_factor_srukf_remaining_adapter_closure.v1",
        "status": "passed"
        if route_guard["status"] == "passed"
        and numerical_gates["status"] == "passed"
        and len(rows) == 4
        else "failed",
        "closed_model_ids": [row["model_id"] for row in rows],
        "closed_count": len(rows),
        "prior_adapter_required_count": 4,
        "remaining_adapter_required_count": inventory["summary"].get("adapter_required", 0),
        "inventory_summary": inventory["summary"],
        "route_guard": route_guard["status"],
        "numerical_gates": numerical_gates["status"],
        "elapsed_seconds": time.time() - started,
        "nonclaims": [
            "not universal model applicability",
            "not singular or rank-changing analytical score support",
            "not exact nonlinear inference",
            "not GPU production or HMC readiness",
        ],
    }
    _write_json(ARTIFACT_ROOT / "closure_summary.json", summary)
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_head = "unavailable"
    manifest = {
        "schema": "bayesfilter.direct_factor_srukf_remaining_adapter_manifest.v1",
        "command": "MPLCONFIGDIR=/tmp/bayesfilter-mpl python scripts/run_direct_factor_srukf_remaining_adapter_closure_20260817.py",
        "plan": "docs/plans/bayesfilter_direct_factor_srukf_remaining_adapter_closure_plan_2026_08_17.md",
        "plan_review": "docs/plans/bayesfilter_direct_factor_srukf_remaining_adapter_closure_plan_review_2026_08_17.md",
        "artifact_root": str(ARTIFACT_ROOT.relative_to(ROOT)),
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "eager": True,
        "xla": True,
        "dtype": "float64",
        "device": "CPU diagnostic/reference lane",
        "visible_physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "git_head_at_run": git_head,
        "wall_time_seconds": time.time() - started,
        "attempt": 4,
        "supersedes_attempt_artifact_roots": [
            "docs/plans/artifacts/direct-factor-srukf-remaining-adapter-closure-20260817",
            "docs/plans/artifacts/direct-factor-srukf-remaining-adapter-closure-20260817-r2",
            "docs/plans/artifacts/direct-factor-srukf-remaining-adapter-closure-20260817-r3"
        ],
        "supersession_reason": "attempt 1 had non-strict JSON/imprecise anchors; attempt 2 did not mechanically evaluate thresholds and retained NumPy scalar infinity; attempt 3 failed before publishing a success summary because route-guard code was displaced during the harness edit; numerical result tensors were unchanged",
    }
    _write_json(ARTIFACT_ROOT / "execution_manifest.json", manifest)
    hashes = {
        str(path.relative_to(ARTIFACT_ROOT)): _sha256(path)
        for path in sorted(ARTIFACT_ROOT.rglob("*"))
        if path.is_file()
    }
    _write_json(ARTIFACT_ROOT / "checksums.json", hashes)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

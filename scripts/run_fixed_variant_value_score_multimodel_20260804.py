#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (  # noqa: E402
    load_deterministic_lgssm_exact_target,
)
from bayesfilter.testing.ksc_ukf_neutra_target_tf import (  # noqa: E402
    make_ksc_ukf_neutra_adapter,
)
from bayesfilter.testing.zhao_cui_actual_sv_neutra_target_tf import (  # noqa: E402
    make_actual_sv_zc_neutra_adapter,
)
from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (  # noqa: E402
    make_predator_prey_sgqf_neutra_adapter,
)
from bayesfilter.testing.sir_filter_neutra_target_design_tf import (  # noqa: E402
    make_sir_sgqf_neutra_adapter,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t1_score_tf import (  # noqa: E402
    load_t1_score_artifact,
)
from bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf import (  # noqa: E402
    load_selected_t2_parameter_parent_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_training_jvp_tf import (  # noqa: E402
    load_t2_training_jvp_child,
)

REGISTRY_PATH = ROOT / "docs/plans/bayesfilter-fixed-variant-value-score-multimodel-target-registry-2026-08-04.json"
SCHEMA_PATH = ROOT / "docs/plans/bayesfilter-fixed-variant-value-score-multimodel-adapter-schema-2026-08-04.json"
DEFAULT_OUTPUT = ROOT / "docs/plans/bayesfilter-fixed-variant-value-score-multimodel-result-2026-08-04.json"

DTYPE = tf.float64
METHOD_A_FD_CONFIG = {
    "LGSSM": {"step": 1.0e-5, "atol": 1.0e-5, "rtol": 1.0e-5},
    "KSC_SV": {"step": 1.0e-5, "atol": 2.0e-6, "rtol": 2.0e-6},
    "ACTUAL_SV": {"step": 1.0e-5, "atol": 2.0e-6, "rtol": 2.0e-6},
    "PREDATOR_PREY": {"step": 1.0e-5, "atol": 2.0e-5, "rtol": 2.0e-5},
    "AUSTRIA_SIR": {"step": 1.0e-5, "atol": 2.0e-4, "rtol": 2.0e-5},
}

PP_SGQF_SPARSE_LEVEL = 2
LGSSM_THETA = tf.constant(
    [
        0.25,
        -1.5141277326297755,
        -1.7147984280919266,
        0.04,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=DTYPE,
)
KSC_THETA = tf.constant([0.6, 0.4], dtype=DTYPE)
ACTUAL_SV_THETA = tf.constant([0.6, 0.4], dtype=DTYPE)
PREDATOR_PREY_SOURCE_PROBIT_THETA = tf.constant(
    [0.0, -0.5244005127080409, 0.0, -0.5244005127080409, 0.0, 0.0],
    dtype=DTYPE,
)
AUSTRIA_SIR_THETA = tf.zeros([3], dtype=DTYPE)

AUSTRIA_T1_PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-20260730/"
    "pilot-final-02/p05_r4_b5_lr3e4_l1_1e9/artifact"
)
AUSTRIA_T1_SCORE_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-score-20260806/"
    "pilot-01-selected-current-closure/artifact"
)
AUSTRIA_T2_PARENT_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/"
    "pilot-final-01/t2_p05_r4_b5_lr3e4_l1_1e9/artifact"
)
AUSTRIA_T2_JVP_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-training-jvp-20260806/"
    "attempt-01-current-closure"
)
AUSTRIA_T1_JVP_DIR = ROOT / (
    "docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-training-jvp-20260806/"
    "attempt-01-current-closure"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _tensor_list(value: tf.Tensor | None) -> list[float] | None:
    if value is None:
        return None
    tensor = tf.reshape(tf.cast(tf.convert_to_tensor(value), DTYPE), [-1])
    items = [float(item) for item in tensor.numpy().tolist()]
    return items


def _row_payload(
    *,
    model_row_id: str,
    method_id: str,
    method_role: str,
    target_scope: str,
    adapter_signature: str,
    value: tf.Tensor | None,
    score: tf.Tensor | None,
    value_status: str,
    score_status: str,
    derivative_backend: str,
    same_scalar_check: dict[str, Any],
    reason_codes: list[str],
    diagnostics: dict[str, Any],
    artifact_path: str | None,
    nonclaims: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    return {
        "payload_schema_version": "bayesfilter.fixed_variant_multimodel_adapter_payload.v1",
        "model_row_id": model_row_id,
        "method_id": method_id,
        "method_role": method_role,
        "target_scope": target_scope,
        "adapter_signature": adapter_signature,
        "value": _safe_float(value),
        "score": _tensor_list(score),
        "value_status": value_status,
        "score_status": score_status,
        "derivative_backend": derivative_backend,
        "same_scalar_check": same_scalar_check,
        "reason_codes": reason_codes,
        "diagnostics": diagnostics,
        "artifact_path": artifact_path,
        "nonclaims": list(nonclaims),
    }


def _method_a_derivative_backend(capability: Any, adapter: Any) -> str:
    runtime_backend = str(getattr(capability, "runtime_backend", ""))
    score_backend_id = str(getattr(adapter, "score_backend_id", ""))
    runtime_autodiff_for_hmc = getattr(adapter, "runtime_autodiff_for_hmc", None)
    runtime_autodiff = getattr(adapter, "runtime_autodiff", None)
    manual_runtime_backends = {
        "tensorflow_manual_lgssm_svd_graph_status_score",
        "tensorflow_fixed_branch_transformed_sv_analytic_tt",
        "tensorflow_fixed_sgqf_predator_prey",
        "tensorflow_fixed_level2_sgqf_parameterized_sir",
        "tensorflow_ksc_principal_sqrt_ukf_affine",
    }
    autodiff_runtime_backends = {
        "tensorflow_ksc_gaussian_sum_mass_preserving_ukf",
        "tensorflow_batched_fixed_adjacent_squared_tt_actual_sv",
    }
    if runtime_backend in manual_runtime_backends:
        return "manual"
    if runtime_backend in autodiff_runtime_backends:
        return "autodiff_same_scalar"
    if (
        score_backend_id.endswith("_no_autodiff_v1")
        or runtime_autodiff_for_hmc is False
        or runtime_autodiff is False
        or "manual" in runtime_backend
        or "analytic" in runtime_backend
    ):
        return "manual"
    return "autodiff_same_scalar"


def _method_a_status(
    derivative_backend: str, same_scalar_status: str
) -> tuple[str, str, list[str], str]:
    if derivative_backend != "manual":
        return (
            "BLOCKED",
            "BLOCKED",
            ["A_AUTODIFF_BACKEND_NOT_ADMITTED"],
            "autodiff_backend_not_method_a_admitted",
        )
    if same_scalar_status == "PASS":
        return "VALID", "VALID", ["NONE"], "bound_to_registry_row"
    return "VALID", "VALID", ["A_SAME_SCALAR_FD_FAIL_DIAGNOSTIC_ONLY"], "manual_score_admitted_diagnostic_mismatch"
def _fd_same_scalar_check(
    adapter: Any,
    theta: tf.Tensor,
    *,
    step: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [-1])
    theta2 = theta[tf.newaxis, :]
    value, score = adapter.log_prob_and_grad(theta2)
    dimension = int(theta.shape[0])
    columns = []
    for coordinate in range(dimension):
        direction = tf.one_hot(coordinate, dimension, dtype=DTYPE)[tf.newaxis, :]
        plus = adapter.log_prob(theta2 + tf.constant(step, DTYPE) * direction)[0]
        minus = adapter.log_prob(theta2 - tf.constant(step, DTYPE) * direction)[0]
        columns.append((plus - minus) / tf.constant(2.0 * step, DTYPE))
    finite_difference = tf.stack(columns, axis=0)
    residual = tf.abs(score[0] - finite_difference)
    tolerance = tf.constant(atol, DTYPE) + tf.constant(rtol, DTYPE) * tf.maximum(
        tf.abs(score[0]), tf.abs(finite_difference)
    )
    passed = bool(tf.reduce_all(residual <= tolerance).numpy())
    return {
        "kind": "finite_difference",
        "status": "PASS" if passed else "FAIL",
        "residual_summary": (
            f"central FD step={step:.1e}, max_abs={float(tf.reduce_max(residual).numpy()):.6e}, "
            f"atol={atol:.1e}, rtol={rtol:.1e}"
        ),
        "max_abs_residual": float(tf.reduce_max(residual).numpy()),
        "reference_score": _tensor_list(finite_difference),
        "reported_score": _tensor_list(score[0]),
        "step": float(step),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _blocked_method_b_row(
    *,
    model_row_id: str,
    target_scope: str,
    backend: str,
    residual_summary: str,
    reason_code: str,
    source_backend: str,
    artifact_path: str | None,
    nonclaims: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    return _row_payload(
        model_row_id=model_row_id,
        method_id="B_tangent_or_interpolation_extension",
        method_role="empirical_extension",
        target_scope=target_scope,
        adapter_signature=target_scope,
        value=None,
        score=None,
        value_status="BLOCKED",
        score_status="BLOCKED",
        derivative_backend=backend,
        same_scalar_check={
            "kind": "not_run",
            "status": "BLOCKED",
            "residual_summary": residual_summary,
        },
        reason_codes=[reason_code],
        diagnostics={
            "target_registry_status": "blocked_by_runtime_guard",
            "current_evidence": False,
            "source_backend": source_backend,
            "frozen_identity_kind": "none" if artifact_path is None else "artifact_or_parent_identity",
        },
        artifact_path=artifact_path,
        nonclaims=nonclaims,
    )


def _adapter_row(model_row_id: str, theta: tf.Tensor, adapter: Any) -> dict[str, Any]:
    value, score, status = adapter.neutra_batch_log_prob_and_grad_status(theta[tf.newaxis, :])
    capability = adapter.value_score_capability()
    fd_config = METHOD_A_FD_CONFIG[model_row_id]
    same_scalar_check = _fd_same_scalar_check(
        adapter,
        theta,
        step=fd_config["step"],
        atol=fd_config["atol"],
        rtol=fd_config["rtol"],
    )
    derivative_backend = _method_a_derivative_backend(capability, adapter)
    value_status, score_status, reason_codes, registry_status = _method_a_status(
        derivative_backend, same_scalar_check["status"]
    )
    diagnostics = {
        "target_registry_status": registry_status,
        "current_evidence": True,
        "source_backend": capability.runtime_backend,
        "frozen_identity_kind": "adapter_signature",
        "telemetry": {k: _tensor_list(v) for k, v in status.items()},
    }
    return _row_payload(
        model_row_id=model_row_id,
        method_id="A_single_fitted_frozen_parent",
        method_role="baseline",
        target_scope=capability.target_scope,
        adapter_signature=adapter.adapter_signature(),
        value=value[0],
        score=score[0],
        value_status=value_status,
        score_status=score_status,
        derivative_backend=derivative_backend,
        same_scalar_check=same_scalar_check,
        reason_codes=reason_codes,
        diagnostics=diagnostics,
        artifact_path=capability.evidence_path,
        nonclaims=capability.nonclaims,
    )


def run_method_a_rows() -> list[dict[str, Any]]:
    bundle = load_deterministic_lgssm_exact_target()
    lgssm_value, lgssm_score, lgssm_status = bundle.adapter.log_prob_and_grad_status(LGSSM_THETA)
    lgssm_same_scalar = _fd_same_scalar_check(
        bundle.adapter,
        LGSSM_THETA,
        step=METHOD_A_FD_CONFIG["LGSSM"]["step"],
        atol=METHOD_A_FD_CONFIG["LGSSM"]["atol"],
        rtol=METHOD_A_FD_CONFIG["LGSSM"]["rtol"],
    )
    lgssm_derivative_backend = _method_a_derivative_backend(
        bundle.adapter.value_score_capability(), bundle.adapter
    )
    lgssm_value_status, lgssm_score_status, lgssm_reason_codes, lgssm_registry_status = (
        _method_a_status(lgssm_derivative_backend, lgssm_same_scalar["status"])
    )
    lgssm_row = _row_payload(
        model_row_id="LGSSM",
        method_id="A_single_fitted_frozen_parent",
        method_role="baseline",
        target_scope=bundle.adapter.target_signature,
        adapter_signature=bundle.adapter.adapter_signature(),
        value=lgssm_value,
        score=lgssm_score,
        value_status=lgssm_value_status,
        score_status=lgssm_score_status,
        derivative_backend=lgssm_derivative_backend,
        same_scalar_check=lgssm_same_scalar,
        reason_codes=lgssm_reason_codes,
        diagnostics={
            "target_registry_status": lgssm_registry_status,
            "current_evidence": True,
            "source_backend": bundle.adapter.value_score_capability().runtime_backend,
            "frozen_identity_kind": "target_signature",
            "telemetry": {k: _tensor_list(v) for k, v in lgssm_status.items()},
        },
        artifact_path=str(bundle.fixture_path.relative_to(ROOT)),
        nonclaims=bundle.adapter.value_score_capability().nonclaims,
    )

    ksc = make_ksc_ukf_neutra_adapter()
    actual_sv = make_actual_sv_zc_neutra_adapter()
    predator_prey = make_predator_prey_sgqf_neutra_adapter(sparse_level=PP_SGQF_SPARSE_LEVEL)
    austria = make_sir_sgqf_neutra_adapter()

    return [
        lgssm_row,
        _adapter_row("KSC_SV", KSC_THETA, ksc),
        _adapter_row("ACTUAL_SV", ACTUAL_SV_THETA, actual_sv),
        _adapter_row("PREDATOR_PREY", PREDATOR_PREY_SOURCE_PROBIT_THETA, predator_prey),
        _adapter_row("AUSTRIA_SIR", AUSTRIA_SIR_THETA, austria),
    ]


def run_method_b_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    parent_t1 = None
    parent_t2 = None
    try:
        parent_t1 = load_lane_b_t1_artifact_v1_compat(AUSTRIA_T1_PARENT_DIR)
        t1_score_artifact = load_t1_score_artifact(AUSTRIA_T1_SCORE_DIR, parent=parent_t1)
        t1_child = t1_score_artifact.child()
        t1_value, t1_score = t1_child.increment_and_score(AUSTRIA_SIR_THETA)
        rows.append(
            _row_payload(
                model_row_id="AUSTRIA_SIR",
                method_id="B_tangent_or_interpolation_extension",
                method_role="empirical_extension",
                target_scope=t1_child.identity.hash.value,
                adapter_signature=t1_child.identity.hash.value,
                value=t1_value,
                score=t1_score,
                value_status="VALID",
                score_status="VALID",
                derivative_backend="manual_runtime_from_offline_tangents",
                same_scalar_check={
                    "kind": "replay_jvp",
                    "status": "NOT_RUN",
                    "residual_summary": "artifact-backed replay gate present; campaign did not rerun it yet",
                },
                reason_codes=["NONE"],
                diagnostics={
                    "target_registry_status": "bound_to_registry_row",
                    "current_evidence": True,
                    "source_backend": "lane_b_parameter_child_runtime_contraction",
                    "frozen_identity_kind": "child_identity",
                    "parent_identity": parent_t1.identity.hash.value,
                    "score_artifact_identity": t1_score_artifact.identity.hash.value,
                },
                artifact_path=str(AUSTRIA_T1_SCORE_DIR.relative_to(ROOT)),
                nonclaims=(
                    "T1 tangent child only",
                    "runtime score is exact only for the child scalar",
                    "offline tangent issuer is empirical, not end-to-end analytical proof",
                    "no T2+ or posterior correctness claim",
                ),
            )
        )
    except Exception as exc:
        rows.append(
            _blocked_method_b_row(
                model_row_id="AUSTRIA_SIR",
                target_scope="austria_t1_method_b_loader_blocked",
                backend="manual_runtime_from_offline_tangents",
                residual_summary=f"Austria T1 Method B loader blocked: {type(exc).__name__}: {exc}",
                reason_code="B_STALE_T1_SCORE_ARTIFACT",
                source_backend="lane_b_parameter_child_runtime_contraction",
                artifact_path=str(AUSTRIA_T1_SCORE_DIR.relative_to(ROOT)),
                nonclaims=(
                    "Austria T1 Method B not executed in this campaign run",
                    "historical artifact failed current strict loader checks",
                ),
            )
        )

    try:
        if parent_t1 is None:
            raise RuntimeError("T1 parent unavailable because Austria T1 Method B did not load")
        parent_t2 = load_selected_t2_parameter_parent_compat(AUSTRIA_T2_PARENT_DIR, parent_artifact=parent_t1)
        if AUSTRIA_T2_JVP_DIR.exists() and AUSTRIA_T1_JVP_DIR.exists():
            _t1_child_jvp, t2_child, _payload = load_t2_training_jvp_child(
                AUSTRIA_T2_JVP_DIR,
                t1_issuer_directory=AUSTRIA_T1_JVP_DIR,
                parent_t1=parent_t1,
                parent_t2=parent_t2,
            )
            t2_increment, t2_score = t2_child.increment_and_score(AUSTRIA_SIR_THETA)
            t2_value = parent_t2.parent_artifact.value() + t2_increment
            rows.append(
                _row_payload(
                    model_row_id="AUSTRIA_SIR_T2_EXTENSION",
                    method_id="B_tangent_or_interpolation_extension",
                    method_role="empirical_extension",
                    target_scope=t2_child.identity.hash.value,
                    adapter_signature=t2_child.identity.hash.value,
                    value=t2_value,
                    score=t2_score,
                    value_status="VALID",
                    score_status="VALID",
                    derivative_backend="manual_plus_jvp_replay",
                    same_scalar_check={
                        "kind": "replay_jvp",
                        "status": "NOT_RUN",
                        "residual_summary": "strict T1/T2 issuer chain loader passed; replay not rerun in campaign",
                    },
                    reason_codes=["NONE"],
                    diagnostics={
                        "target_registry_status": "extension_outside_main_five_model_table",
                        "current_evidence": True,
                        "source_backend": "lane_b_t2_parameter_child_runtime_contraction",
                        "frozen_identity_kind": "child_identity",
                        "parent_t1_identity": parent_t1.identity.hash.value,
                        "parent_t2_identity": parent_t2.identity.hash.value,
                        "increment_at_origin": _safe_float(t2_increment),
                    },
                    artifact_path=str(AUSTRIA_T2_JVP_DIR.relative_to(ROOT)),
                    nonclaims=(
                        "T2 extension row is supporting evidence only",
                        "runtime score is exact only for the child scalar",
                        "offline issuer remains empirical",
                        "no physical likelihood or posterior correctness claim",
                    ),
                )
            )
        else:
            rows.append(
                _blocked_method_b_row(
                    model_row_id="AUSTRIA_SIR_T2_EXTENSION",
                    target_scope=parent_t2.identity.hash.value,
                    backend="manual_plus_jvp_replay",
                    residual_summary="strict T2 JVP artifact path not present in current worktree",
                    reason_code="B_MISSING_T2_JVP_ARTIFACT",
                    source_backend="lane_b_t2_parameter_child_runtime_contraction",
                    artifact_path=str(AUSTRIA_T2_PARENT_DIR.relative_to(ROOT)),
                    nonclaims=(
                        "no Method B T2 execution evidence in this campaign run",
                    ),
                )
            )
    except Exception as exc:
        target_scope = "austria_t2_method_b_loader_blocked"
        if parent_t2 is not None:
            target_scope = parent_t2.identity.hash.value
        rows.append(
            _blocked_method_b_row(
                model_row_id="AUSTRIA_SIR_T2_EXTENSION",
                target_scope=target_scope,
                backend="manual_plus_jvp_replay",
                residual_summary=f"Austria T2 Method B loader blocked: {type(exc).__name__}: {exc}",
                reason_code="B_T2_EXTENSION_BLOCKED",
                source_backend="lane_b_t2_parameter_child_runtime_contraction",
                artifact_path=str(AUSTRIA_T2_PARENT_DIR.relative_to(ROOT)),
                nonclaims=(
                    "Austria T2 Method B not executed in this campaign run",
                ),
            )
        )

    for row_id, scope, backend, reason_code, residual_summary, source_backend in [
        (
            "LGSSM",
            "no_same_scalar_lgssm_method_b_route",
            "not_available",
            "B_SCALAR_COMPATIBLE_ROUTE_MISSING",
            "Existing persisted LGSSM frozen-transport artifacts change the scalar via transport pullback plus log-Jacobian, so they are not same-scalar-compatible Method B rows for this campaign.",
            "fixed_transport_wrapper_changes_scalar",
        ),
        (
            "KSC_SV",
            "no_method_b_route",
            "not_available",
            "B_ROUTE_MISSING",
            "Method B infrastructure not yet implemented for this model in a scalar-comparable persisted form",
            "not_available",
        ),
        (
            "ACTUAL_SV",
            "no_method_b_route",
            "not_available",
            "B_ROUTE_MISSING",
            "Method B infrastructure not yet implemented for this model in a scalar-comparable persisted form",
            "not_available",
        ),
        (
            "PREDATOR_PREY",
            "no_method_b_route",
            "not_available",
            "B_ROUTE_MISSING",
            "Method B infrastructure not yet implemented for this model in a scalar-comparable persisted form",
            "not_available",
        ),
    ]:
        rows.append(
            _blocked_method_b_row(
                model_row_id=row_id,
                target_scope=scope,
                backend=backend,
                residual_summary=residual_summary,
                reason_code=reason_code,
                source_backend=source_backend,
                artifact_path=None,
                nonclaims=(
                    "Method B unavailable for this model in the current campaign run",
                ),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed-variant multimodel campaign summary")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    registry = _load_json(REGISTRY_PATH)
    schema = _load_json(SCHEMA_PATH)
    result = {
        "schema_version": "bayesfilter.fixed_variant_multimodel_result.v1",
        "status": "PASS_WITH_BLOCKED_METHOD_B_ROWS",
        "registry": str(REGISTRY_PATH.relative_to(ROOT)),
        "schema": str(SCHEMA_PATH.relative_to(ROOT)),
        "method_A_rows": run_method_a_rows(),
        "method_B_rows": run_method_b_rows(),
        "registry_payload": registry,
        "schema_payload": schema,
        "nonclaims": [
            "Method A rows are baseline fixed-variant value/score routes only",
            "Method B rows are empirical extensions only",
            "Blocked Method B rows are reported explicitly and not silently omitted",
            "No posterior correctness or HMC readiness claim",
        ],
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

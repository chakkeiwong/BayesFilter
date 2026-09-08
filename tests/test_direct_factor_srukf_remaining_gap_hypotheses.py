"""Negative and bounded-probe tests for remaining SR-UKF gaps.

These tests intentionally verify classification boundaries.  Passing a native
SGQF or Zhao--Cui diagnostic does not make it a direct-factor SR-UKF model.
"""

from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.testing.neutra_model_registry_tf import (
    BLOCKED_CELLS,
    EXECUTABLE_CELLS,
    validate_registry,
)
from bayesfilter.testing.zhao_cui_actual_sv_neutra_target_tf import (
    SCORE_BACKEND_ID,
    make_actual_sv_zc_neutra_adapter,
)
from bayesfilter.testing.ksc_gaussian_sum_ukf_neutra_target_tf import (
    ksc_gaussian_sum_ukf_likelihood_value_score_status,
)
from bayesfilter.testing.ksc_ukf_neutra_target_tf import transformed_ksc_observations
from bayesfilter.highdim.sv_mixture_cut4 import ksc_1998_log_chi_square_mixture
from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
    generate_frozen_exact_sv_dataset_tf,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = (
    ROOT
    / "docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817"
)


def _artifact(relative_path: str) -> dict[str, object]:
    return json.loads((ARTIFACT_ROOT / relative_path).read_text(encoding="utf-8"))


def test_blocked_cells_remain_explicit_and_non_srukf() -> None:
    blocked = {item.cell_id: item for item in BLOCKED_CELLS}
    assert set(blocked) == {"SVX-SGQF", "KSC-UKF", "PP-ZC", "STR-ZC", "SIR-ZC"}
    payload = validate_registry()
    rows = {row["cell_id"]: row for row in payload["blocked"]}
    assert all(row["state"].startswith("TARGET_BLOCKED") for row in rows.values())
    assert "direct-factor" not in rows["PP-ZC"]["reason"].lower()
    assert "SVX-ZC" in {item.cell_id for item in EXECUTABLE_CELLS}


def test_ksc_gaussian_sum_probe_is_finite_but_not_direct_factor_evidence() -> None:
    _states, raw = generate_frozen_exact_sv_dataset_tf(horizon=4)
    mixture = ksc_1998_log_chi_square_mixture()
    theta = tf.constant([[0.0, 0.0], [0.2, -0.3]], tf.float64)
    value, score, status = ksc_gaussian_sum_ukf_likelihood_value_score_status(
        theta,
        transformed_observations=transformed_ksc_observations(raw),
        mixture_weights=mixture.weights,
        mixture_means=mixture.means,
        mixture_variances=mixture.variances,
        component_cap=16,
    )
    tf.debugging.assert_all_finite(value, "KSC diagnostic value")
    tf.debugging.assert_all_finite(score, "KSC diagnostic score")
    tf.debugging.assert_equal(status["status_code"], tf.zeros([2], tf.int32))
    assert status["component_cap"].shape == (2,)


def test_svx_zc_score_backend_has_scoped_external_hmc_evidence() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()
    capability = adapter.value_score_capability()
    assert adapter.score_backend_id == SCORE_BACKEND_ID
    assert capability.xla_hmc_ready is True
    assert capability.full_chain_xla_diagnostic_ready is True
    assert adapter.runtime_autodiff_for_hmc is False
    assert capability.evidence_path.endswith(
        "bayesfilter_direct_factor_srukf_remaining_gaps_closure_execution_result_2026_08_17.md"
    )


def test_gpu3_remaining_gap_artifacts_pin_the_scoped_decisions() -> None:
    ksc = _artifact("ksc-gaussian-sum-gpu3/result.json")
    sgqf = _artifact("svx-sgqf-gpu3/result.json")
    svx_zc = _artifact("svx-zc-gpu3/result.json")

    assert ksc["gpu_xla_canary_passed"] is True
    assert ksc["candidate"]["passed_caps"] == [7, 16, 32, 64, 128, 256]
    assert ksc["gpu_xla_canary"]["value_gap_to_cpu"] < 1.0e-12
    assert ksc["gpu_xla_canary"]["score_gap_to_cpu"] < 1.0e-12

    assert sgqf["decision"] == "KEEP_SVX_SGQF_TARGET_BLOCKED_NO_LEVEL_PASSED"
    assert sgqf["passed"] is False
    assert [row["level"] for row in sgqf["selection_rows"]] == [10, 12, 16, 20, 24]
    assert all(row["passed"] is False for row in sgqf["selection_rows"])
    assert min(
        row["prefix_dense_value_gap_per_observation"]
        for row in sgqf["selection_rows"]
    ) > sgqf["thresholds"]["prefix_dense_value_gap_per_observation"]

    assert svx_zc["decision"] == "PASS_CURRENT_SVX_ZC_GPU_XLA_GATE"
    assert svx_zc["passed"] is True
    assert svx_zc["target_signature"] == (
        "deccdda78028706d0987322d30b9798f0f4d8b518c6773451338e83bf14d1cab"
    )
    assert svx_zc["adapter_signature"] == (
        "a91537bf016dbc4294621e7c638fd8d9432b1d9f6d70502e22167963015a312b"
    )
    assert svx_zc["gates"]["cpu_gpu_value_max_abs"] < 1.0e-12
    assert svx_zc["gates"]["cpu_gpu_score_max_abs"] < 1.0e-12
    assert svx_zc["gates"]["same_program_fd_max_abs"] < 2.0e-6
    assert svx_zc["gpu_memory_policy"]["all_physical_devices_memory_growth"] is True


def test_svx_zc_terminal_hmc_evidence_matches_the_restored_target_scope() -> None:
    terminal_path = (
        ROOT
        / "docs/plans/artifacts/bayesfilter-svx-zc-value-validation-neutra-hmc-20260802"
        / "sequential-hmc-attempt01/SVX-ZC/result.json"
    )
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))

    assert terminal["passed"] is True
    assert terminal["decision"] == "PASS_ONE_SEED_TRUTH_TAIL"
    assert terminal["target_signature"] == (
        "deccdda78028706d0987322d30b9798f0f4d8b518c6773451338e83bf14d1cab"
    )
    assert terminal["sequential"]["decision"] == "ADMIT_SEQUENTIAL_FIXED_NEUTRA_HMC_KERNEL"
    assert terminal["sequential"]["config"]["jit_compile"] is True
    assert terminal["gpu_memory_policy"]["all_physical_devices_memory_growth"] is True


def test_rank_change_and_branch_events_are_value_only_or_score_invalid() -> None:
    from bayesfilter.nonlinear.rectangular_srukf_tf import (
        TFRectangularSRUKFModel,
        tf_rectangular_srukf_value,
    )

    model = TFRectangularSRUKFModel(
        tf.constant([[0.0]], tf.float64),
        tf.constant([[[0.5]]], tf.float64),
        tf.constant([[[0.1]]], tf.float64),
        tf.constant([[[0.0], [0.0]]], tf.float64),
        lambda state, process: state + process,
        lambda state: tf.concat([state, state], axis=-1),
    )

    result = tf_rectangular_srukf_value(
        tf.constant([[[0.1, 0.1]]], tf.float64), model, jit_compile=False
    )
    assert bool(result.diagnostics["value_only"])
    assert result.diagnostics["rank_branch_status"].numpy() == b"value_only_rank_discovery"
    assert bool(tf.reduce_all(tf.math.is_finite(result.log_likelihood)).numpy())

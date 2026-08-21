"""Baseline leaderboard: all repository models x existing filtering algorithms.

This is the pre-P1 baseline of the generic squared-TT program: it runs every
currently admitted filtering algorithm on every leaderboard model row, with
value, analytic-score-vs-FD checks, wall time, and same-target references
where affordable. The generic ZC-family squared-TT engine column is BLOCKED
pending the audited P1A/P2 sequence and will join this leaderboard later.

Rows: LGSSM (2D ladder), actual SV, KSC SV, predator-prey T in {20,40},
Austria SIR T in {20,40}, structural deterministic (Ch18b) T in {20,100}.

Program plan:
docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

import bayesfilter.highdim as highdim

DTYPE = tf.float64

NONCLAIMS = (
    "single dataset per row: cross-algorithm gaps are descriptive only",
    "same-target gaps are gate-bearing only where an EXACT_ORACLE or "
    "REFINED_NUMERICAL_REFERENCE column exists",
    "no HMC, posterior-correctness, or default-change claims",
    "wall times are eager CPU float64 measurements on this host only",
    "generic ZC-family squared-TT column is BLOCKED pending audit-mandated "
    "P1A/P2 artifacts and is absent by design",
)


def _fd_score(value_fn: Callable[[tf.Tensor], tf.Tensor], theta: tf.Tensor, step: float) -> tf.Tensor:
    theta = tf.convert_to_tensor(theta, DTYPE)
    columns = []
    for axis in range(int(theta.shape[0])):
        direction = tf.one_hot(axis, int(theta.shape[0]), dtype=DTYPE) * step
        columns.append((value_fn(theta + direction) - value_fn(theta - direction)) / (2.0 * step))
    return tf.stack(columns)


def _relerr(candidate: tf.Tensor, reference: tf.Tensor) -> float:
    return float(
        (
            tf.linalg.norm(candidate - reference)
            / tf.maximum(tf.constant(1.0, DTYPE), tf.linalg.norm(reference))
        ).numpy()
    )


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def _cell(
    *,
    algorithm: str,
    value: tf.Tensor,
    score: tf.Tensor | None,
    value_fn: Callable[[tf.Tensor], tf.Tensor] | None,
    theta: tf.Tensor,
    fd_step: float,
    wall_value: float,
    reference_authority: str,
    claim: str,
    notes: str = "",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "algorithm": algorithm,
        "log_likelihood": float(tf.reshape(value, []).numpy()),
        "wall_seconds_value": wall_value,
        "reference_authority": reference_authority,
        "claim": claim,
    }
    if notes:
        row["notes"] = notes
    if score is not None and value_fn is not None:
        fd = _fd_score(value_fn, theta, fd_step)
        row["score_norm"] = float(tf.linalg.norm(score).numpy())
        row["score_fd_relative_error"] = _relerr(tf.reshape(score, [-1]), fd)
        row["score_status"] = "analytic"
    elif score is not None:
        row["score_norm"] = float(tf.linalg.norm(score).numpy())
        row["score_status"] = "analytic_unchecked"
    else:
        row["score_status"] = "none"
    return row


# ---------------------------------------------------------------------------
# Row 1: LGSSM 2D (exact Kalman oracle; SVD-UKF / cubature / cut4 columns)
# ---------------------------------------------------------------------------


def _lgssm_row() -> dict[str, Any]:
    from bayesfilter.testing import (
        make_nonlinear_accumulation_first_derivatives_tf,
        make_nonlinear_accumulation_model_tf,
        model_b_observations_tf,
    )
    from bayesfilter.nonlinear.svd_sigma_point_derivatives_tf import (
        tf_svd_cubature_score,
        tf_svd_cut4_score,
        tf_svd_ukf_score,
    )
    from bayesfilter.nonlinear.sigma_points_tf import tf_svd_sigma_point_log_likelihood
    from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_log_likelihood

    # Model B is the standing nonlinear-accumulation control; for the LGSSM
    # oracle row we use the deterministic 18D exact-target bundle instead.
    from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
        load_deterministic_lgssm_exact_target,
    )

    bundle = load_deterministic_lgssm_exact_target()
    adapter = bundle.adapter
    theta = tf.convert_to_tensor(bundle.raw_truth, DTYPE)

    def exact_value(current: tf.Tensor) -> tf.Tensor:
        return adapter.log_prob(current[tf.newaxis, :])[0]

    (value, score, _status), wall = _timed(
        lambda: adapter.neutra_batch_log_prob_and_grad_status(theta[tf.newaxis, :])
    )
    cells = [
        _cell(
            algorithm="exact_kalman_qr",
            value=value[0],
            score=score[0],
            value_fn=exact_value,
            theta=theta,
            fd_step=1e-5,
            wall_value=wall,
            reference_authority="EXACT_ORACLE",
            claim="EXACT_ORACLE",
            notes="18D deterministic triangular LGSSM posterior target",
        )
    ]
    return {
        "model": "LGSSM",
        "state_dim": 4,
        "observation_dim": 4,
        "horizon": "frozen bundle",
        "parameter_dim": int(theta.shape[0]),
        "theta": [float(v) for v in theta.numpy()],
        "reference": "exact Kalman (same target)",
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Rows 2-3: actual SV and KSC SV (dense references + TT/Kalman/SGQF/UKF)
# ---------------------------------------------------------------------------


def _sv_rows(fd_step: float) -> list[dict[str, Any]]:
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        generate_frozen_exact_sv_dataset_tf,
    )

    _states, observations = generate_frozen_exact_sv_dataset_tf(seed=83121, horizon=20)
    gamma = tf.constant([0.60], DTYPE)
    beta = tf.constant([0.40], DTYPE)
    sigma = tf.constant([1.00], DTYPE)
    theta = tf.stack(
        [tf.constant(0.2533471031357997, DTYPE), tf.math.log(beta[0])]
    )

    rows: list[dict[str, Any]] = []

    # ---- actual SV (exact transformed target) ----
    dense, wall_dense = _timed(
        lambda: highdim.exact_transformed_sv_independent_panel_dense_reference(
            observations, gamma=gamma, beta=beta, sigma=sigma, order=401, radius=8.0
        )
    )

    def _sv_physical(current: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        gam = tf.reshape(
            0.5 * (1.0 + tf.math.erf(current[0] / tf.sqrt(tf.constant(2.0, DTYPE)))), [1]
        )
        bet = tf.reshape(tf.exp(current[1]), [1])
        return gam, bet

    def sgqf_value(current: tf.Tensor) -> tf.Tensor:
        gam, bet = _sv_physical(current)
        return highdim.exact_transformed_sv_independent_panel_fixed_sgqf_filter(
            observations, gamma=gam, beta=bet, sigma=sigma, sparse_level=2
        ).log_likelihood

    sgqf, wall_sgqf = _timed(
        lambda: highdim.exact_transformed_sv_independent_panel_fixed_sgqf_filter(
            observations, gamma=gamma, beta=beta, sigma=sigma, sparse_level=2
        )
    )
    sgqf_score = highdim.exact_transformed_sv_independent_panel_fixed_sgqf_score(
        observations, gamma=gamma, beta=beta, sigma=sigma, sparse_level=2
    )

    def zc_tt_value(current: tf.Tensor) -> tf.Tensor:
        gam, bet = _sv_physical(current)
        return _sv_tt_filter(observations, gam, bet, sigma).log_likelihood

    tt, wall_tt = _timed(lambda: _sv_tt_filter(observations, gamma, beta, sigma))

    cells = [
        _cell(
            algorithm="dense_reference_o401",
            value=dense.log_likelihood,
            score=None,
            value_fn=None,
            theta=theta,
            fd_step=fd_step,
            wall_value=wall_dense,
            reference_authority="REFINED_NUMERICAL_REFERENCE",
            claim="REFINED_NUMERICAL_REFERENCE",
            notes="order-401 radius-8 dense grid; two-step refinement certified in prior artifacts",
        ),
        _cell(
            algorithm="fixed_sgqf_l2",
            value=sgqf.log_likelihood,
            score=sgqf_score.score,
            value_fn=sgqf_value,
            theta=theta,
            fd_step=fd_step,
            wall_value=wall_sgqf,
            reference_authority="dense_reference_o401",
            claim="CERTIFIED_APPROXIMATION",
        ),
        _cell(
            algorithm="zc_family_scalar_tt_frozen",
            value=tt.log_likelihood,
            score=None,
            value_fn=None,
            theta=theta,
            fd_step=fd_step,
            wall_value=wall_tt,
            reference_authority="dense_reference_o401",
            claim="CERTIFIED_APPROXIMATION",
            notes="existing per-model scalar fixed-design TT (comparator; not the generic engine)",
        ),
    ]
    dense_value = float(dense.log_likelihood.numpy())
    for cell_row in cells[1:]:
        cell_row["same_target_gap"] = abs(cell_row["log_likelihood"] - dense_value)
    rows.append(
        {
            "model": "ACTUAL_SV",
            "state_dim": 1,
            "observation_dim": 1,
            "horizon": 20,
            "parameter_dim": 2,
            "theta": [float(v) for v in theta.numpy()],
            "reference": "dense exact-transformed (REFINED_NUMERICAL_REFERENCE)",
            "cells": cells,
        }
    )

    # ---- KSC SV (mixture surrogate target) ----
    mixture = highdim.ksc_1998_log_chi_square_mixture()
    model = highdim.StochasticVolatilitySSM(sigma=sigma[0])
    theta_axis = model.unconstrained_from_physical(gamma=gamma[0], beta=beta[0])

    dense_ksc, wall_dense_ksc = _timed(
        lambda: highdim.scalar_sv_mixture_dense_reference(
            model, theta_axis, observations, mixture=mixture, order=401, radius=8.0
        )
    )
    kalman, wall_kalman = _timed(
        lambda: highdim.independent_panel_sv_mixture_kalman_filter(
            observations, gamma=gamma, beta=beta, sigma=sigma, mixture=mixture
        )
    )

    def ksc_kalman_value(current: tf.Tensor) -> tf.Tensor:
        gam, bet = _sv_physical(current)
        return highdim.independent_panel_sv_mixture_kalman_filter(
            observations, gamma=gam, beta=bet, sigma=sigma, mixture=mixture
        ).log_likelihood

    cut4, wall_cut4 = _timed(
        lambda: highdim.independent_panel_sv_mixture_cut4_filter(
            observations, gamma=gamma, beta=beta, sigma=sigma, mixture=mixture
        )
    )
    ksc_sgqf, wall_ksc_sgqf = _timed(
        lambda: highdim.independent_panel_sv_mixture_fixed_sgqf_filter(
            observations, gamma=gamma, beta=beta, sigma=sigma, mixture=mixture
        )
    )
    ksc_sgqf_score = highdim.independent_panel_sv_mixture_fixed_sgqf_score(
        observations, gamma=gamma, beta=beta, sigma=sigma, mixture=mixture
    )

    def ksc_sgqf_value(current: tf.Tensor) -> tf.Tensor:
        gam, bet = _sv_physical(current)
        return highdim.independent_panel_sv_mixture_fixed_sgqf_filter(
            observations, gamma=gam, beta=bet, sigma=sigma, mixture=mixture
        ).log_likelihood

    ksc_ukf, wall_ksc_ukf = _timed(
        lambda: highdim.independent_panel_sv_mixture_ukf_filter(
            observations, gamma=gamma, beta=beta, sigma=sigma, mixture=mixture
        )
    )
    ksc_ukf_score = highdim.independent_panel_sv_mixture_ukf_score(
        observations, gamma=gamma, beta=beta, sigma=sigma, mixture=mixture
    )

    def ksc_ukf_value(current: tf.Tensor) -> tf.Tensor:
        gam, bet = _sv_physical(current)
        return highdim.independent_panel_sv_mixture_ukf_filter(
            observations, gamma=gam, beta=bet, sigma=sigma, mixture=mixture
        ).log_likelihood

    ksc_cells = [
        _cell(
            algorithm="dense_ksc_reference_o401",
            value=dense_ksc.log_likelihood,
            score=None,
            value_fn=None,
            theta=theta,
            fd_step=fd_step,
            wall_value=wall_dense_ksc,
            reference_authority="REFINED_NUMERICAL_REFERENCE",
            claim="REFINED_NUMERICAL_REFERENCE",
        ),
        _cell(
            algorithm="mixture_kalman_exact_enumeration",
            value=kalman.log_likelihood,
            score=None,
            value_fn=ksc_kalman_value,
            theta=theta,
            fd_step=fd_step,
            wall_value=wall_kalman,
            reference_authority="dense_ksc_reference_o401",
            claim="CERTIFIED_APPROXIMATION",
            notes="exact component enumeration + Gaussian collapse (GPB1-style)",
        ),
        _cell(
            algorithm="mixture_cut4",
            value=cut4.log_likelihood,
            score=None,
            value_fn=None,
            theta=theta,
            fd_step=fd_step,
            wall_value=wall_cut4,
            reference_authority="dense_ksc_reference_o401",
            claim="CERTIFIED_APPROXIMATION",
        ),
        _cell(
            algorithm="mixture_fixed_sgqf",
            value=ksc_sgqf.log_likelihood,
            score=ksc_sgqf_score.score,
            value_fn=ksc_sgqf_value,
            theta=theta,
            fd_step=fd_step,
            wall_value=wall_ksc_sgqf,
            reference_authority="dense_ksc_reference_o401",
            claim="CERTIFIED_APPROXIMATION",
        ),
        _cell(
            algorithm="mixture_ukf",
            value=ksc_ukf.log_likelihood,
            score=ksc_ukf_score.score,
            value_fn=ksc_ukf_value,
            theta=theta,
            fd_step=fd_step,
            wall_value=wall_ksc_ukf,
            reference_authority="dense_ksc_reference_o401",
            claim="CERTIFIED_APPROXIMATION",
        ),
    ]
    dense_ksc_value = float(dense_ksc.log_likelihood.numpy())
    for cell_row in ksc_cells[1:]:
        cell_row["same_target_gap"] = abs(cell_row["log_likelihood"] - dense_ksc_value)
    rows.append(
        {
            "model": "KSC_SV",
            "state_dim": 1,
            "observation_dim": 1,
            "horizon": 20,
            "parameter_dim": 2,
            "theta": [float(v) for v in theta.numpy()],
            "reference": "dense KSC mixture (REFINED_NUMERICAL_REFERENCE)",
            "cells": ksc_cells,
        }
    )
    return rows


def _sv_tt_filter(observations, gamma, beta, sigma):
    convention = highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )
    product_basis = highdim.ProductBasis(
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), 48)], convention
    )
    config = highdim.FixedBranchFilterConfig(
        fit_config=highdim.FixedTTFitConfig(
            ranks=(1, 1),
            ridge=1e-12,
            max_sweeps=2,
            sweep_order=(0,),
            row_budget=512,
            column_budget=128,
            dense_matrix_byte_budget=200_000,
            normal_matrix_byte_budget=100_000,
            condition_number_warning=1e10,
            condition_number_veto=1e14,
            holdout_tolerance=5e-4,
        ),
        density_tau=0.0,
        normalizer_floor=1e-12,
        denominator_floor=1e-12,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(
            highdim.AffineCoordinateMap(
                offset=tf.constant([0.0], DTYPE), matrix=tf.constant([[8.0]], DTYPE)
            ),
        ),
        measure_convention=convention,
        deterministic_seed="baseline-leaderboard-sv-tt",
        product_basis=product_basis,
        initial_cores=(
            highdim.TTCore(tf.ones([1, product_basis.bases[0].basis_dim, 1], DTYPE)),
        ),
        fit_quadrature_order=141,
    )
    return highdim.exact_transformed_sv_independent_panel_zhaocui_tt_filter(
        observations,
        gamma=gamma,
        beta=beta,
        sigma=sigma,
        config=config,
        branch_seed_prefix="baseline-leaderboard-sv-tt",
    )


# ---------------------------------------------------------------------------
# Row 4: predator-prey, T in {20, 40} (SVD-UKF and SGQF analytic scores)
# ---------------------------------------------------------------------------


def _predator_prey_rows(fd_step: float, horizons: tuple[int, ...]) -> list[dict[str, Any]]:
    from bayesfilter.highdim.models import p30_predator_prey_fixture_model
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        pp_ukf_likelihood_value_score_status,
    )
    from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
        pp_sgqf_likelihood_value_score_status,
    )
    from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_cloud

    theta = tf.constant(
        [0.0, -0.5244005127080409, 0.0, -0.5244005127080409, 0.0, 0.0], DTYPE
    )
    model = p30_predator_prey_fixture_model()
    cloud = tf_fixed_sgqf_cloud(dim=2, sparse_level=2)

    rows = []
    for horizon in horizons:
        with tf.device("/CPU:0"):
            _states, observations = model.simulate(
                theta=model.true_parameters(), final_time=horizon - 1, seed=81104
            )
        y = tf.convert_to_tensor(observations, DTYPE)

        (ukf_value, ukf_score, ukf_status), wall_ukf = _timed(
            lambda: pp_ukf_likelihood_value_score_status(theta[tf.newaxis, :], observations=y)
        )

        def ukf_value_fn(current: tf.Tensor) -> tf.Tensor:
            value, _s, _st = pp_ukf_likelihood_value_score_status(
                current[tf.newaxis, :], observations=y
            )
            return value[0]

        (sgqf_value, sgqf_score, sgqf_status), wall_sgqf = _timed(
            lambda: pp_sgqf_likelihood_value_score_status(
                theta[tf.newaxis, :],
                observations=y,
                nodes=cloud.points,
                weights=cloud.weights,
            )
        )

        def sgqf_value_fn(current: tf.Tensor) -> tf.Tensor:
            value, _s, _st = pp_sgqf_likelihood_value_score_status(
                current[tf.newaxis, :],
                observations=y,
                nodes=cloud.points,
                weights=cloud.weights,
            )
            return value[0]

        cells = [
            _cell(
                algorithm="svd_ukf_principal_sqrt",
                value=ukf_value[0],
                score=ukf_score[0],
                value_fn=ukf_value_fn,
                theta=theta,
                fd_step=fd_step,
                wall_value=wall_ukf,
                reference_authority="cross_algorithm_only",
                claim="SURROGATE_USEFULNESS",
                notes="additive-Gaussian RK4 closure; no dense same-target reference run here",
            ),
            _cell(
                algorithm="fixed_sgqf_l2",
                value=sgqf_value[0],
                score=sgqf_score[0],
                value_fn=sgqf_value_fn,
                theta=theta,
                fd_step=fd_step,
                wall_value=wall_sgqf,
                reference_authority="cross_algorithm_only",
                claim="SURROGATE_USEFULNESS",
            ),
        ]
        cells[1]["cross_algorithm_gap_vs_svd_ukf"] = abs(
            cells[1]["log_likelihood"] - cells[0]["log_likelihood"]
        )
        rows.append(
            {
                "model": f"PREDATOR_PREY_T{horizon}",
                "state_dim": 2,
                "observation_dim": 2,
                "horizon": horizon,
                "parameter_dim": 6,
                "theta": [float(v) for v in theta.numpy()],
                "reference": "none (cross-algorithm descriptive only)",
                "cells": cells,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Row 5: Austria SIR d=18, T in {20, 40} (SVD-UKF and level-2 SGQF)
# ---------------------------------------------------------------------------


def _sir_rows(fd_step: float, horizons: tuple[int, ...]) -> list[dict[str, Any]]:
    from bayesfilter.highdim.models import zhao_cui_sir_austria_model
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        SIR_STATE_DIM,
        sir_sgqf_likelihood_value_score_status,
        sir_ukf_likelihood_value_score_status,
    )
    from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_level2_axis_cloud

    theta = tf.zeros([3], DTYPE)
    cloud = tf_fixed_sgqf_level2_axis_cloud(SIR_STATE_DIM)

    rows = []
    for horizon in horizons:
        with tf.device("/CPU:0"):
            _states, all_observations = zhao_cui_sir_austria_model().simulate(
                final_time=horizon, seed=81101
            )
        y = tf.convert_to_tensor(all_observations, DTYPE)[1 : horizon + 1]

        (ukf_value, ukf_score, _st), wall_ukf = _timed(
            lambda: sir_ukf_likelihood_value_score_status(theta[tf.newaxis, :], observations=y)
        )

        def ukf_value_fn(current: tf.Tensor) -> tf.Tensor:
            value, _s, _t = sir_ukf_likelihood_value_score_status(
                current[tf.newaxis, :], observations=y
            )
            return value[0]

        (sgqf_value, sgqf_score, _st2), wall_sgqf = _timed(
            lambda: sir_sgqf_likelihood_value_score_status(
                theta[tf.newaxis, :], observations=y, nodes=cloud.points, weights=cloud.weights
            )
        )

        def sgqf_value_fn(current: tf.Tensor) -> tf.Tensor:
            value, _s, _t = sir_sgqf_likelihood_value_score_status(
                current[tf.newaxis, :], observations=y, nodes=cloud.points, weights=cloud.weights
            )
            return value[0]

        cells = [
            _cell(
                algorithm="svd_ukf_principal_sqrt",
                value=ukf_value[0],
                score=ukf_score[0],
                value_fn=ukf_value_fn,
                theta=theta,
                fd_step=fd_step,
                wall_value=wall_ukf,
                reference_authority="cross_algorithm_only",
                claim="DIAGNOSTIC_ONLY",
                notes="d=18; no independent same-target reference exists at this dimension",
            ),
            _cell(
                algorithm="fixed_sgqf_l2_axis",
                value=sgqf_value[0],
                score=sgqf_score[0],
                value_fn=sgqf_value_fn,
                theta=theta,
                fd_step=fd_step,
                wall_value=wall_sgqf,
                reference_authority="cross_algorithm_only",
                claim="DIAGNOSTIC_ONLY",
            ),
        ]
        cells[1]["cross_algorithm_gap_vs_svd_ukf"] = abs(
            cells[1]["log_likelihood"] - cells[0]["log_likelihood"]
        )
        rows.append(
            {
                "model": f"AUSTRIA_SIR_T{horizon}",
                "state_dim": 18,
                "observation_dim": 9,
                "horizon": horizon,
                "parameter_dim": 3,
                "theta": [float(v) for v in theta.numpy()],
                "reference": "none at d=18 (DIAGNOSTIC_ONLY per audit)",
                "cells": cells,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Row 6: structural deterministic (Ch18b), T in {20, 100}
# ---------------------------------------------------------------------------


def _structural_rows(fd_step: float, horizons: tuple[int, ...]) -> list[dict[str, Any]]:
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_TRUTH_PHYSICAL,
        simulate_structural_trajectories_tf,
        structural_truth_source,
        structural_ukf_likelihood_value_score_status,
    )

    theta = structural_truth_source()

    rows = []
    for horizon in horizons:
        states, observations, residuals = simulate_structural_trajectories_tf(
            STRUCTURAL_TRUTH_PHYSICAL[None, :],
            horizon=horizon,
            seed=tf.constant((20260716, 15001), tf.int32),
        )
        max_residual = float(tf.reduce_max(tf.abs(residuals)).numpy())
        y = observations[0]

        (value, score, status), wall = _timed(
            lambda: structural_ukf_likelihood_value_score_status(
                theta[tf.newaxis, :], observations=y
            )
        )

        def value_fn(current: tf.Tensor) -> tf.Tensor:
            v, _s, _t = structural_ukf_likelihood_value_score_status(
                current[tf.newaxis, :], observations=y
            )
            return v[0]

        cells = [
            _cell(
                algorithm="structural_svd_ukf",
                value=value[0],
                score=score[0],
                value_fn=value_fn,
                theta=theta,
                fd_step=fd_step,
                wall_value=wall,
                reference_authority="cross_algorithm_only",
                claim="SURROGATE_USEFULNESS",
                notes=(
                    "Ch18b structural UKF: integration over (x_{t-1}, eps_t), "
                    "deterministic completion computed not noised; "
                    f"simulation completion residual max {max_residual:.2e}"
                ),
            )
        ]
        rows.append(
            {
                "model": f"STRUCTURAL_DETERMINISTIC_T{horizon}",
                "state_dim": 2,
                "observation_dim": 1,
                "horizon": horizon,
                "parameter_dim": 5,
                "theta": [float(v) for v in theta.numpy()],
                "reference": "none in this baseline (dense (x,eps) reference is a P4 artifact)",
                "cells": cells,
            }
        )
    return rows


# ---------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _write_markdown(path: Path, result: dict[str, Any], json_path: Path) -> None:
    lines = [
        "# Baseline Leaderboard (pre-generic-squared-TT)",
        "",
        f"- JSON: `{json_path}`",
        f"- Commit: `{result['git_commit']}` | TF {result['tensorflow_version']} | CPU float64",
        "",
        "## Nonclaims",
        "",
    ]
    lines.extend(f"- {claim}" for claim in result["nonclaims"])
    for row in result["rows"]:
        lines.extend(
            [
                "",
                f"## {row['model']} (n={row['state_dim']}, m={row['observation_dim']}, "
                f"T={row['horizon']}, p={row['parameter_dim']})",
                "",
                "| Algorithm | log-lik | same-target gap | score FD relerr | wall s | claim |",
                "|---|---|---|---|---|---|",
            ]
        )
        for cell_row in row["cells"]:
            gap = cell_row.get("same_target_gap", cell_row.get("cross_algorithm_gap_vs_svd_ukf"))
            gap_text = f"{gap:.3e}" if gap is not None else "-"
            fd = cell_row.get("score_fd_relative_error")
            fd_text = f"{fd:.2e}" if fd is not None else cell_row["score_status"]
            lines.append(
                f"| {cell_row['algorithm']} | {cell_row['log_likelihood']:.6f} | {gap_text} "
                f"| {fd_text} | {cell_row['wall_seconds_value']:.2f} | {cell_row['claim']} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--fd-step", type=float, default=1e-5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()

    started = time.time()
    rows: list[dict[str, Any]] = []
    rows.append(_lgssm_row())
    rows.extend(_sv_rows(args.fd_step))
    rows.extend(_predator_prey_rows(args.fd_step, (20, 40)))
    rows.extend(_sir_rows(args.fd_step, (20, 40)))
    rows.extend(_structural_rows(args.fd_step, (20, 100)))

    result = {
        "schema_version": "baseline_leaderboard.v1",
        "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "host": platform.node(),
        "python_version": platform.python_version(),
        "tensorflow_version": tf.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": _git_commit(),
        "fd_step": args.fd_step,
        "rows": rows,
        "generic_squared_tt_column": "BLOCKED_PENDING_P1A_P2_AUDIT_ARTIFACTS",
        "wall_time_seconds": time.time() - started,
        "nonclaims": list(NONCLAIMS),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.markdown_output:
        markdown = Path(args.markdown_output)
        markdown.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown, result, output)
    print(json.dumps({"rows": len(rows), "wall_time_seconds": result["wall_time_seconds"]}))


if __name__ == "__main__":
    main()

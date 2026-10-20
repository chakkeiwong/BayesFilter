"""C2 Gaussian-reference branch-axis value filter (Hermite, no boxes).

Specification: docs/plans/bayesfilter-c2-gaussian-reference-derivation-note-2026-08-24.md
(status REVIEWED; material review 2026-08-24, VERDICT AGREE after
repairs). Selection basis: monograph ch38 §40.8-40.9.

Mirrors `run_value_filter_branch_axis_adapted` with the reviewed
changes, and deletes — does not port — the bounded-program machinery
(deletion list, note §4.5): no containment shrink, no truncation-mass
correction, no box constants (hw, ±n log 2), no kappa window scaling.
The per-step frozen triangular maps ARE the whitening; the reference is
the standard normal measure on R^{2n}; mass matrices are the identity
(HermiteBasis1D), so every Gram contraction is a plain core product.

Reference-typed conversion (reviewed note §1, review item 1 CORRECT):

    log F_{t}(u_new) = log f + log g
      + log p_ret(u_old) + log eta_n(u_old) - log|det L_old|
      + log|det L_cc| + log|det L_pp| - log eta_2n(u_new),

with the eta terms row-dependent (quadratic) and theta-free under
frozen maps. Defensive policy (D3 as repaired, findings F2-F4):
tau_t = clamp(eps_rel^2, TAU_MIN, TAU_MAX) with eps_rel^2 the
Z_h-normalized weighted fit residual. Rows follow the optimal weighted
least-squares law (Cohen & Migliorati 2017): frozen Sobol pushed through
the per-axis induced-mixture CDF q1 = eta * mean_k He~_k^2 with inverse
Christoffel weights — at degree 0 this IS Phi^{-1}(Sobol) against the
reference. Raw eta-rows are exponentially sample-hungry in the degree
(measured 2026-08-24: Gram cond 1.3e6 at ell=13/N=2048, engine
overcounts +22..47 nats/step via unseen tail mass; weighted law: cond
1.28). Fail-closed on endpoint hits. Value path only; the adjoint
engine port follows the certified-node inventory of the note.
"""

from __future__ import annotations

import math
from typing import Callable

import tensorflow as tf

from bayesfilter.highdim.bases import HermiteBasis1D, ProductBasis
from bayesfilter.highdim.diagnostics import (
    DensityMeasure,
    MassMeasure,
    MeasureConvention,
)
from bayesfilter.highdim.filtering import AffineCoordinateMap
from bayesfilter.highdim.retained_quadratic_form_tf import (
    RetainedQuadraticForm,
    prefix_row_vectors,
    retained_quadratic_form_from_squared_tt,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import (
    DiscreteIndicatorBasis1D,
    EngineConfig,
    _design_rows,
    _fixed_als_fit,
    _initial_tt_cores,
)
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64

# Reviewed note D3 clamp (findings F2-F4): floor inherits the validated
# fixed-policy no-harm evidence; cap = declared per-step bar / 25.
TAU_MIN = 1e-6
TAU_MAX = 1e-4

_HERMITE_CONVENTION = MeasureConvention(
    density_measure=DensityMeasure.REFERENCE_MEASURE,
    mass_measure=MassMeasure.REFERENCE_MEASURE,
    reference_weight_name="standard_normal",
    physical_coordinate_name="x",
    reference_coordinate_name="u",
)


def _hermite_product_basis(dimension: int, degree: int) -> ProductBasis:
    return ProductBasis(
        [HermiteBasis1D(max_degree=degree) for _ in range(dimension)],
        _HERMITE_CONVENTION,
    )


def _christoffel_beta(dimension: int) -> float:
    """Dimension-aware Christoffel mixture weight (A3 calibration,
    2026-08-25). Per-axis importance weights multiply across axes, so
    the product ESS fraction decays geometrically in the axis count:
    at ell=13, beta=0.5 measured ESS 1400/2048 at d=4 (certified n=2
    scopes) but 111/2048 and 408/8192 at d=8 — the starved regime that
    produced non-finite retention in the first A3 run. beta=0.10 at
    d=8 measured ESS 5347/8192 at axis-Gram cond 1.31 (vs 1.16 for
    beta=0.5): the Chernoff-side cost of the lighter mixture is
    negligible at the d>4 operating row count while the ESS gain is
    ~13x. d<=4 keeps the certified beta=0.5. Evidence:
    check_c2_hermite_rowlaw_mechanism_20260824.py (beta table)."""

    return 0.5 if dimension <= 4 else 0.10


def _christoffel_axis_table(degree: int, beta: float) -> tuple[tf.Tensor, tf.Tensor]:
    """Grid CDF of the per-axis induced mixture q1 = eta * mean_k He~_k^2.

    Optimal weighted least squares (Cohen & Migliorati 2017): rows drawn
    from the Christoffel-adjusted mixture with weights 1/(mean_k He~_k^2)
    give a near-isometric empirical Gram at N ~ K log K, where raw
    eta-rows fail exponentially in the degree (Var(He~_k^2) ~ 4^k;
    measured at ell=13, N=2048: cond 1.3e6 raw vs 1.28 weighted). At
    degree 0 the mixture IS eta and the row law degenerates to
    Phi^{-1}(Sobol) exactly.
    """

    grid = tf.linspace(tf.constant(-14.0, DTYPE), tf.constant(14.0, DTYPE), 40001)
    values = HermiteBasis1D(max_degree=degree).evaluate(grid)
    log_eta1 = -0.5 * (math.log(2.0 * math.pi) + tf.square(grid))
    # defensive half-mixture q1 = eta * (1/2 + 1/2 cbar): bounded weights
    # 1/(1/2 + cbar/2) <= 2 per axis keep the effective sample size at
    # O(N) (pure product-Christoffel weights collapsed ESS to ~1% of N,
    # measured 2026-08-24), while w * cbar <= 2 preserves the matrix-
    # Chernoff Gram guarantee up to a factor 2 (measured: cond 1.32 at
    # ell=13, N=2048, ESS 1400).
    christoffel_bar = tf.reduce_mean(tf.square(values), axis=1)
    density = tf.exp(log_eta1) * ((1.0 - beta) + beta * christoffel_bar)
    cdf = tf.cumsum(density)
    cdf = cdf / cdf[-1]
    return grid, cdf


def _christoffel_rows(
    config: EngineConfig,
    count: int,
    dimension: int,
    seed: tuple[int, int],
    degree: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Frozen Sobol rows pushed through the per-axis induced-mixture CDF.

    Returns (rows [N, d], fit weights [N] summing to one). The weights
    are the tensorized inverse Christoffel factors: since
    q1 = eta * mean_k He~_k^2, the ratio eta/q1 = 1/(mean_k He~_k^2),
    so log w = -sum_axes log(mean_k He~_k(u_axis)^2) up to the
    normalization absorbed below. Deterministic, seed-stable, frozen
    before the fit (V1 contract unchanged).
    """

    uniform = _design_rows(config, count, dimension, seed)
    max_abs = float(tf.reduce_max(tf.abs(uniform)).numpy())
    if max_abs >= 1.0:
        raise ValueError(
            "christoffel rows: uniform design touched the open-interval "
            "boundary; inverse CDF would be ill-defined (fail closed)"
        )
    probabilities = 0.5 * (uniform + 1.0)
    beta = _christoffel_beta(dimension)
    grid, cdf = _christoffel_axis_table(degree, beta)
    flat = tf.reshape(probabilities, [-1])
    upper = tf.clip_by_value(
        tf.searchsorted(cdf, flat, side="left"), 1, int(cdf.shape[0]) - 1
    )
    cdf_hi = tf.gather(cdf, upper)
    cdf_lo = tf.gather(cdf, upper - 1)
    grid_hi = tf.gather(grid, upper)
    grid_lo = tf.gather(grid, upper - 1)
    fraction = (flat - cdf_lo) / tf.maximum(cdf_hi - cdf_lo, 1e-300)
    rows = tf.reshape(grid_lo + fraction * (grid_hi - grid_lo), tf.shape(probabilities))
    if not bool(tf.reduce_all(tf.math.is_finite(rows)).numpy()):
        raise ValueError("christoffel rows: non-finite row after inverse CDF")
    values = HermiteBasis1D(max_degree=degree).evaluate(tf.reshape(rows, [-1]))
    log_mixture = tf.reshape(
        tf.math.log(
            (1.0 - beta) + beta * tf.reduce_mean(tf.square(values), axis=1)
        ),
        tf.shape(rows),
    )
    log_weight = -tf.reduce_sum(log_mixture, axis=1)
    weights = tf.exp(log_weight - tf.reduce_logsumexp(log_weight))
    # Class-A observability (campaign plan CF2): the effective sample
    # size of the importance weights is computed here anyway — emit it.
    row_ess = 1.0 / tf.reduce_sum(tf.square(weights))
    return rows, weights, float(row_ess.numpy())


def _log_eta(points: tf.Tensor) -> tf.Tensor:
    """Row-wise log standard-normal density at the points' dimension."""

    dim = tf.cast(tf.shape(points)[1], DTYPE)
    half_log_two_pi = tf.constant(0.5 * math.log(2.0 * math.pi), DTYPE)
    return -dim * half_log_two_pi - 0.5 * tf.reduce_sum(tf.square(points), axis=1)


def _check_hint(mean: tf.Tensor, cov: tf.Tensor, n: int) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.convert_to_tensor(mean, DTYPE)
    cov = tf.convert_to_tensor(cov, DTYPE)
    if mean.shape.as_list() != [n] or cov.shape.as_list() != [n, n]:
        raise ValueError("moment hint shape mismatch")
    if not bool(
        (
            tf.reduce_all(tf.math.is_finite(mean))
            & tf.reduce_all(tf.math.is_finite(cov))
        ).numpy()
    ):
        raise ValueError("moment hint nonfinite")
    chol = tf.linalg.cholesky(cov)
    if not bool(tf.reduce_all(tf.math.is_finite(chol)).numpy()):
        raise ValueError("moment hint covariance not positive definite")
    return mean, chol


def _clamped_tau(weighted_fit_rms: float, z_h: tf.Tensor) -> tuple[tf.Tensor, float]:
    """tau_t = clamp(eps_rel^2, TAU_MIN, TAU_MAX); eps_rel^2 = rms^2 / Z_h."""

    eps_rel_sq = float(weighted_fit_rms) ** 2 / max(float(z_h.numpy()), 1e-300)
    tau_value = min(max(eps_rel_sq, TAU_MIN), TAU_MAX)
    return tf.constant(tau_value, DTYPE), eps_rel_sq


def _logdet_lower(matrix: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.math.log(tf.abs(tf.linalg.diag_part(matrix))))


def _student_t_log_const(nu: float) -> float:
    """Constant of log(t_nu(u)/eta(u)) per axis (closed form)."""

    return float(
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        + 0.5 * math.log(2.0 / nu)
    )


def _log_student_t_ratio(points: tf.Tensor, nu: float) -> tf.Tensor:
    """log lambda_mu at points [N, d]: sum_axes log(t_nu/eta) per axis.

    lambda_mu is a probability density w.r.t. mu = N(0, I) by
    construction (each axis factor integrates to one against eta), so
    the increment identity Z = e^{-c}(1+tau)Z_h and the F7 oracle-gate
    subtraction are unchanged by the escalation (campaign plan C2; the
    prior review verified this closed-form claim)."""

    const = tf.constant(_student_t_log_const(nu), DTYPE)
    u_sq = tf.square(points)
    per_axis = (
        const
        + 0.5 * u_sq
        - ((nu + 1.0) / 2.0) * tf.math.log1p(u_sq / nu)
    )
    return tf.reduce_sum(per_axis, axis=1)


def student_t_margin(nu: float, alpha: float) -> float:
    """Per-axis domination margin M(nu, alpha) = sup_u [alpha u^2/2
    - log lambda_1(u)] for a whitened tail excess alpha in (0, 1)
    (review F1 tail model: log F ~ alpha u^2/2 along the worst ray).
    Closed-form maximizer s* = (nu+1)/(1-alpha) - nu."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s_star = max((nu + 1.0) / (1.0 - alpha) - nu, 0.0)
    return (
        -(1.0 - alpha) * s_star / 2.0
        + ((nu + 1.0) / 2.0) * math.log1p(s_star / nu)
        - _student_t_log_const(nu)
    )


def student_t_nu_criterion(alpha_max: float, cap_per_axis: float) -> float:
    """Two-sided nu selection (campaign plan C2, review CF5 repair):
    the LARGEST nu (lightest tails, least bulk dilution) whose per-axis
    margin satisfies M(nu, alpha_max) <= cap_per_axis. M is monotone
    increasing in nu, so the largest admissible nu exists by bisection;
    domination itself holds for every finite nu (the margin cap, tied
    to the ratio guard at tau >= TAU_MIN, is what selects)."""

    lo, hi = 1.5, 500.0
    if student_t_margin(lo, alpha_max) > cap_per_axis:
        raise ValueError(
            "no admissible nu: margin cap violated even at nu=1.5 — "
            "re-declare the cap or the hint class (fail closed)"
        )
    if student_t_margin(hi, alpha_max) <= cap_per_axis:
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if student_t_margin(mid, alpha_max) <= cap_per_axis:
            lo = mid
        else:
            hi = mid
    return lo


def run_value_filter_branch_axis_gaussian(
    adapter,
    observations: tf.Tensor,
    config: EngineConfig,
    *,
    predictive_moment_hint: Callable[[int, tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    initial_moment_hint: Callable[[tf.Tensor], tuple[tf.Tensor, tf.Tensor]],
    defensive_nu: float | None = None,
) -> tuple[tf.Tensor, list[dict]]:
    """Gaussian-reference value filter. `predictive_moment_hint(t, y_t)`
    returns JOINT moments (mean [2n], cov [2n,2n]) of (x_t, x_{t-1}) in
    (current, previous) order for t >= 1; `initial_moment_hint(y_0)`
    returns the t=0 moments (mean [n], cov [n,n]) of x_0 | y_0. Hints
    are frozen step inputs (M2/M3/M1-DETACHED contracts unchanged).
    `defensive_nu`: None = reference floor (lambda == 1, the default);
    a float = product Student-t floor per the D3 escalation — the
    EXPECTED configuration for the SV arm (review F1), selected by
    `student_t_nu_criterion` per declared scope, never on claim data."""

    if config.quadrature_order is not None:
        raise ValueError("gaussian engine is defined for scattered rows only")
    if predictive_moment_hint is None or initial_moment_hint is None:
        raise ValueError("gaussian engine requires frozen moment hints")

    n = adapter.state_dim
    observations = tf.convert_to_tensor(observations, DTYPE)
    horizon = int(observations.shape[0])
    current_basis = _hermite_product_basis(n, config.basis_degree)
    basis_dim = int(current_basis.bases[0].basis_dim)

    log_likelihood = tf.constant(0.0, DTYPE)
    retained: RetainedQuadraticForm | None = None
    diagnostics: list[dict] = []

    for t in range(horizon):
        if t == 0:
            m_c, l_cc = _check_hint(*initial_moment_hint(observations[0]), n)
            map_c = AffineCoordinateMap(offset=m_c, matrix=l_cc)
            rows, weights, row_ess = _christoffel_rows(
                config, config.row_count, n, (config.seed, 17), config.basis_degree
            )
            x_current = m_c[None, :] + tf.einsum("ij,nj->ni", l_cc, rows)
            conversion = _logdet_lower(l_cc) - _log_eta(rows)
            log_f = (
                adapter.initial_log_density(x_current)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(
                tf.cast(tf.shape(log_f)[0], DTYPE)
            )
            sqrt_target = tf.exp(0.5 * (log_f - shift))
            cores, fit_diag = _fixed_als_fit(
                current_basis, rows, sqrt_target, weights,
                _initial_tt_cores(n, basis_dim, config.rank), config,
            )
            suffix_core = tf.zeros([int(cores[-1].right_rank), basis_dim, 1], DTYPE)
            suffix_core = tf.tensor_scatter_nd_update(suffix_core, [[0, 0, 0]], [1.0])
            extended = tuple(cores) + (TTCore(suffix_core),)
            extended_basis = _hermite_product_basis(n + 1, config.basis_degree)
            base = retained_quadratic_form_from_squared_tt(
                extended, extended_basis, split_index=n, tau=0.0,
                prefix_basis=current_basis, coordinate_map=map_c,
            )
            step_extra: dict = {}
        else:
            map_old = retained.coordinate_map
            l_old = tf.convert_to_tensor(map_old.matrix, DTYPE)
            m_old = tf.convert_to_tensor(map_old.offset, DTYPE)
            joint_mean, joint_chol = _check_hint(
                *predictive_moment_hint(t, observations[t]), 2 * n
            )
            m_c = joint_mean[:n]
            m_p = joint_mean[n:]
            l_cc = joint_chol[:n, :n]
            l_pc = joint_chol[n:, :n]
            l_pp = joint_chol[n:, n:]
            map_c = AffineCoordinateMap(offset=m_c, matrix=l_cc)

            gram = retained.suffix_gram
            floor_scale = tf.linalg.trace(gram) / tf.cast(tf.shape(gram)[0], DTYPE)
            chol = tf.linalg.cholesky(
                gram
                + tf.constant(config.branch_gram_floor, DTYPE)
                * floor_scale
                * tf.eye(tf.shape(gram)[0], dtype=DTYPE)
            )
            branch_count = retained.boundary_rank + 1
            u_rows, u_weights, row_ess = _christoffel_rows(
                config, config.row_count, 2 * n, (config.seed, 100 + t),
                config.basis_degree,
            )
            u_c = u_rows[:, :n]
            u_p = u_rows[:, n:]
            x_current = m_c[None, :] + tf.einsum("ij,nj->ni", l_cc, u_c)
            x_previous = (
                m_p[None, :]
                + tf.einsum("ij,nj->ni", l_pc, u_c)
                + tf.einsum("ij,nj->ni", l_pp, u_p)
            )
            # re-express previous rows in the OLD whitened coordinates —
            # defined for every row (full support; no containment exists)
            u_old = tf.transpose(
                tf.linalg.triangular_solve(
                    l_old, tf.transpose(x_previous - m_old[None, :]), lower=True
                )
            )
            if not bool(tf.reduce_all(tf.math.is_finite(u_old)).numpy()):
                raise ValueError("gaussian engine: non-finite retained re-expression")
            logdet_c = _logdet_lower(l_cc)
            logdet_p = _logdet_lower(l_pp)
            logdet_old = _logdet_lower(l_old)
            # reviewed note §1 display: row-dependent eta-ratio conversion
            conversion = (
                logdet_c
                + logdet_p
                - logdet_old
                + _log_eta(u_old)
                - _log_eta(u_rows)
            )
            log_g = (
                adapter.transition_log_density(x_current, x_previous)
                + adapter.observation_log_density(x_current, observations[t])
                + conversion
            )
            v_prev = tf.einsum(
                "na,ab->nb",
                prefix_row_vectors(retained.prefix_cores, retained.prefix_basis, u_old),
                chol,
            )
            if defensive_nu is None:
                floor_values = retained.tau * tf.ones(
                    [int(u_rows.shape[0])], DTYPE
                )
            else:
                floor_values = retained.tau * tf.exp(
                    _log_student_t_ratio(u_old, defensive_nu)
                )
            sum_sq = tf.reduce_sum(tf.square(v_prev), axis=1) + floor_values
            log_f = tf.math.log(sum_sq) + log_g
            shift = tf.reduce_logsumexp(log_f) - tf.math.log(
                tf.cast(tf.shape(log_f)[0], DTYPE)
            )
            sqrt_g_shifted = tf.exp(0.5 * (log_g - shift))
            amplitudes = tf.concat(
                [v_prev, tf.sqrt(floor_values)[:, None]],
                axis=1,
            )
            targets = amplitudes * sqrt_g_shifted[:, None]
            g_codes = tf.tile(
                tf.range(branch_count, dtype=DTYPE)[None, :],
                [int(u_rows.shape[0]), 1],
            )
            full_rows = tf.concat(
                [
                    tf.repeat(u_rows[:, :n], branch_count, axis=0),
                    tf.reshape(g_codes, [-1, 1]),
                    tf.repeat(u_rows[:, n:], branch_count, axis=0),
                ],
                axis=1,
            )
            sqrt_target = tf.reshape(targets, [-1])
            weights = tf.reshape(tf.repeat(u_weights, branch_count, axis=0), [-1])
            mixed_basis = ProductBasis(
                list(current_basis.bases)
                + [DiscreteIndicatorBasis1D(branch_count)]
                + list(_hermite_product_basis(n, config.basis_degree).bases),
                current_basis.convention,
            )
            mixed_dims = [basis_dim] * n + [branch_count] + [basis_dim] * n
            cores0 = tuple(
                TTCore(
                    0.3
                    * tf.random.stateless_normal(
                        [
                            1 if axis == 0 else config.rank,
                            mixed_dims[axis],
                            1 if axis == 2 * n else config.rank,
                        ],
                        tf.constant((config.seed, 7000 + 31 * t + axis), tf.int32),
                        dtype=DTYPE,
                    )
                )
                for axis in range(2 * n + 1)
            )
            cores, fit_diag = _fixed_als_fit(
                mixed_basis, full_rows, sqrt_target, weights, cores0, config
            )
            base = retained_quadratic_form_from_squared_tt(
                tuple(cores), mixed_basis, split_index=n, tau=0.0,
                prefix_basis=current_basis, coordinate_map=map_c,
            )
            step_extra = {
                "map_logdet_c": float(logdet_c.numpy()),
                "map_logdet_p": float(logdet_p.numpy()),
                "u_old_max": float(tf.reduce_max(tf.abs(u_old)).numpy()),
            }
        z_h_new = base.z_complete_ref
        tau_t, eps_rel_sq = _clamped_tau(fit_diag["weighted_fit_rms"], z_h_new)
        retained_new = RetainedQuadraticForm(
            prefix_cores=base.prefix_cores, suffix_gram=base.suffix_gram,
            tau=tau_t * z_h_new, z_complete_ref=(1.0 + tau_t) * z_h_new,
            prefix_basis=base.prefix_basis, coordinate_map=base.coordinate_map,
        )
        if t == 0:
            log_increment = shift + tf.math.log(retained_new.z_complete_ref)
        else:
            log_increment = (
                shift
                + tf.math.log(retained_new.z_complete_ref)
                - tf.math.log(retained.z_complete_ref)
            )
        log_likelihood += log_increment
        retained = retained_new
        diagnostics.append(
            {
                "time_index": t,
                "log_increment": float(log_increment.numpy()),
                "tau_t": float(tau_t.numpy()),
                "eps_rel_sq": eps_rel_sq,
                "row_ess": row_ess,
                "tie_flag": False,
                **fit_diag,
                **step_extra,
            }
        )
    return log_likelihood, diagnostics


__all__ = [
    "TAU_MAX",
    "TAU_MIN",
    "run_value_filter_branch_axis_gaussian",
]

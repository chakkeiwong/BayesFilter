"""Observation-guided adjacent-state TT proposal extension (TensorFlow only).

Implements a recursive likelihood-weighted SGQF guide, exact affine pullback,
L1-selected Hermite regression, both retained marginal and upper conditional
KR consumers, and exact SV density factors. This is not the author's TT-cross
implementation. See docs/plans/observation-aware-tt-repair-complete-program-20260913.md.
Host code constructs rules/charts and records diagnostics; repeated fitting and
sampling kernels have stable signatures and default to XLA. No pfor is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
from typing import Callable

import tensorflow as tf

from bayesfilter.nonlinear.fixed_sgqf_tf import tf_fixed_sgqf_cloud
from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
    GaussianHermiteRetainedProposal, _normalized_hermite_values,
    _paired_right_environments, _prefix_row_vectors, _log_standard_normal,
    inverse_hermite_polynomial_kr,
)
from bayesfilter.highdim import pair_block_tt_tf as pairtt

D = tf.float64
LOG2PI = math.log(2.0 * math.pi)
EPS = 2.220446049250313e-16


def joint_sgqf_row_sampler(model, current: Chart, condition: Chart, rows: int, seed: int,
                           *, epsilon: float = 0.2, guide_current=None, guide_condition=None):
    """Draw paired regression rows from a frozen SGQF Gaussian mixture.

    The first component is the product reference ``rho``.  The second draws
    current ``x`` from the SGQF posterior and a Gaussian backward conditional
    for ``z``; returning ``log rho/log s`` makes the weighted objective
    auditable.  The mixture is formed in coordinate space and has the bound
    ``rho/s <= 1/epsilon``.
    """
    if not 0. < epsilon <= 1.:
        raise ValueError("epsilon must be in (0,1]")
    # Optional charts may differ from the physical Gaussian guide. Never
    # reinterpret chart scale as guide covariance when transforming rows.
    guide_current = current if guide_current is None else guide_current
    guide_condition = condition if guide_condition is None else guide_condition
    d = model.dimension
    base = tf.random.stateless_normal([rows, 2*d], [seed, 0], dtype=D)
    # Gaussian backward conditional z|x for z~N(m,P), x|z~N(Az,Q).
    A, Q = model.transition, model.sigma**2 * tf.eye(d, dtype=D)
    Pz = guide_condition.factor @ tf.transpose(guide_condition.factor)
    mz = guide_condition.mean
    S = A @ Pz @ tf.transpose(A) + Q
    K = tf.transpose(tf.linalg.solve(S, A @ Pz))
    Pback = Pz - K @ S @ tf.transpose(K)
    back_factor = spd_factor(Pback, "backward SGQF covariance")
    x2 = guide_current.forward(base[:, :d])
    Az = tf.linalg.matvec(A, mz)
    z2 = mz + tf.linalg.matmul(x2 - Az[None, :], K, transpose_b=True)
    z2 = z2 + tf.linalg.matmul(base[:, d:], back_factor, transpose_b=True)
    guided = tf.concat([current.inverse(x2), condition.inverse(z2)], axis=1)
    choose = tf.random.stateless_uniform([rows], [seed, 1], dtype=D) < epsilon
    coordinates = tf.where(choose[:, None], base, guided)
    log_rho = _log_standard_normal(coordinates)
    # Compute s_joint in coordinate space for the guided component.
    x_coord = current.forward(coordinates[:, :d])
    z_coord = condition.forward(coordinates[:, d:])
    log_px = guide_current.log_prob(x_coord) + current.logdet
    residual = z_coord - (mz[None, :] + tf.linalg.matmul(x_coord - Az[None, :], K, transpose_b=True))
    standardized = tf.transpose(tf.linalg.triangular_solve(back_factor, tf.transpose(residual)))
    log_pback = _log_standard_normal(standardized) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(back_factor)))
    log_sg = log_px + log_pback + condition.logdet
    # s = eps*rho + (1-eps)*sg; evaluate the mixture at the selected rows.
    log_s = tf.reduce_logsumexp(tf.stack([tf.math.log(tf.constant(epsilon, D)) + log_rho,
                                          tf.math.log(tf.constant(1.-epsilon, D)) + log_sg], axis=0), axis=0)
    logweights = log_rho - log_s
    weights = tf.exp(logweights)
    if float(tf.reduce_max(weights).numpy()) > 1./epsilon + 1e-9:
        raise ValueError("deterministic-mixture importance-weight bound violated")
    return coordinates, logweights, {"epsilon": epsilon,
        "maximum_rho_over_s": tf.reduce_max(weights),
        "row_ess": tf.square(tf.reduce_sum(weights))/tf.reduce_sum(tf.square(weights)),
        "guided_fraction": tf.reduce_mean(tf.cast(~choose, D))}


def pair_fit_from_log_target(log_target: Callable, dimension: int, seed: int,
                             *, rows: int = 1024, degree: int = 3, rank: int = 3,
                             sweeps: int = 4, proximal_steps: int = 128,
                             penalty: float = 0., row_sampler=None, jit_compile=True):
    """Fit a pair-block amplitude on disjoint calibration/validation/audit rows."""
    raw_splits = []
    for split in range(3):
        if row_sampler is None:
            row = tf.random.stateless_normal([rows, 2*dimension], [seed, split], dtype=D)
            logw = tf.zeros([rows], D)
            wi = {"maximum_rho_over_s": tf.constant(1., D), "row_ess": tf.cast(rows, D)}
        else:
            row, logw, wi = row_sampler(rows, seed + 100*split)
        row = tf.stack([row[:, :dimension], row[:, dimension:]], axis=-1)
        logvalue = log_target(row)
        raw_splits.append((row, logvalue, tf.exp(logw), wi, logw))
    # The scale is a training-only normalization and is frozen for validation
    # and audit, so split changes cannot silently alter the fitted target.
    scale = tf.reduce_logsumexp(raw_splits[0][1] + raw_splits[0][4]) - tf.math.log(tf.cast(rows, D))
    splits = []
    for row, logvalue, weights, wi, _logw in raw_splits:
        target = tf.exp(.5 * (logvalue - scale))
        splits.append((row, pairtt._pair_features(row, degree), target, weights, wi, scale))
    candidates = []
    for penalty_value in (0., 1e-5, 1e-3) if penalty == -1 else (penalty,):
        cores, fitdiag = pairtt.fit_pair_features(splits[0][1], splits[0][2], splits[0][3],
            degree=degree, rank=rank, sweeps=sweeps, proximal_steps=proximal_steps,
            penalty=penalty_value, initial=None, jit_compile=jit_compile)
        train = tf.sqrt(tf.reduce_sum(splits[0][3] * tf.square(pairtt.evaluate_pair_cores(cores, splits[0][0]) - splits[0][2])) /
                        tf.reduce_sum(splits[0][3] * splits[0][2]**2))
        candidates.append((penalty_value, cores, fitdiag, train))
    # Validation selection is explicit; audit rows are evaluated only here.
    def residual(candidate, split):
        values = pairtt.evaluate_pair_cores(candidate[1], split[0])
        return tf.sqrt(tf.reduce_sum(split[3] * tf.square(values - split[2])) /
                       tf.reduce_sum(split[3] * split[2]**2))
    selected = min(candidates, key=lambda x: float(residual(x, splits[1])))
    audit = residual(selected, splits[2])
    return selected[1], {"selected_l1": selected[0], "fit": selected[2],
                         "bond_spectra": pairtt.pair_bond_spectra(selected[1]),
                         "train_relative_rms": selected[3],
                         "validation_relative_rms": residual(selected, splits[1]),
                         "l1_selection": [{"l1": c[0], "validation_relative_rms": residual(c, splits[1])} for c in candidates],
                         "audit_relative_rms": audit, "row_sampling": splits[0][4],
                         "log_target_scale": splits[0][5],
                         "audit_used_for_selection": False}


@dataclass(frozen=True)
class PairRetainedProposal:
    """Retained current marginal obtained by integrating each paired past axis."""
    cores: tuple[tf.Tensor, ...]
    chart: Chart
    z: tf.Tensor
    tau: tf.Tensor
    time_index: int

    def physical_log_density(self, x):
        u = self.chart.inverse(x)
        quadratic = pairtt.pair_retained_quadratic(self.cores, u)
        return (tf.math.log(quadratic + self.tau) + _log_standard_normal(u)
                - tf.math.log(self.z + self.tau) - self.chart.logdet)


@dataclass(frozen=True)
class PairTTStep:
    cores: tuple[tf.Tensor, ...]
    current_chart: Chart
    conditioning_chart: Chart | None
    tau: tf.Tensor
    fit_diagnostics: dict
    time_index: int
    retained_proposal: PairRetainedProposal


def build_pair_tt_path(model, observations, guide_path, *, guided=True, seed=9100,
                       degree=3, rank=3, rows=1024, sweeps=4,
                       proximal_steps=128, defensive_mass=.05,
                       importance_rows=False, jit_compile=True, on_step=None):
    """Build the paired adjacent-state path used by the repair experiment."""
    result, retained = [], None
    for t, observation in enumerate(tf.unstack(observations)):
        current = guide_path[t][1 if guided else 0]
        condition = None if t == 0 else guide_path[t-1][1]
        d = model.dimension

        def log_target(row):
            row = tf.convert_to_tensor(row, D)
            u, v = row[:, :, 0], row[:, :, 1]
            x, previous = current.forward(u), (condition.forward(v) if condition is not None else tf.zeros_like(u))
            value = model.observation_log_prob(x, observation) + current.logdet - _log_standard_normal(u)
            if t == 0:
                # The unused past coordinate is retained under its Gaussian
                # reference measure.  Dividing by rho(v) would instead make
                # the joint target non-integrable in that dummy coordinate.
                value += model.prior_log_prob(x)
            else:
                value += model.transition_log_prob(x, previous) + retained.physical_log_density(previous)
                value += condition.logdet - _log_standard_normal(v)
            return value

        sampler = None
        if importance_rows and t > 0:
            sampler = lambda count, local_seed: joint_sgqf_row_sampler(
                model, current, condition, count, local_seed)
        cores, info = pair_fit_from_log_target(log_target, d, seed + t,
            rows=rows, degree=degree, rank=rank, sweeps=sweeps,
            proximal_steps=proximal_steps, penalty=-1, row_sampler=sampler, jit_compile=jit_compile)
        mass = pairtt.pair_total_mass(cores)
        finite(mass, "pair fitted TT mass")
        if float(mass.numpy()) <= 0.:
            raise ValueError("nonpositive pair TT mass")
        tau = mass * defensive_mass / (1. - defensive_mass)
        retained = PairRetainedProposal(cores, current, mass, tau, t)
        step = PairTTStep(cores, current, condition, tau, info, t, retained)
        result.append(step)
        if on_step is not None:
            on_step(t, info)
    return result


class PairConditionalSamplingError(ValueError):
    """A rejected sample, retaining diagnostics at the host reporting boundary."""

    def __init__(self, time_index, seed, diagnostics):
        self.time_index = time_index
        self.seed = seed
        self.diagnostics = diagnostics
        super().__init__(
            f"pair conditional failed finite/bracket/CDF check: t={time_index}, "
            f"seed={seed}, finite={bool(diagnostics['finite'])}, "
            f"bracket={bool(diagnostics['cdf_bracket_valid'])}, "
            f"cdf_residual={float(diagnostics['cdf_residual']):.17g}")


def sample_pair_tt_step(step: PairTTStep, previous: tf.Tensor, seed: int, jit_compile=True):
    """Draw the particle-specific conditional from one pair-block step."""
    n, d = int(previous.shape[0]), int(step.current_chart.mean.shape[0])
    if step.conditioning_chart is None:
        # The initial target is handled by the existing scalar proposal path.
        raise ValueError("pair conditional is defined for t>=1")
    v = step.conditioning_chart.inverse(previous)
    mix = tf.random.stateless_uniform([n], [seed, 0], dtype=D)
    uniforms = tf.random.stateless_uniform([n, d], [seed, 1], minval=EPS, maxval=1.-EPS, dtype=D)
    noise = tf.random.stateless_normal([n, d], [seed, 2], dtype=D)
    sampler = pairtt.compiled_pair_sampler([c.shape for c in step.cores], jit_compile)
    u, logq, diag = sampler(step.cores, v, step.tau, mix, uniforms, noise)
    if not bool(diag["finite"]) or float(diag["cdf_residual"]) > 1e-8:
        raise PairConditionalSamplingError(step.time_index, seed, diag)
    x = step.current_chart.forward(u)
    logq = logq - step.current_chart.logdet
    finite(x, "pair proposal draw")
    finite(logq, "pair proposal density")
    return x, logq, diag


def finite(value, name):
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        raise ValueError(f"non-finite {name}")


def spd_factor(covariance, name):
    covariance = (covariance + tf.transpose(covariance)) * 0.5
    finite(covariance, name)
    eigenvalues = tf.linalg.eigvalsh(covariance)
    margin = float(tf.reduce_min(eigenvalues).numpy())
    bound = 64 * EPS * int(covariance.shape[0]) * float(tf.reduce_max(tf.abs(eigenvalues)).numpy())
    if margin <= bound:
        raise ValueError(f"{name} is not numerically SPD: {margin} <= {bound}")
    return tf.linalg.cholesky(covariance)


@dataclass(frozen=True)
class Chart:
    mean: tf.Tensor
    factor: tf.Tensor

    @classmethod
    def from_moments(cls, mean, covariance):
        return cls(tf.convert_to_tensor(mean, D), spd_factor(covariance, "chart covariance"))

    @property
    def logdet(self):
        return tf.reduce_sum(tf.math.log(tf.linalg.diag_part(self.factor)))

    def forward(self, u):
        return self.mean + tf.linalg.matmul(u, self.factor, transpose_b=True)

    def inverse(self, x):
        return tf.transpose(tf.linalg.triangular_solve(self.factor, tf.transpose(x - self.mean)))

    def log_prob(self, x):
        return _log_standard_normal(self.inverse(x)) - self.logdet


@dataclass(frozen=True)
class SVModel:
    transition: tf.Tensor
    covariance0: tf.Tensor
    beta: float = 0.4
    sigma: float = 1.0

    @property
    def dimension(self):
        return int(self.transition.shape[0])

    def observation_log_prob(self, x, y, log_beta=None):
        lb = tf.math.log(tf.constant(self.beta, D)) if log_beta is None else log_beta
        logvariance = 2 * lb + x
        return -0.5 * tf.reduce_sum(LOG2PI + logvariance + tf.square(y) * tf.exp(-logvariance), axis=-1)

    def transition_log_prob(self, x, previous):
        residual = (x - tf.linalg.matmul(previous, self.transition, transpose_b=True)) / self.sigma
        return _log_standard_normal(residual) - self.dimension * math.log(self.sigma)

    def prior_log_prob(self, x):
        return Chart(tf.zeros([self.dimension], D), tf.linalg.cholesky(self.covariance0)).log_prob(x)


def gaussian_moments(x, weights):
    mean = tf.einsum("n,ni->i", weights, x)
    centered = x - mean
    return mean, tf.einsum("n,ni,nj->ij", weights, centered, centered)


def sgqf_update(model: SVModel, predictive: Chart, observation, clouds):
    """Signed quadrature of g, x*g and xx'*g; no Kalman cross-covariance update.

    Keep every attempted rule, reject nonpositive mass/covariance, and use the
    highest valid predeclared level. Denser-rule discrepancies remain visible.
    """
    records, accepted = [], []
    for level, cloud in clouds:
        x = predictive.forward(cloud.points)
        logg = model.observation_log_prob(x, observation)
        shift = tf.reduce_max(logg)
        signed = cloud.weights * tf.exp(logg - shift)
        mass = tf.reduce_sum(signed)
        record = {"level": level, "points": int(cloud.points.shape[0]),
                  "negative_rule_weights": int(tf.reduce_sum(tf.cast(cloud.weights < 0, tf.int32)).numpy()),
                  "scaled_mass": float(mass.numpy())}
        try:
            finite(mass, "SGQF mass")
            if float(mass.numpy()) <= 0:
                raise ValueError("nonpositive SGQF likelihood integral")
            weights = signed / mass
            mean, cov = gaussian_moments(x, weights)
            chart = Chart.from_moments(mean, cov)
            standardized = chart.inverse(x)
            third = tf.einsum("n,ni->i", weights, standardized ** 3)
            fourth = tf.einsum("n,ni->i", weights, standardized ** 4)
            record.update(status="valid", mean=mean, covariance=cov,
                          standardized_third=third, standardized_fourth=fourth,
                          log_evidence=tf.math.log(mass) + shift)
            accepted.append((chart, record))
        except ValueError as exc:
            record.update(status="invalid", reason=str(exc))
        records.append(record)
    if not accepted:
        raise ValueError(f"all SGQF levels invalid: {records}")
    chosen, info = accepted[-1]
    discrepancy = None
    if len(accepted) > 1:
        previous, _ = accepted[-2]
        discrepancy = float(tf.linalg.norm(chosen.inverse(previous.mean[None, :])).numpy())
    return chosen, {"rules": records, "selected_level": info["level"],
                    "successive_mean_mahalanobis": discrepancy,
                    "log_evidence": info["log_evidence"]}


def build_guide_path(model, observations, levels=(2, 3, 4, 5)):
    clouds = [(level, tf_fixed_sgqf_cloud(model.dimension, level)) for level in levels]
    mean, cov = tf.zeros([model.dimension], D), model.covariance0
    path, records = [], []
    for t, y in enumerate(tf.unstack(observations)):
        if t:
            mean = tf.linalg.matvec(model.transition, mean)
            cov = model.transition @ cov @ tf.transpose(model.transition) + model.sigma**2 * tf.eye(model.dimension, dtype=D)
        predictive = Chart.from_moments(mean, cov)
        posterior, info = sgqf_update(model, predictive, y, clouds)
        mean, cov = posterior.mean, posterior.factor @ tf.transpose(posterior.factor)
        path.append((predictive, posterior))
        records.append(info)
    return path, records


def features(rows, degree):
    return tf.stack([_normalized_hermite_values(rows[:, k], degree)
                     for k in range(int(rows.shape[1]))], axis=1)


def evaluate_cores(cores, feat):
    left = tf.ones([tf.shape(feat)[0], 1], D)
    for k, core in enumerate(cores):
        left = tf.einsum("na,akb,nk->nb", left, core, feat[:, k, :])
    return left[:, 0]


_FITTERS = {}


def compiled_fitter(axes, degree=3, rank=3, sweeps=4, proximal_steps=128, jit_compile=True):
    key = (axes, degree, rank, sweeps, proximal_steps, jit_compile)
    if key in _FITTERS:
        return _FITTERS[key]
    ranks = (1,) + (rank,) * (axes - 1) + (1,)
    shapes = tuple((ranks[i], degree + 1, ranks[i+1]) for i in range(axes))
    signatures = [tf.TensorSpec([None, axes, degree+1], D), tf.TensorSpec([None], D),
                  tf.TensorSpec([], D), tuple(tf.TensorSpec(s, D) for s in shapes)]

    @tf.function(input_signature=signatures, jit_compile=jit_compile, autograph=False)
    def fit(feat, target, penalty, initial):
        cores = list(initial)
        objective_increases = []
        for _ in range(sweeps):
            for axis in range(axes):
                left = tf.ones([tf.shape(feat)[0], 1], D)
                for j in range(axis):
                    left = tf.einsum("na,akb,nk->nb", left, cores[j], feat[:, j, :])
                right = tf.ones([tf.shape(feat)[0], 1], D)
                for j in range(axes - 1, axis, -1):
                    right = tf.einsum("akb,nk,nb->na", cores[j], feat[:, j, :], right)
                design = tf.reshape(tf.einsum("na,nk,nb->nakb", left, feat[:, axis, :], right),
                                    [tf.shape(feat)[0], -1])
                n = tf.cast(tf.shape(feat)[0], D)
                gram = tf.linalg.matmul(design, design, transpose_a=True) / n
                rhs = tf.linalg.matvec(design, target, transpose_a=True) / n
                lipschitz = tf.reduce_max(tf.linalg.eigvalsh(gram))
                old = tf.reshape(cores[axis], [-1])
                def objective(v):
                    return 0.5 * tf.tensordot(v, tf.linalg.matvec(gram, v), 1) - tf.tensordot(v, rhs, 1) + penalty * tf.reduce_sum(tf.abs(v))
                def body(k, value):
                    proposal = value - (tf.linalg.matvec(gram, value) - rhs) / lipschitz
                    value = tf.sign(proposal) * tf.maximum(tf.abs(proposal) - penalty / lipschitz, 0.0)
                    return k + 1, value
                _, value = tf.while_loop(lambda k, _: k < proximal_steps, body,
                                         (tf.constant(0), old), parallel_iterations=1)
                objective_increases.append(objective(value) - objective(old))
                cores[axis] = tf.reshape(value, shapes[axis])
        return tuple(cores), tf.reduce_max(tf.stack(objective_increases))
    _FITTERS[key] = fit
    return fit


def fit_amplitude(log_target: Callable, axes, seed, degree=3, rank=3,
                  rows=1024, sweeps=4, l1_grid=(0.0, 1e-5, 1e-3), jit_compile=True):
    """Fit in L2(standard Gaussian); select L1 on independent validation only."""
    designs = [tf.random.stateless_normal([rows, axes], [seed, split], dtype=D) for split in range(3)]
    logs = [log_target(x) for x in designs]
    for val in logs:
        finite(val, "pulled-back log target")
    # Scale comes only from training rows. Target RMS is one on those rows.
    logscale = tf.reduce_logsumexp(logs[0]) - tf.math.log(tf.cast(rows, D))
    targets = [tf.exp(0.5 * (val - logscale)) for val in logs]
    feats = [features(x, degree) for x in designs]
    ranks = (1,) + (rank,) * (axes-1) + (1,)
    initial = []
    for axis in range(axes):
        shape = (ranks[axis], degree+1, ranks[axis+1])
        c = 0.03 * tf.random.stateless_normal(shape, [seed + 1, axis], dtype=D)
        c = tf.tensor_scatter_nd_add(c, [[0, 0, 0]], [tf.constant(1.0, D)])
        initial.append(c)
    fit = compiled_fitter(axes, degree, rank, sweeps, jit_compile=jit_compile)
    candidates = []
    for penalty in l1_grid:
        cores, increase = fit(feats[0], targets[0], tf.constant(penalty, D), tuple(initial))
        train = tf.sqrt(tf.reduce_mean(tf.square(evaluate_cores(cores, feats[0]) - targets[0])))
        validation = tf.reduce_sum(tf.square(evaluate_cores(cores, feats[1])-targets[1])) / tf.reduce_sum(tf.square(targets[1]))
        finite(validation, "validation residual")
        finite(increase, "core objective")
        if float(increase.numpy()) > 1e-7:
            raise ValueError("proximal core objective increased")
        candidates.append((float(validation.numpy()), cores, {"l1": penalty, "train_rms": train,
                           "validation_relative_squared_error": validation, "maximum_core_objective_increase": increase}))
    selected = min(range(len(candidates)), key=lambda i: candidates[i][0])
    _, cores, _ = candidates[selected]
    audit = tf.sqrt(tf.reduce_sum(tf.square(evaluate_cores(cores, feats[2])-targets[2])) / tf.reduce_sum(tf.square(targets[2])))
    finite(audit, "audit relative RMS")
    return cores, {"selected_l1": l1_grid[selected], "candidates": [c[2] for c in candidates],
                   "audit_relative_rms": audit, "log_target_scale": logscale,
                   "rows_per_split": rows, "seeds": [[seed, j] for j in range(3)],
                   "audit_used_for_selection": False}


@dataclass(frozen=True)
class TTStep:
    cores: tuple[tf.Tensor, ...]
    current_chart: Chart
    conditioning_chart: Chart | None
    tau: tf.Tensor
    fit_diagnostics: dict
    time_index: int

    @property
    def dimension(self):
        return int(self.current_chart.mean.shape[0])

    def retained(self):
        d = self.dimension
        suffix = _paired_right_environments(self.cores[d:], tf.ones([1, 1], D))[0]
        z = _paired_right_environments(self.cores[:d], suffix)[0][0, 0]
        snapshot_hash = hashlib.sha256()
        for value in (*self.cores, self.current_chart.mean, self.current_chart.factor, self.tau):
            snapshot_hash.update(tf.io.serialize_tensor(value).numpy())
        return GaussianHermiteRetainedProposal(
            self.cores[:d], suffix, z, self.tau,
            self.current_chart.mean, self.current_chart.factor, None,
            self.time_index, snapshot_hash.hexdigest(),
        )


def build_tt_path(model, observations, guide_path, *, guided, seed=9100,
                  degree=3, rank=3, rows=1024, sweeps=4, defensive_mass=0.05,
                  jit_compile=True, on_step=None):
    if not 0 < defensive_mass < 1:
        raise ValueError("defensive mass must be in (0,1)")
    result, retained = [], None
    for t, observation in enumerate(tf.unstack(observations)):
        current = guide_path[t][1 if guided else 0]
        condition = None if t == 0 else guide_path[t-1][1]
        d = model.dimension
        def log_target(reference):
            u = reference[:, :d]
            x = current.forward(u)
            logvalue = model.observation_log_prob(x, observation) + current.logdet - _log_standard_normal(u)
            if t == 0:
                return logvalue + model.prior_log_prob(x)
            v = reference[:, d:]
            previous = condition.forward(v)
            return (logvalue + model.transition_log_prob(x, previous)
                    + retained.physical_log_density(previous) + condition.logdet - _log_standard_normal(v))
        cores, info = fit_amplitude(log_target, d if t == 0 else 2*d, seed+t,
                                   degree, rank, rows, sweeps, jit_compile=jit_compile)
        mass = _paired_right_environments(cores, tf.ones([1, 1], D))[0][0, 0]
        finite(mass, "fitted TT mass")
        if float(mass.numpy()) <= 0:
            raise ValueError("nonpositive TT mass")
        tau = mass * defensive_mass / (1-defensive_mass)
        step = TTStep(cores, current, condition, tau, info, t)
        retained = step.retained()
        result.append(step)
        if on_step:
            on_step(t, info)
    return result


def conditional_suffix_environment(cores, conditioning):
    state = tf.ones([tf.shape(conditioning)[0], 1], D)
    degree = int(cores[0].shape[1])-1 if cores else 0
    for axis in range(len(cores)-1, -1, -1):
        basis = _normalized_hermite_values(conditioning[:, axis], degree)
        state = tf.einsum("akb,nk,nb->na", cores[axis], basis, state)
    return tf.einsum("na,nb->nab", state, state)


_SAMPLERS = {}


def compiled_conditional_sampler(step, jit_compile=True):
    d = step.dimension
    shapes = tuple(tuple(int(v) for v in c.shape) for c in step.cores)
    key = (shapes, d, jit_compile)
    if key in _SAMPLERS:
        return _SAMPLERS[key]
    # Passing cores/charts as tensors prevents one trace per observation.
    @tf.function(input_signature=[tuple(tf.TensorSpec(s, D) for s in shapes),
        tf.TensorSpec([], D), tf.TensorSpec([None, len(shapes)-d], D),
        tf.TensorSpec([None], D), tf.TensorSpec([None, d], D), tf.TensorSpec([None, d], D)],
        jit_compile=jit_compile, autograph=False)
    def sample(cores, tau, conditioning, mixture_u, uniforms, normal_noise):
        suffix = conditional_suffix_environment(cores[d:], conditioning)
        left_mass = tf.ones([1, 1], D)
        for core in cores[:d]:
            left_mass = tf.einsum("ac,akb,ckd->bd", left_mass, core, core)
        z = tf.einsum("ab,nab->n", left_mass, suffix)
        inverse = inverse_hermite_polynomial_kr(cores[:d], suffix, uniforms, reverse=True)
        selected = mixture_u < z/(z+tau)
        u = tf.where(selected[:, None], inverse["reference_points"], normal_noise)
        vector = _prefix_row_vectors(cores[:d], u)
        quadratic = tf.einsum("na,nab,nb->n", vector, suffix, vector)
        logq = tf.math.log(quadratic+tau) + _log_standard_normal(u) - tf.math.log(z+tau)
        return u, logq, inverse["maximum_inverse_cdf_residual"], inverse["cdf_bracket_valid"], inverse["finite"], tf.reduce_min(z)
    _SAMPLERS[key] = sample
    return sample


def sample_tt_step(step, previous, seed, jit_compile=True):
    n, d = int(previous.shape[0]), step.dimension
    condition = (tf.zeros([n, 0], D) if step.conditioning_chart is None
                 else step.conditioning_chart.inverse(previous))
    # Independent pseudo-random innovations, never assigned sorted quantiles.
    mixture_u = tf.random.stateless_uniform([n], [seed, 0], dtype=D)
    u = tf.random.stateless_uniform([n, d], [seed, 1], minval=EPS, maxval=1-EPS, dtype=D)
    noise = tf.random.stateless_normal([n, d], [seed, 2], dtype=D)
    sampler = compiled_conditional_sampler(step, jit_compile)
    reference, logq, residual, bracket, valid, mass = sampler(step.cores, step.tau, condition, mixture_u, u, noise)
    if not bool((bracket & valid).numpy()) or float(residual.numpy()) > 1e-9:
        raise ValueError("conditional Hermite KR validity failure")
    x = step.current_chart.forward(reference)
    logq = logq - step.current_chart.logdet
    finite(x, "proposal draw")
    finite(logq, "actual proposal density")
    return x, logq, {"cdf_residual": residual, "minimum_polynomial_conditional_mass": mass}

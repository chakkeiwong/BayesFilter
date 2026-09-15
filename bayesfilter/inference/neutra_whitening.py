"""TF finite-bank NeuTra whitening diagnostics with correct proposal semantics.

The typical Gaussian proposal supports self-normalized posterior moments;
fixed-radius shells do not. Tail/mixture importance diagnostics cannot veto an
otherwise exact Gaussian map. Statistics alone cannot admit a frozen transport:
the separate integrity path checks inverse, Jacobian, target replay and local
projected precision. Neither path certifies global Gaussianity or convergence.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from .neutra_artifacts import load_frozen_neutra_artifact


@dataclass(frozen=True)
class WhiteningPolicy:
    density_rms: tuple[float, float, float] = (.5, .5, 1.)
    score_rms: tuple[float, float, float] = (.5, .5, 1.)
    coordinate_rms: tuple[float, float, float] = (1., 1., 1.)
    weighted_score_rms: float = .5
    weighted_coordinate_rms: float = 1.
    ess_fraction: float = .2
    mean_max: float = .2
    eigen_min: float = .5
    eigen_max: float = 2.
    condition_max: float = 4.
    offdiagonal_max: float = .35
    replay_tolerance: float = 1e-8
    score_tolerance: float = 1e-7
    curvature_relative_tolerance: float = .05
    curvature_steps: tuple[float, float] = (.001, .0005)
    curvature_directions: int = 16
    curvature_points: int = 8

    def __post_init__(self):
        for limits in (self.density_rms, self.score_rms, self.coordinate_rms):
            if len(limits) != 3 or any(not math.isfinite(x) or x <= 0 for x in limits):
                raise ValueError('three finite positive bank thresholds required')
        if not 0 < self.eigen_min < self.eigen_max or not 0 < self.ess_fraction <= 1:
            raise ValueError('invalid eigenvalue or ESS limits')
        if min(self.curvature_directions, self.curvature_points) <= 0:
            raise ValueError('positive bounded curvature shape required')


@dataclass(frozen=True)
class WhiteningBank:
    """A reproducible actual proposal, never just a caller-supplied log_q."""

    kind: str
    seed: tuple[int, int]
    role: str
    z: tf.Tensor
    log_proposal: tf.Tensor | None

    def identity(self):
        return {'kind': self.kind, 'seed': list(self.seed), 'role': self.role,
                'rows': int(self.z.shape[0]), 'dimension': int(self.z.shape[1]),
                'tensor_sha256': hashlib.sha256(tf.io.serialize_tensor(self.z).numpy()).hexdigest()}


@dataclass(frozen=True)
class WhiteningTargetBatch:
    value: Any
    score: Any
    eligible: Any
    target_signature: str


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _finite(*values):
    return all(bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in values)


def _native(value):
    return value.numpy().tolist()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _normal_log_prob(z, sd):
    dimension = tf.cast(tf.shape(z)[1], tf.float64)
    return -.5*dimension*tf.constant(math.log(2*math.pi), tf.float64)-dimension*tf.math.log(
        tf.constant(sd, tf.float64))-.5*tf.reduce_sum(z*z, axis=1)/(sd*sd)


def proposal_log_prob(kind, z):
    """Density with respect to Lebesgue measure; the shell has none."""
    if kind == 'typical':
        return _normal_log_prob(z, 1.)
    if kind == 'tail':
        return _normal_log_prob(z, 2.)
    if kind == 'mixture':
        logs = tf.stack((_normal_log_prob(z, 1.), _normal_log_prob(z, 2.)), axis=1)
        return tf.reduce_logsumexp(logs, axis=1)-tf.constant(math.log(2), tf.float64)
    if kind == 'shell':
        return None
    raise ValueError('unknown whitening proposal')


def make_whitening_bank(*, kind, dimension, rows, seed, role):
    """Make IID normals, independent uniform sphere directions, or a mixture.

    The mixture draws one independent Bernoulli component per row; normalized
    shell draws live on a sphere and cannot estimate full-posterior moments.
    """
    _require(dimension > 0 and rows > 1 and len(seed) == 2 and bool(role), 'invalid bank dimensions/seed/role')
    if kind not in ('typical', 'shell', 'tail', 'mixture'):
        raise ValueError('unknown whitening proposal')
    seed = tuple(seed)
    z = tf.random.stateless_normal([rows, dimension], seed, dtype=tf.float64)
    if kind == 'shell':
        lengths = tf.linalg.norm(z, axis=1, keepdims=True)
        _require(bool(tf.reduce_all(lengths > 0).numpy()), 'zero shell direction')
        z = z/lengths*tf.sqrt(tf.constant(dimension, tf.float64))
    elif kind == 'tail':
        z = 2*z
    elif kind == 'mixture':
        selector_seed = tf.random.experimental.stateless_fold_in(tf.constant(seed, tf.int32), 1)
        select = tf.random.stateless_uniform([rows, 1], selector_seed, dtype=tf.float64) < .5
        z = tf.where(select, z, 2*z)
    return WhiteningBank(kind, seed, role, z, proposal_log_prob(kind, z))


def validate_whitening_bank(bank):
    _require(isinstance(bank, WhiteningBank) and bank.z.dtype == tf.float64
             and bank.z.shape.rank == 2 and _finite(bank.z), 'invalid/nonfinite whitening bank')
    recreated = make_whitening_bank(kind=bank.kind, dimension=int(bank.z.shape[1]),
        rows=int(bank.z.shape[0]), seed=bank.seed, role=bank.role)
    _require(bool(tf.reduce_all(bank.z == recreated.z).numpy()), 'bank bytes differ from declared proposal/seed')
    if bank.kind == 'shell':
        _require(bank.log_proposal is None, 'shell cannot carry a posterior importance denominator')
    else:
        q = tf.convert_to_tensor(bank.log_proposal, tf.float64)
        _require(q.shape == recreated.log_proposal.shape and _finite(q)
                 and bool(tf.reduce_all(q == recreated.log_proposal).numpy()), 'wrong proposal density')


class WhiteningBankLedger:
    """Persistent no-reuse ledger; reservations survive incomplete evaluations.

    The surrounding phase controller owns process exclusivity and lineage.
    Each bank reservation is an ordinary exclusive JSON file. Observed banks
    are never made fresh by changing the map name, role label or output path.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def reserve(self, bank, *, map_id, target_signature, purpose='qualification'):
        validate_whitening_bank(bank)
        # Exclude the cosmetic role and map name: neither makes consumed data
        # independent again. Campaigns must share this ledger across attempts.
        identity = bank.identity()
        # Reusing the same RNG stream with a different proposal, dimension or
        # row count also destroys independence (or extends an inspected bank).
        key = _digest({'seed': identity['seed']})
        record = {**identity, 'map_id': map_id, 'target_signature': target_signature, 'purpose': purpose}
        try:
            with (self.directory/(key+'.json')).open('x') as f:
                json.dump(record, f, indent=2, sort_keys=True, allow_nan=False)
                f.write('\n')
        except FileExistsError as error:
            raise ValueError('qualification bank was already reserved or inspected') from error
        return key


def _matrix_summary(matrix):
    symmetric = .5*(matrix+tf.transpose(matrix))
    eigen = tf.linalg.eigvalsh(symmetric)
    diagonal = tf.linalg.diag(tf.linalg.diag_part(symmetric))
    denominator = tf.linalg.norm(diagonal)
    _require(_finite(eigen) and float(denominator.numpy()) > 0, 'degenerate covariance/precision')
    minimum, maximum = float(eigen[0].numpy()), float(eigen[-1].numpy())
    return {'minimum_eigenvalue': minimum, 'maximum_eigenvalue': maximum,
            'condition': maximum/minimum if minimum > 0 else None,
            'offdiagonal_ratio': float((tf.linalg.norm(symmetric-diagonal)/denominator).numpy())}


def _matrix_pass(summary, policy):
    return (summary['minimum_eigenvalue'] >= policy.eigen_min
            and summary['maximum_eigenvalue'] <= policy.eigen_max
            and summary['condition'] is not None and summary['condition'] <= policy.condition_max
            and summary['offdiagonal_ratio'] <= policy.offdiagonal_max)


def summarize_whitening_bank(bank, log_density, total_score, eligible, *, policy):
    """Compute proposal-aware TF statistics, without claiming map integrity.

    d=ell_z+||z||^2/2 is centered by the unweighted bank mean. For typical
    weights w, score RMS=sqrt(sum_i w_i mean_j r_ij^2), r=score_z+z. This
    averages coordinates once. All rows must be eligible; none are dropped.
    """
    validate_whitening_bank(bank)
    z = bank.z
    ell = tf.convert_to_tensor(log_density, tf.float64)
    score = tf.convert_to_tensor(total_score, tf.float64)
    valid = tf.convert_to_tensor(eligible, tf.bool)
    _require(ell.shape == z.shape[:-1] and score.shape == z.shape and valid.shape == ell.shape,
             'whitening value/score/eligibility shape mismatch')
    _require(_finite(ell, score) and bool(tf.reduce_all(valid).numpy()), 'every whitening row must be finite and eligible')
    residual = score+z
    density = ell+.5*tf.reduce_sum(z*z, axis=1)
    centered = density-tf.reduce_mean(density)
    density_rms = tf.sqrt(tf.reduce_mean(centered**2))
    coordinate_rms = tf.sqrt(tf.reduce_mean(residual**2, axis=0))
    rms = tf.sqrt(tf.reduce_mean(residual**2))
    _require(_finite(density, centered, density_rms, coordinate_rms, rms), 'nonfinite residual statistic')
    result = {'bank': bank.identity(), 'density_rms': float(density_rms.numpy()),
              'score_rms': float(rms.numpy()), 'coordinate_rms': _native(coordinate_rms),
              'maximum_coordinate_rms': float(tf.reduce_max(coordinate_rms).numpy()),
              'integrity_checked': False, 'posterior_moments_applied': bank.kind == 'typical'}
    # The optional mixture is explanatory only. Use tail residual limits to
    # report its descriptive comparisons, never return admission from it.
    index = {'typical': 0, 'shell': 1, 'tail': 2, 'mixture': 2}[bank.kind]
    gates = {'density': result['density_rms'] <= policy.density_rms[index],
             'score': result['score_rms'] <= policy.score_rms[index],
             'coordinate_score': result['maximum_coordinate_rms'] <= policy.coordinate_rms[index]}
    if bank.kind != 'shell':
        log_weights = ell-bank.log_proposal
        _require(_finite(log_weights), 'nonfinite importance log weights')
        weights = tf.nn.softmax(log_weights)
        _require(_finite(weights) and bool(tf.reduce_any(weights > 0).numpy()), 'degenerate importance weights')
        ess = 1./tf.reduce_sum(weights**2)
        mean = tf.reduce_sum(weights[:, None]*z, axis=0)
        centered_z = z-mean
        covariance = tf.matmul(centered_z, weights[:, None]*centered_z, transpose_a=True)
        weighted_coordinate = tf.sqrt(tf.reduce_sum(weights[:, None]*residual**2, axis=0))
        weighted_rms = tf.sqrt(tf.reduce_sum(weights*tf.reduce_mean(residual**2, axis=1)))
        # A tail proposal can concentrate all representable weight on one row,
        # even for an exact Gaussian map. Undefined descriptive covariance is
        # reported, not smuggled into a tail continuation veto.
        covariance_summary = None
        if bank.kind == 'typical' or float(tf.linalg.norm(tf.linalg.diag_part(covariance)).numpy()) > 0:
            covariance_summary = _matrix_summary(covariance)
        importance = {'ess': float(ess.numpy()), 'ess_fraction': float((ess/int(z.shape[0])).numpy()),
                      'mean_max': float(tf.reduce_max(tf.abs(mean)).numpy()), 'mean': _native(mean),
                      'weighted_score_rms': float(weighted_rms.numpy()),
                      'maximum_weighted_coordinate_rms': float(tf.reduce_max(weighted_coordinate).numpy()),
                      'covariance': covariance_summary, 'descriptive_only': bank.kind != 'typical'}
        result['importance'] = importance
        if bank.kind == 'typical':
            gates.update(ess=importance['ess_fraction'] >= policy.ess_fraction,
                         mean=importance['mean_max'] <= policy.mean_max,
                         covariance=_matrix_pass(importance['covariance'], policy),
                         weighted_score=importance['weighted_score_rms'] <= policy.weighted_score_rms,
                         weighted_coordinate_score=importance['maximum_weighted_coordinate_rms'] <= policy.weighted_coordinate_rms)
    result.update(gates=gates, statistics_passed=all(gates.values()) and bank.kind != 'mixture',
                  optional_diagnostic=bank.kind == 'mixture', policy=asdict(policy))
    return result


def qualify_frozen_whitening_bank(payload, evaluator, bank, *, policy, ledger, curvature_seed, chunk_rows=32):
    """Check a canonical frozen map, then its statistical and curvature gates.

    evaluator(theta) returns explicit target signature and row eligibility.
    Every call is batched and counted before invocation. Jacobian construction
    explicitly disables pfor and runs outside XLA. The caller's phase owns
    resource caps, source/data lineage, fresh-process replay and replicate/map
    aggregation; this result alone cannot promote a research campaign.
    """
    validate_whitening_bank(bank)
    _require(bank.kind != 'mixture', 'mixture is an optional diagnostic, not a qualification bank')
    signature = payload['target_signature']
    frozen = load_frozen_neutra_artifact(payload, expected_target_signature=signature).transport
    repeated = load_frozen_neutra_artifact(payload, expected_target_signature=signature).transport
    dimension = int(bank.z.shape[1])
    _require(frozen.parameter_dim == dimension and chunk_rows > 1, 'map/bank dimension or batch shape mismatch')
    key = ledger.reserve(bank, map_id=payload['transport_id'], target_signature=signature)
    result = {'passed': False, 'integrity_checked': False, 'bank_reservation': key, 'target_rows': 0, 'target_calls': 0}

    def evaluate(theta):
        result['target_rows'] += int(theta.shape[0])
        result['target_calls'] += 1
        batch = evaluator(tf.identity(theta))
        _require(isinstance(batch, WhiteningTargetBatch) and batch.target_signature == signature,
                 'qualification target signature/status object mismatch')
        value = tf.convert_to_tensor(batch.value, tf.float64)
        score = tf.convert_to_tensor(batch.score, tf.float64)
        eligible = tf.convert_to_tensor(batch.eligible, tf.bool)
        _require(value.shape == theta.shape[:-1] and score.shape == theta.shape and eligible.shape == value.shape,
                 'qualification target shape mismatch')
        _require(_finite(value, score) and bool(tf.reduce_all(eligible).numpy()), 'nonfinite/ineligible qualification target')
        return value, score

    def transformed(z):
        theta = frozen.forward_z_to_theta_batch(z)
        value, raw_score = evaluate(theta)
        score = frozen.pullback_score_batch(z, raw_score)+frozen.log_abs_det_jacobian_score_batch(z)
        ell = value+frozen.log_abs_det_jacobian_batch(z)
        _require(_finite(theta, ell, score), 'nonfinite transformed target')
        return ell, score

    try:
        logs, scores = [], []
        min_singular = math.inf
        inverse_error = score_error = 0.
        for start in range(0, int(bank.z.shape[0]), chunk_rows):
            z = bank.z[start:start+chunk_rows]
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(z)
                theta = frozen.forward_z_to_theta_batch(z)
                logdet = frozen.log_abs_det_jacobian_batch(z)
            # Batch-native map independence is guaranteed by the loaded IAF
            # topology. Non-pfor derivatives allocate [chunk,D,D], not N^2 D^2.
            jacobian = tape.batch_jacobian(theta, z, experimental_use_pfor=False, parallel_iterations=1)
            del tape
            singular = tf.linalg.svd(jacobian, compute_uv=False)
            _require(_finite(theta, logdet, singular) and bool(tf.reduce_all(singular > 0).numpy()),
                     'invalid finite-positive Jacobian')
            min_singular = min(min_singular, float(tf.reduce_min(singular).numpy()))
            inverse = repeated.inverse_theta_to_z_batch(theta)
            tf.debugging.assert_near(inverse, z, atol=policy.replay_tolerance, rtol=policy.replay_tolerance)
            inverse_error = max(inverse_error, float(tf.reduce_max(tf.abs(inverse-z)).numpy()))
            replay_theta = repeated.forward_z_to_theta_batch(z)
            tf.debugging.assert_near(replay_theta, theta, atol=policy.replay_tolerance, rtol=policy.replay_tolerance)
            replay_logdet = repeated.log_abs_det_jacobian_batch(z)
            tf.debugging.assert_near(replay_logdet, logdet, atol=policy.replay_tolerance, rtol=policy.replay_tolerance)
            svd_logdet = tf.reduce_sum(tf.math.log(singular), axis=1)
            tf.debugging.assert_near(svd_logdet, logdet, atol=policy.replay_tolerance, rtol=policy.replay_tolerance)
            value, raw_score = evaluate(theta)
            replay_value, replay_raw_score = evaluate(replay_theta)
            tf.debugging.assert_near(replay_value, value, atol=policy.replay_tolerance, rtol=policy.replay_tolerance)
            tf.debugging.assert_near(replay_raw_score, raw_score, atol=policy.score_tolerance, rtol=policy.score_tolerance)
            analytical = frozen.pullback_score_batch(z, raw_score)+frozen.log_abs_det_jacobian_score_batch(z)
            with tf.GradientTape() as tape:
                tape.watch(z)
                live_theta = repeated.forward_z_to_theta_batch(z)
                surrogate = tf.reduce_sum(live_theta*tf.stop_gradient(raw_score), axis=1)
                surrogate += repeated.log_abs_det_jacobian_batch(z)
            automatic = tape.gradient(surrogate, z, output_gradients=tf.ones_like(surrogate))
            tf.debugging.assert_near(analytical, automatic, atol=policy.score_tolerance, rtol=policy.score_tolerance)
            score_error = max(score_error, float(tf.reduce_max(tf.abs(analytical-automatic)).numpy()))
            logs.append(value+logdet)
            scores.append(analytical)
        stats = summarize_whitening_bank(bank, tf.concat(logs, 0), tf.concat(scores, 0),
            tf.ones([int(bank.z.shape[0])], tf.bool), policy=policy)
        k = min(policy.curvature_directions, dimension)
        directions, _ = tf.linalg.qr(tf.random.stateless_normal([dimension, k], curvature_seed, dtype=tf.float64))
        points = tf.concat((tf.zeros([1, dimension], tf.float64), bank.z[:policy.curvature_points]), axis=0)
        curvature = []
        for point in tf.unstack(points):
            matrices = []
            antisymmetry = []
            for step in policy.curvature_steps:
                offsets = tf.transpose(directions)*step
                _vp, gp = transformed(point[None, :]+offsets)
                _vm, gm = transformed(point[None, :]-offsets)
                # Row i is minus the directional score difference projected on
                # each direction: Q.T precision Q, up to a harmless transpose.
                raw = -tf.matmul((gp-gm)/(2*step), directions)
                sym = .5*(raw+tf.transpose(raw))
                norm = float(tf.linalg.norm(sym).numpy())
                _require(norm > 0 and _finite(sym), 'degenerate projected precision')
                antisymmetry.append(float((tf.linalg.norm(raw-tf.transpose(raw))/2).numpy())/norm)
                matrices.append(sym)
            relative = float((tf.linalg.norm(matrices[0]-matrices[1])/tf.linalg.norm(matrices[1])).numpy())
            summaries = [_matrix_summary(matrix) for matrix in matrices]
            curvature.append({'spectra': summaries, 'relative_step_difference': relative,
                              'relative_antisymmetry': antisymmetry,
                              'passed': all(_matrix_pass(summary, policy) for summary in summaries)
                              and relative <= policy.curvature_relative_tolerance
                              and max(antisymmetry) <= policy.curvature_relative_tolerance})
        result.update(statistics=stats, curvature=curvature, integrity_checked=True,
                      minimum_jacobian_singular_value=min_singular, inverse_error=inverse_error,
                      score_replay_error=score_error,
                      passed=stats['statistics_passed'] and all(c['passed'] for c in curvature))
    except Exception as error:  # noqa: BLE001 -- numerical/target failures produce an immutable nonpass.
        result['error'] = f'{type(error).__name__}: {error}'
    return result

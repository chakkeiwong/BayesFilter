"""Opt-in, physical-width guarded training of canonical three-stage IAFs.

For row outputs, d_w=(T_new(z)-T_old(z)) L0^-T. L0 is the fixed initial
position-factor ruler, not part of the learned map. Adam proposals are damped
against finite-bank bounds; this is not a global trust region or a convergence
certificate. Numerical kernels use the configured TF/XLA execution policy;
process dispatch, transaction decisions and artifact work stay on the host.

The host transaction reuses NeuTraReverseKLTrainer's raw detached-score kernel,
but never calls its legacy update methods or optimizer. The caller supplies a
batch-native evaluator and binds its complete source/data lineage in its run
manifest. For durable runs, event_sink must persist target reservations and
terminal records synchronously; reconcile an unfinished reservation on recovery.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from . import neutra_artifacts, neutra_scale_aware_kernels, neutra_training, neutra_transport_core


@dataclass(frozen=True)
class ScaleAwareUpdatePolicy:
    """Explicit engineering budgets in initial Mahalanobis width units.

    RMS averages squared row Euclidean norms once, without dividing by D.
    p95 uses nearest rank; the zero row is separate from the Gaussian bank.
    """

    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    gradient_clip_norm: float = 10.0
    zero_limit: float = 0.02
    rms_limit: float = 0.05
    quantile_limit: float = 0.10
    maximum_limit: float = 0.20
    log_scale_limit: float = 0.02
    guard_rows: int = 64
    release_rows: int = 32
    maximum_halvings: int = 30
    release_boundaries: tuple[int, int] = (100, 200)
    training_batch_size: int = 32
    replay_tolerance: float = 1e-8
    score_tolerance: float = 1e-7

    def __post_init__(self):
        positive = (self.learning_rate, self.epsilon, self.gradient_clip_norm,
                    self.zero_limit, self.rms_limit, self.quantile_limit,
                    self.maximum_limit, self.log_scale_limit, self.replay_tolerance,
                    self.score_tolerance)
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError('finite positive scale-aware budgets required')
        if not 0 < self.beta1 < 1 or not 0 < self.beta2 < 1:
            raise ValueError('Adam beta values must lie strictly between zero and one')
        if min(self.guard_rows, self.release_rows, self.training_batch_size) <= 1:
            raise ValueError('training and probe banks require more than one row')
        if not 0 <= self.maximum_halvings <= 30:
            raise ValueError('backtracking must be bounded by at most 30 halvings')
        a, b = self.release_boundaries
        if a < 100 or b-a < 100:
            raise ValueError('each protected release interval requires at least 100 accepted updates')


@dataclass(frozen=True)
class ScaleAwareTargetBatch:
    """Detached values/scores at exactly theta in the declared target order.

    Eligibility is explicit: finite values alone cannot authorize use of a
    numerically rejected filter/likelihood result. Padding is also finite and
    eligible; row counts exclude it only from the gradient/loss reductions.
    """

    theta: Any
    value: Any
    score: Any
    eligible: Any
    target_signature: str


class ScaleAwareReleaseRequired(RuntimeError):
    """A frozen-map release check is due before another training update."""


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _finite(*values):
    return all(bool(tf.reduce_all(tf.math.is_finite(tf.cast(x, tf.float64))).numpy()) for x in values)


def _native(value):
    return value.numpy().tolist()


def _json_value(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _snapshot(variables):
    return tuple(tf.identity(x) for x in variables)


def _assign(variables, values):
    for variable, value in zip(variables, values, strict=True):
        variable.assign(value)


def _specs(variables):
    return [{'shape': list(x.shape), 'dtype': str(x.dtype)} for x in variables]


def width_displacement_metrics(before, after, lower, scales_before, scales_after):
    """Measure d_theta L0^-T using a lower triangular solve, never an inverse.

    Input row zero is the origin, followed by the nonzero probe bank. Scales
    concatenate stages evaluated at their own actual composed inputs. A wrong
    transpose loses correlation protection even if diagonal fixtures pass.
    """
    _require(_finite(before, after, lower, scales_before, scales_after), 'nonfinite width-guard input')
    delta = tf.transpose(tf.linalg.triangular_solve(lower, tf.transpose(after-before)))
    lengths = tf.linalg.norm(delta, axis=-1)
    rows = lengths[1:]
    count = int(rows.shape[0])
    _require(count > 1, 'width bank must contain zero and more than one probe')
    values = (lengths[0], tf.sqrt(tf.reduce_mean(tf.square(rows))),
              tf.sort(rows)[math.ceil(.95*count)-1], tf.reduce_max(rows),
              tf.reduce_max(tf.abs(scales_after-scales_before)))
    _require(_finite(*values), 'nonfinite standardized displacement')
    return dict(zip(('zero', 'rms', 'p95', 'maximum', 'log_scale'),
                    (float(v.numpy()) for v in values), strict=True))


class ScaleAwareNeuTraTrainer:
    """One guarded transaction for direct, external-score and chunked updates.

    evaluator(theta) must return ScaleAwareTargetBatch from a batch-native
    target. Training/release/guard streams must be distinct. A rejected update
    restores every map/Adam value, keeps work charged and stops this instance.
    Loading a stopped checkpoint preserves that stop; it is not a retry path.
    """

    def __init__(self, target, initializer_config, *, policy: ScaleAwareUpdatePolicy,
                 evaluator: Callable[[tf.Tensor], ScaleAwareTargetBatch],
                 training_seed: Sequence[int], guard_seed: Sequence[int],
                 release_seeds: Sequence[Sequence[int]], event_sink: Callable | None = None):
        cfg = initializer_config
        _require(cfg.initialization_mode == 'quadratic_triangular_anchor_v1'
                 and cfg.stages == 3 and cfg.scale_transform == 'identity'
                 and cfg.anchor_release_steps == 0 and cfg.gradient_clip_mode == 'none',
                 'guarded trainer requires the exact three-IAF direct-log raw-gradient initializer')
        self.policy = policy
        self.evaluator = evaluator
        self.event_sink = event_sink
        self.training_seed = tuple(training_seed)
        self.guard_seed = tuple(guard_seed)
        self.release_seeds = tuple(tuple(seed) for seed in release_seeds)
        streams = (self.training_seed, self.guard_seed, *self.release_seeds)
        _require(len(self.release_seeds) == 2 and all(len(s) == 2 for s in streams)
                 and len(set(streams)) == 4, 'four distinct stateless training/guard/release seed pairs required')
        self._base = neutra_training.NeuTraReverseKLTrainer(target, cfg)
        self.variables = self._base.variables
        self.variable_keys = self._base.transport.variable_keys
        self.dimension = cfg.dimension
        self.target_signature = cfg.target_signature
        self.lower = tf.constant(cfg.initial_anchor_factor, tf.float64)
        _require(_finite(self.lower) and self.lower.shape == (self.dimension, self.dimension)
                 and bool(tf.reduce_all(tf.linalg.diag_part(self.lower) > 0).numpy())
                 and bool(tf.reduce_all(self.lower == tf.linalg.band_part(self.lower, -1, 0)).numpy()),
                 'finite positive-diagonal lower position ruler required')
        self.variable_stages = tuple(int(key.split(']')[0].split('[')[1])//2+1 for key in self.variable_keys)
        _require(len(self.variables) == 27 and set(self.variable_stages) == {1, 2, 3},
                 'unexpected effectful variable inventory')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=policy.learning_rate,
            beta_1=policy.beta1, beta_2=policy.beta2, epsilon=policy.epsilon)
        self.optimizer.build(self.variables)
        self.accepted_updates = 0
        self.attempted_updates = 0
        self.draw_counter = 0
        self.target_rows = 0
        self.external_target_rows = 0
        self.active_stages = 1
        self.preflight_passed = False
        self.stopped = None
        self.records = []
        self.releases = []
        self.events = []
        self._last_before = None
        self.guard = self._probe_bank(self.guard_seed, policy.guard_rows)
        # These are deterministic artifact-preparation banks. XLA and ordinary
        # TF implement different stateless-normal streams for the same seed;
        # prepare the admitted CPU stream once, outside numerical transactions.
        self._release_banks = tuple(self._probe_bank(seed, policy.release_rows) for seed in self.release_seeds)
        self._prepared_training = None
        self._identity = _json_value({
            'initializer': cfg.manifest_payload(), 'policy': asdict(policy),
            'training_seed': self.training_seed, 'guard_seed': self.guard_seed,
            'release_seeds': self.release_seeds, 'tensorflow': tf.__version__,
            'sources': {Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                        for p in (__file__, neutra_training.__file__, neutra_artifacts.__file__,
                                  neutra_scale_aware_kernels.__file__, neutra_transport_core.__file__)}})
        shape = tf.TensorSpec([None, self.dimension], tf.float64)
        self._kernels = neutra_scale_aware_kernels.ScaleAwareNumericalKernels(self)
        self._map_graph = self._kernels.map
        self._gradient_graph = tf.function(self._base._external_gradients_impl,
            input_signature=[shape, tf.TensorSpec([None], tf.float64), shape, tf.TensorSpec([None], tf.float64)],
            autograph=False, jit_compile=bool(cfg.jit_compile))

    def _probe_bank(self, seed, count):
        return tf.concat((tf.zeros([1, self.dimension], tf.float64),
                          tf.random.stateless_normal([count, self.dimension], seed, dtype=tf.float64)), axis=0)

    def _map_impl(self, z):
        return self._kernels._map_impl(z)

    def _map(self, z):
        z = tf.convert_to_tensor(z, tf.float64)
        _require(z.shape.rank == 2 and z.shape[1] == self.dimension, 'invalid latent batch')
        output = self._map_graph(z)
        _require(bool(output[3].numpy()), 'nonfinite or noninvertible map')
        return output[:3]

    def forward_and_logdet(self, z):
        theta, ld, _scale = self._map(z)
        return theta, ld

    def _event(self, payload):
        event = {'index': len(self.events), **payload}
        if self.event_sink is not None:
            self.event_sink(_json_value(event))
        self.events.append(event)

    def _check_batch(self, theta, batch):
        _require(isinstance(batch, ScaleAwareTargetBatch), 'explicit target eligibility batch required')
        _require(batch.target_signature == self.target_signature, 'target signature mismatch')
        supplied_theta = tf.convert_to_tensor(batch.theta, tf.float64)
        value = tf.convert_to_tensor(batch.value, tf.float64)
        score = tf.convert_to_tensor(batch.score, tf.float64)
        eligible = tf.convert_to_tensor(batch.eligible, tf.bool)
        _require(supplied_theta.shape == theta.shape and value.shape == theta.shape[:-1]
                 and score.shape == theta.shape and eligible.shape == theta.shape[:-1], 'target batch shape mismatch')
        _require(bool(self._kernels.check_batch(theta, supplied_theta, value, score, eligible).numpy()),
                 'nonfinite, ineligible, stale or reordered target batch')
        return value, score

    def _evaluate(self, theta, role):
        count = int(theta.shape[0])
        self.target_rows += count
        self._event({'kind': 'target_reservation', 'role': role, 'rows': count,
                     'cumulative_rows': self.target_rows, 'accepted_updates': self.accepted_updates})
        batch = self.evaluator(tf.identity(theta))
        result = self._check_batch(theta, batch)
        self._event({'kind': 'target_complete', 'role': role, 'rows': count})
        return result

    def _inside(self, metrics):
        p = self.policy
        return all(metrics[key] <= limit for key, limit in zip(
            ('zero', 'rms', 'p95', 'maximum', 'log_scale'),
            (p.zero_limit, p.rms_limit, p.quantile_limit, p.maximum_limit, p.log_scale_limit), strict=True))

    def frozen_transport_payload(self, *, transport_id):
        return self._base.frozen_transport_payload(transport_id=transport_id,
                                                   target_signature=self.target_signature)

    def _frozen(self):
        return neutra_artifacts.load_frozen_neutra_artifact(
            self.frozen_transport_payload(transport_id='guarded-frozen-replay'),
            expected_target_signature=self.target_signature).transport

    def _inverse_check(self, probes, frozen=None):
        if frozen is None:
            valid = self._kernels.inverse(probes)
            frozen = self._kernels._frozen
        else:
            valid = self._kernels.check_external_inverse(probes, frozen)
        _require(bool(valid.numpy()), 'nonfinite or inconsistent frozen inverse replay')
        return frozen

    def _replay(self, probes, role):
        self._inverse_check(probes)
        theta, _ld = self.forward_and_logdet(probes)
        value, score = self._evaluate(theta, role+'_live')
        replay_theta, _replay_ld = self._kernels.frozen_map(probes)
        replay_value, replay_score = self._evaluate(replay_theta, role+'_frozen')
        _require(bool(self._kernels.replay(probes, score, replay_score, value, replay_value).numpy()),
                 'inconsistent transformed value or total-score replay')

    def preflight(self):
        """Verify the exact initial factor and eligible target before updating."""
        if self.preflight_passed:
            return
        _require(self.stopped is None and self.accepted_updates == 0, 'preflight requires fresh active state')
        try:
            theta, _ld = self.forward_and_logdet(self.guard)
            c = tf.constant(self._base.config.initial_output_shift, tf.float64)
            standardized = tf.transpose(tf.linalg.triangular_solve(self.lower, tf.transpose(theta-c)))
            tf.debugging.assert_near(standardized, self.guard, atol=self.policy.replay_tolerance,
                                     rtol=self.policy.replay_tolerance)
            self._replay(self.guard, 'preflight')
            self.preflight_passed = True
        except Exception as error:
            self.stopped = f'preflight: {type(error).__name__}: {error}'
            raise

    def _ready(self):
        _require(self.stopped is None, f'trainer stopped: {self.stopped}')
        if self.active_stages < 3 and self.accepted_updates == self.policy.release_boundaries[self.active_stages-1]:
            raise ScaleAwareReleaseRequired('independent frozen release required before next update')
        self.preflight()

    def prepare_training_stream(self, total_updates):
        """Prepare original TF stateless draws without changing any owned state.

        This is artifact/data preparation, outside the XLA numerical hot path.
        A seed alone does not make eager and XLA RNG interchangeable. Keeping
        the original latent tensors also makes checkpoint continuation exact.
        The cache is reproducible from seeds and need not be a checkpoint slot.
        """
        _require(total_updates >= self.draw_counter, 'prepared stream ends before current draw counter')
        stream = []
        for counter in range(total_updates):
            seed = tf.random.experimental.stateless_fold_in(tf.constant(self.training_seed, tf.int32), counter)
            stream.append(tf.random.stateless_normal([self.policy.training_batch_size, self.dimension],
                                                     seed, dtype=tf.float64))
        self._prepared_training = tuple(stream)

    def train_step(self):
        """Consume one owned latent batch and execute one guarded transaction."""
        self._ready()
        if self._prepared_training is not None:
            _require(self.draw_counter < len(self._prepared_training), 'prepared training stream exhausted')
            z = self._prepared_training[self.draw_counter]
        else:
            _require(not self._base.config.jit_compile,
                     'XLA training requires prepare_training_stream to preserve the admitted random stream')
            seed = tf.random.experimental.stateless_fold_in(tf.constant(self.training_seed, tf.int32), self.draw_counter)
            z = tf.random.stateless_normal([self.policy.training_batch_size, self.dimension], seed, dtype=tf.float64)
        self.draw_counter += 1
        return self._transaction((z,), None, (self.policy.training_batch_size,))

    def train_step_with_external_value_score(self, z, batch: ScaleAwareTargetBatch):
        """Guard an already evaluated batch; stale/reordered theta is rejected."""
        self._ready()
        z = tf.convert_to_tensor(z, tf.float64)
        return self._transaction((z,), (batch,), (int(z.shape[0]),))

    def train_step_with_external_value_score_chunks(self, z_chunks, batches, row_counts):
        """Sum raw fixed-shape chunk gradients before one clip and transaction."""
        self._ready()
        chunks = tuple(tf.convert_to_tensor(z, tf.float64) for z in z_chunks)
        counts = tuple(row_counts)
        _require(bool(chunks) and len(chunks) == len(batches) == len(counts), 'inconsistent chunk sequences')
        _require(all(z.shape == chunks[0].shape for z in chunks), 'all padded chunks need one fixed shape')
        return self._transaction(chunks, tuple(batches), counts)

    def _transaction(self, chunks, batches, counts):
        _require(all(type(n) is int and 0 < n <= int(z.shape[0]) for n, z in zip(counts, chunks, strict=True))
                 and sum(counts) > 1, 'positive valid-row counts and a batch larger than one required')
        before_vars, before_slots = self._kernels.snapshot()
        self.attempted_updates += 1
        record = {'attempt': self.attempted_updates, 'accepted': False, 'fractions': [],
                  'previous_step': self.accepted_updates, 'active_stages': self.active_stages}
        committed = False
        try:
            self._event({'kind': 'update_started', 'attempt': self.attempted_updates})
            before, _ld, scales_before = self._map(self.guard)
            raw = []
            for i, (z, count) in enumerate(zip(chunks, counts, strict=True)):
                theta, _ld, _scale = self._map(z)
                if batches is None:
                    value, score = self._evaluate(theta, 'gradient')
                else:
                    self.external_target_rows += int(z.shape[0])
                    self._event({'kind': 'external_target_received', 'rows': int(z.shape[0])})
                    value, score = self._check_batch(theta, batches[i])
                raw.append(self._kernels.gradient(z, value, score, tf.constant(count, tf.int32)))
            loss, gradients, gradient_valid = self._kernels.aggregate(tuple(raw), tf.constant(sum(counts), tf.int32))
            _require(bool(gradient_valid.numpy()), 'nonfinite full-batch loss or gradient')
            proposal, proposal_valid, iteration = self._kernels.proposals[self.active_stages-1](gradients)
            _require(bool(proposal_valid.numpy()), 'nonfinite Adam proposal or slot')
            count, inside, valid, history = self._kernels.damp(
                before_vars, proposal, before, scales_before, tf.constant(self.active_stages, tf.int32))
            count = int(count.numpy())
            # These Python scalars are durable diagnostic records, not numerical
            # proposal computation. The bounded damping loop ran entirely in TF.
            record['fractions'] = [
                {'fraction': 2.0**(-i), 'metrics': self._kernels.metrics_payload(row)}
                for i, row in enumerate(history.numpy().tolist()[:count])]
            _require(bool(valid.numpy()), 'nonfinite map or standardized displacement')
            if bool(inside.numpy()):
                fraction = record['fractions'][-1]['fraction']
                metrics = record['fractions'][-1]['metrics']
                self._inverse_check(self.guard)
                training_rows = tf.concat([z[:n] for z, n in zip(chunks, counts, strict=True)], axis=0)
                theta, _ld, _scale = self._map(tf.concat((training_rows, self.guard), axis=0))
                self._evaluate(theta, 'candidate')
                next_step = self.accepted_updates+1
                _require(int(iteration.numpy()) == next_step, 'optimizer iteration mismatch')
                record.update(accepted=True, step=next_step, fraction=fraction, metrics=metrics,
                              loss=float(loss.numpy()), target_rows=self.target_rows,
                              external_target_rows=self.external_target_rows)
                self._event({'kind': 'update_accepted', **record})
                self.accepted_updates = next_step
                self._last_before = before_vars
                committed = True
            if not committed:
                record['reason'] = 'physical_width_budget_exhausted'
        except Exception as error:  # noqa: BLE001 -- rollback and retain every failed transaction.
            record['reason'] = f'{type(error).__name__}: {error}'
        finally:
            if not committed:
                self._kernels.restore(before_vars, before_slots)
                record['accepted'] = False
                self.stopped = record.get('reason', 'interrupted transaction; reconcile before continuation')
        self.records.append(record)
        if not committed:
            self._event({'kind': 'update_rejected', **record})
        return _json_value(record)

    def release_next_stage(self):
        """Release one stage after fresh frozen replay and last-step checking."""
        _require(self.stopped is None and self.preflight_passed, 'release requires an active preflighted trainer')
        _require(self.active_stages < 3 and self.accepted_updates == self.policy.release_boundaries[self.active_stages-1],
                 'release is only allowed at its declared accepted-update boundary')
        _require(self._last_before is not None, 'missing preceding map for independent release check')
        current, slots = _snapshot(self.variables), _snapshot(self.optimizer.variables)
        result = {'boundary': self.accepted_updates, 'stage': self.active_stages+1, 'passed': False}
        probes = self._release_banks[self.active_stages-1]
        try:
            self._event({'kind': 'release_started', 'boundary': self.accepted_updates,
                         'seed': list(self.release_seeds[self.active_stages-1])})
            after, _ld, scale_after = self._map(probes)
            _assign(self.variables, self._last_before)
            before, _ld, scale_before = self._map(probes)
            _assign(self.variables, current)
            metric_values, metric_valid = self._kernels.width(before, after, scale_before, scale_after)
            _require(bool(metric_valid.numpy()), 'nonfinite independent release displacement')
            metrics = self._kernels.metrics_payload(metric_values.numpy().tolist())
            _require(self._inside(metrics), 'independent last-update release budget failed')
            self._replay(probes, 'release')
            result.update(passed=True, metrics=metrics)
            self._event({'kind': 'release_accepted', **result})
            self.active_stages += 1
        except Exception as error:  # noqa: BLE001 -- a failed release always stops this campaign.
            result.update(passed=False, reason=f'{type(error).__name__}: {error}')
            self.stopped = 'release: '+result['reason']
        finally:
            _assign(self.variables, current)
            _assign(self.optimizer.variables, slots)
        self.releases.append(result)
        return _json_value(result)

    def state_payload(self):
        """JSON checkpoint of all effectful state, seeds and work accounting."""
        state = {'schema': 'bayesfilter.neutra.scale_aware_state.v1', 'identity': self._identity,
                 'variable_keys': list(self.variable_keys), 'variable_specs': _specs(self.variables),
                 'variables': [_native(v) for v in self.variables],
                 'optimizer_specs': _specs(self.optimizer.variables),
                 'optimizer_variables': [_native(v) for v in self.optimizer.variables],
                 'accepted_updates': self.accepted_updates, 'attempted_updates': self.attempted_updates,
                 'draw_counter': self.draw_counter, 'target_rows': self.target_rows,
                 'external_target_rows': self.external_target_rows, 'active_stages': self.active_stages,
                 'preflight_passed': self.preflight_passed, 'stopped': self.stopped,
                 'records': self.records, 'releases': self.releases, 'events': self.events,
                 'guard': _native(self.guard), 'ruler': _native(self.lower),
                 'last_before': None if self._last_before is None else [_native(v) for v in self._last_before]}
        return _json_value({**state, 'state_hash': _digest(state)})

    def restore_state(self, payload: Mapping[str, Any]):
        """Validate a full checkpoint before changing any live variable or slot."""
        state = dict(payload)
        supplied_hash = state.pop('state_hash', None)
        _require(supplied_hash == _digest(state), 'scale-aware state hash mismatch')
        _require(state['schema'] == 'bayesfilter.neutra.scale_aware_state.v1'
                 and state['identity'] == self._identity, 'scale-aware source/target/config/seed identity mismatch')
        _require(state['variable_keys'] == list(self.variable_keys)
                 and state['variable_specs'] == _specs(self.variables)
                 and state['optimizer_specs'] == _specs(self.optimizer.variables), 'variable/optimizer inventory mismatch')
        _require(state['guard'] == _native(self.guard) and state['ruler'] == _native(self.lower), 'ruler or guard changed')
        variables = self._validated_values(self.variables, state['variables'])
        optimizer = self._validated_values(self.optimizer.variables, state['optimizer_variables'])
        previous = None if state['last_before'] is None else self._validated_values(self.variables, state['last_before'])
        for name in ('accepted_updates', 'attempted_updates', 'draw_counter', 'target_rows', 'external_target_rows'):
            _require(type(state[name]) is int and state[name] >= 0, f'invalid counter: {name}')
        steps, stages = state['accepted_updates'], state['active_stages']
        _require(int(optimizer[0].numpy()) == steps and state['attempted_updates'] >= steps, 'step/attempt mismatch')
        _require(stages in (1, 2, 3) and type(stages) is int, 'invalid stage count')
        a, b = self.policy.release_boundaries
        _require((stages != 1 or steps <= a) and (stages != 2 or a <= steps <= b)
                 and (stages != 3 or steps >= b), 'inconsistent stage/update boundary')
        _require(type(state['preflight_passed']) is bool and (steps == 0 or state['preflight_passed']), 'missing preflight')
        _require(previous is not None or steps == 0, 'missing last accepted pre-update map')
        _require(len(state['records']) == state['attempted_updates']
                 and sum(r['accepted'] is True for r in state['records']) == steps, 'incomplete update history')
        _require(sum(r['passed'] is True for r in state['releases']) == stages-1, 'incomplete release history')
        for i, value in enumerate(optimizer):
            variable = self.optimizer.variables[i]
            if any(variable is slot for slot in self.optimizer._velocities):
                _require(bool(tf.reduce_all(value >= 0).numpy()), 'negative Adam second moment')
            if variable is self.optimizer.learning_rate:
                expected = tf.constant(self.policy.learning_rate, dtype=variable.dtype)
                _require(bool(tf.reduce_all(value == expected).numpy()), 'constant learning rate changed')
        # No assignment occurs above this line: corrupt restore is transactional.
        _assign(self.variables, variables)
        _assign(self.optimizer.variables, optimizer)
        self._last_before = previous
        for name in ('accepted_updates', 'attempted_updates', 'draw_counter', 'target_rows',
                     'external_target_rows', 'active_stages', 'preflight_passed', 'stopped'):
            setattr(self, name, state[name])
        self.records = _json_value(state['records'])
        self.releases = _json_value(state['releases'])
        self.events = _json_value(state['events'])

    @staticmethod
    def _validated_values(variables, rows):
        _require(len(rows) == len(variables), 'checkpoint array count mismatch')
        validated = []
        for variable, raw in zip(variables, rows, strict=True):
            dtype = tf.as_dtype(variable.dtype)
            value = tf.convert_to_tensor(raw, dtype=dtype)
            _require(value.shape == variable.shape and _finite(value), 'checkpoint array shape/nonfinite failure')
            if dtype.is_integer:
                original = tf.convert_to_tensor(raw, tf.float64)
                _require(bool(tf.reduce_all(tf.cast(value, tf.float64) == original).numpy()), 'fractional integer state')
            validated.append(value)
        return tuple(validated)

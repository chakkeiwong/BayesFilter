"""Independent batch-native TF fixture targets for guarded NeuTra S0 tests.

No MacroFinance target, qualification bank or retained draws are used. Known
potentials are differentiated directly, independently of the detached-score
trainer. Tiny CPU training here is solely an implementation diagnostic.
"""

from dataclasses import replace

import tensorflow as tf
from bayesfilter.inference.neutra_scale_aware_training import (
    ScaleAwareNeuTraTrainer,
    ScaleAwareTargetBatch,
    ScaleAwareUpdatePolicy,
)
from bayesfilter.inference.neutra_training import quadratic_anchor_neutra_config


class KnownTarget:
    def __init__(self, lower, kind='gaussian', signature='1'*64):
        self.lower = tf.convert_to_tensor(lower, tf.float64)
        self.kind = kind
        self.parameter_dim = int(self.lower.shape[0])
        self.parameter_names = tuple(f'x{i}' for i in range(self.parameter_dim))
        self.signature = signature
        self.config = self
        self.calls = self.rows = 0
        self.invalid = False
        self.fail_at_call = None
        self.invalid_after_width = None

    def target_signature(self):
        return self.signature

    def adapter_signature(self):
        return '2'*64

    def signature_payload(self):
        return {'parameter_transform': {'orientation': 'identity', 'inverse_orientation': 'identity'}}

    def value(self, theta):
        x = tf.transpose(tf.linalg.triangular_solve(self.lower, tf.transpose(theta)))
        if self.kind == 'banana':
            y = x[:, 1:]-.15*(x[:, :1]**2-1.)
            return -.5*(x[:, 0]**2+tf.reduce_sum(y**2, axis=1))
        if self.kind == 'funnel':
            v = x[:, 0]
            return -.5*(v/1.5)**2-.5*(self.parameter_dim-1)*v-.5*tf.reduce_sum(x[:, 1:]**2, axis=1)*tf.exp(-v)
        if self.kind == 'quartic':
            return -tf.reduce_sum(.5*x**2+.03*x**4, axis=1)
        return -.5*tf.reduce_sum(x**2, axis=1)

    def evaluate(self, theta):
        theta = tf.convert_to_tensor(theta, tf.float64)
        self.calls += 1
        self.rows += int(theta.shape[0])
        if self.fail_at_call == self.calls:
            raise RuntimeError('injected target exception')
        with tf.GradientTape() as tape:
            tape.watch(theta)
            value = self.value(theta)
        score = tape.gradient(value, theta, output_gradients=tf.ones_like(value))
        eligible = tf.fill([int(theta.shape[0])], not self.invalid)
        if self.invalid_after_width is not None:
            x = tf.transpose(tf.linalg.triangular_solve(self.lower, tf.transpose(theta)))
            eligible &= tf.reduce_all(tf.abs(x) < self.invalid_after_width, axis=1)
        return ScaleAwareTargetBatch(theta, value, score, eligible, self.signature)


def build(kind='gaussian', lower=None, *, jit=False, policy=None, stream=0, event_sink=None):
    if lower is None:
        lower = tf.constant([[1., 0.], [.3, .8]], tf.float64)
    target = KnownTarget(lower, kind)
    cfg = quadratic_anchor_neutra_config(dimension=target.parameter_dim,
        initial_output_shift=[0.]*target.parameter_dim,
        initial_output_scale_log=tf.math.log(tf.linalg.diag_part(target.lower)).numpy().tolist(),
        initial_anchor_factor=target.lower.numpy().tolist(),
        target_parameter_names=target.parameter_names, target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(), anchor_estimator_signature='3'*64,
        initialization_seed=(20260915, 8200+stream), anchor_release_steps=0,
        scale_transform='identity', jit_compile=jit)
    model = ScaleAwareNeuTraTrainer(target, cfg, policy=policy or ScaleAwareUpdatePolicy(),
        evaluator=target.evaluate, training_seed=(20260915, 8100+stream),
        guard_seed=(20260915, 8300+stream), release_seeds=((20260915, 7201+stream*10),
        (20260915, 7202+stream*10)), event_sink=event_sink)
    return model, target


def take_steps(model, count):
    records = []
    for _ in range(count):
        record = model.train_step()
        if not record['accepted']:
            raise AssertionError(record)
        records.append(record)
    return records


def integration_cases():
    cases = []
    for i, kind in enumerate(('gaussian', 'quartic', 'banana', 'funnel')):
        model, target = build(kind, stream=20+i)
        records = take_steps(model, 3)
        cases.append({'case': kind, 'passed': True, 'steps': 3,
                      'fractions': [r['fraction'] for r in records], 'target_rows': target.rows})
    # Real boundaries, not a reduced cadence hidden behind a test setting.
    model, target = build(stream=30)
    take_steps(model, 100)
    release2 = model.release_next_stage()
    if not release2['passed']:
        raise AssertionError(release2)
    take_steps(model, 100)
    release3 = model.release_next_stage()
    if not release3['passed']:
        raise AssertionError(release3)
    take_steps(model, 1)
    cases.append({'case': 'sequential_201_updates', 'passed': model.active_stages == 3,
                  'steps': model.accepted_updates, 'releases': model.releases,
                  'target_rows': target.rows})
    # A support boundary is an ordinary target eligibility failure, not a
    # request to shrink the support or redraw a more favorable bank.
    model, target = build(stream=31)
    model.preflight()
    before = model.state_payload()
    target.invalid_after_width = .01
    record = model.train_step()
    after = model.state_payload()
    exact = all(after[k] == before[k] for k in ('variables', 'optimizer_variables', 'active_stages'))
    cases.append({'case': 'bounded_support_rejection', 'passed': not record['accepted'] and exact,
                  'target_rows': target.rows, 'reason': record['reason']})
    # A narrow correlated fixture plus a positive change of physical units.
    base = tf.constant([[1., 0.], [.0002, .0004]], tf.float64)
    for i, units in enumerate(([1., 1.], [1e3, 1e-3])):
        model, target = build(lower=base*tf.constant(units, tf.float64)[:, None], stream=32+i)
        records = take_steps(model, 2)
        cases.append({'case': f'narrow_units_{i}', 'passed': all(model._inside(r['metrics']) for r in records),
                      'target_rows': target.rows, 'fractions': [r['fraction'] for r in records]})
    return cases


def malformed_batch(batch, defect):
    if defect == 'signature':
        return replace(batch, target_signature='9'*64)
    if defect == 'theta':
        return replace(batch, theta=tf.reverse(batch.theta, axis=[0]))
    if defect == 'eligibility':
        return replace(batch, eligible=tf.zeros_like(batch.eligible))
    if defect == 'nan_score':
        return replace(batch, score=tf.fill(batch.score.shape, tf.constant(float('nan'), tf.float64)))
    raise ValueError(defect)

"""Diagnostic factory examples for machine benchmarks; no posterior admission.

The Kalman example calls the existing batched analytical QR-filter authority.
No filter, gradient or HMC algorithm is reimplemented in this module.
"""

import hashlib
import json
import math
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference.posterior_adapter import ValueScoreCapability


class GaussianBenchmarkTarget:
    def __init__(self, scale=1.):
        self.scale = float(scale)
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError('scale must be positive and finite')

    def adapter_signature(self):
        return hashlib.sha256(json.dumps({'diagnostic': type(self).__name__, 'scale': self.scale},
                                         sort_keys=True).encode()).hexdigest()

    def value_score_capability(self):
        return ValueScoreCapability(value_score_authority='analytical_manual',
            xla_hmc_ready=True, full_chain_xla_diagnostic_ready=True,
            target_scope='hmc-chain-benchmark-fixture-v1',
            runtime_backend='tensorflow', evidence_path=__file__, score_provenance='analytical_manual',
            nonclaims=('Fixed-kernel diagnostic fixture; no tuning or posterior authority.',))

    def log_prob_and_grad(self, theta):
        return -.5 * self.scale * tf.reduce_sum(theta * theta, axis=-1), -self.scale * theta


class KalmanBenchmarkTarget(GaussianBenchmarkTarget):
    def __init__(self):
        super().__init__()
        from bayesfilter.testing.tf_hmc_readiness import QRStaticLGSSMTarget
        self.target = QRStaticLGSSMTarget.default()

    def adapter_signature(self):
        from bayesfilter.testing import (
            lgssm_generic_target_adapter_tf,
            tf_hmc_readiness,
        )

        payload = {'diagnostic': type(self).__name__,
                   'observations': self.target.observations.numpy().tolist(),
                   'prior_scale': self.target.prior_scale.numpy().tolist(),
                   'jitter': float(self.target.jitter.numpy()),
                   'sources': {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                               for module in (lgssm_generic_target_adapter_tf, tf_hmc_readiness)}}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def log_prob_and_grad(self, theta):
        if theta.dtype != tf.float64:
            raise TypeError('the diagnostic Kalman fixture requires float64')
        from bayesfilter.testing.lgssm_generic_target_adapter_tf import (
            lgssm_gaussian_prior_log_prob_and_grad,
            lgssm_qr_log_likelihood_and_grad,
        )
        value, score = lgssm_qr_log_likelihood_and_grad(theta, source_target=self.target)
        prior, derivative = lgssm_gaussian_prior_log_prob_and_grad(theta, prior_scale=self.target.prior_scale)
        return value + prior, score + derivative


def gaussian_target(scale=1.):
    return GaussianBenchmarkTarget(scale)


def kalman_target():
    return KalmanBenchmarkTarget()

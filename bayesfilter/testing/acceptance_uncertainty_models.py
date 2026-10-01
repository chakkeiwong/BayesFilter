"""Tiny fixed-kernel CPU integration fixtures; no learned-map/posterior claim."""
from __future__ import annotations

import math
import tensorflow as tf
import tensorflow_probability as tfp

from .inference_validation.targets import ValidationTarget
from .inference_validation.ssm_targets import make_ssm_validation_target
from .inference_validation.procedures import FrozenTransition, initial_starts
from .inference_validation.designs import digest


class ResidualFunnelTarget(ValidationTarget):
    """Supplied analytic partial whitening, not a learned NeuTra architecture.

    v=3*z0, u=exp(delta)*z[1:], delta=.5*tanh(z0/2). With x=exp(v/2)*u,
    the centered funnel target plus full Jacobian equals a standard normal
    density for z0 and N(0,1) densities for u, plus 2*delta. The constant
    log(3) cancels the v~N(0,9) normalizing scale. Residual child curvature
    exp(2*delta) lies in [exp(-1), exp(1)].
    """
    def __init__(self):
        super().__init__("funnel_noncentered", jit_compile=True)

    def log_density(self, q):
        delta = .5*tf.tanh(q[..., :1]/2.)
        u = tf.exp(delta)*q[..., 1:]
        return -.5*tf.reduce_sum(tf.concat([q[..., :1]**2, u*u], -1), -1) + 2*delta[..., 0]

    def adapter_signature(self):
        return digest({"analytic_residual_funnel": "v1", "scale": 3., "amplitude": .5})


MODEL_CASES = (
    ("gaussian", 1.2), ("ssm_lgssm_qr", .08), ("ssm_lgssm_near_unit", .04),
    ("ssm_nonlinear", .02), ("mixture", .4), ("funnel_noncentered", .8),
    ("residual_funnel", .8),
)


def fixed_model_trace(name, epsilon, *, draws=512, seed=(20261001, 91)):
    if name == "residual_funnel":
        target = ResidualFunnelTarget()
    elif name.startswith("ssm_"):
        target = make_ssm_validation_target(name)
    else:
        target = ValidationTarget(name)
    start = initial_starts(target, "dispersed")
    transition = FrozenTransition(target, chains=4, step_size=epsilon, leapfrog_steps=3)

    @tf.function(input_signature=[tf.TensorSpec(start.shape, tf.float64), tf.TensorSpec([2], tf.int32)],
                 autograph=False, jit_compile=True)
    def sample(position, stream):
        return tfp.mcmc.sample_chain(num_results=draws, num_burnin_steps=32,
            current_state=position, kernel=transition.kernel, seed=stream,
            trace_fn=lambda state, kr: (kr.log_accept_ratio, kr.is_accepted))

    states, (ratios, accepted) = sample(start, tf.constant(seed, tf.int32))
    return {"states": states, "log_accept_ratio": ratios, "is_accepted": accepted,
            "metadata": {"model": name, "target_signature": target.adapter_signature(),
                "epsilon": epsilon, "L": 3, "draws_per_chain": draws, "chain_count": 4,
                "discarded_warmup": 32, "seed": seed, "jit_compile": True,
                "device": "CPU reference; GPU intentionally hidden", "fixed_kernel": True,
                "allocation_provenance": "tiny integration fixture; no tuning of these settings",
                "nonclaims": "no convergence, calibrated stationary acceptance or posterior correctness"}}

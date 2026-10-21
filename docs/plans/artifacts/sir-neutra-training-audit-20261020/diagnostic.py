"""Tiny independent CPU/XLA diagnostic, not a training or inference route."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import time
import tensorflow as tf
import tensorflow_probability as tfp
from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
from bayesfilter.inference import neutra_transport_core as core

started = time.monotonic()
cfg = NeuTraTransportConfig.hoffman_author_iaf(
    3, conditional_scale_cap=2., seed=(20261020, 31), dtype="float64")
flow = NeuTraTransport(cfg)
z = tf.constant([[.4, -.7, 1.1], [-1.3, .8, -.5], [2., 1.3, -.4], [.1, -.3, .6]], tf.float64)
matrix = tf.constant([[2., .2, -.1], [.2, 1.3, .15], [-.1, .15, .8]], tf.float64)

def log_density(x):
    return -.5*tf.reduce_sum(x*tf.matmul(x, matrix), axis=1)-.025*tf.reduce_sum(x**4, axis=1)

def target(x):
    # Opaque target: the trainer may only consume the supplied first score.
    return (tf.stop_gradient(log_density(x)),
            tf.stop_gradient(-tf.matmul(x, matrix)-.1*x**3), tf.ones([4], tf.bool))

@tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float64)], jit_compile=True, autograph=False)
def compare_gradient(latent):
    result = core.reverse_kl_evaluate(flow, target, latent, estimator="standard")
    with tf.GradientTape() as tape:
        x, ld = flow.forward_and_logdet(latent)
        loss = tf.reduce_mean(-log_density(x)-ld)
    direct = tape.gradient(loss, flow.trainable_variables)
    errors = [tf.reduce_max(tf.abs(a-b)) for a, b in zip(result["gradients"], direct)]
    return result["valid"], result["loss"]-loss, tf.reduce_max(errors), tf.linalg.global_norm(direct)

valid, loss_delta, gradient_error, gradient_norm = compare_gradient(z)
assert bool(valid) and float(gradient_norm) > .01
assert abs(float(loss_delta)) < 1.e-11 and float(gradient_error) < 1.e-10

# Isolate one stage to test its formula, not to define a new training profile.
stage = NeuTraTransport(replace(cfg, stages=1)).stages[0]
stage.biases[-1].assign([.1, -.2, .25, .2, -.4, .3])

def same_conditioner(x):
    s, mu = core.iaf_network(stage, x)
    return mu, s

author = tfp.bijectors.Invert(tfp.bijectors.MaskedAutoregressiveFlow(same_conditioner))
local, local_ld = stage.forward_and_logdet(z)
s, mu = core.iaf_network(stage, z)
author_y = author.forward(z)
author_ld = author.forward_log_det_jacobian(z, event_ndims=1)
formula_error = tf.reduce_max(tf.abs(author_y-(z-mu)*tf.exp(-s)))
source_output_difference = tf.reduce_max(tf.abs(local-author_y))
source_logdet_difference = tf.reduce_max(tf.abs(local_ld-author_ld))
assert float(formula_error) < 1.e-12 and float(source_output_difference) > .01

def reparameterized_conditioner(x):
    local_s, local_mu = core.iaf_network(stage, x)
    return -local_mu*tf.exp(-local_s), -local_s

equivalent = tfp.bijectors.Invert(tfp.bijectors.MaskedAutoregressiveFlow(reparameterized_conditioner))
equivalent_error = tf.reduce_max(tf.abs(local-equivalent.forward(z)))
assert float(equivalent_error) < 1.e-12

report = dict(
    role="CPU-only independent engineering/source diagnostic; not SIR or training-quality evidence",
    cpu_only=True, gpu_intentionally_hidden=True, jit_compile_gradient_check=True,
    tensorflow=tf.__version__, tensorflow_probability=tfp.__version__,
    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    seed=[20261020, 31], loss_difference=float(loss_delta),
    standard_gradient_max_abs_error=float(gradient_error), reference_gradient_norm=float(gradient_norm),
    author_inverted_maf_formula_max_abs_error=float(formula_error),
    same_conditioner_local_vs_author_output_max_abs_difference=float(source_output_difference),
    same_conditioner_local_vs_author_logdet_max_abs_difference=float(source_logdet_difference),
    explicitly_reparameterized_conditioner_output_max_abs_error=float(equivalent_error),
    interpretation="Paper Eq.4 chain rule passes; local and author same-weight parameterizations differ. Algebraic reparameterization agrees, but is not identical author network training.",
    wall_seconds=time.monotonic()-started,
    plan="docs/plans/surrogate-hmc-sir-neutra-training-audit-plan-2026-10-20.md",
)
Path(__file__).with_name("diagnostic.json").write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report, indent=2))

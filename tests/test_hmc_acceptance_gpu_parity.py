"""Scheduled explicit-device parity; CPU runs do not substitute for this gate."""
import os

import numpy as np  # Independent diagnostic comparison only.
import pytest
import tensorflow as tf
from tensorflow_probability.python.mcmc.internal.leapfrog_integrator import SimpleLeapfrogIntegrator

from bayesfilter.inference.native_tfp_hmc import reviewed_independent_chain_target_fn
from bayesfilter.inference.hmc_acceptance_statistics import replicated_acceptance_statistics
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from tests.test_hmc_acceptance_protocol import policy


pytestmark = pytest.mark.skipif(os.environ.get("BAYESFILTER_REQUIRE_GPU_VALIDATION") != "1",
    reason="scheduled trusted GPU validation with explicit memory-growth provenance")


@pytest.mark.parametrize("name",["gaussian","ssm_lgssm_qr","ssm_nonlinear"])
def test_explicit_momentum_cpu_gpu_xla_target_and_proposal_parity(name):
    assert tf.config.list_logical_devices("GPU"), "GPU must be visible; no CPU fallback"
    outputs=[]
    for device in ("/CPU:0","/GPU:0"):
        with tf.device(device):
            target=ValidationTarget(name)
            width=target.parameter_dim
            center=([.65,.4,.75] if width==3 else [.5,-1.5] if name.startswith("ssm") else [.2,-.3])
            initial=tf.constant([center]*4,tf.float64)
            momenta=tf.reshape(tf.linspace(tf.constant(-.25,tf.float64),tf.constant(.35,tf.float64),4*width),[4,width])
            target_fn=reviewed_independent_chain_target_fn(target,chain_count=4,parameter_dim=width)
            integrator=SimpleLeapfrogIntegrator(target_fn,[tf.constant(.001,tf.float64)],2)
            @tf.function(input_signature=[tf.TensorSpec([4,width],tf.float64),tf.TensorSpec([4,width],tf.float64)],
                         autograph=False,jit_compile=True)
            def proposal(q,p):
                return integrator([p],[q])
            result=proposal(initial,momenta)
            assert all(device.replace('/','') in t.device for t in tf.nest.flatten(result))
            outputs.append(tf.nest.map_structure(lambda t:t.numpy(),result))
            proposal(initial+tf.constant(.00001,tf.float64),momenta)
            assert proposal.experimental_get_tracing_count()==1
    for cpu,gpu in zip(tf.nest.flatten(outputs[0]),tf.nest.flatten(outputs[1])):
        np.testing.assert_allclose(gpu,cpu,rtol=2e-5,atol=2e-5)


def test_same_trial_vectors_give_cpu_gpu_decision_and_interval_parity():
    p=policy(base_repetitions=64,max_repetitions=64,max_candidates=2)
    rows=[[.60,.67,.73,.80],[.62,.69,.71,.78]]*32
    reports=[]
    for device in ("/CPU:0","/GPU:0"):
        with tf.device(device):
            values=tf.constant(rows,tf.float64)
            assert device.replace('/','') in values.device
            reports.append(replicated_acceptance_statistics(values,policy=p,
                stage="verification",evidence_rungs=(1,),jit_compile=True))
    assert reports[0]["acceptance_decision"]==reports[1]["acceptance_decision"]
    for key in ("start_means","pooled_interval","covariance_of_mean"):
        np.testing.assert_allclose(reports[1][key],reports[0][key],rtol=2e-12,atol=2e-12)

"""Small CPU/reference candidate-set example; no posterior or ranking claim."""
from __future__ import annotations
import os
import sys
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tensorflow as tf
from bayesfilter.inference import (
    HMCControllerConfig, HMCCandidateExecutionConfig, HMCAcceptancePolicy,
    HMCKernelTuningConfig, FixedTransportHMCKernelTuningConfig,
    ValueScoreCapability, tune_hmc_kernel, tune_fixed_transport_hmc_kernel,
)

class AnalyticGaussianAdapter:
    parameter_dim = 2
    def adapter_signature(self):
        return "docs-candidate-set-gaussian-v1"
    def value_score_capability(self):
        return ValueScoreCapability(value_score_authority="graph_native", xla_hmc_ready=False,
            runtime_backend="tensorflow", target_scope="docs_candidate_set",
            evidence_path=__file__, nonclaims=("small CPU reference fixture",))
    def log_prob_and_grad(self, position):
        position = tf.convert_to_tensor(position, tf.float64)
        return -.5 * tf.reduce_sum(position**2, axis=-1), -position

adapter = AnalyticGaussianAdapter()
starts = tf.constant([[-1.,-.5],[-.3,.2],[.4,-.2],[1.,.5]], tf.float64)
# This reduced grid bounds the documentation smoke; the ordinary convenience
# policy uses L=(3,5,9,13,18,25) and optional pilots/refinement.
search = HMCControllerConfig(primary_l_grid=(3,5),
    epsilon_by_l=((3,(1.1,1.3)),(5,(1.1,1.3))), evidence_rungs=(1,))
execution = HMCCandidateExecutionConfig(measurement_num_results=64,
    verification_num_results=64, num_warmup_steps=8, seed=(20260914,71),
    acceptance_policy=HMCAcceptancePolicy(), target_status_trace_policy="none",
    use_xla=False, non_xla_reason="small documentation CPU reference")

run = tune_hmc_kernel(adapter=adapter, initial_position=starts,
    config=HMCKernelTuningConfig.smoke(mass_policy="fixed_identity",
        target_scope="docs_candidate_set"),
    search_config=search, execution_config=execution,
    target_lineage={"model":"standard Gaussian", "data":"none", "prior":"standard normal"},
    source_paths=[__file__])
assert run.result.payload()["nominee_id"] is None
assert len(run.result.candidates) >= 4
print(run.result.completion_status, run.result.verified_candidate_ids)

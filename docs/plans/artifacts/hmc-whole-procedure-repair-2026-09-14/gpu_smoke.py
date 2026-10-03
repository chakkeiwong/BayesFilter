"""Bounded GPU/XLA engineering smoke; no posterior or ranking evidence."""
import os
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
    raise RuntimeError("GPU memory growth must be configured before TensorFlow import")
import json, sys, time, platform, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.set_soft_device_placement(False)
import tensorflow_probability as tfp
from bayesfilter.inference import (
    ValueScoreCapability, HMCAcceptancePolicy, HMCCandidateExecutionConfig,
    HMCControllerConfig, HMCKernelTuningConfig, tune_hmc_kernel,
    resume_hmc_candidate_set_tuning,
    build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result,
    load_hmc_candidate_retained_runner,
)
class Gaussian:
    parameter_dim = 2
    def adapter_signature(self):
        return "whole-procedure-gpu-gaussian-v1"
    def value_score_capability(self):
        return ValueScoreCapability(value_score_authority="graph_native", xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True, runtime_backend="tensorflow",
            target_scope="whole-procedure-gpu", evidence_path=__file__,
            nonclaims=("engineering fixture only",))
    def log_prob_and_grad(self, state):
        return -.5 * tf.reduce_sum(state * state, axis=-1), -state
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
execution = HMCCandidateExecutionConfig(measurement_num_results=128, verification_num_results=128,
    num_warmup_steps=8, seed=(20260915,31), use_xla=True, target_status_trace_policy="none",
    acceptance_policy=HMCAcceptancePolicy(practical_region=(.55,.85),repair_region=(.5,.9)),
    chunk_max_results=256)
search = HMCControllerConfig(primary_l_grid=(3,5), epsilon_by_l=((3,(1.1,1.3,1.5)),(5,(1.1,1.3,1.5))),
    evidence_rungs=(1,), total_budget_units=40, repair_reserve_units=3, max_candidates=6)
target=Gaussian()
from bayesfilter.inference.hmc_candidate_set_execution import _tensor_payload, _tensor_from_payload
with tf.device("/GPU:0"):
    host_boundary_probe = _tensor_from_payload(_tensor_payload(tf.constant([1., 2.], tf.float64)))
tf.debugging.assert_equal(host_boundary_probe, tf.constant([1., 2.], tf.float64))
starts=tf.constant([[-1.,-.5],[-.3,.2],[.4,-.2],[1.,.5]],tf.float64)
first=tune_hmc_kernel(adapter=target, initial_position=starts,
    config=HMCKernelTuningConfig.smoke(mass_policy="fixed_identity",use_xla=True,
        target_scope="whole-procedure-gpu",max_attempts=1),
    execution_config=execution,search_config=search,
    target_lineage={"model":"standard Gaussian","prior":"standard normal","data":"none"},
    source_paths=[__file__],output_dir=root/'tuning',max_work_items=1)
run=resume_hmc_candidate_set_tuning(root/'tuning/tuning_checkpoint.json',adapter=target)
assert run.result.verified_candidate_ids
assert len(run.result.observations)>len(first.result.observations)
binding=run.adapter._execution_binding
member=build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(
    candidate_set_result=run.result,candidate_id=run.result.verified_candidate_ids[0],retained_binding=binding)
path=member.export(root/'member.json')
loaded=load_hmc_candidate_retained_runner(path,adapter=target,claim_eligible=True)
block=loaded.run(num_results=16,seed=(20260915,91),output_dir=root/'retained')
result={"passed":True,"role":"engineering GPU/XLA smoke; no posterior or ranking claim",
    "verified_candidate_ids":run.result.verified_candidate_ids,"candidate_count":len(run.result.candidates),
    "completion_status":run.result.completion_status,"gpu_memory_policy":memory,
    "execution_policy":binding._runtime,"samples_devices":sorted(set(e['samples_device'] for e in binding._evidence.values())),
    "allocator":tf.config.experimental.get_memory_info("GPU:0"),"jit_compile":True,
    "versions":{"python":platform.python_version(),"tensorflow":tf.__version__,"tfp":tfp.__version__},
    "wall_seconds":time.monotonic()-started,"command":sys.argv,
    "git_commit":subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()}
(root/'result.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
print(json.dumps(result,indent=2,default=str))

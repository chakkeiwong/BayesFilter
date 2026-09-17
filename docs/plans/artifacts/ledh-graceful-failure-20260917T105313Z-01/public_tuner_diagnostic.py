"""Real LG1 -> registered public tuner; deliberately CPU diagnostic-only."""
import json,os,sys
from pathlib import Path
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
import tensorflow as tf
from bayesfilter.highdim.ledh_program_fixtures_tf import make_linear_fixture_target
from bayesfilter.inference import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FourChainMeanBandAcceptancePolicy,FrozenPositionOnlyForce,FrozenTargetPotential,
    TensorFlowHMCKernelTuningConfig,bind_neural_force_hmc_tuning_runner,
    tune_hmc_kernel,build_retained_bound_hmc_archive_runner_from_tuning_result,
)
from bayesfilter.inference.neural_force_hmc import PositionOnlyForceEvaluation,TargetPotentialEvaluation
from bayesfilter.inference.hmc_tensorflow_tuning import ENDPOINT_STATUS_TRACE_FIELDS
root=Path(sys.argv[1]);out=Path(os.environ['LEDH_PROGRAM_RESULT_DIR'])
dataset=json.loads((root/'lg1-data-01/data/LG1-calibration-00.json').read_text())
controls=json.loads((root/'lg1-calibration-diagnostic-01/diagnostic.json').read_text())['controls']
target=make_linear_fixture_target(dataset,16,controls,dtype=tf.float64)
# The public tuner currently requires non-XLA; this is explicitly a graph-only
# reference exception. A stable polymorphic batch dimension covers B=1 and B=4.
evaluate=tf.function(target.unconstrained,input_signature=[tf.TensorSpec([None,1],tf.float64)],autograph=False)
def force(z):
    _,score,status=evaluate(z)
    return PositionOnlyForceEvaluation(-score,status['force_valid'])
def potential(z):
    value,_,status=evaluate(z)
    return TargetPotentialEvaluation(-value,status)
scope='LG1-public-binding-diagnostic-'+target.identity
class Adapter:
    target_scope=scope
    def adapter_signature(self):return target.identity
bound=bind_neural_force_hmc_tuning_runner(
    force=FrozenPositionOnlyForce(force,identity=target.identity,
        semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS),
    target=FrozenTargetPotential(potential,identity=target.identity),target_scope=scope)
cfg=TensorFlowHMCKernelTuningConfig(parameter_dimension=1,evidence_role='diagnostic_only',
    mass_window_results=(1,),step_adaptation_results=1,verification_results=1,
    max_leapfrog_steps=1,initial_step_size=.01,
    budget_provenance='one-transition public LG1 binding diagnostic',
    initial_step_size_provenance='convenience mechanics-only',
    geometry_provenance='unconstrained scale .1 mechanics-only',target_scope=scope,
    acceptance_policy=FourChainMeanBandAcceptancePolicy(overall_band=(0.,1.),per_chain_band=(0.,1.)),
    target_accept_prob=.7,verification_repair_rounds=0,step_repair_factor=2.,mass_shrinkage=.1,
    covariance_jitter=1e-9,eigenvalue_floor=1e-9,max_condition_number=1e8,seed=(20260917,901))
result=tune_hmc_kernel(adapter=Adapter(),initial_position=tf.zeros([1],tf.float64),
    parameter_scales=tf.constant([.1],tf.float64),config=cfg,runner_binding=bound,output_dir=out/'tuner')
tf.debugging.assert_all_finite(result.final_raw_state,'accepted state')
assert result.endpoint_status_trace.shape==(3,4,13)
assert not bool(result.handoff_eligible) and not bool(result.passed)
try:
    build_retained_bound_hmc_archive_runner_from_tuning_result(tuning_result=result,runner_binding=bound)
except ValueError as exc:
    assert 'not admission-capable' in str(exc)
    rejected=str(exc)
else:raise AssertionError('diagnostic admitted retained inference')
report=dict(classification='public_binding_diagnostic_not_admission',cpu_only_gpu_intentionally_hidden=True,
    jit_compile=False,non_xla_exception='public TensorFlow tuner has not qualified XLA',
    target_identity=target.identity,binding=dict(bound.payload()),fixture=dict(target.manifest),
    health_passed=bool(result.health_passed),handoff_eligible=bool(result.handoff_eligible),
    final_raw_state=result.final_raw_state.numpy().tolist(),
    endpoint_status_trace_fields=ENDPOINT_STATUS_TRACE_FIELDS,
    endpoint_status_trace=result.endpoint_status_trace.numpy().tolist(),
    admission_rejection=rejected,artifact=result.artifact_manifest_path)
(out/'public-binding.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ('classification','health_passed','handoff_eligible','artifact')}))

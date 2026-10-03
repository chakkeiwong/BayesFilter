"""Verified tuning member to posterior precision: a small engineering example.

Counts, tolerances and seeds below are explicit Gaussian fixture choices. They
are not defaults for a scientific model. GPU/XLA is the normal mode;
--cpu-reference selects a small CPU/non-XLA reference check.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def run_example(root: Path, *, cpu_reference=False):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    if cpu_reference:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=not cpu_reference)
    from bayesfilter.inference import (
        HMCAcceptancePolicy, HMCCandidateExecutionConfig, HMCControllerConfig,
        PrecomputedMassArtifact, bind_hmc_candidate_set_execution, tune_hmc_kernel,
        build_retained_bound_hmc_archive_runner_from_candidate_set_result,
        HMCPrecisionTarget, HMCPrecisionPolicy, HMCPosteriorAssessmentPolicy,
        SequentialNeuTraHMCConfig, run_hmc_posterior,
    )
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from docs.examples.hmc_candidate_set_retained import GaussianTarget

    target = GaussianTarget()
    binding = bind_hmc_candidate_set_execution(adapter=target,
        initial_position=tf.constant([[-1.,-.5],[-.3,.2],[.4,-.2],[1.,.5]], tf.float64),
        mass_artifact=PrecomputedMassArtifact(position=[0.,0.],
            covariance=tf.eye(2,dtype=tf.float64), factor=tf.eye(2,dtype=tf.float64),
            adapter_signature=target.adapter_signature(), position_role='reference_center',
            covariance_source='known standard Gaussian fixture'),
        target_scope='docs_candidate_retained', scope_id='docs-posterior', search_id='example-1',
        target_lineage={'model':'standard_normal_2d', 'data':'none', 'prior':'target_itself'},
        source_paths=[__file__], epsilon_domain=(.1,1.9), repair_factor=1.1, max_repairs_per_family=0,
        config=HMCCandidateExecutionConfig(measurement_num_results=128,
            verification_num_results=128, num_warmup_steps=8, seed=(20260914,11),
            acceptance_policy=HMCAcceptancePolicy(practical_region=(.55,.85), repair_region=(.5,.9)),
            use_xla=not cpu_reference, target_status_trace_policy='per_chain_step',
            non_xla_reason='Gaussian documentation CPU reference' if cpu_reference else None))
    tuning = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        candidate_set_adapter=binding.typed_adapter, output_dir=root/'tuning',
        config=HMCControllerConfig(primary_l_grid=(3,), epsilon_by_l=((3,(1.3,)),),
            total_budget_units=10, repair_reserve_units=3))
    if not tuning.result.verified_candidate_ids:
        raise RuntimeError('bounded fixture produced no verified member')
    # This fixture has one L. For a real grid select a member explicitly;
    # descriptive acceptance or ESS differences do not establish a ranking.
    member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=tuning.result, candidate_id=tuning.result.verified_candidate_ids[0],
        retained_binding=binding)
    member.export(root/'member.json')
    policy = HMCPosteriorAssessmentPolicy(warmup_bulk_ess_min=50., warmup_tail_ess_min=50.,
        warmup_consecutive_checks=2, retained_bulk_ess_min=100., retained_tail_ess_min=100.,
        precision=HMCPrecisionPolicy((HMCPrecisionTarget('x', mcse_absolute_max=.10),
            HMCPrecisionTarget('y', kind='quantile', probability=.9, mcse_absolute_max=.15)),
            method='lugsail', jit_compile=not cpu_reference))
    config = SequentialNeuTraHMCConfig(step_size=member.step_size,
        num_leapfrog_steps=member.num_leapfrog_steps, jit_compile=not cpu_reference,
        warmup_seed=(20260915,17), retained_seed=(20260915,19),
        warmup_chunk_results=256, warmup_min_results=256, warmup_check_window_results=256,
        warmup_max_results=1024, retained_chunk_results=512, retained_min_results=512,
        retained_max_results=2048, assessment_policy=policy)
    with DurableTensorCheckpoint(root/'posterior_chunks', {
        'member_hash':member.member_hash, 'transform':'identity', 'quantities':'x_y_v1',
        'parameter_names':['x','y'], 'consumer_diagnostic':'none', 'policy':policy.payload(),
    }) as store:
        result = run_hmc_posterior(member=member, config=config, parameter_names=('x','y'),
            checkpoint_store=store)
    return {**{k:v for k,v in result.items() if not k.startswith('private_')},
        'memory_policy':memory, 'runtime_policy':binding._runtime,
        'source_closure':binding._spec['source_closure'],
        'interpretation':'Gaussian engineering smoke; no model-specific burn-in or default promotion'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cpu-reference', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = run_example(args.output_dir, cpu_reference=args.cpu_reference)
    result['wall_seconds'] = time.monotonic()-started
    (args.output_dir/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','equilibration_status','precision_status','wall_seconds')}))


if __name__ == '__main__':
    main()

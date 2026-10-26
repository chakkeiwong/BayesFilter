"""Actual MacroFinance public-path diagnostic; no posterior admission claim.

The supplied covariance is a local hypothesis. This bounded two-candidate
exercise does not replace a complete tuning grid or admit the external target.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from diagnose_modern_hmc_macrofinance import BASE, CONSUMER, PLAN, RESULT, digest, write
GPU_PLAN='docs/plans/bayesfilter-modern-hmc-public-gpu-continuation-2026-10-24.md'


def policy():
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    return HMCReplicatedAcceptancePolicy(
        trial_num_results=64, discarded_prefix=0, base_repetitions=32,
        max_repetitions=32, max_candidates=2, search_family_alpha=.05,
        verification_family_alpha=.05, diagnostic_family_alpha=.05,
        temporal_tolerance=.1, qualification_mode='lower_bound',
        qualification_region=(.55, 1.))


def build(inputs, *, device='CPU', gpu_evidence=None):
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=device=='GPU')
    tf.config.experimental.enable_tensor_float_32_execution(False)
    if device == 'GPU':
        from modern_hmc_macrofinance_gpu_support import build_measured_diagnostic, EVIDENCE
        original,adapter,measured,provenance=build_measured_diagnostic(
            inputs,memory,plan=GPU_PLAN,gpu_evidence=EVIDENCE if gpu_evidence is None else gpu_evidence)
        center=tf.constant(inputs['initial_position'],tf.float64)[None,:]
        for left,right in zip(original.log_prob_and_grad(center),adapter.log_prob_and_grad(center)):
            tf.debugging.assert_equal(left,right)
        return adapter,memory
    from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood
    from bayesfilter.linear.kalman_qr_derivatives_tf import tf_qr_sqrt_kalman_score_factors_batched_static
    import one_country_affine_macro_phase9_hmc as consumer
    adapter = consumer._build_generic_target_with_runtime(inputs['observations'],
        dataset_id=inputs['design']['dataset_id'],
        runtime=(tf, tf_kalman_log_likelihood, tf_qr_sqrt_kalman_score_factors_batched_static, memory),
        diagnostic_admission={'role': 'source_bound_reference_diagnostic_only', 'plan': PLAN})
    return adapter, memory


def binding(adapter, inputs, paths, *, use_xla=False):
    import tensorflow as tf
    from bayesfilter.inference import HMCCandidateExecutionConfig, PrecomputedMassArtifact
    from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    covariance = tf.constant(inputs['initial_covariance'], tf.float64)
    center = tf.constant(inputs['initial_position'], tf.float64)
    if covariance.shape != (9, 9) or center.shape != (9,):
        raise ValueError('actual consumer requires nine-dimensional supplied geometry')
    tf.debugging.assert_all_finite(covariance, 'nonfinite supplied covariance')
    tf.debugging.assert_near(covariance, tf.transpose(covariance), atol=1e-12, rtol=1e-12)
    eigenvalues = tf.linalg.eigvalsh(covariance)
    tf.debugging.assert_positive(eigenvalues, 'supplied covariance is not positive definite')
    factor = tf.linalg.cholesky(covariance)
    tf.debugging.assert_near(factor @ tf.transpose(factor), covariance, atol=1e-12, rtol=1e-12)
    offsets = tf.repeat(tf.constant([[-.15], [-.05], [.05], [.15]], tf.float64), 9, axis=1)
    starts = center + offsets @ tf.transpose(factor)
    p = policy()
    config = HMCCandidateExecutionConfig(
        measurement_num_results=p.trial_num_results, verification_num_results=p.trial_num_results,
        num_warmup_steps=p.discarded_prefix, seed=(20261024, 901 if use_xla else 401), acceptance_policy=p,
        target_status_trace_policy='per_chain_step' if use_xla else 'none', use_xla=use_xla,
        non_xla_reason=None if use_xla else 'actual consumer explicitly lacks XLA qualification; CPU engineering diagnostic',
        chain_mode='batched', chunk_max_results=p.trial_num_results,
        transition_profile=HMCTransitionProfile('hmc', 'uniform_duration'))
    mass = PrecomputedMassArtifact(position=center, factor=factor, covariance=covariance,
        adapter_signature=adapter.adapter_signature(), position_role='reference_center',
        covariance_source='saved P18 same-input covariance hypothesis; not refitted or promoted')
    bound = bind_hmc_candidate_set_execution(adapter=adapter, initial_position=starts,
        target_scope='one-country-affine-phase9-synthetic', target_lineage={
            'dataset_id': inputs['design']['dataset_id'], 'inputs': inputs,
            'prior': 'chart Gaussian scale 0.35', 'geometry_role': 'supplied hypothesis'},
        config=config, source_paths=paths, scope_id='macrofinance-gpu-health-mechanics' if use_xla else 'macrofinance-r3-local-mechanics',
        search_id='20261024-public-gpu-health' if use_xla else '20261024-first-public-diagnostic', epsilon_domain=(.1/1.25, .1*1.25),
        repair_factor=1.25, max_repairs_per_family=0, mass_artifact=mass)
    geometry = dict(eigenvalues=[float(v) for v in eigenvalues],
        condition_number=float(eigenvalues[-1]/eigenvalues[0]),
        cholesky_reconstruction_error=float(tf.reduce_max(tf.abs(factor @ tf.transpose(factor)-covariance))),
        initial_positions=starts.numpy().tolist(), role='supplied_geometry_hypothesis')
    return bound, geometry


def price(bound, output, mark):
    import tensorflow as tf
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateRecord
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_payload, _trace_payload
    rows = []
    for ordinal, steps in enumerate((3, 5), 1):
        candidate = HMCTuningCandidateRecord.create(bound.scope, leapfrog_steps=steps,
            epsilon=.1, creation_ordinal=ordinal)
        for repeat in range(2):
            seed = (20261024, (801 if bound.config.use_xla else 101) + (ordinal-1)*2 + repeat)
            mark('price', dict(L=steps, repeat=repeat, seed=seed))
            start = time.monotonic()
            result = bound._run(candidate, bound.initial_active_state, 8, seed)
            failures = bound.health_failures(bound.initial_active_state, result.samples, result.trace)
            if bound.config.use_xla and ('device:GPU:' not in result.samples.device or not result.metadata['use_xla']):
                raise ValueError('public pricing did not execute GPU/XLA')
            row = dict(L=steps, repeat=repeat, seed=seed, elapsed_seconds=time.monotonic()-start,
                health_failures=list(failures), runtime=result.metadata,
                mean_acceptance_reporting_only=float(tf.reduce_mean(tf.exp(tf.minimum(result.trace['log_accept_ratio'], 0.)))))
            write(output/f'price-L{steps}-repeat{repeat}.json', dict(**row,
                samples=_tensor_payload(result.samples), trace=_trace_payload(result.trace)))
            rows.append(row)
            if failures:
                raise ValueError('pricing health failure; preserve trace and investigate')
    warm = [r['elapsed_seconds'] for r in rows if r['repeat'] == 1]
    projected_native = 64 * (64/8) * sum(warm)
    return dict(rows=rows, projected_native_seconds=projected_native,
        projection_basis='64 measurement-plus-verification trials per L, 64/8 horizon scaling',
        projection_limit='excludes full-horizon compilation, evidence diagnostics, serialization and contention')


def tune(adapter, bound, output, remaining):
    from bayesfilter.inference import (HMCControllerConfig, tune_hmc_kernel,
        export_hmc_candidate_retained_runners)
    from bayesfilter.inference.hmc_acceptance_protocol import replicated_evidence_preflight
    from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence
    p = bound.config.acceptance_policy
    price = replicated_evidence_preflight(p, evidence_rungs=(1,), candidate_cap=2,
        leapfrog_steps=(3, 5), transition_profile=bound.config.transition_profile)
    search = HMCControllerConfig(primary_l_grid=(3, 5),
        epsilon_by_l=((3, (.1,)), (5, (.1,))), total_budget_units=7,
        repair_reserve_units=1, candidate_reserve_units=3, allow_repair_from_free_pool=False,
        max_candidates=2, evidence_rungs=(1,),
        max_gradient_work=price['initial_cohort_mandatory_gradient_work'],
        max_wall_time_seconds=remaining, replicated_acceptance_policy=p)
    write(output/'design.json', dict(policy=p.payload(), search=search.payload(), work_price=price))
    run = tune_hmc_kernel(adapter=adapter, initial_position=bound.initial_active_state,
        config=search, candidate_set_adapter=bound.typed_adapter, output_dir=output/'tuning')
    audit = audit_native_evidence(run.result, bound)
    write(output/'native-audit.json', audit)
    paths = export_hmc_candidate_retained_runners(candidate_set_result=run.result,
        retained_binding=bound, output_dir=output/'members')
    if set(paths) != set(run.result.verified_candidate_ids):
        raise ValueError('exported member set differs from verified set')
    write(output/'members.json', {key:str(path) for key,path in paths.items()})
    return dict(completion_status=run.result.completion_status, final_status=run.result.final_status,
        candidate_states=dict(run.result.candidate_states), verified=list(paths),
        total_candidates=len(run.result.candidates), native_audit=audit,
        all_verified_members_exported=True, no_ranking=True)


def reload_members(adapter, source, output, mark, *, use_xla=False):
    import tensorflow as tf
    from bayesfilter.inference import load_hmc_candidate_retained_runners
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload
    paths = json.loads((source/'members.json').read_text())
    if not paths:
        return dict(status='no_verified_members', verified=[], continuation_tested=False)
    members = load_hmc_candidate_retained_runners(paths.values(), adapter=adapter)
    if set(members) != set(paths):
        raise ValueError('reloaded membership changed')
    rows = []
    for index, (cid, member) in enumerate(sorted(members.items())):
        original = json.loads(Path(paths[cid]).read_text())
        tf.debugging.assert_equal(member.initial_active_state, _tensor_from_payload(original['verified_endpoint']))
        if member.claim_eligible or member.posterior_convergence_authority:
            raise ValueError('diagnostic member acquired unsupported scientific authority')
        seed = (20261024, (1001 if use_xla else 501)+index)
        if seed in member._tuning_seeds():
            raise ValueError('retained stream collides with tuning stream')
        mark('fresh_member_continuation', dict(candidate_id=cid, seed=seed))
        result = member.run(num_results=8, seed=seed, output_dir=output/f'member-{index}')
        if use_xla and 'device:GPU:' not in result['samples'].device:
            raise ValueError('reloaded member did not execute on GPU')
        rows.append(dict(candidate_id=cid, member_hash=member.member_hash, seed=seed,
            archive=result['archive_path'], endpoint_preserved=True,
            finite_continuation=bool(tf.reduce_all(tf.math.is_finite(result['samples'])))))
    return dict(verified=sorted(members), rows=rows, continuation_tested=True,
                all_verified_members_reloaded=True)


def main():
    global PLAN, RESULT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('price', 'tune', 'reload'), required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--wall-seconds', type=float, required=True)
    parser.add_argument('--device',choices=('CPU','GPU'),default='CPU')
    args = parser.parse_args()
    gpu=args.device=='GPU'
    if gpu:
        PLAN=GPU_PLAN
        RESULT='docs/plans/bayesfilter-modern-hmc-public-gpu-result-2026-10-24.md'
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(status='running', mode=args.mode, plan=PLAN, result=RESULT,
        gpu_intentionally_hidden=not gpu, use_xla=gpu, posterior_qualification=False,
        default_promoted=False, evidence_role='actual_consumer_gpu_diagnostic_composition' if gpu else 'actual_consumer_cpu_engineering_diagnostic')
    paths = list((BASE/'bayesfilter').rglob('*.py')) + [Path(__file__).resolve(),
        Path(__file__).with_name('diagnose_modern_hmc_macrofinance.py').resolve(), args.inputs.resolve()]
    paths += [CONSUMER/name for name in ('one_country_affine_macro_phase9_hmc.py',
        'one_country_affine_macro_governance.py', 'one_country_affine_macro_transition_2x2.py',
        'one_country_affine_macro_prior_measurement_contract.py')]
    if gpu:paths.append(Path(__file__).with_name('modern_hmc_macrofinance_gpu_support.py').resolve())
    before = {str(p):digest(p) for p in paths}
    manifest = dict(command=[sys.executable,*sys.argv], started_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
        source_sha256=before, inputs_sha256=digest(args.inputs), plan=PLAN, result=RESULT,
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
            'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        wall_seconds=args.wall_seconds, source_worktree=str(Path(__file__).resolve().parents[1]),
        runtime_authority=str(BASE), mode=args.mode, seeds={'tuning':[20261024,901 if gpu else 401],
            'price':[[20261024,i] for i in range(801 if gpu else 101,805 if gpu else 105)],
            'retained':[[20261024,i] for i in ((1001,1002) if gpu else (501,502))]})
    write(args.output/'manifest.json',manifest)
    def mark(stage, details=None):
        write(args.output/'progress.json',dict(stage=stage,details=details,elapsed_seconds=time.monotonic()-started))
    try:
        if os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
            raise ValueError('memory growth required before TensorFlow import')
        if not gpu and os.environ.get('CUDA_VISIBLE_DEVICES') != '-1':
            raise ValueError('explicit CPU hiding required before TensorFlow import')
        sys.path.insert(0,str(CONSUMER)); sys.path.insert(0,str(BASE))
        if gpu:
            from bayesfilter.runtime.gpu_provenance import selected_nvidia_gpu,query_nvidia_smi_gpus
            manifest['physical_gpu']=selected_nvidia_gpu(query_nvidia_smi_gpus())
            manifest['gpu_trust_basis']='trusted_escalated_gpu_execution'
            write(args.output/'manifest.json',manifest)
        inputs=json.loads(args.inputs.read_text())
        mark('build_actual_adapter')
        adapter,memory=build(inputs,device=args.device)
        import tensorflow as tf
        import tensorflow_probability as tfp
        manifest.update(tensorflow=tf.__version__,tensorflow_probability=tfp.__version__,
            dtype='float64',tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),memory_policy=memory)
        write(args.output/'manifest.json',manifest)
        report.update(memory_policy=memory,adapter_signature=adapter.adapter_signature(),
            target_status_telemetry_available=callable(getattr(adapter,'target_status_telemetry',None)),
            target_status_policy='per_chain_step' if gpu else 'none: consumer has graph assertions but no structured telemetry',
            application_qualification_gap='external consumer not integrated; posterior unqualified' if gpu else 'structured per-proposal filter status unavailable')
        from bayesfilter.runtime.execution_budget import execution_budget, execution_progress
        with execution_budget(deadline=started+args.wall_seconds), execution_progress(mark):
            if args.mode == 'reload':
                if args.source is None: raise ValueError('reload requires source output')
                report['execution']=reload_members(adapter,args.source,args.output,mark,use_xla=gpu)
            else:
                bound,geometry=binding(adapter,inputs,paths,use_xla=gpu)
                report['geometry']=geometry
                report['execution']=(price(bound,args.output,mark) if args.mode=='price' else
                    tune(adapter,bound,args.output,args.wall_seconds-(time.monotonic()-started)))
        report['sources_unchanged']=all(digest(p)==h for p,h in before.items())
        if not report['sources_unchanged']: raise ValueError('source changed during diagnostic')
        if gpu:report['allocator_bytes']=tf.config.experimental.get_memory_info('GPU:0')
        report['status']='completed'
    except Exception as exc:
        report.update(status='error',error=repr(exc),traceback=traceback.format_exc())
    report.update(elapsed_seconds=time.monotonic()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','error','elapsed_seconds','mode')}))
    return 0 if report['status']=='completed' else 1


if __name__=='__main__':
    raise SystemExit(main())

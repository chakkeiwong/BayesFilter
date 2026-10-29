#!/usr/bin/env python3
"""M10 scoped protection non-harm and optimistic HMC cost screens; no inference."""
import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'docs/plans/surrogate-hmc-nonlinear-controls-2026-09-21.md'
os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
sys.path.insert(0, str(ROOT))
from scripts.qualify_surrogate_hmc_nonlinear_consumer import save, sha, snapshot, serial
from scripts.calibrate_surrogate_hmc_sv_controls import no_new_failures


def optimistic_forecast(records, remaining):
    """A lower cost screen cannot issue a launch allowance."""
    if len(records) != 5 or records[0]['compilation_included'] is not True:
        raise ValueError('one compilation and four steady complete transitions required')
    if any(r['compilation_included'] for r in records[1:]):
        raise ValueError('steady costs cannot include compilation')
    if any(not math.isfinite(r[k]) or r[k] <= 0 for r in records for k in ('wall_seconds','cpu_worker_seconds')):
        raise ValueError('finite positive measured transition costs required')
    costs = {key: 1.5*(records[0][timing]+3250*min(r[timing] for r in records[1:]))
             for key,timing in (('gpu_seconds','wall_seconds'),('cpu_worker_seconds','cpu_worker_seconds'))}
    return dict(status=('UNDER_BUDGETED_INITIAL_ALLOCATION' if any(costs[k] > remaining[k] for k in costs)
                        else 'REQUIRES_COMPLETE_FORECAST'), optimistic_cost=costs,
        remaining_family=remaining, transitions=3250, safety_factor=1.5,
        authorizes_inference=False, omitted=['nomination','references','heuristics','analysis','additional leapfrogs'])


def execute(out):
    manifest = json.loads((out/'run-manifest.json').read_text())
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if len(tf.config.list_physical_devices('GPU')) != 1:
        raise ValueError('exactly one designated GPU required')
    from bayesfilter.testing.surrogate_hmc_nonlinear_consumer import make_nonlinear_fixture
    from bayesfilter.testing.surrogate_hmc_nonlinear_reference import PRIOR_SD
    from bayesfilter.inference.surrogate_hmc_dctr_selection import MULTIPLIERS, protection_controls, issue_selection
    sources = manifest['source_hashes']
    manifest.update(memory_policy=memory, gpu_status=[d.name for d in tf.config.list_logical_devices('GPU')],
                    tensorflow=tf.__version__,tf32=True,jit_compile=True)
    save(out/'run-manifest.json',manifest)
    scope = dict(model='M10', attempt=1, replicate=0, horizon=2, particle_count=1024,
        state_dimension=1, parameter_dimension=2, dtype='float32', tf32=True, jit_compile=True,
        backend='TensorFlow GPU canonical analytical DCTR', reset='contract_e_chol_v1',
        chart='identity log-noise multipliers', prior='independent Normal(0,log(2)^2)',
        chunk_policy='dpf_transport_exact_divisor_cap3000_v1', chunk_size=1024,
        tuning_controls=['reset_ridge','correction_lm_damping','correction_lm_scale_floor',
                         'correction_trust_radius','pairwise_rms_cap','coordinate_cap'],
        prepared_data_regimes=['nominal','stress'], fixed_input_seed_date=20260920)
    save(out/'scope.json',scope)
    manifest['seeds'] = dict(fixed_inputs={str(i):[20260920,16200+i] for i in (20,21,22)},
                            transitions={arm:[[20260921,17100+100*(arm == 'C')+i] for i in range(5)]
                                         for arm in ('B','C')})
    save(out/'run-manifest.json',manifest)
    nominal = tf.constant([[0.,0.],[.15,-.1],[-.15,.1],[-.3,-.3]],tf.float32)
    stress = tf.constant([[-2*PRIOR_SD,-2*PRIOR_SD],[-2*PRIOR_SD,2*PRIOR_SD],
                          [2*PRIOR_SD,-2*PRIOR_SD],[2*PRIOR_SD,2*PRIOR_SD]],tf.float32)
    fields = ('filtering_mean','filtering_covariance','post_reset_mean','post_reset_covariance')
    started, cpu_started = time.monotonic(), time.process_time()

    def resource_check():
        if (time.monotonic()-started > manifest['wall_budget']-3
                or time.process_time()-cpu_started > manifest['cpu_budget']-3
                or tf.config.experimental.get_memory_info('GPU:0')['peak'] > 4*1024**3):
            raise RuntimeError('M10 control resource limit')

    curves, data_paths, datasets = {}, {}, {}
    reference = json.loads((Path(manifest['reference_root'])/'report.json').read_text())
    for partition in ('calibration','validation'):
        directory = out/partition
        directory.mkdir()
        datasets[partition] = {}
        provenance = {}
        for situation in ('nominal','stress'):
            binding = reference['partitions'][partition+'-'+situation]
            path = Path(binding['data_file'])
            if sha(path) != binding['data_sha256']:
                raise ValueError('frozen control data changed')
            data = json.loads(path.read_text())
            if (data.get('role'),data.get('regime'),data.get('horizon'),data.get('dtype')) != (partition,situation,2,'float32'):
                raise ValueError('wrong M10 control dataset')
            datasets[partition][situation] = data
            provenance[situation] = dict(path=str(path),sha256=sha(path),seed=data['seed'])
        data_paths[partition] = directory/'data.json'
        save(data_paths[partition],dict(model_id='M10',attempt=1,partition=partition,
            original_datasets=provenance, data=datasets[partition]))
        manifest.setdefault('partition_data',{})[partition] = dict(path=str(data_paths[partition]),
                                                                 sha256=sha(data_paths[partition]))
        manifest['seeds'][partition] = {k:v['seed'] for k,v in provenance.items()}
        save(out/'run-manifest.json',manifest)
        outputs, diagnostics = [], []
        for index,multiplier in enumerate(MULTIPLIERS):
            current, situations, fixtures = {}, {}, {}
            begin = time.monotonic()
            for situation,positions in (('nominal',nominal),('stress',stress)):
                fixture = make_nonlinear_fixture(datasets[partition][situation],
                    controls=protection_controls(multiplier),particle_count=1024)
                evaluate = fixture.compile(4,filter_moments=True)
                value,force,status = evaluate(positions)
                value.numpy()
                valid = status['value_valid'] & status['force_valid']
                finite = tf.math.is_finite(value) & tf.reduce_all(tf.math.is_finite(force),axis=1)
                for key in fields:
                    finite &= tf.reduce_all(tf.math.is_finite(status[key]),axis=list(range(1,status[key].shape.rank)))
                healthy = valid & finite
                flagged = ~valid & ((status['value_failure_code'] != 0) | (status['force_failure_code'] != 0))
                situations[situation] = serial(dict(positions=positions,value=value,force=force,status=status,
                    healthy_rows=healthy,all_healthy=bool(tf.reduce_all(healthy)),
                    all_failures_flagged=bool(tf.reduce_all(healthy | flagged))))
                fixtures[situation] = dict(fixture.manifest)
                current[situation] = (value,force,status)
                resource_check()
            arm = dict(multiplier=multiplier,controls=protection_controls(multiplier),
                fixtures=fixtures,situations=situations,wall_seconds=time.monotonic()-begin)
            save(directory/f'arm-{index}.json',arm)
            diagnostics.append(arm)
            outputs.append(current)
            print(partition,multiplier,{k:v['all_healthy'] for k,v in situations.items()},flush=True)
        pairs = []
        for index in range(3):
            changes = {}
            for situation in ('nominal','stress'):
                a,b = outputs[index][situation],outputs[index+1][situation]
                components = {'value':(a[0],b[0]),'force':(a[1],b[1])}
                components.update({k:(a[2][k],b[2][k]) for k in fields})
                changes[situation] = {k:float(tf.reduce_max(tf.abs(x-y)/(1+tf.abs(y)))) for k,(x,y) in components.items()}
            maximum = max(changes['nominal'].values())
            healthy = all(diagnostics[i]['situations']['nominal']['all_healthy'] for i in (index,index+1))
            safe = all(diagnostics[i]['situations']['stress']['all_failures_flagged'] for i in (index,index+1))
            no_new = no_new_failures(diagnostics[index]['situations']['stress']['healthy_rows'],
                                     diagnostics[index+1]['situations']['stress']['healthy_rows'])
            pairs.append(dict(stronger_multiplier=MULTIPLIERS[index],weaker_multiplier=MULTIPLIERS[index+1],
                healthy_nominal=healthy,stress_safe=safe,no_new_stress_failure=no_new,
                normalized_maximum_change=maximum,tolerance=1e-3,component_changes=changes,
                non_harm_passed=healthy and safe and no_new and math.isfinite(maximum) and maximum <= 1e-3))
        curves[partition] = directory/'curve.json'
        save(curves[partition],dict(scope=scope,source_hashes=sources,data_sha256=sha(data_paths[partition]),
            pairs=pairs,arms=[dict(path=str(directory/f'arm-{i}.json'),sha256=sha(directory/f'arm-{i}.json')) for i in range(4)]))
    try:
        selection = issue_selection(output=out/'selection.json',scope=scope,evidence_paths=curves,
            partition_data_paths=data_paths,source_hashes=sources)
    except ValueError as exc:
        if str(exc) != 'NO_STABLE_CONTROL_PAIR':
            raise
        selection = None
    report = dict(status='SELECTED_FOR_PROPOSAL_EXPERIMENT' if selection else 'NO_STABLE_CONTROL_PAIR',
        selected=selection,scope=scope,posterior_comparisons='NOT_RUN',heuristic_dominance='NOT_RUN',
        inference_attempts_consumed=0,scoped_control_attempts_consumed=1,
        memory_policy=memory,tf32=True,jit_compile=True,forecast={})
    save(out/'report.json',report)
    if selection:
        report['forecast'] = measure_cost(datasets['calibration']['nominal'],selection['controls'],
                                         out,nominal,resource_check,manifest)
    report['allocator'] = tf.config.experimental.get_memory_info('GPU:0')
    save(out/'report.json',report)
    return 0 if selection else 2


def measure_cost(dataset,controls,out,nominal,resource_check,manifest):
    import tensorflow as tf
    from bayesfilter.testing.surrogate_hmc_nonlinear_consumer import make_nonlinear_consumer
    from bayesfilter.testing.surrogate_hmc_lg1_tf import endpoint_status
    from bayesfilter.inference.neural_force_hmc import (
        FrozenTargetPotential,TargetPotentialEvaluation,NeuralForceHMCConfig,neural_force_hmc_transition)
    consumer = make_nonlinear_consumer(dataset,controls=controls,batch_size=4)

    def finite_endpoint(z):
        value,_,status = consumer.evaluate_force(z)
        value = tf.where(status['value_valid'],tf.cast(value,tf.float64),tf.constant(float('nan'),tf.float64))
        return TargetPotentialEvaluation(-value,endpoint_status(value))

    finite = FrozenTargetPotential(finite_endpoint,identity=consumer.fixture.identity+':frozen-DCTR-scalar')
    config = NeuralForceHMCConfig(step_size=.02,num_leapfrog_steps=1,
                                  inverse_mass_diagonal=(1.,1.),dtype='float64')
    record_by_arm = {}
    for arm,endpoint in (('B',consumer.endpoint),('C',finite)):
        position = tf.cast(nominal,tf.float64)
        potential = endpoint.function(position).potential
        @tf.function(input_signature=[tf.TensorSpec([4,2],tf.float64),tf.TensorSpec([4],tf.float64),
                                      tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
        def transition(q,u,seed):
            return neural_force_hmc_transition(q,u,consumer.force,endpoint,config,seed)
        records = []
        for i in range(5):
            start,cpustart = time.monotonic(),time.process_time()
            seed = [20260921,17100+100*(arm == 'C')+i]
            step = transition(position,potential,tf.constant(seed,tf.int32))
            step.position.numpy()
            records.append(dict(compilation_included=i == 0,wall_seconds=time.monotonic()-start,
                cpu_worker_seconds=time.process_time()-cpustart,seed=seed,trace=serial(step.trace._asdict())))
            save(out/f'cost-transitions-{arm}.json',records)
            if (not bool(tf.reduce_all(tf.math.is_finite(step.position)))
                    or not bool(tf.reduce_all(tf.math.is_finite(step.trace.delta_h)))
                    or bool(tf.reduce_any(step.trace.endpoint_status['numerical_failure']))
                    or bool(tf.reduce_any(step.trace.force_fallback)) or bool(tf.reduce_any(step.trace.divergence))):
                raise ValueError('invalid '+arm+' calibration transition; cost forecast unavailable')
            resource_check()
            position,potential = step.position,step.potential
        record_by_arm[arm] = records
    spent_cpu = time.process_time()+manifest['parent_cpu_before_worker']
    spent_wall = time.monotonic()-manifest['started_monotonic']
    remaining = {k:max(0.,manifest['family_remaining'][k]-spent)
                 for k,spent in (('cpu_worker_seconds',spent_cpu),('gpu_seconds',spent_wall))}
    forecasts = {arm:optimistic_forecast(records,remaining) for arm,records in record_by_arm.items()}
    joint = {k:sum(row['optimistic_cost'][k] for row in forecasts.values()) for k in remaining}
    result = dict(arms=forecasts,combined_optimistic_cost=joint,
        combined_status='UNDER_BUDGETED_INITIAL_ALLOCATION' if any(joint[k] > remaining[k] for k in joint)
                        else 'REQUIRES_COMPLETE_FORECAST',
        remaining_family=remaining,B_endpoint='local513 mechanics-only optimistic cost proxy',
        C_endpoint='frozen selected canonical DCTR scalar',authorizes_inference=False)
    save(out/'cost-forecast.json',result)
    return result


def main():
    started,cpustart = time.monotonic(),time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference-root',type=Path)
    parser.add_argument('--qualification-root',type=Path)
    parser.add_argument('--cpu-budget',type=float,default=350.)
    parser.add_argument('--wall-budget',type=float,default=450.)
    parser.add_argument('--family-cpu-budget',type=float,default=1800.)
    parser.add_argument('--family-gpu-budget',type=float,default=7200.)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        return execute(out)
    if (('-m10-controls-' not in out.name and not (out.name == 'P3-M10-controls'
            and out.parent.name.startswith('surrogate-hmc-all-models-'))) or '-tests-' in out.name):
        parser.error('fresh M10 controls output marker required')
    from bayesfilter.inference.surrogate_hmc_program import m10_consumer_state, m10_control_attempts
    state = m10_consumer_state(ROOT,additional_reports=(
        [args.qualification_root/'report.json'] if args.qualification_root else ()))
    if not state['qualified']:
        raise ValueError('current-source M10 consumer qualification required')
    qualified_root = Path(state['qualified']['report']).parent
    qualification = json.loads((qualified_root/'run-manifest.json').read_text())
    reference = Path(qualification['parent_reference'])
    if args.reference_root and args.reference_root.resolve() != reference.resolve():
        raise ValueError('reference/qualification mismatch')
    history = m10_control_attempts(ROOT)
    prior_cost = history['spent']
    if history['reports']:
        raise ValueError('prior control attempt requires checked reuse or prospective repair, not automatic rerun')
    totals = dict(cpu_worker_seconds=0.,gpu_seconds=0.)
    for path in (ROOT/'docs/plans/artifacts').glob('surrogate-hmc-all-models-*/accounting.json'):
        entry = json.loads(path.read_text())
        charge = entry.get('charged_this_run',entry.get('charged',{}))
        for key in totals:
            value = float(charge[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid campaign accounting')
            totals[key] += value
    if not all(math.isfinite(v) and v > 0 for v in
               (args.cpu_budget,args.wall_budget,args.family_cpu_budget,args.family_gpu_budget)):
        raise ValueError('positive finite resource ceilings required')
    state['family_remaining'] = {k:min(state['family_remaining'][k],v) for k,v in
        (('cpu_worker_seconds',args.family_cpu_budget),('gpu_seconds',args.family_gpu_budget))}
    cpu = min(args.cpu_budget,350.-prior_cost['cpu_worker_seconds'],state['family_remaining']['cpu_worker_seconds'],54000.-totals['cpu_worker_seconds'])
    wall = min(args.wall_budget,450.-prior_cost['gpu_seconds'],state['family_remaining']['gpu_seconds'],226800.-totals['gpu_seconds'])
    if min(cpu,wall) < 3:
        raise ValueError('M10 control allocation exhausted')
    out.mkdir(parents=True,exist_ok=False)
    sources = {k:v for k,v in qualification['source_hashes'].items() if k.startswith('bayesfilter/')}
    for name in (PLAN,'scripts/calibrate_surrogate_hmc_nonlinear_controls.py',
                 'scripts/calibrate_surrogate_hmc_sv_controls.py','scripts/qualify_surrogate_hmc_nonlinear_consumer.py'):
        sources[name] = sha(ROOT/name)
    snapshot(out,sources)
    manifest = dict(command=[sys.executable,*sys.argv],git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        environment=sys.prefix,plan=PLAN,result_file=str(out/'report.json'),source_hashes=sources,
        qualification_report=state['qualified'],qualification_manifest_sha256=sha(qualified_root/'run-manifest.json'),
        reference_root=str(reference),reference_manifest_sha256=sha(reference/'run-manifest.json'),
        data_version='frozen M10 T2 nominal/stress calibration/validation; claim data unused',
        family_remaining=state['family_remaining'],cpu_budget=cpu,wall_budget=wall,started_monotonic=started,
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),memory_policy='pending trusted GPU configuration')
    save(out/'run-manifest.json',manifest)
    save(out/'report.json',dict(status='RUNNING',posterior_comparisons='NOT_RUN'))
    child_cpu = cpu-(time.process_time()-cpustart)-2.
    child_wall = wall-(time.monotonic()-started)-1.
    manifest['parent_cpu_before_worker'] = time.process_time()-cpustart
    save(out/'run-manifest.json',manifest)
    def limit():
        resource.setrlimit(resource.RLIMIT_CPU,(max(1,math.floor(child_cpu)),max(2,math.ceil(child_cpu))))
    try:
        with (out/'worker.log').open('w') as log:
            process = subprocess.Popen([sys.executable,__file__,'--worker','--output',str(out)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limit)
            try:
                code = process.wait(timeout=max(.1,child_wall))
            except subprocess.TimeoutExpired:
                process.kill();process.wait();code=124
    except Exception as exc:
        save(out/'launch-error.json',dict(error=f'{type(exc).__name__}: {exc}'))
        code = 125
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    charged_cpu = usage.ru_utime+usage.ru_stime-before.ru_utime-before.ru_stime+time.process_time()-cpustart
    charged_wall = time.monotonic()-started
    manifest = json.loads((out/'run-manifest.json').read_text())
    report = json.loads((out/'report.json').read_text())
    if code != 0 and (report['status'] == 'RUNNING' or (report.get('selected') and not report.get('forecast'))):
        report['status'] = 'STOPPED_RESOURCE_LIMIT' if code in (124,-24,-9) else 'EXECUTION_FAILED'
    if any(sha(ROOT/name) != digest for name,digest in manifest['source_hashes'].items()):
        report['status'],code = 'SOURCE_DRIFT',2
    manifest.update(wall_seconds=charged_wall,cpu_worker_seconds=charged_cpu,exit_code=code)
    report.update(wall_seconds=charged_wall,cpu_worker_seconds=charged_cpu,exit_code=code)
    report['family_remaining_after_run'] = {
        k:max(0.,manifest['family_remaining'][k]-cost) for k,cost in
        (('cpu_worker_seconds',charged_cpu),('gpu_seconds',charged_wall))}
    save(out/'run-manifest.json',manifest);save(out/'report.json',report)
    save(out/'accounting.json',dict(charged_this_run=dict(gpu_seconds=charged_wall,cpu_worker_seconds=charged_cpu)))
    print(report['status'],'CPU',round(charged_cpu,3),'GPU wall',round(charged_wall,3),flush=True)
    return code if code >= 0 else 2


if __name__ == '__main__':
    raise SystemExit(main())

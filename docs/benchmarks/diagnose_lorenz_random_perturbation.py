"""Diagnostic only: native Octave and autonomous TF random-input sensitivity.

NumPy/SciPy are used only in the independent source-reference lane. The TF
lane uses TensorFlow kernels and standard-library reporting, never NumPy math.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def variants():
    rows = [dict(id='zero', arm='zero', epsilon=0., direction=0, sign=1)]
    for arm in ('normal', 'uniform', 'combined'):
        for direction in range(4):
            for sign in (-1, 1):
                rows.append(dict(id=f'{arm}_d{direction}_s{sign}', arm=arm,
                                 epsilon=1e-8, direction=direction, sign=sign))
    for exponent in (-10, -6):
        for sign in (-1, 1):
            rows.append(dict(id=f'ladder_e{exponent}_s{sign}', arm='combined',
                             epsilon=10.**exponent, direction=0, sign=sign))
    return rows


def oq(value):
    return "'" + str(value).replace("'", "''") + "'"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Source anchor mismatch: ' + old)
    return text.replace(old, new)


def source_run(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    start = time.monotonic()
    fixture = json.loads((args.reference / 'fixture.json').read_text())
    baseline = Path(fixture['reference_root'])
    source = args.source_lock / 'author-source/StateSpaceModels/Lorenz'
    lock = json.loads((args.source_lock / 'source-lock.json').read_text())
    assert lock['status'] == 'PASS'
    for item in lock['verified_snapshot_files']:
        relative = Path(item['path']).relative_to('.localresources/code/heng-controlled-smc')
        assert sha(args.source_lock / 'author-source' / relative) == item['sha256']
    for item in lock['runtime_dependencies']:
        assert sha(args.root/item['path']) == item['sha256']
    for name in ('cSMC_Resample.m','cSMC_VaryF.m','RungeKutta4.m','Systematic_Resampling.m','SimulatedData.mat'):
        assert sha(source/name) == sha(baseline/'sources'/name)
    stats = args.root / '.localresources/code/octave-statistics-1.5.0/inst'
    # Capture verified the exact author archive and this mvnrnd dependency.
    baseline_summary = json.loads((baseline / 'summary.json').read_text())
    for key, expected in baseline_summary['dependency_sha256'].items():
        assert sha(Path(baseline_summary[key])) == expected
    for case in fixture['cases']:
        path = baseline / f"author-F{case['forcing']:.4f}-seed{case['seed']}.mat"
        assert sha(path) == case['reference_archive_sha256']
    observed = args.output / 'observed'
    observed.mkdir()
    sampler = (stats / 'distributions/mvnrnd.m').read_text()
    sampler = replace_once(sampler, '  s = randn(n,d)*U + mu;', '''  z = randn(n,d);
  global P_N P_DN P_EPS P_SIGN P_ARM P_NC;
  P_NC = P_NC + 1; assert(P_NC <= 505);
  assert(isequal(z,P_N(:,:,P_NC)));
  if P_ARM == 1 || P_ARM == 3
    z = z + P_SIGN * P_EPS * double(P_DN(:,:,P_NC));
  end
  s = z*U + mu;''')
    (observed / 'mvnrnd.m').write_text(sampler)
    resample = (source / 'Systematic_Resampling.m').read_text()
    resample = replace_once(resample, '    U = rand(1);', '''    U = rand(1);
    global P_U P_DU P_EPS P_SIGN P_ARM P_UC P_REFLECT;
    P_UC = P_UC + 1; assert(P_UC <= 505);
    assert(isequal(U,P_U(P_UC)));
    if P_ARM == 2 || P_ARM == 3
      U = U + P_SIGN * P_EPS * double(P_DU(P_UC));
      P_REFLECT = P_REFLECT + (U<=0 || U>=1);
      U = mod(U,2); if U > 1, U = 2-U; end
      U = min(max(U,realmin),1-eps);
    end''')
    (observed / 'Systematic_Resampling.m').write_text(resample)
    spec = dict(cases=fixture['cases'], variants=variants(), direction_seeds=list(range(2026100601, 2026100605)))
    write_json(args.output / 'spec.json', spec)
    spec_lines = ['spec=struct();']
    for i, case in enumerate(spec['cases'], 1):
        spec_lines.append(f"spec.cases({i})=struct('case',{case['case']},'forcing',{case['forcing']:.17g},'seed',{case['seed']});")
    for i, v in enumerate(spec['variants'], 1):
        spec_lines.append(f"spec.variants({i})=struct('id',{oq(v['id'])},'arm',{oq(v['arm'])},'epsilon',{v['epsilon']:.17g},'direction',{v['direction']},'sign',{v['sign']});")
    spec_lines.append('spec.direction_seeds=['+' '.join(map(str,spec['direction_seeds']))+'];')
    spec_matlab = '\n'.join(spec_lines)
    driver = f'''1;
function putbin(path,value,kind)
 fid=fopen(path,'wb','ieee-le');assert(fid>0);fwrite(fid,value,kind);fclose(fid);
end
root={oq(args.root)};output={oq(args.output)};source={oq(source)};
baseline={oq(baseline)};capture={oq(args.reference)};stats={oq(stats)};
addpath(stats);addpath(fullfile(stats,'distributions'));addpath(source);cd(source);
driver=fileread(fullfile(source,'cSMC_VaryF.m'));
prefix=strsplit(driver,'% Vary F');assert(numel(prefix)==2);eval(prefix{{1}});
assert(d==8 && N==512 && T==100 && I==4);
assert(strcmp(Parameters.Adaptive,'yes') && strcmp(Parameters.Approximator,'quadratic'));
cd(output);addpath(fullfile(output,'observed'),'-begin');
assert(strcmp(which('mvnrnd'),fullfile(output,'observed/mvnrnd.m')));
assert(strcmp(which('Systematic_Resampling'),fullfile(output,'observed/Systematic_Resampling.m')));
assert(strcmp(which('cSMC_Resample'),fullfile(source,'cSMC_Resample.m')));
{spec_matlab}
global P_N P_U P_DN P_DU P_EPS P_SIGN P_ARM P_NC P_UC P_REFLECT;
dn=cell(4,1);du=cell(4,1);
for j=1:4
 rand('state',spec.direction_seeds(j));
 dn{{j}}=int8(2*(rand(N,d,505)>.5)-1);du{{j}}=int8(2*(rand(1,505)>.5)-1);
 putbin(fullfile(output,sprintf('direction%d_normal.bin',j-1)),permute(dn{{j}},[2,1,3]),'schar');
 putbin(fullfile(output,sprintf('direction%d_uniform.bin',j-1)),du{{j}},'schar');
end
rows={{}};
for ci=1:numel(spec.cases)
 cs=spec.cases(ci);F=cs.forcing;seed=cs.seed;
 Lorenz=@(x)((PT*x'-PP*x').*(P*x')-x'+F)';
 Transition.Mean=@(t,x)RungeKutta4(x,interval,Lorenz);
 rand('state',seed);randn('state',seed);P_N=zeros(N,d,505);P_U=zeros(1,505);
 for k=1:505, P_N(:,:,k)=randn(N,d);P_U(k)=rand(1);end
 putbin(fullfile(output,sprintf('case%d_normal.bin',cs.case)),permute(P_N,[2,1,3]),'double');
 putbin(fullfile(output,sprintf('case%d_uniform.bin',cs.case)),P_U,'double');
 original=load(fullfile(baseline,sprintf('author-F%.4f-seed%d.mat',F,seed)));
 savedrng=load(fullfile(capture,sprintf('author-F%.4f-seed%d.mat',F,seed)));
 for vi=1:numel(spec.variants)
  v=spec.variants(vi);P_EPS=v.epsilon;P_SIGN=v.sign;
  P_DN=dn{{v.direction+1}};P_DU=du{{v.direction+1}};
  P_ARM=0;if strcmp(v.arm,'normal'),P_ARM=1;elseif strcmp(v.arm,'uniform'),P_ARM=2;elseif strcmp(v.arm,'combined'),P_ARM=3;end
  P_NC=0;P_UC=0;P_REFLECT=0;rand('state',seed);randn('state',seed);
  row=struct('case',cs.case,'forcing',F,'seed',seed,'variant',v.id,'valid',false,'error','','stages',{{{{}}}});
  tic;result={{}};
  try
   result=cSMC_Resample(Parameters,Initial,Transition,Observation,InitialPolicy);
   row.valid=true;
  catch err
   row.error=err.message;
   if ~isempty(strfind(err.message,'assert')),rethrow(err);end
  end
  row.seconds=toc;row.normal_calls=P_NC;row.uniform_calls=P_UC;row.uniform_reflections=P_REFLECT;
  if row.valid
   row.valid=isreal(result{{end,2}}.LogNormConst) && all(isfinite(result{{end,2}}.LogNormConst));
   if ~row.valid,row.error='Nonreal or nonfinite source likelihood';end
  end
  if row.valid
   if vi==1
    assert(isequal(result,original.result));
    assert(isequal(rand('state'),savedrng.final_rand));assert(isequal(randn('state'),savedrng.final_randn));
    assert(isequal(InitialPolicy,result{{1,1}}));row.zero_control_exact=true;
   end
   for st=1:size(result,1)
    sm=result{{st,2}};sr=struct('stage',st-1,'logZ',sm.LogNormConst(end),'logZ_path',sm.LogNormConst,'min_ess',min(sm.ESS),'ancestry_changed',[]);
    if st<=size(original.result,1),sr.ancestry_changed=sum(sm.Ancestry(:)~=original.result{{st,2}}.Ancestry(:));end
    row.stages{{end+1}}=sr;
   end
   row.terminal_logZ=result{{end,2}}.LogNormConst(end);
   row.stage_count=size(result,1);row.adaptive_stop=row.stage_count<5;
  else
   row.terminal_logZ=[];row.stage_count=0;row.adaptive_stop=false;
  end
  rows{{end+1}}=row;save('-mat7-binary',fullfile(output,'source-rows.mat'),'rows');
  fprintf('PERTURB_SOURCE case=%d variant=%s valid=%d stages=%d seconds=%.3f\\n',cs.case,v.id,row.valid,row.stage_count,row.seconds);fflush(stdout);
 end
end
assert(numel(rows)==174);
'''
    (args.output / 'run-source.m').write_text(driver)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
    command = ['/usr/bin/octave-cli', '--no-gui', '--quiet', str(args.output / 'run-source.m')]
    with (args.output / 'octave.log').open('w') as log:
        subprocess.run(command, cwd=args.output, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1160)
    arrays = {}
    for case in fixture['cases']:
        for kind, shape in [('normal', [5,101,512,8]), ('uniform', [5,101])]:
            name = f"case{case['case']}_{kind}"
            file = args.output / (name + '.bin')
            arrays[name] = dict(file=file.name, shape=shape, dtype='float64', sha256=sha(file))
    for direction in range(4):
        for kind, shape in [('normal', [5,101,512,8]), ('uniform', [5,101])]:
            name = f'direction{direction}_{kind}'
            file = args.output / (name + '.bin')
            arrays[name] = dict(file=file.name, shape=shape, dtype='int8', sha256=sha(file))
    write_json(args.output / 'fixture.json', dict(schema='lorenz_perturbation_v1', arrays=arrays,
        capture_root=str(args.reference), capture_sha256=sha(args.reference/'fixture.json'), **spec))
    # Explicit independent-reference MAT decoding; never imported by the TF lane.
    import numpy as np
    from scipy.io import loadmat
    def plain(value):
        if isinstance(value,dict):
            return {k:plain(v) for k,v in value.items()}
        if isinstance(value,np.ndarray):
            return plain(value.tolist())
        if isinstance(value,np.generic):
            return value.item()
        if isinstance(value,(list,tuple)):
            return [plain(v) for v in value]
        return value
    rows = plain(loadmat(args.output/'source-rows.mat',simplify_cells=True)['rows'])
    for row in rows:
        row['valid']=bool(row['valid'])
        if isinstance(row['stages'],dict):
            row['stages']=[row['stages']]
        if not row['valid']:
            row['terminal_logZ']=None
        if not isinstance(row['error'],str):
            row['error']=''
    write_json(args.output/'source-rows.json',rows)
    assert len(rows) == 174
    assert all(r.get('zero_control_exact') for r in rows if r['variant'] == 'zero')
    summary = dict(status='PASS', scope='SOURCE_RANDOM_INPUT_SENSITIVITY', replication='INCOMPLETE',
        source_valid=sum(r['valid'] for r in rows), runs=len(rows), zero_controls_exact=6,
        cpu_only=True, gpu_intentionally_hidden=True, command=command, algorithm_seconds=time.monotonic()-start,
        fixture_sha256=sha(args.output/'fixture.json'), rows_sha256=sha(args.output/'source-rows.json'))
    write_json(args.output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def tf_run(args):
    start = time.monotonic()
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
    sys.path.insert(0, str(args.root))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    sources = args.output/'sources'
    for relative in ('bayesfilter/replication/controlled_smc_tf.py', 'bayesfilter/runtime/gpu_memory_policy.py'):
        assert sha(args.root/relative) == sha(sources/relative)
    implementation = sources/'bayesfilter/replication/multivariate_controlled_smc_tf.py'
    spec = importlib.util.spec_from_file_location('frozen_perturb_core', implementation)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    fixture = json.loads((args.reference/'fixture.json').read_text())
    capture = Path(fixture['capture_root'])
    assert sha(capture/'fixture.json') == fixture['capture_sha256']
    original = json.loads((capture/'fixture.json').read_text())
    dtype = tf.float64

    def load(directory, index, key):
        item = index['arrays'][key]
        path = directory/item['file']
        assert sha(path) == item['sha256'], key
        raw_type = tf.int8 if item.get('dtype') == 'int8' else dtype
        with tf.device('/CPU:0'):
            value = tf.io.decode_raw(tf.io.read_file(str(path)), raw_type, little_endian=True)
            return tf.cast(tf.reshape(value, item['shape']), dtype)

    def engine(forcing, solver):
        covariance = lambda value: [[value if i==j else 0. for j in range(8)] for i in range(8)]
        return core.MultivariateGaussianControlledSMC(observations=101, particles=512, replicas=1,
            dimension=8, observation_dimension=6, initial_mean=[0.]*8,
            initial_covariance=covariance(.01), transition_covariance=covariance(.001),
            transition_mean=lambda x: core.lorenz_rk4(x, tf.constant(forcing,dtype)),
            observation_log_density=lambda y,x: -.5/1e-4*tf.reduce_sum((y-x[...,:6])**2,-1),
            solver=solver, jit_compile=True, precision_policy='none', first_failure_guards=False)

    @tf.function(input_signature=[tf.TensorSpec([5,101,512,8],dtype),tf.TensorSpec([5,101],dtype),
        tf.TensorSpec([5,101,512,8],dtype),tf.TensorSpec([5,101],dtype),tf.TensorSpec([],dtype),
        tf.TensorSpec([],tf.int32)], jit_compile=True, autograph=False)
    def perturb(normal, uniform, dn, du, step, arm):
        normal = normal + tf.where((arm==1)|(arm==3),step,tf.zeros([],dtype))*dn
        candidate = uniform + step*du
        reflected = tf.math.floormod(candidate,tf.constant(2.,dtype))
        reflected = tf.where(reflected>1,2-reflected,reflected)
        reflected = tf.clip_by_value(reflected,tf.constant(sys.float_info.min,dtype),tf.constant(1.-sys.float_info.epsilon,dtype))
        uniform = tf.where((arm==2)|(arm==3),reflected,uniform)
        return normal,uniform

    rows = []
    resume_provenance = None
    if args.resume:
        prior_manifest=json.loads((args.resume/'manifest.json').read_text())
        assert prior_manifest['status'] in ('TIMED_OUT','FAILED'), 'Resume only a stopped attempt'
        prior_source=args.resume/'sources/bayesfilter/replication/multivariate_controlled_smc_tf.py'
        assert sha(prior_source)==sha(implementation), 'Numerical implementation changed'
        prior_rows=args.resume/'tf-rows.json'
        assert sha(prior_rows)==prior_manifest['output_sha256']['tf-rows.json']
        rows=json.loads(prior_rows.read_text())
        assert prior_manifest['command'][prior_manifest['command'].index('--reference')+1]==str(args.reference)
        resume_provenance=dict(output_root=str(args.resume),rows_sha256=sha(prior_rows),completed_rows=len(rows))
    completed={(r['case'],r['solver'],r['variant']) for r in rows}
    trace_counts = {}
    directions = [(load(args.reference,fixture,f'direction{d}_normal'),load(args.reference,fixture,f'direction{d}_uniform')) for d in range(4)]
    with tf.device('/GPU:0'):
        engines = {}
        for case in fixture['cases']:
            cid = case['case']
            normal = load(args.reference, fixture, f'case{cid}_normal')
            uniform = load(args.reference, fixture, f'case{cid}_uniform')
            initial = [load(capture,original,f'case{cid}_stage0_{k}')[:,None] for k in ('A','b','c')]
            y = load(capture,original,f'case{cid}_y')[:,None]
            initial_logz = load(capture,original,f'case{cid}_stage0_logZ')
            initial_ancestry = tf.cast(load(capture,original,f'case{cid}_stage0_ancestors'),tf.int32)
            zero_ancestry = {}
            for solver in ('source_normal_equations','centered_qr'):
                pending=[v for v in fixture['variants'] if (cid,solver,v['id']) not in completed]
                if not pending:
                    continue
                key=(case['forcing'],solver)
                if key not in engines:
                    engines[key]=engine(case['forcing'],solver)
                current=engines[key]
                # Reconstruct zero ancestry only for an interrupted case/solver;
                # retain its prior reported row and check the deterministic replay.
                selected=([fixture['variants'][0]] if pending[0]['id']!='zero' else [])+pending
                for variant in selected:
                    arm = {'zero':0,'normal':1,'uniform':2,'combined':3}[variant['arm']]
                    draws,u = perturb(normal,uniform,*directions[variant['direction']],
                        tf.constant(variant['sign']*variant['epsilon'],dtype),tf.constant(arm))
                    row = dict(case=cid, forcing=case['forcing'], seed=case['seed'], solver=solver,
                               variant=variant['id'],valid=True,error='',stages=[])
                    tick=time.monotonic()
                    policy=list(initial)
                    for stage in range(5):
                        if stage:
                            fit=current.fit(*policy,result['x'],result['log_incremental_weight'])
                            assert 'GPU:0' in fit['A'].device
                            if not bool(fit['valid'].numpy()):
                                row.update(valid=False,error='INVALID_FITTED_POLICY',failed_stage=stage)
                                break
                            policy=[fit[k] for k in ('A','b','c')]
                        result=current.forward(*policy,y,draws[stage,:,None],u[stage,:,None])
                        assert 'GPU:0' in result['x'].device
                        if not bool(result['valid'].numpy()):
                            row.update(valid=False,error='INVALID_FORWARD',failed_stage=stage)
                            break
                        logz=float(result['logZ'][-1,0].numpy())
                        minimum=float(tf.reduce_min(result['ess']).numpy())
                        ancestry=result['ancestors'][:,0]
                        change=None
                        if variant['id']=='zero':
                            zero_ancestry[solver,stage]=ancestry
                            if stage==0:
                                delta=tf.abs(result['logZ'][:,0]-initial_logz)
                                assert bool(tf.reduce_all(delta<=1e-7+1e-9*tf.abs(initial_logz)).numpy()),'Initial forward likelihood control failed'
                                assert bool(tf.reduce_all(ancestry==initial_ancestry).numpy()),'Initial ancestry control failed'
                        if (solver,stage) in zero_ancestry:
                            change=int(tf.reduce_sum(tf.cast(ancestry!=zero_ancestry[solver,stage],tf.int32)).numpy())
                        row['stages'].append(dict(stage=stage,logZ=logz,min_ess=minimum,ancestry_changed=change,
                            logZ_path=result['logZ'][:,0].numpy().tolist()))
                        if stage>=1 and minimum*100/512>90:
                            break
                    row.update(seconds=time.monotonic()-tick,stage_count=len(row['stages']),
                        terminal_logZ=row['stages'][-1]['logZ'] if row['valid'] else None,
                        adaptive_stop=row['valid'] and len(row['stages'])<5)
                    # Preserve the final fitted policy, even when the next forward failed.
                    with tf.device('/CPU:0'):
                        for name,value in zip(('A','b','c'),policy):
                            tf.io.write_file(str(args.output/f'case{cid}_{solver}_{variant["id"]}_{name}.tensor'),tf.io.serialize_tensor(value))
                    key=(cid,solver,variant['id'])
                    if key in completed:
                        prior=next(r for r in rows if (r['case'],r['solver'],r['variant'])==key)
                        assert row['valid']==prior['valid'] and row['terminal_logZ']==prior['terminal_logZ'] and row['stage_count']==prior['stage_count'], 'Resume zero replay changed'
                    else:
                        rows.append(row)
                    write_json(args.output/'tf-rows.json',rows)
                    print(json.dumps({k:row[k] for k in ('case','solver','variant','valid','error','stage_count','seconds')}),flush=True)
        for (forcing,solver),current in engines.items():
            for label in ('fit','forward'):
                count=getattr(current,label).experimental_get_tracing_count()
                assert count==1,'Unexpected retracing'
                trace_counts[f'{forcing}_{solver}_{label}']=count
    assert len(rows)==348
    summary=dict(status='PASS',scope='AUTONOMOUS_TF_RANDOM_INPUT_SENSITIVITY',replication='INCOMPLETE',
        runs=len(rows),valid=sum(r['valid'] for r in rows),gpu_memory_policy=memory,jit_compile=True,
        tf32=False,dtype='float64',trace_counts=trace_counts,perturb_trace_count=perturb.experimental_get_tracing_count(),
        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),algorithm_seconds=time.monotonic()-start,
        rows_sha256=sha(args.output/'tf-rows.json'),tensorflow_version=tf.__version__,tensorflow=tf.__version__,device='GPU:0',resume=resume_provenance,
        devices=[d.name for d in tf.config.list_physical_devices('GPU')])
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','output','reference'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-lock',type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--mode',choices=('source','tf'),required=True)
    args=parser.parse_args()
    (source_run if args.mode=='source' else tf_run)(args)


if __name__=='__main__':
    main()

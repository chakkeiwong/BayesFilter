"""Bounded diagnostic/qualification harness; never an algorithmic NumPy lane.

The pilot is calibration only. Source execution is original Octave code.
The port invokes the existing complete LorenzRefinement GPU/XLA controller.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
PLAN = 'docs/plans/controlled-smc-downstream-qualification-20261009.md'
RESULT = 'docs/plans/controlled-smc-downstream-qualification-result-20261009.md'
PYTHON = '/home/chakwong/anaconda3/envs/tftwogpu/bin/python'
GPU = 'GPU-68251639-fe82-8f81-3ccc-2953c32e805b'
FILES = (
    'bayesfilter/replication/controlled_smc_tf.py',
    'bayesfilter/replication/multivariate_controlled_smc_tf.py',
    'bayesfilter/replication/lorenz_refinement_tf.py',
    'bayesfilter/runtime/gpu_memory_policy.py',
    'docs/benchmarks/run_lorenz_downstream_qualification.py',
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def oq(value):
    return "'" + str(value).replace("'", "''") + "'"


def replace_once(value, old, new):
    if value.count(old) != 1:
        raise ValueError('Source anchor changed: ' + old)
    return value.replace(old, new)


def verify(output):
    spec = json.loads((output / 'spec.json').read_text())
    for path, digest in spec['candidate_sha256'].items():
        if sha(ROOT / path) != digest:
            raise ValueError('Frozen candidate changed: ' + path)
    for path, digest in spec['source_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen source/dependency changed: ' + path)
    return spec


def source(output):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    spec = verify(output)
    original = Path(spec['source_root'])
    stats = ROOT / '.localresources/code/octave-statistics-1.5.0/inst'
    observed = output / 'observed'
    observed.mkdir()
    sampler = (stats / 'distributions/mvnrnd.m').read_text()
    sampler = replace_once(sampler, '  s = randn(n,d)*U + mu;', '''  z = randn(n,d);
  global Q_N Q_NC;
  Q_NC = Q_NC + 1; assert(Q_NC <= 505);
  assert(isequal(z,Q_N(:,:,Q_NC)));
  s = z*U + mu;''')
    (observed / 'mvnrnd.m').write_text(sampler)
    resampling = replace_once((original / 'Systematic_Resampling.m').read_text(),
        '    U = rand(1);', '''    U = rand(1);
    global Q_U Q_UC;
    Q_UC = Q_UC + 1; assert(Q_UC <= 505);
    assert(isequal(U,Q_U(Q_UC)));''')
    (observed / 'Systematic_Resampling.m').write_text(resampling)
    cases = '\n'.join(f"cases({i})=struct('id',{r['id']},'forcing',{r['forcing']:.17g},'seed',{r['seed']},'control',{int(r['control'])});"
                      for i, r in enumerate(spec['cases'], 1))
    driver = f'''1;
function putbin(path,value)
 fid=fopen(path,'wb','ieee-le'); assert(fid>0); fwrite(fid,value,'double'); fclose(fid);
end
function putrows(path,value)
 save('-mat7-binary',path,'value');
end
output={oq(output)}; source={oq(original)}; stats={oq(stats)};
addpath(stats); addpath(fullfile(stats,'distributions')); addpath(source); cd(source);
prefix=strsplit(fileread(fullfile(source,'cSMC_VaryF.m')),'% Vary F'); assert(numel(prefix)==2); eval(prefix{{1}});
assert(d==8 && N==512 && T==100 && I==4 && size(Observation.y,1)==101);
assert(strcmp(Parameters.Adaptive,'yes') && strcmp(Parameters.Approximator,'quadratic'));
putbin(fullfile(output,'observations.bin'),Observation.y');
assert(strcmp(which('cSMC_Resample'),fullfile(source,'cSMC_Resample.m')));
global Q_N Q_U Q_NC Q_UC;
{cases}
rows={{}}; cd(output);
for ci=1:numel(cases)
 cs=cases(ci); F=cs.forcing;
 Lorenz=@(x)((PT*x'-PP*x').*(P*x')-x'+F)';
 Transition.Mean=@(t,x)RungeKutta4(x,interval,Lorenz);
 pristine={{}}; pristine_error='';
 if cs.control
  if ci>1, rmpath(fullfile(output,'observed')); clear mvnrnd Systematic_Resampling; end
  rand('state',cs.seed); randn('state',cs.seed);
  try, pristine=cSMC_Resample(Parameters,Initial,Transition,Observation,InitialPolicy);
  catch err, pristine_error=err.message; end
  pristine_rand=rand('state'); pristine_randn=randn('state');
  addpath(fullfile(output,'observed'),'-begin'); clear mvnrnd Systematic_Resampling;
 end
 assert(strcmp(which('mvnrnd'),fullfile(output,'observed/mvnrnd.m')));
 assert(strcmp(which('Systematic_Resampling'),fullfile(output,'observed/Systematic_Resampling.m')));
 rand('state',cs.seed); randn('state',cs.seed); Q_N=zeros(N,d,505); Q_U=zeros(1,505);
 for k=1:505, Q_N(:,:,k)=randn(N,d); Q_U(k)=rand(1); end
 if {int(spec.get('preserve_random_tapes', True))}
  putbin(fullfile(output,sprintf('trial%d_normal.bin',cs.id)),permute(Q_N,[2,1,3]));
  putbin(fullfile(output,sprintf('trial%d_uniform.bin',cs.id)),Q_U);
 end
 rand('state',cs.seed); randn('state',cs.seed); Q_NC=0; Q_UC=0;
 row=struct('id',cs.id,'forcing',F,'seed',cs.seed,'valid',false,'error','','terminal_logZ',[],'initial_logZ',[],'stage_count',0,'min_ess',[]);
 tic; result={{}};
 try, result=cSMC_Resample(Parameters,Initial,Transition,Observation,InitialPolicy);
 catch err, row.error=err.message; if ~isempty(strfind(err.message,'assert')), rethrow(err); end; end
 native_error=row.error; row.seconds=toc; row.normal_calls=Q_NC; row.uniform_calls=Q_UC;
 if ~isempty(result)
  row.valid=isreal(result{{end,2}}.LogNormConst) && all(isfinite(result{{end,2}}.LogNormConst));
  for ri=1:size(result,1)
   for side=1:2
    fields=fieldnames(result{{ri,side}});
    for fi=1:numel(fields)
     value=result{{ri,side}}.(fields{{fi}});
     if isnumeric(value), row.valid=row.valid && isreal(value) && all(isfinite(value(:))); end
    end
   end
   row.valid=row.valid && all(result{{ri,2}}.ESS>0) && all(result{{ri,2}}.ESS<=N*(1+1e-12));
  end
  row.validity_checks='all_stages_finite_ess_and_native_sampler_covariance';
  if row.valid
   row.initial_logZ=result{{1,2}}.LogNormConst(end); row.terminal_logZ=result{{end,2}}.LogNormConst(end);
   row.stage_count=size(result,1); row.min_ess=min(result{{end,2}}.ESS);
  else, row.error='Invalid source stage (nonreal, nonfinite or invalid ESS)'; end
 end
 if cs.control
  assert(isequaln(result,pristine)); assert(strcmp(native_error,pristine_error));
  assert(isequal(rand('state'),pristine_rand)); assert(isequal(randn('state'),pristine_randn));
  row.pristine_control_exact=true;
 end
 rows{{end+1}}=row; putrows(fullfile(output,'source-rows.mat'),rows);
 fprintf('SOURCE id=%d F=%.4f valid=%d stages=%d seconds=%.3f\\n',cs.id,F,row.valid,row.stage_count,row.seconds); fflush(stdout);
end
'''
    (output / 'source-driver.m').write_text(driver)
    command = ['/usr/bin/octave-cli', '--no-gui', '--quiet', str(output / 'source-driver.m')]
    with (output / 'octave.log').open('x') as log:
        subprocess.run(command, cwd=output, check=True, stdout=log, stderr=subprocess.STDOUT)
    # Independent source-reference decoding only; never imported by the TF worker.
    import numpy as np
    from scipy.io import loadmat
    def plain(value):
        if isinstance(value, dict):
            return {k:plain(v) for k,v in value.items()}
        if isinstance(value, np.ndarray):
            return plain(value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (list,tuple)):
            return [plain(v) for v in value]
        return value
    rows = plain(loadmat(output/'source-rows.mat', simplify_cells=True)['value'])
    if isinstance(rows, dict):
        rows = [rows]
    for row in rows:
        row['id'], row['seed'] = int(row['id']), int(row['seed'])
        row['valid'] = bool(row['valid'])
        if not isinstance(row['error'], str):
            row['error'] = ''
        if not row['valid']:
            row['terminal_logZ'], row['initial_logZ'], row['min_ess'] = None, None, None
    write(output/'source-rows.json',rows)
    if len(rows) != len(spec['cases']):
        raise ValueError('Incomplete source rows')
    arrays = {p.name: sha(p) for p in output.glob('*.bin')}
    write(output / 'source-summary.json', dict(status='PASS', runs=len(rows),
        valid=sum(bool(r['valid']) for r in rows), arrays_sha256=arrays,
        rows_sha256=sha(output / 'source-rows.json'), command=command,
        cpu_only=True, gpu_intentionally_hidden=True, octave_version=spec['octave_version'],
        pristine_controls=sum(bool(r.get('pristine_control_exact')) for r in rows)))


def port(output):
    spec = verify(output)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['BAYESFILTER_PRELOAD_CUSTOM_OP'] = '0'
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.lorenz_refinement_tf import LorenzRefinement
    source_summary = json.loads((output / 'source-summary.json').read_text())
    if sha(output / 'source-rows.json') != source_summary['rows_sha256']:
        raise ValueError('Source rows changed')
    source_rows = {r['id']: r for r in json.loads((output / 'source-rows.json').read_text())}

    def load(name, shape):
        path = output / name
        if sha(path) != source_summary['arrays_sha256'][name]:
            raise ValueError('Random tape/data changed: ' + name)
        with tf.device('/CPU:0'):
            return tf.reshape(tf.io.decode_raw(tf.io.read_file(str(path)), tf.float64), shape)

    observations = load('observations.bin', [101, 1, 6])
    engines, rows, trace_counts = {}, [], {}
    with tf.device('/GPU:0'):
        for case in spec['cases']:
            forcing = case['forcing']
            if forcing not in engines:
                engines[forcing] = LorenzRefinement(forcing=forcing, jit_compile=True)
            engine = engines[forcing]
            normal = load(f"trial{case['id']}_normal.bin", [5, 101, 1, 512, 8])
            uniform = load(f"trial{case['id']}_uniform.bin", [5, 101, 1])
            started = time.monotonic()
            result = engine.run(observations, normal, uniform, tf.constant(4, tf.int32))
            if 'GPU:0' not in result['history']['logZ'].device:
                raise RuntimeError('Port did not execute on GPU')
            valid = bool(result['valid'].numpy())
            iterations = int(result['iterations'].numpy())
            initial = float(result['history']['logZ'][0, -1, 0].numpy())
            terminal = float(result['history']['logZ'][iterations, -1, 0].numpy())
            minimum = float(tf.reduce_min(result['history']['ess'][iterations]).numpy())
            valid = valid and math.isfinite(terminal)
            reference = source_rows[case['id']]
            apf_difference = (initial - reference['initial_logZ']) if reference['valid'] else None
            apf_ok = apf_difference is None or abs(apf_difference) <= 1e-7 + 1e-9 * abs(reference['initial_logZ'])
            stop_ok = not valid or iterations == 4 or (iterations >= 1 and minimum * 100 / 512 > 90)
            row = dict(**case, valid=valid, terminal_logZ=terminal if valid else None,
                initial_logZ=initial if math.isfinite(initial) else None,
                stage_count=iterations + 1, min_ess=minimum if math.isfinite(minimum) else None,
                attempted_iterations=int(result['attempted_iterations'].numpy()),
                rejected_refit=bool(result['rejected_refit'].numpy()),
                seconds=time.monotonic()-started, apf_difference=apf_difference,
                apf_control_passed=apf_ok, stopping_control_passed=stop_ok)
            rows.append(row)
            write(output / 'tf-rows.json', rows)
            print(json.dumps(row, allow_nan=False), flush=True)
            if not apf_ok or not stop_ok:
                raise ValueError('Initial APF or stopping control failed')
            if not valid:
                # Artifact I/O is CPU-only and outside every numerical graph.
                with tf.device('/CPU:0'):
                    def save_failure(tree, prefix=''):
                        for name, value in tree.items():
                            label = prefix + name
                            if isinstance(value, dict):
                                save_failure(value, label + '-')
                            else:
                                tf.io.write_file(str(output / f"failed-{case['id']}-{label}.tensor"), tf.io.serialize_tensor(value))
                    save_failure(result)
                if spec.get('stop_on_additional_failure') and reference['valid']:
                    write(output / 'candidate-rejection.json', dict(
                        status='ADDITIONAL_TF_FAILURE', case=case, source=reference, port=row,
                        rows_sha256=sha(output/'tf-rows.json'), gpu_memory_policy=memory,
                        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'),
                        jit_compile=True, dtype='float64', tf32=False,
                        trace_count=engine.run.experimental_get_tracing_count(),
                        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
                        next_action='Q3: localize the first rejected fit/forward operation using preserved tapes and tensor diagnostics'))
                    raise SystemExit(20)
            count = engine.run.experimental_get_tracing_count()
            if count != 1:
                raise RuntimeError('Unexpected controller retracing')
            trace_counts[str(forcing)] = count
    verify(output)
    write(output / 'tf-summary.json', dict(status='PASS', runs=len(rows), valid=sum(r['valid'] for r in rows),
        rows_sha256=sha(output / 'tf-rows.json'), gpu_memory_policy=memory,
        gpu_allocator=tf.config.experimental.get_memory_info('GPU:0'), jit_compile=True,
        dtype='float64', tf32=False, trace_counts=trace_counts,
        tensorflow_version=tf.__version__, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
        gpu_devices=[d.name for d in tf.config.list_physical_devices('GPU')]))


def describe(log_source, log_port):
    """Planning diagnostics only; no confidence-interval qualification here."""
    n = len(log_source)
    if n < 8:
        return dict(pairs=n, status='INSUFFICIENT_VALID_PAIRS')
    ms, mp = statistics.mean(log_source), statistics.mean(log_port)
    vs, vp = statistics.variance(log_source), statistics.variance(log_port)
    offset = max(log_source + log_port)
    zs, zp = ([math.exp(x-offset) for x in values] for values in (log_source, log_port))
    ezs, ezp = statistics.mean(zs), statistics.mean(zp)
    likelihood_difference = math.log(ezp/ezs)
    likelihood_if = [b/ezp-a/ezs for a, b in zip(zs, zp)]
    likelihood_se = math.sqrt(statistics.variance(likelihood_if)/n)
    out = dict(pairs=n, status='CALIBRATION_ONLY', mean_logZ_source=ms, mean_logZ_port=mp,
        mean_likelihood_log_ratio=likelihood_difference,
        mean_likelihood_percent=100*math.expm1(likelihood_difference),
        likelihood_paired_influence_se=likelihood_se,
        max_likelihood_share_source=max(zs)/sum(zs), max_likelihood_share_port=max(zp)/sum(zp),
        source_logZ_range=[min(log_source), max(log_source)],
        source_variance_logZ=vs, port_variance_logZ=vp)
    if vs <= 0 or vp <= 0 or ezs <= 0:
        return dict(out, moment_status='ZERO_VARIANCE_UNRESOLVED')
    variance_if = [(b-mp)**2/vp-(a-ms)**2/vs for a, b in zip(log_source, log_port)]
    variance_se = math.sqrt(statistics.variance(variance_if)/n)
    source_kurtosis = statistics.mean([(x-ms)**4 for x in log_source])/(statistics.pvariance(log_source)**2)
    power_factor = statistics.NormalDist().inv_cdf(.95) + statistics.NormalDist().inv_cdf(.90)
    need_mean = math.ceil(power_factor**2 * 2*statistics.variance(zs)/ezs**2 / math.log(1.01)**2)
    need_variance = math.ceil(power_factor**2 * 2*max(0,source_kurtosis-1) / math.log(1.05)**2)
    out.update(variance_ratio=vp/vs, log_variance_ratio_paired_influence_se=variance_se,
        source_logZ_kurtosis=source_kurtosis,
        source_only_independent_arm_required_n=max(300,need_mean,need_variance),
        source_only_required_n_mean=need_mean, source_only_required_n_variance=need_variance,
        source_relative_variance_logZ=vs/ms**2 if ms else None,
        qualification=False, inference='Descriptive pilot; sample-size approximation assumes adequate moments and equal populations')
    return out


def analyze(output):
    spec = verify(output)
    a = json.loads((output / 'source-rows.json').read_text())
    b = json.loads((output / 'tf-rows.json').read_text())
    if len(a) != len(spec['cases']) or len(b) != len(a):
        raise ValueError('Incomplete comparison')
    if [r['id'] for r in a] != [r['id'] for r in b]:
        raise ValueError('Pair ordering mismatch')
    strata = []
    for forcing in sorted({r['forcing'] for r in a}):
        pairs = [(x,y) for x,y in zip(a,b) if x['forcing']==forcing]
        valid = [(x,y) for x,y in pairs if x['valid'] and y['valid']]
        source_valid = sum(bool(x['valid']) for x,y in pairs)
        extra = sum(bool(x['valid']) and not y['valid'] for x,y in pairs)
        row = dict(forcing=forcing, attempted=len(pairs), source_valid=source_valid,
            port_valid=sum(y['valid'] for x,y in pairs), extra_port_failures=extra,
            source_failures=[x['id'] for x,y in pairs if not x['valid']],
            extra_failure_ids=[x['id'] for x,y in pairs if x['valid'] and not y['valid']],
            zero_extra_failure_upper95=1-.05**(1/source_valid) if extra==0 and source_valid else None,
            source_seconds=sum(x['seconds'] for x,y in pairs),
            port_seconds=sum(y['seconds'] for x,y in pairs),
            source_stage_counts=[x['stage_count'] for x,y in pairs],
            port_stage_counts=[y['stage_count'] for x,y in pairs],
            continuous_estimand='conditional on both implementations being valid' if len(valid)!=len(pairs) else 'all sampled runs',
            metrics=describe([x['terminal_logZ'] for x,y in valid], [y['terminal_logZ'] for x,y in valid]))
        strata.append(row)
    extra = sum(r['extra_port_failures'] for r in strata)
    summary = dict(status='PASS', scope='COMPLETE_ADAPTIVE_CSMC_CALIBRATION', replication='INCOMPLETE',
        scientific_verdict='ADDITIONAL_PORT_FAILURE_REQUIRES_LOCALIZATION' if extra else 'CALIBRATION_COMPLETE_QUALIFICATION_OPEN',
        qualification=False, strata=strata, spec_sha256=sha(output/'spec.json'),
        source_rows_sha256=sha(output/'source-rows.json'), tf_rows_sha256=sha(output/'tf-rows.json'))
    write(output / 'summary.json', summary)
    return summary


def pilot(output):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_controlled_smc_kdm as master
    current = master.status()
    if current['running']:
        raise RuntimeError('Existing campaign run: ' + repr(current['running']))
    previous = current.get('csmc_qualification', [])
    if len(previous) >= 3:
        raise RuntimeError('Three-attempt pilot limit reached')
    caps = {r: 3600-sum(v['used_seconds'][r] for v in previous) for r in ('cpu','gpu')}
    for resource in ('cpu','gpu'):
        if caps[resource] <= 0 or current['tranche_remaining_seconds'][resource] < caps[resource]:
            raise RuntimeError('Insufficient remaining pilot budget')
    output.mkdir(parents=False, exist_ok=False)
    lock_root = BASE / 'attempt139_c0_source_lock'
    lock = json.loads((lock_root / 'source-lock.json').read_text())
    if lock['status'] != 'PASS':
        raise ValueError('Invalid author source lock')
    source_root = lock_root / 'author-source/StateSpaceModels/Lorenz'
    hashes = {}
    for item in lock['required_files']['lorenz']:
        path = source_root / Path(item['path']).name
        if sha(path) != item['sha256']:
            raise ValueError('Source lock mismatch')
        hashes[str(path)] = item['sha256']
    for item in lock['runtime_dependencies']:
        path = ROOT / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Runtime dependency mismatch')
        hashes[str(path)] = item['sha256']
    candidates = {p: sha(ROOT/p) for p in FILES}
    for relative in FILES:
        target = output / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    cases = [dict(id=i*32+j, forcing=forcing, seed=261009001+i*10000+j, control=j==0)
             for i, forcing in enumerate((2.5,4.8801,8.5)) for j in range(32)]
    spec = dict(schema='csmc_downstream_pilot_v1', phase='CALIBRATION_NOT_QUALIFICATION',
        source_root=str(source_root), source_sha256=hashes, candidate_sha256=candidates,
        octave_version=lock['octave'], candidate='LorenzRefinement.run/centered_qr/none/FP64/GPU/XLA',
        cases=cases, particles=512, observations=101, maximum_refinements=4,
        seed_policy='independent disjoint blocks, paired native source tapes',
        plan=PLAN, phase_budget_seconds=caps, gpu_uuid=GPU)
    write(output / 'spec.json', spec)
    manifest = dict(status='RUNNING', stage='CSMC_DOWNSTREAM_CALIBRATION', resource='mixed',
        started_unix_seconds=time.time(), supervisor_pid=os.getpid(),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        command=[PYTHON,'-B',str(Path(__file__).resolve()),'pilot','--output',str(output)],
        environment=PYTHON, output_root=str(output), plan=PLAN, result_file=RESULT,
        executions=[], charged_seconds=dict(cpu=0.,gpu=0.), spec_sha256=sha(output/'spec.json'))
    write(output / 'manifest.json', manifest)
    master.refresh_checkpoint()
    try:
        for mode, resource in (('source','cpu'),('tf','gpu')):
            verify(output)
            command = [PYTHON,'-B',str(Path(__file__).resolve()),mode,'--output',str(output)]
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='-1' if mode=='source' else GPU,
                TF_FORCE_GPU_ALLOW_GROWTH='true', BAYESFILTER_PRELOAD_CUSTOM_OP='0',
                OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
            started = time.monotonic()
            with (output / (mode+'.log')).open('x') as log:
                child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                execution = dict(mode=mode, resource=resource, command=command,
                    pid=child.pid, started_unix_seconds=time.time(), timeout_seconds=caps[resource],
                    status='RUNNING', cpu_only=mode=='source', gpu_intentionally_hidden=mode=='source')
                manifest['executions'].append(execution)
                write(output / 'manifest.json', manifest)
                master.refresh_checkpoint()
                try:
                    code = child.wait(timeout=caps[resource])
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL)
                        child.wait()
                    code = -1
                execution.update(status='COMPLETE' if code==0 else 'FAILED', returncode=code,
                    wall_seconds=time.monotonic()-started)
                manifest['charged_seconds'][resource] += execution['wall_seconds']
                write(output / 'manifest.json', manifest)
                if code:
                    raise RuntimeError(mode+' worker failed; see '+str(output/(mode+'.log')))
        summary = analyze(output)
        manifest.update(status='COMPLETE', decision={k:summary[k] for k in ('status','scope','replication','scientific_verdict')})
    except BaseException as exc:
        manifest.update(status='FAILED', error=str(exc), traceback=traceback.format_exc())
        if not (output/'summary.json').exists():
            write(output/'summary.json',dict(status='FAILED',scope='CSMC_DOWNSTREAM_CALIBRATION',replication='INCOMPLETE',error=str(exc)))
    finally:
        manifest.update(wall_seconds=time.time()-manifest['started_unix_seconds'], ended_unix_seconds=time.time())
        write(output/'manifest.json',manifest)
        ledger_path = BASE/'reference-diagnostics.json'
        ledger = json.loads(ledger_path.read_text())
        next_action = ('Review the complete calibration, localize any additional failures, and cost a frozen untouched qualification; exact coefficients are diagnostic only.'
                       if manifest['status']=='COMPLETE' else 'Repair the localized calibration harness/resource failure and retry within the unchanged pilot budget; preserve prior evidence.')
        for resource in ('cpu','gpu'):
            ledger.append(dict(id=output.name+'_'+resource,resource=resource,
                charged_seconds=manifest['charged_seconds'][resource],
                summary=str((output/'summary.json').relative_to(ROOT)),summary_sha256=sha(output/'summary.json'),
                result_file=RESULT,plan_file=PLAN,stage='CSMC_DOWNSTREAM_CALIBRATION_'+manifest['status'],
                next_action=next_action,after_attempt=current['attempts'],
                executions=[dict(path=str(output.relative_to(ROOT)),resource=resource,wall_seconds=manifest['charged_seconds'][resource])]))
        write(ledger_path,ledger)
        master.refresh_checkpoint()
    print(json.dumps({k:manifest[k] for k in ('status','output_root','charged_seconds')}),flush=True)
    return 0 if manifest['status']=='COMPLETE' else 1



def source_grid(output):
    """Independent source-only calibration in bounded, isolated 32-run cells."""
    sys.path.insert(0, str(ROOT/'scripts'))
    import run_controlled_smc_kdm as master
    current = master.status()
    if current['running']:
        raise RuntimeError('Existing campaign run')
    if current['tranche_remaining_seconds']['cpu'] < 7200:
        raise RuntimeError('Insufficient source calibration budget')
    output.mkdir(parents=False, exist_ok=False)
    template = json.loads((BASE/'csmc-qualification-20261009-165753/spec.json').read_text())
    template['candidate_sha256'] = {p:sha(ROOT/p) for p in FILES}
    template.update(phase='SOURCE_ONLY_FULL_GRID_CALIBRATION', preserve_random_tapes=False)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    grid_text = subprocess.check_output(['/usr/bin/octave-cli','--quiet','--eval',
        "fprintf(' %.17g',linspace(2.5,8.5,40));"],env=env,text=True)
    grid = [float(v) for v in grid_text.split()] + [4.8801]
    if len(grid)!=41:
        raise ValueError('Unexpected source grid')
    manifest = dict(status='RUNNING',stage='CSMC_SOURCE_GRID_CALIBRATION',resource='mixed',
        started_unix_seconds=time.time(),supervisor_pid=os.getpid(),git_commit=current.get('git_commit'),
        command=[PYTHON,'-B',str(Path(__file__).resolve()),'source-grid','--output',str(output)],
        environment=PYTHON,output_root=str(output),plan=PLAN,result_file=RESULT,
        executions=[],charged_seconds=dict(cpu=0.,gpu=0.),
        maximum_workers=4,phase_budget_seconds=dict(cpu=7200,gpu=0))
    manifest['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    specs=[]
    for i,forcing in enumerate(grid):
        child_output=output/f'cell{i:02d}'
        child_output.mkdir()
        spec=dict(template,cases=[dict(id=i*32+j,forcing=forcing,seed=261020001+i*10000+j,control=j==0)
                                 for j in range(32)],source_grid_index=i if i<40 else None)
        write(child_output/'spec.json',spec)
        specs.append(child_output)
    write(output/'spec.json',dict(schema='csmc_source_grid_calibration_v1',grid=grid,
        independent_runs_per_cell=32,source_grid_command=['octave-cli','linspace(2.5,8.5,40)'],
        child_spec_sha256={p.name:sha(p/'spec.json') for p in specs},candidate_sha256=template['candidate_sha256'],
        qualification=False,plan=PLAN))
    write(output/'manifest.json',manifest)
    master.refresh_checkpoint()

    def worker(child_output):
        command=[PYTHON,'-B',str(Path(__file__).resolve()),'source','--output',str(child_output)]
        started=time.monotonic()
        with (child_output/'worker.log').open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=child.wait(timeout=150)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGTERM)
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait()
                code=-1
        return dict(mode='source',resource='cpu',command=command,pid=child.pid,
                    wall_seconds=time.monotonic()-started,status='COMPLETE' if code==0 else 'FAILED',
                    returncode=code,output=str(child_output),cpu_only=True,gpu_intentionally_hidden=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(worker,p):p for p in specs}
            for future in concurrent.futures.as_completed(futures):
                execution=future.result()
                manifest['executions'].append(execution)
                manifest['charged_seconds']['cpu']+=execution['wall_seconds']
                write(output/'manifest.json',manifest)
                print(json.dumps({k:execution[k] for k in ('output','status','wall_seconds')}),flush=True)
        if any(e['status']!='COMPLETE' for e in manifest['executions']):
            raise RuntimeError('One or more isolated source calibration cells failed; inspect worker logs')
        strata=[]
        for child_output in specs:
            rows=json.loads((child_output/'source-rows.json').read_text())
            values=[r['terminal_logZ'] for r in rows if r['valid']]
            strata.append(dict(cell=child_output.name,forcing=rows[0]['forcing'],attempted=len(rows),
                source_valid=len(values),failure_ids=[r['id'] for r in rows if not r['valid']],
                median_seconds=statistics.median([r['seconds'] for r in rows]),
                rows_sha256=sha(child_output/'source-rows.json'),metrics=describe(values,values)))
        summary=dict(status='PASS',scope='SOURCE_ONLY_ORIGINAL_GRID_CALIBRATION',replication='INCOMPLETE',
                     scientific_verdict='SOURCE_GRID_CALIBRATION_COMPLETE_QUALIFICATION_OPEN',
                     qualification=False,strata=strata)
        write(output/'summary.json',summary)
        manifest['status']='COMPLETE'
    except BaseException as exc:
        manifest.update(status='FAILED',error=str(exc),traceback=traceback.format_exc())
        write(output/'summary.json',dict(status='FAILED',scope='SOURCE_ONLY_GRID_CALIBRATION',replication='INCOMPLETE',error=str(exc)))
    finally:
        manifest.update(wall_seconds=time.time()-manifest['started_unix_seconds'],ended_unix_seconds=time.time())
        write(output/'manifest.json',manifest)
        ledger_path=BASE/'reference-diagnostics.json'
        ledger=json.loads(ledger_path.read_text())
        ledger.append(dict(id=output.name+'_cpu',resource='cpu',charged_seconds=manifest['charged_seconds']['cpu'],
            summary=str((output/'summary.json').relative_to(ROOT)),summary_sha256=sha(output/'summary.json'),
            plan_file=PLAN,result_file=RESULT,stage='CSMC_SOURCE_GRID_CALIBRATION_'+manifest['status'],
            next_action='Review full-grid source cost/moments, freeze an affordable untouched paired qualification with unchanged margins, then execute; report any precision limit.',
            after_attempt=current['attempts'],executions=[dict(path=str(output.relative_to(ROOT)),resource='cpu',wall_seconds=manifest['charged_seconds']['cpu'])]))
        write(ledger_path,ledger)
        master.refresh_checkpoint()
    print(json.dumps(dict(status=manifest['status'],charged_seconds=manifest['charged_seconds'],output_root=str(output))),flush=True)
    return 0 if manifest['status']=='COMPLETE' else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('pilot','source-grid','source','tf','analyze'))
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.parent != BASE and not (args.mode=='source' and output.parent.parent==BASE):
        parser.error('Output must be a unique directory under the campaign root')
    if args.mode in ('pilot','source-grid'):
        raise SystemExit((pilot if args.mode=='pilot' else source_grid)(output))
    {'source':source,'tf':port,'analyze':analyze}[args.mode](output)


if __name__=='__main__':
    main()

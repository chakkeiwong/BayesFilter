"""Independent CPU reference: capture native author random draws, without refitting changes.

NumPy/SciPy are confined to reference MAT decoding and binary fixture export.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def instrument_sampler(text):
    old = '  s = randn(n,d)*U + mu;'
    new = ("  p0_normal = randn(n,d);\n"
           "  s = p0_normal*U + mu;\n"
           "  global P0_NORMAL_DRAWS;\n"
           "  P0_NORMAL_DRAWS{end+1} = p0_normal;")
    if text.count(old) != 1:
        raise ValueError('Unexpected reference mvnrnd operation')
    return text.replace(old, new)


def instrument_resampling(text):
    old = '    U = rand(1);'
    if text.count(old) != 1:
        raise ValueError('Unexpected reference systematic uniform')
    return text.replace(old, old + '\n    global P0_UNIFORM_DRAWS;\n'
                        '    P0_UNIFORM_DRAWS{end+1} = U;')


def octave_quote(path):
    return "'" + str(path).replace("'", "''") + "'"


def driver_text(root, output, baseline, source):
    """Run unchanged and observed copies in separate Octave processes."""
    return f"""args=argv(); arm=args{{1}};
root={octave_quote(root)}; output={octave_quote(output)}; baseline={octave_quote(baseline)};
source={octave_quote(source)};
stats=fullfile(root,'.localresources/code/octave-statistics-1.5.0/inst');
addpath(stats);addpath(fullfile(stats,'distributions'));addpath(source);cd(source);
driver=fileread(fullfile(source,'cSMC_VaryF.m'));
prefix=strsplit(driver,'% Vary F');assert(numel(prefix)==2);eval(prefix{{1}});
cd(output);
if strcmp(arm,'observed')
 addpath(fullfile(output,'observed'),'-begin');
 assert(strcmp(which('mvnrnd'),fullfile(output,'observed/mvnrnd.m')));
 assert(strcmp(which('Systematic_Resampling'),fullfile(output,'observed/Systematic_Resampling.m')));
else
 assert(strcmp(which('mvnrnd'),fullfile(stats,'distributions/mvnrnd.m')));
 assert(strcmp(which('Systematic_Resampling'),fullfile(source,'Systematic_Resampling.m')));
end
assert(strcmp(which('cSMC_Resample'),fullfile(source,'cSMC_Resample.m')));
assert(d==8 && T==100 && N==512 && I==4);
assert(strcmp(Parameters.Approximator,'quadratic') && strcmp(Parameters.Adaptive,'yes'));
global P0_NORMAL_DRAWS P0_UNIFORM_DRAWS;
for F=[2.5,4.8801,8.5]
 Lorenz=@(x)((PT*x'-PP*x').*(P*x')-x'+F)';
 Transition.Mean=@(t,x)RungeKutta4(x,interval,Lorenz);
 for seed=[20260929,20260930]
  rand('state',seed);randn('state',seed);
  P0_NORMAL_DRAWS={{}};P0_UNIFORM_DRAWS={{}};
  result=cSMC_Resample(Parameters,Initial,Transition,Observation,InitialPolicy);
  filename=sprintf('author-F%.4f-seed%d.mat',F,seed);
  original=load(fullfile(baseline,filename));assert(isequal(result,original.result));
  final_rand=rand('state');final_randn=randn('state');
  if strcmp(arm,'unchanged')
   save('-mat7-binary',fullfile(output,['rng-' filename]),'final_rand','final_randn');
  else
   saved=load(fullfile(output,['rng-' filename]));
   assert(isequal(final_rand,saved.final_rand) && isequal(final_randn,saved.final_randn));
   assert(numel(P0_NORMAL_DRAWS)==size(result,1)*(T+1));
   assert(numel(P0_UNIFORM_DRAWS)==size(result,1)*(T+1));
   normal=cat(3,P0_NORMAL_DRAWS{{:}});uniform=cell2mat(P0_UNIFORM_DRAWS);
   save('-mat7-binary',fullfile(output,filename),'normal','uniform','final_rand','final_randn');
  end
  fprintf('P0_CAPTURE arm=%s F=%.4f seed=%d stages=%d exact=1\\n',arm,F,seed,size(result,1));
 end
end
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--source-lock', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    import numpy as np
    from scipy.io import loadmat
    lock = json.loads((args.source_lock/'source-lock.json').read_text())
    assert lock['status'] == 'PASS'
    source = args.source_lock/'author-source/StateSpaceModels/Lorenz'
    for item in lock['verified_snapshot_files']:
        relative = Path(item['path']).relative_to('.localresources/code/heng-controlled-smc')
        assert sha(args.source_lock/'author-source'/relative) == item['sha256']
    for item in lock['runtime_dependencies']:
        assert sha(args.root/item['path']) == item['sha256']
    baseline = json.loads((args.reference/'summary.json').read_text())
    assert baseline['status'] == 'PASS'
    for name in ('cSMC_Resample.m','cSMC_VaryF.m','RungeKutta4.m','Systematic_Resampling.m','SimulatedData.mat'):
        assert sha(source/name) == sha(args.reference/'sources'/name), name
    for key, expected in baseline['dependency_sha256'].items():
        assert sha(Path(baseline[key])) == expected, key
    observed = args.output/'observed'; observed.mkdir()
    sampler = args.root/'.localresources/code/octave-statistics-1.5.0/inst/distributions/mvnrnd.m'
    for path, instrument in ((sampler,instrument_sampler),(source/'Systematic_Resampling.m',instrument_resampling)):
        original = path.read_text(); modified = instrument(original)
        (observed/path.name).write_text(modified)
        (args.output/(path.stem+'.diff')).write_text(''.join(difflib.unified_diff(
            original.splitlines(True),modified.splitlines(True),fromfile=str(path),tofile='observed/'+path.name)))
    driver = args.output/'capture.m'
    driver.write_text(driver_text(args.root,args.output,args.reference,source))
    commands = []
    def address_limit():
        resource.setrlimit(resource.RLIMIT_AS, (8*1024**3,8*1024**3))
    for arm in ('unchanged','observed'):
        command = ['octave','--no-gui','--quiet',str(driver),arm]
        commands.append(command)
        with (args.output/(arm+'.log')).open('x') as log:
            subprocess.run(command,cwd=args.output,stdout=log,stderr=subprocess.STDOUT,
                           check=True,timeout=max(1,75-(time.monotonic()-started)),preexec_fn=address_limit)
    index = dict(schema='lorenz_forward_reference_v1',cases=[],arrays={},reference_root=str(args.reference),
                 source_sha256=sha(source/'cSMC_Resample.m'),sampler_sha256=sha(sampler),
                 source_results_exactly_unchanged=True,source_rng_exactly_unchanged=True,
                 sampler_covariance_rule="Sigma + eps*norm(Sigma,'fro')*I before Cholesky; original eigen fallback")
    def save(name,value):
        assert np.isrealobj(value)
        array = np.ascontiguousarray(value,dtype='<f8')
        assert np.isfinite(array).all()
        path = args.output/(name+'.bin'); path.write_bytes(array.tobytes())
        index['arrays'][name] = dict(file=path.name,shape=list(array.shape),sha256=sha(path))
    for forcing in (2.5,4.8801,8.5):
        for seed in (20260929,20260930):
            filename = 'author-F%.4f-seed%d.mat' % (forcing,seed)
            ref = loadmat(args.reference/filename,simplify_cells=True)
            tape = loadmat(args.output/filename,simplify_cells=True)
            result = ref['result']; count = result.shape[0]; case = len(index['cases'])
            assert tape['normal'].shape == (512,8,101*count)
            assert tape['uniform'].shape == (101*count,)
            save(f'case{case}_y',ref['y'])
            for stage in range(count):
                policy, smc = [v if isinstance(v,dict) else vars(v) for v in result[stage]]
                prefix = f'case{case}_stage{stage}_'
                for key in ('A','b','c'): save(prefix+key,policy[key])
                save(prefix+'x',np.transpose(smc['Trajectory'],(2,0,1)))
                save(prefix+'ancestors',smc['Ancestry'].T-1)
                for key, source_key in (('lw','LogIncrementalWeight'),('logZ','LogNormConst'),('ess','ESS')):
                    value=smc[source_key]; save(prefix+key,value.T if key=='lw' else value)
                save(prefix+'normal',np.transpose(tape['normal'][:,:,stage*101:(stage+1)*101],(2,0,1)))
                save(prefix+'uniform',tape['uniform'][stage*101:(stage+1)*101])
            index['cases'].append(dict(case=case,forcing=forcing,seed=seed,stages=count,
                reference_archive_sha256=sha(args.reference/filename),capture_archive_sha256=sha(args.output/filename)))
    assert len(index['cases'])==6 and sum(c['stages'] for c in index['cases'])==22
    (args.output/'fixture.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    summary=dict(status='PASS',scope='AUTHOR_FORWARD_DRAW_CAPTURE',replication='INCOMPLETE',
        source_results_exactly_unchanged=True,source_rng_exactly_unchanged=True,cases=6,stages=22,
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='INDEPENDENT_REFERENCE_ONLY',
        commands=commands,algorithm_seconds=time.monotonic()-started,fixture_sha256=sha(args.output/'fixture.json'))
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary))


if __name__=='__main__': main()

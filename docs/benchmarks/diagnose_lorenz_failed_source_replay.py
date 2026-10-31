"""Independent CPU source replay; NumPy/SciPy only for reference I/O and analysis."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.io import loadmat, savemat
import tensorflow as tf


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--mode', choices=['failed_replay'], required=True)
    args = p.parse_args()
    started = time.monotonic()
    checks = []
    def flag(name, good):
        checks.append(dict(name=name, passed=bool(good)))
        if not good:
            raise RuntimeError(name)
    source_path = args.output/'sources/cSMC_Resample.m'
    source = source_path.read_text()
    start = source.index('    switch Approximator')
    end = source.index('    % Pre-allocate\n    cSMC = cell')
    body = source[start:end]
    head = source[source.index('    N = Parameters.Particles;'):
                  source.index('    function SMC = RunSMC')]
    entry = '    function NewPolicy = ApproxDP(SMC,Policy)\n'
    global_insert = '        global LORENZ_REPLAY_AUDIT;\n'
    anchor = '            UpdatedA = squeeze(Policy.A(t,:,:)) + AHat;\n'
    guard = """            if t==1; AuditBase=Initial.Precision; else; AuditBase=Transition.Ginv; end
            AuditPrecision=AuditBase+2*UpdatedA;
            assert(isreal(AuditPrecision) && all(isfinite(AuditPrecision(:))));
            AuditEigen=eig(0.5*(AuditPrecision+AuditPrecision'));
            AuditMargin=min(AuditEigen)/max(abs(AuditEigen));
            LORENZ_REPLAY_AUDIT{end+1}=struct('time',t-1,'min_eigenvalue',min(AuditEigen),...
                'margin',AuditMargin,'BetaHat',BetaHat,'Value',Value,'UpdatedA',UpdatedA,...
                'gram_rcond',rcond(DesignMatrix'*DesignMatrix));
            if AuditMargin<=64*eps; error('LorenzReplay:InvalidPrecision','Invalid Gaussian precision'); end
"""
    flag('unique_extraction_anchors', body.count(entry)==1 and body.count(anchor)==1)
    modified = body.replace(entry, entry+global_insert).replace(anchor, anchor+guard)
    flag('only_observability_and_guard', modified.replace(global_insert,'').replace(guard,'')==body)
    wrapper = 'function Output=author_lorenz_fit_replay(Parameters,Initial,Transition,SMC,Policy)\n'+head+modified
    wrapper += '    Output=ApproxDP(SMC,Policy);\nend\n'
    (args.output/'sources/author_lorenz_fit_replay.m').write_text(wrapper)
    baseline_manifest = json.loads((args.reference/'manifest.json').read_text())
    source_hashes = baseline_manifest['source_sha256']
    flag('unchanged_author_source', sha(source_path) in source_hashes.values())
    inputs = {}
    tensors = {}
    for key in ('A','b','c','x','log_incremental_weight'):
        path = args.candidate/'rejected_batch'/(key+'.tensor')
        inputs[str(path)] = sha(path)
        tensors[key] = tf.io.parse_tensor(path.read_bytes(),tf.float64).numpy()
    fixture = json.loads(args.fixture.read_text())['metadata']
    diag = json.loads((args.candidate/'failure-diagnostics.json').read_text())
    flag('frozen_failed_scope', diag['grid_index']==0 and diag['counts'][0]==3)
    data = dict(stage_count=3, row_index=0, forcing=2.5,
                initial_covariance=np.asarray(fixture['initial_covariance']),
                transition_covariance=np.asarray(fixture['transition_covariance']))
    for stage in range(1,4):
        data['case%d'%stage] = {key:tensors[key][stage-1,:,0] for key in tensors}
        data['case%d'%stage]['expected_A'] = tensors['A'][stage,:,0]
        data['case%d'%stage]['expected_b'] = tensors['b'][stage,:,0]
        data['case%d'%stage]['expected_c'] = tensors['c'][stage,:,0]
    savemat(args.output/'input.mat',data)
    command=['octave','--no-gui','--quiet',
             str(args.output/'sources/reference_lorenz_failed_source_replay.m'),
             str(Path.cwd()),str(args.output),str(args.reference)]
    subprocess.run(command, check=True)
    result = loadmat(args.output/'replay.mat', simplify_cells=True)
    flag('all_healthy_native_fits_exact', int(result['healthy_count'])==16 and bool(result['healthy_exact']))
    flags = np.atleast_1d(result['candidate_valid']).astype(bool)
    failure_times = np.atleast_1d(result['failure_time']).astype(int)
    minimum_eigen = np.atleast_1d(result['minimum_eigenvalue'])
    flag('all_failed_cloud_replays_recorded', len(flags)==3 and np.all(np.isfinite(minimum_eigen)))
    saved_qr_eigen = []
    for stage in range(1,4):
        a = tensors['A'][stage,:,0]
        base = np.broadcast_to(np.linalg.inv(data['transition_covariance']),a.shape).copy()
        base[0] = np.linalg.inv(data['initial_covariance'])
        eig = np.linalg.eigvalsh(base+2*a)
        saved_qr_eigen.append(float(np.min(eig)))
    traces = result['candidate_traces']
    rows=[]
    for i in range(3):
        rows.append(dict(stage=i+1, source_valid=bool(flags[i]), failure_time=int(failure_times[i]),
            source_minimum_eigenvalue=float(minimum_eigen[i]),
            saved_tf_qr_minimum_eigenvalue=saved_qr_eigen[i],
            evaluated_times=int(np.atleast_1d(result['evaluated_times'])[i])))
    x2fx = Path(str(result['x2fx_path']))
    summary=dict(status='PASS',scope='LORENZ_FAILED_CLOUD_AUTHOR_REPLAY',replication='INCOMPLETE',
        scientific_verdict=('AUTHOR_SOURCE_ALSO_FAILS_ON_FROZEN_TF_INPUTS' if not np.all(flags)
                            else 'AUTHOR_SOURCE_VALID_ON_FROZEN_TF_INPUTS'),
        cpu_only=True,gpu_intentionally_hidden=True,numpy_role='INDEPENDENT_REFERENCE_ONLY',
        checks=checks,healthy_native_fits_exact=int(result['healthy_count']),rows=rows,
        source_sha256=sha(source_path),generated_wrapper_sha256=sha(args.output/'sources/author_lorenz_fit_replay.m'),
        extracted_body_sha256=hashlib.sha256(body.encode()).hexdigest(),x2fx_sha256=sha(x2fx),
        input_sha256=inputs,command=command,octave_version=str(result['octave_version']),
        algorithm_seconds=time.monotonic()-started,
        nonclaims=['No MATLAB equivalence','No new filter efficacy or default-readiness evidence',
                   'Source summary reconstruction on fixed TF inputs, not original source random trajectory'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='PASS',healthy_exact=16,rows=rows,scientific_verdict=summary['scientific_verdict'])))


if __name__=='__main__':
    main()

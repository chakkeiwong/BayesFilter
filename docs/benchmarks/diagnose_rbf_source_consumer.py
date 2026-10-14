"""Independent CPU-only source audit; SciPy is only a diagnostic MAT reader."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import difflib
import json
from pathlib import Path
import subprocess
import time


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items() if not k.startswith('__')}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'_fieldnames'):return {k:plain(getattr(value,k)) for k in value._fieldnames}
    if hasattr(value,'tolist'):return plain(value.tolist())
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output;started=time.monotonic();snapshot=output/'sources'
    source=(snapshot/'cSMC_Resample.m').read_text()
    probability='DiscreteSample(NormalisedPolicyWeights(n,:), 1)'
    initial='        SMC.Trajectory(:,1) = X;'
    later='            SMC.Trajectory(:,t+1) = X;'
    assert source.count(probability)==source.count(initial)==source.count(later)==1
    for name,repair in [('observed',False),('corrected',True)]:
        edited=source.replace('function cSMC = cSMC_Resample(',f'function cSMC = cSMC_Resample_{name}(',1)
        edited=edited.replace(initial,initial+'\n        SMC.AuditComponents(:,1) = SelectComponents(:);')
        actual='NormalisedPolicyWeights(Ancestor,:)' if repair else 'NormalisedPolicyWeights'
        extra='\n            SMC.AuditComponents(:,t+1) = SelectComponents(:);\n'
        extra+=f'            SMC.AuditComponentWeightsTV(:,t) = .5 * sum(abs({actual} - NormalisedPolicyWeights(Ancestor,:)),2);'
        edited=edited.replace(later,later+extra)
        if repair:edited=edited.replace(probability,'DiscreteSample(NormalisedPolicyWeights(Ancestor(n),:), 1)')
        (snapshot/f'cSMC_Resample_{name}.m').write_text(edited)
        (output/f'{name}.diff').write_text(''.join(difflib.unified_diff(source.splitlines(True),edited.splitlines(True),fromfile='pinned_author',tofile=name)))
    command=['/usr/bin/octave-cli','--no-gui','--quiet',str(snapshot/'reference_rbf_consumer.m'),'/home/chakwong/BayesFilter',str(output)]
    subprocess.run(command+['audit'],check=True)
    from scipy.io import loadmat
    data=plain(loadmat(output/'export.mat',simplify_cells=True));summary=data['summary']
    (output/'fixture.json').write_text(json.dumps(data['fixture'],allow_nan=False)+'\n')
    probe_started=time.monotonic()
    try:
        subprocess.run(command+['probe'],check=True,timeout=20)
        summary['nnls_cost_probe']=plain(loadmat(output/'probe.mat',simplify_cells=True))['probe']
    except subprocess.TimeoutExpired:
        summary['nnls_cost_probe']={'status':'TIMEOUT','cap_seconds':20,'scope':'COST_ONLY'}
    summary.update(reference_command=command,wall_seconds=time.monotonic()-started,
                   nnls_probe_wall_seconds=time.monotonic()-probe_started,
                   nonclaims=['No Table2 replication','No learning or performance ranking','TF fitting port remains absent'])
    (output/'summary.json').write_text(json.dumps(summary,allow_nan=False,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':main()

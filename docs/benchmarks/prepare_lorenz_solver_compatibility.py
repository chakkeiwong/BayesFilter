"""Independent CPU/NumPy reference preparation for saved solver diagnostics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();start=time.monotonic()
    import numpy as np
    from reference_octave_system_lapack import SystemLapack
    campaign=args.root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    args.output.mkdir(parents=True,exist_ok=False)
    library=Path('/usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3')
    assert sha(library)=='426b4bf536065cc63ef879444dad76dbc9f85e36e5cde906269acd6f177c5151'
    solver=SystemLapack(library);inputs={};values={};rows=[]

    def load(directory,name):
        fixture=directory/'fixture.json';inputs[str(fixture)]=sha(fixture)
        item=json.loads(fixture.read_text())['arrays'][name]
        path=directory/item['file'];assert sha(path)==item['sha256']
        inputs[str(path)]=sha(path)
        return np.fromfile(path,dtype='<f8').reshape(item['shape'])

    earlier=campaign/'attempt92_c2_lorenz_matrix_crossover'
    reference=campaign/'attempt94_c1_lorenz_matrix_fallback'
    answer=load(reference,'ordinary_beta')
    source_summary=reference/'summary.json';inputs[str(source_summary)]=sha(source_summary)
    branches=json.loads(source_summary.read_text())['diagnostics']['cached_types']
    matrices=[load(earlier,n) for n in ('native_gram','tf_gram')]
    rhs=[load(earlier,n) for n in ('native_rhs','tf_rhs')]
    designs=[load(earlier,n) for n in ('native_design','tf_design')]
    targets=load(earlier,'target')
    collected={k:[] for k in ('gram','rhs','beta','design','target')}

    def append(g,h,b,d,y,label,branch):
        actual,details=solver.solve(g,h)
        assert np.array_equal(actual,b),label
        assert details['branch']==branch,label
        for name,value in zip(collected,(g,h,b,d,y)):collected[name].append(value)
        rows.append(dict(label=label,coefficient_exact=True,**details))

    for arm in range(4):
        for row in range(32):
            append(matrices[arm//2][row],rhs[arm%2][row],answer[arm,row],
                   designs[arm//2][row],targets[row],f'crossover_arm{arm}_row{row}',branches[arm][row])
    trace=campaign/'offset-localization-20261006-02'
    meta=json.loads((trace/'manifest.json').read_text())
    path=trace/'regressions.npz';assert sha(path)==meta['output_sha256']['regressions.npz']
    inputs[str(path)]=sha(path)
    source_summary=trace/'summary.json';inputs[str(source_summary)]=sha(source_summary)
    source=json.loads(source_summary.read_text());assert source['source_result_exact'] and source['source_rng_exact']
    saved=np.load(path)
    for t in range(101):
        append(*(saved[k][t] for k in collected),label=f'offset_time{t}',branch=source['rows'][t]['branch'])
    arrays={}
    for name,sequence in collected.items():
        value=np.asarray(sequence,dtype='<f8');path=args.output/(name+'.bin');value.tofile(path)
        arrays[name]=dict(file=path.name,shape=list(value.shape),sha256=sha(path))
    fixture=dict(status='PASS',diagnostic_only=True,source_coefficients_exact=len(rows),
        rows=rows,arrays=arrays,library_sha256=sha(library),source_svd_relative_cutoff=2.**-53,
        seeds=[20260929,20260930],numpy_role='independent reference only')
    (args.output/'fixture.json').write_text(json.dumps(fixture,indent=2,allow_nan=False)+'\n')
    manifest=dict(status='PASS',command=[sys.executable,*sys.argv],environment=sys.executable,
        cpu_only=True,gpu_intentionally_hidden=True,git_commit=subprocess.check_output(
            ['git','rev-parse','HEAD'],cwd=args.root,text=True).strip(),
        plan='docs/plans/controlled-smc-tf-solver-compatibility-20261006.md',
        result_file='docs/plans/controlled-smc-tf-solver-compatibility-result-20261006.md',
        ended_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,
        source_sha256={str(Path(__file__)):sha(Path(__file__)),
            str(Path(__file__).with_name('reference_octave_system_lapack.py')):sha(Path(__file__).with_name('reference_octave_system_lapack.py'))},
        input_artifacts_sha256=inputs,output_sha256={i['file']:i['sha256'] for i in arrays.values()})
    manifest['output_sha256']['fixture.json']=sha(args.output/'fixture.json')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',source_coefficients_exact=len(rows),
        branches={b:sum(r['branch']==b for r in rows) for b in sorted(set(r['branch'] for r in rows))},
        wall_seconds=manifest['wall_seconds'])))


if __name__=='__main__':main()

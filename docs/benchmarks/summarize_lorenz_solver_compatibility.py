"""Independent CPU reference/post-run NumPy analysis; never a runtime dependency."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();start=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import tensorflow as tf
    from reference_octave_system_lapack import SystemLapack
    assert not tf.config.list_physical_devices('GPU')
    base=args.root/'docs/plans/artifacts/controlled-smc-kdm-replication-20260928'
    fixture=base/'solver-compatibility-reference-20261006-01';inputs={}
    index=json.loads((fixture/'fixture.json').read_text());arrays={}
    for key,item in index['arrays'].items():
        f=fixture/item['file'];assert sha(f)==item['sha256'];inputs[str(f)]=sha(f)
        arrays[key]=np.fromfile(f,dtype='<f8').reshape(item['shape'])
    library=Path('/usr/lib/x86_64-linux-gnu/openblas-pthread/liblapack.so.3')
    assert sha(library)==index['library_sha256']

    class InspectLapack(SystemLapack):
        spectrum=None

        def _call(self,name,*args):
            super()._call(name,*args)
            if name=='dgelsd' and args[11].value>0:
                self.spectrum=args[7].copy()

    solver=InspectLapack(library);spectra=[]
    for i,source in enumerate(index['rows']):
        solver.spectrum=None
        beta,meta=solver.solve(arrays['gram'][i],arrays['rhs'][i])
        assert np.array_equal(beta,arrays['beta'][i]) and meta['branch']==source['branch']
        spectra.append(solver.spectrum)
    candidates={}
    for label,name in (('gpu','attempt159_p0_tf_solver_compatibility'),('cpu','attempt160_p0_tf_solver_cpu_reference')):
        directory=base/name;manifest=json.loads((directory/'manifest.json').read_text())
        for key in ('conditional_beta','svd_spectra'):
            f=directory/(key+'.tensor');inputs[str(f)]=sha(f)
            assert [value for name,value in manifest['output_sha256'].items() if Path(name).name==f.name]==[sha(f)]
            # TensorFlow decodes the saved artifact; NumPy is confined to this
            # explicitly independent reporting/reference computation.
            candidates[(label,key)]=tf.io.parse_tensor(f.read_bytes(),tf.float64).numpy()
        candidates[(label,'summary')]=json.loads((directory/'summary.json').read_text())
    rows=[]

    def error(g,b,h):
        return float(np.linalg.norm(g@b-h,np.inf)/(np.linalg.norm(g,np.inf)*np.linalg.norm(b,np.inf)+np.linalg.norm(h,np.inf)))

    for i,source in enumerate(index['rows']):
        g=arrays['gram'][i];h=arrays['rhs'][i]
        row=dict(label=source['label'],source_branch=source['branch'],
            source_backward_error=error(g,arrays['beta'][i],h),
            source_minimum_singular_ratio=float(spectra[i][-1]/spectra[i][0]) if spectra[i] is not None else None)
        for label in ('gpu','cpu'):
            row[label+'_backward_error']=error(g,candidates[(label,'conditional_beta')][i],h)
            row[label+'_minimum_singular_ratio']=candidates[(label,'summary')]['rows'][i]['minimum_singular_ratio']
        rows.append(row)
    selected=[row for row in rows if row['label'] in ('offset_time20','offset_time25','offset_time28')]
    summary=dict(status='PASS',diagnostic_only=True,source_coefficients_exact=229,
        arithmetic='independent NumPy float64 post-run reporting',cpu_only=True,gpu_intentionally_hidden=True,
        source_cutoff=2.**-53,rows=rows,selected_rows=selected,
        backward_error_definition='infinity_norm(G beta - h) / (infinity_norm(G)*infinity_norm(beta)+infinity_norm(h))',
        maxima={label:max(r[label+'_backward_error'] for r in rows) for label in ('source','gpu','cpu')},
        inference='Small backward error is not forward/policy agreement; values are explanatory diagnostics.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    manifest=dict(status='PASS',command=[sys.executable,*sys.argv],environment=sys.executable,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.root,text=True).strip(),
        plan='docs/plans/controlled-smc-tf-solver-compatibility-20261006.md',
        result_file='docs/plans/controlled-smc-tf-solver-compatibility-result-20261006.md',
        seeds=[20260929,20260930],cpu_only=True,gpu_intentionally_hidden=True,
        wall_seconds=time.monotonic()-start,input_artifacts_sha256=inputs,
        source_sha256={str(Path(__file__)):sha(Path(__file__)),str(Path(__file__).with_name('reference_octave_system_lapack.py')):sha(Path(__file__).with_name('reference_octave_system_lapack.py'))},
        output_sha256={'summary.json':sha(args.output/'summary.json')},library_sha256=sha(library))
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',selected_rows=selected,maxima=summary['maxima'],wall_seconds=manifest['wall_seconds'])))


if __name__=='__main__':main()

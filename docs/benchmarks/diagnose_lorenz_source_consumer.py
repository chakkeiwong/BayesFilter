"""Independent CPU Octave source diagnostic; SciPy only reads its MAT boundary."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import argparse
import json
from pathlib import Path
import subprocess


def plain(value):
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items() if not k.startswith('__')}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if hasattr(value,'_fieldnames'):return {key:plain(getattr(value,key)) for key in value._fieldnames}
    if hasattr(value,'tolist'):return plain(value.tolist())
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    output=parser.parse_args().output;root=Path('/home/chakwong/BayesFilter')
    command=['/usr/bin/octave-cli','--no-gui','--quiet',str(output/'sources/reference_lorenz_consumer.m'),str(root),str(output)]
    subprocess.run(command,cwd=root,check=True)
    from scipy.io import loadmat
    data=plain(loadmat(output/'export.mat',simplify_cells=True))
    data['summary']['reference_command']=command
    for key in ('fixture','summary'):(output/(key+'.json')).write_text(json.dumps(data[key],allow_nan=False)+'\n')
    print(json.dumps(data['summary']))


if __name__=='__main__':main()

"""Extract preserved full author evidence and replay the actual GPU/XLA consumer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--fixture')
    parser.add_argument('--reference-result',required=True);args=parser.parse_args()
    output=Path(args.output);sources=output/'sources';env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1'
    command=['/usr/bin/octave-cli','--no-gui','--quiet',str(sources/'export_controlled_smc_full_fixture.m'),
             args.reference_result,str(output/'full-fixture.mat')]
    subprocess.run(command,env=env,check=True)
    # Explicit independent-reference serialization, isolated from GPU numerical code.
    conversion="from scipy.io import loadmat; import json,sys; from diagnose_controlled_smc_reference import plain; from pathlib import Path; p=Path(sys.argv[1]); d=plain(loadmat(p/'full-fixture.mat',simplify_cells=True)['fixture']); (p/'full-fixture.json').write_text(json.dumps(d,allow_nan=False))"
    subprocess.run([sys.executable,'-B','-c',conversion,str(output)],cwd=sources,env=env,check=True)
    numerical=[sys.executable,'-B',str(sources/'diagnose_controlled_smc_tf.py'),
               '--output',str(output),'--fixture',str(output/'full-fixture.json'),'--parity-only']
    print(json.dumps(dict(reference_export=command,gpu_command=numerical)),flush=True)
    return subprocess.call(numerical,env=os.environ.copy())


if __name__=='__main__':sys.exit(main())

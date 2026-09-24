"""Sequential bounded development fits; uses the existing public validation engine."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    args=parser.parse_args()
    inventory=json.loads(args.inventory.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    records=[]
    for row in inventory['fits']:
        destination=args.output/row['name']
        command=[sys.executable,str(root/'docs/benchmarks/check_hmc_remaining_2026_09_24.py'),
                 '--source',str(args.source),'--output',str(destination),'--seconds',str(row['seconds']),
                 '--device','gpu','--gpu',str(args.gpu),'--',
                 '/home/ubuntu/anaconda3/envs/tfgpu/bin/python',
                 str(args.source/'docs/benchmarks/fit_hmc_remaining_2026_09_24.py'),
                 '--design',row['design'],'--output',str(destination/'fit'),
                 '--seconds',str(row['seconds']-10)]
        code=subprocess.call(command)
        records.append({'name':row['name'],'exit_code':code,'directory':str(destination)})
        (args.output/'progress.json').write_text(json.dumps({'planned':len(inventory['fits']),'records':records},indent=2)+'\n')
        if code:
            raise SystemExit(code)
    print(json.dumps({'completed':len(records),'output':str(args.output)}),flush=True)


if __name__=='__main__':main()

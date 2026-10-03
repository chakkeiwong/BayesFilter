"""One isolated complete public HMC fit for the declared A--F program."""
import argparse
import datetime as dt
import json
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,required=True)
    args=parser.parse_args()
    from bayesfilter.testing.inference_validation.designs import resolve_suite
    from bayesfilter.testing.inference_validation.execution import configure_worker,source_state
    from bayesfilter.testing.inference_validation.storage import read_json,write_json,file_hash
    design,=resolve_suite(read_json(args.design))
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    manifest={'command':sys.argv,'design':design.payload(),'design_sha256':file_hash(args.design),
              'design_identity':design.identity,'source':source_state(),'seed':design.seed,
              'python':sys.executable,'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
              'plan_file':'docs/plans/bayesfilter-hmc-remaining-gap-program-2026-09-24.md',
              'result_file':str(args.output/'result.json'),'role':'development_not_confirmation',
              'data_version':'declared analytic law and fixed data in design',
              'runner_strategy':'dynamic_binding_scoped_reuse'}
    manifest['runtime']=configure_worker(design)
    write_json(args.output/'manifest.json',manifest)
    try:
        if design.engine == 'reference_mean':
            from bayesfilter.testing.inference_validation.engines.reference_mean import run
            result = run(design, args.output, deadline=started+args.seconds)
        else:
            from bayesfilter.testing.inference_validation.engines.pipeline import run_replication
            result=run_replication(design,args.output,0,deadline=started+args.seconds,reuse_leapfrog_graphs=True)
        write_json(args.output/'result.json',result)
    finally:
        write_json(args.output/'exit.json',{'elapsed_seconds':time.monotonic()-started,
                                          'result_present':(args.output/'result.json').exists()})


if __name__=='__main__':main()

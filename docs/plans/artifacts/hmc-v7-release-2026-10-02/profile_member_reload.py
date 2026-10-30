"""Profile a frozen saved-member reload; diagnostics only, no posterior run."""
import argparse
import cProfile
import io
import json
import os
from pathlib import Path
import pstats
import signal
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--member',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args()
source,member,output=args.source.resolve(),args.member.resolve(),args.output.resolve()
assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
sys.path.insert(0,str(source));os.chdir(source)
from bayesfilter.inference.hmc_candidate_set_retained import load_hmc_candidate_retained_runner
from bayesfilter.testing.inference_validation.targets import ValidationTarget

config=json.loads((member.parent/'configuration.json').read_text())
target=ValidationTarget(config['target'],config['parameters'],config['data'])
def expired(_signal,_frame):
    raise TimeoutError('bounded diagnostic profile cap')
signal.signal(signal.SIGALRM,expired)
profile=cProfile.Profile()
start=time.monotonic()
status='incomplete'
try:
    signal.alarm(300)
    profile.enable()
    runner=load_hmc_candidate_retained_runner(member,adapter=target)
    profile.disable()
    status='reload_passed'
except TimeoutError:
    profile.disable()
    status='profile_cap_reached_without_reload'
finally:
    signal.alarm(0)
    profile.dump_stats(str(output/'profile.pstats'))
    text=io.StringIO()
    pstats.Stats(profile,stream=text).sort_stats('cumulative').print_stats(50)
    (output/'profile.txt').write_text(text.getvalue())
    result=dict(status=status,wall_seconds=time.monotonic()-start,source=str(source),member=str(member),
                native_sampling=False,scope='bounded profiling only; no full-search or posterior claim')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

"""Resume the reviewed A11 stages; no selection or scientific-policy changes."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/chakwong/BayesFilter')
AR = ROOT/'docs/plans/artifacts/observation-tt-protected-fitting-20260917-01'
NR = ROOT/'docs/benchmarks/artifacts/observation_tt_protected_fitting_20260917'
CAL = NR/'attempt-calibration-01'
CONF = NR/'attempt-confirmation-01'


def record(stage, **values):
    state = dict(stage=stage, time_utc=datetime.now(timezone.utc).isoformat(),
                 pid=os.getpid(), **values)
    (AR/'continuation-state.json').write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps(state), flush=True)


deadline = time.monotonic()+14400
record('waiting_for_calibration')
while True:
    try:
        manifest = json.loads((CAL/'run_manifest.json').read_text())
    except json.JSONDecodeError:
        time.sleep(2)
        continue
    status = manifest['status']
    if status == 'COMPLETE':
        break
    if status == 'FAILED' or time.monotonic() > deadline:
        record('calibration_failed_or_watch_timeout', calibration_status=status)
        sys.exit(1)
    time.sleep(20)

# The consumer checks the frozen controls, source hashes and unique output root.
command = [sys.executable, 'docs/benchmarks/run_observation_tt_protected_fitting.py',
           '--stage', 'confirmation', '--attempt', '1',
           '--calibration-root', str(CAL), '--output-root', str(CONF),
           '--wall-budget-seconds', '5400']
record('confirmation_launch', command=command, calibration_wall_seconds=manifest['wall_seconds'])
environment = dict(os.environ, CUDA_VISIBLE_DEVICES='1', TF_FORCE_GPU_ALLOW_GROWTH='true',
                   TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1')
returncode = subprocess.call(command, cwd=ROOT, env=environment)
if returncode:
    record('confirmation_failed', returncode=returncode)
    sys.exit(returncode)
for stage, root in [('calibration', CAL), ('confirmation', CONF)]:
    subprocess.run([sys.executable, str(AR/'inspect_result.py'), str(root),
                    str(AR/f'{stage}-inspection.json')], cwd=ROOT, check=True)
record('complete')

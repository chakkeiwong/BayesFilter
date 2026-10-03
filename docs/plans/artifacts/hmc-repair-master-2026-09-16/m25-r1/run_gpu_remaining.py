"""Serial remainder of the fixed M25 GPU inventory after its successful pilot."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
pilot = json.loads((root/'gpu-exact-2251-r1/worker/result.json').read_text())
assert pilot['status'] == 'complete' and pilot['verified_candidate_ids']
cases = [('exact', 2026092252), ('partial', 2026092251), ('partial', 2026092252),
         ('partial_half', 2026092251), ('partial_half', 2026092252)]
state = {'planned_remaining': cases, 'completed': [], 'status': 'running'}
path = root/'gpu-queue-progress.json'
for kind, seed in cases:
    name = f'gpu-{kind}-{str(seed)[-4:]}-r1'
    command = [sys.executable, str(root/'run_gpu_attempt.py'), '--name', name,
               '--map', kind, '--seed', str(seed)]
    state['active'] = name
    path.write_text(json.dumps(state, indent=2)+'\n')
    run = subprocess.run(command)
    state['completed'].append({'name': name, 'returncode': run.returncode})
    if run.returncode:
        state['status'] = 'requires_infrastructure_or_capacity_review'
        break
else:
    state['status'] = 'complete_requires_terminal_audit'
state['active'] = None
path.write_text(json.dumps(state, indent=2)+'\n')
print(json.dumps(state), flush=True)

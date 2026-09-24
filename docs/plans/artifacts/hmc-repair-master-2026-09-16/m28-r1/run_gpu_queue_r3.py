"""Serial four-cell M28 execution; outer receipts account for nested children once."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
source = root / 'source-r1'
assert (source / 'source_snapshot.json').is_file()
assert all((root / 'designs-r1' / f'm28-{k}-{s}.json').is_file()
           for k in ('exact', 'residual') for s in (2026092381, 2026092382))
for kind in ('exact', 'residual'):
    for seed in (2026092381, 2026092382):
        name = f'gpu-{kind}-{seed}-r3'
        command = [sys.executable, str(root / 'run_attempt.py'), '--name', name,
            '--seconds', '1000', '--source-root', str(source), '--device', 'gpu', '--gpu', '2',
            '--', sys.executable, '-m', 'bayesfilter.testing.inference_validation', 'run',
            str(root / 'designs-r1' / f'm28-{kind}-{seed}.json'), '--output',
            str(root / name / 'fits'), '--max-workers', '1']
        print('START ' + name, flush=True)
        with (root / (name + '-meter.log')).open('x') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        receipt_path = root / name / 'execution.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        print(json.dumps({'attempt': name, 'exit_code': code, 'receipt': receipt}), flush=True)
        if receipt is None or receipt['launch_failure'] is not None:
            raise SystemExit('Missing terminal receipt or unavailable device; review before continuation')
print('M28 FOUR-CELL QUEUE COMPLETE', flush=True)

"""CPU-only healthy-equivalence and failed-guard regression on frozen inputs."""
import json
import os
from pathlib import Path
import runpy
import sys

campaign = Path(__file__).resolve().parent
out = Path(os.environ['LEDH_PROGRAM_RESULT_DIR'])
sys.argv = [str(campaign / 'preserve_baseline.py'), str(out / 'baseline.json')]
runpy.run_path(sys.argv[0], run_name='__main__')
original = json.loads((campaign / 'baseline.json').read_text())
latest = json.loads((out / 'baseline.json').read_text())
assert latest['checks']['healthy'] == original['checks']['healthy']
for check in latest['checks'].values():
    assert 'exception' not in check, check
assert latest['checks']['upper_nan']['result'][0] is False
assert latest['checks']['infinite_input']['result'][0] is False
assert latest['checks']['shape_2_1']['result'] == [2, 1]
bad = latest['checks']['invalid_actual_batch']['result']
assert bad[0] == [{'__nonfinite_float__': '-Infinity'}]
assert bad[1] == [[0.0]]
assert bad[2]['value_valid'] == [False]
assert bad[2]['force_valid'] == [False]
result = dict(healthy_equivalence='passed', containment='passed',
              gpu_intentionally_hidden=True, comparison='exact equality to preserved current-source fixture',
              source='baseline.json', healthy=latest['checks']['healthy'])
(out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print('Exact healthy equivalence and invalid-guard assertions passed.')

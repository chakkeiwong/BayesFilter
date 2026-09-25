"""Post-run diagnostic: attribute archived changed-input record differences."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
OLD = ROOT / 'run-01754/lifecycle-runtime-5.json'
NEW = ROOT / 'run-01821/lifecycle-original-runtime-comparisons-5.json'


def differences(left, right, path='result'):
    if isinstance(right, dict):
        return [entry for key in right if key != 'jit_compile'
                for entry in differences(left[key], right[key], path + '.' + key)]
    if isinstance(right, list):
        return [entry for index, (a, b) in enumerate(zip(left, right, strict=True))
                for entry in differences(a, b, path + f'[{index}]')]
    if isinstance(right, (str, bool, int)) or right is None:
        equal = left == right
    else:
        equal = (left == right or (math.isnan(left) and math.isnan(right))
                 or abs(left - right) <= 1e-10 + 1e-10 * abs(right))
    return [] if equal else [{'path': path, 'candidate': left, 'baseline': right}]


old = json.loads(OLD.read_text())['observations']
new = json.loads(NEW.read_text())['comparisons']
rows = []
for index in range(2):
    original = new[index * 2]['expected']
    repaired = new[index * 2]['actual']
    previous = old[index]['after']
    rows.append({'input_index': index,
        'previous_vs_original': differences(previous, original),
        'repaired_vs_original': differences(repaired, original),
        'repaired_vs_previous': differences(repaired, previous),
        'factor_fit_bitwise_equal_previous_repaired': (
            previous['diagnostics']['history'][0]['fit'] ==
            repaired['diagnostics']['history'][0]['fit'])})
report = {'role': 'archived_record_attribution_only',
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD, NEW)},
    'absolute_tolerance': 1e-10, 'relative_tolerance': 1e-10,
    'excluded_field': 'jit_compile execution metadata absent from original schema',
    'rows': rows,
    'nonclaims': ['No new numerical execution, terminal qualification or cause of fitter drift.']}
output = ROOT / 'lifecycle-changed-input-attribution-01822.json'
with output.open('x') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
with (ROOT / Path(__file__).name).open('x') as handle:
    handle.write(Path(__file__).read_text())
print(json.dumps({'output': str(output), 'summary': [
    {key: len(value) if isinstance(value, list) else value for key, value in row.items()}
    for row in rows]}, indent=2))

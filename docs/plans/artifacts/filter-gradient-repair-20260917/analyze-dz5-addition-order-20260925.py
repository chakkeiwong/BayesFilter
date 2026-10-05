"""Independent diagnostic analysis; failed full replay is never qualification."""

import hashlib
import json
import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle_errors(oracle):
    assert len(oracle['values']) == len(oracle['scores']) == len(oracle['bank']) == 185
    assert oracle['validity'] == [True] * 185 and oracle['branch_status'] == [0] * 185
    assert all(len(row) == 23 for row in oracle['scores'] + oracle['bank'])
    assert all(math.isfinite(x) for row in oracle['scores'] for x in row)
    assert all(math.isfinite(x) for x in oracle['values'])
    errors = []
    for index, step in enumerate((1e-3, 5e-4)):
        estimates = []
        for coordinate, scale in enumerate(oracle['prior_scale']):
            assert math.isfinite(scale) and scale > 0
            for block, multiplier in enumerate((-2, -1, 1, 2)):
                expected = list(oracle['bank'][0])
                expected[coordinate] += multiplier * step * scale
                assert oracle['bank'][1 + 92 * index + 23 * block + coordinate] == expected
            offset = 1 + 92 * index + coordinate
            a, b, c, d = (oracle['values'][offset + 23 * block] for block in range(4))
            estimates.append((a - 8 * b + 8 * c - d) / (12 * step * scale))
        absolute = [abs(a - b) for a, b in zip(oracle['scores'][0], estimates, strict=True)]
        allowed = [1e-8 + 1e-7 * abs(x) for x in estimates]
        normalized = max(a / b for a, b in zip(absolute, allowed, strict=True))
        saved = oracle['checks'][index]
        assert saved['step_prior_sd'] == step and saved['passed'] and normalized <= 1
        assert saved['finite_difference'] == estimates
        assert saved['absolute_error'] == absolute and saved['allowed_error'] == allowed
        assert saved['max_scaled_error'] == normalized
        errors.append(normalized)
    return errors


def analyze(root):
    result = {'role': 'diagnostic_overlay_evidence_not_runtime_qualification', 'runs': {}}
    prior = json.loads((root / 'run-03901/dz5-score-oracle.json').read_text())
    for number, expected_pass in ((3907, True), (3908, False)):
        directory = root / f'run-{number:05d}'
        run = json.loads((directory / 'run.json').read_text())
        report = json.loads((directory / 'dz5-snapshot-import.json').read_text())
        oracle = json.loads((directory / 'dz5-score-oracle.json').read_text())
        assert oracle['bank'] == prior['bank']
        command = json.loads((directory / 'isolated-dz5-command.json').read_text())
        cases = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
        assert len(cases) == 1 and bool(list(cases[0])) != expected_pass
        assert (run['state'] == 'passed') == (run['exit_code'] == 0) == expected_pass
        assert report['passed'] == report['replay_exact'] == expected_pass
        assert report['trace_count'] == 1 and not report['host_callbacks']
        assert not report['unexpected_modules'] and not report['post_target_unexpected_modules']
        assert report['optimizer_options'].get('arithmetic_optimization', True)
        assert report['cuda_visible_devices'] == '-1' and not report['jit_compile']
        assert not report['adapter_admitted']
        assert command['child_sha256'] == sha(directory / 'isolated-dz5-import.py')
        assert report['executable_overlay']['modified_function_sha256'] == sha(directory / 'diagnostic-srukf-overlay.py')
        snapshot = Path(command['command'][command['command'].index('/tmp/dz5-source') - 1])
        manifest_sha = sha(snapshot / 'manifest.json')
        assert manifest_sha == oracle['snapshot_manifest_sha256'] == command['snapshot_manifest_sha256']
        manifest = json.loads((snapshot / 'manifest.json').read_text())
        for path, entry in manifest['sources'].items():
            assert sha(snapshot / path.lstrip('/')) == entry['sha256']
        for module in report['post_target_loaded_modules'].values():
            assert manifest['sources'][module['path']]['sha256'] == module['sha256']
        replays = []
        for path in sorted(directory.glob('dz5-score-replay*.json')):
            replay = json.loads(path.read_text())
            assert replay['values'] == oracle['values']
            assert replay['validity'] == oracle['validity'] and replay['branch_status'] == oracle['branch_status']
            delta = [abs(a - b) for x, y in zip(oracle['scores'], replay['scores'], strict=True)
                for a, b in zip(x, y, strict=True)]
            assert len(delta) == 4255 and all(math.isfinite(x) for x in delta)
            assert max(delta) == replay['score_max_absolute_error']
            assert replay['value_max_absolute_error'] == 0
            assert (max(delta) == 0) == expected_pass
            replays.append({'path': path.name, 'changed_scores': sum(x != 0 for x in delta),
                'score_max_absolute_error': max(delta)})
        assert len(replays) == (4 if expected_pass else 1)
        graphs = json.loads((directory / 'ordered-additions.json').read_text())
        protected = []
        for row in graphs:
            assert sha(directory / row['path']) == row['sha256']
            if row['stage'] == 'post_optimization_graph':
                protected.extend(row['ordered_functions'])
                assert all(len(node['input']) != 5 for node in row['addn'] if 'while' in node['name'])
        assert protected and all(row['noinline'] and row['ops'] == ['AddV2', 'Identity'] for row in protected)
        result['runs'][str(number)] = {'finite_difference_oracle_passed': True,
            'evaluated_observations': report['evaluated_observations'],
            'normalized_errors': oracle_errors(oracle), 'replays': replays,
            'all_replays_exact': expected_pass, 'protected_functions': len(protected)}
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1])), indent=2))

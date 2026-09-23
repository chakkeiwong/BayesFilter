"""Bounded native-memory containment with sequential fresh numerical workers.

This diagnostic uses the actual current DZ5 parent function and the qualified
public-block churn test. It does not alter library algorithms or promise native
eviction inside a process.
"""

import ast
import gc
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from tests.test_filter_repair_dz5_supervision import ROOT as DZ5_ROOT
from tests.test_filter_repair_dz5_supervision import SUPERVISOR


def _host_memory():
    fields = {line.split(':', 1)[0]: int(line.split()[1]) * 1024
        for line in Path('/proc/self/smaps_rollup').read_text().splitlines()
        if line.startswith(('Rss:', 'Pss:', 'Private_Clean:', 'Private_Dirty:'))}
    fields['map_count'] = len(Path('/proc/self/maps').read_text().splitlines())
    return fields


def test_fresh_workers_contain_public_block_native_residency(request):
    source = SUPERVISOR.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'supervise')
    imports = '\n'.join(ast.get_source_segment(source, node) for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)))
    excerpt = imports + '\n' + ast.get_source_segment(source, function)
    namespace = {'REPO_ROOT': DZ5_ROOT}
    exec(compile(excerpt, str(SUPERVISOR), 'exec'), namespace)  # noqa: S102 - exact inspected parent
    root = Path(request.config.getoption('xmlpath')).parent
    checkout = Path(__file__).resolve().parents[1]
    initial = _host_memory()
    rows = []
    for index in range(2):
        directory = root / f'contained-{index}'
        directory.mkdir()
        command = [sys.executable, str(checkout / 'scripts/filter_repair_test_worker.py'),
            '-q', str(checkout / 'tests/test_filter_repair_block_public_churn.py')
                + '::test_bounded_public_block_callback_and_shape_churn',
            f'--junitxml={directory / "junit.xml"}']
        environment = dict(os.environ)
        # The actual parent uses its repository as cwd; explicit PYTHONPATH
        # selects this repair checkout without editing the external runtime.
        environment['PYTHONPATH'] = str(checkout)
        receipt = namespace['supervise'](command, timeout_seconds=300,
            log_path=directory / 'worker.log', environment=environment)
        gone = not Path(f'/proc/{receipt["pid"]}').exists()
        events = [json.loads(line) for line in Path(receipt['progress_path']).read_text().splitlines()]
        row = {'index': index, 'receipt': receipt, 'process_reaped': gone,
            'parent_after_exit': _host_memory(), 'heartbeat_count': sum(x['event'] == 'heartbeat' for x in events)}
        rows.append(row)
        with (root / 'containment-stages.jsonl').open('a') as output:
            output.write(json.dumps(row, allow_nan=False) + '\n')
        assert gone and not receipt['timed_out'] and receipt['returncode'] == 0, receipt
        tests = list(ET.parse(directory / 'junit.xml').getroot().iter('testcase'))
        assert len(tests) == 1 and not any(case.findall(tag) for case in tests for tag in ('failure', 'error', 'skipped'))
        snapshots = []
        with (directory / 'block-churn-stages.jsonl').open() as handle:
            for line in handle:
                record = json.loads(line)
                snapshots.append({'stage': record['stage'], 'cycle': record['cycle'],
                    'rss_bytes': record['memory']['rollup']['Rss'], 'map_count': record['map_count'],
                    'gpu_allocator': record['memory']['gpu']})
        row['child_stages'] = snapshots
        row['result_sha256'] = hashlib.sha256((directory / 'block-public-churn.json').read_bytes()).hexdigest()
        row['junit_sha256'] = hashlib.sha256((directory / 'junit.xml').read_bytes()).hexdigest()
        gc.collect()
        row['parent_after_reading_artifacts'] = _host_memory()
    report = {'schema': 'filter_process_containment.v1', 'supervisor_source': str(SUPERVISOR),
        'supervisor_sha256': hashlib.sha256(source.encode()).hexdigest(),
        'supervisor_function_sha256': hashlib.sha256(excerpt.encode()).hexdigest(),
        'numerical_source_checkout': str(checkout), 'parent_before': initial,
        'parent_after': _host_memory(), 'workers': rows,
        'runtime_device': os.environ['CUDA_VISIBLE_DEVICES'],
        'nonclaims': ['Two sequential workers qualify this bounded lifecycle only.',
            'No native in-process eviction, infinite-lifetime safety or exact peak bound.',
            'Child GPU sharing observations are attribution, not cost comparison.',
            'Actual DZ5 numerical integration and model transitions remain separate.']}
    with (root / 'process-containment.json').open('x') as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write('\n')
    assert SUPERVISOR.read_text() == source
    assert rows[0]['receipt']['pid'] != rows[1]['receipt']['pid']
    assert all(row['heartbeat_count'] > 0 for row in rows)

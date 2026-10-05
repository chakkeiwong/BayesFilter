"""Bind public-helper/caller evidence and unchanged complete-preparation owners."""

import ast
import hashlib
import json
import statistics
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.filter_repair_cost_provenance import validate_cost_device
from tests.test_filter_repair_c2_preparation import compare

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')
SOURCES = ('bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py',
           'bayesfilter/highdim/c2_mixture_ukf_apf_tf.py')


def test_public_helper_caller_and_resource_evidence(request):
    groups = {f'c2_preparation_public_{name}_{device}'
              for name in ('helpers', 'callers') for device in ('cpu', 'gpu')}
    groups |= {f'c2_preparation_public_cost_{arm}_gpu' for arm in ('original', 'current')}
    records = {}
    failures = []
    for path in sorted(RAW.glob('run-*/run.json')):
        if int(path.parent.name[4:]) <= 5525:
            continue
        row = json.loads(path.read_text())
        group = row['key'][1]
        if group not in groups:
            continue
        if row['state'] != 'passed':
            failures.append({'run': path.parent.name, 'state': row['state']})
            continue
        for source in SOURCES:
            assert hashlib.sha256((ROOT/source).read_bytes()).hexdigest() == row['source_sha256'][source]
        cases = ET.parse(path.parent/'junit.xml').findall('.//testcase')
        assert cases and all(c.find(tag) is None for c in cases for tag in ('error', 'failure', 'skipped'))
        provenance = next(json.loads(line) for line in (path.parent/'process.log').read_text().splitlines()
                          if line.startswith('{"tensorflow_version"'))
        policy = provenance['gpu_memory_policy']
        assert policy['configured_before_logical_device_initialization']
        assert policy['all_physical_devices_memory_growth']
        if row['device'] == 'CPU':
            assert provenance['cuda_visible_devices'] == '-1'
        else:
            assert policy['physical_devices']
        records[group] = (path.parent, row, provenance, len(cases))
    assert set(records) == groups
    callers = []
    for device in ('cpu', 'gpu'):
        path, _, _, _ = records[f'c2_preparation_public_callers_{device}']
        value = json.loads((path/'c2-actual-candidate-callers.json').read_text())
        assert value['reference_and_standalone_fallbacks_blocked']
        assert len(value['records']) == 11
        callers.append({'device': device, 'run': path.name,
                        'families': [r['family'] for r in value['records']]})
    costs = {}
    for arm in ('original', 'current'):
        path, row, provenance, _ = records[f'c2_preparation_public_cost_{arm}_gpu']
        value = json.loads((path/'c2-public-helper-cost.json').read_text())
        validate_cost_device(row, provenance, value['device_observation'])
        exited = row['process_exit_observation']
        assert not exited['errors'] and not exited['proc_entry_present']
        assert not any(p['pid'] == exited['pid'] for p in exited['gpu_processes'])
        costs[arm] = value
    assert records['c2_preparation_public_cost_original_gpu'][1]['gpu_uuid'] == records['c2_preparation_public_cost_current_gpu'][1]['gpu_uuid']
    assert records['c2_preparation_public_cost_original_gpu'][1]['source_sha256'] == records['c2_preparation_public_cost_current_gpu'][1]['source_sha256']
    summaries = []
    for before, after in zip(costs['original']['rows'], costs['current']['rows'], strict=True):
        assert before['helper'] == after['helper']
        compare(before['first_result'], after['first_result'])
        assert after['trace_count'] == 1
        samples = after['fixed_root_snapshots']
        assert samples['128']['VmRSS']-samples['64']['VmRSS'] <= 16*2**20
        assert samples['128']['allocator']['current'] == samples['64']['allocator']['current']
        summaries.append({'helper': before['helper'],
            'warm_seconds_before': statistics.median(before['warm_seconds']),
            'warm_seconds_after': statistics.median(after['warm_seconds']),
            'cold_seconds_before': before['cold_seconds'], 'cold_seconds_after': after['cold_seconds'],
            'warm_memory_before': before['memory']['warm'], 'warm_memory_after': after['memory']['warm'],
            'reuse': samples, 'inference': 'one process pair; descriptive only'})

    # The source files also contain already measured preparation owners. Prove
    # their callable bodies stayed unchanged rather than silently accepting
    # stale whole-file hashes or remeasuring a different numerical program.
    unchanged = []
    allowed = {'bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py':
                   {'FrozenGaussianStateProposal.sample_with_seed'},
               'bayesfilter/highdim/c2_mixture_ukf_apf_tf.py':
                   {'compile_k1_apf_proposal', 'sample_k1_apf_step'}}

    def functions(source):
        tree = ast.parse(source)
        result = {}

        def walk(body, prefix=''):
            for node in body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[prefix+node.name] = ast.dump(node, include_attributes=False)
                elif isinstance(node, ast.ClassDef):
                    walk(node.body, prefix+node.name+'.')
        walk(tree.body)
        return result

    for source in SOURCES:
        original = functions(subprocess.check_output(['git', 'show', 'd91e269c8:'+source], cwd=ROOT))
        current = functions((ROOT/source).read_text())
        for name, body in original.items():
            if name in allowed[source]:
                continue
            assert current[name] == body, (source, name)
            unchanged.append({'path': source, 'function': name,
                              'ast_sha256': hashlib.sha256(body.encode()).hexdigest()})
    result = {'schema': 'filter_repair.c2_public_closure.v1',
        'evidence': {group: {'run': path.name, 'tests': count,
            'manifest_sha256': hashlib.sha256((path/'run.json').read_bytes()).hexdigest()}
            for group, (path, _, _, count) in records.items()},
        'failed_workers_preserved': failures, 'callers': callers, 'descriptive_costs': summaries,
        'unchanged_preparation_functions': unchanged,
        'nonclaims': ['No whole-program closure, canonical LEDH admission or deferred-algorithm qualification.',
                     'No general speed ranking, native eviction or arbitrary-shape capacity.']}
    (Path(request.config.getoption('xmlpath')).parent/'c2-public-closure-readback.json').write_text(
        json.dumps(result, indent=2)+'\n')

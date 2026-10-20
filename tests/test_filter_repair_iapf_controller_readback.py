"""Verify the preserved iAPF boundary failure and its arithmetic diagnosis."""

import hashlib
import json
from pathlib import Path

from tests.test_filter_repair_ledh_seeded_readback import _load, _provenance
from tests.test_filter_repair_nonlinear_scope import RAW, ROOT


def test_iapf_controller_diagnosis_readback(request):
    for number in (4863, 4865, 4868):
        run = json.loads((RAW/f'run-{number:05d}'/'run.json').read_text())
        assert run['state'] == 'failed' and run['test_evidence']['failure'] == 1
        report = _load(number, 'iapf-controller.json')
        assert not report['complete']
        case = report['rows'][-1]
        assert case['reference']['action'] == 'final' and case['candidate']['action'] == 0
        assert case['reference']['cv'] < case['tau'] <= case['candidate']['cv']
    cpu = _load(4867, 'iapf-controller.json')
    assert cpu['complete'] and not cpu['runtime_migrated']
    assert len(cpu['rows']) == 12 and len(cpu['invalid']) == 7
    graph = cpu['graph']
    assert graph['jit_compile'] and graph['trace_count'] == 1 and not graph['host_callbacks']
    assert hashlib.sha256((RAW/'run-04867/iapf-controller.hlo.txt').read_bytes()).hexdigest() == graph['hlo_sha256']
    cpu_probe = _load(4866, 'iapf-controller-localization.json')
    gpu_probe = _load(4869, 'iapf-controller-localization.json')
    for number, probe in ((4866, cpu_probe), (4869, gpu_probe)):
        assert probe['reference_decision']['cv'] == probe['reference']['cv']
        assert not probe['runtime_repaired']
        for item in probe['scalar_divisions'].values():
            assert item['reference'] == item['compiled']
        provenance = _provenance(number)
        assert provenance['gpu_memory_policy']['configured_before_logical_device_initialization']
    const = cpu_probe['compiled_constant_divisors']
    assert const['values'] == cpu_probe['reference']['values']
    assert const['total'] == cpu_probe['reference']['total']
    assert const['mean'] != cpu_probe['reference']['mean']
    assert cpu_probe['constant_diagnostic_matches_owner_cv']
    dynamic = cpu_probe['compiled_dynamic_divisors']
    assert dynamic['mean'] == cpu_probe['reference']['mean']
    assert dynamic['root'] == cpu_probe['reference']['root']
    assert dynamic['cv'] != cpu_probe['reference']['cv']
    gpu = gpu_probe['compiled_barrier_dynamic_divisors']
    assert gpu['shifted'] == gpu_probe['reference']['shifted']
    assert gpu['values'][0] == gpu_probe['reference']['values'][0]
    assert gpu['values'][1] != gpu_probe['reference']['values'][1]
    assert gpu['cv'] == gpu_probe['actual_owner']['cv']
    fixture_root = ROOT/'tests/fixtures/filter_repair_iapf_controller_20260929'
    fixture = json.loads((fixture_root/'fixture.json').read_text())
    assert hashlib.sha256((ROOT/'bayesfilter/score_study/iapf_adapter.py').read_bytes()).hexdigest() == fixture['source_sha256']
    report = {'schema': 'filter_repair_iapf_controller_diagnosis.v1',
        'cpu_qualified_run': 4867, 'gpu_boundary_failure_run': 4868,
        'cpu_localization_run': 4866, 'gpu_localization_run': 4869,
        'current_runtime_unchanged': True, 'primitive_admitted': False,
        'cpu_reference_cv': cpu_probe['reference']['cv'], 'gpu_candidate_cv': gpu['cv'],
        'decision': 'Boundary mismatch remains a veto; no full adaptive migration yet.'}
    (Path(request.config.getoption('xmlpath')).parent/'iapf-controller-diagnosis-readback.json').write_text(
        json.dumps(report, indent=2)+'\n')

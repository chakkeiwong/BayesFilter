"""Check extraction evidence and extend the existing CPU/GPU campaign ledger."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def tree(path):
    return ast.parse(path.read_text())


def definitions(parsed):
    return {n.name: n for n in parsed.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}


def assignments(parsed):
    result = {}
    for node in parsed.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node
    return result


def dump(node):
    return ast.dump(node, include_attributes=False)


previous_path = ROOT.parent / 'm7-bootstrap-r1/reconciliation.json'
previous = load(previous_path)
baseline = load(ROOT / 'baseline.json')
for row in baseline['files']:
    assert sha(ROOT / 'baseline' / row['path']) == row['sha256'], row['path']
for kind in ('scripted', 'real'):
    assert load(ROOT / f'before-{kind}.json') == load(ROOT / f'after-{kind}.json'), kind

old_tree = tree(ROOT / 'baseline/bayesfilter/inference/hmc_kernel_tuning.py')
new_tree = tree(REPO / 'bayesfilter/inference/hmc_mass_adaptation.py')
remaining_tree = tree(REPO / 'bayesfilter/inference/hmc_kernel_tuning.py')
old, new, remaining = map(definitions, (old_tree, new_tree, remaining_tree))
parity = load(ROOT / 'ast-parity.json')
names = set(load(ROOT / 'definitions.json'))
assert set(old) - set(remaining) == names
assert set(new) == names
for row in parity['definitions']:
    name = row['name']
    assert hashlib.sha256(dump(old[name]).encode()).hexdigest() == row['before_ast_sha256']
    assert hashlib.sha256(dump(new[name]).encode()).hexdigest() == row['after_ast_sha256']
    if name == '_run_p4_windowed_boundary_attempt':
        normalized = tree(REPO / 'bayesfilter/inference/hmc_mass_adaptation.py')
        definition = definitions(normalized)[name]
        owners = [n for n in ast.walk(definition) if isinstance(n, ast.keyword) and n.arg == 'owner_file']
        assert len(owners) == 1 and owners[0].value.value == 'hmc_mass_adaptation.py'
        owners[0].value = ast.Constant(value='hmc_kernel_tuning.py')
        assert dump(definition) == dump(old[name])
    else:
        assert dump(new[name]) == dump(old[name]), name
assert [name for name in remaining if dump(remaining[name]) != dump(old[name])] == ['_g2_source_file_for_site']
old_constants, new_constants = assignments(old_tree), assignments(new_tree)
constant_names = load(ROOT / 'moved-constants.json')
assert all(dump(old_constants[n]) == dump(new_constants[n]) for n in constant_names)
assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and 'numpy' in ast.unparse(n)
               for n in ast.walk(new_tree))

cases = {}
for name in ('affected-tests.xml', 'compatibility-retry.xml', 'historical-handoff-tests.xml'):
    for case in ET.parse(ROOT / name).iter('testcase'):
        nodeid = case.get('classname') + '::' + case.get('name')
        status = 'skipped' if case.find('skipped') is not None else (
            'failed' if case.find('failure') is not None or case.find('error') is not None else 'passed')
        cases[nodeid] = {'status': status, 'latest_batch': name}
counts = Counter(row['status'] for row in cases.values())
assert counts == {'passed': 352, 'skipped': 1}, counts

attempts = []
for path in sorted(ROOT.glob('*-run.json')):
    row = load(path)
    attempts.append({**row, 'manifest_path': str(path.relative_to(REPO)),
                     'manifest_sha256': sha(path), 'log_sha256': sha(Path(row['log']))})
assert len(attempts) == 8
assert [row['manifest_path'].split('/')[-1] for row in attempts if row['returncode']] == ['affected-tests-run.json']
measured = sum(row['elapsed_seconds'] for row in attempts)
assert measured <= 700.
overhead = 120.
prior = previous['budget_seconds']
budget = {
    'cpu_measured': measured, 'cpu_overhead': overhead, 'cpu_charge': measured + overhead,
    'gpu_charge': 0.,
    'total_cpu_charged': prior['total_cpu_charged'] + measured + overhead,
    'total_gpu_charged': prior['total_gpu_charged'],
    'remaining_cpu': prior['remaining_cpu'] - measured - overhead,
    'remaining_gpu': prior['remaining_gpu'],
    'remaining_m7_cpu_allocation': prior['remaining_m7_cpu_allocation'] - measured - overhead,
    'remaining_m7_gpu_allocation': prior['remaining_m7_gpu_allocation'],
    'unfinished_launched_reservations': 0.,
}
assert budget['remaining_m7_cpu_allocation'] >= 0
installation = load(ROOT / 'guide-installation.json')
assert sha(REPO / installation['official_path']) == installation['installed_sha256']
assert sha(REPO / installation['archival_build']) == installation['installed_sha256']
guide = load(ROOT / 'guide-r1/build-manifest.json')
for path, expected in guide['source_sha256'].items():
    assert sha(REPO / path) == expected, path

paths = ['bayesfilter/inference/' + name for name in (
    '__init__.py', 'hmc_bootstrap.py', 'hmc_preparation_common.py', 'hmc_kernel_tuning.py',
    'hmc_mass_adaptation.py', 'hmc_geometry.py', 'hmc_warmup.py', 'hmc_preparation.py',
    'hmc_budget_policy.py', 'hmc_candidate_set_execution.py')]
paths += ['tests/' + name for name in (
    'test_hmc_mass_adaptation_extraction.py', 'test_hmc_kernel_tuning_windowed_mass.py',
    'test_hmc_candidate_set_execution.py', 'test_hmc_kernel_tuning_public_api.py',
    'test_hmc_kernel_tuning_p4_registry_public_chain.py', 'test_hmc_master_repair.py',
    'test_hmc_warmup.py', 'test_hmc_kernel_tuning_outer_loop.py',
    'test_hmc_tuning_documentation_contract.py', 'test_hmc_bootstrap_extraction.py',
    'test_hmc_kernel_tuning_bootstrap.py', 'test_hmc_geometry_extraction.py',
    'test_hmc_kernel_tuning_geometry.py')]
paths += ['docs/reference/hmc-tuning-interface.md', 'docs/chapters/ch21b_hmc_tuning_interfaces.tex',
          'docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md',
          'docs/plans/bayesfilter-hmc-repair-m7-result-2026-09-17.md']
evidence_paths = [*ROOT.glob('*.py'), ROOT/'baseline.json', ROOT/'definitions.json',
                  ROOT/'moved-constants.json', ROOT/'ast-parity.json', ROOT/'guide-installation.json']
evidence_paths += [ROOT/name for name in ('before-scripted.json', 'before-real.json',
                  'after-scripted.json', 'after-real.json', 'affected-tests.xml',
                  'compatibility-retry.xml', 'historical-handoff-tests.xml')]
report = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'question': 'Windowed preparation implementation ownership with unchanged numerical behavior',
    'plan_file': paths[-2], 'result_file': paths[-1], 'command': [sys.executable, str(Path(__file__))],
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
    'environment': {'python': sys.executable, 'tensorflow': version('tensorflow'),
                    'tensorflow_probability': version('tfp-nightly'),
                    'tensorflow_probability_distribution': 'tfp-nightly',
                    'gpu_status': 'intentionally hidden in all numerical test workers',
                    'current_process_framework_imported': False},
    'seeds': 'Recorded in capture_windowed.py and imported saved test fixtures; unchanged by extraction',
    'data': 'Synthetic diagnostic/test fixtures only',
    'previous_reconciliation': {'path': str(previous_path.relative_to(REPO)), 'sha256': sha(previous_path)},
    'baseline_verified': baseline, 'source_sha256': {path: sha(REPO / path) for path in paths},
    'evidence_sha256': {str(p.relative_to(REPO)): sha(p) for p in evidence_paths},
    'engineering': {'exact_scripted_cases': 14, 'exact_live_reference_runs': 1,
                    'live_reference_fields': 16, 'moved_definitions': 76,
                    'identical_definition_asts': 75, 'physical_owner_relocations': 1,
                    'identical_constants_and_type_aliases': 22,
                    'unchanged_remaining_definitions': len(remaining) - 1,
                    'distinct_passed': counts['passed'], 'distinct_skipped': counts['skipped']},
    'test_cases': cases, 'attempts': attempts, 'budget_seconds': budget,
    'guide_installation': installation,
    'failure_and_repair': 'New test incorrectly requested a complete registry before the later P4 seed. Test now inspects the consumed stage entry. Runtime closure validation is unchanged; retry passes.',
    'post_test_source_edit': 'Only blank gaps left by removed definitions were shortened; checked ASTs unchanged.',
    'scientific_interpretation': 'CPU engineering parity only; earlier frozen GPU findings retain their source identity. No defaults or diagnostic roles changed.',
}
with (ROOT / 'reconciliation.json').open('x') as handle:
    json.dump(report, handle, indent=2)
    handle.write('\n')
print(json.dumps({'engineering': report['engineering'], 'budget_seconds': budget}, indent=2))

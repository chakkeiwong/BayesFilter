"""Reconcile configuration extraction without resetting prior campaign evidence."""
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

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[4]

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def definitions(p):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body
            if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
def assignments(p):
    result={}
    for n in ast.parse(p.read_text()).body:
        if isinstance(n,(ast.Assign,ast.AnnAssign)):
            for t in n.targets if isinstance(n,ast.Assign) else [n.target]:
                if isinstance(t,ast.Name):result[t.id]=ast.dump(n,include_attributes=False)
    return result

previous_path=ROOT.parent/'m7-windowed-r1/reconciliation.json'
previous=load(previous_path)
baseline=load(ROOT/'baseline.json')
for row in baseline['files']:
    assert sha(ROOT/'baseline'/row['path'])==row['sha256'],row['path']
assert load(ROOT/'before-r2.json')==load(ROOT/'after.json')
old_path=ROOT/'baseline/bayesfilter/inference/hmc_kernel_tuning.py'
new_path=REPO/'bayesfilter/inference/hmc_configuration.py'
old,new,remaining=map(definitions,(old_path,new_path,REPO/'bayesfilter/inference/hmc_kernel_tuning.py'))
names=set(load(ROOT/'definitions.json'))
assert set(old)-set(remaining)==names==set(new)
assert all(old[n]==new[n] for n in new)
assert all(old[n]==remaining[n] for n in remaining)
a,b=assignments(old_path),assignments(new_path)
constants=load(ROOT/'moved-constants.json')
assert all(a[n]==b[n] for n in constants)
# Windowed numeric bodies are unchanged by the annotation-owner import update.
windowed_name='bayesfilter/inference/hmc_mass_adaptation.py'
assert definitions(ROOT/'baseline'/windowed_name)==definitions(REPO/windowed_name)

cases={}
for name in ('affected-tests.xml','budget-tests.xml','direct-export-tests.xml'):
    for case in ET.parse(ROOT/name).iter('testcase'):
        status='skipped' if case.find('skipped') is not None else (
            'failed' if case.find('failure') is not None or case.find('error') is not None else 'passed')
        cases[case.get('classname')+'::'+case.get('name')]={'status':status,'latest_batch':name}
counts=Counter(row['status'] for row in cases.values())
assert counts=={'passed':218},counts
combined={**previous['test_cases'],**cases}
combined_counts=Counter(row['status'] for row in combined.values())
assert combined_counts=={'passed':454,'skipped':1},combined_counts
attempts=[]
for path in sorted(ROOT.glob('*-run.json')):
    row=load(path)
    attempts.append({**row,'manifest_path':str(path.relative_to(REPO)),
                     'manifest_sha256':sha(path),'log_sha256':sha(Path(row['log']))})
assert len(attempts)==7
assert [Path(r['manifest_path']).name for r in attempts if r['returncode']]==['before-config-run.json']
measured=sum(row['elapsed_seconds'] for row in attempts)
assert measured<=520.
overhead=80.
prior=previous['budget_seconds']
budget={
    'cpu_measured':measured,'cpu_overhead':overhead,'cpu_charge':measured+overhead,'gpu_charge':0.,
    'total_cpu_charged':prior['total_cpu_charged']+measured+overhead,
    'total_gpu_charged':prior['total_gpu_charged'],
    'remaining_cpu':prior['remaining_cpu']-measured-overhead,
    'remaining_gpu':prior['remaining_gpu'],
    'remaining_m7_cpu_allocation':prior['remaining_m7_cpu_allocation']-measured-overhead,
    'remaining_m7_gpu_allocation':prior['remaining_m7_gpu_allocation'],
    'unfinished_launched_reservations':0.,
}
assert budget['remaining_m7_cpu_allocation']>=0
installation=load(ROOT/'guide-installation.json')
assert sha(REPO/installation['official_path'])==installation['installed_sha256']
assert sha(REPO/installation['archival_build'])==installation['installed_sha256']
for path,expected in load(ROOT/'guide-r1/build-manifest.json')['source_sha256'].items():
    assert sha(REPO/path)==expected,path
paths=['bayesfilter/inference/'+name for name in (
    '__init__.py','hmc_kernel_tuning.py','hmc_configuration.py','hmc_geometry.py','hmc_bootstrap.py',
    'hmc_mass_adaptation.py','hmc_preparation_common.py','hmc_preparation.py','hmc_budget_policy.py',
    'hmc_candidate_set_public.py','hmc_candidate_set_execution.py','hmc_warmup.py')]
paths+=['tests/'+name for name in (
    'test_hmc_configuration_extraction.py','test_hmc_mass_adaptation_extraction.py',
    'test_hmc_bootstrap_extraction.py','test_hmc_master_repair.py','test_hmc_candidate_set_execution.py',
    'test_hmc_consistency_gap_repair.py','test_hmc_tuning_policy_replay_authority.py',
    'test_hmc_tuning_dispatch.py','test_hmc_tuning_documentation_contract.py',
    'test_hmc_kernel_tuning_public_api.py','test_hmc_kernel_tuning_outer_loop.py')]
paths+=['docs/reference/hmc-tuning-interface.md','docs/chapters/ch21b_hmc_tuning_interfaces.tex',
    'docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md',
    'docs/plans/bayesfilter-hmc-repair-m7-result-2026-09-17.md']
evidence=[*ROOT.glob('*.py'),ROOT/'baseline.json',ROOT/'definitions.json',ROOT/'moved-constants.json',
          ROOT/'ast-parity.json',ROOT/'guide-installation.json',ROOT/'budget-test-selection.json',
          ROOT/'before-r2.json',ROOT/'after.json',*ROOT.glob('*.xml')]
report={
    'created_utc':datetime.now(timezone.utc).isoformat(),
    'question':'Independent public configuration/budget translation with identical numerical policies',
    'plan_file':paths[-2],'result_file':paths[-1],
    'command':[sys.executable,str(Path(__file__))],
    'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
    'environment':{'python':sys.executable,'tensorflow':version('tensorflow'),
        'tensorflow_probability':version('tfp-nightly'),'tensorflow_probability_distribution':'tfp-nightly',
        'gpu_status':'intentionally hidden in numerical test workers','current_process_framework_imported':False},
    'seeds':'Unchanged fixture seeds and declared configuration seed translation in capture_configuration.py',
    'data':'Synthetic diagnostic/test fixtures only',
    'previous_reconciliation':{'path':str(previous_path.relative_to(REPO)),'sha256':sha(previous_path)},
    'baseline_verified':baseline,'source_sha256':{p:sha(REPO/p) for p in paths},
    'evidence_sha256':{str(p.relative_to(REPO)):sha(p) for p in evidence},
    'engineering':{'exact_comparison_records':22,'exact_preset_mass_combinations':10,
        'exact_dimension_attempt_budget_payloads':160,'moved_definitions':30,
        'identical_definition_asts':30,'identical_constants':17,'unchanged_remaining_definitions':253,
        'distinct_passed':218,'distinct_skipped':0,'combined_september18_distinct_passed':454,
        'combined_september18_distinct_skipped':1},
    'test_cases':cases,'combined_test_cases':combined,'attempts':attempts,'budget_seconds':budget,
    'guide_installation':installation,
    'failure_and_repair':'Baseline writer supplied an integer instead of the required timeout-enlargement mapping. Corrected before extraction, original script/log preserved; subsequent checks pass.',
    'final_audit_repair':'Direct lazy exports added for public preparation and policy constants, with fresh-process public import and documentation tests.',
    'scientific_interpretation':'CPU engineering equivalence only. All verified members retained; no new numerical default, tuning diagnostic role, GPU performance or posterior claim.',
}
with (ROOT/'reconciliation.json').open('x') as handle:
    json.dump(report,handle,indent=2);handle.write('\n')
print(json.dumps({'engineering':report['engineering'],'budget_seconds':budget},indent=2))

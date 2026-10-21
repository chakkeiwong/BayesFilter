"""Static executable wiring audit of one historical SUCCESS report."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

repo = Path(sys.argv[1])
destination = Path(sys.argv[2])
artifact = repo / 'docs/benchmarks/artifacts/ledh_trust_region_lgssm_t50_20260903/result.json'
result = json.loads(artifact.read_text())
revision = result['git_commit']
paths = ['docs/benchmarks/run_ledh_trust_region_phase3_lgssm_t50.py',
         'docs/benchmarks/run_moment_retuned_genut_whole_leaderboard.py',
         'bayesfilter/highdim/cubature_genut_filter.py',
         'bayesfilter/highdim/cubature_genut_adapters.py']
source_records = []
for path in paths:
    recorded = subprocess.check_output(['git','show',revision+':'+path], cwd=repo)
    assert recorded == (repo/path).read_bytes(), path
    source_records.append(dict(path=path, recorded_source_equals_current=True,
                               sha256=hashlib.sha256(recorded).hexdigest()))
def function(path, name):
    return next(n for n in ast.parse((repo/path).read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
def calls(node):
    return sorted({ast.unparse(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)})
evaluator = function(paths[0], '_make_evaluator')
gate = function(paths[0], '_valid')
stage = function(paths[2], 'finite_value_score')
adapter = function(paths[3], 'diagonal_lgssm_candidate_adapter')
evaluator_calls, stage_calls = calls(evaluator), calls(stage)
assert 'finite_value_score' in evaluator_calls
assert not any('canonical_' in c for c in evaluator_calls)
assert 'adapter.transition_value' in stage_calls and 'adapter.observation_value' in stage_calls
assert not any(('canonical_' in c or 'ledh_flow' in c or 'ukf' in c) for c in stage_calls)
transition = next(n for n in adapter.body if isinstance(n, ast.FunctionDef) and n.name == 'transition_value')
report = dict(classification='static_call_chain_check_not_numerical_parity',
    recorded_commit=revision, source_matches=source_records, target=result['target'],
    recorded_status=result['status'], selected_label=result['selected_config']['label'],
    selected_controls=result['selected_config']['controls'],
    rows_count=len(result['selected_config']['rows']), evaluator_calls=evaluator_calls,
    stage_calls=[c for c in stage_calls if c.startswith('adapter.') or 'restore' in c or 'shape' in c],
    validity_gate_source=ast.unparse(gate), transition_source=ast.unparse(transition),
    claim_boundary='This specific SUCCESS report exercised the staged Cubature/GenUT candidate, not the canonical UKF/LEDH-flow recursion. Other prior LGSSM reports are not adjudicated.')
destination.write_text(json.dumps(report, indent=2)+'\n')
print('PASS: four recorded source files match; the SUCCESS evaluator uses the distinct staged candidate.')

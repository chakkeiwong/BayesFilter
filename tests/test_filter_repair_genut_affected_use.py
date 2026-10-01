"""Source-bound diagnostic disposition of reduced GenUT numerical failures."""

import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_reduced_genut_affected_use_disposition(request):
    record_path = ROOT/'docs/plans/filter_gradient_genut_affected_use_sources_20260930.json'
    record = json.loads(record_path.read_text())
    for path, expected in record['sources'].items():
        assert digest(ROOT/path) == expected, path
    calls = []
    for base in ('bayesfilter', 'experiments/dpf_implementation/tf_tfp'):
        for path in sorted((ROOT/base).rglob('*.py')):
            source = path.read_text()
            if 'dual_cap_genut_primal' not in source:
                continue
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
                    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr
                    if name == 'dual_cap_genut_primal':
                        calls.append({'path': str(path.relative_to(ROOT)), 'line': node.lineno, 'call': ast.unparse(node)})
    assert calls == record['direct_runtime_reduced_calls']
    assert len(calls) == 1 and calls[0]['path'] == 'bayesfilter/highdim/genut_guided_proposal_tf.py'
    reduced = ast.parse((ROOT/'bayesfilter/highdim/dual_cap_genut_primal_tf.py').read_text())
    flags = {target.id: node.value.value for node in reduced.body if isinstance(node, ast.Assign)
             and isinstance(node.value, ast.Constant) for target in node.targets if isinstance(target, ast.Name)}
    assert flags['CANONICAL_LEDH_ADMITTED'] is False
    assert 'Primal-only' in ast.get_docstring(reduced)
    score_source = (ROOT/'bayesfilter/highdim/ledh_canonical_score_tf.py').read_text()
    assert 'dual_cap_genut_primal' not in score_source and 'batched_higher_moment_shape_jvp(' in score_source
    assert 'GradientTape' not in score_source and 'ForwardAccumulator' not in score_source
    value_source = (ROOT/'bayesfilter/highdim/ledh_canonical_value_program_tf.py').read_text()
    assert 'fraction_coordinatewise_cap_active' not in value_source
    registry = ast.parse((ROOT/'bayesfilter/highdim/ledh_alg1_contract.py').read_text())
    setting = next(ast.literal_eval(n.value) for n in registry.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'LEDH_PRODUCTION_PROGRAM_V1' for t in n.targets))
    assert setting['filter']['dual_cap_enabled'] and setting['filter']['trust_region_enabled']
    witnesses = []
    for number in (4936, 4938):
        folder = RAW/f'run-{number:05d}'
        run = json.loads((folder/'run.json').read_text())
        assert run['state'] == 'passed'
        value = json.loads((folder/'genut-consumer-value.json').read_text())
        assert [r['runtime_counts'] for r in value['records']] == [[0, 0], [1, 0], [0, 1]]
        assert all(r['maximum_output_error'] == 0. for r in value['records'])
        analytical = json.loads((folder/'genut-consumer-analytical.json').read_text())
        assert analytical['reduced_primal_blocked'] and analytical['runtime_batched_jvp_calls'] == 2
        assert analytical['finite'] and analytical['maximum_output_error'] == 0.
        for path, sha in value['sources'].items():
            # Incoming annealed and shape-preservation score work is separately
            # qualified through05202; it still has no reduced-primal/AD binding.
            if path == 'bayesfilter/highdim/ledh_canonical_score_tf.py':
                continue
            assert digest(ROOT/path) == sha
        witnesses.append({'run': number, 'manifest_sha256': digest(folder/'run.json'),
            'value_sha256': digest(folder/'genut-consumer-value.json'),
            'analytical_sha256': digest(folder/'genut-consumer-analytical.json')})
    merged = RAW/'run-05202'
    assert json.loads((merged/'run.json').read_text())['state'] == 'passed'
    result = {'schema': 'filter_repair.genut_affected_use_readback.v1',
        'source_record_sha256': digest(record_path), 'wiring_witnesses': witnesses,
        'merged_terminal_manifest_sha256': digest(merged/'run.json'),
        'unsupported_uses': record['unsupported_uses'],
        'nonclaims': ['Reduced reverse-AD and cap failures remain failed and unsupported.',
                     'Static inspection and finite wiring fixtures do not prove external callback purity.',
                     'No canonical LEDH, precision repair, posterior or HMC admission.']}
    (Path(request.config.getoption('xmlpath')).parent/'genut-affected-use-readback.json').write_text(
        json.dumps(result, indent=2)+'\n')

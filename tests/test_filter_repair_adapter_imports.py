"""Public export compatibility and fresh-process reference-import isolation."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import audit_filter_gradient_policy as inventory

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / 'tests/fixtures/filter_repair_adapter_exports.json'


def test_export_map_and_order_are_unchanged():
    expected = json.loads(FIXTURE.read_text())
    tree = ast.parse((ROOT / 'bayesfilter/adapters/__init__.py').read_text())
    assignments = {target.id: ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, ast.Assign) for target in node.targets
        if isinstance(target, ast.Name) and target.id in
        ('_EXPORT_MODULES', '_EXPORT_ATTRIBUTES', '__all__')}
    actual = {name: {'module': module,
        'attribute': assignments['_EXPORT_ATTRIBUTES'].get(name, name)}
        for name, module in assignments['_EXPORT_MODULES'].items()}
    assert actual == expected['exports']
    assert assignments['__all__'] == expected['export_order']


def test_inventory_records_implicit_parents_and_lazy_aliases(tmp_path, monkeypatch):
    sources = {
        'alpha/__init__.py': 'from alpha.reference import marker\n',
        'alpha/reference.py': 'import numpy as np\nmarker = 1\n',
        'alpha/runtime.py': 'def kernel():\n    return 1\n',
        'beta/__init__.py': "_EXPORT_MODULES = {'public': 'alpha.runtime'}\n_EXPORT_ATTRIBUTES = {'public': 'kernel'}\n",
        'entry.py': 'from beta import public\ndef run():\n    return public()\n',
    }
    for relative, source in sources.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.setattr(inventory, 'ROOT', tmp_path)
    monkeypatch.setattr(inventory.subprocess, 'check_output',
        lambda args, **kwargs: '\n'.join(sources) if 'ls-files' in args else 'fixture-commit\n')
    result = inventory.audit()
    assert any(row['module'] == 'alpha.runtime' and row['parent'] == 'alpha'
        for row in result['implicit_package_import_edges'])
    assert any(row['caller'] == 'entry.run' and row['target'] == 'alpha.runtime.kernel'
        for row in result['resolved_static_call_edges'])


IMPORT_CHECK = '''
import importlib
import importlib.abc
import json
import sys

case = CASE
reference_modules = ('bayesfilter.adapters.macrofinance', 'bayesfilter.filters',
    'bayesfilter.linear.types', 'bayesfilter.linear.kalman_derivatives_numpy',
    'bayesfilter.results')
def forbidden(name):
    return any(name == item or name.startswith(item + '.') for item in reference_modules)
class NoReferenceImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if forbidden(fullname):
            raise AssertionError('Runtime imported reference module: ' + fullname)
        return None
if case != 'explicit_reference':
    sys.meta_path.insert(0, NoReferenceImport())
import bayesfilter.adapters as adapters
assert not any(forbidden(name) for name in sys.modules)
assert 'bayesfilter.adapters.bgs' not in sys.modules
assert 'BGSPosteriorAdapter' in dir(adapters)
try:
    adapters.no_such_adapter
except AttributeError:
    pass
else:
    raise AssertionError('Unknown export did not raise AttributeError')
if case == 'direct':
    import bayesfilter.adapters.bgs as bgs
    assert bgs.constrained_log_prior_and_score.function_spec.jit_compile
    assert bgs.constrained_log_prior_and_score.input_signature is not None
elif case == 'public':
    from bayesfilter.adapters import BGSPosteriorAdapter, constrained_log_prior_and_score
    from bayesfilter.adapters.bgs import BGSPosteriorAdapter as direct
    assert BGSPosteriorAdapter is direct
    assert constrained_log_prior_and_score.function_spec.jit_compile
elif case == 'aliases':
    from bayesfilter.adapters import bgs_log_abs_det_jacobian, bgs_theta_from_unconstrained, bgs_unconstrained_from_theta
    from bayesfilter.adapters import bgs
    assert bgs_log_abs_det_jacobian is bgs.log_abs_det_jacobian
    assert bgs_theta_from_unconstrained is bgs.theta_from_unconstrained
    assert bgs_unconstrained_from_theta is bgs.unconstrained_from_theta
elif case == 'explicit_reference':
    from bayesfilter.adapters import evaluate_macrofinance_provider_likelihood
    from bayesfilter.adapters.macrofinance import evaluate_macrofinance_provider_likelihood as direct
    assert evaluate_macrofinance_provider_likelihood is direct
if case != 'explicit_reference':
    assert not any(forbidden(name) for name in sys.modules)
print(json.dumps({'case': case, 'passed': True,
    'loaded_reference_modules': sorted(name for name in sys.modules if forbidden(name))}))
'''


@pytest.mark.parametrize('case', ['package', 'direct', 'public', 'aliases', 'explicit_reference'])
def test_real_fresh_process_import(case, request):
    directory = Path(request.config.getoption('xmlpath')).parent
    code = IMPORT_CHECK.replace('CASE', repr(case))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', TF_FORCE_GPU_ALLOW_GROWTH='true')
    result = subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=60, check=False)
    (directory / f'adapter-import-{case}.log').write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report['passed'] and report['case'] == case

"""Diagnostic execution of the current DZ5 parent's exact deadline function.

The external source remains untouched. These standard-library workers test
quiet completion, blocked calls and process-group cleanup, not a model run.
"""

import ast
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

import pytest

ROOT = Path('/home/ubuntu/workspace/MacroFinance-dz5-neutra')
SUPERVISOR = ROOT / 'scripts/run_bayesfilter_estimation.py'
CALLER = ROOT / 'scripts/run_dz5_cdf_proposal.py'


def _running(pid):
    try:
        status = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return status[0] != 'Z'


@pytest.mark.parametrize('case', ('quiet', 'blocked', 'descendant', 'ignores_term'))
def test_actual_dz5_parent_deadline(case, request):
    source, caller = SUPERVISOR.read_text(), CALLER.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'supervise')
    imports = '\n'.join(ast.get_source_segment(source, node) for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom)))
    excerpt = imports + '\n' + ast.get_source_segment(source, function)
    namespace = {'REPO_ROOT': ROOT}
    exec(compile(excerpt, str(SUPERVISOR), 'exec'), namespace)  # noqa: S102 - exact external diagnostic function
    caller_tree = ast.parse(caller)
    assert any(isinstance(node, ast.ImportFrom) and node.module == 'scripts.run_bayesfilter_estimation'
        and any(alias.name == 'supervise' for alias in node.names) for node in ast.walk(caller_tree))
    calls = [node for node in ast.walk(caller_tree) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == 'supervise']
    assert len(calls) == 1 and any(k.arg == 'timeout_seconds' and ast.unparse(k.value) == 'cap'
        for k in calls[0].keywords)
    directory = Path(request.config.getoption('xmlpath')).parent
    child_path = directory / 'descendant-pid.json'
    program = "import time; time.sleep(.25); print('completed', flush=True)"
    timeout = 3.
    if case == 'blocked':
        program, timeout = 'import time; time.sleep(60)', .4
    if case == 'descendant':
        program = ("import json, pathlib, subprocess, sys; "
            "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
            "pathlib.Path(sys.argv[1]).write_text(json.dumps({'pid':child.pid})); child.wait()")
        timeout = .7
    if case == 'ignores_term':
        program = ("import pathlib, signal, sys, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "pathlib.Path(sys.argv[1]).write_text('ready'); time.sleep(60)")
        child_path = directory / 'ignores-term-ready.txt'
        timeout = .7
    environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'TF_FORCE_GPU_ALLOW_GROWTH': 'true'}
    command = [sys.executable, '-c', program, str(child_path)]
    receipt = namespace['supervise'](command, timeout_seconds=timeout,
        log_path=directory / f'dz5-{case}.log', environment=environment)
    child = json.loads(child_path.read_text())['pid'] if case == 'descendant' and child_path.exists() else None
    child_running = child is not None and _running(child)
    try:
        observations = [json.loads(line) for line in Path(receipt['progress_path']).read_text().splitlines()]
        report = {'schema': 'filter_dz5_parent_deadline.v1', 'case': case, 'receipt': receipt,
            'supervisor_source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'supervisor_function_sha256': hashlib.sha256(excerpt.encode()).hexdigest(),
            'caller_source_sha256': hashlib.sha256(caller.encode()).hexdigest(),
            'caller_lines': [calls[0].lineno, calls[0].end_lineno],
            'parent_reaped': not Path(f'/proc/{receipt["pid"]}').exists(),
            'descendant_pid': child, 'descendant_running_after_return': child_running,
            'events': observations,
            'nonclaims': ['No model/target execution, source deployment, GPU cost or initializer qualification.',
                'Ordinary blocked/quiet workers; not adversarial process escape containment.']}
        with (directory / f'dz5-deadline-{case}.json').open('x') as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
        assert SUPERVISOR.read_text() == source and CALLER.read_text() == caller
        assert report['parent_reaped'] and not child_running
        assert observations[0]['event'] == 'started' and observations[-1]['event'] == 'finished'
        assert receipt['timed_out'] == (case != 'quiet')
        assert receipt['elapsed_seconds'] < timeout + 7.
        if case == 'quiet':
            assert receipt['returncode'] == 0 and Path(receipt['log_path']).read_text().strip() == 'completed'
        else:
            assert receipt['returncode'] != 0
        if case == 'descendant':
            assert child is not None, 'Child never started; cleanup was not tested'
        if case == 'ignores_term':
            assert child_path.read_text() == 'ready' and receipt['returncode'] == -signal.SIGKILL
    finally:
        if child_running:
            try:
                if os.getpgid(child) == receipt['pid']:
                    os.killpg(receipt['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass

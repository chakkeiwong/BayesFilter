"""CPU-only checks of native RNG instrumentation and queued consumer wiring."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import run_controlled_smc_kdm as runner


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT/path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = load_module('forward_capture', 'docs/benchmarks/capture_lorenz_forward_reference.py')
comparison = load_module('forward_comparison', 'docs/benchmarks/compare_lorenz_forward_tf.py')


class NativeDrawChecks(unittest.TestCase):
    def test_instrumentation_preserves_actual_octave_values_and_rng_states(self):
        stats = ROOT/'.localresources/code/octave-statistics-1.5.0/inst'
        source = ROOT/'.localresources/code/heng-controlled-smc/StateSpaceModels/Lorenz'
        if not shutil.which('octave') or not all(path.is_file() for path in (
            stats/'distributions/mvnrnd.m', source/'Systematic_Resampling.m'
        )):
            self.skipTest('native RNG reference check needs Octave and pinned author/statistics sources')
        with tempfile.TemporaryDirectory(prefix='forward-rng-check-') as tmp:
            directory = Path(tmp)
            observed = directory/'observed'
            observed.mkdir()
            (observed/'mvnrnd.m').write_text(capture.instrument_sampler((stats/'distributions/mvnrnd.m').read_text()))
            (observed/'Systematic_Resampling.m').write_text(capture.instrument_resampling((source/'Systematic_Resampling.m').read_text()))
            q = capture.octave_quote
            script = f"""args=argv(); arm=args{{1}};
addpath({q(stats)});addpath({q(stats/'distributions')});addpath({q(source)});
if strcmp(arm,'observed')
 addpath({q(observed)},'-begin');
 assert(strcmp(which('mvnrnd'),{q(observed/'mvnrnd.m')}));
 assert(strcmp(which('Systematic_Resampling'),{q(observed/'Systematic_Resampling.m')}));
end
rand('state',123);randn('state',456);
global P0_NORMAL_DRAWS P0_UNIFORM_DRAWS;
P0_NORMAL_DRAWS={{}};P0_UNIFORM_DRAWS={{}};
x=mvnrnd([1,2;3,4;5,6],[2,.3;.3,1]);
anc=Systematic_Resampling([.01;.09;.3;.6]);
r=rand('state');rn=randn('state');
if strcmp(arm,'unchanged')
 save('-mat7-binary','baseline.mat','x','anc','r','rn');
else
 expected=load('baseline.mat');
 assert(isequal(x,expected.x) && isequal(anc,expected.anc));
 assert(isequal(r,expected.r) && isequal(rn,expected.rn));
 assert(numel(P0_NORMAL_DRAWS)==1 && numel(P0_UNIFORM_DRAWS)==1);
 rand('state',123);randn('state',456);
 assert(isequal(P0_NORMAL_DRAWS{{1}},randn(3,2)));
 assert(isequal(P0_UNIFORM_DRAWS{{1}},rand(1)));
 disp('NATIVE_VALUES_AND_RANDOM_STREAM_EXACT');
end
"""
            driver = directory/'test.m'
            driver.write_text(script)
            for arm in ('unchanged', 'observed'):
                result = subprocess.run(['octave', '--no-gui', '--quiet', str(driver), arm],
                                        cwd=directory, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('NATIVE_VALUES_AND_RANDOM_STREAM_EXACT', result.stdout)

    def test_unknown_source_rejected(self):
        for instrument in (capture.instrument_sampler, capture.instrument_resampling):
            with self.assertRaises(ValueError):
                instrument('changed source')


class VerdictChecks(unittest.TestCase):
    def setUp(self):
        self.stages = [dict(valid=True, value_screen_passed=True, ancestry_exact=True)]
        self.failure = dict(precision_rejected=True, forward_rejected=True)

    def test_accepted_author_failure_is_required_reference_behavior(self):
        self.assertEqual(comparison.comparison_verdict(self.stages, self.failure),
                         'FIXED_POLICY_FORWARD_SCREEN_PASSED')

    def test_extra_failure_and_missing_failure_check_cannot_pass(self):
        invalid = copy.deepcopy(self.stages)
        invalid[0]['valid'] = False
        self.assertEqual(comparison.comparison_verdict(invalid, self.failure), 'EXTRA_PORT_INVALID_RESULT')
        self.assertNotEqual(comparison.comparison_verdict([], self.failure), 'FIXED_POLICY_FORWARD_SCREEN_PASSED')
        for field in ('precision_rejected', 'forward_rejected'):
            failed = dict(self.failure, **{field: None})
            self.assertNotEqual(comparison.comparison_verdict(self.stages, failed), 'FIXED_POLICY_FORWARD_SCREEN_PASSED')

    def test_completed_mismatch_is_not_acceptance(self):
        for field in ('value_screen_passed', 'ancestry_exact'):
            stage = dict(self.stages[0], **{field: False})
            self.assertEqual(comparison.comparison_verdict([stage], self.failure),
                             'FORWARD_MISMATCH_REQUIRES_LOCALIZATION')


class QueueChecks(unittest.TestCase):
    def test_comparison_inputs_resolve_successful_reference_arrays(self):
        with tempfile.TemporaryDirectory(prefix='forward-input-check-') as tmp:
            root = Path(tmp)
            records = []
            for phase, index_name in (('p0_forward_capture', 'fixture.json'), ('c1_source_failure_replay', 'failed-policy.json')):
                directory = root/phase
                directory.mkdir()
                (directory/'array.bin').write_bytes(b'fixture')
                (directory/index_name).write_text(json.dumps(dict(arrays={'x': dict(file='array.bin')})))
                (directory/'summary.json').write_text('{"status":"PASS"}')
                records.append((None, dict(phase=phase, status='COMPLETE', decision={'status':'PASS'}, output_root=str(directory))))
            command, inputs = runner.p0_forward_inputs('p0_forward_compare', root/'output', records)
            self.assertEqual(command[2], str(root/'output/sources/compare_lorenz_forward_tf.py'))
            self.assertEqual(command[command.index('--reference')+1], str(root/'p0_forward_capture'))
            self.assertEqual(command[command.index('--failure')+1], str(root/'c1_source_failure_replay'))
            self.assertEqual(len(inputs), 6)
            (root/'p0_forward_capture/array.bin').unlink()
            with self.assertRaises(FileNotFoundError):
                runner.p0_forward_inputs('p0_forward_compare', root/'output', records)
            records[0][1]['decision']['status'] = 'FAIL'
            with self.assertRaisesRegex(RuntimeError, 'Missing successful input'):
                runner.p0_forward_inputs('p0_forward_compare', root/'output', records)

    def readiness(self, allowance, pending=True):
        current = dict(passed=[], remaining_seconds={'cpu': allowance}, tranche_remaining_seconds={'cpu': allowance})
        records = [(None, dict(phase='c1_source_full', status='RUNNING', resource='cpu', timeout_seconds=1000))]
        queue = lambda name: dict(status='WAITING' if pending and name=='c1_source_failure_replay' else 'NOT_QUEUED', supervisor_alive=True)
        with tempfile.TemporaryDirectory(prefix='forward-budget-check-') as tmp:
            root = Path(tmp)
            (root/'opening-budget.json').write_text('{"attempt_limit":148}')
            with patch.object(runner, 'CAMPAIGN', root), patch.object(runner, 'status', return_value=current), \
                 patch.object(runner, 'campaign_attempts', return_value=records), patch.object(runner, 'queue_wait_state'), \
                 patch.object(runner, 'queue_status', side_effect=queue), patch.object(runner.shared, 'charge', return_value=0):
                return runner.p0_forward_readiness('p0_forward_capture')

    def test_pending_and_running_work_reserve_budget(self):
        with self.assertRaisesRegex(RuntimeError, 'Insufficient unreserved'):
            self.readiness(1149)
        self.readiness(1150)

    def test_source_failure_does_not_require_full_source_success_for_port(self):
        for phase in runner.P0_FORWARD_PHASES:
            self.assertIn(phase, runner.ACTIVE_PHASES)
            self.assertNotIn('c1_source_full', runner.PHASES[phase]['dependencies'])
            self.assertNotIn('c2_source_results', runner.PHASES[phase]['dependencies'])
            self.assertEqual(runner.PHASES[phase]['max_attempts'], 3 if phase=='p0_tf_solver_compatibility' else 2 if phase in ('p0_fitting_forward','p0_offset_localization') or phase in runner.P0_PERTURB_PHASES else 1)


if __name__ == '__main__':
    unittest.main()

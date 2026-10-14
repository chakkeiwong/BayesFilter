"""CPU-only checks: failed fits cannot disappear behind diagnostic PASS."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import run_controlled_smc_kdm as runner
spec = importlib.util.spec_from_file_location('fit_forward', ROOT/'docs/benchmarks/compare_lorenz_fitting_forward_tf.py')
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class FittingForwardChecks(unittest.TestCase):
    def test_missing_or_rejected_fit_never_passes(self):
        self.assertEqual(comparison.solver_verdict([]), 'NO_COMPARISON')
        for row in (dict(fit_valid=False), dict(fit_valid=True, forward_valid=None),
                    dict(fit_valid=True, forward_valid=False)):
            self.assertEqual(comparison.solver_verdict([row]),
                             'ADDITIONAL_INVALID_RESULT_ON_SOURCE_VALID_INPUT')

    def test_policy_and_downstream_screens_both_required(self):
        good = dict(fit_valid=True, forward_valid=True, forward_screen_passed=True,
                    policy_screen_passed=True)
        self.assertEqual(comparison.solver_verdict([good]), 'SCOPED_FITTING_FORWARD_SCREEN_PASSED')
        for field in ('forward_screen_passed', 'policy_screen_passed'):
            self.assertEqual(comparison.solver_verdict([dict(good, **{field: False})]),
                             'FITTING_DOWNSTREAM_MISMATCH')

    def test_scientific_verdict_survives_completion_status(self):
        summary = dict(status='PASS', fitting_forward_screen_passed=False,
                       solver_verdicts={'raw':'ADDITIONAL_INVALID_RESULT_ON_SOURCE_VALID_INPUT'})
        self.assertEqual(runner.compact_decision(summary), summary)

    def test_completed_source_arrays_and_forward_control_are_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = []
            for phase in ('p0_forward_capture', 'p0_forward_compare'):
                directory = root/phase
                directory.mkdir()
                (directory/'summary.json').write_text('{"status":"PASS"}')
                records.append((None, dict(phase=phase, status='COMPLETE', decision={'status':'PASS'}, output_root=str(directory))))
            reference = root/'p0_forward_capture'
            (reference/'x.bin').write_bytes(b'saved source')
            (reference/'fixture.json').write_text(json.dumps(dict(arrays={'x':dict(file='x.bin')})))
            command, inputs = runner.p0_forward_inputs('p0_fitting_forward', root/'out', records)
            self.assertEqual(command[2], str(root/'out/sources/compare_lorenz_fitting_forward_tf.py'))
            self.assertEqual(command[command.index('--reference')+1], str(reference))
            self.assertEqual(len(inputs), 4)
            self.assertIn(str(root/'p0_forward_compare/summary.json'), inputs)
            records[-1][1]['status'] = 'FAILED'
            with self.assertRaisesRegex(RuntimeError, 'Missing successful input'):
                runner.p0_forward_inputs('p0_fitting_forward', root/'out', records)


if __name__ == '__main__':
    unittest.main()

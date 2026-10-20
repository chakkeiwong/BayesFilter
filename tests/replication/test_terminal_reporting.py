"""CPU-only reporting checks; no numerical execution or evidence mutation."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_controlled_smc_kdm as runner


class TerminalReportingTests(unittest.TestCase):
    def test_completed_audit_preserves_failed_numeric_and_dtype_screens(self):
        summary = dict(status='PASS', replication='INCOMPLETE',
                       numerical_screen_passed=False,
                       dtype_screen_passed={'float32':False,'float64':True},
                       comparison_verdict='FP64_PASS_FP32_FAIL_UNCHANGED_TOLERANCES')
        self.assertEqual(runner.compact_decision(summary), summary)

    def test_new_compact_decision_preserves_false_comparison_result(self):
        summary = dict(status='PASS', scope='MATCHED_AUTHOR_FIXED_POLICY_FORWARD',
                       replication='INCOMPLETE', forward_screen_passed=False,
                       comparison_verdict='FAILED_POLICY_FORWARD_NOT_CONFIRMED')
        self.assertEqual(runner.compact_decision(summary), summary)

    def test_historical_read_uses_verified_summary_without_mutating_manifest(self):
        for passed in (False, True):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'summary.json'
                summary = dict(status='PASS', scope='MATCHED_AUTHOR_FIXED_POLICY_FORWARD',
                               replication='INCOMPLETE', forward_screen_passed=passed,
                               comparison_verdict='PASS_SCREEN' if passed else 'FAILED_POLICY_FORWARD_NOT_CONFIRMED')
                raw = json.dumps(summary).encode()
                path.write_bytes(raw)
                manifest = dict(phase='p0_forward_compare', status='COMPLETE', output_root=directory,
                                decision={k:summary[k] for k in ('status','scope','replication')},
                                output_sha256={'summary.json':hashlib.sha256(raw).hexdigest()})
                frozen = json.dumps(manifest)
                result = runner.reported_decision(manifest)
                self.assertEqual(result['comparison_verdict'],summary['comparison_verdict'])
                self.assertIs(result['forward_screen_passed'],passed)
                self.assertEqual(result['summary_verification'],'VERIFIED_PRESERVED_SUMMARY')
                self.assertEqual(json.dumps(manifest),frozen)

    def test_modified_or_missing_summary_cannot_report_a_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'summary.json'
            path.write_text(json.dumps(dict(status='PASS',forward_screen_passed=True,
                                            comparison_verdict='PASS_SCREEN')))
            manifest = dict(phase='p0_forward_compare',status='COMPLETE',output_root=directory,
                            decision={'status':'PASS'},output_sha256={'summary.json':'wrong'})
            for exists in (True,False):
                if not exists:
                    path.unlink()
                result = runner.reported_decision(manifest)
                self.assertEqual(result['comparison_verdict'],'RESULT_SUMMARY_UNVERIFIED')
                self.assertIs(result['forward_screen_passed'],False)


    def test_progress_selects_current_work_instead_of_old_source_failure(self):
        for terminal_status in ('RUNNING', 'COMPLETE'):
            with self.subTest(status=terminal_status), tempfile.TemporaryDirectory() as directory:
                old = dict(phase='c1_source_full', status='FAILED', output_root=directory,
                           decision={'status':'FAIL'})
                new = dict(phase='p0_fitting_forward', status=terminal_status, output_root=directory,
                           decision={'status':'PASS', 'fitting_forward_screen_passed':False})
                records = [(Path(directory)/'old.json', old), (Path(directory)/'new.json', new)]
                current = dict(replication_target='author', active_program_stage='P0',
                               attempts=2, running=[], remaining_seconds={'cpu':500,'gpu':500},
                               tranche_remaining_seconds={'cpu':100,'gpu':100},
                               csmc_replication='INCOMPLETE', kdm_replication='INCOMPLETE')
                with (
                    patch.object(runner, 'status', return_value=current),
                    patch.object(runner, 'campaign_attempts', return_value=records),
                    patch.object(runner, 'queue_status', return_value={}),
                    patch.object(runner.shared, 'charge', return_value=0),
                    patch.object(runner.shared, 'alive', return_value=False),
                ):
                    report = runner.progress()
                self.assertEqual(report['attempt']['phase'], 'p0_fitting_forward')
                self.assertEqual(report['attempt']['status'], terminal_status)
                self.assertEqual(report['remaining_seconds'], {'cpu':100, 'gpu':100})
                self.assertEqual(report['inherited_remaining_seconds'], {'cpu':500, 'gpu':500})
                self.assertIs(report['phase_decision']['fitting_forward_screen_passed'], False)
                if terminal_status == 'COMPLETE':
                    self.assertIs(report['fitting_comparison_result']['fitting_forward_screen_passed'], False)
                else:
                    self.assertNotIn('fitting_comparison_result', report)

    def test_completed_reference_requires_scientific_agreement_to_report_verified(self):
        for agrees in (False, True):
            with self.subTest(agrees=agrees), tempfile.TemporaryDirectory() as directory:
                campaign = Path(directory)
                (campaign/'reference-diagnostics.json').write_text('[]')
                (campaign/'opening-budget.json').write_text(json.dumps(
                    {'cpu':100, 'gpu':100, 'tranche':{'cpu':100,'gpu':100}}))
                verdict = 'CONDITIONAL_SOLVER_REFERENCE_' + ('AGREES' if agrees else 'DISAGREES')
                record = dict(phase='p0_solver_system_reference', status='COMPLETE', resource='cpu',
                              decision={'status':'PASS','scientific_verdict':verdict})
                records = [(campaign/'manifest.json', record)]
                snapshot = dict(resources={r:dict(remaining_seconds=100) for r in ('cpu','gpu')},
                                running=[])
                with (
                    patch.object(runner, 'CAMPAIGN', campaign),
                    patch.object(runner, 'campaign_attempts', return_value=records),
                    patch.object(runner, 'source_driver_failures', return_value=[]),
                    patch.object(runner.shared, 'snapshot', return_value=snapshot),
                    patch.object(runner.shared, 'charge', return_value=0),
                ):
                    report = runner.status()
                expected = 'VERIFIED' if agrees else 'MISMATCH'
                self.assertEqual(report['active_program_stage'],
                                 'P0_SOLVER_REFERENCE_'+expected+'_PORT_AND_KDM_INCOMPLETE')

    def test_reference_charge_and_result_survive_repeated_reporting(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            (campaign/'opening-budget.json').write_text(json.dumps(
                {'cpu':100, 'gpu':100, 'tranche':{'cpu':100,'gpu':100}}))
            summary = campaign/'summary.json'
            raw = json.dumps(dict(status='PASS', replication='INCOMPLETE',
                                  scientific_verdict='DIAGNOSTIC_ONLY')).encode()
            summary.write_bytes(raw)
            entry = dict(id='reference', resource='cpu', charged_seconds=1.25,
                         summary=str(summary), summary_sha256=hashlib.sha256(raw).hexdigest(),
                         result_file='reference-result.md', plan_file='reference-plan.md',
                         stage='REFERENCE_COMPLETE_PORT_INCOMPLETE', next_action='Continue',
                         after_attempt=0)
            ledger = campaign/'reference-diagnostics.json'
            ledger.write_text(json.dumps([entry]))
            snapshot = dict(resources={r:dict(remaining_seconds=100) for r in ('cpu','gpu')},
                            running=[])
            with (
                patch.object(runner, 'CAMPAIGN', campaign),
                patch.object(runner, 'campaign_attempts', return_value=[]),
                patch.object(runner, 'source_driver_failures', return_value=[]),
                patch.object(runner.shared, 'snapshot', return_value=snapshot),
            ):
                for _ in range(2):
                    report = runner.status()
                    self.assertEqual(report['used_seconds'], {'cpu':1.25, 'gpu':0})
                    self.assertEqual(report['remaining_seconds']['cpu'],98.75)
                    self.assertEqual(report['active_program_stage'],entry['stage'])
                    self.assertEqual(report['result'],entry['result_file'])
                    self.assertEqual(report['csmc_replication'],'INCOMPLETE')
                summary.write_text('{"status":"PASS","scientific_verdict":"MODIFIED"}')
                report = runner.status()
                self.assertEqual(report['active_program_stage'],
                                 'REFERENCE_DIAGNOSTIC_EVIDENCE_REVIEW_REQUIRED')
                self.assertEqual(report['used_seconds']['cpu'],1.25)
                self.assertEqual(report['reference_diagnostics'][0]['decision']['status'],
                                 'EVIDENCE_UNVERIFIED')
                ledger.write_text(json.dumps([entry,entry]))
                with self.assertRaises(ValueError):
                    runner.status()

    def test_reference_does_not_override_a_later_numbered_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign=Path(directory)
            (campaign/'reference-diagnostics.json').write_text('[]')
            (campaign/'opening-budget.json').write_text(json.dumps(
                {'cpu':100, 'gpu':100, 'tranche':{'cpu':100,'gpu':100}}))
            reference=dict(id='reference',resource='cpu',charged_seconds=2,
                           result_file='reference.md',plan_file='reference-plan.md',
                           stage='OLD_REFERENCE',after_attempt=0,
                           summary_verification='VERIFIED_PRESERVED_SUMMARY',
                           decision={'status':'PASS'})
            record=dict(phase='p0_fitting_forward',status='RUNNING',resource='gpu')
            snapshot=dict(resources={r:dict(remaining_seconds=100) for r in ('cpu','gpu')},
                          running=[])
            with (
                patch.object(runner,'CAMPAIGN',campaign),
                patch.object(runner,'campaign_attempts',return_value=[(campaign/'manifest.json',record)]),
                patch.object(runner,'source_driver_failures',return_value=[]),
                patch.object(runner,'reference_diagnostics',return_value=[reference]),
                patch.object(runner.shared,'snapshot',return_value=snapshot),
                patch.object(runner.shared,'charge',return_value=3),
            ):
                report=runner.status()
            self.assertEqual(report['active_program_stage'],'P0_FITTING_FORWARD_RUNNING')
            self.assertEqual(report['used_seconds'],{'cpu':2,'gpu':3})
            self.assertIsNone(report['active_reference'])



if __name__ == '__main__':
    unittest.main()

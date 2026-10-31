"""CPU-only reporting and budget checks; TensorFlow is intentionally not imported."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from dataclasses import dataclass
import unittest

from scripts import run_surrogate_hmc_ot_five_iterations as diagnostic


def arm(value=-100., score=None, valid=True):
    return dict(value=value, score=[2., 0.] if score is None else score, valid=valid)


class DiagnosticTests(unittest.TestCase):
    def test_signed_score_and_near_zero_scaling(self):
        r = diagnostic.comparison(arm(score=[-2., 0.]), arm(score=[-2.01, .009]))
        self.assertTrue(r["passed"])
        self.assertAlmostEqual(r["max_score_scaled_error"], .009)

    def test_one_material_score_vetoes_unchanged_value(self):
        r = diagnostic.comparison(arm(), arm(score=[2., .011]))
        self.assertEqual(r["status"], "MATERIAL_DIFFERENCE")

    def test_nonfinite_and_incomplete_are_invalid(self):
        for value in (float("nan"), "nan"):
            self.assertEqual(diagnostic.comparison(arm(), arm(value=value))["status"], "INVALID_ARM")
        self.assertEqual(diagnostic.comparison(arm(), arm(valid=False))["status"], "INVALID_ARM")

    def test_small_percentage_preserves_absolute_and_centered_change(self):
        rows = []
        for i, delta in enumerate((.2, .9)):
            b, c = arm(), arm(value=-100.+delta)
            rows.append(dict(dataset="saved", position=i, baseline=b, candidate=c,
                             comparison=diagnostic.comparison(b, c)))
        r = diagnostic.summarize(rows)
        self.assertEqual(r["status"], diagnostic.PASS)
        self.assertAlmostEqual(r["max_abs_centered_log_likelihood_delta"], .7)
        self.assertAlmostEqual(r["max_abs_log_likelihood_delta"], .9)

    def test_invalid_row_cannot_be_hidden_by_valid_rows(self):
        r = diagnostic.summarize([dict(comparison=diagnostic.comparison(arm(), arm(valid=False)))])
        self.assertEqual(r["status"], "INVALID_ARM")

    def test_only_requested_controls_change(self):
        @dataclass(frozen=True)
        class Controls:
            reset_steps: int = 80
            reset_epsilon: float = .5
            flow_steps: int = 32
            contraction: str = "direct"
        original = Controls()
        candidate = diagnostic.candidate_controls(original)
        self.assertEqual((candidate.reset_epsilon, candidate.reset_steps), (1., 5))
        self.assertEqual(candidate.flow_steps, original.flow_steps)
        self.assertEqual(candidate.contraction, original.contraction)
        self.assertEqual(original.reset_steps, 80)

    def test_total_budget_and_attempt_limit(self):
        ledger = dict(attempts=[], max_attempts=120)
        self.assertEqual(diagnostic.available_timeout(ledger, 9000, 20), 20)
        self.assertEqual(diagnostic.available_timeout(ledger, 10, 20), 10)
        self.assertEqual(diagnostic.available_timeout(ledger, 9000, 9000), 2400)
        self.assertEqual(diagnostic.available_timeout(dict(attempts=[], max_attempts=0), 9000, 9000), 0)

    def test_active_worker_refuses_overlap(self):
        with self.assertRaises(RuntimeError):
            diagnostic.available_timeout(dict(attempts=[dict(status="RUNNING")], max_attempts=120), 10, 10)


if __name__ == "__main__":
    unittest.main()

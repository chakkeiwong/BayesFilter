"""CPU-only provenance guards; importing this driver must not initialize TF."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "marginal_driver", ROOT / "scripts/run_surrogate_hmc_marginal_weight_campaign.py")
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


class CampaignProvenanceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "output"

    def args(self, **updates):
        values = dict(output=self.output, baseline_source=None, worker=None,
                      only="m13-t1", particles=256, budget_seconds=21600., gpu="test-gpu")
        values.update(updates)
        return argparse.Namespace(**values)

    def test_missing_baseline_fails_before_launch(self):
        with patch.object(driver.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "missing pre-change baseline"):
                driver.master(self.args(only=None))
            launch.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_direct_worker_cannot_fall_back_to_current_package(self):
        with self.assertRaisesRegex(ValueError, "missing pre-change baseline"):
            driver.setup(self.args(worker="regression-before"))

    def test_current_checkout_is_not_a_baseline(self):
        with self.assertRaisesRegex(ValueError, "not the current checkout"):
            driver.run_configuration(self.args(only=None, baseline_source=ROOT))

    def test_snapshot_changes_are_observable(self):
        baseline = Path(self.temp.name) / "baseline"
        for name in ("bayesfilter/__init__.py", "bayesfilter/highdim/ledh_canonical_score_tf.py",
                     "bayesfilter/highdim/ledh_canonical_batch_fused_tf.py"):
            path = baseline / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# saved baseline\n")
        args = self.args(only=None, baseline_source=baseline)
        before = driver.run_configuration(args)
        (baseline / "bayesfilter/__init__.py").write_text("# changed baseline\n")
        after = driver.run_configuration(args)
        self.assertNotEqual(before["baseline_sha256"], after["baseline_sha256"])

    def test_resume_rejects_changed_particle_count(self):
        args = self.args()
        self.output.mkdir()
        (self.output / "campaign.json").write_text(json.dumps(dict(
            source_sha256=driver.source_identity(), configuration=driver.run_configuration(args))))
        with patch.object(driver.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(ValueError, "run configuration changed"):
                driver.master(self.args(particles=4096))
            launch.assert_not_called()

    def test_invalid_phase_and_missing_regression_dependency(self):
        for phases in ("m13-typo", "m13-t1,m13-t1"):
            with self.assertRaisesRegex(ValueError, "distinct supported phase names"):
                driver.run_configuration(self.args(only=phases))
        with self.assertRaisesRegex(ValueError, "preceding saved-baseline regression"):
            driver.run_configuration(self.args(only="regression-after"))
        with self.assertRaisesRegex(ValueError, "direct regression-after"):
            driver.run_configuration(self.args(worker="regression-after", only=None))


if __name__ == "__main__":
    unittest.main()

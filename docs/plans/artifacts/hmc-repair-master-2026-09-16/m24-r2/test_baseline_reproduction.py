"""Before/after engineering reproduction from the preserved pre-repair modules."""
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib.style.core
import arviz as az
import numpy as np
from scipy.signal import lfilter
import tensorflow as tf

from bayesfilter.inference import hmc_bootstrap, hmc_convergence
from tests.test_hmc_kernel_tuning_bootstrap import _ToyGaussianAdapter, _geometry, _config, _fake_result

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent / "m21-r2/source-r1/bayesfilter/inference"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_before_after():
    old = load("baseline_bootstrap", BASELINE / "hmc_bootstrap.py")
    old_convergence = load("baseline_convergence", BASELINE / "hmc_convergence.py")
    config_fields = dict(screen_num_results=16, max_repairs=0, target_scope="kernel_bootstrap_toy_gaussian")
    runner = lambda *args: _fake_result(acceptance=.6846376853336343, binary_rate=.8125, count=16)
    before = old.run_hmc_bootstrap_screen(adapter=_ToyGaussianAdapter(), geometry=_geometry(),
        config=old.HMCBootstrapScreenConfig(**config_fields), run_full_chain=runner)
    after = hmc_bootstrap.run_hmc_bootstrap_screen(adapter=_ToyGaussianAdapter(), geometry=_geometry(),
        config=_config(**config_fields), run_full_chain=runner)
    assert before.rounds[0].classification == "repair"
    assert before.rounds[0].repair_action == "increase_epsilon_recompute_l"
    assert after.rounds[0].classification == "passed"
    rng = np.random.default_rng(5521)
    values = lfilter([1.], [1., .95], rng.normal(size=(4, 769, 2)), axis=1)[:, 512:]
    samples = tf.constant(values.transpose(1, 0, 2), tf.float64)
    reports = []
    for module in (old_convergence, hmc_convergence):
        result = module.rank_normalized_hmc_diagnostics(samples, parameter_names=("a", "b"),
            thresholds=module.RankNormalizedHMCThresholds())
        reports.append([row["bulk_ess"] for row in result["parameter_diagnostics"]])
    reference = [float(az.ess(values[:, :, i], method="bulk")) for i in range(2)]
    np.testing.assert_allclose(reports[1], reference, rtol=2e-11, atol=2e-11)
    assert not np.allclose(reports[0], reference, rtol=2e-11, atol=2e-11)
    record = dict(baseline_package="m21-r2/source-r1, f9c86f41a", scientific_authority=False,
        bootstrap=dict(mean_probability=.6846376853336343, binary_rate=.8125,
            before=before.rounds[0].classification, before_action=before.rounds[0].repair_action,
            after=after.rounds[0].classification),
        public_bulk_ess=dict(before=reports[0], after=reports[1], independent_arviz=reference),
        limitations="Synthetic engineering reproduction, not a MacroFinance replay or posterior qualification")
    output = Path(__import__("os").environ["HMC_REPAIR_TEST_OUTPUT"])
    (output / "baseline-reproduction.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")

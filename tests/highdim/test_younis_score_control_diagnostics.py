"""Real consumer parity and shared-correction wiring; CPU/XLA references only."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _check_consumer(proposal, nonlinear):
    from unittest.mock import patch
    import tensorflow as tf
    from tests.highdim.test_younis_score_master_provider_consumers_tf import fixture
    row, context = fixture(proposal)
    from bayesfilter.highdim import ledh_unified_correction_tf as correction
    from bayesfilter.score_study.adapters import evaluate_gaussian
    from bayesfilter.score_study.nonlinear_adapter import evaluate_nonlinear
    settings = context["study"]["settings"]
    if nonlinear:
        row.update(model="nonlinear_scalar", estimator="nonlinear_analytical", sgqf_level=2)
        settings.update(transition_curve=.2, observation_curve=.12,
            reference=dict(points=401, radius=9., relative_tolerance=1e-7, tail_tolerance=1e-9))
    else:
        settings["dimension"] = 2  # Pairwise fields must be exercised, not scalar placeholders.
    evaluate = evaluate_nonlinear if nonlinear else evaluate_gaussian
    ordinary = evaluate(row, context)
    original = correction.batched_higher_moment_shape_jvp
    seen = set()

    def observed(*args, **kwargs):
        result = dict(original(*args, **kwargs))
        seen.update(result)
        result["wiring_probe"] = tf.reduce_sum(result["target_skew"], axis=-1)
        return result

    with patch.object(correction, "batched_higher_moment_shape_jvp", observed):
        reported = evaluate({**row, "collect_control_diagnostics": True}, context)
    tf.debugging.assert_near(tf.constant(ordinary["value"], tf.float64),
        tf.constant(reported["value"], tf.float64), atol=2e-11, rtol=2e-11)
    tf.debugging.assert_near(tf.constant(ordinary["score"], tf.float64),
        tf.constant(reported["score"], tf.float64), atol=2e-10, rtol=2e-10)
    diagnostics = reported["diagnostics"]["control_diagnostics"]
    assert seen, "consumer did not call the shared correction"
    assert {"higher_moment_" + key for key in seen - {"particles", "particles_tangent"}} <= diagnostics.keys()
    tf.debugging.assert_near(tf.constant(diagnostics["higher_moment_wiring_probe"], tf.float64),
        tf.reduce_sum(tf.constant(diagnostics["higher_moment_target_skew"], tf.float64), axis=-1))
    assert len(diagnostics["particle_ess_per_time"]) == settings["horizon"]
    assert all(0 < ess <= settings["particles"] * (1 + 1e-12)
               for ess in diagnostics["particle_ess_per_time"])
    assert max(diagnostics["transport_row_sum_error_per_time"]) < 1e-10
    assert all(diagnostics["higher_moment_valid"])
    if not nonlinear:
        mask = tf.constant(diagnostics["higher_moment_pairwise_target_mask"])
        assert mask.shape == (settings["horizon"], 2, 2)
        assert bool(tf.reduce_any(mask > 0)), "multidimensional pairwise route was inactive"


@pytest.mark.parametrize("proposal,nonlinear", [("ledh", False), ("ledh", True),
                                                 ("sgqf", True), ("kdm_covariance", True)])
def test_actual_consumer_reports_shared_diagnostics_without_changing_score(proposal, nonlinear):
    code = "import runpy,sys; runpy.run_path(sys.argv[1])['_check_consumer'](sys.argv[2],sys.argv[3]=='1')"
    result = subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve()),
                             proposal, "1" if nonlinear else "0"],
        cwd=Path(__file__).resolve().parents[2], env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1",
            "BAYESFILTER_PRELOAD_CUSTOM_OP": "0"}, capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("bad", ["nonfinite", "nonpositive", "invalid"])
def test_materialization_preserves_invalid_diagnostics_in_failure(bad):
    import tensorflow as tf
    from bayesfilter.score_study.contracts import DiagnosticFailure
    from bayesfilter.score_study.control_diagnostics_tf import materialize_control_diagnostics
    values = {"higher_moment_valid": tf.constant([bad != "invalid"]),
        "predicted_covariances_minimum_eigenvalue_per_time": tf.constant([0. if bad == "nonpositive" else 1.]),
        "particle_ess_per_time": tf.constant([float("nan") if bad == "nonfinite" else 8.])}
    with pytest.raises(DiagnosticFailure, match="control diagnostics invalid"):
        materialize_control_diagnostics(values)

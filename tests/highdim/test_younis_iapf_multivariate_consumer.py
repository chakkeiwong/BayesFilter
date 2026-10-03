"""Multivariate endpoint wiring, preserved admission boundaries and failure guards.

Independent numerical R agreement is preserved by the adaptive-consumer campaign;
these tests prevent the actual public consumer from regressing to scalar-only
execution or bypassing the checked general kernels.
"""
import os
from pathlib import Path
import subprocess
import sys


def _check_public_multivariate_consumer():
    import math
    from unittest.mock import patch
    from bayesfilter.score_study.adapters import evaluate_gaussian
    from bayesfilter.score_study import iapf_adapter, fitted_twist_tf, iapf_fit_tf
    from tests.highdim.test_younis_score_master_provider_consumers_tf import fixture
    for d, o in ((2, 1), (5, 3)):
        row, context = fixture("iapf")
        settings = context["study"]["settings"]
        settings.update(dimension=d, observation_dimension=o, particles=64, horizon=4)
        row["iapf"] = dict(k=2, tau=.5, max_iterations=12, max_particles=512,
            mean_bound=4., sd_lower=.2, sd_upper=4., max_fit_steps=2000,
            max_backtracks=30, fit_tolerance=1e-7, floor_ratio=.01,
            fit_theta=settings["theta"])
        with patch.object(iapf_adapter, "execute_iapf", wraps=iapf_adapter.execute_iapf) as adapter, \
             patch.object(fitted_twist_tf, "make_fitted_twist_kernel", wraps=fitted_twist_tf.make_fitted_twist_kernel) as filter_factory, \
             patch.object(iapf_fit_tf, "make_density_recursive_fit_kernel", wraps=iapf_fit_tf.make_density_recursive_fit_kernel) as fit_factory:
            result = evaluate_gaussian(row, context)
        assert adapter.call_count == 1
        assert filter_factory.call_count >= 5 and fit_factory.call_count >= 3
        assert all(call.args[:2] == (d, o) for call in filter_factory.call_args_list)
        assert all(call.args[:2] == (d, o) for call in fit_factory.call_args_list)
        diagnostics = result["diagnostics"]
        assert result["numerical_validity"] == "pass"
        assert math.isfinite(result["value"]) and all(map(math.isfinite, result["score"]))
        assert diagnostics["fit_iterations"][-1]["action"] == "final"
        assert all(item["density_fit_valid"] and item["density_fit_converged"]
                   for item in diagnostics["fit_iterations"][:-1])
        assert len(diagnostics["fit"]["centers"]) == 4
        assert all(len(center) == d for center in diagnostics["fit"]["centers"])
        streams = diagnostics["fit_seed_records"]
        fitting = {tuple(v) for k, v in streams.items() if not k.startswith("iapf_final")}
        final = {tuple(v) for k, v in streams.items() if k.startswith("iapf_final")}
        assert fitting.isdisjoint(final)
        assert all(count == 1 for count in diagnostics["fit_trace_counts"] + diagnostics["run_trace_counts"])


def _check_rejection_boundaries():
    import pytest
    from bayesfilter.score_study.iapf_adapter import execute_iapf
    settings = dict(dimension=2, observation_dimension=1, particles=64, horizon=4,
                    dtype="float64", jit_compile=True)
    with pytest.raises(ValueError, match="verified study context"):
        execute_iapf(dict(model="gaussian_all_parameters", role="claim"), settings, None, None, None)
    with pytest.raises(ValueError, match="nonlinear conditional means require scalar"):
        execute_iapf(dict(model="nonlinear_scalar", role="mechanics"),
            {**settings, "transition_curve": .1, "observation_curve": .2}, None, None, None)
    for d, o in ((0, 1), (2, 0), (2.5, 1), (True, 1)):
        with pytest.raises(ValueError, match="positive integer"):
            execute_iapf(dict(model="gaussian_all_parameters", role="mechanics"),
                         {**settings, "dimension": d, "observation_dimension": o}, None, None, None)


def _isolated(name):
    result = subprocess.run([sys.executable, "-c",
        "import runpy,sys; runpy.run_path(sys.argv[1])[sys.argv[2]]()", str(Path(__file__).resolve()), name],
        cwd=Path(__file__).resolve().parents[2], env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1",
            "BAYESFILTER_PRELOAD_CUSTOM_OP": "0", "TF_NUM_INTEROP_THREADS": "1", "TF_NUM_INTRAOP_THREADS": "1"},
        capture_output=True, text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr


def test_public_consumer_reaches_general_filter_and_fitter_for_multivariate_models():
    _isolated("_check_public_multivariate_consumer")


def test_multivariate_extension_preserves_claim_nonlinear_and_dimension_boundaries():
    _isolated("_check_rejection_boundaries")

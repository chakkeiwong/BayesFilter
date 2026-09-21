"""Windowed preparation ownership, compatibility and live seed provenance."""
import hashlib
import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
import typing

import pytest

import bayesfilter.inference as public
from bayesfilter.inference import hmc_kernel_tuning as historical
from bayesfilter.inference import hmc_mass_adaptation as windowed


def test_public_and_historical_windowed_imports_share_the_implementation():
    for name in ("HMCStagedTimeoutPolicy", "HMCWindowedMassStageConfig",
                 "HMCWindowedMassStageResult", "WINDOWED_MASS_STAGE_NONCLAIMS",
                 "run_hmc_windowed_mass_stage", "build_operational_fixed_mass_hmc_adapter"):
        assert getattr(public, name) is getattr(windowed, name)
        assert getattr(historical, name) is getattr(windowed, name)
    from bayesfilter.inference.hmc_budget_policy import HMCStagedTimeoutPolicy
    assert HMCStagedTimeoutPolicy is windowed.HMCStagedTimeoutPolicy
    assert typing.get_type_hints(windowed.HMCWindowedMassStageResult)["config"] is windowed.HMCWindowedMassStageConfig


@pytest.mark.parametrize("name", ["HMCStagedTimeoutPolicy", "HMCWindowedMassStageConfig",
                                 "HMCWindowedMassStageResult"])
def test_historical_pickle_globals_resolve_to_the_extracted_windowed_types(name):
    old_global = f"cbayesfilter.inference.hmc_kernel_tuning\n{name}\n.".encode("ascii")
    assert pickle.loads(old_global) is getattr(windowed, name)


def test_windowed_public_imports_do_not_load_the_historical_tuner():
    code = """
import sys
from bayesfilter.inference import (
    HMCStagedTimeoutPolicy, HMCWindowedMassStageConfig, HMCWindowedMassStageResult,
    run_hmc_windowed_mass_stage, build_operational_fixed_mass_hmc_adapter,
)
for definition in (HMCStagedTimeoutPolicy, HMCWindowedMassStageConfig,
                   HMCWindowedMassStageResult, run_hmc_windowed_mass_stage,
                   build_operational_fixed_mass_hmc_adapter):
    assert definition.__module__ == 'bayesfilter.inference.hmc_mass_adaptation'
assert 'bayesfilter.inference.hmc_kernel_tuning' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[1],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))


def test_numerical_source_closure_includes_the_windowed_implementation():
    from bayesfilter.inference.hmc_candidate_set_execution import _source_closure
    from tests.test_hmc_kernel_tuning_geometry import Adapter

    path = Path(inspect.getsourcefile(windowed.run_hmc_windowed_mass_stage)).resolve()
    closure = _source_closure(Adapter(), [__file__])
    assert closure[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_live_seed_source_coverage_matches_the_actual_seed_operations():
    from tests.test_hmc_kernel_tuning_windowed_mass import _g1a_manifest_sites_from_final_sources

    coverage = historical._public_p4_seed_source_coverage_payload()
    sites = {row["site_id"]: row for row in _g1a_manifest_sites_from_final_sources()}
    assert set(sites) == set(coverage["source_site_contracts"])
    for site_id, row in coverage["source_site_contracts"].items():
        assert Path(row["source_path"]).resolve() == Path(sites[site_id]["source_path"])
    path = Path(windowed.__file__).resolve()
    files = {row["path"]: row["sha256"] for row in coverage["source_files"]}
    assert files["bayesfilter/inference/hmc_mass_adaptation.py"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_extracted_p4_stage_consumes_its_seed_in_the_live_registry(monkeypatch):
    from bayesfilter.hmc_route_contract import OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
    from tests.test_hmc_kernel_tuning_windowed_mass import _stage_config

    registry = historical._build_public_p4_seed_use_registry()
    config = _stage_config(algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
                          engineering_probe_covariance_multiplier=2.)
    captured = []

    def capture(**kwargs):
        captured.append(kwargs["stage_seed"])
        assert kwargs["g2_seed_use_registry"] is registry
        return None, None, {"diagnostic_fixture": True}, None, None

    monkeypatch.setattr(windowed, "_operational_windowed_mass_capture", capture)
    result = windowed._run_p4_windowed_boundary_attempt(
        adapter=object(), geometry=SimpleNamespace(), hmc_adapter_signature="test",
        stage_mass_artifact=SimpleNamespace(),
        mass_window_seed_kernel={"step_size": .1, "num_leapfrog_steps": 3},
        windowed_config=windowed._windowed_mass_stage_internal_config(None, mass_policy=config.mass_policy),
        config=config, target_scope=config.target_scope, attempt_state=None,
        route_decision=SimpleNamespace(), progress_callback=None, attempt_index=0,
        registry=registry,
    )
    assert result[7:9] == (None, None)
    assert captured == [historical._derive_seed(config.seed, stage_index=0)]
    entries = registry._ordered_entries()
    assert len(entries) == 1
    assert entries[0]["key"] == "phase4/stage"
    assert entries[0]["owner_file"] == "hmc_mass_adaptation.py"


@pytest.mark.parametrize("stop_at", ["windowed", "handoff"])
def test_automatic_preparation_calls_the_extracted_stage_and_handoff(monkeypatch, stop_at):
    from bayesfilter.inference import hmc_bootstrap, hmc_geometry
    from bayesfilter.inference.hmc_preparation import prepare_operational_windowed_mass_handoff
    from tests.test_hmc_kernel_tuning_windowed_mass import _ToyGaussianAdapter, _geometry, _bootstrap

    geometry, bootstrap = _geometry(), _bootstrap()
    monkeypatch.setattr(hmc_geometry, "initialize_hmc_kernel_geometry", lambda **kwargs: geometry)
    monkeypatch.setattr(hmc_bootstrap, "run_hmc_bootstrap_screen", lambda **kwargs: bootstrap)
    stage_result = SimpleNamespace(passed=True, final_status="passed",
        operational_warmup_result=SimpleNamespace(operational_metric_update_count=1,
                                                 metric_adaptation_status="metric_updated"))
    seen = []

    def stage(**kwargs):
        assert kwargs["geometry"] is geometry
        assert kwargs["bootstrap"] is bootstrap
        seen.append("windowed")
        if stop_at == "windowed":
            raise RuntimeError("extracted windowed reached")
        return stage_result

    def handoff(**kwargs):
        assert kwargs["windowed_stage"] is stage_result
        seen.append("handoff")
        raise RuntimeError("extracted handoff reached")

    monkeypatch.setattr(windowed, "run_hmc_windowed_mass_stage", stage)
    monkeypatch.setattr(windowed, "build_operational_fixed_mass_hmc_adapter", handoff)
    with pytest.raises(RuntimeError, match=f"extracted {stop_at} reached"):
        prepare_operational_windowed_mass_handoff(
            adapter=_ToyGaussianAdapter(), initial_position=[0., 0.],
            config=historical.HMCKernelTuningConfig.smoke(
                target_scope="kernel_windowed_mass_toy_gaussian", use_xla=False),
        )
    assert seen == (["windowed"] if stop_at == "windowed" else ["windowed", "handoff"])

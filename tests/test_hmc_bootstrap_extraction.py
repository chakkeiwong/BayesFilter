"""Bootstrap ownership, public call-chain and physical seed/source coverage."""
import hashlib
import inspect
import os
from pathlib import Path
import pickle
import subprocess
import sys
import typing

import pytest

import bayesfilter.inference as public
from bayesfilter.inference import hmc_bootstrap as bootstrap
from bayesfilter.inference import hmc_kernel_tuning as historical


def test_public_and_historical_bootstrap_imports_share_the_implementation():
    for name in ("BOOTSTRAP_SCREEN_NONCLAIMS", "HMCBootstrapScreenConfig",
                 "HMCBootstrapRepairRound", "HMCBootstrapScreenResult", "run_hmc_bootstrap_screen"):
        assert getattr(public, name) is getattr(bootstrap, name)
        assert getattr(historical, name) is getattr(bootstrap, name)
    assert public.build_bootstrap_fixed_mass_adapter is historical._build_bootstrap_fixed_mass_adapter
    assert public.BootstrapFixedMassAdapter is historical._BootstrapFixedMassLatentValueScoreAdapter
    assert typing.get_type_hints(bootstrap.run_hmc_bootstrap_screen)["return"] is bootstrap.HMCBootstrapScreenResult


@pytest.mark.parametrize("name", ["HMCBootstrapScreenConfig", "HMCBootstrapRepairRound",
                                 "HMCBootstrapScreenResult", "_BootstrapFixedMassLatentValueScoreAdapter"])
def test_historical_pickle_globals_resolve_to_the_same_bootstrap_types(name):
    old_global = f"cbayesfilter.inference.hmc_kernel_tuning\n{name}\n.".encode("ascii")
    assert pickle.loads(old_global) is getattr(bootstrap, name)


def test_bootstrap_can_load_without_importing_the_historical_tuner():
    code = """
import sys
from bayesfilter.inference import HMCBootstrapScreenConfig, run_hmc_bootstrap_screen
assert HMCBootstrapScreenConfig.__module__ == 'bayesfilter.inference.hmc_bootstrap'
assert run_hmc_bootstrap_screen.__module__ == 'bayesfilter.inference.hmc_bootstrap'
assert 'bayesfilter.inference.hmc_kernel_tuning' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[1],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))


def test_source_closure_includes_both_extracted_modules():
    from bayesfilter.inference.hmc_candidate_set_execution import _source_closure
    from bayesfilter.inference import hmc_preparation_common
    from tests.test_hmc_kernel_tuning_geometry import Adapter

    closure = _source_closure(Adapter(), [__file__])
    for module in (bootstrap, hmc_preparation_common):
        path = str(Path(module.__file__).resolve())
        assert closure[path] == hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_live_seed_coverage_follows_the_extracted_implementation():
    from tests.test_hmc_kernel_tuning_bootstrap import _ToyGaussianAdapter, _geometry, _config, _scripted_runner

    coverage = historical._public_p4_seed_source_coverage_payload()
    implementation = Path(inspect.getsourcefile(bootstrap.run_hmc_bootstrap_screen)).resolve()
    for site_id, row in coverage["source_site_contracts"].items():
        if site_id.startswith(("hmc_kernel_tuning.run_hmc_bootstrap_screen.",
                               "hmc_kernel_tuning._bootstrap_screen_config.")):
            assert Path(row["source_path"]).resolve() == implementation
    files = {row["path"]: row["sha256"] for row in coverage["source_files"]}
    assert files["bayesfilter/inference/hmc_bootstrap.py"] == hashlib.sha256(implementation.read_bytes()).hexdigest()
    registry = historical._build_public_p4_seed_use_registry()
    runner, calls = _scripted_runner([.95, .70])
    result = bootstrap.run_hmc_bootstrap_screen(
        adapter=_ToyGaussianAdapter(), geometry=_geometry(), config=_config(),
        run_full_chain=runner, _g2_seed_use_registry=registry,
    )
    assert result.passed
    assert [call[2] for call in calls] == [(20260621, 30), (20260621, 31)]


def test_automatic_preparation_calls_the_extracted_bootstrap(monkeypatch):
    from bayesfilter.inference.hmc_preparation import prepare_operational_windowed_mass_handoff
    from tests.test_hmc_kernel_tuning_bootstrap import _ToyGaussianAdapter

    def reached(**kwargs):
        assert kwargs["geometry"].target_dimension == 2
        raise RuntimeError("extracted bootstrap reached")

    monkeypatch.setattr(bootstrap, "run_hmc_bootstrap_screen", reached)
    with pytest.raises(RuntimeError, match="extracted bootstrap reached"):
        prepare_operational_windowed_mass_handoff(
            adapter=_ToyGaussianAdapter(), initial_position=[0., 0.],
            config=historical.HMCKernelTuningConfig.smoke(
                target_scope="kernel_bootstrap_toy_gaussian", use_xla=False),
        )

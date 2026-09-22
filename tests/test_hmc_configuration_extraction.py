"""Preparation configuration ownership and compatibility; no policy changes."""
import hashlib
import os
from pathlib import Path
import pickle
import subprocess
import sys
import typing

import pytest

import bayesfilter.inference as public
from bayesfilter.inference import hmc_configuration as configuration
from bayesfilter.inference import hmc_kernel_tuning as historical


def test_public_and_historical_configuration_share_definitions():
    for name in ('HMCKernelTuningConfig', 'HMCTuneVerifyRepairLoopConfig',
                 'HMCGeometryScaledBudgetTimingPolicy', 'resolve_ordinary_hmc_selection_policy',
                 'ORDINARY_SHARED_EPSILON_SCREEN_POLICY_ID',
                 'ORDINARY_LEGACY_JOINT_L_EPSILON_POLICY_ID',
                 'ORDINARY_ENGINEERING_JOINT_L_EPSILON_POLICY_ID'):
        assert getattr(public, name) is getattr(configuration, name)
        assert getattr(historical, name) is getattr(configuration, name)
    from bayesfilter.inference.hmc_budget_policy import HMCGeometryScaledBudgetTimingPolicy
    assert HMCGeometryScaledBudgetTimingPolicy is configuration.HMCGeometryScaledBudgetTimingPolicy
    assert typing.get_type_hints(configuration._public_budget_policy_factory)['config'] is configuration.HMCKernelTuningConfig


@pytest.mark.parametrize('name', ['HMCKernelTuningConfig', 'HMCTuneVerifyRepairLoopConfig',
                                 'HMCGeometryScaledBudgetTimingPolicy', '_HMCAttemptBudgetPolicy'])
def test_historical_pickle_globals_keep_the_same_configuration_types(name):
    old_global = f'cbayesfilter.inference.hmc_kernel_tuning\n{name}\n.'.encode('ascii')
    assert pickle.loads(old_global) is getattr(configuration, name)


def test_public_configuration_and_budget_load_without_historical_executor():
    code = """
import sys
from bayesfilter.inference import HMCKernelTuningConfig, HMCGeometryScaledBudgetTimingPolicy
from bayesfilter.inference.hmc_configuration import (
    _public_budget_policy_factory, _public_loop_config, _phase7_windowed_stage_config,
)
from bayesfilter.inference.hmc_budget_policy import HMCStagedTimeoutPolicy
from bayesfilter.inference import ORDINARY_SHARED_EPSILON_SCREEN_POLICY_ID
for preset in ('smoke', 'standard', 'diagnostic', 'diagnostic_plus', 'serious'):
    config = getattr(HMCKernelTuningConfig, preset)()
    assert config.__class__.__module__ == 'bayesfilter.inference.hmc_configuration'
    assert _public_budget_policy_factory(config)(9, 0).phase4_warmup_steps > 0
    stage = _phase7_windowed_stage_config(_public_loop_config(config), attempt_index=0)
    assert stage.mass_policy == config.mass_policy
assert 'bayesfilter.inference.hmc_kernel_tuning' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[1],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))


def test_automatic_preparation_imports_and_runs_without_historical_executor():
    code = """
import sys
from bayesfilter.inference import HMCKernelTuningConfig
from bayesfilter.inference import hmc_geometry, hmc_bootstrap, hmc_mass_adaptation
from bayesfilter.inference import prepare_operational_windowed_mass_handoff
from types import SimpleNamespace
geometry = SimpleNamespace(artifact_hash='geometry', target_dimension=2)
bootstrap = SimpleNamespace(artifact_hash='bootstrap', final_status='passed',
                            rounds=(), selected_kernel_payload=None)
hmc_geometry.initialize_hmc_kernel_geometry = lambda **kwargs: geometry
hmc_bootstrap.run_hmc_bootstrap_screen = lambda **kwargs: bootstrap
seen = []
def stage(**kwargs):
    assert kwargs['_attempt_budget_policy'].target_dimension == 2
    assert kwargs['geometry'] is geometry
    seen.append('windowed')
    raise RuntimeError('stage reached')
hmc_mass_adaptation.run_hmc_windowed_mass_stage = stage
try:
    prepare_operational_windowed_mass_handoff(
        adapter=object(), initial_position=[0., 0.],
        config=HMCKernelTuningConfig.smoke(target_scope='config-wiring'))
except RuntimeError as exc:
    assert str(exc) == 'stage reached', str(exc)
else:
    raise AssertionError('stage was not reached')
assert seen == ['windowed']
assert 'bayesfilter.inference.hmc_kernel_tuning' not in sys.modules
"""
    subprocess.run([sys.executable, '-c', code], check=True,
                   cwd=Path(__file__).resolve().parents[1],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='-1'))


def test_source_closure_includes_the_actual_configuration_owner():
    from bayesfilter.inference.hmc_candidate_set_execution import _source_closure
    from tests.test_hmc_kernel_tuning_geometry import Adapter
    path = Path(configuration.__file__).resolve()
    assert _source_closure(Adapter(), [__file__])[str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()

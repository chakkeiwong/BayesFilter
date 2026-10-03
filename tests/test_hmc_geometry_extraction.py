"""Public compatibility and dependency boundaries for initial geometry."""
import os
from pathlib import Path
import pickle
import subprocess
import sys

import bayesfilter.inference as public
from bayesfilter.inference import hmc_geometry as geometry
from bayesfilter.inference import hmc_kernel_tuning as historical


def test_public_and_historical_geometry_imports_share_the_implementation():
    for name in ("HMCGeometryInitializationConfig", "HMCGeometryInitializationResult",
                 "initialize_hmc_kernel_geometry", "GEOMETRY_INITIALIZATION_NONCLAIMS"):
        assert getattr(public, name) is getattr(geometry, name)
        assert getattr(historical, name) is getattr(geometry, name)


def test_historical_pickle_global_resolves_to_the_same_geometry_class():
    # Protocol-0 GLOBAL records encode the module/name used by older pickles.
    old_global = b"cbayesfilter.inference.hmc_kernel_tuning\nHMCGeometryInitializationConfig\n."
    assert pickle.loads(old_global) is geometry.HMCGeometryInitializationConfig
    config = geometry.HMCGeometryInitializationConfig(seed=(17, 19))
    assert pickle.loads(pickle.dumps(config)).payload() == config.payload()


def test_initial_geometry_can_load_without_importing_the_historical_tuner():
    code = """
import sys
from bayesfilter.inference import HMCGeometryInitializationConfig, initialize_hmc_kernel_geometry
assert HMCGeometryInitializationConfig.__module__ == 'bayesfilter.inference.hmc_geometry'
assert initialize_hmc_kernel_geometry.__module__ == 'bayesfilter.inference.hmc_geometry'
assert 'bayesfilter.inference.hmc_kernel_tuning' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True,
                   cwd=Path(__file__).resolve().parents[1],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))


def test_numerical_source_closure_includes_the_extracted_implementation():
    from bayesfilter.inference.hmc_candidate_set_execution import _source_closure
    from tests.test_hmc_kernel_tuning_geometry import Adapter
    closure = _source_closure(Adapter(), [__file__])
    assert str(Path(geometry.__file__).resolve()) in closure

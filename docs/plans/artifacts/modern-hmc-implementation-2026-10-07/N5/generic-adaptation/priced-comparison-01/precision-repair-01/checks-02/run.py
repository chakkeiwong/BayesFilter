import sys, importlib.util
from pathlib import Path
sys.path.insert(0, '/home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/precision-repair-01/source')
spec=importlib.util.spec_from_file_location("scripts.calibrate_hmc_stopped_mcse", '/home/ubuntu/python/BayesFilter/scripts/calibrate_hmc_stopped_mcse.py')
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)
import bayesfilter.inference.hmc_precision as precision
assert Path(precision.__file__).resolve()==Path('/home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/precision-repair-01/source/bayesfilter/inference/hmc_precision.py')
import pytest
raise SystemExit(pytest.main(['--noconftest', '--import-mode=importlib', '-q', 'tests/test_hmc_precision_batch_maximum.py', 'tests/test_hmc_precision_covariance_reference.py', 'tests/test_hmc_stopped_mcse_calibration.py', 'tests/test_hmc_quantile_precision_reference.py', '--junitxml=/home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/priced-comparison-01/precision-repair-01/checks-02/tests.xml']))

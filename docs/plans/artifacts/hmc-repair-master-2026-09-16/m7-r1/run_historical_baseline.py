import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"baseline-package"))
import bayesfilter
import pytest
print("baseline package:",bayesfilter.__file__)
raise SystemExit(pytest.main(['-q', '--disable-warnings', 'tests/test_hmc_kernel_tuning_windowed_mass.py::test_windowed_mass_stage_private_progress_callback_is_allowlisted', 'tests/test_hmc_kernel_tuning_windowed_mass.py::test_real_operational_route_with_generous_timeout_never_uses_legacy', 'tests/test_hmc_kernel_tuning_windowed_mass.py::test_g1a_source_coverage_manifest_binds_every_preboundary_seed_site', 'tests/test_hmc_kernel_tuning_windowed_mass.py::test_p4_multiplier_is_explicit_private_and_propagates_without_hmc', 'tests/test_hmc_kernel_tuning_windowed_mass.py::test_g1a_p4_compatibility_failure_is_static_and_exception_redacted', 'tests/test_hmc_kernel_tuning_windowed_mass.py::test_phase7_initial_state_preserves_ordinary_legacy_bank_without_hmc', 'tests/test_hmc_kernel_tuning_windowed_mass.py::test_phase7_initial_state_rejects_legacy_bank_when_p4_is_configured', '--junitxml=docs/plans/artifacts/hmc-repair-master-2026-09-16/m7-r1/historical-baseline-tests.xml']))

"""Fresh target evidence for the isolated initializer adapter, never old admission."""

from pathlib import Path

import pytest

from tests import test_filter_repair_dz5_merged as merged
from tests import test_filter_repair_dz5_score_oracle as oracle
from tests import test_filter_repair_dz5_snapshot as snapshot_test

SNAPSHOT = Path('/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/'
    'filter-gradient-repair-20260917/dz5-initializer-adapter-20260928-r1')


def audited_child(child):
    return child.replace("    report['passed'] = True", """
    forbidden_modules = [name for name in sys.modules if any(
        name == prefix or name.startswith(prefix + '.') for prefix in
        ('filters', 'inference.hmc', 'inference.mass_matrix', 'inference.posterior_adapter'))]
    report['forbidden_local_runtime_imports'] = forbidden_modules
    assert not forbidden_modules, forbidden_modules
    report['passed'] = True""")


@pytest.mark.parametrize('batch', [1, 4, 46, 68])
def test_current_initializer_target(request, monkeypatch, batch):
    monkeypatch.setenv('FILTER_REPAIR_DZ5_BATCH', str(batch))
    snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=audited_child(merged.target_child()), child_timeout_seconds=840,
        scope='fresh_target_comparison_for_initializer_only_no_HMC_admission')


def test_current_initializer_score_oracle(request, monkeypatch):
    monkeypatch.setenv('FILTER_REPAIR_DZ5_SCORE_JIT', '1')
    loaded_audit = merged.TARGET_CHECK[merged.TARGET_CHECK.index('    # Re-audit all actual project imports'):]
    child = merged.target_child().replace(merged.TARGET_CHECK, oracle.ORACLE_CHECK + loaded_audit)
    snapshot_test.run_isolated_snapshot(request, snapshot=SNAPSHOT,
        child=audited_child(child), child_timeout_seconds=840,
        scope='fresh_independent_score_oracle_for_initializer_only_no_HMC_admission')

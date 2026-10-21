"""Diagnostic harness guards against using the wrong saved point."""
import importlib.util
from pathlib import Path

path = Path(__file__).resolve().parents[1] / 'scripts/run_sir_fresh_reference.py'
spec = importlib.util.spec_from_file_location('sir_fresh_reference', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_saved_point_is_physical_theta_not_chart_position():
    source = module.read(module.SOURCE)
    data, row, theta = module.selected_input(source)
    assert data['data_seed'] == 26101401
    assert row['position'] == 2
    assert theta == [-0.017499999999999998]*3
    assert row['baseline']['score'][1] == -1.3257831440132581


def test_one_rung_cannot_establish_particle_stability():
    result = module.assess([{'log_likelihood':0.,'score':[1.],
                            'jackknife_mcse_log_likelihood':0.,'jackknife_mcse_score':[0.]}],{})
    assert not result['uncertainty_resolved']
    assert result['status'] == 'PARTICLE_LADDER_PENDING'

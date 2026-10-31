"""The optimistic cost screen may defer an arm, but never admits inference."""
import pytest

from scripts.calibrate_surrogate_hmc_nonlinear_controls import optimistic_forecast


def measurements():
    return [dict(compilation_included=i == 0,wall_seconds=10. if i == 0 else 1.,
                 cpu_worker_seconds=5. if i == 0 else .5) for i in range(5)]


def test_forecast_preserves_per_arm_and_cpu_limit():
    result = optimistic_forecast(measurements(),dict(gpu_seconds=10000.,cpu_worker_seconds=1000.))
    assert result['status'] == 'UNDER_BUDGETED_INITIAL_ALLOCATION'
    assert result['optimistic_cost'] == dict(gpu_seconds=4890.,cpu_worker_seconds=2445.)
    assert result['authorizes_inference'] is False
    result = optimistic_forecast(measurements(),dict(gpu_seconds=10000.,cpu_worker_seconds=10000.))
    assert result['status'] == 'REQUIRES_COMPLETE_FORECAST'
    assert result['authorizes_inference'] is False


@pytest.mark.parametrize('bad', [0.,-1.,float('nan'),float('inf')])
def test_invalid_timing_cannot_nominate_a_launch(bad):
    records = measurements()
    records[2]['cpu_worker_seconds'] = bad
    with pytest.raises(ValueError,match='positive measured'):
        optimistic_forecast(records,dict(gpu_seconds=10000.,cpu_worker_seconds=10000.))

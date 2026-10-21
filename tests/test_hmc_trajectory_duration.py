"""Pure duration-law checks independent of any fitted state-space model."""
import math

import pytest

from bayesfilter.inference.hmc_trajectory_duration import compile_mean_duration
from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile


@pytest.mark.parametrize("family", ["hmc", "mams"])
@pytest.mark.parametrize("trajectory", ["fixed", "uniform_duration"])
@pytest.mark.parametrize("count", [1, 2, 3, 9, 18, 101])
def test_exact_executed_duration_moments_and_compilation(family, trajectory, count):
    profile = HMCTransitionProfile(family, trajectory)
    epsilon = 0.37
    mean = epsilon * ((count + 1) / 2 if trajectory == "uniform_duration" else count)
    item = compile_mean_duration(profile=profile, epsilon=epsilon,
                                 requested_mean_duration=mean,
                                 max_steps_cap=101)
    assert item.frozen.max_steps == count
    assert item.realized_mean_duration == pytest.approx(mean)
    explicit_second = (epsilon**2 * sum(n*n for n in range(1, count+1)) / count
                       if trajectory == "uniform_duration" else mean**2)
    assert item.realized_second_moment == pytest.approx(explicit_second)
    assert item.frozen.payload()["rounding"] == (
        "max_1_ceil_uniform_times_max_steps" if trajectory == "uniform_duration"
        else "fixed_count")
    assert item.payload()["tuning_artifact_authority"] is False


@pytest.mark.parametrize("trajectory,error_factor", [("fixed", .5),
                                                       ("uniform_duration", .25)])
def test_nearest_count_error_bound_and_ties_up(trajectory, error_factor):
    profile = HMCTransitionProfile("mams", trajectory)
    for epsilon in (.001, .2, 1.7):
        for target in (epsilon, 1.125*epsilon, 5.5*epsilon, 31.273*epsilon):
            item = compile_mean_duration(profile=profile, epsilon=epsilon,
                                         requested_mean_duration=target,
                                         max_steps_cap=1000)
            assert abs(item.quantization_error) <= error_factor*epsilon+1e-12
    tie = compile_mean_duration(profile=profile, epsilon=1.,
                                requested_mean_duration=(2.25 if trajectory == "uniform_duration"
                                                         else 2.5), max_steps_cap=10)
    assert tie.frozen.max_steps == (4 if trajectory == "uniform_duration" else 3)


@pytest.mark.parametrize("kwargs,reason", [
    ({"epsilon": 0.5, "requested_mean_duration": 0.2}, "one-step minimum"),
    ({"epsilon": 0.5, "requested_mean_duration": 10.}, "step-count cap"),
    ({"epsilon": math.inf, "requested_mean_duration": 1.}, "epsilon"),
    ({"epsilon": 0.5, "requested_mean_duration": math.nan}, "requested mean"),
])
def test_infeasible_request_never_silently_clips(kwargs, reason):
    with pytest.raises(ValueError, match=reason):
        compile_mean_duration(profile=HMCTransitionProfile("mams", "uniform_duration"),
                              max_steps_cap=8, **kwargs)

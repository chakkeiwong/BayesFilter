"""Finite-support centering and consumer wiring for diagnostic controls."""
import itertools
import json
import time

import pytest
import tensorflow as tf

from bayesfilter.score_study.innovation_controls_tf import make_innovation_control_kernel


def test_continuation_streams_exclude_previous_finals_and_each_other(tmp_path, monkeypatch):
    from docs.benchmarks import diagnose_younis_iapf_innovation_control as driver
    monkeypatch.setattr(driver, "CONTINUATION_ROOT", tmp_path)
    first, checked = driver.seed_inventory("pinned", "pinned01")
    assert checked["old_seed_pairs"] == 22720
    assert checked["new_seed_pairs"] == 4800
    run = tmp_path / "pinned01"
    run.mkdir()
    (run / "seeds.json").write_text(json.dumps(first))
    fresh, checked = driver.seed_inventory("fresh", "fresh01")
    assert checked["old_seed_pairs"] == 27520
    assert checked["new_seed_pairs"] == 3840
    first_pairs = {tuple(pair) for pairs in first.values() for pair in pairs}
    assert not first_pairs.intersection(tuple(pair) for pairs in fresh.values() for pair in pairs)
    with pytest.raises(RuntimeError, match="seed collision"):
        driver.seed_inventory("pinned", "pinned01")


def test_continuation_retry_accounts_for_consumed_work_and_ceiling(tmp_path, monkeypatch):
    from docs.benchmarks import diagnose_younis_iapf_innovation_control as driver
    monkeypatch.setattr(driver, "CONTINUATION_ROOT", tmp_path)
    run = tmp_path / "failed01"
    run.mkdir()
    counts = dict(filter_calls=7000, adaptive_fits=0, fixed_cloud_fits=0)
    record = dict(status="stopped", continuation_stage="pinned", wall_seconds=12.,
                  budget=dict(counts=counts), prior_budget=dict.fromkeys(counts, 0))
    (run / "manifest.json").write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="insufficient remaining"):
        driver.continuation_budget(tmp_path / "retry02", "pinned", time.monotonic())
    record["budget"]["counts"]["filter_calls"] = 10
    (run / "manifest.json").write_text(json.dumps(record))
    budget, seconds, launch, expected = driver.continuation_budget(tmp_path / "retry02", "pinned", time.monotonic())
    assert (budget.counts["filter_calls"], seconds, launch, expected) == (10, 12., 2, 1116)
    with pytest.raises(RuntimeError, match="pinned stage must complete"):
        driver.continuation_budget(tmp_path / "fresh02", "fresh", time.monotonic())


@pytest.mark.parametrize("copies", [1, 2])
def test_centering_preserves_expectation_for_a_nonstandard_discrete_noise_law(copies):
    # This law has E[Z]=1/2 and E[Z^2]=5/2. Ideal-normal centering is wrong.
    kernel = make_innovation_control_kernel(1, 1, 1, copies, "float64")
    controls, raw, corrected = [], [], []
    for values in itertools.product((-1., 2.), repeat=copies+1):
        z, *reference = values
        control = kernel(tf.constant([[[z]]], tf.float64),
                         tf.constant([[[[r]]] for r in reference], tf.float64))
        statistic = z**3 + 2*z
        controls.append(control)
        raw.append(statistic)
        corrected.append(statistic-tf.reduce_sum(control*tf.constant([.7, -.3], tf.float64)))
    tf.debugging.assert_near(tf.reduce_mean(controls, 0), tf.zeros([2], tf.float64), atol=1e-14)
    tf.debugging.assert_near(tf.reduce_mean(corrected), tf.reduce_mean(tf.constant(raw, tf.float64)), atol=1e-14)
    assert kernel.experimental_get_tracing_count() == 1


def test_controls_handle_multiple_times_coordinates_and_reference_batches():
    kernel = make_innovation_control_kernel(4, 3, 2, 2, "float32")
    z = tf.reshape(tf.range(24, dtype=tf.float32)-10, [3, 4, 2])
    # Each reference batch permutes the same particles, so both moments match.
    references = tf.stack([tf.reverse(z, [1]), tf.roll(z, 1, axis=1)])
    tf.debugging.assert_equal(kernel(z, references), tf.zeros([12], tf.float64))


def test_diagnostic_observer_uses_exact_noise_inputs_of_the_single_filter_call():
    from docs.benchmarks.diagnose_younis_iapf_innovation_control import observe
    initial = tf.constant([[.2], [.3]], tf.float32)
    process = tf.constant([[[.4], [.5]], [[.6], [.7]]], tf.float32)
    draws = (initial, process, "ancestors", "mixture")
    references = object()
    calls = []

    def filter_kernel(*args):
        calls.append(args)
        assert args[2] is initial and args[3] is process
        return ("value", "fixed_score", "cloud", "fisher", "ancestor_controls")

    def control_kernel(noise, supplied_reference):
        assert supplied_reference is references
        tf.debugging.assert_equal(noise, tf.concat([initial[None], process], 0))
        return "innovation_controls"

    outputs, controls = observe(filter_kernel, control_kernel, "theta", "obs", draws, ("fit",), references)
    assert len(calls) == 1 and outputs[3] == "fisher" and controls == "innovation_controls"


def test_rejects_unspecified_precision_or_invalid_shapes():
    with pytest.raises(ValueError):
        make_innovation_control_kernel(4, 3, 1, 0)
    with pytest.raises(ValueError):
        make_innovation_control_kernel(4, 3, 1, dtype_name="float16")


def test_launch_device_is_the_recorded_physical_gpu_not_an_ordinal():
    from docs.benchmarks.diagnose_younis_iapf_innovation_control import reference_gpu_uuid
    identifier = "GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a"
    assert reference_gpu_uuid({"cuda_visible_devices": identifier}) == identifier
    for value in (None, "0", "-1", "0,1", identifier+",GPU-second"):
        with pytest.raises(ValueError):
            reference_gpu_uuid({"cuda_visible_devices": value})

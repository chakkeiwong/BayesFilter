"""Independent diagnostic references for the registered LEDH seeded endpoint.

NumPy and the frozen eager recurrence are test authorities only. No canonical
method, posterior or HMC admission follows from these execution checks.
"""

import gc
import hashlib
import importlib
import json
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_alg1_contract import ENTRY_POINTS
from bayesfilter.highdim.ledh_canonical_filter_tf import (
    canonical_value_and_diagnostics,
    make_seeded_canonical_value_program,
)
from bayesfilter.highdim.ledh_canonical_value_program_tf import (
    make_canonical_value_program,
)
from bayesfilter.ops.ledh_random_compat_tf import (
    integer_seed_words,
    pcg64_next,
    pcg64_offset_initial_state,
    seeded_value_inputs,
)
from tests.test_filter_repair_ledh_value_native import (
    _compare,
    _frozen_reference,
)
from tests.test_filter_repair_ledh_value_native import (
    authorities as _native_authorities,
)


@pytest.fixture(scope="module")
def authorities():
    yield from _native_authorities.__wrapped__()


def _save(request, name, report):
    directory = Path(request.config.getoption("xmlpath")).parent
    (directory / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")


def _reference_generator(seed):
    material = np.random.SeedSequence(seed).generate_state(2, dtype=np.uint32)
    return tf.random.Generator.from_seed((int(material[0]) << 31) ^ int(material[1]))


def _reference_inputs(seed, resample_seed, dtype, horizon=3, stages=3):
    generator = _reference_generator(seed)
    return (
        generator.normal([8, 2], dtype=dtype),
        tf.stack([generator.normal([8, 2], dtype=dtype) for _ in range(horizon)]),
        tf.constant(np.stack([
            np.random.default_rng(resample_seed + time).uniform(size=stages)
            for time in range(horizon)
        ]), tf.float64),
    )


@pytest.mark.parametrize("seed", [0, 2**32 - 1, 2**64 - 1, 2**128 - 1, 2**160 - 1])
def test_seed_offset_carry(seed, request):
    words = tf.constant(integer_seed_words(seed), tf.uint32)

    @tf.function(input_signature=[tf.TensorSpec(words.shape, tf.uint32),
                                 tf.TensorSpec([], tf.int32)],
                 jit_compile=True, autograph=False)
    def owner(entropy, offset):
        state, increment = pcg64_offset_initial_state(entropy, offset)
        following, raw, uniform = pcg64_next(state, increment)
        return state, increment, following, raw, uniform

    records = []
    for offset in (0, 1, 2, 17, 2**31 - 1):
        state, increment, following, raw, uniform = owner(words, tf.constant(offset))
        expected = np.random.PCG64(seed + offset)
        split = lambda value: [value >> 64, value & ((1 << 64) - 1)]
        np.testing.assert_array_equal(state, split(expected.state["state"]["state"]))
        np.testing.assert_array_equal(increment, split(expected.state["state"]["inc"]))
        np.testing.assert_array_equal(raw, expected.random_raw())
        np.testing.assert_array_equal(following, split(expected.state["state"]["state"]))
        np.testing.assert_array_equal(uniform, np.random.default_rng(seed + offset).uniform())
        records.append({"offset": offset, "raw": int(raw), "uniform": float(uniform)})
    assert owner.experimental_get_tracing_count() == 1
    _save(request, f"seeded-offset-{seed}", {"seed": seed, "exact": True,
          "records": records, "trace_count": 1, "device": raw.device})


@pytest.mark.parametrize("dtype", [tf.float64, tf.float32])
def test_complete_draw_schedule(dtype, request):
    @tf.function(input_signature=[tf.TensorSpec([5], tf.uint32),
                                 tf.TensorSpec([5], tf.uint32)],
                 jit_compile=True, autograph=False)
    def owner(seed, resample_seed):
        return seeded_value_inputs(seed, resample_seed, horizon=3, particle_count=8,
                                   dimension=2, dtype=dtype, stage_count=3,
                                   annealed_resampling=True)

    cases = []
    for seed, resample_seed in ((2**160 - 1, 2**160 - 1), (2**128 + 42, 2**128 + 17)):
        args = (tf.constant(integer_seed_words(seed), tf.uint32),
                tf.constant(integer_seed_words(resample_seed), tf.uint32))
        actual = owner(*args)
        expected = _reference_inputs(seed, resample_seed, dtype)
        tolerance = 1e-12 if dtype == tf.float64 else 1e-5
        for a, b in zip(actual[:2], expected[:2], strict=True):
            np.testing.assert_allclose(a, b, atol=tolerance, rtol=tolerance)
        np.testing.assert_array_equal(actual[2], expected[2])
        for a, b in zip(actual, owner(*args), strict=True):
            np.testing.assert_array_equal(a, b)
        cases.append({"seed": seed, "resample_seed": resample_seed,
                      "actual": [a.numpy().tolist() for a in actual],
                      "expected": [a.numpy().tolist() for a in expected]})
    assert owner.experimental_get_tracing_count() == 1
    hlo = owner.experimental_get_compiler_ir(*args)(stage="hlo")
    graph = owner.get_concrete_function().graph.as_graph_def()
    operations = {n.op for n in graph.node}
    operations.update(n.op for f in graph.library.function for n in f.node_def)
    assert not operations & {"PyFunc", "EagerPyFunc", "PyFuncStateless", "XlaHostCompute"}
    _save(request, f"seeded-schedule-{dtype.name}", {"cases": cases,
          "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(), "trace_count": 1,
          "device": actual[0].device, "exact_uniforms": True})


def _fixture(authorities, case):
    baseline, fixture, source_sha = authorities
    horizon = 1 if case == "one_step" else 3
    model = fixture._lgssm_model(13, horizon=horizon)
    callbacks = fixture._callbacks_for_lgssm(model)
    observations = tf.constant(model["observations"], tf.float64)
    if case == "invalid_initial":
        callbacks = replace(callbacks, initial_mean=tf.constant([float("nan"), 0.], tf.float64))
    if case == "invalid_prediction":
        original = callbacks.transition_mean_fn
        callbacks = replace(callbacks, transition_mean_fn=lambda points, time: tf.where(
            time == 1, tf.fill(tf.shape(points), tf.constant(float("nan"), tf.float64)),
            original(points, time)))
    if case == "invalid_observation":
        observations = tf.tensor_scatter_nd_update(observations, [[1, 0]],
                                                   [tf.constant(float("nan"), tf.float64)])
    stages = 3 if case in {"composed", "annealed"} else 1
    controls = {"flow_substeps": 3, "temper_stages": stages,
                "annealed_resampling": case == "annealed",
                "flow_prior_cap": 0.8 if stages == 3 else float("inf"),
                "sinkhorn_steps": 2, "balance_steps": 2,
                "dual_cap_enabled": case == "dual_trust",
                "trust_region_enabled": case == "dual_trust"}
    return baseline, callbacks, observations, controls, source_sha


@pytest.mark.parametrize("case", ["composed", "annealed", "dual_trust", "one_step",
                                  "invalid_initial", "invalid_prediction", "invalid_observation"])
def test_registered_seeded_endpoint(authorities, case, request):
    baseline, callbacks, observations, controls, source_sha = _fixture(authorities, case)
    registration = next(e for e in ENTRY_POINTS if e.callable_name == "canonical_value_and_diagnostics")
    endpoint = getattr(importlib.import_module(registration.module), registration.callable_name)
    assert endpoint is canonical_value_and_diagnostics
    owner = make_seeded_canonical_value_program(
        callbacks, tf.TensorSpec(observations.shape, observations.dtype),
        particle_count=8, **controls)
    records = []
    for seed, shift in ((123, 0.), (124, 0.1)):
        changed_observations = observations + shift
        resample_seed = 17 + (seed - 123)
        inputs = _reference_inputs(seed, resample_seed, observations.dtype,
                                   int(observations.shape[0]), controls["temper_stages"])
        # The frozen diagnostic wrapper indexes supplied uniforms from seed17.
        expected = _frozen_reference(baseline, callbacks, changed_observations, inputs, controls)
        args = (changed_observations, tf.constant(integer_seed_words(seed), tf.uint32),
                tf.constant(integer_seed_words(resample_seed), tf.uint32))
        actual = owner(*args)
        public = endpoint(callbacks, changed_observations, particle_count=8, seed=seed,
                          resample_seed=resample_seed, **controls)
        record = {"seed": seed, "resample_seed": resample_seed,
                  "actual": {k: v.numpy().tolist() for k, v in actual.items()},
                  "expected": {k: v.numpy().tolist() for k, v in expected.items() if k != "model_id"}}
        records.append(record)
        _save(request, f"seeded-endpoint-{case}", {"records": records, "baseline_sha": source_sha})
        record["comparison"] = _compare(actual, expected)
        record["public_comparison"] = _compare(public, expected)
        np.testing.assert_array_equal(public["model_id"], callbacks.model_id.encode())
        completed = int(actual["marginal_steps_completed"])
        assert public["per_step_marginal_valid"].shape == [completed]
        for key, value in owner(*args).items():
            np.testing.assert_array_equal(value, actual[key], err_msg=key)
        if case.startswith("invalid"):
            assert not bool(public["program_valid"])
            assert int(public["numerical_failure_code"]) == 1
    assert owner.experimental_get_tracing_count() == 1
    hlo = owner.experimental_get_compiler_ir(*args)(stage="hlo")
    assert hlo == owner.experimental_get_compiler_ir(
        observations, tf.constant([125], tf.uint32), tf.constant([19], tf.uint32))(stage="hlo")
    graph = owner.get_concrete_function().graph.as_graph_def()
    operations = {n.op for n in graph.node}
    operations.update(n.op for f in graph.library.function for n in f.node_def)
    assert not operations & {"PyFunc", "EagerPyFunc", "PyFuncStateless", "XlaHostCompute"}
    assert owner.function_spec.jit_compile is True
    reference = weakref.ref(owner)
    del owner
    gc.collect()
    assert reference() is None
    _save(request, f"seeded-endpoint-{case}", {"records": records,
          "baseline_sha": source_sha, "trace_count": 1, "owner_collected": True,
          "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(), "device": actual["value"].device,
          "registered_module": registration.module, "enclosing_xla": True})


def test_mutable_callback_refresh(authorities, request):
    baseline, callbacks, observations, controls, source_sha = _fixture(authorities, "composed")
    state = {"shift": 0.0}
    original = callbacks.observation_log_density_fn
    callbacks = replace(callbacks, observation_log_density_fn=lambda points, observation, time:
                        original(points, observation, time) + state["shift"] * tf.cast(time + 1, tf.float64))
    values = []
    for shift in (0.0, 0.125):
        state["shift"] = shift
        result = canonical_value_and_diagnostics(callbacks, observations, particle_count=8,
                                                  seed=123, **controls)
        inputs = _reference_inputs(123, 17, observations.dtype)
        expected = _frozen_reference(baseline, callbacks, observations, inputs, controls)
        _compare(result, expected)
        values.append(float(result["raw_value"]))
    np.testing.assert_allclose(values[1] - values[0], 0.125 * (1 + 2 + 3), atol=1e-6, rtol=1e-6)
    _save(request, "seeded-callback-refresh", {"raw_values": values,
          "expected_difference": 0.75, "baseline_sha": source_sha,
          "tensor_time_and_mutable_closure_checked": True})


def test_dual_seeded_localization(authorities, request):
    """Preserve the healthy GPU mismatch; separate RNG from enclosing context."""
    baseline, callbacks, observations, controls, source_sha = _fixture(authorities, "dual_trust")
    spec = tf.TensorSpec(observations.shape, observations.dtype)
    supplied = make_canonical_value_program(callbacks, spec, particle_count=8, **controls)
    seeded = make_seeded_canonical_value_program(callbacks, spec, particle_count=8, **controls)

    @tf.function(input_signature=[tf.TensorSpec([1], tf.uint32), tf.TensorSpec([1], tf.uint32)],
                 jit_compile=True, autograph=False)
    def random_inputs(seed, resample_seed):
        return seeded_value_inputs(seed, resample_seed, horizon=3, particle_count=8,
                                   dimension=2, dtype=tf.float64, stage_count=1)

    args = (tf.constant([123], tf.uint32), tf.constant([17], tf.uint32))
    original_inputs = _reference_inputs(123, 17, tf.float64, stages=1)
    native_inputs = random_inputs(*args)
    records = {
        "frozen_original_inputs": _frozen_reference(baseline, callbacks, observations, original_inputs, controls),
        "frozen_native_inputs": _frozen_reference(baseline, callbacks, observations, native_inputs, controls),
        "supplied_original_inputs": supplied(observations, *original_inputs),
        "supplied_native_inputs": supplied(observations, *native_inputs),
        "seeded": seeded(observations, *args),
    }
    serialized = {name: {k: v.numpy().tolist() for k, v in record.items() if k != "model_id"}
                  for name, record in records.items()}
    comparisons = {}
    for name, record in records.items():
        if name.startswith("frozen"):
            continue
        try:
            comparisons[name] = {"passed": True, "errors": _compare(record, records["frozen_original_inputs"])}
        except AssertionError as error:
            comparisons[name] = {"passed": False, "error": str(error)}
    report = {"schema": "filter_ledh_seeded_localization.v1", "records": serialized,
              "comparisons": comparisons, "baseline_sha": source_sha,
              "original_inputs": [v.numpy().tolist() for v in original_inputs],
              "native_inputs": [v.numpy().tolist() for v in native_inputs],
              "input_max_errors": [float(tf.reduce_max(tf.abs(a-b))) for a,b in
                                    zip(original_inputs, native_inputs, strict=True)],
              "device": records["seeded"]["value"].device,
              "role": "explanatory_only_not_equivalence"}
    _save(request, "seeded-dual-localization", report)
    assert all(bool(row["numerical_valid"]) for row in records.values())
    assert supplied.experimental_get_tracing_count() == seeded.experimental_get_tracing_count() == 1


def test_seeded_reset_localization(authorities, request, monkeypatch):
    """Isolate shared reset and trust inputs; precision arms are diagnostics."""
    from bayesfilter.highdim import genut_guided_proposal_tf as proposal

    baseline, callbacks, observations, controls, source_sha = _fixture(authorities, "dual_trust")
    inputs = _reference_inputs(123, 17, tf.float64, stages=1)
    resets, trusts = [], []
    restore = proposal._restore_cloud_primal
    trust = proposal.higher_moment_shape_jvp

    def capture_trust(*args, **kwargs):
        result = trust(*args, **kwargs)
        trusts.append((args, kwargs, result))
        return result

    def capture_reset(*args, **kwargs):
        result = restore(*args, **kwargs)
        resets.append((args, kwargs, result))
        return result

    with monkeypatch.context() as capture:
        capture.setattr(baseline, "_restore_cloud_primal", capture_reset)
        capture.setattr(proposal, "higher_moment_shape_jvp", capture_trust)
        expected = _frozen_reference(baseline, callbacks, observations, inputs, controls)
    assert bool(expected["numerical_valid"])

    def serialized(record):
        return {k: v.numpy().tolist() for k, v in record.items() if k != "model_id"}

    def build_kernel(implementation, options, signature, jit):
        def invoke(*args):
            return implementation(*args, **options)
        return tf.function(invoke, input_signature=signature, jit_compile=jit, autograph=False)

    reports = {}
    compiled_resets = {}
    for label, calls, implementation in (("reset", resets, restore), ("trust", trusts, trust)):
        options = calls[0][1]
        mode_kernels = {}
        for dtype in (tf.float32, tf.float64):
            signature = [tf.TensorSpec(a.shape, dtype) for a in calls[0][0]]
            for mode, jit in (("graph", False), ("xla", True)):
                mode_kernels[dtype.name, mode] = build_kernel(implementation, options, signature, jit)
        if label == "reset":
            compiled_resets = mode_kernels
        rows = []
        for args, _, reference in calls:
            row = {"inputs": [a.numpy().tolist() for a in args],
                   "modes": {"float32_eager": serialized(reference)}}
            for dtype in (tf.float32, tf.float64):
                operands = tuple(tf.cast(a, dtype) for a in args)
                if dtype == tf.float64:
                    row["modes"]["float64_eager"] = serialized(implementation(*operands, **options))
                for mode in ("graph", "xla"):
                    row["modes"][f"{dtype.name}_{mode}"] = serialized(mode_kernels[dtype.name, mode](*operands))
            rows.append(row)
        reports[label] = {"options": options, "steps": rows}

    trajectories = {"frozen_eager": serialized(expected)}
    for mode in ("graph", "xla"):
        kernel = compiled_resets["float32", mode]
        with monkeypatch.context() as intervention:
            intervention.setattr(baseline, "_restore_cloud_primal", lambda *args, _kernel=kernel, **kwargs: _kernel(*args))
            result = _frozen_reference(baseline, callbacks, observations, inputs, controls)
        trajectories[f"frozen_with_{mode}_reset"] = serialized(result)
    _save(request, "seeded-reset-localization", {"schema": "filter_ledh_seeded_reset_localization.v1",
        "role": "explanatory_only_not_runtime_precision_change", "baseline_sha": source_sha,
        "components": reports, "trajectories": trajectories, "device": expected["value"].device})


def test_seeded_reset_tf32_localization(request):
    """TF32 toggle is a reference diagnostic, never a runtime default change."""
    from bayesfilter.highdim import genut_guided_proposal_tf as proposal

    path = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917/run-04675/seeded-reset-localization.json")
    prior = json.loads(path.read_text())
    initial_tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
    reports = {}
    try:
        for label, implementation in (("reset", proposal._restore_cloud_primal),
                                       ("trust", proposal.higher_moment_shape_jvp)):
            component = prior["components"][label]
            rows = []
            for row in component["steps"]:
                inputs = tuple(tf.constant(v, tf.float32) for v in row["inputs"])
                modes = {}
                for tf32 in (True, False):
                    tf.config.experimental.enable_tensor_float_32_execution(tf32)
                    result = implementation(*inputs, **component["options"])
                    modes[str(tf32)] = {k: v.numpy().tolist() for k, v in result.items()}
                rows.append({"modes": modes,
                             "float64_eager": row["modes"]["float64_eager"],
                             "float32_xla": row["modes"]["float32_xla"]})
            reports[label] = rows
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(initial_tf32)
    _save(request, "seeded-reset-tf32-localization", {"components": reports,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "initial_tf32": initial_tf32,
        "restored_tf32": tf.config.experimental.tensor_float_32_execution_enabled(),
        "role": "diagnostic_reference_toggle_no_runtime_change"})
    assert tf.config.experimental.tensor_float_32_execution_enabled() == initial_tf32


def test_seeded_lm_tf32_localization(request, monkeypatch):
    """Check whether only the tiny LM products account for TF32 sensitivity."""
    from bayesfilter.highdim import higher_moment_contract_e as moment
    from bayesfilter.highdim.genut_shape_lm_tf import scaled_lm_coefficients_jvp

    path = Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917/run-04675/seeded-reset-localization.json")
    source = json.loads(path.read_text())["components"]["trust"]
    initial_tf32 = tf.config.experimental.tensor_float_32_execution_enabled()
    reports = []
    try:
        for saved in source["steps"]:
            args = tuple(tf.constant(v, tf.float32) for v in saved["inputs"])
            calls = []

            def capture(*operands, captured=calls, **options):
                result = scaled_lm_coefficients_jvp(*operands, **options)
                captured.append((operands, options, result))
                return result

            tf.config.experimental.enable_tensor_float_32_execution(True)
            with monkeypatch.context() as patch:
                patch.setattr(moment, "scaled_lm_coefficients_jvp", capture)
                original = moment.higher_moment_shape_jvp(*args, **source["options"])

            def lm_no_tf32(*operands, **options):
                tf.config.experimental.enable_tensor_float_32_execution(False)
                try:
                    return scaled_lm_coefficients_jvp(*operands, **options)
                finally:
                    tf.config.experimental.enable_tensor_float_32_execution(True)

            with monkeypatch.context() as patch:
                patch.setattr(moment, "scaled_lm_coefficients_jvp", lm_no_tf32)
                isolated = moment.higher_moment_shape_jvp(*args, **source["options"])
            primitives = []
            for operands, options, before in calls:
                after = lm_no_tf32(*operands, **options)
                high = scaled_lm_coefficients_jvp(*(tf.cast(v, tf.float64) for v in operands), **options)
                primitives.append({"inputs": [v.numpy().tolist() for v in operands],
                    "options": options, "before": {k:v.numpy().tolist() for k,v in before.items()},
                    "no_tf32": {k:v.numpy().tolist() for k,v in after.items()},
                    "float64": {k:v.numpy().tolist() for k,v in high.items()}})
            reports.append({"original": {k:v.numpy().tolist() for k,v in original.items()},
                "only_lm_no_tf32": {k:v.numpy().tolist() for k,v in isolated.items()},
                "float64_eager": saved["modes"]["float64_eager"], "lm_calls": primitives})
    finally:
        tf.config.experimental.enable_tensor_float_32_execution(initial_tf32)
    _save(request, "seeded-lm-tf32-localization", {"steps": reports,
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "role": "diagnostic_isolated_LM_toggle", "tf32_restored": initial_tf32})

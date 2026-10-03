"""Opt-in full-chain typed-XLA mechanics qualification, not posterior evidence."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import resource
import time
from types import SimpleNamespace

import pytest


if os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") != "visible":
    pytest.skip("requires an explicitly trusted visible-GPU launch", allow_module_level=True)
if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1"}:
    raise RuntimeError("qualification requires exactly one approved GPU")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("memory growth must be configured before TensorFlow import")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


GPU_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)

from bayesfilter.inference import (
    BoundRetainedHMCArchiveConfig,
    FourChainMeanBandAcceptancePolicy,
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    TensorFlowHMCKernelTuningConfig,
    bind_neural_force_hmc_tuning_runner,
    build_retained_bound_hmc_archive_runner_from_tuning_result,
    load_tensorflow_hmc_tuning_result,
    tune_hmc_kernel,
)
from bayesfilter.inference.neural_force_hmc import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
)


@tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64)], jit_compile=True)
def potential(position):
    return 0.5 * tf.reduce_sum(tf.square(position), axis=-1)


@tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64)], jit_compile=True)
def proposal_field(position):
    return 0.9 * position


@tf.function(
    input_signature=[tf.TensorSpec([4, 2], tf.float64)] * 2, jit_compile=True,
)
def state_checks(first, second):
    return tf.reduce_all(tf.math.is_finite(first)), tf.reduce_all(first == second)


def assert_same_state(first, second):
    finite, equal = state_checks(first, second)
    assert bool(tf.get_static_value(finite))
    assert bool(tf.get_static_value(equal))


def test_typed_tuning_retained_reload_and_rejection_xla(tmp_path):
    started = time.monotonic()
    root = Path(os.environ.get("BAYESFILTER_XLA_QUALIFICATION_ROOT", str(tmp_path)))
    root.mkdir(parents=True, exist_ok=True)
    binding = bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(
            function=proposal_field, identity="typed-xla-nonexact-gaussian-field-v1",
            semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
            coordinate_system="raw",
        ),
        target=FrozenTargetPotential(
            function=potential, identity="typed-xla-gaussian-potential-v1",
            coordinate_system="raw", includes_chart_log_jacobian=True,
        ),
        target_scope="typed_xla_synthetic_mechanics_v1",
    )
    adapter = SimpleNamespace(
        target_scope="typed_xla_synthetic_mechanics_v1",
        adapter_signature=lambda: "typed-xla-synthetic-v1",
    )
    config = TensorFlowHMCKernelTuningConfig(
        parameter_dimension=2, evidence_role="candidate", mass_window_results=(2,),
        step_adaptation_results=4, verification_results=4, max_leapfrog_steps=4,
        initial_step_size=0.01, budget_provenance="tiny compilation/replay regression only",
        initial_step_size_provenance="inherited synthetic handoff fixture, not BGS tuning",
        geometry_provenance="synthetic diagonal scale hypothesis",
        target_scope=adapter.target_scope,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(0.0, 1.0), per_chain_band=(0.0, 1.0),
        ),
        target_accept_prob=0.7, verification_repair_rounds=2, step_repair_factor=2.0,
        mass_shrinkage=0.5, covariance_jitter=0.0, eigenvalue_floor=1e-10,
        max_condition_number=1e8, seed=(11, 23), use_xla=True,
    )
    initial = tf.constant([[-0.2, 0.1], [0.1, -0.2], [0.2, 0.3], [-0.1, -0.3]], tf.float64)
    scales = tf.constant([1.0, 2.0], tf.float64)
    with tf.device("/GPU:0"):
        tuning = tune_hmc_kernel(
            adapter=adapter, initial_position=initial, config=config,
            parameter_scales=scales, runner_binding=binding, output_dir=root / "tuning",
        )
        assert bool(tf.get_static_value(tuning.handoff_eligible))
        concrete = tuning.graph_function.get_concrete_function(initial, scales)
        assert concrete.function_def.attr["_XlaMustCompile"].b
        replay = tuning.graph_function(initial, scales)
        assert_same_state(tuning.final_chain_state, replay["final_chain_state"])
        assert tuning.final_chain_state.device.endswith("GPU:0")
        hlo = tuning.graph_function.experimental_get_compiler_ir(initial, scales)(stage="hlo")
        (root / "tuning-hlo.txt").write_text(hlo)
        loaded = load_tensorflow_hmc_tuning_result(
            tuning.artifact_manifest_path, adapter=adapter, runner_binding=binding,
        )
        assert loaded.config.use_xla is True
        runner = build_retained_bound_hmc_archive_runner_from_tuning_result(
            tuning_result=loaded, runner_binding=binding,
        )
        pilot = runner.run(BoundRetainedHMCArchiveConfig(
            num_results=3, seed=(11, 24), output_dir=root / "pilot",
            budget_provenance="three-transition compiled archive regression",
        ))
        assert_same_state(pilot.initial_chain_state, tuning.final_chain_state)
        assert bool(tf.get_static_value(pilot.health_passed))
        assert pilot.final_chain_state.device.endswith("GPU:0")
        extension = runner.run(BoundRetainedHMCArchiveConfig(
            num_results=3, seed=(11, 25), output_dir=root / "extension",
            budget_provenance="three-transition compiled predecessor regression",
            continuation_manifest=pilot.archive_manifest_path,
        ))
        assert_same_state(extension.initial_chain_state, pilot.final_chain_state)
        assert extension.final_chain_state.device.endswith("GPU:0")
        pilot_manifest = json.loads(Path(pilot.archive_manifest_path).read_text())
        assert pilot_manifest["use_xla"] is True
        assert len(pilot_manifest["tensors"]) == 11
        failed = tune_hmc_kernel(
            adapter=adapter, initial_position=initial,
            config=replace(config, acceptance_policy=FourChainMeanBandAcceptancePolicy(
                overall_band=(0.0, 0.0), per_chain_band=(0.0, 0.0),
            ), verification_repair_rounds=0),
            parameter_scales=scales, runner_binding=binding, output_dir=root / "rejected",
        )
        assert not bool(tf.get_static_value(failed.handoff_eligible))
        with pytest.raises(tf.errors.InvalidArgumentError, match="handoff-eligible"):
            build_retained_bound_hmc_archive_runner_from_tuning_result(
                tuning_result=failed, runner_binding=binding,
            )
    record = {
        "question": "full-chain typed GPU XLA compilation and archive mechanics",
        "passed": True, "canonical_bgs_qualified": False, "posterior_admitted": False,
        "tensorflow": tf.__version__, "gpu_policy": GPU_POLICY,
        "visible_gpu": os.environ["CUDA_VISIBLE_DEVICES"],
        "device": tuning.final_chain_state.device,
        "allocator": tf.config.experimental.get_memory_info("GPU:0"),
        "host_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "wall_seconds": time.monotonic() - started,
        "seed": list(config.seed), "retained_seeds": [[11, 24], [11, 25]],
        "tuning_manifest": tuning.artifact_manifest_path,
        "pilot_manifest": pilot.archive_manifest_path,
        "extension_manifest": extension.archive_manifest_path,
        "rejected_manifest": failed.artifact_manifest_path,
    }
    (root / "qualification.json").write_text(json.dumps(record, indent=2) + "\n")

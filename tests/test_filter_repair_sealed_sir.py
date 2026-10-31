"""Preserve the existing sealed SIR dataset independently of compiler rounding."""

import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (
    SIR_OBSERVATION_SHA256,
    SIR_RUNTIME_FP32_OBSERVATION_SHA256,
    SIR_STATE_SHA256,
    SIR_WRONG_TIME_ORDER_SHA256,
    generate_sealed_lane_b_dataset,
    tensor_sha256,
)


def test_sealed_dataset_keeps_all_original_hashes(request):
    states, observations, all_observations = generate_sealed_lane_b_dataset()
    assert tensor_sha256(states) == SIR_STATE_SHA256
    assert tensor_sha256(observations) == SIR_OBSERVATION_SHA256
    assert tensor_sha256(all_observations[:-1]) == SIR_WRONG_TIME_ORDER_SHA256
    assert tensor_sha256(tf.cast(observations, tf.float32)) == SIR_RUNTIME_FP32_OBSERVATION_SHA256
    # The bounded campaign records the imported source tree and run identity.
    # Export the freshly reproduced pinned baseline for an immutable dataset
    # source; no historical filtering result is consumed.
    junit = request.config.getoption("xmlpath")
    if junit:
        path = Path(junit).parent / "sealed-sir-dataset.json"
        with path.open("x") as output:
            json.dump({"states": states.numpy().tolist(),
                       "all_observations": all_observations.numpy().tolist()}, output,
                      allow_nan=False, indent=2)
            output.write("\n")


def test_sealed_tensors_compile_and_reuse_one_signature():
    from bayesfilter.highdim.sealed_sir_dataset_tf import sealed_sir_tensors

    first = sealed_sir_tensors()
    second = sealed_sir_tensors()
    assert tensor_sha256(first[0]) == tensor_sha256(second[0]) == SIR_STATE_SHA256
    assert "HloModule" in sealed_sir_tensors.experimental_get_compiler_ir()(stage="hlo")
    assert sealed_sir_tensors.experimental_get_tracing_count() == 1

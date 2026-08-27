"""Construct an honestly labeled neural-force binding without launching HMC."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

import tensorflow as tf

from bayesfilter.inference import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    bind_neural_force_hmc_tuning_runner,
)


def build_binding() -> Any:
    """Bind a deterministic non-gradient proposal field to an exact potential."""

    def proposal_field(position: tf.Tensor) -> tf.Tensor:
        position = tf.convert_to_tensor(position, tf.float64)
        return tf.stack(
            (
                position[..., 0] + 0.2 * position[..., 1],
                0.4 * position[..., 1],
            ),
            axis=-1,
        )

    def exact_endpoint_potential(position: tf.Tensor) -> tf.Tensor:
        position = tf.convert_to_tensor(position, tf.float64)
        return 0.5 * tf.reduce_sum(tf.square(position), axis=-1)

    return bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(
            function=proposal_field,
            identity="docs-deterministic-proposal-field-v1",
            semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
            coordinate_system="raw",
        ),
        target=FrozenTargetPotential(
            function=exact_endpoint_potential,
            identity="docs-exact-endpoint-potential-v1",
            coordinate_system="raw",
        ),
        target_scope="docs_neural_force_binding",
    )


def main() -> Mapping[str, Any]:
    """Return the public identity payload; no tuner or chain runner is called."""

    payload = build_binding().payload()
    assert payload["artifact_authority"] is False
    assert payload["force_semantics"] == (
        DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS
    )
    assert payload["endpoint_target_coordinate_system"] == "raw"
    return payload


if __name__ == "__main__":
    print(main()["binding_hash"])

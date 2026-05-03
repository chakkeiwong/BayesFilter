import pytest

from bayesfilter import StatePartition, StructuralFilterConfig, validate_filter_config


def test_partition_validates_dimension_and_roles():
    partition = StatePartition(
        state_names=("m", "lag_m", "acc"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        auxiliary_indices=(2,),
        innovation_dim=1,
    )

    assert partition.state_dim == 3
    assert partition.is_mixed
    assert partition.role_of(2) == "auxiliary"


def test_partition_rejects_overlapping_indices():
    with pytest.raises(ValueError, match="appears in both"):
        StatePartition(
            state_names=("m", "lag_m"),
            stochastic_indices=(0,),
            deterministic_indices=(0,),
            innovation_dim=1,
        )


def test_partition_rejects_missing_coverage():
    with pytest.raises(ValueError, match="does not cover"):
        StatePartition(
            state_names=("m", "lag_m"),
            stochastic_indices=(0,),
            deterministic_indices=(),
            innovation_dim=1,
        )


def test_mixed_full_state_requires_explicit_label_and_opt_in():
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    config = StructuralFilterConfig(
        integration_space="full_state",
        deterministic_completion="approximate",
        approximation_label="legacy_full_state_gaussian",
        allow_full_state_for_mixed=True,
    )

    validate_filter_config(partition, config)


def test_mixed_full_state_without_label_fails_closed():
    partition = StatePartition(
        state_names=("m", "lag_m"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    with pytest.raises(ValueError, match="requires explicit opt-in"):
        validate_filter_config(
            partition,
            StructuralFilterConfig(
                integration_space="full_state",
                deterministic_completion="required",
            ),
        )

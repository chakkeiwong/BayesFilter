from __future__ import annotations

from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.highdim.models import PredatorPreySSM
from bayesfilter.highdim.zhao_cui_predator_prey_proposal_tf import (
    PredatorPreyProposalSpec,
    evaluate_predator_prey_proposal_candidate,
    fit_predator_prey_proposal_candidate,
    make_tuning_artifact,
    select_l1_candidate,
    select_structural_candidate,
)
from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
    generate_source_order_predator_prey_dataset_tf,
)


def _observations(seed: int) -> tf.Tensor:
    model = PredatorPreySSM(dtype=tf.float64)
    _states, observations = model.simulate(model.true_parameters(), 20, seed)
    return observations[1:]


@pytest.fixture(scope="module")
def protocol_fixture():
    calibration = _observations(83105)
    validation = _observations(83106)
    audit_observations = _observations(83107)
    zero_spec = PredatorPreyProposalSpec(
        degree=0,
        rank=1,
        coordinate_scale=8.0,
        defensive_tau=1e-3,
        l1_weight=0.0,
        ridge=1e-6,
        train_steps=0,
        cdf_grid_size=17,
        cdf_bisection_steps=8,
    )
    positive_spec = replace(zero_spec, l1_weight=1e-6)
    calibration_candidate = fit_predator_prey_proposal_candidate(
        observations=calibration,
        spec=zero_spec,
        calibration_order=2,
        validation_order=3,
        calibration_seed=8310501,
        validation_seed=8310601,
        data_role="calibration",
    )
    zero = fit_predator_prey_proposal_candidate(
        observations=validation,
        spec=zero_spec,
        calibration_order=2,
        validation_order=3,
        calibration_seed=8310602,
        validation_seed=8310603,
        data_role="validation",
    )
    positive = fit_predator_prey_proposal_candidate(
        observations=validation,
        spec=positive_spec,
        calibration_order=2,
        validation_order=3,
        calibration_seed=8310602,
        validation_seed=8310603,
        data_role="validation",
    )
    selected, selection = select_l1_candidate((zero, positive))
    # Audit fitting is final-only: controls are inherited from validation and
    # the audit observations cannot select structure or L1.
    audit_candidate = fit_predator_prey_proposal_candidate(
        observations=audit_observations,
        spec=selected.spec,
        calibration_order=2,
        validation_order=3,
        calibration_seed=8310701,
        validation_seed=8310702,
        data_role="audit",
        frozen_control_source_scope_id=selected.scope_id,
    )
    audit = evaluate_predator_prey_proposal_candidate(
        audit_candidate,
        observations=audit_observations,
        design_order=3,
        design_seed=8310701,
    )
    tuning = make_tuning_artifact(
        calibration_candidate=calibration_candidate,
        selected_candidate=selected,
        audit_candidate=audit_candidate,
        calibration_observation_hash=calibration_candidate.observation_hash,
        validation_observation_hash=selected.observation_hash,
        audit=audit,
        selection_diagnostics={"l1_selection": selection},
    )
    return {
        "calibration": calibration,
        "validation": validation,
        "audit_observations": audit_observations,
        "zero": zero,
        "positive": positive,
        "calibration_candidate": calibration_candidate,
        "selected": selected,
        "audit_candidate": audit_candidate,
        "audit": audit,
        "tuning": tuning,
    }


def test_sealed_observations_require_repository_tuning_artifact() -> None:
    _states, sealed = generate_source_order_predator_prey_dataset_tf()
    sealed = tf.cast(sealed, tf.float64)
    spec = PredatorPreyProposalSpec(degree=0, rank=1, train_steps=0)
    with pytest.raises(ValueError, match="requires a repository-issued tuning artifact"):
        fit_predator_prey_proposal_candidate(
            observations=sealed,
            spec=spec,
            calibration_order=2,
            validation_order=3,
            data_role="sealed_claim_preparation",
        )


def test_candidate_and_audit_identity_reject_tampering(protocol_fixture) -> None:
    with pytest.raises(ValueError, match="caller-stamped proposal candidate identity"):
        replace(protocol_fixture["zero"], data_role="audit")
    with pytest.raises(ValueError, match="caller-stamped proposal audit identity"):
        replace(protocol_fixture["audit"], audit_id="0" * 64)


def test_l1_selection_holds_non_l1_controls_and_roles_fixed(protocol_fixture) -> None:
    selected, diagnostics = select_l1_candidate(
        (protocol_fixture["zero"], protocol_fixture["positive"])
    )
    assert selected.spec.structural_payload() == protocol_fixture[
        "zero"
    ].spec.structural_payload()
    assert diagnostics["audit_data_used_for_selection"] is False

    with pytest.raises(ValueError, match="caller-stamped proposal candidate identity"):
        # The repository identity rejects role relabelling before selection.
        replace(
            protocol_fixture["zero"],
            data_role="calibration",
            scope_id="0" * 64,
        )

    changed_structure = fit_predator_prey_proposal_candidate(
        observations=protocol_fixture["validation"],
        spec=replace(protocol_fixture["positive"].spec, coordinate_scale=9.0),
        calibration_order=2,
        validation_order=3,
        calibration_seed=8310501,
        validation_seed=8310601,
        data_role="validation",
    )
    with pytest.raises(ValueError, match="hold all non-L1 controls fixed"):
        select_l1_candidate((protocol_fixture["zero"], changed_structure))


def test_structural_selection_rejects_validation_candidates(protocol_fixture) -> None:
    with pytest.raises(ValueError, match="calibration-role candidates only"):
        select_structural_candidate((protocol_fixture["zero"],))

    selected, _ = select_structural_candidate(
        (protocol_fixture["calibration_candidate"],)
    )
    assert selected.data_role == "calibration"


def test_tuning_identity_and_audit_binding_reject_tampering(protocol_fixture) -> None:
    with pytest.raises(ValueError, match="caller-stamped tuning artifact identity"):
        replace(protocol_fixture["tuning"], artifact_id="0" * 64)
    with pytest.raises(ValueError, match="final-only audit fitting"):
        make_tuning_artifact(
            calibration_candidate=protocol_fixture["calibration_candidate"],
            selected_candidate=protocol_fixture["positive"],
            audit_candidate=protocol_fixture["calibration_candidate"],
            calibration_observation_hash=protocol_fixture[
                "calibration_candidate"
            ].observation_hash,
            validation_observation_hash=protocol_fixture[
                "positive"
            ].observation_hash,
            audit=protocol_fixture["audit"],
            selection_diagnostics={},
        )


def test_audit_fit_requires_validation_control_scope(protocol_fixture) -> None:
    with pytest.raises(ValueError, match="validation-issued frozen control scope"):
        fit_predator_prey_proposal_candidate(
            observations=protocol_fixture["audit_observations"],
            spec=protocol_fixture["selected"].spec,
            calibration_order=2,
            validation_order=3,
            data_role="audit",
        )

    assert (
        protocol_fixture["audit_candidate"].fit_manifest[
            "frozen_control_source_scope_id"
        ]
        == protocol_fixture["selected"].scope_id
    )


def test_claim_controls_must_exactly_match_tuning_artifact(protocol_fixture) -> None:
    _states, sealed = generate_source_order_predator_prey_dataset_tf()
    sealed = tf.cast(sealed, tf.float64)
    mismatched = replace(protocol_fixture["tuning"].selected_spec, coordinate_scale=9.0)
    with pytest.raises(ValueError, match="claim controls differ from the tuning artifact"):
        fit_predator_prey_proposal_candidate(
            observations=sealed,
            spec=mismatched,
            tuning_artifact=protocol_fixture["tuning"],
            calibration_order=2,
            validation_order=3,
            data_role="sealed_claim_preparation",
        )

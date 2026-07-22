from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_campaign import (
    CampaignCellLedger,
    NeuTraCampaignError,
    SeparateCampaignArchive,
    TypedNeuTraTargetIdentity,
    admit_independent_posterior_recomposition,
    deterministic_cpu_sample_partitions,
    generate_cpu_sample_batch,
    issue_typed_neutra_target_identity,
    load_campaign_neutra_transport,
    campaign_fixed_transport_adapter,
    load_validated_p0_registry,
    require_typed_neutra_target,
    run_campaign_neutra_hmc,
    train_campaign_neutra,
)
from bayesfilter.inference.neutra_hmc import SequentialNeuTraHMCConfig
from bayesfilter.inference.neutra_training import PlainDenseIAFTrainingConfig
from bayesfilter.testing.multimodel_neutra_p1_canary_tf import (
    SYNTHETIC_CANARY_SCOPE,
    SyntheticGaussianCampaignAdapter,
    make_synthetic_gaussian_contract,
    synthetic_exponential_chart_jacobian_value_score,
    synthetic_gaussian_likelihood_value_score,
    synthetic_gaussian_prior_value_score,
    synthetic_gaussian_final_posterior_value_score,
)


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
P0_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
    "phase-p0/attempt-04-20260715T1658"
)
P0_REGISTRY_SHA256 = (
    "eba02073b8b2f4a2b648128ace5163356cf5971a5c66724bd82048e97d522a3d"
)


def wrong_zero_jacobian_value_score(
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    values = tf.convert_to_tensor(theta, tf.float64)
    return tf.zeros(tf.shape(values)[:-1], tf.float64), tf.zeros_like(values)


def recomposition_via_production_assembler(
    theta: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    adapter = SyntheticGaussianCampaignAdapter()
    return adapter.log_prob_and_grad(theta)


def _points() -> tf.Tensor:
    return tf.constant(
        [[-0.75, 0.50], [0.0, 0.0], [0.60, -0.40]], tf.float64
    )


def _recomposition(adapter: SyntheticGaussianCampaignAdapter):
    return admit_independent_posterior_recomposition(
        adapter=adapter,
        points=_points(),
        prior_value_score_fn=synthetic_gaussian_prior_value_score,
        likelihood_value_score_fn=synthetic_gaussian_likelihood_value_score,
        jacobian_value_score_fn=synthetic_exponential_chart_jacobian_value_score,
    )


def _identity(adapter: SyntheticGaussianCampaignAdapter | None = None):
    target = SyntheticGaussianCampaignAdapter() if adapter is None else adapter
    return target, issue_typed_neutra_target_identity(
        program_id=PROGRAM_ID,
        scope_kind="synthetic_canary",
        scope_id=SYNTHETIC_CANARY_SCOPE,
        adapter=target,
        recomposition=_recomposition(target),
    )


def _affine_payload(identity, *, target_signature: str | None = None):
    signature = identity.target_signature if target_signature is None else target_signature
    return {
        "schema": "bayesfilter.neutra.frozen_affine_diag.v1",
        "transport_id": "p1-test-affine",
        "dimension": 2,
        "target_signature": signature,
        "log_jacobian_available": True,
        "shift": [0.0, 0.0],
        "raw_scale": [0.0, 0.0],
        "training_state_hash": None,
    }


def test_independent_recomposition_passes_and_omitted_jacobian_fails() -> None:
    adapter = SyntheticGaussianCampaignAdapter()
    admission = _recomposition(adapter)

    assert admission.passed is True
    assert admission.maximum_absolute_value_error <= admission.value_tolerance
    assert admission.maximum_absolute_score_error <= admission.score_tolerance
    assert {row["role"] for row in admission.component_identities} == {
        "prior",
        "filter_likelihood",
        "unconstraining_jacobian",
    }

    with pytest.raises(NeuTraCampaignError, match="recomposition mismatch"):
        admit_independent_posterior_recomposition(
            adapter=adapter,
            points=_points(),
            prior_value_score_fn=synthetic_gaussian_prior_value_score,
            likelihood_value_score_fn=synthetic_gaussian_likelihood_value_score,
            jacobian_value_score_fn=wrong_zero_jacobian_value_score,
        )


def test_recomposition_rejects_final_assembler_reuse() -> None:
    adapter = SyntheticGaussianCampaignAdapter()
    with pytest.raises(NeuTraCampaignError, match="production final assembler"):
        admit_independent_posterior_recomposition(
            adapter=adapter,
            points=_points(),
            prior_value_score_fn=recomposition_via_production_assembler,
            likelihood_value_score_fn=synthetic_gaussian_likelihood_value_score,
            jacobian_value_score_fn=synthetic_exponential_chart_jacobian_value_score,
        )
    with pytest.raises(NeuTraCampaignError, match="production final assembler"):
        admit_independent_posterior_recomposition(
            adapter=adapter,
            points=_points(),
            prior_value_score_fn=synthetic_gaussian_final_posterior_value_score,
            likelihood_value_score_fn=synthetic_gaussian_likelihood_value_score,
            jacobian_value_score_fn=synthetic_exponential_chart_jacobian_value_score,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("data_hash", "sha256:p1-changed-observations"),
        ("prior_hash", "sha256:p1-changed-prior"),
        ("filter_hash", "sha256:p1-changed-filter"),
        ("transform_hash", "sha256:p1-changed-chart"),
        ("model_hash", "sha256:p1-changed-model"),
    ),
)
def test_typed_identity_changes_with_mathematical_target(field, value) -> None:
    base_adapter, base = _identity()
    del base_adapter
    changed_adapter = SyntheticGaussianCampaignAdapter(
        make_synthetic_gaussian_contract(**{field: value})
    )
    changed = issue_typed_neutra_target_identity(
        program_id=PROGRAM_ID,
        scope_kind="synthetic_canary",
        scope_id=SYNTHETIC_CANARY_SCOPE,
        adapter=changed_adapter,
        recomposition=_recomposition(changed_adapter),
    )

    assert base.mathematical_target_signature != changed.mathematical_target_signature
    assert base.target_signature != changed.target_signature


def test_typed_identity_changes_with_dtype() -> None:
    class Float32SyntheticAdapter(SyntheticGaussianCampaignAdapter):
        dtype = tf.float32

        def __init__(self) -> None:
            super().__init__()
            self._adapter_signature = "f" * 64

    _, base = _identity()
    changed_adapter = Float32SyntheticAdapter()
    with pytest.raises(NeuTraCampaignError, match="dtype mismatch"):
        _recomposition(changed_adapter)
    assert base.dtype == "float64"


def test_forged_or_cross_adapter_identity_fails_closed() -> None:
    adapter, identity = _identity()
    forged = replace(identity, target_signature="0" * 64)
    with pytest.raises(NeuTraCampaignError, match="target signature mismatch"):
        require_typed_neutra_target(forged, adapter=adapter)

    other = SyntheticGaussianCampaignAdapter()
    with pytest.raises(NeuTraCampaignError, match="another adapter"):
        require_typed_neutra_target(identity, adapter=other)

    caller_constructed = TypedNeuTraTargetIdentity(
        **{
            **identity.__dict__,
            "_issuer": object(),
        }
    )
    with pytest.raises(NeuTraCampaignError, match="invalid issuer"):
        require_typed_neutra_target(caller_constructed, adapter=adapter)


def test_post_issuance_hmc_callable_replacement_fails_closed(monkeypatch) -> None:
    adapter, identity = _identity()

    def changed_log_prob_and_grad(self, theta):
        return synthetic_gaussian_prior_value_score(theta)

    monkeypatch.setattr(
        SyntheticGaussianCampaignAdapter,
        "log_prob_and_grad",
        changed_log_prob_and_grad,
    )
    with pytest.raises(NeuTraCampaignError, match="callable changed"):
        require_typed_neutra_target(identity, adapter=adapter)


def test_post_issuance_status_callable_replacement_fails_closed(monkeypatch) -> None:
    adapter, identity = _identity()

    def changed_target_status_telemetry(self, theta):
        values = tf.convert_to_tensor(theta, tf.float64)
        leading = tf.shape(values)[:-1]
        return {
            "status_code": tf.ones(leading, tf.int32),
            "valid_pre_regularized_score": tf.zeros(leading, tf.bool),
        }

    monkeypatch.setattr(
        SyntheticGaussianCampaignAdapter,
        "target_status_telemetry",
        changed_target_status_telemetry,
    )
    with pytest.raises(NeuTraCampaignError, match="status callable changed"):
        require_typed_neutra_target(identity, adapter=adapter)


def test_p0_registry_and_scope_identity_cannot_issue_or_load() -> None:
    registry = load_validated_p0_registry(
        P0_ROOT / "target_registry.json",
        expected_file_sha256=P0_REGISTRY_SHA256,
    )
    adapter = SyntheticGaussianCampaignAdapter()
    row = registry["cells"][0]

    with pytest.raises(NeuTraCampaignError, match="not eligible"):
        issue_typed_neutra_target_identity(
            program_id=PROGRAM_ID,
            scope_kind="model_cell",
            scope_id=row["cell_id"],
            adapter=adapter,
            recomposition=_recomposition(adapter),
            registry_row=row,
            registry_artifact_sha256=P0_REGISTRY_SHA256,
        )

    _, identity = _identity(adapter)
    with pytest.raises(Exception, match="target_signature mismatch"):
        load_campaign_neutra_transport(
            identity=identity,
            adapter=adapter,
            payload=_affine_payload(identity, target_signature=row["scope_identity"]),
        )


def test_cross_target_transport_fails_closed() -> None:
    adapter, identity = _identity()
    changed_adapter = SyntheticGaussianCampaignAdapter(
        make_synthetic_gaussian_contract(prior_hash="sha256:other-prior")
    )
    changed = issue_typed_neutra_target_identity(
        program_id=PROGRAM_ID,
        scope_kind="synthetic_canary",
        scope_id="P1-SYNTHETIC-GAUSSIAN-OTHER",
        adapter=changed_adapter,
        recomposition=_recomposition(changed_adapter),
    )
    with pytest.raises(Exception, match="target_signature mismatch"):
        load_campaign_neutra_transport(
            identity=identity,
            adapter=adapter,
            payload=_affine_payload(identity, target_signature=changed.target_signature),
        )


def test_transformed_adapter_preserves_base_status_telemetry() -> None:
    adapter, identity = _identity()
    loaded = load_campaign_neutra_transport(
        identity=identity,
        adapter=adapter,
        payload=_affine_payload(identity),
    )
    transformed = campaign_fixed_transport_adapter(
        identity=identity,
        adapter=adapter,
        loaded_artifact=loaded,
    )
    status = transformed.target_status_telemetry(
        tf.constant([[-0.2, 0.1], [0.2, -0.1]], tf.float64)
    )
    assert bool(tf.reduce_all(tf.equal(status["status_code"], 0)).numpy()) is True
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy()) is True


def test_blocked_cell_cannot_advance_or_reject_recipe() -> None:
    registry = load_validated_p0_registry(
        P0_ROOT / "target_registry.json",
        expected_file_sha256=P0_REGISTRY_SHA256,
    )
    ledger = CampaignCellLedger(registry)
    cell = registry["cells"][0]["cell_id"]

    with pytest.raises(NeuTraCampaignError, match="target-repair phase"):
        ledger.transition(
            cell_id=cell,
            new_state="VALUE_SCORE_ADMITTED",
            evidence_path="not-used.json",
        )
    with pytest.raises(NeuTraCampaignError, match="cannot execute or reject"):
        ledger.record_recipe_rejection(
            cell_id=cell,
            family="plain_dense_iaf",
            recipe_id="unrun",
            evidence_path="not-used.json",
        )
    assert set(ledger.payload()["states"].values()) == {"TARGET_BLOCKED"}


def test_cell_rejection_requires_every_frozen_family() -> None:
    registry = {
        "cells": [{"cell_id": "TEST-CELL", "state": "COMPARATOR_ADMITTED"}]
    }
    ledger = CampaignCellLedger(registry)
    ledger.record_recipe_rejection(
        cell_id="TEST-CELL",
        family="plain_dense_iaf",
        recipe_id="recipe-a",
        evidence_path="a.json",
    )
    with pytest.raises(NeuTraCampaignError, match="missing=.*enhanced"):
        ledger.reject_cell_candidates(
            cell_id="TEST-CELL", evidence_path="premature.json"
        )
    ledger.record_recipe_rejection(
        cell_id="TEST-CELL",
        family="enhanced",
        recipe_id="recipe-b",
        evidence_path="b.json",
    )
    result = ledger.reject_cell_candidates(
        cell_id="TEST-CELL", evidence_path="complete.json"
    )
    assert result["to_state"] == "CELL_CANDIDATE_REJECTED"


def test_cell_ledger_persists_append_only_events(tmp_path) -> None:
    event_path = tmp_path / "events.jsonl"
    ledger = CampaignCellLedger(
        {"cells": [{"cell_id": "TEST-CELL", "state": "COMPARATOR_ADMITTED"}]},
        event_path=event_path,
    )
    ledger.record_recipe_rejection(
        cell_id="TEST-CELL",
        family="plain_dense_iaf",
        recipe_id="recipe-a",
        evidence_path="a.json",
    )
    rows = [json.loads(line) for line in event_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["event_type"] == "recipe_outcome"
    assert rows[0]["status"] == "RECIPE_REJECTED"


def test_cpu_sample_partition_is_worker_count_invariant() -> None:
    one = deterministic_cpu_sample_partitions(
        root_seed=(20260715, 301),
        sample_count=23,
        batch_size=6,
        worker_count=1,
        domain="p1-training-samples",
    )
    four = deterministic_cpu_sample_partitions(
        root_seed=(20260715, 301),
        sample_count=23,
        batch_size=6,
        worker_count=4,
        domain="p1-training-samples",
    )
    assert tuple((row.start_index, row.sample_count, row.seed) for row in one) == tuple(
        (row.start_index, row.sample_count, row.seed) for row in four
    )
    assert tuple(row.worker_index for row in four) == (0, 1, 2, 3)
    first = tf.concat(
        tuple(generate_cpu_sample_batch(row, dimension=3) for row in one), axis=0
    )
    second = tf.concat(
        tuple(generate_cpu_sample_batch(row, dimension=3) for row in four), axis=0
    )
    tf.debugging.assert_equal(first, second)
    assert first.shape == (23, 3)


def test_archive_keeps_warmup_and_retained_disjoint(tmp_path) -> None:
    adapter, identity = _identity()
    archive = SeparateCampaignArchive(
        output_root=tmp_path / "samples", identity=identity, adapter=adapter
    )
    values = tf.reshape(tf.range(24, dtype=tf.float64), (3, 4, 2))
    warmup = archive(
        stage="warmup",
        chunk_index=0,
        latent_samples=values,
        model_samples=values,
        seed=(1, 2),
        cumulative=False,
    )
    retained = archive(
        stage="retained",
        chunk_index=0,
        latent_samples=values + 1.0,
        model_samples=values + 1.0,
        seed=(3, 4),
        cumulative=False,
    )

    assert Path(warmup["latent_path"]).parent != Path(retained["latent_path"]).parent
    assert "/warmup/" in warmup["latent_path"]
    assert "/retained/" in retained["latent_path"]
    assert json.loads(
        (Path(warmup["latent_path"]).parent / "metadata.json").read_text()
    )["warmup_excluded_from_posterior"] is True
    with pytest.raises(NeuTraCampaignError, match="already exists"):
        archive(
            stage="warmup",
            chunk_index=0,
            latent_samples=values,
            model_samples=values,
            seed=(1, 2),
            cumulative=False,
        )


def test_campaign_training_requires_fresh_root_and_gpu_memory_policy(tmp_path) -> None:
    adapter, identity = _identity()
    gpu_config = PlainDenseIAFTrainingConfig(
        target_signature=identity.target_signature,
        dimension=2,
        affine_center=(0.0, 0.0),
        affine_factor=((1.0, 0.0), (0.0, 1.0)),
        output_dir=tmp_path / "gpu-training",
        steps=1,
        batch_size=2,
    )
    with pytest.raises(NeuTraCampaignError, match="memory-growth metadata"):
        train_campaign_neutra(
            identity=identity,
            adapter=adapter,
            config=gpu_config,
            freeze_transport_id="not-run",
        )

    existing = tmp_path / "existing"
    existing.mkdir()
    cpu_config = replace(
        gpu_config,
        output_dir=existing,
        device="/CPU:0",
        require_gpu=False,
    )
    with pytest.raises(NeuTraCampaignError, match="fresh output"):
        train_campaign_neutra(
            identity=identity,
            adapter=adapter,
            config=cpu_config,
            freeze_transport_id="not-run",
        )


def test_typed_sequential_hmc_writes_separate_archives(tmp_path, monkeypatch) -> None:
    import bayesfilter.inference.neutra_hmc as neutra_hmc

    adapter, identity = _identity()
    loaded = load_campaign_neutra_transport(
        identity=identity,
        adapter=adapter,
        payload=_affine_payload(identity),
    )
    monkeypatch.setattr(
        neutra_hmc,
        "rank_normalized_split_rhat_summary",
        lambda draws, *, rhat_max: {
            "passed": True,
            "rhat_threshold": float(rhat_max),
            "draw_count_per_chain": int(tf.shape(draws)[0]),
        },
    )
    archive = SeparateCampaignArchive(
        output_root=tmp_path / "sequential", identity=identity, adapter=adapter
    )
    result = run_campaign_neutra_hmc(
        identity=identity,
        adapter=adapter,
        loaded_artifact=loaded,
        initial_state=tf.constant(
            [[-0.3, 0.2], [-0.1, -0.2], [0.1, 0.2], [0.3, -0.2]], tf.float64
        ),
        parameter_names=("theta_0", "theta_1"),
        config=SequentialNeuTraHMCConfig(
            step_size=0.2,
            num_leapfrog_steps=2,
            warmup_seed=(20260715, 601),
            retained_seed=(20260715, 701),
            warmup_chunk_results=4,
            warmup_min_results=4,
            warmup_check_window_results=4,
            warmup_max_results=4,
            retained_chunk_results=4,
            retained_min_results=4,
            retained_max_results=4,
        ),
        archive_callback=archive,
    )

    assert result["passed"] is True
    assert result["warmup_results_per_chain"] == 4
    assert result["retained_results_per_chain"] == 4
    assert result["warmup_excluded_from_posterior"] is True
    assert (tmp_path / "sequential" / "warmup" / "cumulative").is_dir()
    assert (tmp_path / "sequential" / "retained" / "cumulative").is_dir()

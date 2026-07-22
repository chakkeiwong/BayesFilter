from __future__ import annotations

import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.inference.neutra_campaign import (
    admit_independent_posterior_recomposition,
    issue_typed_neutra_target_identity,
    load_campaign_neutra_transport,
    run_campaign_plain_hmc,
    train_campaign_neutra,
)
from bayesfilter.inference.neutra_hmc import BatchedHMCConfig
from bayesfilter.inference.neutra_training import PlainDenseIAFTrainingConfig
from bayesfilter.testing.multimodel_neutra_p1_canary_tf import (
    SYNTHETIC_CANARY_SCOPE,
    SyntheticGaussianCampaignAdapter,
    synthetic_exponential_chart_jacobian_value_score,
    synthetic_gaussian_likelihood_value_score,
    synthetic_gaussian_prior_value_score,
)


def _identity():
    adapter = SyntheticGaussianCampaignAdapter()
    recomposition = admit_independent_posterior_recomposition(
        adapter=adapter,
        points=tf.constant([[-0.5, 0.25], [0.0, 0.0], [0.5, -0.25]], tf.float64),
        prior_value_score_fn=synthetic_gaussian_prior_value_score,
        likelihood_value_score_fn=synthetic_gaussian_likelihood_value_score,
        jacobian_value_score_fn=synthetic_exponential_chart_jacobian_value_score,
    )
    identity = issue_typed_neutra_target_identity(
        program_id="multimodel-neutra-filter-posterior-20260715",
        scope_kind="synthetic_canary",
        scope_id=SYNTHETIC_CANARY_SCOPE,
        adapter=adapter,
        recomposition=recomposition,
    )
    return adapter, identity


def test_cpu_hidden_campaign_training_freeze_replay_and_hmc_smoke(tmp_path) -> None:
    adapter, identity = _identity()
    config = PlainDenseIAFTrainingConfig(
        target_signature=identity.target_signature,
        dimension=2,
        affine_center=(0.0, 0.0),
        affine_factor=((1.0, 0.0), (0.0, 1.0)),
        output_dir=tmp_path / "training",
        seed=(20260715, 401),
        hidden_layers=(4,),
        stage_count=1,
        steps=2,
        batch_size=8,
        checkpoint_every=1,
        heartbeat_every=1,
        device="/CPU:0",
        require_gpu=False,
    )
    trained = train_campaign_neutra(
        identity=identity,
        adapter=adapter,
        config=config,
        freeze_transport_id="p1-cpu-test-transport",
    )
    payload = json.loads(trained.frozen_payload_path.read_text(encoding="utf-8"))
    loaded = load_campaign_neutra_transport(
        identity=identity, adapter=adapter, payload=payload
    )
    smoke = run_campaign_plain_hmc(
        identity=identity,
        adapter=adapter,
        initial_state=tf.constant(
            [[-0.4, 0.2], [-0.2, -0.1], [0.2, 0.1], [0.4, -0.2]], tf.float64
        ),
        config=BatchedHMCConfig(
            num_results=8,
            num_burnin_steps=4,
            step_size=0.25,
            num_leapfrog_steps=2,
            seed=(20260715, 402),
        ),
    )

    assert trained.completed_steps == 2
    assert loaded.manifest.target_signature == identity.target_signature
    assert smoke["diagnostics"]["health_passed"] is True
    assert smoke["samples"].shape == (8, 4, 2)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"

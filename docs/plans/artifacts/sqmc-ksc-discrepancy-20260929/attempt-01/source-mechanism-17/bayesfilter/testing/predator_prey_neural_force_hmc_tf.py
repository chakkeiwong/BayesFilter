"""P4 corrected neural-force HMC campaign for predator-prey UKF and SGQF."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tensorflow as tf

from bayesfilter.inference.neural_force_campaign import (
    bind_transformed_neural_force_target,
    generate_neural_force_supervision,
    validate_value_only_endpoint_parity,
)
from bayesfilter.inference.neural_force_hmc import FrozenPositionOnlyForce
from bayesfilter.inference.neural_force_training import (
    ScalarResidualForceTrainingConfig,
    train_scalar_residual_force,
)
from bayesfilter.testing import lgssm_neural_force_hmc_pilot_tf as campaign


CELLS = ("PP-UKF", "PP-SGQF")
DIMENSION = 6
PARAMETER_NAMES = ("r", "K", "a", "s", "u", "v")
ARCHIVE_ROOT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4"
)
TRANSPORT_ROOTS = {
    cell: ARCHIVE_ROOT / cell / "training/final/wide_lr5e3/attempt-01"
    for cell in CELLS
}
LATENT_ROOTS = {
    cell: ARCHIVE_ROOT
    / cell
    / "neutra-confirmation/attempt-02/samples/retained/cumulative"
    for cell in CELLS
}
TRAIN_ROWS = 2048
HELDOUT_ROWS = 1024


class PredatorPreyNeuralForceError(RuntimeError):
    """Raised when a P4 identity, endpoint, or evidence gate fails closed."""


def load_context(cell: str) -> Mapping[str, Any]:
    """Replay one target identity, transport, endpoint scalar, and comparators."""

    from docs.benchmarks import (
        run_multimodel_neutra_p4_predator_prey_neutra_confirmation as confirmation,
    )
    from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_training as training
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.ssm import stable_ssm_target_signature
    from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
        PP_TRUTH_PHYSICAL,
        generate_frozen_predator_prey_dataset_tf,
        pp_ukf_posterior_value_only,
    )

    cell = _cell(cell)
    identity_root = training.base.IDENTITY_ROOTS[cell]
    identity_reference = training.base._verify_source_root(identity_root)
    identity_payload = training._read_mapping(identity_root / "target_identity.json")
    source = training._read_mapping(identity_root / "result.json")
    _states, observations = generate_frozen_predator_prey_dataset_tf()
    if cell == "PP-UKF":
        from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
            make_predator_prey_ukf_neutra_adapter,
        )

        adapter = make_predator_prey_ukf_neutra_adapter(observations=observations)

        def raw_value(raw: tf.Tensor) -> tf.Tensor:
            return pp_ukf_posterior_value_only(
                raw, observations=adapter.observations
            )
    else:
        from bayesfilter.testing.predator_prey_sgqf_neutra_target_tf import (
            make_predator_prey_sgqf_neutra_adapter,
            pp_sgqf_posterior_value_only,
        )

        if source.get("selected_level") != 2:
            raise PredatorPreyNeuralForceError("historical SGQF level drift")
        adapter = make_predator_prey_sgqf_neutra_adapter(
            sparse_level=2, observations=observations
        )

        def raw_value(raw: tf.Tensor) -> tf.Tensor:
            return pp_sgqf_posterior_value_only(
                raw,
                observations=adapter.observations,
                nodes=adapter.nodes,
                weights=adapter.weights,
            )

    if (
        stable_ssm_target_signature(adapter.contract)
        != identity_payload["mathematical_target_signature"]
        or adapter.adapter_signature() != identity_payload["adapter_signature"]
    ):
        raise PredatorPreyNeuralForceError(
            "mathematical target or adapter signature changed"
        )
    target_signature = str(identity_payload["target_signature"])
    transport_root = TRANSPORT_ROOTS[cell]
    expected_training_hash = confirmation.EXPECTED_TRAINING_RESULT_SHA256[cell]
    training_reference = training._verify_result_root(
        transport_root, expected_training_hash, require_passed=True
    )
    training_result = training._read_mapping(transport_root / "result.json")
    payload_path = Path(str(training_result["payload"]["path"]))
    if training._file_sha256(payload_path) != training_result["payload"]["file_sha256"]:
        raise PredatorPreyNeuralForceError("frozen transport payload hash drift")
    loaded = load_frozen_neutra_artifact(
        training._read_mapping(payload_path),
        expected_target_signature=target_signature,
    )
    if (
        loaded.artifact_signature != training_result["transport_artifact_signature"]
        or loaded.manifest.transport_hash != training_result["transport_hash"]
    ):
        raise PredatorPreyNeuralForceError("frozen transport binding drift")
    transformed = FixedTransportValueScoreAdapter(
        base_adapter=adapter,
        transport=loaded.transport,
        target_scope=f"{cell}:corrected-neural-force-p4",
        evidence_path=__file__,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=False,
        require_batch_native=True,
    )
    identity = SimpleNamespace(
        target_signature=target_signature,
        mathematical_target_signature=identity_payload["mathematical_target_signature"],
        parameter_dim=DIMENSION,
    )

    def endpoint_potential(position: tf.Tensor) -> tf.Tensor:
        z = tf.convert_to_tensor(position, tf.float64)
        raw = loaded.transport.forward_batch(z)
        return -(
            raw_value(raw) + loaded.transport.log_abs_det_jacobian_batch(z)
        )

    binding = bind_transformed_neural_force_target(
        adapter=transformed,
        endpoint_potential_function=endpoint_potential,
        target_signature=identity.target_signature,
        transport_signature=loaded.manifest.transport_hash,
        dimension=DIMENSION,
    )
    comparator, comparator_reference, comparator_samples = (
        confirmation._load_comparator(
            tf=tf, cell=cell, expected_target_signature=identity.target_signature
        )
    )
    latent, latent_reference = _load_latent_archive(
        cell, expected_target_signature=identity.target_signature
    )
    return {
        "cell": cell,
        "adapter": adapter,
        "identity": identity,
        "identity_reference": identity_reference,
        "training_result": training_result,
        "training_reference": training_reference,
        "loaded": loaded,
        "transformed": transformed,
        "binding": binding,
        "comparator": comparator,
        "comparator_reference": comparator_reference,
        "comparator_samples": comparator_samples,
        "latent": latent,
        "latent_reference": latent_reference,
        "truth_physical": PP_TRUTH_PHYSICAL,
    }


def prepare_supervision(context: Mapping[str, Any]) -> Mapping[str, Any]:
    flat = tf.reshape(context["latent"], [-1, DIMENSION])
    required = TRAIN_ROWS + HELDOUT_ROWS
    if int(flat.shape[0]) < required:
        raise PredatorPreyNeuralForceError("preserved latent archive is too short")
    train_positions = flat[:TRAIN_ROWS]
    heldout_positions = flat[TRAIN_ROWS:required]
    parity = validate_value_only_endpoint_parity(
        context["binding"],
        tf.concat((train_positions[:4], heldout_positions[:4]), axis=0),
        absolute_tolerance=2.0e-8,
    )
    return {
        "train": generate_neural_force_supervision(
            context["binding"], train_positions
        ),
        "heldout": generate_neural_force_supervision(
            context["binding"], heldout_positions
        ),
        "parity": parity,
        "train_rows": TRAIN_ROWS,
        "heldout_rows": HELDOUT_ROWS,
        "disjoint_archive_slices": True,
    }


def train_target_specific_grid(
    context: Mapping[str, Any], supervision: Mapping[str, Any], output_root: Path
) -> Mapping[str, Any]:
    """Screen small target-local recipes, then train a fresh selected force."""

    rows = []
    recipes = (
        ((12, 12), 1.0e-3, 128),
        ((12, 12), 5.0e-3, 128),
        ((24, 24), 1.0e-3, 256),
        ((24, 24), 5.0e-3, 256),
    )
    for index, (layers, learning_rate, batch_size) in enumerate(recipes):
        recipe_id = f"w{layers[0]}_lr{learning_rate:g}_b{batch_size}"
        config = ScalarResidualForceTrainingConfig(
            target_signature=context["identity"].target_signature,
            transport_signature=context["loaded"].manifest.transport_hash,
            dimension=DIMENSION,
            hidden_layers=layers,
            output_dir=output_root / "screen" / recipe_id,
            seed=(20260717, 61000 + 1000 * CELLS.index(context["cell"]) + index),
            steps=500,
            batch_size=batch_size,
            learning_rate=learning_rate,
            heartbeat_every=100,
            device="/GPU:0",
            require_gpu=True,
        )
        result = train_scalar_residual_force(
            train_positions=supervision["train"].positions,
            train_potentials=supervision["train"].potentials,
            train_forces=supervision["train"].forces,
            heldout_positions=supervision["heldout"].positions,
            heldout_potentials=supervision["heldout"].potentials,
            heldout_forces=supervision["heldout"].forces,
            config=config,
        )
        rows.append(
            {
                "recipe_id": recipe_id,
                "layers": layers,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "heldout": dict(result.metrics["heldout"]),
                "elapsed_seconds": result.runtime_metadata["elapsed_seconds"],
                "result_path": str(result.result_path),
            }
        )
    viable = tuple(
        row for row in rows if row["heldout"]["predictions_all_finite"] is True
    )
    if not viable:
        raise PredatorPreyNeuralForceError("no finite target-specific force recipe")
    selected = min(
        viable,
        key=lambda row: (
            row["heldout"]["standardized_force_rmse"],
            row["heldout"]["centered_standardized_potential_rmse"],
            row["recipe_id"],
        ),
    )
    final_config = ScalarResidualForceTrainingConfig(
        target_signature=context["identity"].target_signature,
        transport_signature=context["loaded"].manifest.transport_hash,
        dimension=DIMENSION,
        hidden_layers=tuple(selected["layers"]),
        output_dir=output_root / "final" / selected["recipe_id"],
        seed=(20260717, 62000 + 1000 * CELLS.index(context["cell"])),
        steps=5000,
        batch_size=int(selected["batch_size"]),
        learning_rate=float(selected["learning_rate"]),
        heartbeat_every=250,
        device="/GPU:0",
        require_gpu=True,
    )
    final = train_scalar_residual_force(
        train_positions=supervision["train"].positions,
        train_potentials=supervision["train"].potentials,
        train_forces=supervision["train"].forces,
        heldout_positions=supervision["heldout"].positions,
        heldout_potentials=supervision["heldout"].potentials,
        heldout_forces=supervision["heldout"].forces,
        config=final_config,
    )
    return {"screen": tuple(rows), "selected": selected, "final": final}


def run_cell(context: Mapping[str, Any], output_root: Path) -> Mapping[str, Any]:
    from docs.benchmarks import (
        run_multimodel_neutra_p4_predator_prey_neutra_confirmation as confirmation,
    )

    started = time.monotonic()
    supervision = prepare_supervision(context)
    training = train_target_specific_grid(
        context, supervision, output_root / "force-training"
    )
    target = context["binding"].hmc_target()
    initial = tf.gather(
        supervision["heldout"].positions, tf.constant((0, 17, 33, 49), tf.int32)
    )

    def true_force(position: tf.Tensor) -> tf.Tensor:
        _potential, force = context["binding"].potential_and_force(position)
        return force

    arms = {
        "zero_residual": FrozenPositionOnlyForce(
            lambda position: position,
            f"{context['cell']}-zero-residual-gaussian-force",
        ),
        "learned_residual": training["final"].frozen.hmc_force(),
    }
    tuning = {}
    runs = {}
    for index, (arm_id, force) in enumerate(arms.items()):
        tuning[arm_id] = campaign.tune_force(
            force=force,
            target=target,
            initial_position=initial,
            transform=context["loaded"].transport,
            step_sizes=(0.2, 0.4, 0.6, 0.8),
            leapfrog_steps=(6, 10),
            seed_offset=63000 + CELLS.index(context["cell"]) * 1000 + index * 100,
        )
        selected = tuning[arm_id]["selected"]
        runs[arm_id] = campaign.run_sequential_arm(
            arm_id=arm_id,
            force=force,
            target=target,
            initial_position=initial,
            transform=context["loaded"].transport,
            parameter_names=(
                "r_source_probit",
                "K_source_probit",
                "a_source_probit",
                "s_source_probit",
                "u_source_probit",
                "v_source_probit",
            ),
            step_size=selected.step_size,
            num_leapfrog_steps=selected.num_leapfrog_steps,
            output_root=output_root / "sampling",
            seed_base=65000 + CELLS.index(context["cell"]) * 5000 + index * 1000,
        )

    arm_decisions = {}
    for arm_id, run in runs.items():
        source_samples = run["private_retained_raw"]
        if run["passed"]:
            from bayesfilter.testing.predator_prey_ukf_neutra_target_tf import (
                source_chart_physical_parameters,
            )

            shape = tf.shape(source_samples)
            physical_flat, _ = source_chart_physical_parameters(
                tf.reshape(source_samples, [-1, DIMENSION])
            )
            physical = tf.reshape(physical_flat, shape)
            truth_tail = campaign.truth_tail_summary(
                physical, context["truth_physical"], PARAMETER_NAMES
            )
            agreement = confirmation._physical_mean_agreement(
                tf=tf,
                tfp=__import__("tensorflow_probability"),
                candidate_source_samples=source_samples,
                comparator_source_samples=context["comparator_samples"],
            )
        else:
            truth_tail = None
            agreement = None
        passed = bool(
            run["passed"]
            and truth_tail is not None
            and truth_tail["passed"]
            and agreement is not None
            and agreement["passed"]
        )
        arm_decisions[arm_id] = {
            "passed": passed,
            "truth_tail": truth_tail,
            "plain_hmc_physical_mean_agreement": agreement,
        }
    representative = (
        "learned_residual"
        if arm_decisions["learned_residual"]["passed"]
        else "zero_residual"
        if arm_decisions["zero_residual"]["passed"]
        else None
    )
    return {
        "schema": "bayesfilter.predator_prey_neural_force_hmc_p4_result.v1",
        "cell": context["cell"],
        "passed": representative is not None,
        "decision": (
            "HNN_VALIDITY_CONFIRMED_ONE_SEED"
            if arm_decisions["learned_residual"]["passed"]
            else "ZERO_RESIDUAL_CORRECTED_FORCE_VALID_ONLY"
            if representative is not None
            else "P4_CANDIDATES_NOT_CONFIRMED"
        ),
        "representative_arm": representative,
        "target_signature": context["identity"].target_signature,
        "transport_signature": context["loaded"].manifest.transport_hash,
        "value_only_endpoint_parity": supervision["parity"],
        "supervision": {
            "train_rows": supervision["train_rows"],
            "heldout_rows": supervision["heldout_rows"],
            "disjoint_archive_slices": True,
            "source_reference": context["latent_reference"],
        },
        "training": {
            "screen": training["screen"],
            "selected": training["selected"],
            "final_result_path": str(training["final"].result_path),
            "final_metrics": training["final"].metrics,
        },
        "tuning": {
            name: {
                "selected": {
                    "candidate_id": value["selected"].candidate_id,
                    "step_size": value["selected"].step_size,
                    "num_leapfrog_steps": value["selected"].num_leapfrog_steps,
                },
                "rows": value["rows"],
            }
            for name, value in tuning.items()
        },
        "runs": {name: campaign.json_ready(value) for name, value in runs.items()},
        "arm_decisions": campaign.json_ready(arm_decisions),
        "plain_hmc_reference": context["comparator_reference"],
        "historical_true_gradient_reference": context["training_reference"],
        "elapsed_seconds": time.monotonic() - started,
        "statistically_supported_ranking": False,
        "descriptive_only": (
            "acceptance, runtime, loss, RMSE, and between-arm differences",
        ),
        "nonclaims": (
            "one fixture and one deterministic filter posterior only",
            "no filter ranking or latent-model exactness claim",
            "no stochastic superiority claim",
        ),
    }


def run_smoke(context: Mapping[str, Any]) -> Mapping[str, Any]:
    from bayesfilter.inference.neural_force_hmc import (
        NeuralForceHMCConfig,
        sample_neural_force_hmc,
    )

    flat = tf.reshape(context["latent"], [-1, DIMENSION])
    points = tf.gather(flat, tf.constant((0, 17, 33, 49), tf.int32))
    parity = validate_value_only_endpoint_parity(
        context["binding"], points, absolute_tolerance=2.0e-8
    )
    target = context["binding"].hmc_target()
    force = FrozenPositionOnlyForce(
        lambda position: position, f"{context['cell']}-zero-residual-smoke"
    )
    config = NeuralForceHMCConfig(
        step_size=0.2,
        num_leapfrog_steps=2,
        inverse_mass_diagonal=(1.0,) * DIMENSION,
        dtype="float64",
    )

    @tf.function(jit_compile=True)
    def compiled(position: tf.Tensor, potential: tf.Tensor):
        return sample_neural_force_hmc(
            position,
            potential,
            force,
            target,
            config,
            num_warmup=0,
            num_results=4,
            seed=tf.constant((20260717, 60999), tf.int32),
        )

    chain = compiled(points, target.function(points))
    health = campaign.chain_health(chain)
    return {
        "schema": "bayesfilter.predator_prey_neural_force_hmc_p4_smoke.v1",
        "cell": context["cell"],
        "passed": bool(parity["passed"] and health["passed"]),
        "value_only_endpoint_parity": parity,
        "health": health,
        "acceptance_rate": float(
            tf.reduce_mean(tf.cast(chain.accepted, tf.float64)).numpy()
        ),
    }


def _load_latent_archive(
    cell: str, *, expected_target_signature: str
) -> tuple[tf.Tensor, Mapping[str, Any]]:
    root = LATENT_ROOTS[cell]
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata.get("target_signature") != expected_target_signature
        or metadata.get("sample_shape") != [4000, 4, 6]
        or metadata.get("stage") != "retained"
    ):
        raise PredatorPreyNeuralForceError("latent supervision archive binding failed")
    tensor_path = Path(metadata["latent_path"])
    value = tf.io.parse_tensor(tf.io.read_file(str(tensor_path)), tf.float64)
    value = tf.ensure_shape(value, (4000, 4, 6))
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        raise PredatorPreyNeuralForceError("latent supervision archive is nonfinite")
    return value, {
        "metadata_path": str(metadata_path),
        "tensor_path": str(tensor_path),
        "sample_shape": (4000, 4, 6),
        "target_signature": expected_target_signature,
    }


def _cell(value: str) -> str:
    cell = str(value).upper()
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {CELLS}")
    return cell


__all__ = [
    "CELLS",
    "load_context",
    "prepare_supervision",
    "run_cell",
    "run_smoke",
]

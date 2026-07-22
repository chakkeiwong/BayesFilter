"""P5 corrected neural-force HMC campaigns for SIR-SGQF and structural UKF."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tensorflow as tf

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
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
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.testing import lgssm_neural_force_hmc_pilot_tf as campaign
from bayesfilter.ssm import stable_ssm_target_signature


CELLS = ("SIR-SGQF", "STR-UKF")


class P5NeuralForceError(RuntimeError):
    pass


def load_context(cell: str) -> Mapping[str, Any]:
    cell = str(cell).upper()
    if cell == "SIR-SGQF":
        return _load_sir()
    if cell == "STR-UKF":
        return _load_structural()
    raise ValueError(f"cell must be one of {CELLS}")


def _load_sir() -> Mapping[str, Any]:
    from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_training as common
    from docs.benchmarks import run_multimodel_neutra_p6_sir_sgqf_hmc as hmc
    from docs.benchmarks import run_multimodel_neutra_p6_sir_sgqf_neutra_confirmation as confirmation
    from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
        generate_frozen_sir_dataset_tf,
        make_sir_sgqf_neutra_adapter,
        sir_sgqf_posterior_value_only,
    )

    identity_root = hmc.IDENTITY_ROOT
    identity_reference = hmc._verify_root(identity_root, hmc.IDENTITY_RESULT_SHA256)
    identity_payload = common._read_mapping(identity_root / "target_identity.json")
    _states, observations, _all = generate_frozen_sir_dataset_tf()
    adapter = make_sir_sgqf_neutra_adapter(observations=observations)
    _require_identity(adapter, identity_payload)
    target_signature = identity_payload["target_signature"]
    training_root = confirmation.TRAINING_ROOT
    training_reference = common._verify_result_root(
        training_root, confirmation.TRAINING_RESULT_SHA256, require_passed=True
    )
    training_result = common._read_mapping(training_root / "result.json")
    loaded = _load_transport(common, training_result, target_signature)
    comparator, comparator_reference, comparator_samples = confirmation._load_comparator(
        tf=tf, expected_target_signature=target_signature
    )
    latent_root = Path(
        "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/"
        "phase-p6/SIR-SGQF/neutra-confirmation/attempt-01/samples/retained/cumulative"
    )
    latent, latent_reference = _load_single_latent(
        latent_root, target_signature, (4000, 4, 3)
    )

    def raw_value(raw: tf.Tensor) -> tf.Tensor:
        return sir_sgqf_posterior_value_only(
            raw, observations=adapter.observations, nodes=adapter.nodes,
            weights=adapter.weights
        )

    return _finish_context(
        cell="SIR-SGQF", dimension=3, adapter=adapter,
        target_signature=target_signature, loaded=loaded, raw_value=raw_value,
        latent=latent, latent_reference=latent_reference,
        identity_reference=identity_reference, training_reference=training_reference,
        comparator=comparator, comparator_reference=comparator_reference,
        comparator_samples=comparator_samples,
        source_names=("log_kappa_scale", "log_nu_scale", "log_observation_noise_scale"),
        physical_names=("kappa", "nu", "observation_noise_sd"),
        truth_physical=tf.constant((0.1, 18.0, 10.0), tf.float64),
        step_sizes=(0.2, 0.4, 0.6, 0.8), leapfrog_steps=(6, 10),
    )


def _load_structural() -> Mapping[str, Any]:
    from docs.benchmarks import run_multimodel_neutra_p4_predator_prey_training as common
    from docs.benchmarks import run_structural_ukf_neutra_training as training
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_TRUTH_PHYSICAL,
        generate_frozen_structural_dataset_tf,
        make_structural_ukf_neutra_adapter,
        structural_ukf_posterior_value_only,
    )

    identity_root = training.IDENTITY_ROOT
    identity_reference = common._verify_result_root(
        identity_root, training.IDENTITY_RESULT_SHA256, require_passed=True
    )
    identity_payload = common._read_mapping(identity_root / "target_identity.json")
    _states, observations = generate_frozen_structural_dataset_tf()
    adapter = make_structural_ukf_neutra_adapter(observations=observations)
    _require_identity(adapter, identity_payload)
    target_signature = identity_payload["target_signature"]
    training_root = Path(
        "docs/plans/artifacts/structural-ukf-neutra-truth-tail-20260717/"
        "training/final/dim6_lr5e3/attempt-02/segment-5000"
    )
    training_result = common._read_mapping(training_root / "result.json")
    training_reference = {
        "root": str(training_root),
        "result_sha256": common._file_sha256(training_root / "result.json"),
    }
    loaded = _load_transport(common, training_result, target_signature)
    roots = tuple(
        Path(
            "docs/plans/artifacts/structural-ukf-neutra-truth-tail-20260717/"
            f"confirmation/attempt-03/samples/retained/chunk-{index:04d}"
        )
        for index in range(2)
    )
    chunks = []
    references = []
    for root in roots:
        value, reference = _load_single_latent(root, target_signature, (2000, 4, 5))
        chunks.append(value)
        references.append(reference)
    latent = tf.concat(chunks, axis=0)

    def raw_value(raw: tf.Tensor) -> tf.Tensor:
        return structural_ukf_posterior_value_only(
            raw, observations=adapter.observations
        )

    context = _finish_context(
        cell="STR-UKF", dimension=5, adapter=adapter,
        target_signature=target_signature, loaded=loaded, raw_value=raw_value,
        latent=latent, latent_reference={"chunks": references},
        identity_reference=identity_reference, training_reference=training_reference,
        comparator=None,
        comparator_reference={
            "status": "BLOCK_STR_UKF_SOURCE_GEOMETRY",
            "path": "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p5/STR-UKF/plain-hmc/attempt-02/result.json",
            "historical_owner_adjudication": "posthoc_bulk_ess_900_not_used_prospectively",
        },
        comparator_samples=None,
        source_names=("rho_source_probit", "sigma_source_probit", "phi_source_probit", "gamma_source_probit", "R_source_probit"),
        physical_names=("rho", "sigma", "phi", "gamma", "R"),
        truth_physical=STRUCTURAL_TRUTH_PHYSICAL,
        step_sizes=(0.025, 0.05, 0.1, 0.2), leapfrog_steps=(8, 12),
    )
    return {**context, "structural_invariant_required": True}


def _finish_context(**values: Any) -> Mapping[str, Any]:
    loaded = values["loaded"]
    adapter = values["adapter"]
    dimension = values["dimension"]
    raw_value = values.pop("raw_value")

    def endpoint(position: tf.Tensor) -> tf.Tensor:
        z = tf.convert_to_tensor(position, tf.float64)
        raw = loaded.transport.forward_batch(z)
        return -(raw_value(raw) + loaded.transport.log_abs_det_jacobian_batch(z))

    transformed = FixedTransportValueScoreAdapter(
        base_adapter=adapter, transport=loaded.transport,
        target_scope=f"{values['cell']}:corrected-neural-force-p5",
        evidence_path=__file__, xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=False, require_batch_native=True,
    )
    binding = bind_transformed_neural_force_target(
        adapter=transformed, endpoint_potential_function=endpoint,
        target_signature=values["target_signature"],
        transport_signature=loaded.manifest.transport_hash,
        dimension=dimension,
    )
    return {**values, "transformed": transformed, "binding": binding}


def _require_identity(adapter: Any, payload: Mapping[str, Any]) -> None:
    if (
        stable_ssm_target_signature(adapter.contract) != payload["mathematical_target_signature"]
        or adapter.adapter_signature() != payload["adapter_signature"]
    ):
        raise P5NeuralForceError("mathematical target or adapter signature drift")


def _load_transport(common: Any, result: Mapping[str, Any], target_signature: str) -> Any:
    path = Path(result["payload"]["path"])
    if common._file_sha256(path) != result["payload"]["file_sha256"]:
        raise P5NeuralForceError("frozen transport payload hash drift")
    loaded = load_frozen_neutra_artifact(
        common._read_mapping(path), expected_target_signature=target_signature
    )
    if (
        loaded.artifact_signature != result["transport_artifact_signature"]
        or loaded.manifest.transport_hash != result["transport_hash"]
    ):
        raise P5NeuralForceError("frozen transport identity drift")
    return loaded


def _load_single_latent(root: Path, signature: str, shape: tuple[int, ...]):
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if metadata["target_signature"] != signature or tuple(metadata["sample_shape"]) != shape:
        raise P5NeuralForceError("latent archive binding drift")
    path = Path(metadata["latent_path"])
    value = tf.ensure_shape(
        tf.io.parse_tensor(tf.io.read_file(str(path)), tf.float64), shape
    )
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        raise P5NeuralForceError("latent archive nonfinite")
    return value, {"metadata_path": str(root / "metadata.json"), "latent_path": str(path), "shape": shape}


def prepare_supervision(context: Mapping[str, Any]):
    dimension = context["dimension"]
    flat = tf.reshape(context["latent"], [-1, dimension])
    train_positions = flat[:2048]
    heldout_positions = flat[2048:3072]
    parity = validate_value_only_endpoint_parity(
        context["binding"], tf.concat((train_positions[:4], heldout_positions[:4]), 0),
        absolute_tolerance=2e-8,
    )
    train = generate_neural_force_supervision(context["binding"], train_positions)
    heldout = generate_neural_force_supervision(context["binding"], heldout_positions)
    return {"train": train, "heldout": heldout, "parity": parity}


def train_grid(context: Mapping[str, Any], supervision: Mapping[str, Any], root: Path):
    d = context["dimension"]
    recipes = (
        ((2*d, 2*d), 1e-3, 128), ((2*d, 2*d), 5e-3, 128),
        ((4*d, 4*d), 1e-3, 256), ((4*d, 4*d), 5e-3, 256),
    )
    rows = []
    for index, (layers, lr, batch) in enumerate(recipes):
        recipe = f"w{layers[0]}_lr{lr:g}_b{batch}"
        config = ScalarResidualForceTrainingConfig(
            target_signature=context["target_signature"],
            transport_signature=context["loaded"].manifest.transport_hash,
            dimension=d, hidden_layers=layers, output_dir=root/"screen"/recipe,
            seed=(20260718, 71000 + CELLS.index(context["cell"])*1000 + index),
            steps=500, batch_size=batch, learning_rate=lr,
            heartbeat_every=100, device="/GPU:0", require_gpu=True,
        )
        result = train_scalar_residual_force(
            train_positions=supervision["train"].positions,
            train_potentials=supervision["train"].potentials,
            train_forces=supervision["train"].forces,
            heldout_positions=supervision["heldout"].positions,
            heldout_potentials=supervision["heldout"].potentials,
            heldout_forces=supervision["heldout"].forces, config=config,
        )
        rows.append({"recipe":recipe,"layers":layers,"learning_rate":lr,"batch_size":batch,"heldout":dict(result.metrics["heldout"]),"elapsed_seconds":result.runtime_metadata["elapsed_seconds"]})
    selected = min(
        (row for row in rows if row["heldout"]["predictions_all_finite"]),
        key=lambda row:(row["heldout"]["standardized_force_rmse"],row["heldout"]["centered_standardized_potential_rmse"],row["recipe"]),
    )
    config = ScalarResidualForceTrainingConfig(
        target_signature=context["target_signature"],
        transport_signature=context["loaded"].manifest.transport_hash,
        dimension=d, hidden_layers=tuple(selected["layers"]),
        output_dir=root/"final"/selected["recipe"],
        seed=(20260718, 72000 + CELLS.index(context["cell"])*1000),
        steps=5000, batch_size=selected["batch_size"],
        learning_rate=selected["learning_rate"], heartbeat_every=250,
        device="/GPU:0", require_gpu=True,
    )
    final = train_scalar_residual_force(
        train_positions=supervision["train"].positions,
        train_potentials=supervision["train"].potentials,
        train_forces=supervision["train"].forces,
        heldout_positions=supervision["heldout"].positions,
        heldout_potentials=supervision["heldout"].potentials,
        heldout_forces=supervision["heldout"].forces, config=config,
    )
    return {"screen":rows,"selected":selected,"final":final}


def run_cell(context: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    import tensorflow_probability as tfp
    supervision = prepare_supervision(context)
    training = train_grid(context, supervision, root/"force-training")
    initial = tf.gather(supervision["heldout"].positions, (0,17,33,49))
    target = context["binding"].hmc_target()
    arms = {
        "zero_residual": FrozenPositionOnlyForce(lambda x:x, f"{context['cell']}-zero"),
        "learned_residual": training["final"].frozen.hmc_force(),
    }
    tuning, runs = {}, {}
    for index,(name,force) in enumerate(arms.items()):
        tuning[name] = campaign.tune_force(
            force=force,target=target,initial_position=initial,
            transform=context["loaded"].transport,
            step_sizes=context["step_sizes"],leapfrog_steps=context["leapfrog_steps"],
            seed_offset=73000+CELLS.index(context["cell"])*2000+index*200,
        )
        selected=tuning[name]["selected"]
        runs[name]=campaign.run_sequential_arm(
            arm_id=name,force=force,target=target,initial_position=initial,
            transform=context["loaded"].transport,
            parameter_names=context["source_names"],step_size=selected.step_size,
            num_leapfrog_steps=selected.num_leapfrog_steps,
            output_root=root/"sampling",
            seed_base=75000+CELLS.index(context["cell"])*5000+index*1000,
        )
    decisions={}
    for name,run in runs.items():
        source=run["private_retained_raw"]
        if run["passed"]:
            physical=_physical(context,source)
            truth=campaign.truth_tail_summary(physical,context["truth_physical"],context["physical_names"])
            agreement=_agreement(context,source,tfp)
        else:
            truth=agreement=None
        passed=bool(run["passed"] and truth and truth["passed"] and (agreement is None or agreement["passed"]))
        decisions[name]={"passed":passed,"truth_tail":truth,"agreement":agreement}
    representative="learned_residual" if decisions["learned_residual"]["passed"] else "zero_residual" if decisions["zero_residual"]["passed"] else None
    return {
        "schema":"bayesfilter.sir_structural_neural_force_hmc_p5_result.v1",
        "cell":context["cell"],"passed":representative is not None,
        "decision":"HNN_VALIDITY_CONFIRMED_ONE_SEED" if decisions["learned_residual"]["passed"] else "ZERO_RESIDUAL_CORRECTED_FORCE_VALID_ONLY" if representative else "P5_CANDIDATES_NOT_CONFIRMED",
        "representative_arm":representative,"target_signature":context["target_signature"],
        "transport_signature":context["loaded"].manifest.transport_hash,
        "value_only_endpoint_parity":supervision["parity"],
        "structural_invariant_required":context.get("structural_invariant_required",False),
        "structural_artificial_noise_allowed":False if context["cell"]=="STR-UKF" else None,
        "training":{"screen":training["screen"],"selected":training["selected"],"final_result_path":str(training["final"].result_path),"final_metrics":training["final"].metrics},
        "tuning":{n:{"selected":{"candidate_id":v["selected"].candidate_id,"step_size":v["selected"].step_size,"num_leapfrog_steps":v["selected"].num_leapfrog_steps},"rows":v["rows"]} for n,v in tuning.items()},
        "runs":{n:campaign.json_ready(v) for n,v in runs.items()},
        "arm_decisions":campaign.json_ready(decisions),
        "identity_reference":context["identity_reference"],"transport_reference":context["training_reference"],"comparator_reference":context["comparator_reference"],
        "statistically_supported_ranking":False,
        "nonclaims":["one named deterministic filter posterior and one fixture","no latent-model exactness or arm superiority claim"],
    }


def _physical(context:Mapping[str,Any],source:tf.Tensor)->tf.Tensor:
    if context["cell"]=="SIR-SGQF":
        return tf.constant((0.1,18.0,10.0),tf.float64)*tf.exp(source)
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import structural_source_chart
    shape=tf.shape(source); flat,_=structural_source_chart(tf.reshape(source,[-1,5])); return tf.reshape(flat,shape)


def _agreement(context:Mapping[str,Any],source:tf.Tensor,tfp:Any):
    if context["cell"]!="SIR-SGQF": return None
    from docs.benchmarks.run_multimodel_neutra_p6_sir_sgqf_neutra_confirmation import _physical_mean_agreement
    return _physical_mean_agreement(tf=tf,tfp=tfp,candidate_source_samples=source,comparator_source_samples=context["comparator_samples"])


def run_smoke(context:Mapping[str,Any])->Mapping[str,Any]:
    from bayesfilter.inference.neural_force_hmc import NeuralForceHMCConfig,sample_neural_force_hmc
    d=context["dimension"]; flat=tf.reshape(context["latent"],[-1,d]); points=tf.gather(flat,(0,17,33,49))
    parity=validate_value_only_endpoint_parity(context["binding"],points,absolute_tolerance=2e-8)
    target=context["binding"].hmc_target(); force=FrozenPositionOnlyForce(lambda x:x,f"{context['cell']}-smoke")
    config=NeuralForceHMCConfig(step_size=0.05,num_leapfrog_steps=2,inverse_mass_diagonal=(1.0,)*d,dtype="float64")
    @tf.function(jit_compile=True)
    def compiled(position,potential): return sample_neural_force_hmc(position,potential,force,target,config,num_warmup=0,num_results=4,seed=tf.constant((20260718,70999),tf.int32))
    chain=compiled(points,target.function(points)); health=campaign.chain_health(chain)
    return {"schema":"bayesfilter.sir_structural_neural_force_hmc_p5_smoke.v1","cell":context["cell"],"passed":bool(parity["passed"] and health["passed"]),"parity":parity,"health":health}

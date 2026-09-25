"""Current public tuning members and common posterior assessment for q20.

Kernel membership depends on acceptance and numerical health. Selection is an
explicit operational choice; posterior precision and reference agreement are
separate, and none of these screens establishes a ranking of methods.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.q20_production_config import PARAMETERS, digest, frozen_scope_hash, scoped_seed, write_json
from bayesfilter.inference.q20_production_training import source_snapshot
from bayesfilter.inference import (
    HMCAcceptancePolicy, HMCCandidateExecutionConfig, HMCControllerConfig,
    PrecomputedMassArtifact, bind_hmc_candidate_set_execution, tune_hmc_kernel,
    build_retained_bound_hmc_archive_runner_from_candidate_set_result,
    load_hmc_candidate_retained_runner, HMCPrecisionTarget, HMCPrecisionPolicy,
    HMCPosteriorAssessmentPolicy, SequentialNeuTraHMCConfig, run_hmc_posterior,
)
from bayesfilter.inference.neutra_hmc import SequentialExactTransitionConfig, run_sequential_exact_transition
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint


def posterior_policy(config, *, parameter_names=PARAMETERS):
    p = config["posterior"]
    targets = [HMCPrecisionTarget(name, mcse_sd_ratio_max=p["mean_mcse_sd"]) for name in parameter_names]
    targets += [HMCPrecisionTarget(name, kind="quantile", probability=probability,
                                  mcse_sd_ratio_max=p["quantile_mcse_sd"])
                for name in parameter_names for probability in p["quantiles"]]
    targets.append(HMCPrecisionTarget("positive_theta_2", mcse_absolute_max=p["event_mcse"]))
    return HMCPosteriorAssessmentPolicy(retained_bulk_ess_min=p["bulk_ess"],
        retained_tail_ess_min=p["tail_ess"], quantities_id="q20_physical_coordinates_and_theta2_sign_v1",
        precision=HMCPrecisionPolicy(tuple(targets), method="autocorrelation", jit_compile=config["jit_compile"]))


def posterior_quantities(draws):
    return {"positive_theta_2": tf.cast(draws[:, :, 2] > 0., tf.float64)}


def sequential_kwargs(config, label):
    p = config["posterior"]
    return dict(warmup_seed=scoped_seed(config, "posterior-warmup", label),
        retained_seed=scoped_seed(config, "posterior-retained", label),
        warmup_chunk_results=p["warmup_chunk"], warmup_min_results=p["warmup_min"],
        warmup_check_window_results=p["warmup_window"], warmup_max_results=p["warmup_max"],
        warmup_rhat_max=p["warmup_rhat"], retained_chunk_results=p["retained_chunk"],
        retained_min_results=p["retained_min"], retained_max_results=p["retained_max"],
        retained_rhat_max=p["retained_rhat"], minimum_chain_count=p["chains"],
        assessment_policy=posterior_policy(config))


def draw_start_bank(config, bridge, beta, label):
    """Prior proposals conditioned only for overdispersed chain initialization."""
    maximum = config["starts"]["max_proposals"]
    with tf.device("/CPU:0"):
        noise = tf.random.stateless_normal([maximum, bridge.parameter_dim],
            scoped_seed(config, "start-bank", label, beta), dtype=tf.float64)
        proposals = bridge.prior_center + tf.sqrt(tf.convert_to_tensor(bridge.prior_variance, tf.float64)) * noise
    selected = {False: [], True: []}
    evaluated = 0
    batch = config["training"]["batch_size"]
    for start in range(0, maximum, batch):
        points = proposals[start:start+batch]
        values, scores, status = bridge.value_score_status(points, tf.constant(beta, tf.float64))
        valid = status["bridge_valid"] & tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(scores), axis=-1)
        for index, valid_row in enumerate(valid.numpy().tolist()):
            evaluated += 1
            sign = bool(points[index, 2] > 0.)
            if valid_row and len(selected[sign]) < config["starts"]["per_sign"]:
                selected[sign].append(points[index])
        if all(len(rows) == config["starts"]["per_sign"] for rows in selected.values()):
            return tf.stack(selected[False] + selected[True]), {"proposals_evaluated": evaluated,
                "seed": list(scoped_seed(config, "start-bank", label, beta)),
                "role": "initialization_only_no_target_conditioning"}
    raise ValueError("start bank exhausted without two valid starts per sign")


def tuning_configs(config, label):
    h = config["tuning"]
    execution = HMCCandidateExecutionConfig(measurement_num_results=h["measurement"],
        verification_num_results=h["verification"], pilot_num_results=h["pilot"],
        num_warmup_steps=h["startup"], seed=scoped_seed(config, "tuning", label),
        acceptance_policy=HMCAcceptancePolicy(target=h["target_acceptance"],
            practical_region=tuple(h["practical_region"]), repair_region=tuple(h["repair_region"])),
        target_status_trace_policy="per_chain_step", use_xla=config["jit_compile"], chain_mode="batched",
        chunk_max_results=h["chunk_max_results"],
        non_xla_reason="explicit tiny CPU reference smoke" if not config["jit_compile"] else None)
    search = HMCControllerConfig(primary_l_grid=tuple(h["l_grid"]), initial_epsilon=h["initial_epsilon"],
        total_budget_units=h["total_budget_units"], repair_reserve_units=h["repair_reserve_units"],
        pilot_enabled=True, candidate_reserve_units=3, evidence_rungs=tuple(h["evidence_rungs"]),
        max_candidates=h["max_candidates"], max_wall_time_seconds=h["max_wall_seconds"])
    return execution, search


def validate_training_export(record, config, adapter, beta):
    if record["frozen_scope_hash"] != frozen_scope_hash(config) or record["sources"] != source_snapshot():
        raise ValueError("trained map has a different protocol/source scope")
    if record["assessment"]["beta"] != beta:
        raise ValueError("map was assessed at another temperature")
    if not record["assessment"]["map_reliability"]["passed"]:
        raise ValueError("map failed numerical reliability")
    if config["role"] != "smoke" and (record["role"] == "smoke" or not record["assessment"]["decision"]["development_eligible"]):
        raise ValueError("map training is not assessed for development; continue or repair training")
    return load_frozen_neutra_artifact(record["frozen_transport"], expected_target_signature=adapter.adapter_signature())


def tune_scope(config, bridge, root, *, method, beta=1., training_export=None,
               initial_position=None, max_work_items=None, resume=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    if method not in {"identity", "classical", "neutra"}:
        raise ValueError("ensemble requires every individual temperature/chart scope")
    adapter = bridge.fixed_beta_adapter(beta)
    label = f"{method}-beta{beta:g}"
    record = None
    if method == "neutra":
        if training_export is None:
            raise ValueError("NeuTra tuning requires an assessed frozen map")
        record = json.loads(Path(training_export).read_text()) if isinstance(training_export, (str, Path)) else training_export
        loaded = validate_training_export(record, config, adapter, beta)
        label += "-" + record["candidate"]["id"]
    elif training_export is not None:
        raise ValueError("ordinary comparator must not consume a learned map")
    if resume is not None:
        # Copy the complete numerical store so continuation writes fresh files
        # while preserving the original attempt and all completed chunks.
        import shutil
        from bayesfilter.inference.hmc_candidate_set_checkpoint import resume_hmc_candidate_set_tuning, load_numerical_tuning_checkpoint
        previous_binding, _ = load_numerical_tuning_checkpoint(resume, adapter=adapter)
        if previous_binding._spec["target_lineage"].get("frozen_scope_hash") != frozen_scope_hash(config):
            raise ValueError("resumed tuning protocol differs")
        shutil.copytree(Path(resume).parent, root / "tuning")
        run = resume_hmc_candidate_set_tuning(root / "tuning" / Path(resume).name, adapter=adapter,
                                             max_work_items=max_work_items)
        binding = run.adapter._execution_binding
    else:
        starts, start_receipt = (draw_start_bank(config, bridge, beta, label) if initial_position is None
                                else (tf.convert_to_tensor(initial_position, tf.float64), {"role": "explicit_fixture"}))
        write_json(root / "starts.json", {**start_receipt, "positions": starts.numpy().tolist()})
        execution, search = tuning_configs(config, label)
        sources = source_snapshot()
        repo = Path(__file__).resolve().parents[2]
        source_paths = [repo / path for path in sources]
        common = dict(adapter=adapter, target_lineage={"model": "ssl_lstm_q20", "data": bridge.target_signature,
            "prior": "gaussian_sd4", "beta": beta, "method": method, "role": config["role"],
            "frozen_scope_hash": frozen_scope_hash(config),
            "chart_root": None if record is None else record["candidate"]["root"]}, config=execution, source_paths=source_paths,
            scope_id=label, search_id=digest([config, label])[:20],
            epsilon_domain=tuple(config["tuning"]["epsilon_domain"]),
            repair_factor=config["tuning"]["repair_factor"],
            max_repairs_per_family=config["tuning"]["max_repairs_per_family"])
        if method == "classical":
            from bayesfilter.inference.hmc_kernel_tuning import HMCKernelTuningConfig, prepare_operational_windowed_mass_handoff
            from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution_from_preparation
            cfg = HMCKernelTuningConfig.serious(target_scope=adapter.target_scope, use_xla=config["jit_compile"],
                chain_execution_mode="tf_function", target_status_trace_policy="per_chain_step",
                metric_update_requirement="require_operational_update",
                seed=scoped_seed(config, "classical-preparation", label),
                public_timeout_budget_s=config["tuning"]["max_wall_seconds"])
            write_json(root / "preparation-policy.json", cfg.payload())
            started = time.monotonic()
            preparation = prepare_operational_windowed_mass_handoff(adapter=adapter,
                initial_position=starts[0], config=cfg, parameter_scales=tf.fill([bridge.parameter_dim], tf.constant(4., tf.float64)))
            from dataclasses import replace
            common["config"] = replace(execution, preparation_elapsed_seconds=time.monotonic()-started)
            binding = bind_hmc_candidate_set_execution_from_preparation(preparation=preparation, **common)
            from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution_new_starts
            binding = bind_hmc_candidate_set_execution_new_starts(binding=binding,
                initial_position=starts, config=common["config"], scope_id=label,
                search_id=common["search_id"], max_repairs_per_family=config["tuning"]["max_repairs_per_family"])
            # Controller construction materializes epsilon_by_l. Replacing
            # only initial_epsilon leaves that already-materialized cohort
            # unchanged, outside the preparation's stability bound.
            bounded_epsilon = max(binding.scope.epsilon_domain[0],
                min(search.initial_epsilon, binding.scope.epsilon_domain[1]))
            search = replace(search, initial_epsilon=bounded_epsilon,
                             epsilon_by_l=())
        elif method == "identity":
            mass = PrecomputedMassArtifact(position=tf.zeros([bridge.parameter_dim], tf.float64),
                covariance=tf.eye(bridge.parameter_dim, dtype=tf.float64), factor=tf.eye(bridge.parameter_dim, dtype=tf.float64),
                adapter_signature=adapter.adapter_signature(), position_role="fixed_identity",
                covariance_source="declared naive identity comparator")
            binding = bind_hmc_candidate_set_execution(initial_position=starts, mass_artifact=mass,
                                                       target_scope=adapter.target_scope, **common)
        else:
            binding = bind_hmc_candidate_set_execution(initial_position=loaded.transport.inverse_theta_to_z_batch(starts),
                frozen_transport_payload=record["frozen_transport"], start_coordinates="active", target_scope=adapter.target_scope, **common)
        if method == "neutra":
            from bayesfilter.inference import tune_fixed_transport_hmc_kernel
            run = tune_fixed_transport_hmc_kernel(base_adapter=adapter, fixed_transport=binding.fixed_transport,
                initial_position=binding.initial_active_state, candidate_set_adapter=binding.typed_adapter,
                config=search, output_dir=root / "tuning", max_work_items=max_work_items)
        else:
            run = tune_hmc_kernel(adapter=adapter, initial_position=binding.initial_active_state,
                candidate_set_adapter=binding.typed_adapter, config=search,
                output_dir=root / "tuning", max_work_items=max_work_items)
    return _export_tuning_result(config, root, method, beta, label, run, binding)


def _export_tuning_result(config, root, method, beta, label, run, binding):
    members = {}
    for candidate_id in run.result.verified_candidate_ids:
        member = build_retained_bound_hmc_archive_runner_from_candidate_set_result(
            candidate_set_result=run.result, candidate_id=candidate_id, retained_binding=binding)
        path = member.export(root / f"member-{candidate_id}.json")
        members[candidate_id] = str(path)
    result = {"status": run.result.completion_status, "method": method, "beta": beta, "label": label,
              "config_hash": digest(config), "verified_members": members,
              "tuning_checkpoint": str(root / "tuning" / "tuning_checkpoint.json"),
              "ranking": "none; choose an explicit member", "production_qualified": False}
    write_json(root / "result.json", result)
    return result


def reverify_member(config, bridge, root, *, parent_member_path, beta, label, initial_position):
    """Fresh starts and evidence for the frozen exact kernel; no retuning."""
    from dataclasses import replace
    from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution_new_starts
    from bayesfilter.inference import tune_fixed_transport_hmc_kernel
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    adapter = bridge.fixed_beta_adapter(beta)
    member = load_hmc_candidate_retained_runner(parent_member_path, adapter=adapter)
    lineage = _check_member_protocol(member, config)
    execution = replace(member._binding.config, seed=scoped_seed(config, "reverification", label),
                        preparation_elapsed_seconds=0.)
    binding = bind_hmc_candidate_set_execution_new_starts(binding=member._binding,
        initial_position=initial_position, config=execution, scope_id=label,
        search_id=digest([config, label, member.member_hash])[:20])
    search = HMCControllerConfig(primary_l_grid=(member.num_leapfrog_steps,),
        epsilon_by_l=((member.num_leapfrog_steps, (member.step_size,)),),
        initial_epsilon=member.step_size, total_budget_units=config["tuning"]["total_budget_units"],
        repair_reserve_units=1, max_candidates=1, pilot_enabled=False,
        evidence_rungs=tuple(config["tuning"]["evidence_rungs"]),
        max_wall_time_seconds=config["tuning"]["max_wall_seconds"])
    common = dict(initial_position=binding.initial_active_state, config=search,
                  candidate_set_adapter=binding.typed_adapter, output_dir=root / "tuning")
    if lineage["method"] == "neutra":
        run = tune_fixed_transport_hmc_kernel(base_adapter=adapter, fixed_transport=binding.fixed_transport, **common)
    else:
        run = tune_hmc_kernel(adapter=adapter, **common)
    write_json(root / "frozen-parent.json", {"member_hash": member.member_hash,
        "epsilon": member.step_size, "L": member.num_leapfrog_steps,
        "physical_starts": tf.convert_to_tensor(initial_position).numpy().tolist(),
        "selection": "frozen_exact_pair_reverification_only"})
    return _export_tuning_result(config, root, lineage["method"], beta, label, run, binding)


def _checkpoint_directory(root, previous):
    import shutil
    destination = Path(root) / "chunks"
    if previous is not None:
        shutil.copytree(previous, destination)
    return destination


def _check_member_protocol(member, config):
    member._validate()
    lineage = member._binding._spec["target_lineage"]
    if lineage.get("frozen_scope_hash") != frozen_scope_hash(config):
        raise ValueError("verified member belongs to another protocol")
    if config["role"] != "smoke" and lineage.get("role") == "smoke":
        raise ValueError("smoke member cannot supply serious posterior sampling")
    return lineage


def sample_member(config, bridge, root, *, member_path, label, resume_chunks=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    member = load_hmc_candidate_retained_runner(member_path, adapter=bridge.fixed_beta_adapter(1.))
    _check_member_protocol(member, config)
    kwargs = sequential_kwargs(config, label)
    policy = kwargs["assessment_policy"]
    controller = SequentialNeuTraHMCConfig(step_size=member.step_size, num_leapfrog_steps=member.num_leapfrog_steps,
        jit_compile=config["jit_compile"], energy_error_log_accept_threshold=config["posterior"]["energy_log_accept_alert"], **kwargs)
    chunks = _checkpoint_directory(root, resume_chunks)
    with DurableTensorCheckpoint(chunks, {"member": member.member_hash, "config": digest(config),
            "label": label, "names": list(PARAMETERS), "quantities": policy.quantities_id}) as store:
        result = run_hmc_posterior(member=member, config=controller, parameter_names=PARAMETERS,
                                  quantities_fn=posterior_quantities, checkpoint_store=store)
    _write_posterior_result(root, result, config, bridge, label)
    return result


def build_member_physical_kernel(member, *, state_shape):
    """Verified candidate to the shared exact transition, preserving health vetoes."""
    from bayesfilter.inference.hmc_candidate_set_retained import HMCCandidateRetainedRunner
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import build_fixed_transport_one_step_transition
    if not isinstance(member, HMCCandidateRetainedRunner):
        raise TypeError("an actual verified candidate member is required")
    member._validate()
    adapter = member._binding._active_adapter
    primitive = build_fixed_transport_one_step_transition(adapter, state_shape=state_shape,
        step_size=member.step_size, num_leapfrog_steps=member.num_leapfrog_steps,
        use_xla=member._binding.config.use_xla, capture_health=True)
    @tf.function(input_signature=(tf.TensorSpec(state_shape, tf.float64), tf.TensorSpec([2], tf.int32)),
                 jit_compile=member._binding.config.use_xla, reduce_retracing=False)
    def kernel(physical, seed):
        latent = member._binding.active_positions(physical)
        next_latent, accepted, log_accept, value, score, healthy = primitive(latent, seed)
        next_physical = member._binding.position_samples(next_latent)
        # Propagate a health failure to the enclosing all-temperature check.
        return tf.where(healthy, next_physical, tf.fill(state_shape, tf.constant(float("nan"), tf.float64)))
    return kernel


def sample_ensemble(config, bridge, root, *, members_by_beta, label, resume_chunks=None,
                    physical_baseline=False, initial_by_beta=None):
    from bayesfilter.inference.tempered_transitions_tf import (
        FixedChartKernelMixture, BoundWithinTemperatureKernel, ProperBridgeReplicaExchange,
        ProperReplicaExchangeTransitionProgram,
    )
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    betas = config["training"]["betas"]
    expected = {str(beta) for beta in betas[1:]}
    chart_count = 1 if physical_baseline else config["ensemble"]["charts"]
    if set(members_by_beta) != expected or any(len(v) != chart_count for v in members_by_beta.values()):
        raise ValueError("ensemble needs K verified charts at every positive beta")
    bindings, all_members, starts = [], [], []
    shape = (config["posterior"]["chains"], bridge.parameter_dim)
    for beta in betas:
        if beta == 0.:
            @tf.function(input_signature=(tf.TensorSpec(shape, tf.float64), tf.TensorSpec([2], tf.int32)),
                         jit_compile=config["jit_compile"], reduce_retracing=False)
            def prior_refresh(state, seed):
                return bridge.prior_center + tf.sqrt(tf.convert_to_tensor(bridge.prior_variance, tf.float64)) * tf.random.stateless_normal(shape, seed, dtype=tf.float64)
            kernel, signature = prior_refresh, digest([bridge.signature, "exact_prior_refresh"])
            role = "analytic_beta_zero_prior_refresh"
        else:
            members = [load_hmc_candidate_retained_runner(path, adapter=bridge.fixed_beta_adapter(beta))
                       for path in members_by_beta[str(beta)]]
            if len({m.member_hash for m in members}) != len(members):
                raise ValueError("duplicate chart member is not an ensemble")
            lineages = [_check_member_protocol(m, config) for m in members]
            roots = [row.get("chart_root") for row in lineages]
            if not physical_baseline and (None in roots or len(set(roots)) != len(roots)):
                raise ValueError("ensemble requires independently trained chart roots")
            if physical_baseline and any(row["method"] != "classical" for row in lineages):
                raise ValueError("physical replica exchange requires independently tuned classical kernels")
            if any(m._binding.config.use_xla != config["jit_compile"] for m in members):
                raise ValueError("member XLA policy differs from ensemble")
            all_members.extend(members)
            kernels = [build_member_physical_kernel(m, state_shape=shape) for m in members]
            mixture = FixedChartKernelMixture(kernels, gamma=[1./len(kernels)]*len(kernels),
                                              chart_ids=[m.member_hash for m in members])
            kernel, signature = mixture.transition_state, mixture.selection.signature
            role = "verified_candidate_members_fixed_chart_mixture"
        bindings.append(BoundWithinTemperatureKernel(beta=beta, bridge_signature=bridge.signature,
            kernel_signature=signature, kernel=kernel, mechanics_role=role))
        initial, receipt = (draw_start_bank(config, bridge, beta, label) if initial_by_beta is None else
                            (tf.convert_to_tensor(initial_by_beta[str(beta)], tf.float64), {"role": "matched_physical_starts"}))
        starts.append(initial)
        write_json(root / f"starts-beta{beta:g}.json", {**receipt, "positions": initial.numpy().tolist()})
    program = ProperReplicaExchangeTransitionProgram(ProperBridgeReplicaExchange(bridge, betas), bindings,
                                                     jit_compile=config["jit_compile"])
    initial = program.initial_state(tf.stack(starts))
    controller = SequentialExactTransitionConfig(transition_signature=program.transition_signature,
                                                 **sequential_kwargs(config, label))
    from bayesfilter.inference.hmc_posterior_assessment import validate_sequential_seeds
    forbidden = set().union(*(member._tuning_seeds() for member in all_members))
    validate_sequential_seeds(controller, forbidden=forbidden)
    chunks = _checkpoint_directory(root, resume_chunks)
    with DurableTensorCheckpoint(chunks, {"transition": program.transition_signature,
            "config": digest(config), "label": label, "initial": tf.io.serialize_tensor(initial["state"]).numpy().hex()}) as store:
        def transition(state, *, num_results, seed, stage):
            for member in all_members:
                member._validate()
            seed_list = seed.numpy().tolist()
            return store.run(f"{stage}-{seed_list[0]}-{seed_list[1]}",
                {"num_results": num_results, "seed": seed_list,
                 "state_hash": store.tensor_hash(state["state"]),
                 "identities_hash": store.tensor_hash(state["identities_at_temperature"]),
                 "transition_index": int(state["transition_index"].numpy())},
                lambda: program(state, num_results=num_results, seed=seed, stage=stage))
        result = run_sequential_exact_transition(transition_program=transition,
            initial_transition_state=initial, posterior_state_fn=program.posterior_state,
            parameter_names=PARAMETERS, config=controller, quantities_fn=posterior_quantities)
    _write_posterior_result(root, result, config, bridge, label)
    return result


def _write_posterior_result(root, result, config, bridge, label):
    from bayesfilter.inference.q20_production_comparison import posterior_summary
    from bayesfilter.inference.neutra_hmc import _tensor_tree_python
    retained = result.get("private_retained_model",
                         result.get("private_retained_raw",
                                   result.get("private_retained_beta_one")))
    summary = posterior_summary(config, retained, target_signature=bridge.target_signature,
                                label=label, sequential_passed=result["passed"])
    if retained is not None:
        import hashlib
        data = tf.io.serialize_tensor(retained).numpy()
        path = root / "retained.tensor"
        path.write_bytes(data)
        summary["retained_archive"] = {"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                                       "dtype": "float64", "shape": list(retained.shape)}
    public = _tensor_tree_python({k:v for k,v in result.items() if not k.startswith("private_")})
    write_json(root / "result.json", {"sequential": public, "summary": summary,
               "role": config["role"], "production_qualified": False,
               "reference_status": "not_assessed", "frozen_scope_hash": frozen_scope_hash(config)})

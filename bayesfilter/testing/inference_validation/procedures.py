"""Narrow adapters to real BayesFilter/TFP execution for validation only."""
from __future__ import annotations

from pathlib import Path
import time
import tensorflow as tf
import tensorflow_probability as tfp

from .designs import seed_for
from .targets import ValidationTarget


class FrozenTransition:
    """The same TFP HMC and reviewed score binding used by native BayesFilter.

    Identity and energy mutations are explicit test controls. They cannot issue
    tuning artifacts. One call is a complete HMC transition, not a leapfrog step.
    """
    def __init__(self, target, *, chains, step_size, leapfrog_steps, control="baseline", jit_compile=True):
        from bayesfilter.inference.native_tfp_hmc import reviewed_independent_chain_target_fn
        target.batch_rank_policy = "rank2_required"
        self.target = target
        target_fn = reviewed_independent_chain_target_fn(target, chain_count=chains,
                                                        parameter_dim=target.parameter_dim)
        epsilon = tf.constant(step_size, tf.float64)
        inner = tfp.mcmc.UncalibratedHamiltonianMonteCarlo(target_fn, step_size=epsilon,
                                                         num_leapfrog_steps=leapfrog_steps)
        if control == "wrong_energy":
            class WrongEnergy(tfp.mcmc.TransitionKernel):
                @property
                def is_calibrated(self): return False
                @property
                def parameters(self): return {}
                def bootstrap_results(self, state): return inner.bootstrap_results(state)
                def one_step(self, state, previous_kernel_results, seed=None):
                    proposal, kr = inner.one_step(state, previous_kernel_results, seed=seed)
                    # Reverse the entire MH log ratio while keeping the same
                    # TFP proposal, target and score. This is the activated defect.
                    delta = kr.target_log_prob - previous_kernel_results.target_log_prob
                    return proposal, kr._replace(log_acceptance_correction=-kr.log_acceptance_correction-2*delta)
            kernel = tfp.mcmc.MetropolisHastings(WrongEnergy())
        else:
            kernel = tfp.mcmc.HamiltonianMonteCarlo(target_fn, step_size=epsilon,
                                                  num_leapfrog_steps=leapfrog_steps)
        self.kernel = kernel
        def step(q, seed):
            if control == "identity":
                return q, tf.zeros([chains], tf.float64)
            if control == "two_cycle":
                return -q, tf.zeros([chains], tf.float64)
            out, kr = kernel.one_step(q, kernel.bootstrap_results(q), seed=seed)
            if control == "duplicate_stream":
                out = tf.repeat(out[:1], chains, axis=0)
            return out, kr.log_accept_ratio
        self.step = tf.function(step, input_signature=[tf.TensorSpec([chains,target.parameter_dim],tf.float64),
            tf.TensorSpec([2],tf.int32)], autograph=False, jit_compile=jit_compile)
        self._powered_steps = {1: self.step}
        self._powered_steps_with_health = {}
        self._signature = self.step.input_signature
        self._jit_compile = jit_compile

        def audit_step(q, seed):
            out, results = kernel.one_step(q, kernel.bootstrap_results(q), seed=seed)
            proposal = results.proposed_results
            return {"state": out, "proposed_state": results.proposed_state,
                    "initial_momentum": proposal.initial_momentum[0],
                    "final_momentum": proposal.final_momentum[0],
                    "log_accept_ratio": results.log_accept_ratio,
                    "is_accepted": results.is_accepted}

        # Inspect actual TFP proposal/momentum evidence for the independent
        # Metropolis-energy oracle. This diagnostic has no tuning authority.
        self.audit_step = tf.function(audit_step, input_signature=self._signature,
                                      autograph=False, jit_compile=jit_compile)

    def powered_step(self, power, *, with_health=False):
        """Compose the same complete MH transition K a fixed number of times.

        Gandy--Scott section 2.2 permits K^s in its random-position experiment.
        The second return is the LAST substep's log ratio, never an acceptance
        ratio for the composition. Power one preserves the original stream.
        With health enabled, a third return checks finite states and log ratios
        over every substep, including those preceding a finite final substep.
        """
        if type(power) is not int or not 1 <= power < 2**31:
            raise ValueError("kernel power must be a positive int32 count")
        if with_health:
            if power not in self._powered_steps_with_health:
                def composed_with_health(q, seed):
                    def body(index, state, last_ratio, healthy):
                        sub_seed = (seed if power == 1 else
                                    tf.random.experimental.stateless_fold_in(seed, index))
                        state, last_ratio = self.step(state, sub_seed)
                        healthy = (healthy & tf.reduce_all(tf.math.is_finite(state))
                                   & tf.reduce_all(tf.math.is_finite(last_ratio)))
                        return index + 1, state, last_ratio, healthy
                    _, state, last_ratio, healthy = tf.while_loop(
                        lambda index, *_: index < power, body,
                        (tf.constant(0), q, tf.zeros(q.shape[:1], q.dtype),
                         tf.reduce_all(tf.math.is_finite(q))), parallel_iterations=1)
                    return state, last_ratio, healthy
                self._powered_steps_with_health[power] = tf.function(
                    composed_with_health, input_signature=self._signature,
                    autograph=False, jit_compile=self._jit_compile)
            return self._powered_steps_with_health[power]
        if power not in self._powered_steps:
            def composed(q, seed):
                def body(index, state, last_ratio):
                    sub_seed = tf.random.experimental.stateless_fold_in(seed, index)
                    state, last_ratio = self.step(state, sub_seed)
                    return index + 1, state, last_ratio
                _, state, last_ratio = tf.while_loop(
                    lambda index, *_: index < power, body,
                    (tf.constant(0), q, tf.zeros(q.shape[:1], q.dtype)),
                    parallel_iterations=1)
                return state, last_ratio
            self._powered_steps[power] = tf.function(
                composed, input_signature=self._signature,
                autograph=False, jit_compile=self._jit_compile)
        return self._powered_steps[power]


def initial_starts(target, regime):
    d = target.parameter_dim
    # Data-independent fixture starts. No posterior/reference draws enter here.
    shifts = tf.constant([-1., -.3, .4, 1.], tf.float64)[:,None]
    starts = tf.broadcast_to(shifts, [4,d])
    if regime == "remote": starts += tf.constant(8., tf.float64)
    if regime == "single_mode": starts += tf.constant(-5., tf.float64)
    if regime == "mode_dispersed":
        centers = tf.constant([-1., -1., 1., 1.], tf.float64) * target.parameters.get("separation", 5.)
        starts = tf.concat([(centers + shifts[:, 0])[..., None], starts[:, 1:]], axis=1)
    if regime == "reference": raise ValueError("reference starts are only for invariance")
    if target.target_id == "funnel_noncentered":
        # Preserve the centered fixture's model starts before changing coordinates.
        starts = tf.concat([starts[:, :1], tf.exp(-starts[:, :1]/2)*starts[:, 1:]], -1)
    return starts


def selected_member_ids(design, candidates):
    """Predeclared choice based on tuning records alone, never posterior output."""
    rows = [(c.candidate_id, c.leapfrog_steps) for c in candidates]
    if design.options.get("member_rule", "declared_l_first") == "declared_l_first":
        rows = [(cid, steps) for cid, steps in rows if steps == design.member_l]
    return tuple(cid for cid, _ in sorted(rows)[:1])


def posterior_quantities(design):
    """Only predeclared functions enter stopping; exact truth stays assessor-only."""
    names = design.options.get("global_quantities", [])
    if not names:
        return None, None
    if names != ["left_mode_probability"] or design.scenario.target != "mixture":
        raise ValueError("unsupported posterior quantity definition")
    def quantities(draws):
        return {"left_mode_probability": tf.cast(draws[..., 0] < 0., tf.float64)}
    return "validation.mixture.left_of_zero.v1", quantities


def run_fixed_comparator(member, target, settings, directory, seed_parts, deadline):
    """Use the native verified-member runner for two fixed archived blocks."""
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload
    from .storage import read_json, write_json, write_tensor
    started = time.monotonic()
    previous = None
    draws = None
    archives = []
    for stage, count in (("discarded_warmup", settings["warmup_results"]),
                         ("fixed_retained", settings["retained_results"])):
        archive = directory / stage / "retained_archive.json"
        if archive.exists():
            payload, _ = member._archive(archive)
            positions = _tensor_from_payload(payload["position_samples"])
        else:
            if deadline is not None and time.monotonic() >= deadline:
                return {"status": "unavailable", "reason": "fixed_arm_deadline", "archives": archives}
            block = member.run(num_results=count, seed=seed_for(*seed_parts, stage),
                               output_dir=archive.parent, previous_archive=previous)
            positions = block["position_samples"]
        values = target.to_model(positions)
        write_tensor(directory / (stage + ".tensor"), values)
        previous = archive
        archives.append(str(archive))
        if stage == "fixed_retained":
            draws = values
    result = {"status": "assessed", "draws_path": str(directory / "fixed_retained.tensor"),
              "warmup_results": settings["warmup_results"], "retained_results": int(draws.shape[0]),
              "archives": archives, "elapsed_seconds": time.monotonic() - started,
              "seeding": "independent of stopped arm conditional on same verified member",
              "warmup_excluded_from_estimates": True, "convergence_claim": False}
    write_json(directory / "result.json", result)
    return result


def execute_pipeline(design, destination, *, data=None, fit_id=0, dataset_id=0, deadline=None):
    """Complete public tuning plus actual replay/posterior controller for all members.

    Native checkpoints allow a repeated call at the same path to finish existing
    work. The surrounding executor binds that path to immutable design/source.
    """
    from bayesfilter.inference import (
        HMCControllerConfig, HMCCandidateExecutionConfig, HMCAcceptancePolicy,
        HMCKernelTuningConfig, tune_hmc_kernel, tune_fixed_transport_hmc_kernel,
        FixedTransportHMCKernelTuningConfig, bind_hmc_candidate_set_execution,
        PrecomputedMassArtifact, resume_hmc_candidate_set_tuning,
        load_numerical_tuning_checkpoint,
        build_retained_bound_hmc_archive_runner_from_candidate_set_result,
        load_hmc_candidate_retained_runner, SequentialNeuTraHMCConfig,
        HMCPosteriorAssessmentPolicy, HMCPrecisionPolicy, HMCPrecisionTarget,
        run_hmc_posterior,
    )
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    from .storage import write_json, write_tensor, read_json

    started = time.monotonic()
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    scenario = design.scenario
    target = ValidationTarget(scenario.target, scenario.parameters, data,
        control=scenario.control if scenario.control in {"ignore_data","wrong_score","omit_jacobian"} else "baseline",
        jit_compile=design.device=="gpu")
    starts = initial_starts(target, scenario.start)
    seed = seed_for(design.seed, design.design_id, dataset_id, fit_id, "tuning")
    acceptance_policy = HMCAcceptancePolicy(**design.options.get("acceptance_policy", {}))
    execution = HMCCandidateExecutionConfig(measurement_num_results=design.measurement_draws,
        verification_num_results=design.measurement_draws, num_warmup_steps=8,
        seed=seed, use_xla=design.device=="gpu", target_status_trace_policy="none",
        acceptance_policy=acceptance_policy,
        non_xla_reason="explicit CPU diagnostic validation profile" if design.device!="gpu" else None)
    search_options = dict(design.options.get("search", {}))
    forbidden = {"primary_l_grid", "initial_epsilon", "max_wall_time_seconds"} & search_options.keys()
    if forbidden:
        raise ValueError("search identity fields belong in the design: " + str(sorted(forbidden)))
    search_values = dict(pilot_enabled=True, refinement_rounds=1, max_candidates=100,
        total_budget_units=300, repair_reserve_units=40)
    search_values.update(search_options)
    search = HMCControllerConfig(primary_l_grid=design.l_grid, initial_epsilon=design.step_size,
        max_wall_time_seconds=design.budget_seconds, **search_values)
    common = dict(target_lineage={"model":scenario.target,"data":data,"prior":scenario.parameters,
                                 "control":target.control}, source_paths=[__file__, str(Path(__file__).with_name("targets.py"))])
    tuning_path = root/"tuning"
    tuning_started = time.monotonic()
    run = None
    if (tuning_path/"candidate_set_result.json").exists():
        binding, controller = load_numerical_tuning_checkpoint(tuning_path/"tuning_checkpoint.json",adapter=target)
        result = controller.result()
        from .designs import digest
        from bayesfilter.inference.hmc_candidate_set_artifacts import candidate_set_result_payload, load_candidate_set_result_payload
        final=load_candidate_set_result_payload(tuning_path/"candidate_set_result.json")
        restored=candidate_set_result_payload(result)
        # The native checkpoint measures elapsed time again after writing the
        # final result. Compare immutable numerical/lifecycle facts, not clocks.
        fields=("scope","config","candidates","candidate_states","work_items","verification_receipts",
                "repair_actions","verified_candidate_ids","completion_status")
        if digest({k:final[k] for k in fields}) != digest({k:restored[k] for k in fields}):
            raise ValueError("completed tuning artifact and checkpoint disagree")
    elif (tuning_path/"tuning_checkpoint.json").exists():
        run = resume_hmc_candidate_set_tuning(tuning_path/"tuning_checkpoint.json", adapter=target)
    elif scenario.route == "ordinary":
        cfg = HMCKernelTuningConfig(preset=design.options.get("preparation_preset","standard"),
            use_xla=design.device=="gpu", target_scope="inference_validation", seed=seed,
            candidate_search_bound_expansion_steps=design.options.get("preparation_bound_expansion_steps", 0),
            bootstrap_initialization_rounds=design.options.get("bootstrap_initialization_rounds", 0),
            metric_evidence_policy=design.options.get("metric_evidence_policy", "temporal_information"),
            metric_probe_num_results=design.options.get("metric_probe_num_results", 1),
            preparation_max_restarts=design.options.get("preparation_max_restarts", 0),
            target_accept_prob=acceptance_policy.target, acceptance_band=acceptance_policy.practical_region,
            repair_band=acceptance_policy.repair_region)
        # Automatic preparation accepts one model position and constructs its
        # own four-chain bank. Prepared execution accepts the entire bank.
        run = tune_hmc_kernel(adapter=target, initial_position=starts[0], config=cfg,
            search_config=None if design.options.get("native_search",False) else search,
            execution_config=execution, output_dir=tuning_path, **common)
    elif scenario.route in {"prepared", "fixed_transport"}:
        if scenario.route == "prepared":
            mass = PrecomputedMassArtifact(position=[0.]*target.parameter_dim,
                covariance=tf.eye(target.parameter_dim,dtype=tf.float64), factor=tf.eye(target.parameter_dim,dtype=tf.float64),
                adapter_signature=target.adapter_signature(), position_role="fixture_origin",
                covariance_source="explicit identity hypothesis, no automatic geometry claim")
            binding = bind_hmc_candidate_set_execution(adapter=target, initial_position=starts,
                mass_artifact=mass, target_scope="inference_validation", scope_id="validation",
                search_id=design.identity, epsilon_domain=(.005,3.), repair_factor=2.,max_repairs_per_family=5,
                config=execution, **common)
            run = tune_hmc_kernel(adapter=target, initial_position=starts, config=search,
                candidate_set_adapter=binding.typed_adapter, output_dir=tuning_path)
        else:
            payload = design.options.get("transport_payload") or {
                "schema":"bayesfilter.neutra.frozen_affine_diag.v1", "transport_id":"validation-affine",
                "dimension":target.parameter_dim,"target_signature":target.adapter_signature(),
                "log_jacobian_available":True,"shift":[.2]*target.parameter_dim,"raw_scale":[.1]*target.parameter_dim}
            # Repository codec reconstructs and verifies the transform; no custom target bypass.
            from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
            transport = load_frozen_neutra_artifact(payload,
                expected_target_signature=target.adapter_signature()).transport
            run = tune_fixed_transport_hmc_kernel(base_adapter=target, fixed_transport=transport,
                initial_position=starts, config=FixedTransportHMCKernelTuningConfig(initial_step_size=design.step_size,
                    leapfrog_grid=design.l_grid,use_xla=design.device=="gpu",target_scope="inference_validation"),
                frozen_transport_payload=payload, search_config=search,execution_config=execution,
                output_dir=tuning_path, **common)
    else:
        raise ValueError("numerical public pipeline route required")
    if run is not None:
        binding=run.adapter._execution_binding  # Tested compatibility boundary.
        result=run.result
    from bayesfilter.inference.hmc_candidate_set_artifacts import candidate_set_result_payload
    observed_tuning_path=write_json(root/"tuning_observation.json",candidate_set_result_payload(result))
    tuning_seconds = time.monotonic() - tuning_started
    selected = selected_member_ids(design, (result.replay_candidate(cid) for cid in result.verified_candidate_ids))
    write_json(root/"posterior_selection.json", {
        "rule": design.options.get("member_rule", "declared_l_first"),
        "assessment_scope": design.options.get("posterior_members", "all"),
        "selected_candidate_ids": selected, "verified_candidate_ids": result.verified_candidate_ids,
        "selection_uses": "tuning identity and predeclared L only; no posterior or truth"})
    members=[]
    for index, candidate_id in enumerate(result.verified_candidate_ids):
        candidate = result.replay_candidate(candidate_id)
        if design.options.get("posterior_members", "all") == "selected" and candidate_id not in selected:
            members.append({"candidate_id":candidate_id,"L":candidate.leapfrog_steps,
                "epsilon":candidate.epsilon,"status":"unassessed_by_design"})
            continue
        member_started = time.monotonic()
        directory=root/"members"/candidate_id
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/"result.json").exists():
            members.append(read_json(directory/"result.json")); continue
        if deadline is not None and time.monotonic()>=deadline:
            candidate = result.replay_candidate(candidate_id)
            members.append({"candidate_id":candidate_id,"L":candidate.leapfrog_steps,
                "epsilon":candidate.epsilon,"status":"unfunded"}); continue
        member_file=directory/"member.json"
        if not member_file.exists():
            member=build_retained_bound_hmc_archive_runner_from_candidate_set_result(
                candidate_set_result=result,candidate_id=candidate_id,retained_binding=binding)
            member.export(member_file)
        member=load_hmc_candidate_retained_runner(member_file,adapter=target)
        construction_seconds = time.monotonic() - member_started
        count=max(64,min(design.draws,design.posterior_cap))
        # Count is a declared engineering profile allocation, never a default.
        precision_targets = [
            HMCPrecisionTarget(name, kind="quantile",probability=.5,mcse_absolute_max=design.mcse_tolerance)
            for name in target.spec.parameters]
        if target.spec.finite_variance:
            precision_targets.extend(HMCPrecisionTarget(name, kind="mean", mcse_absolute_max=design.mcse_tolerance)
                                     for name in target.spec.parameters)
        quantities_id, quantities_fn = posterior_quantities(design)
        precision_targets.extend(HMCPrecisionTarget(name, kind="mean", mcse_absolute_max=design.mcse_tolerance)
                                 for name in design.options.get("global_quantities", []))
        policy=HMCPosteriorAssessmentPolicy(precision=HMCPrecisionPolicy(tuple(precision_targets),
            method="lugsail",jit_compile=design.device=="gpu"), quantities_id=quantities_id)
        counts = dict(warmup_chunk_results=count,warmup_min_results=count,warmup_check_window_results=count,
            warmup_max_results=design.posterior_cap,retained_chunk_results=count,retained_min_results=count,
            retained_max_results=design.posterior_cap)
        counts.update(design.options.get("posterior_settings", {}))
        config=SequentialNeuTraHMCConfig(step_size=member.step_size,num_leapfrog_steps=member.num_leapfrog_steps,
            jit_compile=design.device=="gpu", warmup_seed=seed_for(design.seed,design.design_id,dataset_id,fit_id,index,"warmup"),
            retained_seed=seed_for(design.seed,design.design_id,dataset_id,fit_id,index,"retained"),
            **counts,assessment_policy=policy)
        posterior_started = time.monotonic()
        with DurableTensorCheckpoint(directory/"posterior_chunks",{
            "member":member.member_hash,"policy":policy.payload(),"quantities":quantities_id or "model_coordinates.v1",
            "parameters":target.spec.parameters}) as store:
            posterior=run_hmc_posterior(member=member,config=config,parameter_names=target.spec.parameters,
                model_transform=target.to_model,checkpoint_store=store,
                quantities_fn=quantities_fn,
                budget_check=lambda _: deadline is None or time.monotonic() < deadline)
        posterior_seconds = time.monotonic() - posterior_started
        raw=posterior["private_retained_raw"]
        warmup=posterior["private_warmup_raw"]
        if scenario.control=="warmup_leak": raw=tf.concat([warmup,raw],axis=0)
        if scenario.control=="lost_chunk" and int(raw.shape[0]) > 0: raw=raw[1:]
        if scenario.control=="duplicate_stream": raw=tf.repeat(raw[:, :1], 4, axis=1)
        draw_path=write_tensor(directory/"draws.tensor",raw)
        warmup_path=write_tensor(directory/"warmup.tensor",warmup)
        summary={"candidate_id":candidate_id,"L":member.num_leapfrog_steps,"epsilon":member.step_size,
            "member_path":str(member_file),"draws_path":str(draw_path),"warmup_path":str(warmup_path),
            "status":"assessed","posterior":{k:v for k,v in posterior.items() if not k.startswith("private_")},
            "control":scenario.control,"recorded_retained_count":int(raw.shape[0]),
            "timing":{"construction_seconds":construction_seconds,
                      "controller_seconds":posterior_seconds,
                      "compilation_separated":False},
            "duplicate_chains": bool(tf.reduce_all(raw == raw[:, :1])) if int(raw.shape[0]) else None,
            "warmup_exclusion_matches":int(raw.shape[0])==posterior["retained_results_per_chain"]}
        if design.options.get("fixed_comparator") is not None:
            try:
                summary["fixed_comparator"] = run_fixed_comparator(member, target,
                    design.options["fixed_comparator"], directory/"fixed_comparator",
                    (design.seed,design.design_id,dataset_id,fit_id,index,"fixed"),deadline)
            except Exception as exc:
                summary["fixed_comparator"] = {"status":"failed", "exception":type(exc).__name__,
                                                "reason":str(exc)}
                write_json(directory/"fixed_comparator"/"failure.json",summary["fixed_comparator"])
        write_json(directory/"result.json",summary); members.append(summary)
    payload={"tuning_path":str(observed_tuning_path),"completion":result.completion_status,
        "verified_candidate_ids":list(result.verified_candidate_ids),"candidate_count":len(result.candidates),
        "members":members,"source_binding":binding.binding_hash,"numerical_route":scenario.route,
        "data":data,"dataset_id":dataset_id,"fit_id":fit_id,
        "selection":{"candidate_ids":selected,"scope":design.options.get("posterior_members","all"),
                     "rule":design.options.get("member_rule","declared_l_first")},
        "timing":{"tuning_or_reload_seconds":tuning_seconds,
                  "preparation_seconds":binding.config.preparation_elapsed_seconds,
                  "invocation_seconds":time.monotonic()-started,
                  "compilation_separated":False}}
    write_json(root/"pipeline.json",payload)
    return payload

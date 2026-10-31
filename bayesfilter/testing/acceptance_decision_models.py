"""Diagnostic public-tuner model cases for the replicated acceptance policy.

Reference checks never supply geometry, starts, proposal selection or admission
data. A completed case is engineering/delivery evidence, not posterior evidence.
Every invocation preserves a new output root and the native numerical ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from bayesfilter.runtime.execution_budget import execution_budget
from .inference_validation.targets import ValidationTarget
from .inference_validation.designs import digest


EXPECTED_OUTCOMES = {"positive_delivery", "inconclusive_at_cap", "numerical_rejection",
                     "preparation_review_required", "budget_deferred", "bounded_diagnostic"}


def validate_configuration(configuration):
    required = {"schema", "case_id", "target", "parameters", "data", "route", "geometry",
                "active_starts", "epsilon_by_l", "epsilon_domain", "repair_factor",
                "max_repairs_per_family", "policy", "evidence_rungs", "seed", "wall_seconds",
                "expected_outcome", "classification", "provenance"}
    schema = configuration.get("schema")
    if schema in {"bayesfilter.acceptance_model_config.v2", "bayesfilter.acceptance_model_config.v3"}:
        required.add("search")
        if schema.endswith(".v3"):
            required.add("replicated_trial_batch_size")
            batch = configuration.get("replicated_trial_batch_size")
            if type(batch) is not int or not 2 <= batch <= 32:
                raise ValueError("model v3 requires explicit trial batch size in [2, 32]")
    elif schema != "bayesfilter.acceptance_model_config.v1":
        raise ValueError("unsupported model configuration")
    if set(configuration) != required:
        raise ValueError("model configuration requires explicit fields: " + str(sorted(required ^ set(configuration))))
    if configuration["route"] not in {"ordinary", "fixed_transport"}:
        raise ValueError("unsupported model route")
    if (configuration["route"] == "fixed_transport") != (configuration["geometry"].get("kind") in {"exact","residual"}):
        raise ValueError("fixed transport and supplied map geometry must agree")
    if configuration["classification"] not in {"development", "confirmation", "regression", "scheduled_stress"}:
        raise ValueError("explicit evidence classification required")
    if configuration["expected_outcome"] not in EXPECTED_OUTCOMES:
        raise ValueError("explicit expected outcome required")
    if (configuration["expected_outcome"] == "bounded_diagnostic"
            and configuration["classification"] != "development"):
        raise ValueError("legacy bounded diagnostics cannot establish a regression or confirmation")
    if not configuration["provenance"] or not 0 < configuration["wall_seconds"] <= 3600:
        raise ValueError("bounded wall allocation and numerical provenance required")
    policy = HMCReplicatedAcceptancePolicy.from_payload(configuration["policy"])
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
    proposals = tuple((l, tuple(eps)) for l, eps in configuration["epsilon_by_l"])
    if schema in {"bayesfilter.acceptance_model_config.v2", "bayesfilter.acceptance_model_config.v3"}:
        search = HMCControllerConfig.from_payload(configuration["search"])
        if (search.epsilon_by_l != proposals
                or search.primary_l_grid != tuple(l for l, _ in proposals)
                or search.evidence_rungs != tuple(configuration["evidence_rungs"])
                or search.replicated_acceptance_policy != policy
                or search.max_wall_time_seconds != float(configuration["wall_seconds"])):
            raise ValueError("serialized search disagrees with frozen model configuration")
    else:
        search = HMCControllerConfig(primary_l_grid=tuple(l for l,_ in proposals), epsilon_by_l=proposals,
            max_candidates=policy.max_candidates, total_budget_units=policy.max_candidates*12,
            repair_reserve_units=policy.max_candidates*2, evidence_rungs=tuple(configuration["evidence_rungs"]),
            replicated_acceptance_policy=policy, max_wall_time_seconds=float(configuration["wall_seconds"]))
    return policy, search


def outcome_check(expected, *, states, completion, exported, receipts):
    """Assert an announced observable outcome; absence of an oracle is not a pass."""
    if expected not in EXPECTED_OUTCOMES:
        raise ValueError("unknown model outcome")
    if expected == "bounded_diagnostic":
        return None  # Historical development report, never a successful test.
    if completion in {"shared_invalidity", "paused_infrastructure"} or not states:
        return False
    if expected == "budget_deferred":
        return completion == "partial_budget" and not exported
    if completion != "complete":
        return False
    if expected == "positive_delivery":
        return bool(exported) and set(exported) == {cid for cid,s in states.items() if s == "verified"}
    if exported:
        return False
    if expected in {"inconclusive_at_cap", "preparation_review_required"}:
        return set(states.values()) == {expected}
    # A directional rejection or exhaustion of a repair budget is not a
    # numerical failure. Require an actual health/validity veto for each pair.
    return (set(states.values()) == {"promotion_failed"}
            and all(any(r.candidate_id == cid and (r.hard_vetoes or r.promotion_vetoes)
                        for r in receipts) for cid in states))


def evidence_accounting(binding, result):
    """Count unique completed trials and all attempted chunks across rungs."""
    from bayesfilter.inference.hmc_acceptance_trials import initialize_seed_registry
    initialize_seed_registry(binding, result)
    trials, seeds, committed_chunks = {}, {}, {}
    works = {w.work_item_id: w for w in result.work_items}
    candidates = {c.candidate_id: c for c in result.candidates}
    for row in binding._evidence.values():
        work = row["work"]
        ordinals = [t["ordinal"] for t in row.get("trials", ())]
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("duplicate completed trial in one evidence rung")
        for trial in row.get("trials", ()):
            key = (work["candidate_id"], work["stage"], trial["ordinal"])
            fingerprint = digest(trial)
            if key in trials and trials[key][0] != fingerprint:
                raise ValueError("a cumulative trial changed between evidence rungs")
            seed = tuple(trial["seed"])
            if seed in seeds and seeds[seed] != key:
                raise ValueError("independent stages or trials reused a stream")
            seeds[seed] = key
            trials[key] = (fingerprint, trial["scores"] is not None)
            indices = [c["trial_chunk_index"] for c in trial["chunks"]]
            if indices != list(range(len(indices))):
                raise ValueError("duplicate or missing completed trial chunk")
            for chunk in trial["chunks"]:
                chunk_key = (*key, chunk["trial_chunk_index"])
                committed_chunks[chunk_key] = (chunk["count"] * binding.config.acceptance_policy.chain_count,
                    candidates[work["candidate_id"]].leapfrog_steps + 1)
    attempted_transitions = attempted_work = 0
    events = [e for e in result.accounting_events if e["event"] == "numerical_chunk_charged"]
    for event in events:
        c = candidates[works[event["work_item_id"]].candidate_id]
        if event["gradient_work"] != event["transitions"] * (c.leapfrog_steps + 1):
            raise ValueError("attempted gradient cost does not match the candidate")
        attempted_transitions += event["transitions"]
        attempted_work += event["gradient_work"]
    committed_transitions = sum(n for n,_ in committed_chunks.values())
    committed_work = sum(n*factor for n,factor in committed_chunks.values())
    if attempted_work < committed_work or attempted_transitions < committed_transitions:
        raise ValueError("completed trials exceed charged numerical work")
    return {"unique_complete_trials": len(trials),
        "valid_complete_trials": sum(valid for _,valid in trials.values()),
        "invalid_complete_trials": sum(not valid for _,valid in trials.values()),
        "completed_trial_transitions": committed_transitions,
        "attempted_transitions": attempted_transitions, "gradient_work": attempted_work,
        "attempted_work_outside_complete_trials": attempted_work - committed_work,
        "charged_chunks": len(events), "independent_trial_streams": len(seeds)}


def reference_checks(target, positions):
    """Independent density and central-difference score diagnostic, never tuning."""
    from .inference_validation.references.analytic import log_density
    value, score = target.log_prob_and_grad(positions)
    tf.debugging.assert_all_finite(value, "nonfinite reference-probe value")
    tf.debugging.assert_all_finite(score, "nonfinite reference-probe score")
    approximate_no_reference = target.target_id.startswith("ssm_nonlinear")
    if approximate_no_reference:
        # This checks the derivative of the declared finite approximate value,
        # not independent correctness of that likelihood approximation.
        value_error = None
        evaluate = lambda q: target.log_prob_and_grad(q)[0]
    else:
        def evaluate(q):
            values = log_density(target.target_id, q.numpy(), target.parameters, target.data)
            return tf.convert_to_tensor(values, tf.float64)
        reference = evaluate(positions)
        value_error = float(tf.reduce_max(tf.abs(value-reference)))
        tf.debugging.assert_near(value, reference, atol=1e-6, rtol=1e-7,
                                 message="independent density mismatch")
    # h=1e-5 is a checked central-difference diagnostic hypothesis, not a score
    # implementation. Halving h separately exposes an unstable reference.
    errors = []
    differences = []
    for h in (1e-5, 5e-6):
        columns = []
        for j in range(target.parameter_dim):
            delta = tf.one_hot(j, target.parameter_dim, on_value=h, off_value=0., dtype=tf.float64)
            columns.append((evaluate(positions+delta)-evaluate(positions-delta))/(2*h))
        numerical = tf.stack(columns, axis=1)
        tf.debugging.assert_near(score, numerical, atol=2e-4, rtol=2e-4,
                                 message="independent score mismatch")
        errors.append(float(tf.reduce_max(tf.abs(score-numerical))))
        differences.append(numerical)
    return {"density_max_abs_error": value_error, "score_max_abs_errors": errors,
        "difference_step_halving_change": float(tf.reduce_max(tf.abs(differences[0]-differences[1]))),
        "reference_scope": "declared approximate value derivative only" if approximate_no_reference else "independent target value and derivative",
        "posterior_correctness": "not_tested", "admission_effect": "none"}


def run_model(configuration, output, *, manifest=None):
    policy, search = validate_configuration(configuration)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stages = []
    def stage(name):
        stages.append({"stage": name, "elapsed_seconds": time.monotonic()-started})
        (root/"stage_timing.json").write_text(json.dumps(stages,indent=2)+"\n")
    stage("started")
    (root/"configuration.json").write_text(json.dumps(configuration, indent=2)+"\n")
    if manifest is not None:
        (root/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    from bayesfilter.inference import (
        HMCCandidateExecutionConfig, PrecomputedMassArtifact, bind_hmc_candidate_set_execution,
        tune_hmc_kernel, tune_fixed_transport_hmc_kernel,
        export_hmc_candidate_retained_runners, load_hmc_candidate_retained_runners,
        load_numerical_tuning_checkpoint,
    )
    target = ValidationTarget(configuration["target"], configuration["parameters"], configuration["data"])
    starts = tf.constant(configuration["active_starts"], tf.float64)
    if tuple(starts.shape) != (4, target.parameter_dim):
        raise ValueError("four full-dimensional explicitly declared active starts required")
    tf.debugging.assert_all_finite(starts, "invalid start bank")
    geometry = configuration["geometry"]
    extra = {}
    if configuration["route"] == "ordinary":
        if geometry["kind"] == "quadratic_location":
            if target.target_id != "ssm_campaign_location":
                raise ValueError("quadratic preparation is restricted to the analytic K0 fixture")
            # Native target score, not the independent posterior oracle.
            probes = tf.constant([[-1.],[0.],[1.]],tf.float64)
            _, gradient = target.log_prob_and_grad(probes)
            precision = gradient[1,0]-gradient[2,0]
            tf.debugging.assert_positive(precision)
            tf.debugging.assert_near(gradient[0,0]-gradient[1,0],precision,atol=1e-8,rtol=1e-8)
            center = [float(gradient[1,0]/precision)]
            scales = [float(tf.math.rsqrt(precision))]
        elif geometry["kind"] == "explicit_diagonal":
            center, scales = geometry["center"], geometry["scale"]
        else:
            raise ValueError("unsupported fixed geometry")
        if len(center) != target.parameter_dim or len(scales) != target.parameter_dim or any(s<=0 or not math.isfinite(s) for s in scales):
            raise ValueError("invalid full-dimensional frozen geometry")
        factor = tf.linalg.diag(tf.constant(scales, tf.float64))
        extra["mass_artifact"] = PrecomputedMassArtifact(position=center, factor=factor,
            covariance=tf.matmul(factor,factor,transpose_b=True), adapter_signature=target.adapter_signature(),
            position_role="declared_diagnostic_preparation", covariance_source=configuration["provenance"])
        physical_starts = tf.constant(center,tf.float64)[None,:] + starts*tf.constant(scales,tf.float64)[None,:]
    else:
        from .inference_validation.funnel_maps import supplied_funnel_map
        from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
        if geometry["kind"] not in {"exact", "residual"}:
            raise ValueError("this diagnostic requires a declared supplied funnel map")
        supplied = supplied_funnel_map(geometry["kind"])
        if supplied["target"] != target.target_id or supplied["parameters"] != target.parameters:
            raise ValueError("supplied map target must match declared target")
        extra["frozen_transport_payload"] = supplied["transport_payload"]
        transport = load_frozen_neutra_artifact(supplied["transport_payload"], expected_target_signature=target.adapter_signature()).transport
        physical_starts = transport.forward_batch(starts)
    oracle = reference_checks(target, physical_starts)
    stage("reference_checked")
    (root/"reference.json").write_text(json.dumps(oracle,indent=2)+"\n")
    sources = [__file__, str(Path(__file__).parent/"inference_validation/targets.py")]
    if target._ssm is not None:
        if hasattr(target._ssm,"source_paths"):
            sources.extend(target._ssm.source_paths())
        else:
            sources += [str(Path(__file__).parent/"inference_validation/ssm_targets.py"),
                        target.value_score_capability().evidence_path]
    if configuration["route"] == "fixed_transport":
        sources.append(str(Path(__file__).parent/"inference_validation/funnel_maps.py"))
    (root/"source-manifest.json").write_text(json.dumps({str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                                         for p in sources if p},indent=2)+"\n")
    execution = HMCCandidateExecutionConfig(measurement_num_results=policy.trial_num_results,
        verification_num_results=policy.trial_num_results, num_warmup_steps=policy.discarded_prefix,
        seed=tuple(configuration["seed"]), acceptance_policy=policy, target_status_trace_policy="per_chain_step",
        use_xla=True, chain_mode="batched", reuse_leapfrog_graphs=True,
        chunk_max_results=policy.trial_num_results+policy.discarded_prefix,
        replicated_trial_batch_size=configuration.get("replicated_trial_batch_size", 1))
    binding = bind_hmc_candidate_set_execution(adapter=target, initial_position=starts,
        start_coordinates="active", target_scope="inference_validation",
        target_lineage={"model":target.target_id,"data":target.data,"parameters":target.parameters},
        config=execution,source_paths=[p for p in sources if p],scope_id=configuration["case_id"],search_id=digest(configuration),
        epsilon_domain=tuple(configuration["epsilon_domain"]),repair_factor=configuration["repair_factor"],
        max_repairs_per_family=configuration["max_repairs_per_family"],**extra)
    stage("binding_prepared")
    with execution_budget(deadline=started+configuration["wall_seconds"]):
        kwargs = dict(initial_position=starts, config=search, candidate_set_adapter=binding.typed_adapter,output_dir=root/"tuning")
        if configuration["route"] == "ordinary":
            run = tune_hmc_kernel(adapter=target,**kwargs)
        else:
            run = tune_fixed_transport_hmc_kernel(base_adapter=target,fixed_transport=binding.fixed_transport,**kwargs)
    stage("tuning_returned")
    paths = (export_hmc_candidate_retained_runners(candidate_set_result=run.result,
              retained_binding=binding, output_dir=root) if run.result.verified_candidate_ids else {})
    stage("all_members_exported")
    from bayesfilter.inference.hmc_candidate_set_execution import _fresh_numerical_replay_scope
    # Both readers are independent of the live tuning binding. Reconstruct
    # identical raw evidence once within this closeout; retain every file check.
    with _fresh_numerical_replay_scope():
        restored = load_hmc_candidate_retained_runners(paths.values(), adapter=target)
        stage("all_members_reloaded")
        if set(restored) != set(run.result.verified_candidate_ids):
            raise ValueError("retained export/reload omitted or added a verified member")
        exported = list(restored)
        loaded, controller = load_numerical_tuning_checkpoint(root/"tuning/tuning_checkpoint.json",adapter=target)
    stage("checkpoint_reconstructed")
    if controller.result().candidate_states != run.result.candidate_states:
        raise ValueError("native checkpoint changed candidate states")
    accounting = evidence_accounting(binding, run.result)
    stage("accounting_checked")
    positive = bool(exported)
    result = {"schema":"bayesfilter.replicated_acceptance_model_result.v1","case_id":configuration["case_id"],
        "classification":configuration["classification"],"expected_outcome":configuration["expected_outcome"],
        "positive_delivery":positive,"expectation_met":outcome_check(configuration["expected_outcome"],
            states=run.result.candidate_states,completion=run.result.completion_status,
            exported=exported,receipts=run.result.verification_receipts),
        "candidate_states":run.result.candidate_states,"verified_candidate_ids":exported,
        "candidates":[c.payload() for c in run.result.candidates],
        "repair_actions":[action.payload() for action in run.result.repair_actions],
        "search":search.payload(), "search_state":run.result.search_state,
        "work_items":[work.payload() for work in run.result.work_items],
        "completion_status":run.result.completion_status,"checkpoint_recomputed":True,
        "receipts":[{"L":r.exact_l,"epsilon":r.epsilon,"stage":r.stage,"decision":r.decision,
                     "acceptance":r.acceptance,"vetoes":r.promotion_vetoes} for r in run.result.verification_receipts],
        "gradient_work":accounting["gradient_work"],"evidence_accounting":accounting,
        "wall_seconds":time.monotonic()-started,"target_signature":target.adapter_signature(),
        "model_starts":physical_starts.numpy().tolist(),"reference":oracle,
        "posterior_correctness":"not_tested","default_promotion":False,"jit_compile":True,
        "sampling_streams":configuration["seed"],"data":target.data}
    (root/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result

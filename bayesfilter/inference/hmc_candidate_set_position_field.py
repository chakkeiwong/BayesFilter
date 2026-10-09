"""Position-field mechanics using the shared candidate controller.

The repository-issued transition binding retains its conditional mechanics
authority. This adapter cannot issue an exact-score numerical retained member.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCCandidateSetScope, HMCTuningCandidateSetController, HMCInfrastructureFailure, HMCWorkItem, _sha256,
)
from bayesfilter.inference.hmc_candidate_set_adapters import issue_hmc_candidate_set_adapter, run_typed_hmc_candidate_set
from bayesfilter.inference.hmc_candidate_set_artifacts import candidate_set_result_payload, _validate_result_payload, _canonical
from bayesfilter.inference.hmc_candidate_set_execution import (
    _tensor_payload, _tensor_from_payload, _report_rhat, _source_closure, _check_sources, _runtime_policy,
    HMCCandidateExecutionConfig,
)
from bayesfilter.inference.hmc_candidate_set_public import _search, _preflight_output, _preflight_search
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy, evaluate_hmc_acceptance_evidence
from bayesfilter.inference.hmc_candidate_decisions import HMCCandidateDecision
from bayesfilter.inference.hmc_candidate_runtime import chunk_seed, before_numerical_chunk
from bayesfilter.inference.hmc_preparation import HMCPreparationProgress


class _MechanicsCheckpoint:
    """Bounded persistence only; does not grant numerical replay authority."""

    def __init__(self, spec):
        self.spec = spec
        self.partial = {}
        self.preparation_elapsed_seconds = spec["preparation_elapsed_seconds"]
        self._checkpoint_callback = None
        self._deadline = None

    def write_checkpoint(self, result, root):
        from bayesfilter.inference.hmc_candidate_set_checkpoint import _write
        partial = {}
        for work_id, chunks in self.partial.items():
            partial[work_id] = []
            for chunk in chunks:
                digest = _sha256(chunk)
                _write(chunk, root / "position_field_chunks" / (digest + ".json"))
                partial[work_id].append(digest)
        body = {"schema": "bayesfilter.position_field_tuning_checkpoint.v1",
            "preparation_hash": _sha256(self.spec), "result": candidate_set_result_payload(result),
            "partial_chunks": partial}
        if hasattr(self, "_evidence"):
            for digest, evidence in self._evidence.items():
                if _sha256(evidence) != digest:
                    raise ValueError("corrupt replicated mechanics evidence")
                _write(evidence, root / "position_field_evidence" / (digest + ".json"))
            body["schema"] = "bayesfilter.position_field_tuning_checkpoint.v2"
            body["numerical_evidence_hashes"] = list(self._evidence)
        _write({**body, "content_hash": _sha256(body)}, root / "controller_checkpoint.json", replace=True)


class _ReplicatedMechanicsRuntime(_MechanicsCheckpoint):
    """Mechanics backend for the same trial driver used by exact-score tuning."""

    def __init__(self, spec, *, adapter, runner_binding, affine, starts, config):
        super().__init__(spec)
        self.config = HMCCandidateExecutionConfig.from_payload(spec["execution_config"])
        self.scope = HMCCandidateSetScope.from_payload(spec["scope"])
        self.binding_hash = _sha256(spec)
        self.initial_active_state = starts
        self._base_adapter = adapter
        self._runner_binding = runner_binding
        self._affine = affine
        self._mechanics_config = config
        self._runners = {}
        self._evidence = {}
        self._partial = self.partial

    def _run(self, candidate, state, count, seed):
        from types import SimpleNamespace
        from .hmc_tensorflow_tuning import _fixed_kernel, _run_tensor_steps
        key = candidate.leapfrog_steps, count
        if key not in self._runners:
            @tf.function(input_signature=(tf.TensorSpec(self.initial_active_state.shape, tf.float64),
                tf.TensorSpec([], tf.float64), tf.TensorSpec([2], tf.int32)),
                autograph=False, jit_compile=self.config.use_xla)
            def run(position, epsilon, random_seed):
                kernel = _fixed_kernel(binding=self._runner_binding, adapter=self._affine,
                    step_size=epsilon, leapfrog_steps=candidate.leapfrog_steps, config=self._mechanics_config)
                return _run_tensor_steps(kernel=kernel, initial_state=position,
                    num_results=count, seed=random_seed, adaptive=False)
            self._runners[key] = run
        chunk = self._runners[key](state, tf.constant(candidate.epsilon, tf.float64), tf.constant(seed, tf.int32))
        trace = {k: v for k, v in chunk._asdict().items() if k != "states" and not k.startswith("final_")}
        return SimpleNamespace(samples=chunk.states, trace=trace, metadata={
            "authority": "conditional_mechanics_only",
            "seed_layout": "tfp_stateless_vector_state_independent_chain_rows", "root_seed": seed})

    def analyze_trial(self, initial, samples, trace):
        from .hmc_verification import evaluate_hmc_trial_health
        p = self.config.acceptance_policy
        policy = HMCAcceptancePolicy(**{name: getattr(p, name) for name in (
            "min_movement_rate", "max_repeated_state_fraction",
            "min_normalized_return_displacement", "max_abs_log_accept_energy_proxy")})
        w = self.config.num_warmup_steps
        evidence = evaluate_hmc_trial_health(samples=samples[w:], log_accept_ratio=trace["log_accept_ratio"][w:],
            is_accepted=trace["is_accepted"][w:], policy=policy, native_divergence_status="available",
            native_divergence_count=int(tf.reduce_sum(tf.cast(trace["divergence"], tf.int32))))
        failures = tuple("nonfinite_" + name for name, value in (
            ("state", samples), ("log_accept_ratio", trace["log_accept_ratio"]), ("delta_h", trace["delta_h"]))
            if not bool(tf.reduce_all(tf.math.is_finite(value))))
        decision = HMCCandidateDecision.from_evidence(evidence, hard_vetoes=failures)
        if decision.hard_vetoes and decision.evidence_validity == "valid":
            decision = HMCCandidateDecision("unavailable", "candidate_data_invalid",
                hard_vetoes=decision.hard_vetoes, promotion_vetoes=decision.promotion_vetoes)
        return {**decision.payload(), "health_evidence": evidence.payload(),
                "diagnostic_alerts": evidence.payload().get("diagnostic_alerts", ())}

    def evidence_analysis(self, numerical):
        from .hmc_acceptance_trials import evidence_analysis
        return evidence_analysis(self, numerical)

    def trial_seed_lineage(self, seed):
        return (tuple(seed),)  # The mechanics runner uses one batched state.

    def observe(self, work, candidate):
        from .hmc_acceptance_trials import observe
        from .hmc_candidate_set_tuning import HMCSharedInvalidity
        try:
            _check_sources(self.spec["source_closure"])
        except ValueError as exc:
            raise HMCSharedInvalidity(str(exc)) from exc
        if _sha256(self._runner_binding.payload()) != self.spec["runner_binding_hash"]:
            raise HMCSharedInvalidity("position-field transition changed")
        if work.candidate_record_hash != candidate.candidate_record_hash:
            raise ValueError("mechanics trial candidate mismatch")
        return observe(self, work, candidate)

    def work_cost(self, work, candidate):
        from .hmc_acceptance_trials import work_cost
        return work_cost(self, work, candidate)


def _issued_adapter(spec, *, adapter, runner_binding):
    from bayesfilter.inference.hmc_tensorflow_tuning import (
        TensorFlowHMCKernelTuningConfig, _adapter_signature, _TensorFlowAffineAdapter,
        _TensorFlowAffineTransform, _fixed_kernel, _run_tensor_steps, _TensorChunk,
    )
    config = TensorFlowHMCKernelTuningConfig.from_payload(spec["config"])
    execution = HMCCandidateExecutionConfig.from_payload(spec["execution_config"])
    if _adapter_signature(adapter) != spec["adapter_signature"] or _sha256(runner_binding.payload()) != spec["runner_binding_hash"]:
        raise ValueError("position-field target or transition binding changed")
    if _sha256(_runtime_policy()) != spec["runtime_hash"]:
        raise ValueError("position-field execution policy changed")
    _check_sources(spec["source_closure"])
    prepared = {k: _tensor_from_payload(v) for k, v in spec["preparation"].items()}
    affine = _TensorFlowAffineAdapter(_TensorFlowAffineTransform(prepared["center"], prepared["factor"]),
                                     spec["adapter_signature"], config.target_scope)
    starts = prepared["initial_active_state"]
    policy = execution.acceptance_policy
    scope = HMCCandidateSetScope.from_payload(spec["scope"])
    from .hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    if isinstance(policy, HMCReplicatedAcceptancePolicy):
        runtime = _ReplicatedMechanicsRuntime(spec, adapter=adapter, runner_binding=runner_binding,
            affine=affine, starts=starts, config=config)
        return replace(issue_hmc_candidate_set_adapter(scope=scope, adapter_kind="ordinary",
            observe=runtime.observe, work_cost=runtime.work_cost,
            source_dependency_closure=spec["source_closure"], target_preparation_identity=scope.target_preparation_identity,
            transition_identity=scope.transition_identity), _checkpoint_runtime=runtime)
    runners = {}
    runtime = _MechanicsCheckpoint(spec)

    def work_count(work):
        base = (execution.pilot_num_results or execution.measurement_num_results) if work.stage == "pilot" else getattr(
            execution, work.stage + "_num_results")
        return base * work.evidence_multiplier

    def work_cost(work, candidate):
        remaining = work_count(work) + execution.num_warmup_steps - sum(
            chunk["count"] for chunk in runtime.partial.get(work.work_item_id, ()))
        return {"transitions": 4 * remaining,
            "gradient_work": 4 * remaining * (candidate.leapfrog_steps + 1),
            "cost_basis": "conservative_position_field_work_estimate", "charge_mode": "chunk"}

    def observe(work, candidate):
        try:
            _check_sources(spec["source_closure"])
        except ValueError as exc:
            from bayesfilter.inference.hmc_candidate_set_tuning import HMCSharedInvalidity
            raise HMCSharedInvalidity(str(exc)) from exc
        if _sha256(runner_binding.payload()) != spec["runner_binding_hash"]:
            from bayesfilter.inference.hmc_candidate_set_tuning import HMCSharedInvalidity
            raise HMCSharedInvalidity("position-field transition changed")
        count = work_count(work)
        total = count + execution.num_warmup_steps
        def get_runner(take):
            key = candidate.leapfrog_steps, take
            if key in runners:
                return runners[key]
            @tf.function(input_signature=(tf.TensorSpec(starts.shape, tf.float64),
                tf.TensorSpec([], tf.float64), tf.TensorSpec([2], tf.int32)),
                autograph=False, jit_compile=config.use_xla)
            def run(state, epsilon, seed):
                kernel = _fixed_kernel(binding=runner_binding, adapter=affine, step_size=epsilon,
                    leapfrog_steps=candidate.leapfrog_steps, config=config)
                return _run_tensor_steps(kernel=kernel, initial_state=state, num_results=take, seed=seed, adaptive=False)
            runners[key] = run
            return run
        digest = hashlib.sha256(json.dumps([execution.seed, work.work_item_id,
            work.candidate_record_hash, work.stage, scope.search_id], sort_keys=True).encode()).digest()
        seed = tuple(int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4))
        started = time.monotonic()
        pieces = runtime.partial.setdefault(work.work_item_id, [])
        done = sum(p["count"] for p in pieces)
        state = _tensor_from_payload(pieces[-1]["final_state"]) if pieces else starts
        while done < total:
            take = min(total-done, execution.chunk_max_results)
            before_numerical_chunk(runtime, work, candidate, count=take, chains=4, index=len(pieces))
            current_seed = chunk_seed(seed, len(pieces))
            chunk_started = time.monotonic()
            try:
                chunk = get_runner(take)(state, tf.constant(candidate.epsilon, tf.float64), tf.constant(current_seed, tf.int32))
            except (tf.errors.ResourceExhaustedError, tf.errors.UnavailableError,
                    tf.errors.DeadlineExceededError, tf.errors.AbortedError) as exc:
                raise HMCInfrastructureFailure(str(exc)) from exc
            pieces.append({"count": take, "seed": current_seed, "work": work.payload(),
                "initial_state": _tensor_payload(state), "elapsed_seconds": time.monotonic()-chunk_started,
                **{k: _tensor_payload(v) for k, v in chunk._asdict().items()}})
            state = chunk.final_state
            done += take
            if runtime._checkpoint_callback is not None:
                runtime._checkpoint_callback()
        chunk = _TensorChunk(**{k: (_tensor_from_payload(pieces[-1][k]) if k.startswith("final_") else
            tf.concat([_tensor_from_payload(p[k]) for p in pieces], axis=0)) for k in _TensorChunk._fields})
        warmup = execution.num_warmup_steps
        evidence = evaluate_hmc_acceptance_evidence(samples=chunk.states[warmup:],
            log_accept_ratio=chunk.log_accept_ratio[warmup:], is_accepted=chunk.is_accepted[warmup:], policy=policy,
            native_divergence_status="available", native_divergence_count=int(tf.reduce_sum(tf.cast(chunk.divergence, tf.int32))))
        # Warmup is excluded from acceptance statistics, never from health.
        health_failures = tuple("nonfinite_" + name for name, value in (
            ("state", chunk.states), ("log_accept_ratio", chunk.log_accept_ratio),
            ("delta_h", chunk.delta_h)) if not bool(tf.reduce_all(tf.math.is_finite(value))))
        decision = HMCCandidateDecision.from_evidence(evidence, hard_vetoes=health_failures)
        observation = {**decision.payload(), "acceptance": evidence.pooled_mean,
            "stream_id": work.work_item_id, "seed_lineage": seed, "draw_range": (warmup, total),
            "acceptance_evidence": evidence.payload(), "rhat_reporting_only": _report_rhat(chunk.states[warmup:]),
            "elapsed_seconds": sum(p["elapsed_seconds"] for p in pieces),
            "resume_call_elapsed_seconds": time.monotonic()-started,
            "chunks": [{k: p[k] for k in ("count", "seed", "elapsed_seconds")} for p in pieces],
            "numerical_mechanics": {"states": _tensor_payload(chunk.states),
                "log_accept_ratio": _tensor_payload(chunk.log_accept_ratio),
                "delta_h": _tensor_payload(chunk.delta_h),
                "is_accepted": _tensor_payload(chunk.is_accepted), "divergence": _tensor_payload(chunk.divergence),
                "force_fallback": _tensor_payload(chunk.force_fallback)},
            "numerical_handoff_authority": False}
        runtime.partial.pop(work.work_item_id, None)
        return observation
    return replace(issue_hmc_candidate_set_adapter(scope=scope, adapter_kind="ordinary", observe=observe,
        work_cost=work_cost,
        source_dependency_closure=spec["source_closure"], target_preparation_identity=scope.target_preparation_identity,
        transition_identity=scope.transition_identity), _checkpoint_runtime=runtime)


def run_shared_position_field_tuning(*, adapter, initial_position, config, output_dir,
        parameter_scales, runner_binding, search_config=None, execution_config=None, max_work_items=None):
    from bayesfilter.inference.hmc_tensorflow_tuning import (
        _normalize_initial_chain_positions, _adapter_signature, _build_tuning_graph,
    )
    _preflight_output(output_dir)
    _preflight_search(search_config, max_leapfrog_steps=config.max_leapfrog_steps,
                      max_work_items=max_work_items)
    if execution_config is not None and not isinstance(execution_config, HMCCandidateExecutionConfig):
        raise TypeError("execution_config must be HMCCandidateExecutionConfig")
    if execution_config is not None and execution_config.reuse_leapfrog_graphs:
        raise ValueError("reuse_leapfrog_graphs requires the exact-score TFP execution binding")
    if execution_config is not None and execution_config.replicated_trial_batch_size > 1:
        raise ValueError("replicated trial batching requires the exact-score TFP execution binding")
    if parameter_scales is None:
        raise ValueError("position-field preparation requires parameter_scales")
    if execution_config is None and config.verification_results < HMCAcceptancePolicy().min_decisions_per_chain:
        raise ValueError("shared position-field tuning needs at least 64 decisions per evidence rung; smaller historical graphs are diagnostics only")
    if not config.use_xla and not (config.non_xla_reason or (
            execution_config is not None and execution_config.non_xla_reason)):
        raise ValueError("position-field non-XLA execution requires an explicit non_xla_reason")
    policy = (execution_config.acceptance_policy if execution_config is not None else
        HMCAcceptancePolicy(target=config.target_accept_prob,
            practical_region=config.acceptance_policy.overall_band, repair_region=config.acceptance_policy.per_chain_band))
    execution = execution_config or HMCCandidateExecutionConfig(
        measurement_num_results=config.verification_results, verification_num_results=config.verification_results,
        pilot_num_results=config.step_adaptation_results if search_config is None or search_config.pilot_enabled else None,
        num_warmup_steps=0, seed=config.seed, acceptance_policy=policy,
        target_status_trace_policy=config.target_status_trace_policy, use_xla=config.use_xla,
        non_xla_reason=config.non_xla_reason)
    if execution.use_xla != config.use_xla or execution.target_status_trace_policy != config.target_status_trace_policy:
        raise ValueError("preparation and candidate execution policies disagree")
    from .hmc_acceptance_trials import validate_execution_protocol
    validate_execution_protocol(search_config, execution)
    if execution.chain_mode != "serial":
        raise ValueError("position-field batched transitions do not support threaded chain_mode")
    from bayesfilter.hmc_ordinary_selection_policy import ORDINARY_BROAD_PRIMARY_L_GRID
    grid = tuple(l for l in ORDINARY_BROAD_PRIMARY_L_GRID if l <= config.max_leapfrog_steps)
    if not grid:
        raise ValueError("max_leapfrog_steps excludes the broad L grid")
    with HMCPreparationProgress(output_dir, max_wall_time_seconds=(
            None if search_config is None else search_config.max_wall_time_seconds)) as progress:
        _runtime_policy()
        started = time.monotonic()
        runner_binding.validate_public_context(target_scope=config.target_scope,
            target_status_trace_policy=config.target_status_trace_policy,
            chain_execution_mode=config.chain_execution_mode, use_xla=config.use_xla)
        starts, _ = _normalize_initial_chain_positions(initial_position, config=config)
        signature = _adapter_signature(adapter)
        progress.phase("affine_mass_graph_started")
        prepared = _build_tuning_graph(config, runner_binding, signature, preparation_only=True)(starts,
            tf.convert_to_tensor(parameter_scales, tf.float64))
        progress.phase("affine_mass_graph_completed")
        if any(value.dtype.is_floating and not bool(tf.reduce_all(tf.math.is_finite(value))) for value in prepared.values()):
            raise ValueError("position-field preparation is nonfinite")
        epsilon = float(prepared["initial_step_size"])
        search = _search(search_config, epsilon=epsilon, grid=grid)
        closure = _source_closure(adapter, [__file__, str(Path(__file__).with_name("hmc_tensorflow_tuning.py"))])
        preparation = {k: _tensor_payload(v) for k, v in prepared.items()}
        execution_policy = dict(execution.payload())
        execution_policy.pop("preparation_elapsed_seconds")
        scope = HMCCandidateSetScope(scope_id=config.target_scope, search_id=_sha256(search.payload())[:20],
            target_signature=signature, mass_signature=_sha256(preparation), coordinate_system="position_field_affine",
            start_bank_signature=_sha256(preparation["initial_active_state"]),
            warmup_protocol=_sha256({"preparation": config.payload(), "execution": execution_policy}),
            adapter_signature=signature, source_dependency_hash=_sha256(closure),
            target_preparation_identity=_sha256(preparation),
            transition_identity=_sha256({"runner": runner_binding.payload(), "execution": execution_policy}),
            epsilon_domain=(min(e for _, v in search.epsilon_by_l for e in v)/config.step_repair_factor**(config.verification_repair_rounds+1),
                            max(e for _, v in search.epsilon_by_l for e in v)*config.step_repair_factor**(config.verification_repair_rounds+1)),
            repair_factor=config.step_repair_factor, max_repairs_per_family=config.verification_repair_rounds,
            use_xla=config.use_xla)
        spec = {"schema": "bayesfilter.position_field_candidate_preparation.v1", "config": config.payload(),
            "execution_config": execution.payload(),
            "config_interpretation": "legacy preparation config; shared search and execution config own candidate stages",
            "preparation_elapsed_seconds": time.monotonic()-started, "chunk_max_results": execution.chunk_max_results,
            "adapter_signature": signature, "runner_binding_hash": _sha256(runner_binding.payload()),
            "source_closure": closure, "runtime_hash": _sha256(_runtime_policy()),
            "preparation": preparation, "scope": scope.payload()}
        if output_dir is not None:
            from bayesfilter.inference.hmc_candidate_set_checkpoint import _write
            _write({**spec, "content_hash": _sha256(spec)}, Path(output_dir) / "position_field_preparation.json")
        typed = _issued_adapter(spec, adapter=adapter, runner_binding=runner_binding)
        progress.phase("frozen_execution_ready")
    return run_typed_hmc_candidate_set(typed, search, output_dir=output_dir,
        max_work_items=max_work_items, _preparation_elapsed_seconds=progress.elapsed_seconds)


def resume_position_field_candidate_tuning(path, *, adapter, runner_binding, max_work_items=None):
    path = Path(path)
    spec = json.loads((path.parent / "position_field_preparation.json").read_text())
    digest = spec.pop("content_hash", None)
    if digest != _sha256(spec):
        raise ValueError("position-field preparation checksum mismatch")
    checkpoint = json.loads(path.read_text())
    digest = checkpoint.pop("content_hash", None)
    if (checkpoint.get("schema") not in {"bayesfilter.position_field_tuning_checkpoint.v1", "bayesfilter.position_field_tuning_checkpoint.v2"}
            or digest != _sha256(checkpoint) or checkpoint["preparation_hash"] != _sha256(spec)):
        raise ValueError("position-field checkpoint identity mismatch")
    result = checkpoint["result"]
    body = dict(result)
    result_hash = body.pop("result_hash", None)
    if hashlib.sha256(_canonical(body)).hexdigest() != result_hash:
        raise ValueError("position-field result checksum mismatch")
    _validate_result_payload(result)
    controller = HMCTuningCandidateSetController.from_result_payload(result)
    typed = _issued_adapter(spec, adapter=adapter, runner_binding=runner_binding)
    if _sha256(controller.scope.payload()) != _sha256(typed.scope.payload()):
        raise ValueError("position-field checkpoint scope mismatch")
    works = {w.work_item_id: w for w in controller.result().work_items}
    runtime = typed._checkpoint_runtime
    if isinstance(runtime, _ReplicatedMechanicsRuntime):
        from .hmc_acceptance_trials import validate_execution_protocol, validate_chunks
        validate_execution_protocol(controller.config, runtime.config)
        if checkpoint["schema"] != "bayesfilter.position_field_tuning_checkpoint.v2":
            raise ValueError("replicated mechanics require a versioned trial checkpoint")
        runtime._replicated_evidence_rungs = controller.config.evidence_rungs
        for digest in checkpoint["numerical_evidence_hashes"]:
            numerical = json.loads((path.parent / "position_field_evidence" / (digest + ".json")).read_text())
            if _sha256(numerical) != digest:
                raise ValueError("mechanics evidence checksum mismatch")
            work = HMCWorkItem.from_payload(numerical["work"])
            if work.work_item_id not in works or any(_sha256(v) != _sha256(works[work.work_item_id].payload().get(k))
                    for k, v in work.payload().items() if k != "status"):
                raise ValueError("mechanics evidence work mismatch")
            if _sha256(runtime.evidence_analysis(numerical)) != _sha256(numerical["analysis"]):
                raise ValueError("mechanics evidence analysis mismatch")
            runtime._evidence[digest] = numerical
        observed = set()
        for row in result.get("observations", ()):
            raw = row["observation"]
            numerical = runtime._evidence.get(raw.get("numerical_evidence_hash"))
            if (numerical is None or row["work_item_id"] in observed
                    or row["work_item_id"] != numerical["work"]["work_item_id"]
                    or row["candidate_id"] != numerical["candidate"]["candidate_id"]
                    or row["stage"] != numerical["work"]["stage"]
                    or any(_sha256(raw.get(k)) != _sha256(v) for k, v in numerical["analysis"].items())):
                raise ValueError("mechanics observation differs from its numerical evidence")
            observed.add(row["work_item_id"])
        if any(w.status == "completed" and w.work_item_id not in observed for w in works.values()):
            raise ValueError("mechanics completed work is missing evidence")
        for work_id, hashes in checkpoint["partial_chunks"].items():
            if work_id not in works or works[work_id].status == "completed":
                raise ValueError("mechanics partial work identity mismatch")
            chunks = []
            for digest in hashes:
                chunk = json.loads((path.parent / "position_field_chunks" / (digest + ".json")).read_text())
                if _sha256(chunk) != digest:
                    raise ValueError("mechanics trial chunk checksum mismatch")
                chunks.append(chunk)
            validate_chunks(runtime, works[work_id], chunks)
            runtime._partial[work_id] = chunks
        return run_typed_hmc_candidate_set(typed, controller.config, output_dir=path.parent,
            max_work_items=max_work_items, _resume_controller=controller)
    if checkpoint["schema"] != "bayesfilter.position_field_tuning_checkpoint.v1":
        raise ValueError("legacy mechanics cannot consume trial evidence")
    for work_id, hashes in checkpoint["partial_chunks"].items():
        if work_id not in works or works[work_id].status == "completed":
            raise ValueError("position-field partial work mismatch")
        chunks = []
        state = spec["preparation"]["initial_active_state"]
        for chunk_hash in hashes:
            chunk = json.loads((path.parent / "position_field_chunks" / (chunk_hash + ".json")).read_text())
            if _sha256(chunk) != chunk_hash or chunk["initial_state"] != state:
                raise ValueError("position-field chunk checksum or continuity mismatch")
            work = HMCWorkItem.from_payload(chunk["work"])
            if any(_sha256(v) != _sha256(works[work_id].payload()[k]) for k, v in work.payload().items() if k != "status"):
                raise ValueError("position-field chunk work mismatch")
            samples = _tensor_from_payload(chunk["states"])
            if samples.shape[0] != chunk["count"] or _tensor_payload(samples[-1]) != chunk["final_state"]:
                raise ValueError("position-field chunk state/count mismatch")
            state = chunk["final_state"]
            chunks.append(chunk)
        execution = HMCCandidateExecutionConfig.from_payload(spec["execution_config"])
        work = works[work_id]
        count = ((execution.pilot_num_results or execution.measurement_num_results) if work.stage == "pilot"
                 else getattr(execution, work.stage + "_num_results")) * work.evidence_multiplier
        if sum(c["count"] for c in chunks) > count + execution.num_warmup_steps:
            raise ValueError("position-field checkpoint exceeds evidence allocation")
        typed._checkpoint_runtime.partial[work_id] = chunks
    return run_typed_hmc_candidate_set(typed, controller.config, output_dir=path.parent,
        max_work_items=max_work_items, _resume_controller=controller)

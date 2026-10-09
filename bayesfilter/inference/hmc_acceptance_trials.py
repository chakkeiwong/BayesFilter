"""Shared finite-trial driver for experimental replicated acceptance evidence.

The runtime supplies a repository-owned transition and health implementation.
Trials reset to the frozen start bank; chunks within a trial continue its state.
This module does not create a public tuning entry point or grant artifact authority.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from .hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from .hmc_candidate_decisions import HMCCandidateDecision, HMCCandidateExecutionFailure
from .hmc_candidate_runtime import (
    chunk_seed, before_numerical_chunk, grouped_incremental_chunk_checkpoint,
)
from .hmc_candidate_set_tuning import (
    HMCWorkItem, _sha256, _json_native_sha256, HMCInfrastructureFailure, HMCSharedInvalidity,
)


NUMERICAL_SCHEMA = "bayesfilter.hmc_replicated_numerical_evidence.v2"


def validate_execution_protocol(search, execution):
    policy = execution.acceptance_policy
    replicated = isinstance(policy, HMCReplicatedAcceptancePolicy)
    declared = None if search is None else search.replicated_acceptance_policy
    if replicated != (declared is not None) or (replicated and declared != policy):
        raise ValueError("search and execution must declare the same replicated acceptance policy")


def work_seed(runtime, work):
    identity = {"seed": runtime.config.seed, "scope": runtime.scope.payload(),
                "candidate": work.candidate_record_hash, "stage": work.stage,
                "protocol": runtime.config.acceptance_policy.identity}
    digest = bytes.fromhex(_sha256(identity))
    return tuple(int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4))


def trial_seed(root, ordinal):
    digest = bytes.fromhex(_sha256({"stage_seed": root, "trial_ordinal": ordinal}))
    return tuple(int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4))


def trial_size(runtime):
    policy = runtime.config.acceptance_policy
    return policy.discarded_prefix + policy.trial_num_results


def _register_chunk_seed(runtime, work, seed, ordinal, within):
    registry = runtime._trial_seed_registry
    lineage = runtime.trial_seed_lineage(seed)
    for index, actual in enumerate(lineage):
        identity = (work.candidate_record_hash, work.stage, ordinal, within, index)
        actual = tuple(actual)
        if actual in registry and registry[actual] != identity:
            raise HMCSharedInvalidity("duplicate independent-trial random stream")
        registry[actual] = identity


def initialize_seed_registry(runtime, result):
    """Restore attempted streams as well as completed traces before any retry."""
    runtime._trial_seed_registry = {}
    if isinstance(result, Mapping):
        works = {w["work_item_id"]: HMCWorkItem.from_payload(w) for w in result["work_items"]}
        steps = {c["candidate_id"]: c["leapfrog_steps"] for c in result["candidates"]}
        events = result["accounting_events"]
    else:
        works = {w.work_item_id: w for w in result.work_items}
        steps = {c.candidate_id: c.leapfrog_steps for c in result.candidates}
        events = result.accounting_events
    # These inputs are fixed during this validation. Rebuild the local maps on
    # every call so a later replay cannot reuse trust in changed evidence.
    roots = {key: work_seed(runtime, work) for key, work in works.items()
             if work.trial_range is not None}
    expected_seeds = {}

    def expected_seed(work_id, ordinal, within):
        key = (work_id, ordinal, within)
        if key not in expected_seeds:
            expected_seeds[key] = chunk_seed(trial_seed(roots[work_id], ordinal), within)
        return expected_seeds[key]

    per_trial = (trial_size(runtime)+runtime.config.chunk_max_results-1)//runtime.config.chunk_max_results
    attempted = set()
    for event in events:
        if event["event"] == "numerical_chunk_charged":
            work = works[event["work_item_id"]]
            if work.trial_range is not None:
                if type(event.get("chunk_index")) is not int or event["chunk_index"] < 0:
                    raise ValueError("attempted trial chunk index mismatch")
                ordinal = work.trial_range[0]+event["chunk_index"]//per_trial
                within = event["chunk_index"] % per_trial
                if (ordinal >= work.trial_range[1] or ordinal != event.get("trial_ordinal")
                        or within != event.get("trial_chunk_index")
                        or tuple(event.get("seed", ())) != expected_seed(work.work_item_id, ordinal, within)):
                    raise ValueError("attempted trial stream accounting mismatch")
                count = min(runtime.config.chunk_max_results,
                            trial_size(runtime)-within*runtime.config.chunk_max_results)
                transitions = count * runtime.config.acceptance_policy.chain_count
                if (type(event.get("transitions")) is not int
                        or event["transitions"] != transitions
                        or event.get("gradient_work") != transitions * (steps[work.candidate_id]+1)):
                    raise ValueError("attempted trial cost differs from frozen chunk extent")
                attempted.add((work.work_item_id, ordinal, within, tuple(event["seed"])))
                _register_chunk_seed(runtime, work, event["seed"], event["trial_ordinal"], event["trial_chunk_index"])
    # A lost native call may be retried and charged again. Each persisted chunk
    # nevertheless needs at least one charge; aggregate totals cannot show this.
    groups = [(row["work"]["work_item_id"], row.get("chunks", ()))
              for row in runtime._evidence.values()]
    groups.extend(runtime._partial.items())
    checked_batches = set()
    for work_id, chunks in groups:
        if work_id not in works or works[work_id].trial_range is None:
            raise ValueError("persisted trial work missing from attempted accounting")
        for chunk in chunks:
            key = (work_id, chunk["trial_ordinal"], chunk["trial_chunk_index"], tuple(chunk["seed"]))
            if key not in attempted:
                raise ValueError("persisted trial chunk has no charged attempt")
            metadata = chunk.get("runtime", {})
            if runtime.config.replicated_trial_batch_size > 1:
                batch, row = metadata.get("trial_batch_size"), metadata.get("trial_batch_row")
                if type(batch) is not int or type(row) is not int or not 0 <= row < batch:
                    raise ValueError("persisted trial batch identity is invalid")
                first = chunk["trial_ordinal"] - row
                group_key = (work_id, first, batch)
                if group_key not in checked_batches:
                    for ordinal in range(first, first + batch):
                        seed = expected_seed(work_id, ordinal, 0)
                        if (work_id, ordinal, 0, seed) not in attempted:
                            raise ValueError("persisted trial batch has an uncharged row")
                    checked_batches.add(group_key)
    # A native exception returns no trace; it still cannot erase attempted work.
    for evidence in runtime._evidence.values():
        if "execution_failure" not in evidence:
            continue
        work_id = evidence["work"]["work_item_id"]
        work = works[work_id]
        chunk_index = len(evidence["chunks"])
        ordinal, within = work.trial_range[0] + chunk_index//per_trial, chunk_index % per_trial
        batch = min(runtime.config.replicated_trial_batch_size, work.trial_range[1]-ordinal)
        for offset in range(batch):
            seed = expected_seed(work_id, ordinal+offset, within)
            if (work_id, ordinal+offset, within, seed) not in attempted:
                raise ValueError("failed trial or batch has an uncharged attempted row")


def _validate_work(runtime, work):
    policy = runtime.config.acceptance_policy
    if not isinstance(policy, HMCReplicatedAcceptancePolicy) or work.trial_range is None:
        raise ValueError("replicated work and a replicated policy are required together")
    if work.trial_range[1] != policy.repetition_target(work.evidence_multiplier):
        raise ValueError("trial count differs from its frozen evidence rung")
    if (work.evidence_rung == 0) != (work.trial_range[0] == 0):
        raise ValueError("trial predecessor differs from its evidence rung")
    rungs = runtime._replicated_evidence_rungs
    if work.evidence_rung >= len(rungs):
        raise ValueError("trial look exceeds the frozen schedule")
    expected_start = 0 if work.evidence_rung == 0 else policy.repetition_target(rungs[work.evidence_rung-1])
    if (work.evidence_multiplier != rungs[work.evidence_rung]
            or work.trial_range[0] != expected_start):
        raise ValueError("trial range differs from its frozen look schedule")


def work_cost(runtime, work, candidate, *, remaining=True):
    _validate_work(runtime, work)
    count = (work.trial_range[1] - work.trial_range[0]) * trial_size(runtime)
    if remaining:
        count -= sum(chunk["count"] for chunk in runtime._partial.get(work.work_item_id, ()))
    if count < 0:
        raise ValueError("partial work exceeds the replicated allocation")
    transitions = count * runtime.config.acceptance_policy.chain_count
    return {"transitions": transitions, "gradient_work": transitions * (candidate.leapfrog_steps+1),
            "cost_basis": "complete independent trials including every discarded prefix",
            "charge_mode": "chunk", "leapfrog_steps": candidate.leapfrog_steps}


def validate_chunks(runtime, work, chunks, *, complete=False, return_decoded=False):
    from .hmc_candidate_set_execution import _tensor_from_payload, _tensor_payload, _trace_from_payload
    _validate_work(runtime, work)
    total = trial_size(runtime)
    size = runtime.config.chunk_max_results
    per_trial = (total + size - 1)//size
    maximum = (work.trial_range[1]-work.trial_range[0])*per_trial
    if len(chunks) > maximum or (complete and len(chunks) != maximum):
        raise ValueError("chunk inventory differs from the declared trial allocation")
    state = runtime.initial_active_state
    decoded = []
    root = work_seed(runtime, work)
    expected_work = {key: _sha256(value) for key, value in work.payload().items() if key != "status"}
    batch_seeds = {}
    for index, chunk in enumerate(chunks):
        ordinal = work.trial_range[0] + index//per_trial
        within = index % per_trial
        if within == 0:
            state = runtime.initial_active_state
        count = min(size, total-within*size)
        expected_seed = chunk_seed(trial_seed(root, ordinal), within)
        if (chunk.get("trial_ordinal") != ordinal or chunk.get("trial_chunk_index") != within
                or chunk["count"] != count or tuple(chunk["seed"]) != expected_seed
                or chunk["binding_hash"] != runtime.binding_hash
                or chunk["initial_state"] != _tensor_payload(state)):
            raise ValueError("trial chunk identity, count, seed or reset continuity mismatch")
        actual_work = {key: _sha256(value) for key, value in chunk["work"].items() if key != "status"}
        if actual_work != expected_work:
            raise ValueError("trial chunk work mismatch")
        if runtime.config.replicated_trial_batch_size > 1:
            metadata = chunk.get("runtime", {})
            batch = metadata.get("trial_batch_size")
            row = metadata.get("trial_batch_row")
            if (type(batch) is not int or not 1 <= batch <= runtime.config.replicated_trial_batch_size
                    or type(row) is not int or not 0 <= row < batch
                    or ordinal-row < work.trial_range[0] or ordinal-row+batch > work.trial_range[1]
                    or metadata.get("seed_layout") != "independent_original_tfp_sample_chain_streams_v1"):
                raise ValueError("trial batch identity mismatch")
            batch_key = (ordinal-row, batch)
            if batch_key not in batch_seeds:
                batch_seeds[batch_key] = [list(chunk_seed(trial_seed(root, ordinal-row+i), 0))
                                          for i in range(batch)]
            if metadata.get("trial_batch_seeds") != batch_seeds[batch_key]:
                raise ValueError("trial batch streams differ from the original per-trial streams")
        samples = _tensor_from_payload(chunk["samples"])
        if tuple(samples.shape) != (count, *runtime.initial_active_state.shape):
            raise ValueError("trial chunk sample shape mismatch")
        trace = _trace_from_payload(chunk["trace"])
        if return_decoded:
            decoded.append((samples, trace))
        state = samples[-1]
    return decoded if return_decoded else None


def _prior(runtime, work):
    if work.predecessor_work_id is None:
        return [], None
    rows = [(digest, value) for digest, value in runtime._evidence.items()
            if value["work"]["work_item_id"] == work.predecessor_work_id]
    if len(rows) != 1:
        raise ValueError("missing or duplicate predecessor trial evidence")
    digest, previous = rows[0]
    prior_work = HMCWorkItem.from_payload(previous["work"])
    if (_json_native_sha256(previous) != digest or previous.get("schema") != NUMERICAL_SCHEMA
            or "execution_failure" in previous or prior_work.stage != work.stage
            or prior_work.candidate_record_hash != work.candidate_record_hash
            or prior_work.trial_range[1] != work.trial_range[0]
            or prior_work.evidence_rung + 1 != work.evidence_rung):
        raise ValueError("trial predecessor protocol mismatch")
    if _sha256(runtime.evidence_analysis(previous)) != _sha256(previous["analysis"]):
        raise ValueError("trial predecessor analysis mismatch")
    return list(previous["trials"]), digest


def _assemble_trials(runtime, work, chunks, *, decoded_chunks=None):
    import tensorflow as tf
    from .hmc_candidate_set_execution import (
        _tensor_from_payload, _trace_from_payload, _tensor_payload, _trace_payload)
    from .hmc_acceptance_statistics import complete_trial_scores
    total = trial_size(runtime)
    size = runtime.config.chunk_max_results
    per_trial = (total+size-1)//size
    if decoded_chunks is not None and len(decoded_chunks) != len(chunks):
        raise ValueError("decoded chunk inventory mismatch")
    trials = []
    root = work_seed(runtime, work)
    for offset in range(0, len(chunks), per_trial):
        group = chunks[offset:offset+per_trial]
        if len(group) != per_trial:
            raise ValueError("an incomplete trial cannot supply a score")
        decoded = (decoded_chunks[offset:offset+per_trial] if decoded_chunks is not None else
                   [(_tensor_from_payload(c["samples"]), _trace_from_payload(c["trace"])) for c in group])
        if len(decoded) == 1:
            samples, trace = decoded[0]
        else:
            samples = tf.concat([sample for sample, _ in decoded], axis=0)
            traces = [trace for _, trace in decoded]
            trace = tf.nest.map_structure(lambda *parts: tf.concat(parts, axis=0), *traces)
        health = runtime.analyze_trial(runtime.initial_active_state, samples, trace)
        scores = None
        if health["evidence_validity"] == "valid":
            scores = complete_trial_scores(trace["log_accept_ratio"][runtime.config.num_warmup_steps:],
                policy=runtime.config.acceptance_policy, jit_compile=runtime.config.use_xla)
        ordinal = work.trial_range[0] + offset//per_trial
        trials.append({"ordinal": ordinal, "seed": trial_seed(root, ordinal),
            "chunks": group, "samples": _tensor_payload(samples), "trace": _trace_payload(trace),
            "health": health, "scores": scores})
    return trials


def _analyze_trials(runtime, work, trials, rungs, *, compact_health=True):
    from .hmc_acceptance_statistics import replicated_acceptance_statistics
    health = [HMCCandidateDecision.from_observation(t["health"]) for t in trials]
    hard = tuple(dict.fromkeys(reason for h in health for reason in h.hard_vetoes))
    vetoes = tuple(dict.fromkeys(reason for h in health for reason in h.promotion_vetoes))
    validity = ("shared_execution_invalid" if any(h.evidence_validity == "shared_execution_invalid" for h in health)
                else "candidate_data_invalid" if any(h.evidence_validity != "valid" for h in health) else "valid")
    statistics = None
    decision = "unavailable"
    if validity == "valid":
        statistics = replicated_acceptance_statistics([t["scores"]["start_scores"] for t in trials],
            policy=runtime.config.acceptance_policy, stage=work.stage, evidence_rungs=tuple(rungs),
            window_scores=[t["scores"]["window_scores"] for t in trials],
            jit_compile=runtime.config.use_xla)
        decision = statistics["acceptance_decision"]
        if "path_return_resonance_detected" in vetoes:
            decision = "repair_trajectory"
    classified = HMCCandidateDecision(decision, validity, hard_vetoes=hard, promotion_vetoes=vetoes)
    if compact_health:
        from collections import Counter
        health_payload = {
            "schema": "bayesfilter.hmc_replicated_acceptance_evidence.v2",
            "health_summary": {
                "trial_count": len(health),
                "evidence_validity_counts": dict(Counter(h.evidence_validity for h in health)),
                "hard_veto_trial_counts": dict(Counter(x for h in health for x in set(h.hard_vetoes))),
                "promotion_veto_trial_counts": dict(Counter(x for h in health for x in set(h.promotion_vetoes))),
                "detailed_records": "numerical_evidence.trials[*].health",
            },
        }
    else:
        # Historical evidence keeps its original recomputed representation.
        health_payload = {"schema": "bayesfilter.hmc_replicated_acceptance_evidence.v1",
                          "health": [t["health"] for t in trials]}
    return {**classified.payload(), "acceptance": None if statistics is None else statistics["pooled_mean"],
        "acceptance_evidence": {**health_payload, "statistics": statistics,
            "planned_trials": work.trial_range[1], "completed_trials": len(trials),
            "invalid_trials": sum(h.evidence_validity != "valid" for h in health)},
        "engineering_invalidity_reasons": hard,
        "diagnostic_alerts": tuple(dict.fromkeys(x for t in trials for x in t["health"].get("diagnostic_alerts", ())))}


def _failure_analysis(runtime, work, failure, chunks, *, failed_batch_size=1):
    per_trial = (trial_size(runtime)+runtime.config.chunk_max_results-1)//runtime.config.chunk_max_results
    completed = work.trial_range[0]+len(chunks)//per_trial
    outcomes = {"planned_trials": work.trial_range[1],
            "complete_trials_before_failure": completed, "invalid_trial_ordinal": completed,
            "unstarted_trials": work.trial_range[1]-completed-failed_batch_size}
    if runtime.config.replicated_trial_batch_size > 1:
        outcomes.pop("invalid_trial_ordinal")
        outcomes.update(failed_batch_trial_ordinals=list(range(completed, completed+failed_batch_size)),
            failure_localization="native batch failed; no individual invalid row identified")
    return {**failure.analysis(), "trial_outcomes": outcomes}


def evidence_analysis(runtime, numerical):
    from .hmc_candidate_set_execution import _tensor_payload
    work = HMCWorkItem.from_payload(numerical["work"])
    _validate_work(runtime, work)
    if tuple(numerical["evidence_rungs"]) != runtime._replicated_evidence_rungs:
        raise ValueError("replicated evidence look schedule mismatch")
    if (numerical.get("schema") != NUMERICAL_SCHEMA or numerical["binding_hash"] != runtime.binding_hash
            or tuple(numerical["seed"]) != work_seed(runtime, work)
            or numerical["initial_state"] != _tensor_payload(runtime.initial_active_state)):
        raise ValueError("replicated evidence identity mismatch")
    prior, predecessor = _prior(runtime, work)
    if predecessor != numerical["predecessor_evidence_hash"]:
        raise ValueError("replicated evidence predecessor hash mismatch")
    chunks = numerical["chunks"]
    failed = "execution_failure" in numerical
    decoded = validate_chunks(runtime, work, chunks, complete=not failed, return_decoded=not failed)
    if failed:
        failure = HMCCandidateExecutionFailure.from_payload(numerical["execution_failure"])
        per_trial = (trial_size(runtime)+runtime.config.chunk_max_results-1)//runtime.config.chunk_max_results
        ordinal, within = work.trial_range[0]+len(chunks)//per_trial, len(chunks)%per_trial
        expected = chunk_seed(trial_seed(work_seed(runtime, work), ordinal), within)
        if (failure.chunk_index != len(chunks) or ordinal >= work.trial_range[1]
                or tuple(numerical["attempted_seed"]) != expected
                or numerical.get("samples") is not None or numerical.get("trace") is not None):
            raise ValueError("replicated failure inventory mismatch")
        batch = numerical.get("failed_batch_size", 1)
        expected_batch = min(runtime.config.replicated_trial_batch_size, work.trial_range[1]-ordinal)
        if type(batch) is not int or batch != expected_batch:
            raise ValueError("failed native batch extent mismatch")
        return _failure_analysis(runtime, work, failure, chunks, failed_batch_size=batch)
    trials = prior + _assemble_trials(runtime, work, chunks, decoded_chunks=decoded)
    if (len(trials) != work.trial_range[1] or _json_native_sha256(trials) != _json_native_sha256(numerical["trials"])
            or _sha256(numerical["samples"]) != _sha256(trials[-1]["samples"])
            or _sha256(numerical["trace"]) != _sha256(trials[-1]["trace"])):
        raise ValueError("replicated trial inventory or retained endpoint mismatch")
    schema = numerical["analysis"]["acceptance_evidence"]["schema"]
    if schema not in {"bayesfilter.hmc_replicated_acceptance_evidence.v1",
                      "bayesfilter.hmc_replicated_acceptance_evidence.v2"}:
        raise ValueError("unsupported replicated acceptance evidence summary")
    return _analyze_trials(runtime, work, trials, numerical["evidence_rungs"],
                           compact_health=schema.endswith(".v2"))


def observe(runtime, work, candidate):
    import tensorflow as tf
    from .hmc_candidate_set_execution import _tensor_payload, _tensor_from_payload, _trace_payload, _json_copy, _report_rhat
    _validate_work(runtime, work)
    rungs = runtime._replicated_evidence_rungs
    if work.evidence_multiplier != rungs[work.evidence_rung]:
        raise ValueError("work differs from declared repetition schedule")
    completed = [(digest, row) for digest, row in runtime._evidence.items()
                 if row["work"]["work_item_id"] == work.work_item_id]
    if len(completed) > 1:
        raise ValueError("duplicate completed trial allocation")
    if completed:
        digest, numerical = completed[0]
        if _json_native_sha256(numerical) != digest or _sha256(runtime.evidence_analysis(numerical)) != _sha256(numerical["analysis"]):
            raise ValueError("completed trial allocation cannot be replayed")
        return _observation(work, numerical, digest)
    prior, predecessor = _prior(runtime, work)
    chunks = runtime._partial.setdefault(work.work_item_id, [])
    validate_chunks(runtime, work, chunks)
    total, size = trial_size(runtime), runtime.config.chunk_max_results
    per_trial = (total+size-1)//size
    maximum = (work.trial_range[1]-work.trial_range[0])*per_trial
    root = work_seed(runtime, work)
    failure = None
    failed_batch_size = 1
    while len(chunks) < maximum:
        ordinal, within = work.trial_range[0]+len(chunks)//per_trial, len(chunks)%per_trial
        state = runtime.initial_active_state if within == 0 else _tensor_from_payload(chunks[-1]["samples"])[-1]
        take = min(size, total-within*size)
        seed = chunk_seed(trial_seed(root, ordinal), within)
        batched = runtime.config.replicated_trial_batch_size > 1
        batch = min(runtime.config.replicated_trial_batch_size, maximum-len(chunks)) if batched else 1
        seeds = [chunk_seed(trial_seed(root, ordinal+i), 0) for i in range(batch)] if batched else [seed]
        with grouped_incremental_chunk_checkpoint(runtime):
            for index, attempt_seed in enumerate(seeds):
                _register_chunk_seed(runtime, work, attempt_seed, ordinal+index, within)
                before_numerical_chunk(runtime, work, candidate, count=take,
                    chains=runtime.config.acceptance_policy.chain_count, index=len(chunks)+index,
                    seed=attempt_seed, trial_ordinal=ordinal+index, trial_chunk_index=within)
        started = time.monotonic()
        try:
            outputs = (runtime._run_replicated_batch(candidate, take, seeds) if batched else
                       [runtime._run(candidate, state, take, seed)])
        except (tf.errors.ResourceExhaustedError, tf.errors.UnavailableError,
                tf.errors.DeadlineExceededError, tf.errors.AbortedError) as exc:
            raise HMCInfrastructureFailure(type(exc).__name__ + ": " + str(exc)) from exc
        except Exception as exc:
            classifier = getattr(runtime._base_adapter, "classify_target_exception", None)
            classified = classifier(exc) if callable(classifier) else False
            if type(classified) is not bool:
                raise TypeError("classify_target_exception must return a boolean")
            if not classified:
                raise
            failure = HMCCandidateExecutionFailure(type(exc).__name__, str(exc), len(chunks))
            failed_batch_size = batch
            break
        if len(outputs) != batch:
            raise HMCSharedInvalidity("native trial batch returned an incorrect number of paths")
        native_seconds = time.monotonic()-started
        with grouped_incremental_chunk_checkpoint(runtime):
            for index, result in enumerate(outputs):
                persist_started = time.monotonic()
                chunk = {"count": take, "seed": seeds[index], "initial_state": _tensor_payload(state),
                    "work": work.payload(), "binding_hash": runtime.binding_hash,
                    "trial_ordinal": ordinal+index, "trial_chunk_index": within,
                    "samples": _tensor_payload(result.samples), "trace": _trace_payload(result.trace),
                    "samples_device": result.samples.device, "runtime": result.metadata}
                chunk["elapsed_seconds"] = native_seconds/batch + time.monotonic()-persist_started
                chunks.append(_json_copy(chunk))
    numerical = {"schema": NUMERICAL_SCHEMA, "binding_hash": runtime.binding_hash,
        "candidate": candidate.payload(), "work": work.payload(), "seed": root,
        "initial_state": _tensor_payload(runtime.initial_active_state), "chunks": chunks,
        "predecessor_evidence_hash": predecessor, "evidence_rungs": rungs}
    if failure is not None:
        analysis = _failure_analysis(runtime, work, failure, chunks, failed_batch_size=failed_batch_size)
        numerical.update(execution_failure=failure.payload(), attempted_seed=seed, samples=None, trace=None)
        if runtime.config.replicated_trial_batch_size > 1:
            numerical["failed_batch_size"] = failed_batch_size
        rhat = {"status": "unavailable_after_execution_failure", "role": "reporting_only"}
    else:
        decoded = validate_chunks(runtime, work, chunks, complete=True, return_decoded=True)
        trials = prior + _assemble_trials(runtime, work, chunks, decoded_chunks=decoded)
        analysis = _analyze_trials(runtime, work, trials, rungs)
        numerical.update(trials=trials, samples=trials[-1]["samples"], trace=trials[-1]["trace"])
        rhat = _report_rhat(_tensor_from_payload(trials[-1]["samples"])[runtime.config.num_warmup_steps:])
    numerical["rhat_reporting_only"] = rhat
    numerical["analysis"] = analysis
    numerical["elapsed_seconds"] = sum(c["elapsed_seconds"] for c in chunks)
    numerical["cost"] = work_cost(runtime, work, candidate, remaining=False)
    numerical["samples_device"] = chunks[-1]["samples_device"] if chunks else "unavailable"
    digest = _json_native_sha256(numerical)
    runtime._evidence[digest] = _json_copy(numerical)
    analysis_cache = getattr(runtime, "_analysis_cache", None)
    if failure is None and analysis_cache is not None:
        # Emission used the same checked raw chunks and numerical analysis as
        # replay. Reuse that result under its finished content digest. Later
        # reads still hash current content, and fresh bindings reconstruct it.
        # The position-field mechanics runtime has no analysis cache.
        analysis_cache[digest] = _json_copy(analysis)
    runtime._partial.pop(work.work_item_id, None)
    if runtime._checkpoint_callback is not None:
        runtime._checkpoint_callback()
    return _observation(work, numerical, digest)


def _observation(work, numerical, digest):
    return {**numerical["analysis"], "numerical_evidence_hash": digest,
            "stream_id": work.work_item_id+":"+_sha256({"seed": tuple(numerical["seed"])}),
            "draw_range": (0, 0), "seed_lineage": tuple(numerical["seed"]),
            "rhat_reporting_only": numerical["rhat_reporting_only"],
            "trial_range": work.trial_range, "evidence_unit": "independent_fixed_horizon_trial"}

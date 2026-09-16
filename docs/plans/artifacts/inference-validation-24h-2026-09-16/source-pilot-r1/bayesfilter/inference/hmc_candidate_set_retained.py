"""Validated numerical member replay and durable retained HMC continuation.

All three public builders share this implementation. Claim eligibility is a
backend/target policy check, never posterior convergence or scientific proof.
"""
from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_candidate_runtime import tuning_seed_inventory

from bayesfilter.inference.hmc_candidate_set_artifacts import (
    candidate_set_result_payload, load_candidate_set_result_payload, require_verified_member,
)
from bayesfilter.inference.hmc_candidate_set_execution import (
    EXECUTION_SCHEMA, HMCCandidateExecutionBinding, _ISSUER, _json_copy, _positive_int,
    _seed, _tensor_payload, _tensor_from_payload, _trace_payload, _trace_from_payload, _report_rhat,
)
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCTuningCandidateSetResult, HMCTuningCandidateRecord, HMCWorkItem, _sha256,
)

RETAINED_MEMBER_SCHEMA = "bayesfilter.hmc_candidate_retained_member.v1"
RETAINED_ARCHIVE_SCHEMA = "bayesfilter.hmc_candidate_retained_archive.v1"


def _write_new(payload: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Versioned artifacts are never overwritten, including on retry.
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return destination


def _checked_payload(path: str | Path, schema: str) -> Mapping[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = payload.pop("content_hash", None)
    if payload.get("schema") != schema or digest != _sha256(payload):
        raise ValueError("retained artifact schema/checksum mismatch")
    return payload


def _result_payload(result: Any) -> Mapping[str, Any]:
    if isinstance(result, HMCTuningCandidateSetResult):
        return _json_copy(candidate_set_result_payload(result))
    if isinstance(result, (str, Path)):
        return load_candidate_set_result_payload(result)
    if isinstance(result, Mapping):
        return _json_copy(result)
    raise TypeError("supply a candidate-set result or its checked durable payload")


def _validate_member(result: Any, candidate_id: str, binding: HMCCandidateExecutionBinding) -> tuple[Any, Any, Any]:
    if type(binding) is not HMCCandidateExecutionBinding:
        raise TypeError("retained_binding must be a repository-issued execution binding")
    binding.validate()
    for digest, evidence in binding._evidence.items():
        if _sha256(evidence) != digest or evidence.get("binding_hash") != binding.binding_hash:
            raise ValueError("corrupt numerical evidence inventory")
    payload = _result_payload(result)
    record = require_verified_member(payload, scope_id=binding.scope.scope_id,
        candidate_id=candidate_id, expected_scope=_json_copy(binding.scope.payload()))
    selected = HMCTuningCandidateRecord.from_payload(binding.scope, record)
    endpoint = None
    work_items = {item["work_item_id"]: item for item in payload["work_items"]}
    # Validate every receipt, including failed directional ancestors. The
    # selected child's own fresh verification must pass; ancestors need not.
    for receipt in payload["verification_receipts"]:
        digest = receipt.get("numerical_evidence_hash")
        numerical = binding._evidence.get(digest)
        if numerical is None or _sha256(numerical) != digest:
            raise ValueError("missing or corrupt repository numerical evidence")
        if numerical["binding_hash"] != binding.binding_hash:
            raise ValueError("numerical evidence binding mismatch")
        candidate = HMCTuningCandidateRecord.from_payload(binding.scope, numerical["candidate"])
        if any(receipt[key] != expected for key, expected in {
            "candidate_id": candidate.candidate_id, "candidate_record_hash": candidate.candidate_record_hash,
            "exact_l": candidate.leapfrog_steps, "epsilon": candidate.epsilon,
            "mass_signature": candidate.mass_signature,
        }.items()):
            raise ValueError("numerical receipt candidate mismatch")
        work = HMCWorkItem.from_payload(numerical["work"])
        issued_work = work_items.get(work.work_item_id)
        if issued_work is None or issued_work["status"] != "completed":
            raise ValueError("numerical evidence work is not completed")
        # Captured work is running; the controller subsequently closes it.
        for key, value in work.payload().items():
            if key != "status" and _json_copy(value) != issued_work[key]:
                raise ValueError("numerical work identity mismatch")
        seed = binding.work_seed(work)
        if list(seed) != numerical["seed"] or list(seed) != receipt["seed_lineage"]:
            raise ValueError("numerical evidence seed mismatch")
        if receipt["stream_id"] != work.work_item_id + ":" + _sha256({"seed": seed}):
            raise ValueError("numerical evidence stream mismatch")
        attempt = work.verification_attempt_id or work.work_item_id + ":attempt"
        if receipt["verification_attempt_id"] != attempt:
            raise ValueError("numerical evidence attempt mismatch")
        count = binding.work_count(work)
        warmup = binding.config.num_warmup_steps
        failed_execution = "execution_failure" in numerical
        if receipt["draw_range"] != ([0, 0] if failed_execution else [warmup, warmup + count]):
            raise ValueError("numerical draw range mismatch")
        initial = _tensor_from_payload(numerical["initial_state"])
        if _tensor_payload(initial) != _tensor_payload(binding.initial_active_state):
            raise ValueError("numerical initial state mismatch")
        samples = None if failed_execution else _tensor_from_payload(numerical["samples"])
        if samples is not None and samples.shape[0] != count + warmup:
            raise ValueError("numerical draw count mismatch")
        analysis = binding.evidence_analysis(numerical)
        if _json_copy(analysis) != numerical["analysis"]:
            raise ValueError("numerical evidence recomputation mismatch")
        if analysis["evidence_validity"] == "shared_execution_invalid":
            raise ValueError("shared-invalid numerical evidence disables scope replay")
        for key in ("decision", "acceptance", "hard_vetoes", "evidence_validity", "promotion_vetoes", "repair_eligible"):
            if _json_copy(analysis[key]) != receipt[key]:
                raise ValueError("numerical receipt decision mismatch")
        if (candidate.candidate_id == candidate_id and work.stage == "verification"
                and analysis["decision"] == "passed" and not analysis["hard_vetoes"]
                and not analysis["promotion_vetoes"] and analysis["evidence_validity"] == "valid"):
            endpoint = samples[-1]
    if endpoint is None:
        raise ValueError("selected member lacks passing fresh numerical verification")
    return payload, selected, endpoint


def _run_sequential_member(*, binding, candidate, initial_state, model_transform, config, **kwargs):
    """Delegate the checked member to the repository posterior controller."""
    from bayesfilter.inference.neutra_hmc import (
        run_sequential_neutra_hmc, BatchedHMCConfig, _summarize_batched_hmc_output,
    )
    def run_chunk(state, count, seed, checkpoint_store, checkpoint_index):
        started = time.monotonic()
        runtime = {"checkpoint_reused": True}
        binding.validate()
        def execute():
            result = binding._run(candidate, state, count, seed)
            runtime.update(checkpoint_reused=False, native=result.metadata)
            runtime["call_class"] = ("first_compile_plus_execute" if result.metadata.get("ensemble_call_count")==1
                                     else "repeated_execute")
            return result.samples, result.trace
        if checkpoint_store is None:
            samples, trace = execute()
        else:
            samples, trace = checkpoint_store.run(f"chunk-{checkpoint_index:06d}", {
                "binding_hash": binding.binding_hash,
                "candidate_record_hash": candidate.candidate_record_hash,
                "config": config.payload(chain_count=int(state.shape[0])),
                "active_results": count, "seed": seed,
                "initial_state_sha256": checkpoint_store.tensor_hash(state),
            }, execute)
        failures = binding.health_failures(state, samples, trace)
        chunk = _summarize_batched_hmc_output(initial_state=state,
            samples=samples, trace=trace, chain_count=int(state.shape[0]),
            elapsed_seconds=time.monotonic() - started, config=BatchedHMCConfig(num_results=count, num_burnin_steps=0,
                step_size=candidate.epsilon, num_leapfrog_steps=candidate.leapfrog_steps,
                seed=seed, jit_compile=config.jit_compile,
                energy_error_log_accept_threshold=config.energy_error_log_accept_threshold))
        chunk["diagnostics"].update(single_batched_sample_chain_invocation=False,
            execution_topology=binding.config.chain_mode,
            target_status_trace_policy=binding.config.target_status_trace_policy,
            runtime=runtime,
            member_health_failures=failures,
            health_passed=chunk["diagnostics"]["health_passed"] and not failures)
        chunk["trace"] = trace
        return chunk
    return run_sequential_neutra_hmc(adapter=binding._active_adapter,
        initial_state=initial_state, model_transform=model_transform, config=config,
        _member_runner=run_chunk, **kwargs)


class HMCCandidateRetainedRunner:
    """Exact frozen kernel, selected explicitly from a numerically verified set.

    ``run`` writes active and position samples and their exact endpoint. Pass
    ``previous_archive`` to continue after process exit. No adaptation occurs.
    """

    def __init__(self, token: Any, *, result: Any, candidate_id: str,
                 binding: HMCCandidateExecutionBinding, claim_eligible: bool = False) -> None:
        if token is not _ISSUER:
            raise ValueError("retained runners must be repository-issued")
        payload, candidate, endpoint = _validate_member(result, candidate_id, binding)
        if claim_eligible and (not binding.config.use_xla or binding._runtime["device_type"] != "GPU"):
            raise ValueError("CPU or non-XLA exception is mechanics-only, not claim eligible")
        if claim_eligible and any("execution_failure" not in evidence and "GPU" not in evidence.get("samples_device", "")
                                  for evidence in binding._evidence.values()):
            raise ValueError("claim eligibility requires actual GPU verification evidence")
        self._result = payload
        self._binding = binding
        self.candidate = candidate
        self.initial_active_state = endpoint
        self._claim_eligible = bool(claim_eligible)
        self.member_hash = _sha256({"result_hash": payload["result_hash"],
            "candidate_record_hash": candidate.candidate_record_hash, "binding_hash": binding.binding_hash})

    @property
    def numerical_handoff_authority(self) -> bool:
        return True

    @property
    def artifact_authority(self) -> bool:
        return True

    @property
    def posterior_convergence_authority(self) -> bool:
        return False

    @property
    def claim_eligible(self) -> bool:
        return self._claim_eligible

    @property
    def step_size(self) -> float:
        return self.candidate.epsilon

    @property
    def num_leapfrog_steps(self) -> int:
        return self.candidate.leapfrog_steps

    def _tuning_seeds(self) -> set[tuple[int, int]]:
        works = {row["work_item_id"]: HMCWorkItem.from_payload(row)
                 for row in self._result["work_items"]}
        attempts = ((self._binding.work_seed(works[event["work_item_id"]]), event["chunk_index"])
                    for event in self._result["accounting_events"]
                    if event["event"] == "numerical_chunk_charged")
        return tuning_seed_inventory(self._binding._evidence.values(), self._binding._partial.values(),
                                     attempted_chunks=attempts)

    def run_sequential(self, *, config: Any, model_transform: Any = None, **kwargs: Any) -> Any:
        """Assess one explicitly selected member with the posterior controller.

        Model diagnostics receive the reconstructed original coordinates. The
        posterior controller owns discarded warmup and cumulative R-hat/ESS;
        none of these diagnostics modifies tuning membership.
        """
        self._validate()
        if (config.step_size != self.candidate.epsilon or
                config.num_leapfrog_steps != self.candidate.leapfrog_steps):
            raise ValueError("sequential sampling must preserve the verified candidate kernel")
        if config.jit_compile != self._binding.config.use_xla:
            raise ValueError("sequential execution must preserve the qualified XLA policy")
        from bayesfilter.inference.hmc_posterior_assessment import validate_sequential_seeds
        validate_sequential_seeds(config, forbidden=self._tuning_seeds())
        if any(key in kwargs for key in ("adapter", "initial_state", "_member_runner")):
            raise ValueError("the verified member supplies the sequential adapter and initial state")
        def transform(draws):
            raw = self._binding.position_samples(draws)
            return raw if model_transform is None else model_transform(raw)
        return _run_sequential_member(binding=self._binding, candidate=self.candidate,
            initial_state=self.initial_active_state, model_transform=transform, config=config, **kwargs)

    def export(self, path: str | Path, *, portable: bool = False) -> Path:
        """Write a compact member sharing an evidence bundle, or a portable file.

        Copy the shared bundle with compact member files when moving them.
        ``portable=True`` embeds all evidence for a standalone export.
        """
        self._validate()
        common = {"candidate_set_result": self._result,
            "execution": self._binding._spec, "binding_hash": self._binding.binding_hash,
            "numerical_evidence": self._binding._evidence}
        payload = {"schema": RETAINED_MEMBER_SCHEMA, "member_hash": self.member_hash,
            "candidate_id": self.candidate.candidate_id,
            "verified_endpoint": _tensor_payload(self.initial_active_state)}
        if portable:
            payload.update(common)
        else:
            from bayesfilter.inference.hmc_candidate_set_checkpoint import _persist_once
            bundle = {"schema": "bayesfilter.hmc_candidate_evidence_bundle.v1", **common}
            digest = _sha256(bundle)
            bundle_path = Path(path).parent / ("evidence-" + digest + ".json")
            _persist_once(self._binding, {**bundle, "content_hash": digest}, bundle_path)
            payload.update(schema="bayesfilter.hmc_candidate_retained_member.v2",
                evidence_bundle={"path": bundle_path.name, "content_hash": digest})
        return _write_new({**payload, "content_hash": _sha256(payload)}, path)

    def _validate(self) -> None:
        payload, candidate, endpoint = _validate_member(self._result, self.candidate.candidate_id, self._binding)
        expected = _sha256({"result_hash": payload["result_hash"],
            "candidate_record_hash": candidate.candidate_record_hash, "binding_hash": self._binding.binding_hash})
        if (candidate != self.candidate or expected != self.member_hash
                or _tensor_payload(endpoint) != _tensor_payload(self.initial_active_state)):
            raise ValueError("retained member mutated")
        if self.claim_eligible and (not self._binding.config.use_xla
                or self._binding._runtime["device_type"] != "GPU"
                or any("execution_failure" not in evidence and "GPU" not in evidence.get("samples_device", "") for evidence in self._binding._evidence.values())):
            raise ValueError("retained claim eligibility does not match numerical evidence")

    def _archive(self, path: str | Path) -> tuple[Mapping[str, Any], Any]:
        import tensorflow as tf

        visited = set()
        tuning_seeds = self._tuning_seeds()
        newest = child = None
        while True:
            resolved = str(Path(path).resolve())
            if resolved in visited:
                raise ValueError("retained predecessor cycle")
            visited.add(resolved)
            payload = _checked_payload(path, RETAINED_ARCHIVE_SCHEMA)
            if (payload["member_hash"] != self.member_hash or payload["binding_hash"] != self._binding.binding_hash
                    or payload["candidate"] != _json_copy(self.candidate.payload())):
                raise ValueError("retained archive kernel/member mismatch")
            if payload["health_failures"]:
                raise ValueError("retained archive has failed numerical health")
            samples = _tensor_from_payload(payload["active_samples"])
            initial = _tensor_from_payload(payload["initial_active_state"])
            endpoint = _tensor_from_payload(payload["final_active_state"])
            if samples.shape[0] != payload["num_results"] or _tensor_payload(samples[-1]) != _tensor_payload(endpoint):
                raise ValueError("retained archive endpoint/count mismatch")
            trace = _trace_from_payload(payload["trace"])
            if self._binding.health_failures(initial, samples, trace):
                raise ValueError("retained archive has failed numerical health")
            raw = _tensor_from_payload(payload["position_samples"])
            if not bool(tf.reduce_all(tf.equal(raw, self._binding.position_samples(samples)))):
                raise ValueError("retained position samples disagree with frozen geometry")
            expected_first = _tensor_payload(self.initial_active_state)
            if payload["predecessor"] is None and payload["initial_active_state"] != expected_first:
                raise ValueError("first retained archive does not start at verification endpoint")
            seeds = payload["seed_history"]
            if not seeds or seeds[-1] != payload["seed"] or len({tuple(seed) for seed in seeds}) != len(seeds):
                raise ValueError("invalid retained seed history")
            for seed in seeds:
                _seed(seed)
            if any(tuple(seed) in tuning_seeds for seed in seeds):
                raise ValueError("retained archive seeds must be fresh relative to all tuning chunks")
            if newest is None:
                newest = payload, endpoint
            if child is not None:
                parent = child["predecessor"]
                if (parent["content_hash"] != _sha256(payload)
                        or parent["final_active_state_hash"] != payload["final_active_state"]["sha256"]
                        or _tensor_payload(endpoint) != child["initial_active_state"]
                        or child["seed_history"] != [*seeds, child["seed"]]):
                    raise ValueError("retained predecessor state or history mismatch")
            if payload["predecessor"] is None:
                if seeds != [payload["seed"]]:
                    raise ValueError("first retained archive has invalid history")
                return newest
            child = payload
            path = payload["predecessor"]["path"]

    def run(self, *, num_results: int, seed: tuple[int, int], output_dir: str | Path,
            previous_archive: str | Path | None = None) -> Mapping[str, Any]:
        """Run a fresh retained block, optionally from a preceding block's endpoint.

        Acceptance and R-hat are reported; posterior admission belongs to the
        consumer's sequential convergence and scientific validation procedure.
        """
        from bayesfilter.inference.hmc import _rhat_summary_from_retained_samples
        from bayesfilter.inference.tuning_contract import HMC_TUNING_ORDINARY_RHAT_THRESHOLD
        import tensorflow as tf

        _positive_int(num_results, "num_results")
        seed = _seed(seed)
        self._validate()
        predecessor = None
        state = self.initial_active_state
        history = []
        if previous_archive is not None:
            previous, state = self._archive(previous_archive)
            history = previous["seed_history"]
            predecessor = {"path": str(Path(previous_archive).resolve()), "content_hash": _sha256(previous),
                           "final_active_state_hash": previous["final_active_state"]["sha256"]}
        tuning_seeds = self._tuning_seeds()
        if seed in tuning_seeds or list(seed) in history:
            raise ValueError("retained seed must be fresh relative to tuning and predecessor blocks")
        destination = Path(output_dir) / "retained_archive.json"
        if destination.exists():
            raise FileExistsError(destination)
        result = self._binding._run(self.candidate, state, num_results, seed)
        failures = self._binding.health_failures(state, result.samples, result.trace)
        finite_samples = bool(tf.reduce_all(tf.math.is_finite(result.samples)))
        probability = tf.reduce_mean(tf.exp(tf.minimum(result.trace["log_accept_ratio"], 0.0)))
        payload = {"schema": RETAINED_ARCHIVE_SCHEMA, "member_hash": self.member_hash,
            "binding_hash": self._binding.binding_hash, "candidate": self.candidate.payload(),
            "num_results": num_results, "seed": seed, "seed_history": [*history, list(seed)],
            "predecessor": predecessor, "initial_active_state": _tensor_payload(state),
            "active_samples": _tensor_payload(result.samples),
            "position_samples": _tensor_payload(self._binding.position_samples(result.samples)) if finite_samples else None,
            "final_active_state": _tensor_payload(result.samples[-1]), "trace": _trace_payload(result.trace),
            "health_failures": failures, "claim_eligible": self.claim_eligible,
            "acceptance_reporting_only": {
                "mean_probability": float(probability) if bool(tf.math.is_finite(probability)) else None,
                "realized_rate": float(tf.reduce_mean(tf.cast(result.trace["is_accepted"], tf.float64)))},
            "rhat_reporting_only": _report_rhat(result.samples) if finite_samples else None,
            "runtime": result.metadata,
            "posterior_convergence_authority": False,
            "warmup_draws_included": False, "tuning_draws_included": False}
        payload = _json_copy(payload)
        _write_new({**payload, "content_hash": _sha256(payload)}, destination)
        if failures:
            raise ValueError("retained numerical health failed; preserved archive: " + str(destination))
        return {"archive_path": str(destination.resolve()), "content_hash": _sha256(payload),
                "member_hash": self.member_hash, "num_results": num_results,
                "final_active_state": result.samples[-1], "samples": result.samples,
                "position_samples": self._binding.position_samples(result.samples),
                "rhat_reporting_only": payload["rhat_reporting_only"]}


def build_retained_bound_hmc_archive_runner_from_candidate_set_result(*, candidate_set_result: Any,
        candidate_id: str, retained_binding: HMCCandidateExecutionBinding) -> HMCCandidateRetainedRunner:
    return HMCCandidateRetainedRunner(_ISSUER, result=candidate_set_result,
        candidate_id=candidate_id, binding=retained_binding)


def build_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(*, candidate_set_result: Any,
        candidate_id: str, retained_binding: HMCCandidateExecutionBinding) -> HMCCandidateRetainedRunner:
    return build_retained_bound_hmc_archive_runner_from_candidate_set_result(
        candidate_set_result=candidate_set_result, candidate_id=candidate_id, retained_binding=retained_binding)


def build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_candidate_set_result(*, candidate_set_result: Any,
        candidate_id: str, retained_binding: HMCCandidateExecutionBinding) -> HMCCandidateRetainedRunner:
    """Require exact target/XLA eligibility; this does not certify a posterior."""
    return HMCCandidateRetainedRunner(_ISSUER, result=candidate_set_result,
        candidate_id=candidate_id, binding=retained_binding, claim_eligible=True)


def load_hmc_candidate_retained_runner(path: str | Path, *, adapter: Any,
        claim_eligible: bool = False) -> HMCCandidateRetainedRunner:
    """Reload using the original target; BayesFilter reconstructs frozen geometry."""
    schema = json.loads(Path(path).read_text()).get("schema")
    if schema not in {RETAINED_MEMBER_SCHEMA, "bayesfilter.hmc_candidate_retained_member.v2"}:
        raise ValueError("retained member schema mismatch")
    payload = dict(_checked_payload(path, schema))
    if schema != RETAINED_MEMBER_SCHEMA:
        reference = payload["evidence_bundle"]
        bundle_path = Path(path).parent / reference["path"]
        bundle = _checked_payload(bundle_path, "bayesfilter.hmc_candidate_evidence_bundle.v1")
        if _sha256(bundle) != reference["content_hash"]:
            raise ValueError("retained evidence bundle checksum mismatch")
        for key in ("execution", "binding_hash", "candidate_set_result", "numerical_evidence"):
            payload[key] = bundle[key]
    spec = payload["execution"]
    if spec.get("schema") != EXECUTION_SCHEMA or _sha256(spec) != payload["binding_hash"]:
        raise ValueError("retained execution binding mismatch")
    binding = HMCCandidateExecutionBinding(_ISSUER, adapter=adapter, spec=spec)
    binding._evidence = payload["numerical_evidence"]
    runner = HMCCandidateRetainedRunner(_ISSUER, result=payload["candidate_set_result"],
        candidate_id=payload["candidate_id"], binding=binding, claim_eligible=claim_eligible)
    if runner.member_hash != payload["member_hash"] or _tensor_payload(runner.initial_active_state) != payload["verified_endpoint"]:
        raise ValueError("retained member or verified endpoint mismatch")
    return runner

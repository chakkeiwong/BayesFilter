"""Validated numerical member replay and durable retained HMC continuation.

All three public builders share this implementation. Claim eligibility is a
backend/target policy check, never posterior convergence or scientific proof.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesfilter.inference.hmc_candidate_set_artifacts import (
    candidate_set_result_payload, load_candidate_set_result_payload, require_verified_member,
)
from bayesfilter.inference.hmc_candidate_set_execution import (
    EXECUTION_SCHEMA, HMCCandidateExecutionBinding, _ISSUER, _json_copy, _positive_int,
    _seed, _tensor_payload, _tensor_from_payload, _trace_payload, _trace_from_payload,
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
        count = getattr(binding.config, work.stage + "_num_results")
        warmup = binding.config.num_warmup_steps
        if receipt["draw_range"] != [warmup, warmup + count]:
            raise ValueError("numerical draw range mismatch")
        initial = _tensor_from_payload(numerical["initial_state"])
        if _tensor_payload(initial) != _tensor_payload(binding.initial_active_state):
            raise ValueError("numerical initial state mismatch")
        samples = _tensor_from_payload(numerical["samples"])
        if samples.shape[0] != count + warmup:
            raise ValueError("numerical draw count mismatch")
        trace = _trace_from_payload(numerical["trace"])
        analysis = binding.analyze(initial, samples, trace)
        if _json_copy(analysis) != numerical["analysis"]:
            raise ValueError("numerical evidence recomputation mismatch")
        for key in ("decision", "acceptance", "hard_vetoes"):
            if _json_copy(analysis[key]) != receipt[key]:
                raise ValueError("numerical receipt decision mismatch")
        if (candidate.candidate_id == candidate_id and work.stage == "verification"
                and analysis["decision"] == "passed" and not analysis["hard_vetoes"]):
            endpoint = samples[-1]
    if endpoint is None:
        raise ValueError("selected member lacks passing fresh numerical verification")
    return payload, selected, endpoint


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
        if claim_eligible and any("GPU" not in evidence.get("samples_device", "")
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

    def export(self, path: str | Path) -> Path:
        """Persist geometry, complete candidate result, traces and verified state."""
        self._validate()
        payload = {"schema": RETAINED_MEMBER_SCHEMA, "member_hash": self.member_hash,
            "candidate_id": self.candidate.candidate_id, "candidate_set_result": self._result,
            "execution": self._binding._spec, "binding_hash": self._binding.binding_hash,
            "numerical_evidence": self._binding._evidence,
            "verified_endpoint": _tensor_payload(self.initial_active_state)}
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
                or any("GPU" not in evidence.get("samples_device", "") for evidence in self._binding._evidence.values())):
            raise ValueError("retained claim eligibility does not match numerical evidence")

    def _archive(self, path: str | Path, seen: frozenset[str] = frozenset()) -> tuple[Mapping[str, Any], Any]:
        import tensorflow as tf

        resolved = str(Path(path).resolve())
        if resolved in seen:
            raise ValueError("retained predecessor cycle")
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
        if payload["predecessor"] is not None:
            parent = payload["predecessor"]
            previous, previous_endpoint = self._archive(parent["path"], seen | {resolved})
            if (parent["content_hash"] != _sha256(previous)
                    or parent["final_active_state_hash"] != previous["final_active_state"]["sha256"]
                    or _tensor_payload(previous_endpoint) != payload["initial_active_state"]
                    or seeds != [*previous["seed_history"], payload["seed"]]):
                raise ValueError("retained predecessor state or history mismatch")
        elif seeds != [payload["seed"]]:
            raise ValueError("first retained archive has invalid history")
        return payload, endpoint

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
        tuning_seeds = {tuple(value["seed"]) for value in self._binding._evidence.values()}
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
            "rhat_reporting_only": _rhat_summary_from_retained_samples(
                result.samples, threshold=HMC_TUNING_ORDINARY_RHAT_THRESHOLD) if finite_samples else None,
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
    payload = _checked_payload(path, RETAINED_MEMBER_SCHEMA)
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

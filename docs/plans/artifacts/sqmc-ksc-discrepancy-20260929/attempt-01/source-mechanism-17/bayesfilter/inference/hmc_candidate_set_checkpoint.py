"""Durable numerical tuning checkpoints, including unsuccessful searches.

Immutable evidence files are shared by successive atomic checkpoints. A new
search requires a fresh output directory; resume replaces only its checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from bayesfilter.inference.hmc_candidate_set_artifacts import (
    candidate_set_result_payload, _validate_result_payload,
)
from bayesfilter.inference.hmc_candidate_set_tuning import (
    HMCTuningCandidateSetController, HMCTuningCandidateRecord, HMCWorkItem, _sha256,
)

CHECKPOINT_SCHEMA = "bayesfilter.hmc_numerical_tuning_checkpoint.v1"


def _json_native_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a record already normalized by the execution binding's JSON copy.

    These records contain string-key dictionaries, lists and JSON scalars only.
    Recheck their current contents on every call, without repeating the generic
    arbitrary-object normalization. General candidate identity hashing must
    continue to use _sha256.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _persist_once(binding, payload, path):
    """Avoid reparsing/rewriting unchanged files; recheck external changes."""
    key = str(path.resolve())
    stat = path.stat() if path.exists() else None
    signature = None if stat is None else (stat.st_mtime_ns, stat.st_size)
    if signature is not None and binding._persisted_files.get(key) == signature:
        return
    _write(payload, path)
    stat = path.stat()
    binding._persisted_files[key] = (stat.st_mtime_ns, stat.st_size)


def _write(payload: Mapping[str, Any], path: Path, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if not replace:
        if path.exists():
            if json.loads(path.read_text()) != json.loads(encoded):
                raise ValueError("immutable numerical evidence collision: " + str(path))
            return
        with path.open("x") as stream:
            stream.write(encoded)
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(encoded)
        temporary.replace(path)


def write_numerical_tuning_checkpoint(binding: Any, result: Any, output_dir: str | Path) -> Path:
    root = Path(output_dir)
    if _json_native_sha256(binding._spec) != binding.binding_hash:
        raise ValueError("corrupt execution specification")
    _persist_once(binding, {"execution": binding._spec, "binding_hash": binding.binding_hash}, root / "execution_spec.json")
    for digest, evidence in binding._evidence.items():
        if _json_native_sha256(evidence) != digest:
            raise ValueError("corrupt live numerical evidence")
        _persist_once(binding, evidence, root / "numerical_evidence" / (digest + ".json"))
    partial = {}
    for work_id, chunks in binding._partial.items():
        partial[work_id] = []
        for chunk in chunks:
            digest = _json_native_sha256(chunk)
            _persist_once(binding, chunk, root / "numerical_chunks" / (digest + ".json"))
            partial[work_id].append(digest)
    body = {"schema": CHECKPOINT_SCHEMA, "binding_hash": binding.binding_hash,
            "result": candidate_set_result_payload(result),
            "numerical_evidence_hashes": list(binding._evidence), "partial_chunks": partial}
    destination = root / "tuning_checkpoint.json"
    _write({**body, "content_hash": _sha256(body)}, destination, replace=True)
    return destination


def load_numerical_tuning_checkpoint(path: str | Path, *, adapter: Any):
    from bayesfilter.inference.hmc_candidate_set_execution import (
        HMCCandidateExecutionBinding, _ISSUER, _tensor_from_payload, _tensor_payload, _trace_from_payload,
    )
    path = Path(path)
    payload = json.loads(path.read_text())
    digest = payload.pop("content_hash", None)
    if payload.get("schema") != CHECKPOINT_SCHEMA or digest != _sha256(payload):
        raise ValueError("numerical checkpoint schema/checksum mismatch")
    spec = json.loads((path.parent / "execution_spec.json").read_text())
    if spec["binding_hash"] != payload["binding_hash"] or _sha256(spec["execution"]) != spec["binding_hash"]:
        raise ValueError("checkpoint execution identity mismatch")
    binding = HMCCandidateExecutionBinding(_ISSUER, adapter=adapter, spec=spec["execution"])
    result = payload["result"]
    from bayesfilter.inference.hmc_candidate_set_artifacts import _canonical
    import hashlib
    body = dict(result)
    result_hash = body.pop("result_hash", None)
    if hashlib.sha256(_canonical(body)).hexdigest() != result_hash:
        raise ValueError("checkpoint result checksum mismatch")
    _validate_result_payload(result)
    controller = HMCTuningCandidateSetController.from_result_payload(result)
    if _sha256(controller.scope.payload()) != _sha256(binding.scope.payload()):
        raise ValueError("checkpoint scope mismatch")
    works = {w.work_item_id: w for w in controller.result().work_items}
    for evidence_hash in payload["numerical_evidence_hashes"]:
        evidence = json.loads((path.parent / "numerical_evidence" / (evidence_hash + ".json")).read_text())
        if _sha256(evidence) != evidence_hash or evidence["binding_hash"] != binding.binding_hash:
            raise ValueError("checkpoint numerical evidence checksum mismatch")
        work = HMCWorkItem.from_payload(evidence["work"])
        issued = works.get(work.work_item_id)
        if issued is None or any(_sha256(v) != _sha256(issued.payload()[k])
                                for k, v in work.payload().items() if k != "status"):
            raise ValueError("checkpoint numerical work identity mismatch")
        candidate = HMCTuningCandidateRecord.from_payload(binding.scope, evidence["candidate"])
        if candidate.candidate_record_hash != work.candidate_record_hash:
            raise ValueError("checkpoint numerical candidate mismatch")
        initial = _tensor_from_payload(evidence["initial_state"])
        if (_tensor_payload(initial) != _tensor_payload(binding.initial_active_state)
                or tuple(evidence["seed"]) != binding.work_seed(work)):
            raise ValueError("checkpoint numerical starts, seeds or count mismatch")
        if "execution_failure" not in evidence:
            samples = _tensor_from_payload(evidence["samples"])
            if samples.shape[0] != binding.work_count(work) + binding.config.num_warmup_steps:
                raise ValueError("checkpoint numerical count mismatch")
        if _sha256(binding.evidence_analysis(evidence)) != _sha256(evidence["analysis"]):
            raise ValueError("checkpoint numerical analysis mismatch")
        binding._evidence[evidence_hash] = evidence
    observed = set()
    for observation in result.get("observations", ()):
        key = observation["observation"].get("numerical_evidence_hash")
        if key not in binding._evidence:
            raise ValueError("checkpoint observation is missing numerical evidence")
        evidence = binding._evidence[key]
        raw = observation["observation"]
        if (observation["work_item_id"] in observed or observation["work_item_id"] != evidence["work"]["work_item_id"]
                or observation["candidate_id"] != evidence["candidate"]["candidate_id"]
                or observation["stage"] != evidence["work"]["stage"]
                or any(_sha256(raw.get(k)) != _sha256(v) for k, v in evidence["analysis"].items())):
            raise ValueError("checkpoint observation differs from numerical evidence")
        observed.add(observation["work_item_id"])
    if any(w.status == "completed" and w.work_item_id not in observed for w in works.values()):
        raise ValueError("checkpoint completed work is missing its observation")
    for work_id, hashes in payload["partial_chunks"].items():
        if work_id not in works or works[work_id].status == "completed":
            raise ValueError("checkpoint partial work identity mismatch")
        chunks = []
        state = binding.initial_active_state
        for chunk_hash in hashes:
            chunk = json.loads((path.parent / "numerical_chunks" / (chunk_hash + ".json")).read_text())
            if _sha256(chunk) != chunk_hash or _tensor_payload(state) != chunk["initial_state"]:
                raise ValueError("checkpoint chunk checksum or state continuity mismatch")
            if (chunk["binding_hash"] != binding.binding_hash
                    or any(_sha256(v) != _sha256(works[work_id].payload()[k])
                           for k, v in chunk["work"].items() if k != "status")):
                raise ValueError("checkpoint chunk work or binding mismatch")
            samples = _tensor_from_payload(chunk["samples"])
            if samples.shape[0] != chunk["count"]:
                raise ValueError("checkpoint chunk count mismatch")
            _trace_from_payload(chunk["trace"])
            state = samples[-1]
            chunks.append(chunk)
        if sum(c["count"] for c in chunks) > binding.work_count(works[work_id]) + binding.config.num_warmup_steps:
            raise ValueError("checkpoint exceeds declared evidence allocation")
        binding._partial[work_id] = chunks
    binding.validate()
    return binding, controller


def resume_hmc_candidate_set_tuning(path: str | Path, *, adapter: Any,
                                    max_work_items: int | None = None):
    """Resume unchanged numerical tuning and return the same public run type."""
    from bayesfilter.inference.hmc_candidate_set_adapters import run_typed_hmc_candidate_set
    binding, controller = load_numerical_tuning_checkpoint(path, adapter=adapter)
    return run_typed_hmc_candidate_set(binding.typed_adapter, controller.config,
        output_dir=Path(path).parent, max_work_items=max_work_items, _resume_controller=controller)

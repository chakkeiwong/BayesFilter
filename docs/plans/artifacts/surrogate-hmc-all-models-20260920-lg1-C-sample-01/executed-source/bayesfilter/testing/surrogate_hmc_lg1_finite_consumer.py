"""Scope-checked C consumer for the independently qualified finite DCTR target."""
import hashlib
import json
from pathlib import Path

from bayesfilter.inference import bind_neural_force_hmc_tuning_runner
from .surrogate_hmc_lg1_scoped import prepare_consumer
from .surrogate_hmc_lg1_finite_endpoint import make_finite_endpoint
from .surrogate_hmc_lg1_finite_reference import validate_endpoint_qualification


def prepare_finite_consumer(reference_root, selection, endpoint_root, finite_reference_root):
    consumer, provenance = prepare_consumer(reference_root,selection)
    endpoint, _, _ = make_finite_endpoint(consumer)
    qualification = validate_endpoint_qualification(endpoint_root,endpoint,provenance)
    path = Path(finite_reference_root)/"report.json"
    report = json.loads(path.read_text())
    if report.get("qualified") is not True or report["endpoint_identity"] != endpoint.identity:
        raise ValueError("qualified matching finite-posterior reference required")
    if report["qualification"] != qualification:
        raise ValueError("finite-posterior endpoint qualification drift")
    for key in ("selection_sha256","claim_data_sha256"):
        if report["provenance"][key] != provenance[key]:
            raise ValueError("finite-posterior data or controls drift")
    for name,digest in report["source_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError("finite-posterior reference source drift")
    ref_path=Path(reference_root)/"claim/reference.json"
    if hashlib.sha256(ref_path.read_bytes()).hexdigest()!=report["reference_sha256"]:
        raise ValueError("exact reference drift")
    for grid in report["grids"]:
        for record in grid["tensors"].values():
            if hashlib.sha256((Path(finite_reference_root)/record["path"]).read_bytes()).hexdigest()!=record["sha256"]:
                raise ValueError("finite reference tensor checksum mismatch")
    scope="LG1-attempt4-T10-N1024-claim-C-"+endpoint.identity
    binding=bind_neural_force_hmc_tuning_runner(force=consumer.force,target=endpoint,target_scope=scope)
    class Adapter:
        target_scope=scope
        def adapter_signature(self):
            return scope+":"+binding.binding_hash
    provenance.update(finite_reference_path=str(path.resolve()),finite_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      endpoint_qualification=qualification,endpoint_identity=endpoint.identity)
    return consumer,Adapter(),binding,report,provenance

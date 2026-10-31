"""Scope-checked LG1 experimental consumer, without exact-score admission."""
import hashlib
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.ledh_program_fixtures_tf import source_closure
from bayesfilter.inference.surrogate_hmc_dctr_selection import load_selection
from bayesfilter.inference import bind_neural_force_hmc_tuning_runner
from bayesfilter.testing.surrogate_hmc_lg1_tf import make_lg1_consumer


def prepare_consumer(reference_root, selection_path):
    root = Path(reference_root)
    report = json.loads((root / "report.json").read_text())
    if report.get("qualified") is not True:
        raise ValueError("qualified LG1 reference required")
    provenance = json.loads((root / "run-manifest.json").read_text())
    for name, digest in provenance["source_hashes"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise ValueError("LG1 reference source drift")
    scope = dict(model="LG1", horizon=10, attempt=4, state_dimension=1, parameter_dimension=1,
        particle_count=1024, chunk_extent=1024, dtype="float32", tf32=True, jit_compile=True,
        route="canonical_batch_fused_value_score", reset_contract_id="contract_e_chol_v1",
        law="stationary x0, Q1 R.5, uniform phi[-.95,.95]",
        data_regime="synthetic nominal LG1; separate calibration/validation/claim partitions",
        control_family="full_canonical_geometric_protection_curve_v1")
    hashes = source_closure()
    name = "bayesfilter/inference/surrogate_hmc_dctr_selection.py"
    hashes[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    selected = load_selection(selection_path, expected_scope=scope, expected_source_hashes=hashes)
    data_path = root / "claim/data.json"
    data = json.loads(data_path.read_text())
    reference = json.loads((root / "claim/reference.json").read_text())
    if (reference.get("qualified") is not True or reference.get("data_sha256") !=
            hashlib.sha256(data_path.read_bytes()).hexdigest()):
        raise ValueError("reference and inference data mismatch")
    if (data.get("attempt"), data.get("partition"), data.get("horizon"), data.get("model_id")) != (4, "claim", 10, "LG1"):
        raise ValueError("wrong untouched inference partition")
    consumer = make_lg1_consumer(data, batch_size=4, controls=selected["controls"])
    return consumer, dict(scope=scope, selection_path=str(Path(selection_path).resolve()),
        selection_sha256=hashlib.sha256(Path(selection_path).read_bytes()).hexdigest(),
        claim_data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        consumer=consumer.manifest)


def bind_arm(consumer, arm):
    if arm not in ("A", "B"):
        raise ValueError("this stage implements only exact-reference-endpoint A/B")
    scope = "LG1-attempt4-T10-N1024-claim-" + arm
    force = consumer.exact_force if arm == "A" else consumer.force
    binding = bind_neural_force_hmc_tuning_runner(force=force, target=consumer.endpoint,
                                                 target_scope=scope)
    class Adapter:
        target_scope = scope
        def adapter_signature(self):
            return scope + ":" + binding.binding_hash
    return Adapter(), binding

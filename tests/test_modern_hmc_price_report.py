"""Independent receipt/design mutations; no numerical or performance claims."""
from copy import deepcopy
import json

import pytest

from scripts.report_modern_hmc_frozen_prices import audit_call, check_design_reuse
from scripts.audit_mams_full_model_pricing import digest, payload_digest


def original_design():
    return dict(method="fixed_hmc", target="NL-PP-SGQF", seed=71410,
        geometry_directory="fixed-geometry", execution=dict(use_xla=True, seed=[1, 2]),
        search=dict(schema="bayesfilter.hmc_controller_config.v3",
            repair_trajectory_policy="preserve_mean_duration", repair_mean_duration_max_steps=128,
            epsilon_by_l=[[3, [.5]]], max_candidates=2),
        posterior=dict(mcse_sd_ratio_max=.02))


def corrected_design():
    result = original_design()
    result["search"].pop("repair_trajectory_policy")
    result["search"].pop("repair_mean_duration_max_steps")
    result["search"]["schema"] = "bayesfilter.hmc_controller_config.v2"
    return result


def test_only_documented_nonexecuted_fixed_repair_correction_is_reusable():
    assert check_design_reuse(original_design(), corrected_design()) == "fixed_count_repair_correction"
    assert check_design_reuse(corrected_design(), corrected_design()) == "identical_design"


@pytest.mark.parametrize("mutation", ["target", "start", "execution_seed", "geometry", "root",
    "posterior", "repair_cap", "other_method"])
def test_equal_pairs_alone_cannot_make_changed_design_eligible(mutation):
    old, new = original_design(), corrected_design()
    if mutation == "target": new["target"] = "NL-PP-UKF"
    elif mutation == "start": new["seed"] += 1
    elif mutation == "execution_seed": new["execution"]["seed"][1] += 1
    elif mutation == "geometry": new["geometry_directory"] = "new-geometry"
    elif mutation == "root": new["search"]["epsilon_by_l"][0][1][0] = 1.
    elif mutation == "posterior": new["posterior"]["mcse_sd_ratio_max"] = .04
    elif mutation == "repair_cap": new["search"]["max_candidates"] = 1
    elif mutation == "other_method": old["method"] = new["method"] = "mams"
    with pytest.raises(ValueError):
        check_design_reuse(old, new)


def write(path, value):
    path.write_text(json.dumps(value))


def call_fixture(directory):
    cell = dict(binding_hash="binding", initial_active_state_sha256="start")
    root = dict(index=0, candidate_id="candidate", transition_spec={"profile": "test"},
                epsilon=.5, leapfrog_steps=3)
    metadata = dict(value_score_row_evaluations=32, target_status_row_evaluations=32,
        maximum_reserved_gradient_work=80, use_xla=True, samples_device="GPU:0",
        transition_spec=root["transition_spec"])
    call = dict(repeat=0, seed=[1, 2], transitions=64, native=metadata)
    identity = dict(binding_hash="binding", role="cost_only")
    directory.mkdir()
    write(directory / "identity.json", identity)
    committed = directory / "committed/root-000-repeat-0"
    committed.mkdir(parents=True)
    attempt = directory / "attempt"
    attempt.mkdir()
    inputs = dict(candidate={key: root[key] for key in (
        "candidate_id", "transition_spec", "epsilon", "leapfrog_steps")},
        start_sha256="start", seed=[1, 2], transitions=64)
    write(attempt / "call-start.json", dict(inputs=inputs))
    (committed / "tensor.bin").write_bytes(b"independent-checksum-fixture")
    def tensor(shape):
        return dict(tensor=dict(file="tensor.bin", sha256=digest(committed / "tensor.bin"),
                               shape=shape, dtype="float64"))
    def scalar_tree(value):
        if isinstance(value, dict):
            return dict(mapping={k: scalar_tree(v) for k, v in value.items()})
        return dict(scalar=value)
    trace = {name: tensor([64, 4]) for name in ("trajectory_steps", "value_score_calls_per_chain",
                                               "target_status_calls_per_chain", "log_accept_ratio")}
    trace["target_status_telemetry"] = dict(mapping={name: tensor([64, 4]) for name in (
        "status_code", "valid_pre_regularized_score", "floor_count_value")})
    bundle = dict(schema="bayesfilter.durable_tensor_checkpoint.v1", identity_hash=payload_digest(identity),
        inputs_hash=payload_digest(inputs), metadata=dict(attempt=str(attempt)),
        tree=dict(mapping=dict(samples=tensor([64, 4, 6]), trace=dict(mapping=trace),
                               metadata=scalar_tree(metadata))))
    path = committed / "bundle.json"
    write(path, bundle)
    path.with_name("bundle.sha256").write_text(digest(path))
    return cell, root, call, path


def test_native_receipt_joins_provenance_work_trace_and_sample_dimensions(tmp_path):
    directory = tmp_path / "cell"
    cell, root, call, _ = call_fixture(directory)
    result = audit_call(directory, cell, root, call, dimension=6)
    assert result["actual_work"] == 64
    assert result["tensor_receipts_checked"] == 8


@pytest.mark.parametrize("mutation", ["start", "stream", "count", "native_work", "dtype", "status_shape", "bytes"])
def test_inconsistent_native_receipts_cannot_be_cost_evidence(tmp_path, mutation):
    directory = tmp_path / "cell"
    cell, root, call, path = call_fixture(directory)
    if mutation == "start": cell["initial_active_state_sha256"] = "different-start"
    elif mutation == "stream": call["seed"] = [1, 3]
    elif mutation == "count": call["transitions"] = 32
    elif mutation == "native_work": call["native"]["value_score_row_evaluations"] = 40
    elif mutation == "bytes": (path.parent / "tensor.bin").write_bytes(b"truncated")
    else:
        bundle = json.loads(path.read_text())
        if mutation == "dtype": bundle["tree"]["mapping"]["samples"]["tensor"]["dtype"] = "float32"
        else:
            bundle["tree"]["mapping"]["trace"]["mapping"]["target_status_telemetry"]["mapping"][
                "status_code"]["tensor"]["shape"] = [64]
        write(path, bundle)
        path.with_name("bundle.sha256").write_text(digest(path))
    with pytest.raises(ValueError):
        audit_call(directory, cell, root, call, dimension=6)

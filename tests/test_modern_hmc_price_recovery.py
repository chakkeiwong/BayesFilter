"""Cost-only timeout recovery: preserve receipts, streams and inclusive charges."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.price_modern_hmc_frozen_design import (
    audit_attempts, digest, finalize, planned_rows, pricing_seed, resume_rows,
)
from scripts.report_modern_hmc_frozen_prices import build_report
from scripts.audit_mams_full_model_pricing import payload_digest
from tests.test_modern_hmc_comparison_prices import design
from tests.test_modern_hmc_price_report import call_fixture


def save(path, value):
    path.write_text(json.dumps(value))


def native_call(directory, *, cell_index, root_index, repeat, namespace, elapsed=2.):
    cell, root, call, path = call_fixture(directory)
    root.update(index=root_index, transition_spec=dict(profile="test", max_steps=3, epsilon=.5))
    call.update(status="returned", elapsed_seconds=elapsed, hard_vetoes=[],
                repeat=repeat, seed=list(pricing_seed(namespace, cell_index, root_index, repeat)))
    call["native"]["transition_spec"] = root["transition_spec"]
    call["native"]["sample_chain_call_s"] = elapsed / 2
    bundle = json.loads(path.read_text())
    inputs_path = Path(bundle["metadata"]["attempt"]) / "call-start.json"
    inputs = json.loads(inputs_path.read_text())["inputs"]
    inputs.update(seed=call["seed"], candidate={k: root[k] for k in (
        "candidate_id", "transition_spec", "epsilon", "leapfrog_steps")})
    save(inputs_path, dict(inputs=inputs))
    bundle["inputs_hash"] = payload_digest(inputs)
    bundle["tree"]["mapping"]["metadata"]["mapping"]["transition_spec"] = dict(mapping={
        k: dict(scalar=v) for k, v in root["transition_spec"].items()})
    bundle["tree"]["mapping"]["metadata"]["mapping"]["sample_chain_call_s"] = dict(scalar=elapsed / 2)
    if repeat:
        destination = path.parent.with_name(f"root-{root_index:03d}-repeat-{repeat}")
        path.parent.rename(destination)
        path = destination / "bundle.json"
    save(path, bundle)
    path.with_name("bundle.sha256").write_text(digest(path))
    return cell, root, call


def partial_fixture(tmp_path):
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    save(geometry / "geometry.json", dict(center=[0.]*6))
    save(geometry / "design.json", dict(metadata=dict(data_sha256="data")))
    source = tmp_path / "source"
    source.mkdir()
    (source / "operator.so").write_bytes(b"source-fixture")
    save(source / ".validation-source.json", dict(commit="frozen"))
    driver = tmp_path / "driver.py"
    driver.write_text("# preserved driver\n")
    paths = []
    for method in ("fixed_hmc", "mams"):
        value = design()
        value.update(method=method, geometry_directory=str(geometry))
        value["execution"]["transition_profile"] = "test"
        value["search"]["epsilon_by_l"] = [[3, [.5]]]
        value["search"]["max_candidates"] = 2
        path = tmp_path / f"{method}.json"
        save(path, value)
        paths.append(path)
    original = tmp_path / "original"
    original.mkdir()
    cell, root, call = native_call(original / "cell-00", cell_index=0,
        root_index=0, repeat=0, namespace=100)
    rows = planned_rows(paths)
    rows[0].update(cell, status="running", source_sha256={str(driver): digest(driver)},
        geometry_sha256=digest(geometry / "geometry.json"), data_sha256="data", preparation_seconds=1.)
    rows[0]["roots"][0].update(root, status="running", calls=[call,
        dict(status="running", repeat=1, seed=list(pricing_seed(100, 0, 0, 1)))])
    prior = dict(status="running", rows=rows, source_root=str(source),
        numerical_source=dict(commit="frozen"), compiled_operator_sha256={"operator.so": digest(source / "operator.so")},
        memory_policy=dict(all_physical_devices_memory_growth=True,
                           configured_before_logical_device_initialization=True),
        memory_growth_environment="true", gpu_trust_basis="trusted_escalated_gpu_execution",
        command=["python", str(driver)], driver_sha256=digest(driver), pricing_seed_namespace=100)
    save(original / "result.json", prior)
    prior = finalize(original, status="supervisor_budget_exhausted", elapsed=180.)
    return prior, paths, original


def test_timeout_reuses_only_audited_returns_and_keeps_full_unstarted_inventory(tmp_path):
    prior, paths, original = partial_fixture(tmp_path)
    rows = resume_rows(prior, paths, seed_namespace=101, prior_directory=original)
    assert [r["method"] for r in rows] == ["fixed_hmc", "mams"]
    root = rows[0]["roots"][0]
    assert root["resume_missing_repeats"] == [1]
    assert [c["repeat"] for c in root["calls"]] == [0]
    assert len(root["prior_attempts"]) == 2
    assert root["calls"][0]["checkpoint_directory"] == str(original / "cell-00")
    assert rows[1]["roots"][0]["resume_missing_repeats"] == [0, 1]
    assert rows[1]["resume_binding_identity"] == {}
    assert prior["rows"][0]["roots"][0]["calls"][0].get("checkpoint_directory") is None


@pytest.mark.parametrize("mutation", ["running", "namespace", "design", "order", "geometry", "source",
    "operator", "seed", "checkpoint", "growth", "cost", "driver", "second_recovery", "repeat"])
def test_recovery_rejects_stale_invalid_or_changed_originals(tmp_path, mutation):
    prior, paths, original = partial_fixture(tmp_path)
    namespace = 101
    if mutation == "running": prior["status"] = "running"
    elif mutation == "namespace": namespace = 100
    elif mutation == "design": paths[0].write_text(paths[0].read_text() + " ")
    elif mutation == "order": paths.reverse()
    elif mutation == "geometry": (tmp_path / "geometry/geometry.json").write_text('{"center":[1]}')
    elif mutation == "source": save(tmp_path / "source/.validation-source.json", dict(commit="changed"))
    elif mutation == "operator": (tmp_path / "source/operator.so").write_bytes(b"changed")
    elif mutation == "seed": prior["rows"][0]["roots"][0]["calls"][1]["seed"][1] += 1
    elif mutation == "checkpoint": (original / "cell-00/committed/root-000-repeat-0/tensor.bin").unlink()
    elif mutation == "growth": prior["memory_policy"]["all_physical_devices_memory_growth"] = False
    elif mutation == "cost": prior["supervisor_elapsed_seconds"] += 1
    elif mutation == "driver": (tmp_path / "driver.py").write_text("changed")
    elif mutation == "second_recovery": prior["resume_of"] = str(original / "result.json")
    elif mutation == "repeat": prior["rows"][0]["roots"][0]["calls"].append(
        deepcopy(prior["rows"][0]["roots"][0]["calls"][0]))
    with pytest.raises((ValueError, FileNotFoundError)):
        resume_rows(prior, paths, seed_namespace=namespace, prior_directory=original)


def test_completed_reuse_and_recovery_inclusive_charges_are_not_double_counted(tmp_path):
    prior, paths, original = partial_fixture(tmp_path)
    # A complete first cell may be skipped; a partially completed one retains vetoes.
    root = prior["rows"][0]["roots"][0]
    root["calls"][0]["hard_vetoes"] = ["diagnostic-veto"]
    save(original / "result.json", prior)
    rows = resume_rows(prior, paths, seed_namespace=101, prior_directory=original)
    assert rows[0]["roots"][0]["calls"][0]["hard_vetoes"] == ["diagnostic-veto"]
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    report = deepcopy(prior)
    report.update(rows=rows, pricing_seed_namespace=101, resume_of=str(original / "result.json"),
        resume_sha256=digest(original / "result.json"), pricing_attempts=audit_attempts(prior, original))
    save(recovery / "result.json", report)
    settled = finalize(recovery, status="supervisor_budget_exhausted", elapsed=75.)
    assert settled["current_supervisor_elapsed_seconds"] == 75.
    assert settled["supervisor_elapsed_seconds"] == 255.
    assert len(audit_attempts(settled, recovery)) == 2
    with pytest.raises(ValueError, match="second timeout recovery"):
        resume_rows(settled, paths, seed_namespace=102, prior_directory=recovery)
    prior["supervisor_elapsed_seconds"] += 1
    save(original / "result.json", prior)
    with pytest.raises(ValueError, match="original terminal price receipt changed"):
        audit_attempts(settled, recovery)


@pytest.mark.parametrize("changed_field", [None, "elapsed_seconds", "hard_vetoes", "seed"])
def test_recovery_receipts_support_complete_cost_report_without_overwriting_prices(tmp_path, changed_field):
    prior, paths, original = partial_fixture(tmp_path)
    recovery = tmp_path / "recovery"
    recovery.mkdir()
    rows = resume_rows(prior, paths, seed_namespace=101, prior_directory=original)
    # Build independent native receipts for exactly the three missing calls.
    for i, row in enumerate(rows):
        root = row["roots"][0]
        for repeat in root["resume_missing_repeats"]:
            temporary = recovery / f"temporary-{i}-{repeat}"
            cell, candidate, call = native_call(temporary, cell_index=i, root_index=0,
                repeat=repeat, namespace=101)
            destination = recovery / f"cell-{i:02d}"
            if not destination.exists():
                temporary.rename(destination)
            else:
                (temporary / f"committed/root-000-repeat-{repeat}").rename(
                    destination / f"committed/root-000-repeat-{repeat}")
            # Preserve the helper's attempt path; it contains the declared inputs.
            bundle_path = destination / f"committed/root-000-repeat-{repeat}/bundle.json"
            bundle = json.loads(bundle_path.read_text())
            input_dir = recovery / f"inputs-{i}-{repeat}"
            input_dir.mkdir()
            old_input = Path(bundle["metadata"]["attempt"]) / "call-start.json"
            if not old_input.exists():
                old_input = destination / "attempt/call-start.json"
            (input_dir / "call-start.json").write_bytes(old_input.read_bytes())
            bundle["metadata"]["attempt"] = str(input_dir)
            save(bundle_path, bundle)
            bundle_path.with_name("bundle.sha256").write_text(digest(bundle_path))
            call["checkpoint_directory"] = str(destination)
            root["calls"].append(call)
            root.update(candidate, status="priced")
            row.update(cell, status="completed", source_sha256={str(tmp_path / "driver.py"): digest(tmp_path / "driver.py")},
                geometry_sha256=digest(tmp_path / "geometry/geometry.json"), data_sha256="data", preparation_seconds=1.)
        root["calls"].sort(key=lambda c: c["repeat"])
    report = deepcopy(prior)
    report.update(rows=rows, pricing_seed_namespace=101, resume_of=str(original / "result.json"),
        resume_sha256=digest(original / "result.json"), pricing_attempts=audit_attempts(prior, original))
    save(recovery / "result.json", report)
    finalize(recovery, status="completed", elapsed=75.)
    if changed_field is not None:
        changed = json.loads((recovery / "result.json").read_text())
        reused = changed["rows"][0]["roots"][0]["calls"][0]
        if changed_field == "elapsed_seconds":
            reused[changed_field] += 10.
        elif changed_field == "hard_vetoes":
            reused[changed_field] = ["changed-veto"]
        else:
            reused[changed_field][0] = 101
        save(recovery / "result.json", changed)
        with pytest.raises(ValueError, match="reused price call changed|originating attempt"):
            build_report(recovery / "result.json", paths, available_seconds=1000000.,
                         trial_overhead_seconds=0., fixed_overhead_seconds=0.)
        assert json.loads((original / "result.json").read_text()) == prior
        return
    result = build_report(recovery / "result.json", paths, available_seconds=1000000.,
                          trial_overhead_seconds=0., fixed_overhead_seconds=0.)
    assert result["actual_price_worker_seconds"] == 255.
    assert result["distinct_price_streams"] == 4
    assert [r["root_count"] for r in result["rows"]] == [1, 1]
    assert json.loads((original / "result.json").read_text()) == prior

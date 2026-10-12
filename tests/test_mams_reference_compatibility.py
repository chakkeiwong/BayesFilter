"""Reference identity and corruption controls; synthetic rows are mechanics only."""
import json
from pathlib import Path

import pytest

from scripts import audit_mams_reference_compatibility as audit


def save(path, value):
    path.write_text(json.dumps(value))


def reference_fixture(tmp_path, target="NL-SIR"):
    directory = tmp_path / "reference"
    directory.mkdir()
    horizon = 100 if target == "NL-STR" else 20
    dimension = 3 if target == "NL-SIR" else 5 if target == "NL-STR" else 6
    names = audit.quantity_names(target)
    data = [[0.]] * horizon
    metadata = dict(target_id=target, horizon=horizon, full_horizon=horizon,
                    parameter_dim=dimension, data_sha256="fixture-data")
    design = dict(target=metadata, data=data, powers=[3, 4], replications=8,
                  quantity_names=names, reference_tolerance=.002,
                  standardization="theta=0.5*z; log_jacobian=3*log(0.5)")
    result = dict(status="empirically_refined", target=target,
                  rungs=[dict(power=p) for p in (3, 4)], final_rung="rung-04.json")
    parity = dict(passed=True, value_max_absolute_error=0., score_max_scaled_errors=[0., 0.])
    save(directory / "design.json", design)
    save(directory / "result.json", result)
    save(directory / "parity.json", parity)
    for power in (3, 4):
        save(directory / f"rung-{power:02d}.json", dict(power=power, replications=8,
            points_per_replication=2**power,
            rows=[dict(replication=i, mean=[0.]*len(names), sd=[1.]*len(names)) for i in range(8)],
            mean=[0.]*len(names), sd=[1.]*len(names),
            pointwise_half_width=[0.]*len(names), randomization_mcse=[0.]*len(names)))
    return directory, data


@pytest.mark.parametrize("target", audit.REFERENCES)
def test_all_reference_quantity_and_horizon_contracts(tmp_path, target):
    directory, data = reference_fixture(tmp_path, target)
    summary = audit.reference_summary(directory, target=target, data_sha256="fixture-data", data=data)
    assert summary["quantities"] == audit.quantity_names(target)
    assert len(summary["mean"]) == len(summary["uncertainty"])
    assert summary["exact_nonlinear_posterior"] is False
    assert summary["rigorous_integration_error_bound"] is False


@pytest.mark.parametrize("file,mutate", [
    ("design.json", lambda d: d["target"].update(target_id="NL-STR")),
    ("design.json", lambda d: d["target"].update(horizon=19)),
    ("design.json", lambda d: d.update(data=[[1.]]*20)),
    ("design.json", lambda d: d.update(quantity_names=d["quantity_names"][:-1])),
    ("design.json", lambda d: d.update(standardization="theta=z")),
    ("design.json", lambda d: d.update(powers=[4, 3])),
    ("design.json", lambda d: d.update(replications=4)),
    ("result.json", lambda d: d.update(rungs=[dict(power=3)])),
    ("parity.json", lambda d: d.update(value_max_absolute_error=float("nan"))),
    ("parity.json", lambda d: d.update(score_max_scaled_errors=[0., .1])),
    ("rung-04.json", lambda d: d["rows"][1].update(replication=0)),
    ("rung-04.json", lambda d: d.update(rows=d["rows"][:-1])),
    ("rung-04.json", lambda d: d["rows"][0].update(mean=[float("nan")]*9)),
    ("rung-04.json", lambda d: d.update(pointwise_half_width=[-1.]*9)),
    ("rung-04.json", lambda d: d.update(mean=[1.]*9)),
    ("rung-04.json", lambda d: d.update(points_per_replication=15)),
])
def test_bad_reference_evidence_is_rejected(tmp_path, file, mutate):
    directory, data = reference_fixture(tmp_path)
    value = audit.read(directory / file)
    mutate(value)
    save(directory / file, value)
    with pytest.raises(ValueError):
        audit.reference_summary(directory, target="NL-SIR", data_sha256="fixture-data", data=data)


def test_reference_precision_recomputed_from_rows(tmp_path):
    directory, data = reference_fixture(tmp_path)
    value = audit.read(directory / "rung-04.json")
    value["mean"] = [.01]*9
    for row in value["rows"]:
        row["mean"] = [.01]*9
    save(directory / "rung-04.json", value)
    with pytest.raises(ValueError, match="empirical precision"):
        audit.reference_summary(directory, target="NL-SIR", data_sha256="fixture-data", data=data)


def source_fixture(tmp_path):
    reference = tmp_path / "reference"
    current = tmp_path / "current"
    relative = Path("bayesfilter/testing/target.py")
    for root in (reference / "source", current):
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_text("def value(q):\n    return q*q\n")
    save(reference / "design.json", dict(
        source_closure={str(tmp_path / relative): audit.digest(current / relative)}, inputs={}))
    return reference, current, relative


def test_reviewed_statistics_extension_preserves_target_source_and_rejects_further_changes(tmp_path):
    import ast
    root = Path(__file__).resolve().parents[1]
    relative = Path("bayesfilter/inference/hmc_acceptance_statistics.py")
    source = (root / relative).read_text()
    old_source = source.split("\n\ndef replicated_qualification_attainability", 1)[0]
    before, after = ast.parse(old_source), ast.parse(source)
    additions = [node for node in after.body if isinstance(node, ast.FunctionDef)
                 and node.name == "replicated_qualification_attainability"]
    assert len(additions) == 1 and not additions[0].decorator_list
    after.body = [node for node in after.body if node not in additions]
    assert ast.dump(before) == ast.dump(after)
    reference, current = tmp_path / "reference", tmp_path / "current"
    for directory, text in ((reference / "source", old_source), (current, source)):
        path = directory / relative
        path.parent.mkdir(parents=True)
        path.write_text(text)
    expected_old, expected_new, _ = audit.REVIEWED_PAIRS[str(relative)]
    assert audit.digest(reference / "source" / relative) == expected_old
    assert audit.digest(current / relative) == expected_new
    save(reference / "design.json", dict(source_closure={str(root / relative): expected_old}, inputs={}))
    report = audit.audit_sources(reference, current, tmp_path)
    assert len(report["reviewed_differences"]) == 1
    (current / relative).write_text(source + "\n_UNREVIEWED = True\n")
    with pytest.raises(ValueError, match="unreviewed source difference"):
        audit.audit_sources(reference, current, tmp_path)


@pytest.mark.parametrize("mutation", ["current", "archive", "missing"])
def test_no_unreviewed_source_exclusion(tmp_path, mutation):
    reference, current, relative = source_fixture(tmp_path)
    assert not audit.audit_sources(reference, current, tmp_path)["reviewed_differences"]
    if mutation == "current":
        (current / relative).write_text("def value(q):\n    return q*q*q\n")
    elif mutation == "archive":
        (reference / "source" / relative).write_text("changed")
    else:
        (reference / "source" / relative).unlink()
    with pytest.raises(ValueError, match="source"):
        audit.audit_sources(reference, current, tmp_path)


def bridge_fixture(tmp_path):
    directory, data = reference_fixture(tmp_path)
    current = tmp_path / "current"
    current.mkdir()
    (current / "target.py").write_text("current numerical source")
    summary = audit.reference_summary(directory, target="NL-SIR", data_sha256="fixture-data", data=data)
    design_hash = audit.digest(directory / "design.json")
    report = dict(schema="bayesfilter.mams_reference_compatibility.v1", target="NL-SIR", status="compatible",
        reference_directory=str(directory), summary=summary,
        sources=dict(current_source_sha256={"target.py": audit.digest(current / "target.py")},
            archived_source_sha256={str(directory / "design.json"): design_hash},
            input_sha256={str(directory / "design.json"): design_hash}),
        audit_files_sha256={str(Path(audit.__file__)): audit.digest(audit.__file__)},
        parity=dict(passed=True, all_finite=True, native_status_valid=True,
            original_point_count=4, prepared_point_count=4, points=[[0.]*3]*8,
            value_max_absolute_error=0., score_max_scaled_errors=[0., 0.], quantity_max_scaled_error=0.))
    path = tmp_path / "bridge.json"
    save(path, report)
    return path, current, data


@pytest.mark.parametrize("mutation", [None, "source", "target", "data", "quantity", "parity", "empty"])
def test_bridge_consumer_binds_current_source_and_reference(tmp_path, mutation):
    path, current, data = bridge_fixture(tmp_path)
    report = audit.read(path)
    if mutation == "source":
        (current / "target.py").write_text("new numerical source")
    elif mutation == "target":
        report["target"] = "NL-STR"
    elif mutation == "data":
        data = [[1.]]*20
    elif mutation == "quantity":
        report["summary"]["quantities"][0] = "wrong chart"
    elif mutation == "parity":
        report["parity"]["quantity_max_scaled_error"] = float("nan")
    elif mutation == "empty":
        report["sources"]["current_source_sha256"] = {}
    save(path, report)
    def consume():
        return audit.load_compatible_reference(path, current_source=current, target="NL-SIR", data_sha256="fixture-data", data=data)
    if mutation is None:
        assert consume()["quantities"] == audit.quantity_names("NL-SIR")
    else:
        with pytest.raises(ValueError):
            consume()

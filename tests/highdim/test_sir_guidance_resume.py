"""Diagnostic continuation checks without importing TensorFlow or using a GPU."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("sir_guidance_runner", ROOT / "scripts/run_ledh_sir_guidance.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def fixture(directory):
    fields = ("git_commit", "jit_compile", "dtype", "tf32", "cpu_only",
              "particles", "horizon", "reset_epsilon", "reset_steps", "evidence_role",
              "guide_fitting", "data", "fit_seeds", "theta", "parameter_names", "reference", "tensorflow")
    manifest = {field: field for field in fields}
    manifest.update(status="budget_exhausted", completed_rows=1, wall_seconds=1.,
                    arms=RUNNER.ARMS, seeds={"26100831": list(range(8)), "26100832": list(range(8))},
                    trace_fields=["valid", "moment_valid", "correction_cap_valid"], fits={},
                    source_sha256={"scripts/run_ledh_sir_guidance.py": "old-runner", "numerics.py": "same-numerics"})
    for data_seed in manifest["seeds"]:
        path = directory / f"guides-{data_seed}.json"
        path.write_text("{}")
        manifest["fits"][data_seed] = dict(sha256=RUNNER.digest(path))
    row = dict(data_seed=26100831, seed=0, arm="plain", beta=RUNNER.ARMS["plain"],
               reset_epsilon=1., reset_steps=5, valid=True, steps_completed=[50] * 3,
               score=[0., 1., 2.], value=-1., traces=[[[1., 1., 1.]] * 50] * 3)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    (directory / "rows.jsonl").write_text(json.dumps(row) + "\n")
    current = copy.deepcopy(manifest)
    current["source_sha256"]["scripts/run_ledh_sir_guidance.py"] = "resume-enabled-runner"
    return current, row


def test_preserves_completed_rows_and_provenance(tmp_path):
    current, row = fixture(tmp_path)
    retained, provenance = RUNNER.previous_rows(tmp_path, current)
    assert retained == [row]
    assert provenance["count"] == 1
    assert provenance["original_runner_sha256"] == "old-runner"
    assert provenance["rows_sha256"] == RUNNER.digest(tmp_path / "rows.jsonl")


def test_preserves_failed_evidence_only_after_exact_diagnostic(tmp_path):
    current, row = fixture(tmp_path)
    row.update(valid=False, steps_completed=[4] * 3)
    (tmp_path / "rows.jsonl").write_text(json.dumps(row) + "\n")
    diagnostic = dict(reproduced=True, valid=False, data_seed=row["data_seed"],
                      seed=row["seed"], arm=row["arm"], steps_completed=4,
                      reset_epsilon=1., reset_steps=5,
                      source_manifest_sha256=RUNNER.digest(tmp_path / "manifest.json"))
    diagnostic_path = tmp_path / "diagnostic.json"
    diagnostic_path.write_text(json.dumps(diagnostic))
    retained, provenance = RUNNER.previous_rows(tmp_path, current, diagnostic_path)
    assert retained[0]["valid"] is False
    assert provenance["quarantined_keys"] == [(26100831, 0, "plain")]
    diagnostic["source_manifest_sha256"] = "different-attempt"
    diagnostic_path.write_text(json.dumps(diagnostic))
    with pytest.raises(ValueError):
        RUNNER.previous_rows(tmp_path, current, diagnostic_path)


@pytest.mark.parametrize("fault", ["scope", "implementation", "fit", "duplicate", "invalid", "key", "controls", "running"])
def test_rejects_invalid_continuation(tmp_path, fault):
    current, row = fixture(tmp_path)
    previous = json.loads((tmp_path / "manifest.json").read_text())
    rows = [row]
    if fault == "scope":
        current["horizon"] = 10
    elif fault == "implementation":
        current["source_sha256"]["numerics.py"] = "changed"
    elif fault == "fit":
        (tmp_path / "guides-26100831.json").write_text("changed")
    elif fault == "duplicate":
        rows.append(row)
        previous["completed_rows"] = 2
    elif fault == "invalid":
        row["traces"][0][0][0] = 0.
    elif fault == "key":
        row["seed"] = 99
    elif fault == "controls":
        row["reset_steps"] = 40
    else:
        previous["status"] = "running"
    (tmp_path / "manifest.json").write_text(json.dumps(previous))
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(item) + "\n" for item in rows))
    with pytest.raises(ValueError):
        RUNNER.previous_rows(tmp_path, current)

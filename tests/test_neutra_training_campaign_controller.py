"""CPU-only orchestration checks; no training or scientific quality evidence."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def controller():
    path = Path(__file__).resolve().parents[1]/"scripts/continue_neutra_training_campaign.py"
    spec = importlib.util.spec_from_file_location("neutra_training_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def cohort(tmp_path):
    rows = []
    for family in ("iaf16", "naf16"):
        for seed in range(3):
            name = f"{family}-seed{seed}-u4096"
            request = {"root_seed": seed, "batch_size": 32, "updates": 4096,
                "validation_rungs": [1024, 2048, 4096], "estimator": "path"}
            (tmp_path/f"training-request-{name}-01.json").write_text(json.dumps(request))
            rows.append({"name": name, "status": "completed", "output": str(tmp_path/name)})
    selected = {family: {"learning_rate": .01, "gradient_clip_norm": 20.}
        for family in ("iaf16", "naf16", "naf32", "legacy_control")}
    return {"jobs": rows}, selected


def test_continuation_preserves_seeded_optimizer_handoffs_and_fresh_control(controller, cohort, tmp_path):
    initial, selected = cohort
    decision, requests, retained = controller.continuation_requests(tmp_path, initial, selected)
    assert decision["family"] == "naf16"
    control = requests[0][1]
    assert control["family"] == "legacy_control" and "resume_checkpoint" not in control
    assert control["updates"] == 4096
    for seed, (_, request) in enumerate(requests[1:]):
        assert request["root_seed"] == seed and request["updates"] == 8192
        assert request["resume_checkpoint"] == str(tmp_path/f"naf16-seed{seed}-u4096/checkpoint-004096.json")
        assert request["validation_rungs"] == [6144, 8192]
    assert {row["name"] for row in retained} == {f"iaf16-seed{s}-u4096" for s in range(3)}


def test_candidate_failure_activates_repair_without_hiding_survivors(controller, cohort, tmp_path):
    initial, selected = cohort
    initial["jobs"][-1]["status"] = "candidate_rejected"
    decision, requests, retained = controller.continuation_requests(tmp_path, initial, selected)
    assert decision["family"] == "naf32"
    assert not decision["candidate_failure_rejects_research_direction"]
    assert len(retained) == 5
    assert all(request["updates"] == 4096 and "resume_checkpoint" not in request for _, request in requests)
    assert all(request["family"] == "naf32" for _, request in requests[1:])
    initial["jobs"].pop()
    with pytest.raises(ValueError, match="all six"):
        controller.continuation_requests(tmp_path, initial, selected)


def test_resume_rejects_changed_inputs_and_preserves_compute_reserve(controller, tmp_path):
    path = tmp_path/"request.json"
    request = {"seed": 3, "updates": 4096}
    controller.save_request(path, request)
    controller.save_request(path, request)
    with pytest.raises(ValueError, match="request differs"):
        controller.save_request(path, {**request, "seed": 4})
    assert json.loads(path.read_text()) == request
    controller.save(tmp_path/"accounting-after-calibration.json",
        {"remaining_before_sustained_training_seconds": 100.})
    controller.save(tmp_path/"supplemental-charges.json", {"charges": [{"worker_seconds": 10.}]})
    phases = [{"spent_worker_seconds": 20.}]
    assert controller.check_allocation(tmp_path, phases, 50., 20.) == 70.
    with pytest.raises(ValueError, match="campaign budget"):
        controller.check_allocation(tmp_path, phases, 51., 20.)

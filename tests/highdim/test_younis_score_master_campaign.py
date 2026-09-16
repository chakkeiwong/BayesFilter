"""Coordinator tests: no held-out data opens before all selections exist."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def load_driver():
    path = Path(__file__).resolve().parents[2]/"scripts/run_younis_score_campaign.py"
    spec = importlib.util.spec_from_file_location("score_campaign_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("failure", [None, "execute", "selection"])
def test_all_calibration_and_selections_precede_evaluation(tmp_path, monkeypatch, failure):
    driver = load_driver()
    events = []
    stages = []
    for label in ("cal1", "cal2", "claim"):
        study = tmp_path/(label+".json")
        study.write_text(json.dumps({"rows": [{"id": label}]}))
        stage = {"study": str(study), "output": str(tmp_path/label)}
        if label != "claim": stage["selection"] = str(tmp_path/(label+"-selection.json"))
        stages.append(stage)
    campaign = {"calibration": stages[:2], "evaluation": stages[2:],
                "max_numerical_rows": 3, "wall_seconds": 30,
                "progress_output": str(tmp_path/"progress.json")}
    path = tmp_path/"campaign.json"
    path.write_text(json.dumps(campaign))
    monkeypatch.setattr(sys, "argv", ["driver", "--campaign", str(path), "--action", "run"])
    monkeypatch.setattr(driver, "validate_study", lambda *args: None)
    def execute(study, registry, output, resume):
        events.append("run_"+output.name)
        return {"execution_status": "failed" if failure == "execute" and output.name == "cal2" else "complete", "wall_seconds": .1}
    monkeypatch.setattr(driver, "execute", execute)
    from bayesfilter.score_study import tuning
    def selection(root, destination):
        events.append("select_"+Path(root).name)
        if failure == "selection" and Path(root).name == "cal2": raise ValueError("invalid selection")
        Path(destination).write_text("{}")
    monkeypatch.setattr(tuning, "issue_selection", selection)
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            driver.main()
        assert "run_claim" not in events
        assert not json.loads((tmp_path/"progress.json").read_text())["evaluation_opened"]
    else:
        assert driver.main() == 0
        assert events == ["run_cal1", "select_cal1", "run_cal2", "select_cal2", "run_claim"]
        with pytest.raises(ValueError, match="fresh versioned"):
            driver.main()

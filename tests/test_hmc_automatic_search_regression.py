"""Complete automatic-search checks on tiny exact targets, not sampler rankings."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.extended
@pytest.mark.parametrize("case", ["isotropic", "anisotropic"])
def test_automatic_preparation_and_search_complete_with_verified_members(tmp_path, case):
    root = Path(__file__).resolve().parents[1]
    script = root / "docs/plans/artifacts/hmc-consistency-gap-repair-2026-09-15/automatic_search_check.py"
    destination = tmp_path / case
    # The five-minute limit is a convenience regression ceiling; it is not a
    # target-scale tuning budget or evidence of expected runtime.
    with (tmp_path / "process.log").open("w") as log:
        completed = subprocess.run([sys.executable, str(script), str(destination), case],
            cwd=root, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1",
                "TF_FORCE_GPU_ALLOW_GROWTH": "true", "BAYESFILTER_TEST_DEVICE_SCOPE": "cpu"},
            stdout=log, stderr=subprocess.STDOUT, timeout=300)
    result = json.loads((destination / "result.json").read_text())
    assert completed.returncode == 0, result
    assert result["completion"] == "complete" and result["verified_count"] > 0
    from bayesfilter.inference.hmc_candidate_set_artifacts import load_candidate_set_result_payload
    payload = load_candidate_set_result_payload(destination / "tuning/candidate_set_result.json")
    assert payload["nominee_id"] is None
    assert len(payload["verified_candidate_ids"]) == result["verified_count"]
    assert set(payload["candidate_states"].values()) <= {"verified", "promotion_failed", "inconclusive_at_cap"}
    primary = payload["candidates"][:6]
    assert [candidate["leapfrog_steps"] for candidate in primary] == [3, 5, 9, 13, 18, 25]

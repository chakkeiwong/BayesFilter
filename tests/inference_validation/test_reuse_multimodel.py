"""Exact candidate and posterior parity on geometry-specific public routes.

Small CPU reference allocations test mechanics only. GPU evidence requires the
separate roadmap designs and trusted worker configuration.
"""
import pytest
import tensorflow as tf

from bayesfilter.testing.inference_validation.designs import ScenarioSpec, ValidationDesign
from bayesfilter.testing.inference_validation.procedures import execute_pipeline
from bayesfilter.testing.inference_validation.storage import read_json, read_tensor
from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
from docs.benchmarks.audit_hmc_m30_2026_09_23 import normalize, numerical_rows
from docs.benchmarks.audit_hmc_m29_2026_09_23 import normalized_tuning


def parity_design(target, device="cpu_reference"):
    params, options = {}, {}
    route, epsilon = "prepared", 1.3
    if target == "dirichlet":
        epsilon = .8
    elif target == "residual":
        from bayesfilter.testing.inference_validation.funnel_maps import supplied_funnel_map
        spec = supplied_funnel_map("residual")
        target, params = spec["target"], spec["parameters"]
        route, epsilon = "fixed_transport", .5
        options["transport_payload"] = spec["transport_payload"]
    return ValidationDesign(design_id="m32-reuse-" + target, engine="accuracy",
        scenario=ScenarioSpec(target, route, parameters=params), replications=1,
        draws=64, measurement_draws=64, seed=2026092311, budget_seconds=800,
        device=device, purpose="full candidate lifecycle and posterior parity",
        numerical_provenance="M32 inherited tiny mechanics counts/broad screen; not calibration",
        l_grid=(2, 3), step_size=epsilon, posterior_cap=128,
        options={**options, "acceptance_policy": {"practical_region": (.41, .99), "repair_region": (.405, .995)},
                 "search": {"pilot_enabled": False, "refinement_rounds": 0,
                            "total_budget_units": 24, "repair_reserve_units": 4, "evidence_rungs": (1,)}})


def compare_pair(left, right):
    pairs = [read_json(p / "pipeline.json") for p in (left, right)]
    assert pairs[0]["verified_candidate_ids"]
    assert pairs[0]["verified_candidate_ids"] == pairs[1]["verified_candidate_ids"]
    tuned = []
    rows = []
    for root, pipeline in zip((left, right), pairs):
        tuning = read_json(pipeline["tuning_path"])
        assert not check_inventory(tuning)["failures"]
        # Evidence is keyed by its checked content digest in the repository.
        from bayesfilter.testing.inference_validation.designs import digest
        evidence = [read_json(p) for p in (root / "tuning/numerical_evidence").glob("*.json")]
        numerical, hashes = numerical_rows({digest(e): e for e in evidence})
        rows.append(numerical)
        tuned.append(normalized_tuning(tuning, hashes))
    assert rows[0] == rows[1]
    assert tuned[0] == tuned[1]
    for a, b in zip(pairs[0]["members"], pairs[1]["members"]):
        assert a["candidate_id"] == b["candidate_id"]
        assert a["status"] == b["status"] == "assessed"
        assert a["warmup_exclusion_matches"] and b["warmup_exclusion_matches"]
        for key in ("draws_path", "warmup_path"):
            tf.debugging.assert_equal(read_tensor(a[key]), read_tensor(b[key]))
        assert normalize(a["posterior"], expected_l=a["L"], fit_root=left) == normalize(
            b["posterior"], expected_l=b["L"], fit_root=right)
    return {"verified_members": len(pairs[0]["verified_candidate_ids"]),
            "numerical_receipts": len(rows[0]), "exact_parity": True}


@pytest.mark.parametrize("target", ["rotated_gaussian", "dirichlet", "residual"])
def test_real_public_reuse_parity_across_geometry(tmp_path, target):
    design = parity_design(target)
    for reuse, name in ((False, "static"), (True, "dynamic")):
        execute_pipeline(design, tmp_path / name, reuse_leapfrog_graphs=reuse)
    compare_pair(tmp_path / "static", tmp_path / "dynamic")

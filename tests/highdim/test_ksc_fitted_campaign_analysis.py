"""Independent synthetic fixtures for campaign pairing and clustered inference."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def module(name):
    specification = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    loaded = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loaded)
    return loaded


def fixture():
    analysis = module("analyze_ledh_ksc_fitted_guides")
    rows = [{"condition": condition, "cluster": cluster, "replicate": replicate,
             "arm": arm, "status": "valid",
             "score": [1.0 + cluster / 10.0, 2.0],
             "cloud_sha256": f"{condition}-{cluster}-{replicate}"}
            for condition in analysis.CONDITIONS for cluster in range(8) for replicate in range(8)
            for arm in analysis.CONTROLS + analysis.FITTED]
    references = {condition: {"score": [0.0, 0.0],
                              "gaussian_approximation": {"score": [0.0, 0.0]}}
                  for condition in analysis.CONDITIONS}
    return analysis, rows, references


def test_seed_namespaces_are_disjoint_and_complete():
    runner = module("run_ledh_ksc_fitted_guides")
    design = runner.seed_design()
    groups = {"calibration": [], "validation": [], "final_fits": [], "final_clouds": []}
    for condition in runner.CONDITIONS:
        for values in design["calibration"][condition].values():
            groups["calibration"].extend(values)
        groups["validation"].extend(design["validation"][condition])
        for values in design["final_fits"][condition].values():
            groups["final_fits"].extend(values)
        for values in design["final_clouds"][condition]:
            groups["final_clouds"].extend(values)
    assert [len(values) for values in groups.values()] == [120, 12, 48, 192]
    combined = sum(groups.values(), [])
    assert len(set(combined)) == len(combined)


def test_clustered_intervals_do_not_treat_clouds_as_independent_fits():
    analysis, rows, references = fixture()
    for row in rows:
        if row["arm"] in analysis.FITTED:
            row["score"] = [0.5 + row["cluster"] / 20.0, 1.0]
    actual = analysis.analyze(rows, references, 5.0)
    assert len(actual["contrasts"]) == 24
    contrast = actual["contrasts"][0]
    assert len(contrast["cluster_means"]) == 8
    expected = [-0.75 * (1.0 + cluster / 10.0) ** 2 for cluster in range(8)]
    assert contrast["cluster_means"] == pytest.approx(expected)
    assert actual["heuristic_dominance_verdict"] == "promotion_veto"


@pytest.mark.parametrize("fault", ["missing", "duplicate", "unpaired", "nonfinite"])
def test_invalid_grids_fail_closed(fault):
    analysis, rows, references = fixture()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "unpaired":
        rows[0]["cloud_sha256"] = "wrong"
    else:
        rows[0]["score"][0] = float("nan")
    with pytest.raises(ValueError):
        analysis.analyze(rows, references, 5.0)


def test_ineligible_fit_is_not_silently_removed_or_replaced():
    analysis, rows, references = fixture()
    for row in rows:
        if row["arm"] == analysis.FITTED[0] and row["cluster"] == 0:
            row["status"] = "candidate_ineligible"
            del row["score"]
    actual = analysis.analyze(rows, references, 5.0)
    assert actual["arm_decisions"][analysis.FITTED[0]]["ineligible_rows"] == 24
    assert all(not contrast["available"] for contrast in actual["contrasts"]
               if contrast["arm"] == analysis.FITTED[0])

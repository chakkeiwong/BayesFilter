"""CPU-only checks of conditional post-run SIR comparison reporting."""

import pytest

from scripts.analyze_sir_neural_kdm_completion import (
    HEURISTIC_ARMS, PARTICLE_ARMS, RESTRICTED_ARMS, analyze_completion,
)


def rows():
    neural, restricted = [], []
    datasets = [101, 102, 103, 104]
    for index, dataset in enumerate(datasets):
        common = dict(dataset=dataset, valid=True,
            observation_regime="low" if index < 2 else "high",
            compartment_regime="nearer_boundary" if index == 0 else "interior")
        for arm in PARTICLE_ARMS + RESTRICTED_ARMS:
            loss = 2. if arm.endswith("unadapted") else 1.
            destination = restricted if arm in RESTRICTED_ARMS else neural
            destination.extend(dict(common, arm=arm, loss=loss, blocks=[loss / 2., loss * 1.5],
                replicate=replicate, particle_seed=283030000 + 4 * index + replicate)
                for replicate in range(4))
        neural.extend(dict(common, arm=arm, loss=.5, blocks=[.25, .75], replicate=None,
                           particle_seed=None) for arm in HEURISTIC_ARMS)
    return neural, restricted, datasets


def test_adaptation_success_cannot_hide_heuristic_failure(monkeypatch):
    monkeypatch.setattr("scripts.analyze_sir_neural_kdm_completion.paired_interval",
                        lambda differences: [min(differences), max(differences)])
    result = analyze_completion(*rows())
    adaptation = [row for row in result["comparisons"] if row["regime"] == "all"
                  and row["candidate"] == "csmc_adapted" and row["comparator"] == "csmc_unadapted"]
    assert adaptation[0]["mean_difference"] == -1.
    assert adaptation[0]["paired_cluster_interval95"] == [-1., -1.]
    assert all(not verdict["passed"] for verdict in result["heuristic_dominance"].values())
    assert result["regime_counts"] == dict(all=4, low=2, high=2, nearer_boundary=1, interior=3)
    assert result["neural_primary_decision_unchanged"]
    assert not result["default_ready"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "seed", "regime", "blocks", "unknown"])
def test_invalid_joint_comparison_rejected(defect):
    neural, restricted, datasets = rows()
    if defect == "missing":
        restricted.pop()
    elif defect == "duplicate":
        restricted.append(dict(restricted[0]))
    elif defect == "seed":
        restricted[0]["particle_seed"] += 1
    elif defect == "regime":
        restricted[0]["observation_regime"] = "high"
    elif defect == "blocks":
        restricted[0]["blocks"] = [0., 0.]
    else:
        restricted[0]["arm"] = "unplanned"
    with pytest.raises(ValueError):
        analyze_completion(neural, restricted, datasets)

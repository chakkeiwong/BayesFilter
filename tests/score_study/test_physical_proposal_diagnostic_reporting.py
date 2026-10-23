"""CPU reporting checks; synthetic values have no scientific interpretation."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from bayesfilter.score_study.physical_proposal_diagnostic_campaign import PhysicalCampaign


def _report(candidate_losses, invalid_plain=False):
    campaign = PhysicalCampaign.__new__(PhysicalCampaign)
    campaign.partitions = {"test": [1, 2, 3, 4]}
    rows = []
    arms = ("plain", "conditional", "half", "csmc", "kernel", "csmc_trained",
            "kernel_trained", "prediction", "observation_only", "recursive_gaussian")
    for dataset in campaign.partitions["test"]:
        for arm in arms:
            rows.append(dict(dataset=dataset, regime="low" if dataset <= 2 else "high",
                arm=arm, valid=not (invalid_plain and arm == "plain" and dataset == 1),
                loss=candidate_losses[dataset - 1] if arm.endswith("_trained") else 1.))
    return campaign.summarize(rows)


def test_invalid_baseline_blocks_its_comparison_without_erasing_valid_controls():
    report = _report([.5] * 4, invalid_plain=True)
    comparisons = {(row["comparator"], row["regime"]): row
                   for row in report["comparisons"] if row["candidate"] == "csmc_trained"}
    assert not comparisons["plain", "all"]["valid"]
    assert comparisons["conditional", "all"]["valid"]
    assert comparisons["conditional", "all"]["inference"] == "candidate_lower"
    assert report["invalid_counts"]["plain"] == 1


def test_conditional_heuristic_loss_vetoes_favorable_pooled_average():
    report = _report([1.1, 1.1, .1, .1])
    verdict = report["heuristic_dominance"]["csmc_trained"]
    assert not verdict["passed"]
    assert {row["regime"] for row in verdict["underperformance"]} == {"low"}
    assert not report["default_readiness"]

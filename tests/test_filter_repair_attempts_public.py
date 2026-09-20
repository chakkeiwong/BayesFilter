"""Complete public records through a real second-factor fit and proposal."""

import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as current
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_structured_memory import _inputs


def test_public_actual_two_factor_escalation(request):
    checkpoint = FrozenCheckpoint("93c8e419", "attempts_public")
    frozen = checkpoint.load("bayesfilter.inference.sequential_map_covariance")
    scalar, batched, _ = _inputs(5, 4)
    start = tf.linspace(tf.constant(.002, tf.float64), tf.constant(.004, tf.float64), 5)
    options = {"locator_policy": "center_first", "terminal_score_max_abs": 1e-10,
        "initial_radius": .25, "search_sample_count": 4, "terminal_sample_count": 24,
        "max_attempts": 2, "max_exact_evaluations": 256,
        "refinement_geometry_policy": "factor_correlation", "structured_max_factors": 2,
        "structured_holdout_score_relative_rmse": .001, "record_refinement_movement_diagnostics": True,
        "seed": (2026, 715)}
    records = []
    for module in (frozen, current):
        records.append(module.estimate_sequential_map_covariance(scalar, [start],
            batched_value_and_score_fn=batched,
            config=module.SequentialMapCovarianceConfig(**options)).payload())
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "attempts-public-actual.json").open("x") as handle:
        json.dump({"before": records[0], "after": records[1],
            "checkpoint": checkpoint.revision, "frozen_source_sha256": checkpoint.hashes()}, handle, indent=2)
        handle.write("\n")
    assert frozen.structured_fit_data_program.__globals__["factor"].__name__.startswith(checkpoint.prefix)
    assert frozen.proposal_program.__module__.startswith(checkpoint.prefix)
    for record in records:
        rows = [row for row in record["diagnostics"]["history"] if "proposal_attempts" in row]
        assert rows, "The fixture must actually reach the public proposal path"
        assert any(len(row["proposal_attempts"]) == 2
            and row["proposal_attempts"][1]["proposal_evaluated"]
            and row["fit"]["factor_count"] == 2 and row["fit"]["status"] == "usable" for row in rows)
    _compare(records[1], records[0])

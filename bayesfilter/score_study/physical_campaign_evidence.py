"""Host-side evidence checks for the bounded physical-proposal campaign."""

import math


PARTICLE_ARMS = ("plain", "conditional", "half", "csmc", "kernel",
                 "csmc_trained", "kernel_trained")
HEURISTIC_ARMS = ("prediction", "observation_only", "recursive_gaussian")
PARTICLE_REPLICATIONS = 2
TEST_PARTICLE_SEED_BASE = 276100000


def validate_holdout_rows(rows, datasets):
    """Reject incomplete or invalid paired data before any statistical reduction."""
    if not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("Holdout datasets must be nonempty and unique")
    expected = {
        (dataset, arm, replicate)
        for dataset in datasets for arm in PARTICLE_ARMS
        for replicate in range(PARTICLE_REPLICATIONS)
    } | {(dataset, arm, None) for dataset in datasets for arm in HEURISTIC_ARMS}
    dataset_indices = {dataset: index for index, dataset in enumerate(datasets)}
    seen = set()
    regimes = {}
    required = {"dataset", "arm", "replicate", "regime", "loss", "valid"}
    for row in rows:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("Holdout row is missing required fields")
        if type(row["dataset"]) is not int or not isinstance(row["arm"], str):
            raise ValueError("Malformed holdout key")
        if row["replicate"] is not None and type(row["replicate"]) is not int:
            raise ValueError("Malformed holdout replicate")
        key = (row["dataset"], row["arm"], row["replicate"])
        if key not in expected:
            raise ValueError(f"Unexpected holdout key: {key}")
        if key in seen:
            raise ValueError(f"Duplicate holdout key: {key}")
        seen.add(key)
        if row["valid"] is not True:
            raise ValueError(f"Invalid holdout computation: {key}")
        if (type(row["loss"]) not in (int, float)
                or not math.isfinite(row["loss"]) or row["loss"] < 0.):
            raise ValueError(f"Invalid holdout moment loss: {key}")
        regime = row["regime"]
        if regime not in ("low", "high"):
            raise ValueError(f"Unknown holdout regime: {key}")
        previous = regimes.setdefault(row["dataset"], regime)
        if previous != regime:
            raise ValueError(f"Inconsistent holdout regime: {key}")
        if row["arm"] in PARTICLE_ARMS:
            expected_seed = (TEST_PARTICLE_SEED_BASE
                + PARTICLE_REPLICATIONS * dataset_indices[row["dataset"]] + row["replicate"])
            if type(row.get("particle_seed")) is not int or row["particle_seed"] != expected_seed:
                raise ValueError(f"Unpaired holdout particle seed: {key}")
        elif row.get("particle_seed") is not None:
            raise ValueError(f"Unexpected heuristic particle seed: {key}")
    missing = expected - seen
    if missing:
        raise ValueError(f"Incomplete holdout grid: {len(missing)} missing rows")
    return dict(complete=True, paired=True, rows=len(rows), datasets=len(datasets),
                particle_replications=PARTICLE_REPLICATIONS)

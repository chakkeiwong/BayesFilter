"""Exact diagnostic replay of a failed moment-training test; no selection."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

from bayesfilter.score_study.moment_training_campaign import Campaign, tf


def safe(value):
    if tf.is_tensor(value):
        return safe(value.numpy().tolist())
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.monotonic()
    manifest = json.loads((arguments.source / "manifest.json").read_text())
    hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
              for name, expected in manifest["source_hashes"].items()}
    if not all(hashes.values()):
        raise ValueError("Executed runtime sources changed")
    failed = json.loads((arguments.source / "test.jsonl").read_text().splitlines()[-1])
    if failed["valid"] or failed["method"] != "unguided":
        raise ValueError("Expected the exact failed unguided endpoint")
    campaign = Campaign(argparse.Namespace(output=arguments.output, budget_seconds=600., smoke=False))
    data = campaign.dataset(failed["dataset_seed"])
    saved = json.loads((arguments.source / f"dataset-{failed['dataset_seed']}.json").read_text())
    matches = {name: bool(tf.reduce_all(data[name] == tf.constant(saved[name], tf.float64)))
               for name in ("observations", "label", "score_label")}
    matches["csmc"] = all(bool(tf.reduce_all(actual == tf.constant(expected, tf.float64)))
                          for actual, expected in zip(data["csmc"], saved["csmc"]))
    if not all(matches.values()):
        raise ValueError("Regenerated inputs differ from the saved test")
    outputs = []
    for coordinate in range(3):
        output = campaign.evaluate("csmc", "pooled", tf.zeros([1, 3], tf.float64),
            failed["dataset_seed"], failed["particle_seed"],
            direction=tf.one_hot(coordinate, 3, dtype=tf.float64), multiplier=0.)
        outputs.append(safe(output))
    states, covariances, noises, d_states, d_covariances = campaign.particles_for(failed["particle_seed"])
    value, score, trace = campaign.kernels["csmc"](campaign.theta,
        tf.one_hot(0, 3, dtype=tf.float64), states, covariances, noises,
        data["observations"], d_states, d_covariances,
        *(tf.zeros_like(coefficient) for coefficient in data["csmc"]))
    steps = []
    for index, step in enumerate(trace):
        scalars = {name: tensor for name, tensor in step.items() if tensor.shape.rank == 0}
        nonfinite = {name: int(tf.reduce_sum(tf.cast(~tf.math.is_finite(tensor), tf.int32)))
                     for name, tensor in step.items() if tensor.dtype.is_floating}
        steps.append(dict(time=index + 1, scalars=safe(scalars),
                          nonfinite={name: count for name, count in nonfinite.items() if count}))
    result = dict(role="exact failure localization; no tuning or ranking", failed_key=failed,
        source_hash_checks=hashes, input_matches=matches, outputs=outputs, steps=steps,
        wall_seconds=time.monotonic() - started, calls=campaign.calls + 1,
        allocator=tf.config.experimental.get_memory_info("GPU:0"))
    (arguments.output / "diagnostic.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(valid=[row["valid"] for row in outputs],
        failed_checks=[{name: flag for name, flag in row["checks"].items()
                       if not flag and (name.endswith("_valid") or name.endswith("_finite"))}
                       for row in outputs], wall_seconds=result["wall_seconds"])))


if __name__ == "__main__":
    main()

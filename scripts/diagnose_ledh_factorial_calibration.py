"""CPU-only diagnostic reproduction of factorial calibration nonconvergence."""

import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

from bayesfilter.highdim.covariance_proposal_beta_tf import make_beta_solver
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
from bayesfilter.highdim.sqmc_lgssm_tf import LGSSMSpec
from bayesfilter.score_study.guided_mixture_diagnostic_tf import make_diagnostic_programs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    started = time.monotonic()
    dtype = tf.float64
    spec = LGSSMSpec("diagonal_ar", 2)
    theta = tf.constant([.2, .2, .3, .5], dtype)
    _, pilot = make_diagnostic_programs(32, 5, ProposalControls(reset_steps=5, correction_steps=2))
    contexts = []
    for seed in (218000, 218001):
        observations = spec.simulate(theta, 5, seed)
        variance = 1.
        for _ in range(3):
            variance = .04 * variance + .09
        observations += tf.scatter_nd([[2, 0]], tf.constant([3. * math.sqrt(variance + .25)], dtype), [5, 2])
        result = pilot(theta, observations,
                       tf.random.stateless_normal([32, 2], [seed, 2], dtype=dtype),
                       tf.random.stateless_normal([5, 32, 2], [seed, 3], dtype=dtype),
                       tf.random.stateless_uniform([5, 32, 2], [seed, 4], dtype=dtype),
                       tf.constant([1. / 3.] * 3, dtype), tf.zeros([5, 2, 2], dtype),
                       tf.zeros([5, 2], dtype),
                       tf.random.stateless_normal([5, 32, 2], [seed, 10], dtype=dtype),
                       tf.random.stateless_uniform([5, 32, 2], [seed, 11], dtype=dtype))
        if not bool(result["valid"]):
            raise RuntimeError("invalid pilot context")
        contexts.append(result)
    density = tf.concat([context["log_components"] for context in contexts], 0)
    target = tf.concat([context["log_target"] for context in contexts], 0)
    attempts = []
    for maximum in (2000, 20000):
        result = make_beta_solver(10, 32, dtype, max_steps=maximum)(
            density, target, tf.constant(.1, dtype), tf.constant(1e-7, dtype))
        attempts.append({"max_steps": maximum, "beta": result["beta"].numpy().tolist(),
                         "objective": float(result["objective"]), "gap": float(result["gap"]),
                         "iterations": int(result["iterations"]), "converged": bool(result["converged"]),
                         "required_gap": 1e-7 * max(1., float(result["objective"]))})
    output = {"execution": "CPU/XLA diagnostic; GPU intentionally hidden", "condition": 1,
              "guide": "current", "seeds": [218000, 218001], "attempts": attempts,
              "wall_seconds_excluding_import": time.monotonic() - started,
              "log_components": density.numpy().tolist(), "log_target": target.numpy().tolist()}
    arguments.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in output.items()
                      if key not in ("log_components", "log_target")}))


if __name__ == "__main__":
    main()

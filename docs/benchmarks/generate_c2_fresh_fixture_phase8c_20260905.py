"""Generate a reproducible fresh C2 observation fixture with TensorFlow.

This is an offline data-generation lane. It uses no NumPy numerical path and
does not modify the preserved C2 fixture. The generated observations follow
the same stationary linear state transition and raw stochastic-volatility
observation law used by the claim-free APF diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import tensorflow as tf


DTYPE = tf.float64
ROOT = Path(__file__).resolve().parents[2]
BASE_FIXTURE = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
HORIZON = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor(value: object) -> tf.Tensor:
    return tf.convert_to_tensor(value, DTYPE)


def generate(*, state_seed: int, observation_seed: int) -> dict[str, object]:
    base = json.loads(BASE_FIXTURE.read_text(encoding="utf-8"))
    dimension = int(base["state_dimension"])
    transition = _tensor(base["transition_matrix"])
    stationary = _tensor(base["stationary_covariance"])
    sigma = tf.constant(float(base["sigma"]), DTYPE)
    xi = tf.constant(math.log(float(base["beta"])), DTYPE)
    chol = tf.linalg.cholesky(stationary)
    state_rows = []
    first_noise = tf.random.stateless_normal([dimension], [state_seed, 0], dtype=DTYPE)
    state = tf.linalg.matvec(chol, first_noise)
    state_rows.append(state)
    for time_index in range(1, HORIZON):
        noise = tf.random.stateless_normal(
            [dimension], [state_seed, time_index], dtype=DTYPE
        )
        state = tf.linalg.matvec(transition, state) + sigma * noise
        state_rows.append(state)
    states = tf.stack(state_rows)
    observation_noise = tf.random.stateless_normal(
        [HORIZON, dimension], [observation_seed, 0], dtype=DTYPE
    )
    observations = tf.exp(0.5 * states + xi) * observation_noise
    if not bool(tf.reduce_all(tf.math.is_finite(observations)).numpy()):
        raise ValueError("generated observations are non-finite")
    payload = dict(base)
    payload["observations"] = observations.numpy().tolist()
    payload["model_seed"] = int(state_seed)
    payload["observation_seed"] = int(observation_seed)
    payload["source_generator"] = str(
        Path(__file__).resolve().relative_to(ROOT)
    )
    payload["source_generator_sha256"] = _sha256(Path(__file__).resolve())
    payload["classification"] = "fresh_phase8c_tensorflow_data_generation_diagnostic"
    payload["fresh_fixture_id"] = "c2_sv_phase8c_fresh_observations_v1"
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--state-seed", type=int, required=True)
    parser.add_argument("--observation-seed", type=int, required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = generate(
        state_seed=int(args.state_seed), observation_seed=int(args.observation_seed)
    )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

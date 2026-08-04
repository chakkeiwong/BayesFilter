"""Run the bounded conditional-reference T1 authority."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_conditional_reference_tf import (
    generate_authority_pair,
)


def _jsonable(value):
    if isinstance(value, tf.Tensor):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-count", type=int, default=8192)
    parser.add_argument("--seed-a", type=int, default=92001)
    parser.add_argument("--seed-b", type=int, default=92002)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = Path(
        args.output_dir
        if args.output_dir is not None
        else "docs/plans/artifacts/zhao-cui-austria-sir-conditional-reference-t1-20260801/authority-two-seed-n8192-retry02"
    )
    output.mkdir(parents=True, exist_ok=False)
    result = generate_authority_pair(
        sample_count=args.sample_count, seed_a=args.seed_a, seed_b=args.seed_b
    )
    payload = _jsonable(result)
    (output / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "bayesfilter.zhao_cui.austria_sir.conditional_reference_manifest.v1",
        "reference_id": payload["reference_id"],
        "classification": payload["classification"],
        "command": "CUDA_VISIBLE_DEVICES=-1 python scripts/run_zhao_cui_austria_sir_conditional_reference_t1.py",
        "environment": "tf-gpu conda environment, CPU-only diagnostic",
        "cuda_visible_devices": "-1",
        "sample_count": payload["sample_count"],
        "seeds": [args.seed_a, args.seed_b],
        "artifact": "result.json",
        "nonclaims": payload["nonclaims"],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

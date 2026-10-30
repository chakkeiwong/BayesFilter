"""Reconstruct archived CPU/FP32 particle designs and verify their exact hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    archive = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=archive / "prepared")
    args = parser.parse_args()
    repository = next(p for p in archive.parents if (p / "bayesfilter").is_dir())
    sys.path.insert(0, str(repository))
    # These designs were generated on CPU. Hide CUDA before framework import.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("BAYESFILTER_PRELOAD_CUSTOM_OP", "0")
    import tensorflow as tf
    from bayesfilter.testing.structural_ledh_models_tf import make_structural_design

    tf.config.experimental.enable_tensor_float_32_execution(False)
    inventory = json.loads((archive / "design-inventory.json").read_text())
    args.output_directory.mkdir(parents=True, exist_ok=True)
    restored = existing = 0
    for entry in inventory["designs"]:
        destination = args.output_directory / Path(entry["path"]).name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != entry["sha256"]:
                raise RuntimeError(f"Existing file differs; refusing overwrite: {destination}")
            existing += 1
            continue
        design = make_structural_design(entry["particles"], entry["horizon"],
                                        entry["design_seed"], dtype=tf.float32)
        content = (json.dumps({k: v.numpy().tolist() for k, v in design.items()},
                              indent=2, allow_nan=False) + "\n").encode()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Reconstruction mismatch with TensorFlow {tf.__version__}: {destination}")
        destination.write_bytes(content)
        restored += 1
    print(json.dumps(dict(restored=restored, existing_verified=existing,
                          tensorflow=tf.__version__, device="CPU; CUDA intentionally hidden",
                          output_directory=str(args.output_directory))))


if __name__ == "__main__":
    main()

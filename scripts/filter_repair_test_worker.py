"""Configure the campaign device before pytest imports numerical test modules."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("FILTER_REPAIR_SOURCE_ROOT", str(Path(__file__).resolve().parents[1])))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if __name__ == "__main__":
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=os.environ.get("CUDA_VISIBLE_DEVICES") != "-1")
    print(json.dumps({"tensorflow_version": tf.__version__, "gpu_memory_policy": memory_policy,
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
                      "trust_basis": "owner_designated_managed_session_visible_gpu_trusted"
                      if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" else "explicit_cpu_reference"}), flush=True)
    import pytest

    raise SystemExit(pytest.main(sys.argv[1:]))

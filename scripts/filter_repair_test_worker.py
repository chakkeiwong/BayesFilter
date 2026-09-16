"""Configure the campaign device before pytest imports numerical test modules."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

if __name__ == "__main__":
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    configure_tensorflow_gpu_memory_growth(tf, require_gpu=os.environ.get("CUDA_VISIBLE_DEVICES") != "-1")
    import pytest

    raise SystemExit(pytest.main(sys.argv[1:]))

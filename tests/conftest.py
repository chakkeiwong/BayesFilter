from __future__ import annotations

import os
import sys
from pathlib import Path

_DEVICE_SCOPE = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE", "cpu").strip().lower()
if _DEVICE_SCOPE not in {"cpu", "visible"}:
    raise RuntimeError("BAYESFILTER_TEST_DEVICE_SCOPE must be 'cpu' or 'visible'")
if _DEVICE_SCOPE == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# TensorFlow 2.21 can lose the custom XLA registration if the op is first
# loaded after another compiled graph. Load it before test modules are
# imported; this is intentionally test-scoped, not a global package setting.
if os.environ.get("BAYESFILTER_PRELOAD_CUSTOM_OP") is None:
    os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

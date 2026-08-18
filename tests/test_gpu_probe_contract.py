from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts" / "run_tensorflow_gpu_probe.py"
SELECTOR = ROOT / "scripts" / "select_preferred_gpu.py"
SHELL_HELPER = ROOT / "scripts" / "select_preferred_gpu_env.sh"


def test_probe_records_uuid_tensorflow_and_allocator_provenance() -> None:
    source = PROBE.read_text(encoding="utf-8")
    ast.parse(source)

    assert "selected_nvidia_gpu" in source
    assert "configure_tensorflow_gpu_memory_growth" in source
    assert 'with tf.device("/GPU:0")' in source
    assert "get_memory_info" in source
    assert "nonclaims" in source
    assert "inference.hmc" not in source


def test_selector_and_shell_helper_emit_uuid_and_physical_metadata() -> None:
    selector = SELECTOR.read_text(encoding="utf-8")
    helper = SHELL_HELPER.read_text(encoding="utf-8")

    assert "pci.bus_id" in selector
    assert "memory.used" in selector
    assert 'CUDA_VISIBLE_DEVICES="${BAYESFILTER_SELECTED_GPU_UUID}"' in helper
    assert "BAYESFILTER_SELECTED_GPU_PCI_BUS_ID" in helper

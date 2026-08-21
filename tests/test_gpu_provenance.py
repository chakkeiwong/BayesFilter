from __future__ import annotations

import pytest

from bayesfilter.runtime.gpu_provenance import (
    GPUProvenanceError,
    parse_nvidia_smi_rows,
    selected_nvidia_gpu,
)


def test_selected_gpu_requires_uuid_and_preserves_physical_identity() -> None:
    rows = parse_nvidia_smi_rows(
        "0, GPU-5080, RTX 5080, 00000000:01:00.0, 0, 4096, 12000, 16096\n"
        "1, GPU-4080, RTX 4080, 00000000:09:00.0, 0, 11, 16000, 16011\n"
    )

    selected = selected_nvidia_gpu(rows, cuda_visible_devices="GPU-4080")

    assert selected["nvidia_smi_index"] == 1
    assert selected["uuid"] == "GPU-4080"
    assert selected["pci_bus_id"] == "00000000:09:00.0"


@pytest.mark.parametrize("value", ["", "0", "1", "GPU-a,GPU-b"])
def test_selected_gpu_rejects_ambiguous_cuda_ordinals(value: str) -> None:
    with pytest.raises(GPUProvenanceError, match="stable NVIDIA GPU UUID"):
        selected_nvidia_gpu((), cuda_visible_devices=value)


def test_selected_gpu_fails_when_uuid_is_not_in_snapshot() -> None:
    with pytest.raises(GPUProvenanceError, match="absent"):
        selected_nvidia_gpu((), cuda_visible_devices="GPU-missing")

"""Select the repository-preferred physical GPU from a trusted live snapshot."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.runtime.device_policy import (
    build_trusted_gpu_snapshot,
    select_preferred_gpu,
)


def _probe() -> list[dict[str, object]]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,utilization.gpu,memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    rows = []
    for line in output.splitlines():
        if not line.strip():
            continue
        index, uuid, name, utilization, free_mib, total_mib = (
            part.strip() for part in line.split(",")
        )
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "utilization_gpu_pct": float(utilization),
                "memory_free_mb": float(free_mib),
                "memory_total_mb": float(total_mib),
                "memory_used_mb": float(total_mib) - float(free_mib),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preferred-gpu", type=int, default=1)
    parser.add_argument("--fallback-gpu", type=int, default=0)
    parser.add_argument("--maximum-utilization-percent", type=float, default=50.0)
    parser.add_argument("--minimum-free-mib", type=float, default=8192.0)
    args = parser.parse_args()

    rows = _probe()
    snapshot = build_trusted_gpu_snapshot(
        rows,
        trusted_or_escalated=True,
        source="trusted_nvidia_smi_live_probe",
    )
    selection = select_preferred_gpu(
        (),
        preferred_gpu=args.preferred_gpu,
        fallback_gpu=args.fallback_gpu,
        gpu_snapshot=snapshot,
        busy_memory_fraction=1.0,
        busy_utilization_pct=args.maximum_utilization_percent,
        minimum_free_memory_mb=args.minimum_free_mib,
    )
    if selection.selected_gpu is None:
        raise SystemExit(
            "no eligible GPU: require utilization < "
            f"{args.maximum_utilization_percent:g}% and free memory > "
            f"{args.minimum_free_mib:g} MiB; reason={selection.reason}"
        )
    row = next(item for item in rows if item["index"] == selection.selected_gpu)
    print(
        "\t".join(
            (
                str(row["index"]),
                str(row["uuid"]),
                str(row["name"]),
                f"{float(row['utilization_gpu_pct']):g}",
                f"{float(row['memory_free_mb']):g}",
                selection.reason,
            )
        )
    )


if __name__ == "__main__":
    main()

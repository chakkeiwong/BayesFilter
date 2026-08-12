#!/usr/bin/env python3
"""Build one fixed independent Zhao-Cui bounded-feature teacher artifact."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim.zhao_cui_austria_sir_bounded_teacher_tf import (  # noqa: E402
    build_austria_t1_t2_bounded_teacher,
    save_austria_t1_t2_bounded_teacher,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--t1-seed", type=int, default=98501)
    parser.add_argument("--t2-seed", type=int, default=98502)
    args = parser.parse_args()
    started = time.perf_counter()
    build = build_austria_t1_t2_bounded_teacher(
        sample_count=args.sample_count,
        seeds=(args.t1_seed, args.t2_seed),
    )
    manifest = save_austria_t1_t2_bounded_teacher(
        build, args.output_dir.resolve()
    )
    print(
        json.dumps(
            {
                "status": "PASS_FIXED_SAMPLED_ZHAO_CUI_BOUNDED_TEACHER",
                "manifest": manifest.as_posix(),
                "sample_count": args.sample_count,
                "seeds": [args.t1_seed, args.t2_seed],
                "wall_time_seconds": time.perf_counter() - started,
            }
        )
    )


if __name__ == "__main__":
    main()

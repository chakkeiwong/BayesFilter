#!/usr/bin/env python3
"""CLI for the NeuTra F0 new-fixture plain-HMC comparator."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from bayesfilter.testing.lgssm_new_fixture_plain_hmc_f0_tf import (
        run_f0_plain_hmc_comparator,
    )

    result = run_f0_plain_hmc_comparator()
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("passed", "decision", "artifact_hash")
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())

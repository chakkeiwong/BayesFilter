"""Verify the immutable Phase 7 historical archive without changing it."""

from __future__ import annotations

import os
import sys
from pathlib import Path


sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_serious_authority import (  # noqa: E402
    write_historical_archive_bundle,
)


def main() -> int:
    write_historical_archive_bundle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

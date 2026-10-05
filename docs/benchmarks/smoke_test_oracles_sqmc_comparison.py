#!/usr/bin/env python3
"""Compatibility entrypoint for the canonical SQMC GPU smoke."""

import sys

from docs.benchmarks.run_sqmc_oracle_characterization import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(["--stage", "smoke"])
    raise SystemExit(main())

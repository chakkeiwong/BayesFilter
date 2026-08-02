#!/usr/bin/env python3
"""Retired P76 diagnostic that depended on the SIR-UKF warm-start pilot."""

from __future__ import annotations

import sys


RETIREMENT_REASON = (
    "retired: this diagnostic depends on the owner-excluded SIR-UKF P76 route; "
    "historical P76 artifacts remain provenance only"
)


def main() -> int:
    raise SystemExit(RETIREMENT_REASON)


if __name__ == "__main__":
    sys.exit(main())

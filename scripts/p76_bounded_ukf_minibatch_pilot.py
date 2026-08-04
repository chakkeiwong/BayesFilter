#!/usr/bin/env python3
"""Retired P76 SIR-UKF warm-start pilot entry point."""

from __future__ import annotations

import sys


RETIREMENT_REASON = (
    "retired: UKF does not work for SIR and SIR-UKF is owner-excluded from "
    "testing; historical P76 artifacts remain provenance only"
)


def main() -> int:
    raise SystemExit(RETIREMENT_REASON)


if __name__ == "__main__":
    sys.exit(main())

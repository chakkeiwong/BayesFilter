#!/usr/bin/env python3
"""Retired P77 training entry point built on the SIR-UKF P76 route."""

from __future__ import annotations

import sys


RETIREMENT_REASON = (
    "retired: this training route depends on the owner-excluded SIR-UKF P76 "
    "route; historical P77 artifacts remain provenance only"
)


def main() -> int:
    raise SystemExit(RETIREMENT_REASON)


if __name__ == "__main__":
    sys.exit(main())

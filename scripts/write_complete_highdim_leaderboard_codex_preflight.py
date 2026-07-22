#!/usr/bin/env python3
"""Write the structured result of a trusted noninteractive Codex probe."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import hashlib
from pathlib import Path
from typing import Sequence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--final-message", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--runner-script", type=Path, required=True)
    args = parser.parse_args(argv)
    token = args.final_message.read_text(encoding="utf-8").strip()
    payload = {
        "schema_version": "bayesfilter.complete_highdim_leaderboard.codex_preflight.v1",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "exit_code": args.exit_code,
        "probe_token": token,
        "events_path": str(args.events),
        "final_message_path": str(args.final_message),
        "stderr_path": str(args.stderr),
        "noninteractive": True,
        "trusted_execution": True,
        "runner_script_path": str(args.runner_script.resolve()),
        "runner_script_sha256": _sha256(args.runner_script.resolve()),
        "writer_script_path": str(Path(__file__).resolve()),
        "writer_script_sha256": _sha256(Path(__file__).resolve()),
        "preflight_pass": args.exit_code == 0 and token == "CODEX_PROBE_OK",
        "nonclaims": ["Codex health does not authorize launch or scientific claims"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    return 0 if payload["preflight_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

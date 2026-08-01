"""Materialize one exact, manifest-bound Phase 6 smoke authority."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence


sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name, _value in {
    "TF_NUM_INTRAOP_THREADS": "8",
    "TF_NUM_INTEROP_THREADS": "1",
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}.items():
    os.environ[_name] = _value
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter-phase6-authority")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_smoke_authority import (  # noqa: E402
    AUTHORITY_PATH,
    ConsumedAttempt1EvidenceSession,
    PROPOSAL_MANIFEST_PATH,
    build_smoke_authority,
    expected_smoke_approval_statement,
    parse_smoke_authority,
    verify_default_smoke_authority_proposal_bundle,
    verify_smoke_authority,
    write_phase6_json,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--approval-statement", required=True)
    parser.add_argument("--approval-date", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with ConsumedAttempt1EvidenceSession.open() as consumed_evidence_session:
        _proposal, manifest, _config, _preflight = (
            verify_default_smoke_authority_proposal_bundle(
                python_executable=Path(sys.executable).resolve(),
                consumed_evidence_session=consumed_evidence_session,
            )
        )
        expected = expected_smoke_approval_statement(manifest["artifact_hash"])
        if args.approval_statement != expected:
            raise ValueError("smoke human approval statement mismatch")
        authority = build_smoke_authority(
            proposal_manifest_path=PROPOSAL_MANIFEST_PATH,
            human_approval_statement=args.approval_statement,
            human_approval_date=args.approval_date,
        )
        write_phase6_json(
            AUTHORITY_PATH,
            authority,
            parser=parse_smoke_authority,
            consumed_evidence_session=consumed_evidence_session,
        )
        verify_smoke_authority(
            authority,
            proposal_manifest_path=PROPOSAL_MANIFEST_PATH,
        )
        consumed_evidence_session.verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

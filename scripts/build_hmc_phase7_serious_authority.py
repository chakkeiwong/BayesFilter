"""Materialize one exact manifest-bound serious Phase 7 authority."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter-phase7-authority")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_serious_authority import (  # noqa: E402
    AUTHORITY_PATH,
    PROPOSAL_MANIFEST_PATH,
    PROPOSAL_PATH,
    SeriousInheritedEvidenceSession,
    _artifact_reference_from_snapshot,
    build_serious_authority,
    expected_serious_approval_statement,
    parse_serious_authority,
    verify_serious_authority_proposal_candidate,
    verify_serious_authority_proposal_manifest,
)
from bayesfilter.inference.hmc_smoke_authority import write_phase6_json  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--approval-statement", required=True)
    parser.add_argument("--approval-date", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with SeriousInheritedEvidenceSession.open(
        extra_paths=(PROPOSAL_PATH, PROPOSAL_MANIFEST_PATH)
    ) as evidence:
        manifest = evidence.read_pinned_json(PROPOSAL_MANIFEST_PATH)
        verify_serious_authority_proposal_manifest(
            manifest,
            proposal_payload=evidence.read_pinned_json(PROPOSAL_PATH),
            proposal_bytes=evidence.read_pinned_bytes(PROPOSAL_PATH),
        )
        proposal = evidence.read_pinned_json(PROPOSAL_PATH)
        verify_serious_authority_proposal_candidate(
            proposal,
            python_executable=Path(sys.executable).resolve(),
            evidence_session=evidence,
        )
        if args.approval_statement != expected_serious_approval_statement(
            manifest["artifact_hash"]
        ):
            raise ValueError("serious human approval statement mismatch")
        authority = build_serious_authority(
            approval_statement=args.approval_statement,
            approval_date=args.approval_date,
            proposal_manifest=manifest,
            proposal_manifest_reference=_artifact_reference_from_snapshot(
                path=PROPOSAL_MANIFEST_PATH,
                payload=manifest,
                data=evidence.read_pinned_bytes(PROPOSAL_MANIFEST_PATH),
            ),
        )
        write_phase6_json(
            AUTHORITY_PATH,
            authority,
            parser=parse_serious_authority,
            consumed_evidence_session=evidence,
        )
        evidence.pin_additional(AUTHORITY_PATH)
        parse_serious_authority(evidence.read_pinned_json(AUTHORITY_PATH))
        evidence.verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

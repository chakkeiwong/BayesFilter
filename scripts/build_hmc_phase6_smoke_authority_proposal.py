"""Build the pending Phase 6 smoke proposal and its terminal manifest."""

from __future__ import annotations

import os
import json
import sys
from pathlib import Path


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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter-phase6-proposal")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_smoke_authority import (  # noqa: E402
    ConsumedAttempt1EvidenceSession,
    PROPOSAL_MANIFEST_PATH,
    PROPOSAL_PATH,
    build_default_smoke_authority_proposal,
    build_smoke_authority_proposal_manifest,
    parse_smoke_authority_proposal,
    parse_smoke_authority_proposal_manifest,
    verify_default_smoke_authority_proposal_candidate,
    verify_smoke_authority_proposal_manifest,
    write_phase6_json,
)


def main() -> int:
    python_executable = Path(sys.executable).resolve()
    with ConsumedAttempt1EvidenceSession.open() as consumed_evidence_session:
        proposal = build_default_smoke_authority_proposal(
            python_executable=python_executable
        )
        verify_default_smoke_authority_proposal_candidate(
            proposal,
            python_executable=python_executable,
            consumed_evidence_session=consumed_evidence_session,
        )
        write_phase6_json(
            PROPOSAL_PATH,
            proposal,
            parser=parse_smoke_authority_proposal,
            consumed_evidence_session=consumed_evidence_session,
        )
        manifest = build_smoke_authority_proposal_manifest(proposal_path=PROPOSAL_PATH)
        write_phase6_json(
            PROPOSAL_MANIFEST_PATH,
            manifest,
            parser=parse_smoke_authority_proposal_manifest,
            consumed_evidence_session=consumed_evidence_session,
        )
        verify_smoke_authority_proposal_manifest(
            parse_smoke_authority_proposal_manifest(
                json.loads(PROPOSAL_MANIFEST_PATH.read_text(encoding="utf-8"))
            ),
            proposal_path=PROPOSAL_PATH,
        )
        consumed_evidence_session.verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

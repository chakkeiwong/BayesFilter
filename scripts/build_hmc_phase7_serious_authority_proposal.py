"""Build the pending serious Phase 7 authority proposal and terminal manifest."""

from __future__ import annotations

import os
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter-phase7-proposal")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_serious_authority import (  # noqa: E402
    PROPOSAL_MANIFEST_PATH,
    PROPOSAL_PATH,
    SeriousInheritedEvidenceSession,
    _artifact_reference_from_snapshot,
    build_default_serious_authority_proposal,
    build_serious_authority_proposal_manifest,
    parse_serious_authority_proposal,
    parse_serious_authority_proposal_manifest,
    verify_serious_authority_proposal_candidate,
    verify_serious_authority_proposal_manifest,
)
from bayesfilter.inference.hmc_smoke_authority import write_phase6_json  # noqa: E402


def main() -> int:
    python = Path(sys.executable).resolve()
    with SeriousInheritedEvidenceSession.open() as evidence:
        proposal = build_default_serious_authority_proposal(
            python_executable=python,
            evidence_session=evidence,
        )
        verify_serious_authority_proposal_candidate(
            proposal,
            python_executable=python,
            evidence_session=evidence,
        )
        write_phase6_json(
            PROPOSAL_PATH,
            proposal,
            parser=parse_serious_authority_proposal,
            consumed_evidence_session=evidence,
        )
        evidence.pin_additional(PROPOSAL_PATH)
        proposal_bytes = evidence.read_pinned_bytes(PROPOSAL_PATH)
        proposal_reference = _artifact_reference_from_snapshot(
            path=PROPOSAL_PATH,
            payload=evidence.read_pinned_json(PROPOSAL_PATH),
            data=proposal_bytes,
        )
        manifest = build_serious_authority_proposal_manifest(
            proposal_reference=proposal_reference
        )
        write_phase6_json(
            PROPOSAL_MANIFEST_PATH,
            manifest,
            parser=parse_serious_authority_proposal_manifest,
            consumed_evidence_session=evidence,
        )
        evidence.pin_additional(PROPOSAL_MANIFEST_PATH)
        restored_proposal = evidence.read_pinned_json(PROPOSAL_PATH)
        restored_manifest = evidence.read_pinned_json(PROPOSAL_MANIFEST_PATH)
        verify_serious_authority_proposal_manifest(
            restored_manifest,
            proposal_payload=restored_proposal,
            proposal_bytes=evidence.read_pinned_bytes(PROPOSAL_PATH),
        )
        verify_serious_authority_proposal_candidate(
            restored_proposal,
            python_executable=python,
            evidence_session=evidence,
        )
        evidence.verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Current-source HMC/validation regressions with source and test inventory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]


def sources():
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((REPO / "bayesfilter").rglob("*.py"))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="affected-tests-r2")
    parser.add_argument("--after", default="")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    unavailable = {"test_fixed_transport_hmc_binding_c603_fixture.py":
                       "requires absent external /tmp/dsge_hmc-neutra-handoff-20260705",
                   "test_hmc_academic_campaign.py":
                       "requires absent July09 private migration certificate",
                   "test_hmc_identity_adoption.py":
                       "superseded July adoption ceremony; absent or drifted historical fixtures",
                   "test_hmc_identity_migration_certificate.py":
                       "superseded July migration ceremony; source_payloads requires absent private replay",
                   "test_hmc_serious_authority.py":
                       "superseded hash-bound authority/launch ceremony; outside current academic policy",
                   "test_hmc_smoke_authority.py":
                       "superseded authority/reservation ceremony; outside current academic policy"}
    tests = sorted(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_*.py")
                   if p.name.startswith(("test_hmc_", "test_fixed_transport_")) and p.name not in unavailable
                   and p.name > args.after)
    tests += ["tests/test_neural_force_hmc.py", "tests/test_dense_iaf_neutra_artifact_loader.py",
              "tests/inference_validation"]
    command = [sys.executable, "-m", "pytest", "-q", *tests, "-m", "not extended", "--maxfail=8",
               "--deselect=tests/test_hmc_full_estimation_campaign.py::test_v3_preflight_exposes_private_replay_hash_for_retained_archive",
               "--deselect=tests/test_hmc_identity_integration.py::test_real_selection_provenance_binds_named_selected_attempt_lineage",
               "--deselect=tests/test_hmc_identity_integration.py::test_real_phase3_opt_in_path_persists_evidence_then_reraises_legacy_veto",
               "--deselect=tests/test_hmc_identity_integration.py::test_legacy_phase7_validator_retains_exact_fail_closed_behavior",
               "--junitxml=" + str(ROOT / (args.label + ".xml"))]
    before = sources()
    code = subprocess.run(command, cwd=REPO).returncode
    after = sources()
    with (ROOT / (args.label + "-source.json")).open("x") as handle:
        json.dump({"command": command, "tests": tests, "returncode": code,
                   "unavailable_historical_fixtures": unavailable,
                   "resume_after_file": args.after,
                   "unavailable_individual_nodes": "July09/13 private replay and exact historical validator outcomes; listed in command",
                   "extended_selection": "excluded from this routine batch; current serious public paths have phase-specific multi-model campaigns",
                   "before": before, "after": after,
                   "changed_sources": sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)),
                   "device": "cpu_reference", "gpu_intentionally_hidden": True}, handle, indent=2)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

"""Exact source migration for the 2026-10-18 HMC orchestration repair.

Only the reviewed resume/tuning arguments and this scope-hash call changed
in the runtime module. Numerical sources retain their exact saved hashes.
This compatibility identity never replaces executed-source provenance.
"""
import hashlib
from pathlib import Path


RUNTIME_PATH = "bayesfilter/testing/surrogate_hmc_five_model.py"
SAVED_RUNTIME_SHA256 = "e9d2176cc30b709a0acc3f5f34e627f9a1670624500151e272bb68834a392c1e"
REVIEWED_RUNTIME_SHA256 = "6b59d2f6f087a3b168b77dbb942997917c3bf8d0cf4bb99ba4d3b085192cfa2e"
MIGRATION_ID = "surrogate_hmc_orchestration_only_resume_20261018_v1"


def scope_source_hashes(actual):
    """Normalize only one exact reviewed revision; unknown revisions stay strict."""
    result = dict(actual)
    if result.get(RUNTIME_PATH) == REVIEWED_RUNTIME_SHA256:
        result[RUNTIME_PATH] = SAVED_RUNTIME_SHA256
    return result


def source_migration_provenance(root):
    """Expose actual and compatibility hashes without concealing source edits."""
    root = Path(root)
    actual = hashlib.sha256((root / RUNTIME_PATH).read_bytes()).hexdigest()
    return dict(
        migration_id=MIGRATION_ID if actual == REVIEWED_RUNTIME_SHA256 else None,
        runtime_path=RUNTIME_PATH,
        executed_runtime_sha256=actual,
        tuning_scope_runtime_sha256=scope_source_hashes({RUNTIME_PATH: actual})[RUNTIME_PATH],
        compatibility_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )

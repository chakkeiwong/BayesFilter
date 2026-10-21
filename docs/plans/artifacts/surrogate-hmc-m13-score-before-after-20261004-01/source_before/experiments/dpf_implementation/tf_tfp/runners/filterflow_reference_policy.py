"""Reference policy for BayesFilter-vs-filterflow comparison runners.

The active comparison lane uses the local float64 filterflow branch. Historical
artifacts may still record earlier float32/compatibility branches as provenance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


FILTERFLOW_REFERENCE_BRANCH = "bayesfilter-py311-float64-reference"
FILTERFLOW_REFERENCE_COMMIT = "1e5fbc288c1c11fc18ba01bb4842832e2088b800"
FILTERFLOW_UPSTREAM_BASE_COMMIT = "5d8300ba247c4c17e1a301a22560c24fd0670bfe"
FILTERFLOW_REFERENCE_DTYPE = "float64"
FILTERFLOW_BRANCH_MARKER = "BAYESFILTER_FLOAT64_REFERENCE.md"


def reference_policy() -> dict[str, str]:
    return {
        "future_comparator": "filterflow_float64_reference_branch",
        "branch": FILTERFLOW_REFERENCE_BRANCH,
        "commit": FILTERFLOW_REFERENCE_COMMIT,
        "upstream_base": FILTERFLOW_UPSTREAM_BASE_COMMIT,
        "dtype": FILTERFLOW_REFERENCE_DTYPE,
        "local_reference_status": "BayesFilter audit reference code, not pristine upstream",
        "transition_covariance": "I_2 executable reproduction setting",
        "fixed_target_sinkhorn": "local BayesFilter diagnostic/comparator only",
    }


def validate_filterflow_reference_status(
    status: dict[str, Any],
    *,
    marker_path: Path | None = None,
) -> None:
    if status.get("branch") != FILTERFLOW_REFERENCE_BRANCH:
        raise RuntimeError(f"filterflow branch mismatch: {status.get('branch')}")
    if status.get("commit") != FILTERFLOW_REFERENCE_COMMIT:
        raise RuntimeError(f"filterflow commit mismatch: {status.get('commit')}")
    if marker_path is not None and not marker_path.exists():
        raise RuntimeError(f"missing filterflow float64 branch marker: {marker_path}")

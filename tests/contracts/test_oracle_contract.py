"""Sustained oracle contract: analytical JVP stages vs autodiff reference.

This module defines the binding contract for claim integrity: every analytical
tangent implementation must pass its autodiff oracle gate. Failures block
commits via pre-commit hook enforcement.

Contract authority: Phase 2C Step 2 (2026-09-11)
Tolerance: rtol 1e-6, atol 1e-8 (same as individual gates)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent


ORACLE_TEST_MODULES = [
    "tests/highdim/test_ledh_canonical_score_stages.py",
    "tests/highdim/test_ledh_canonical_score_step.py",
    "tests/highdim/test_ledh_canonical_score_ukf_tangent.py",
    "tests/highdim/test_ledh_canonical_score_full.py",
    "tests/highdim/test_ledh_unified_reset.py",
    "tests/highdim/test_ledh_unified_correction.py",
]

ORACLE_TEST_PATTERNS = [
    "test.*oracle",
    "test.*autodiff.*oracle",
    "test_analytical_tangents_match_autodiff_oracle",
]


def _collect_oracle_tests() -> list[str]:
    """Collect all oracle test node IDs from the contract modules."""
    node_ids = []
    for module in ORACLE_TEST_MODULES:
        module_path = REPO_ROOT / module
        if not module_path.exists():
            pytest.fail(f"Oracle contract module missing: {module}")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                str(module_path),
                "--collect-only",
                "-q",
                "--no-header",
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            pytest.fail(
                f"Oracle test collection failed for {module}:\n{result.stderr}"
            )
        for line in result.stdout.splitlines():
            line = line.strip()
            if "::" not in line:
                continue
            # Match tests containing "oracle" or "autodiff"
            lower = line.lower()
            if "oracle" in lower or "autodiff" in lower:
                node_ids.append(line)
    return node_ids


def test_oracle_contract_modules_exist():
    """All contract modules must exist and be importable."""
    missing = []
    for module in ORACLE_TEST_MODULES:
        if not (REPO_ROOT / module).exists():
            missing.append(module)
    assert not missing, f"Missing oracle contract modules: {missing}"


def test_oracle_contract_battery_passes():
    """Execute the full oracle battery - all must pass to satisfy the contract.

    This is the binding gate: analytical JVP implementations vs autodiff oracle.
    A failure here means a tangent does not match its reference, which violates
    the claim-integrity contract established in Phase 2C.
    """
    oracle_node_ids = _collect_oracle_tests()
    if not oracle_node_ids:
        pytest.fail(
            "Oracle contract battery is empty - collection found no oracle tests"
        )

    # Run the collected oracle tests
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *oracle_node_ids,
            "-v",
            "--tb=short",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={**subprocess.os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
    )

    if result.returncode != 0:
        pytest.fail(
            f"Oracle contract violated: {result.returncode} test(s) failed\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )


def test_oracle_contract_coverage_stable():
    """The oracle test count must not decrease across commits.

    This prevents silent removal of oracle gates. The count may increase
    (new stages added), but never decrease without explicit approval.
    """
    oracle_node_ids = _collect_oracle_tests()
    # Baseline from Phase 2C Step 1 completion (2026-09-11):
    # - 9 canonical score tests (stage/step/ukf/full)
    # - 2 unified reset/correction tests
    BASELINE_ORACLE_COUNT = 11

    actual_count = len(oracle_node_ids)
    assert actual_count >= BASELINE_ORACLE_COUNT, (
        f"Oracle contract coverage dropped: {actual_count} < {BASELINE_ORACLE_COUNT}. "
        f"Collected: {oracle_node_ids}"
    )


__all__ = [
    "test_oracle_contract_modules_exist",
    "test_oracle_contract_battery_passes",
    "test_oracle_contract_coverage_stable",
]

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "docs/benchmarks/audit_multimodel_neutra_p7.py"


def _load_audit_module():
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("multimodel_neutra_p7_audit", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imported = set(sys.modules) - before
    assert "tensorflow" not in imported
    assert "tensorflow_probability" not in imported
    return module


def test_registry_membership_and_corrected_terminal_states() -> None:
    audit = _load_audit_module()

    result = audit._registry_check()

    assert result["passed"] is True
    assert len(result["cell_ids"]) == 11
    assert audit.EXPECTED_STATES["SIR-SGQF"] == "NEUTRA_CONFIRMED"
    assert audit.EXPECTED_STATES["PP-UKF"] == "NEUTRA_CONFIRMED"
    assert audit.EXPECTED_STATES["PP-SGQF"] == "NEUTRA_CONFIRMED"


def test_all_confirmation_cells_have_disjoint_tuning_admission() -> None:
    audit = _load_audit_module()

    pp_ukf = audit._verify_confirmation_cell("PP-UKF", audit.R4_ROOTS["PP-UKF"])
    pp_sgqf = audit._verify_confirmation_cell("PP-SGQF", audit.R4_ROOTS["PP-SGQF"])
    sir = audit._verify_confirmation_cell("SIR-SGQF", audit.R4_ROOTS["SIR-SGQF"])

    assert pp_ukf["tuning_admission"]["classification"] == "TUNING_ADMITTED"
    assert pp_sgqf["tuning_admission"]["classification"] == "TUNING_ADMITTED"
    assert pp_ukf["tuning_admission"]["modern_rhat"] <= 1.01
    assert pp_sgqf["tuning_admission"]["modern_rhat"] <= 1.01
    assert pp_ukf["tuning_admission"]["archive_excluded_from_posterior"] is True
    assert pp_sgqf["tuning_admission"]["archive_excluded_from_posterior"] is True
    assert pp_ukf["final_sampler_diagnostics"]["passed"] is True
    assert pp_sgqf["final_sampler_diagnostics"]["passed"] is True
    assert sir["tuning_admission"]["classification"] == "TUNING_ADMITTED"
    assert sir["tuning_admission"]["modern_rhat"] <= 1.01


def test_blocked_state_semantics_and_static_policy_scan() -> None:
    audit = _load_audit_module()

    blocker_rows = audit._blocked_semantics()
    static_scan = audit._static_policy_scan()

    assert {row["cell_id"] for row in blocker_rows} == {
        "SVX-SGQF",
        "SVX-ZC",
        "KSC-UKF",
        "PP-ZC",
        "STR-UKF",
        "STR-ZC",
        "SIR-UKF",
        "SIR-ZC",
    }
    assert all(row["passed"] is True for row in blocker_rows)
    assert static_scan["passed"] is True
    assert static_scan["numpy_or_host_callback_hits"] == []
    assert audit._inherited_claim_scan()["findings"] == []

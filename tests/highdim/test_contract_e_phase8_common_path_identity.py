from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/benchmarks/diagnose_contract_e_phase8_common_path_identity.py"


def _module():
    spec = importlib.util.spec_from_file_location("phase8_common_path_identity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_common_proposal_weight_value_and_derivative_identities() -> None:
    payload = _module().compute_diagnostic()
    assert payload["status"] == "COMMON_PATH_IDENTITY_PASS"
    assert payload["passed"] is True
    assert payload["checks"]["all_finite"] is True
    for name, check in payload["checks"].items():
        if name != "all_finite":
            assert check["all_close"] is True, name

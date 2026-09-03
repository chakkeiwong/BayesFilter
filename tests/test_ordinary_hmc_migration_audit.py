from __future__ import annotations

from pathlib import Path

from scripts.audit_ordinary_hmc_migration_surface import (
    _numpy_ledger,
    _scan_python_file,
)


def test_static_audit_marks_computed_hmc_imports_unknown(tmp_path: Path) -> None:
    source = """
import importlib
from bayesfilter.inference import tune_hmc_kernel

def load(name, module):
    imported = importlib.import_module(module)
    return getattr(imported, name)
"""
    path = tmp_path / "consumer.py"
    path.write_text(source, encoding="utf-8")

    row = _scan_python_file(path, root_label="test")

    assert row["consumer_role"] == "unknown_dynamic_import"
    assert {
        item["classification"] for item in row["dynamic_imports"]
    } == {"unknown_dynamic_import"}
    assert row["claim_adjacent"] is False


def test_static_audit_separates_unresolved_attributes_from_imports(
    tmp_path: Path,
) -> None:
    path = tmp_path / "consumer.py"
    path.write_text(
        "from bayesfilter.inference import tune_hmc_kernel\n"
        "def read(result, field):\n"
        "    return getattr(result, field)\n",
        encoding="utf-8",
    )

    row = _scan_python_file(path, root_label="test")

    assert row["consumer_role"] == "unknown_dynamic_import"
    assert row["dynamic_imports"][0]["classification"] == (
        "unresolved_dynamic_attribute"
    )


def test_static_audit_numpy_ledger_labels_ordinary_runtime_module(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hmc_kernel_tuning.py"
    path.write_text("import numpy as np\nvalue = np.zeros(2)\n", encoding="utf-8")

    row = _scan_python_file(path, root_label="bayesfilter")
    ledger = _numpy_ledger((row,))

    assert ledger[0]["classification"] == "runtime_candidate"
    assert ledger[0]["calls"][0]["name"] == "numpy.zeros"

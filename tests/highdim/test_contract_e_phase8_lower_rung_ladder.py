from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks


ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / "docs/benchmarks/run_contract_e_phase8_lower_rung_ladder.py"
WORKER = ROOT / "docs/benchmarks/run_contract_e_phase8_lower_rung_node.py"


def _load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


worker = _load("contract_e_lower_rung_worker", WORKER)


def test_old_lower_rung_chunk_ladder_is_archival_and_fail_closed() -> None:
    tree = ast.parse(DRIVER.read_text(encoding="utf-8"))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    first = main.body[0]
    assert isinstance(first, ast.Raise)
    assert "ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY" in ast.unparse(first)


def test_worker_parses_decimal_and_hexadecimal_ridges() -> None:
    assert worker._parse_float("0.125") == 0.125
    assert worker._parse_float("0x1.0000000000000p-3") == 0.125


def test_current_worker_uses_policy_chunk_for_lower_rung() -> None:
    chunks = select_transport_chunks(32)
    assert chunks.row_chunk_size == 32
    assert chunks.col_chunk_size == 32
    source = WORKER.read_text(encoding="utf-8")
    assert "select_transport_chunks(args.num_particles)" in source
    assert "--row-chunk-size" not in source
    assert "--col-chunk-size" not in source


def test_lower_rung_has_a_likelihood_increment_after_first_reset() -> None:
    assert worker.TIME_STEPS == 2
    assert worker.TIME_STEPS > 1


def test_no_reset_infinite_minimum_mass_is_an_inactive_sentinel() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "no_reset_weighted" in source
    assert "inactive_minimum_mass_sentinel" in source


def test_source_hash_contract_matches_current_guarded_closure() -> None:
    realized = {
        name: worker._sha256_path(ROOT / name)
        for name in worker.EXPECTED_SOURCE_SHA256
    }
    assert realized == worker.EXPECTED_SOURCE_SHA256

"""Focused contract tests for the material annealed-SMC supervisor."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_physical_annealed_smc_material_2026_08_10.py"


def _load_runner():
    name = "test_ssl_lstm_q20_physical_annealed_smc_material_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_batch_specs_are_independent_and_match_material_design() -> None:
    runner = _load_runner()
    specs = runner.batch_specs()
    assert len(specs) == 10
    assert [spec["family"] for spec in specs[:8]] == ["central"] * 8
    assert [spec["target_ess_fraction"] for spec in specs[:8]] == [0.80] * 8
    assert [spec["family"] for spec in specs[8:]] == ["sensitivity"] * 2
    assert [spec["target_ess_fraction"] for spec in specs[8:]] == [0.70] * 2
    offsets = [spec["seed_domain_offset"] for spec in specs]
    assert len(set(offsets)) == 10
    assert min(b - a for a, b in zip(sorted(offsets), sorted(offsets)[1:])) >= 10000


def test_child_commands_bind_versioned_output_seed_cess_and_documents() -> None:
    runner = _load_runner()
    for spec in runner.batch_specs():
        command = runner._child_command(spec)
        output = runner._child_output(spec)
        assert output.parent == runner.OUTPUT_ROOT
        assert ".." not in output.parts
        assert command[command.index("--output-root") + 1] == output.as_posix()
        assert float(command[command.index("--target-ess-fraction") + 1]) == spec[
            "target_ess_fraction"
        ]
        assert int(command[command.index("--seed-domain-offset") + 1]) == spec[
            "seed_domain_offset"
        ]
        assert command[command.index("--plan-file") + 1] == runner.PLAN.as_posix()
        assert command[command.index("--result-file") + 1] == runner.RESULT.as_posix()


def test_material_aggregation_uses_terminal_pre_resampling_per_run_measures() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'terminal.get("terminal_pre_resampling")' in source
    assert 'terminal.get("resampled")' in source
    assert '_load_terminal_tensor(child, "normalized_weights"' in source
    assert '_load_terminal_tensor(child, "sign"' in source
    assert "tf.reduce_mean(central_estimates)" in source
    assert "tf.concat" not in source


def test_material_campaign_is_bounded_and_fail_closed() -> None:
    runner = _load_runner()
    source = RUNNER.read_text(encoding="utf-8")
    assert runner.CHILD_TIMEOUT_SECONDS == 900.0
    assert runner.RUNNER_CAP_SECONDS < 4200.0
    assert "start_new_session=True" in source
    assert "os.killpg" in source
    assert "SMC_MATERIAL_HARNESS_FAILED" in source
    assert runner.OUTPUT_ROOT.as_posix().endswith("/r2")


def test_stage_receipt_groups_support_nested_and_historical_schemas() -> None:
    runner = _load_runner()
    pre = {"z": {"path": "pre"}}
    post = {"z": {"path": "post"}}
    assert runner._stage_receipt_groups(
        {"receipts": {"pre": pre, "post": post}}
    ) == {"pre": pre, "post": post}
    assert runner._stage_receipt_groups({"receipts": pre}) == {
        "pre": pre,
        "post": {},
    }

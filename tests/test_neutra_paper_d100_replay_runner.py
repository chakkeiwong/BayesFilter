"""Contract tests for CPU-only exact d100 replay generation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/generate_neutra_paper_d100_exact_replay_2026_08_13.py"
CONSTANTS = ROOT / (
    "docs/plans/artifacts/weighted-forward-kl-positive-controls-2026-08-12/"
    "paper-d100/source-r1/paper_ill_cond_gaussian_d100_constants.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runner_emits_disjoint_hash_bound_cpu_replay(tmp_path: Path) -> None:
    output = tmp_path / "replay"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.run(
        (
            sys.executable,
            str(RUNNER),
            "--output-root",
            str(output),
            "--target",
            "paper_funnel",
            "--gaussian-constants",
            str(CONSTANTS),
            "--training-size",
            "64",
            "--selection-size",
            "32",
            "--audit-size",
            "32",
            "--calibration-size",
            "32",
            "--initial-size",
            "4",
        ),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((output / "replay_manifest.json").read_text())
    assert manifest["schema"] == "bayesfilter.neutra.paper_d100_exact_replay.v1"
    assert manifest["cpu_only"] is True
    assert manifest["cuda_visible_devices"] == "-1"
    assert manifest["optimizer_update_performed"] is False
    assert manifest["sample_wise_loop_or_scalar_fallback"] is False
    assert manifest["partitions_disjoint_by_stateless_seed"] is True
    assert len({tuple(value) for value in manifest["seeds"].values()}) == 5
    for name, receipt in manifest["receipts"].items():
        path = output / receipt["path"]
        assert _sha256(path) == receipt["sha256"]
        tensor = tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64)
        assert tensor.shape == tuple(receipt["shape"])
        assert int(tensor.shape[1]) == 100


def test_runner_source_preserves_cpu_and_no_optimizer_contract() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"' in source
    assert "sample_paper_d100_exact" in source
    assert "WeightedForwardKLNeuTraTrainer" not in source
    assert "MatchedReverseKLNeuTraTrainer" not in source
    assert "train_step(" not in source

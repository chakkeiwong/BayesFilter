#!/usr/bin/env python3
"""Run seed-B's tuned terminal NeuTra transport through sequential HMC."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_chart_a_l10_sequential_hmc_2026_08_04.py"
)
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-seed-b-terminal-neutra-validation-plan-2026-08-07.md"
)
TUNING_ARTIFACT = ROOT / (
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/"
    "r1/tuning/merged-tuning-result.json"
)
DEFAULT_OUTPUT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/"
    "r1/sequential"
)
DEFAULT_PREFLIGHT_OUTPUT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/"
    "r1/sequential-preflight"
)
SCHEMA = "bayesfilter.ssl_lstm.q20_seed_b_terminal_sequential_hmc.v1"
TARGET_SCOPE = (
    "ssl_lstm_neutra_state_complexity_batch_native:q20:"
    "fixed_hmc_api:seed-b-terminal-step-6250:claim_tuning_grid6"
)
ROOT_SEED = (20260807, 41001)
DEFAULT_CAP_SECONDS = 86_400.0


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("q20_chart_a_sequential_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the q=20 sequential harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()
base.SCRIPT = SCRIPT
base.PLAN = PLAN
base.TUNING_ARTIFACT = TUNING_ARTIFACT
base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
base.DEFAULT_PREFLIGHT_OUTPUT = DEFAULT_PREFLIGHT_OUTPUT
base.SCHEMA = SCHEMA
base.TARGET_SCOPE = TARGET_SCOPE
base.ROOT_SEED = ROOT_SEED
base.DEFAULT_CAP_SECONDS = DEFAULT_CAP_SECONDS


def load_frozen_kernel(path: Path = TUNING_ARTIFACT) -> Mapping[str, Any]:
    if not path.exists():
        raise base.CampaignError("seed-B tuning artifact does not exist")
    payload = base._read_json(path)
    if payload.get("passed") is not True or payload.get("final_status") != "passed":
        raise base.CampaignError("seed-B tuning did not nominate a kernel")
    kernel = payload.get("final_kernel_payload")
    if not isinstance(kernel, Mapping):
        raise base.CampaignError("seed-B tuning artifact lacks its final kernel")
    leapfrog = int(kernel.get("num_leapfrog_steps", -1))
    if leapfrog < 2:
        raise base.CampaignError("L=1 is forbidden for sequential q=20 HMC")
    if not math.isfinite(float(kernel.get("step_size", math.nan))) or float(
        kernel["step_size"]
    ) <= 0.0:
        raise base.CampaignError("seed-B tuned step size is invalid")
    if kernel.get("mass_policy") != "fixed_identity_z":
        raise base.CampaignError("seed-B tuned kernel does not use fixed identity-z mass")
    if kernel.get("use_xla") is not True:
        raise base.CampaignError("seed-B tuned kernel is not XLA enabled")
    kernel_hash = base._stable_hash(kernel)
    recorded_hash = payload.get("final_kernel_hash")
    if recorded_hash is not None and str(recorded_hash) != kernel_hash:
        raise base.CampaignError("seed-B tuned kernel hash mismatch")
    base.EXPECTED_KERNEL_HASH = kernel_hash
    return dict(kernel)


def _build_adapter(*, threads: int) -> tuple[Any, Mapping[str, Any], Mapping[str, Any]]:
    from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter
    from ssl_lstm_q20_neutra_seed_b_terminal import build_seed_b_terminal

    bridge, transport, provenance = build_seed_b_terminal(
        threads=threads,
        evidence_path=PLAN.as_posix(),
        target_scope_suffix="fixed_hmc_api:seed-b-terminal-step-6250",
    )
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=transport,
        target_scope=TARGET_SCOPE,
        runtime_backend="ssl_lstm_q20_seed_b_terminal_sequential_cpu_xla",
        evidence_path=PLAN.as_posix(),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
        nonclaims=(
            "CPU/XLA validation exception to GPU training default",
            "sequential sampler screen is not posterior correctness",
        ),
    )
    kernel = load_frozen_kernel()
    bindings = {
        "base_adapter_signature": bridge.adapter_signature(),
        "transformed_adapter_signature": adapter.adapter_signature(),
        "fixed_transport_manifest_hash": adapter.transport_manifest_hash,
    }
    for name, value in bindings.items():
        if str(kernel.get(name)) != str(value):
            raise base.CampaignError(f"seed-B tuned kernel {name} binding mismatch")
    return adapter, {**dict(provenance), **bindings}, kernel


base.load_frozen_kernel = load_frozen_kernel
base._build_adapter = _build_adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run", "worker"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cap-seconds", type=float, default=DEFAULT_CAP_SECONDS)
    parser.add_argument("--chain-index", type=int)
    args = parser.parse_args()
    if args.mode == "worker" and args.chain_index is None:
        parser.error("--chain-index is required for worker mode")
    if args.mode != "worker" and args.chain_index is not None:
        parser.error("--chain-index is only valid for worker mode")
    if not 0.0 < args.cap_seconds <= DEFAULT_CAP_SECONDS:
        parser.error(f"--cap-seconds must be in (0,{DEFAULT_CAP_SECONDS:g}]")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "worker":
        return base._run_worker(args)
    if args.mode == "preflight":
        return base._run_preflight(args)
    import bayesfilter.inference.neutra_hmc as controller

    original = controller.run_sequential_neutra_hmc

    def seed_b_archive(*values: Any, **keywords: Any) -> Any:
        keywords["archive_label"] = "seed-b-terminal"
        return original(*values, **keywords)

    controller.run_sequential_neutra_hmc = seed_b_archive
    try:
        return base._run_campaign(args)
    finally:
        controller.run_sequential_neutra_hmc = original


if __name__ == "__main__":
    raise SystemExit(main())

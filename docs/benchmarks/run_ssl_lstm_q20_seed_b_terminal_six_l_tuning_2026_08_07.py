#!/usr/bin/env python3
"""Tune a fixed-HMC kernel for the clean q=20 seed-B terminal transport."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_chart_a_six_l_fixed_hmc_tuning_2026_08_03.py"
)
SEED_B_LOADER = ROOT / (
    "docs/benchmarks/ssl_lstm_q20_neutra_seed_b_terminal.py"
)
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-seed-b-terminal-neutra-validation-plan-2026-08-07.md"
)
SCHEMA = "bayesfilter.ssl_lstm.q20_seed_b_terminal_six_l_hmc_tuning.v1"
TUNE_SEED_BASE = (20260807, 10100)
SCREEN_SEED_BASE = (20260807, 20100)
VERIFICATION_SEED_BASE = (20260807, 30100)
TARGET_SCOPE = (
    "ssl_lstm_neutra_state_complexity_batch_native:q20:"
    "fixed_hmc_api:seed-b-terminal-step-6250:claim_tuning_grid6"
)


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("q20_chart_a_tuning_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the q=20 tuning harness")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base()
base.SCRIPT = SCRIPT
base.BASE_HARNESS = SEED_B_LOADER
base.PLAN = PLAN
base.SCHEMA = SCHEMA
base.TUNE_SEED_BASE = TUNE_SEED_BASE
base.SCREEN_SEED_BASE = SCREEN_SEED_BASE
base.VERIFICATION_SEED_BASE = VERIFICATION_SEED_BASE


def _load_terminal_harness() -> Any:
    import ssl_lstm_q20_neutra_seed_b_terminal as terminal

    return SimpleNamespace(
        _configure_tensorflow=terminal.configure_cpu_tensorflow,
        _build_chart=lambda _label, *, threads: terminal.build_seed_b_terminal(
            threads=threads,
            evidence_path=PLAN.as_posix(),
            target_scope_suffix="fixed_hmc_api:seed-b-terminal-step-6250",
        ),
    )


def _shifted_seed(seed: tuple[int, int], offset: int) -> tuple[int, int]:
    return seed[0], seed[1] + int(offset)


def _config_for_worker(*, leapfrog: int, candidate_index: int) -> Any:
    from bayesfilter.inference.fixed_transport_hmc_tuning import (
        FixedTransportHMCKernelTuningConfig,
    )

    if int(leapfrog) < 2:
        raise ValueError("L=1 is forbidden for q=20 HMC tuning")
    return FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.5,
        leapfrog_grid=(int(leapfrog),),
        chain_count=4,
        target_accept_prob=0.70,
        acceptance_band=(0.65, 0.75),
        repair_band=(0.55, 0.85),
        budget_schedule=(8, 16, 32),
        tune_num_results=8,
        screen_num_results=16,
        screen_num_burnin_steps=4,
        verification_num_results=64,
        verification_num_burnin_steps=16,
        tune_seed_base=_shifted_seed(TUNE_SEED_BASE, candidate_index * 100),
        screen_seed_base=_shifted_seed(SCREEN_SEED_BASE, candidate_index * 100),
        verification_seed_base=_shifted_seed(
            VERIFICATION_SEED_BASE, candidate_index
        ),
        target_status_trace_policy="per_chain_step",
        target_scope=TARGET_SCOPE,
        use_xla=True,
        output_filename="tuning-result.json",
    )


base._load_base_harness = _load_terminal_harness
base._config_for_worker = _config_for_worker


def _replace_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".seed-b.tmp")
    temporary.write_bytes(base._canonical_bytes(payload))
    temporary.replace(path)


def _normalize_worker_summary(output_root: Path) -> None:
    path = ROOT / output_root / "summary.json"
    payload = base._read_json(path)
    payload.update(
        {
            "schema": SCHEMA,
            "chart": "seed-b-terminal",
            "plan": PLAN.as_posix(),
            "source_harness": SCRIPT.relative_to(ROOT).as_posix(),
            "source_harness_sha256": base._sha256(SCRIPT),
            "nonclaims": [
                "seed-B terminal fixed-HMC kernel candidate tuning only",
                "no sequential HMC, convergence, posterior, or default claim",
            ],
        }
    )
    _replace_json(path, payload)


def _normalize_supervisor_summary(output_root: Path) -> None:
    path = ROOT / output_root / "summary.json"
    payload = base._read_json(path)
    payload.update(
        {
            "schema": SCHEMA,
            "chart": "seed-b-terminal",
            "plan": PLAN.as_posix(),
            "source_harness": SCRIPT.relative_to(ROOT).as_posix(),
            "source_harness_sha256": base._sha256(SCRIPT),
            "nonclaims": [
                "seed-B terminal fixed-HMC kernel candidate tuning only",
                "no sequential HMC, convergence, posterior, or default claim",
            ],
        }
    )
    _replace_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("preflight", "worker", "supervisor"), required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--leapfrog", type=int)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cap-seconds", type=float, default=43_200.0)
    args = parser.parse_args()
    if args.mode == "worker":
        if args.leapfrog is None or args.candidate_index is None or args.threads is None:
            parser.error("worker mode requires leapfrog, candidate-index, and threads")
    elif any(
        value is not None
        for value in (args.leapfrog, args.candidate_index, args.threads)
    ):
        parser.error("only worker mode accepts worker arguments")
    if not 0.0 < args.cap_seconds <= 43_200.0:
        parser.error("--cap-seconds must be in (0,43200]")
    return args


def main() -> int:
    args = parse_args()
    base._validate_static_contract()
    if args.mode == "preflight":
        from ssl_lstm_q20_neutra_seed_b_terminal import (
            binding_payload,
            build_seed_b_terminal,
        )

        bridge, transport, provenance = build_seed_b_terminal(
            threads=1,
            evidence_path=PLAN.as_posix(),
            target_scope_suffix="fixed_hmc_api:seed-b-terminal-step-6250",
        )
        import tensorflow as tf

        values, scores, status = bridge.log_prob_and_grad_status(
            tf.zeros((4, 4), tf.float64)
        )
        status_code = tf.convert_to_tensor(status["status_code"], tf.int32)
        valid = tf.convert_to_tensor(
            status["valid_pre_regularized_score"], tf.bool
        )
        target_passed = bool(
            tf.reduce_all(tf.math.is_finite(values)).numpy()
            and tf.reduce_all(tf.math.is_finite(scores)).numpy()
            and tf.reduce_all(tf.equal(status_code, 0)).numpy()
            and tf.reduce_all(valid).numpy()
        )
        if not target_passed:
            raise RuntimeError("seed-B terminal target/status preflight failed")

        print(
            json.dumps(
                {
                    "status": "PREFLIGHT_PASS",
                    "total_worker_cores": 58,
                    "binding": binding_payload(),
                    "target_scope": TARGET_SCOPE,
                    "target_status_passed": target_passed,
                    "transport_type": type(transport).__name__,
                    "provenance": provenance,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.mode == "worker":
        code = base._run_worker(args)
        _normalize_worker_summary(args.output_root)
        return code
    code = base._run_supervisor(args)
    _normalize_supervisor_summary(args.output_root)
    return code


if __name__ == "__main__":
    raise SystemExit(main())

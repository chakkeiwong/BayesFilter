"""CPU-only QR HMC smoke artifact for BayesFilter v1 readiness."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf  # noqa: E402
import tensorflow_probability as tfp  # noqa: E402

from bayesfilter.testing import QRStaticLGSSMTarget, run_qr_static_lgssm_hmc_smoke  # noqa: E402


@dataclass(frozen=True)
class HMCConfig:
    num_results: int
    num_burnin_steps: int
    step_size: float
    num_leapfrog_steps: int
    seed: tuple[int, int]


def _tensor_to_json(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        array = value.numpy()
        if array.shape == ():
            item = array.item()
            if isinstance(item, (bool, int, float, str)):
                return item
            return float(item)
        return array.tolist()
    return value


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_devices": [
            {"name": device.name, "device_type": device.device_type}
            for device in tf.config.list_logical_devices()
        ],
    }


def _markdown(payload: dict[str, Any], json_path: Path | None) -> str:
    result = payload["result"]
    json_name = str(json_path) if json_path is not None else "stdout"
    return "\n".join(
        [
            "# BayesFilter v1 QR HMC Smoke",
            "",
            "Purpose: record a narrow CPU-only TFP HMC smoke for "
            "`linear_qr_score_hessian_static_lgssm` using QR value autodiff "
            "for the sampler gradient.",
            "",
            "## Claim Scope",
            "",
            "This artifact is a first target-specific HMC smoke.  It is not a "
            "general HMC readiness claim for all BayesFilter filters.",
            "",
            "## Result",
            "",
            f"The JSON file is authoritative: `{json_name}`.",
            "",
            "| Status | Acceptance | Nonfinite Samples | Initial Grad Finite |",
            "| --- | ---: | ---: | --- |",
            "| {status} | {acceptance:.4f} | {nonfinite} | {grad_finite} |".format(
                status=payload["status"],
                acceptance=float(result["acceptance_rate"]),
                nonfinite=int(result["nonfinite_sample_count"]),
                grad_finite=result["initial_gradient_finite"],
            ),
            "",
            "## Interpretation",
            "",
            "Passing means this one small QR target can run a fixed-seed HMC "
            "smoke with finite samples and diagnostics.  Broader posterior "
            "recovery, tuning, GPU, DSGE, MacroFinance switch-over, and SVD-CUT "
            "HMC remain separate gates.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-results", type=int, default=12)
    parser.add_argument("--num-burnin-steps", type=int, default=6)
    parser.add_argument("--step-size", type=float, default=0.03)
    parser.add_argument("--num-leapfrog-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, nargs=2, default=(20260511, 17))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args()

    config = HMCConfig(
        num_results=args.num_results,
        num_burnin_steps=args.num_burnin_steps,
        step_size=args.step_size,
        num_leapfrog_steps=args.num_leapfrog_steps,
        seed=(int(args.seed[0]), int(args.seed[1])),
    )
    start = time.perf_counter()
    diagnostics = run_qr_static_lgssm_hmc_smoke(
        num_results=config.num_results,
        num_burnin_steps=config.num_burnin_steps,
        step_size=config.step_size,
        num_leapfrog_steps=config.num_leapfrog_steps,
        seed=config.seed,
    )
    elapsed = time.perf_counter() - start
    target = QRStaticLGSSMTarget.default()
    result = {key: _tensor_to_json(value) for key, value in diagnostics.items()}
    nonfinite = int(result["nonfinite_sample_count"])
    acceptance = float(result["acceptance_rate"])
    status = "ok" if nonfinite == 0 and 0.1 <= acceptance <= 1.0 else "blocked"
    payload = {
        "benchmark": "bayesfilter_v1_qr_hmc_smoke",
        "claim_scope": "target_specific_smoke_only",
        "target": "linear_qr_score_hessian_static_lgssm",
        "gradient_path": "qr_value_autodiff_score",
        "config": asdict(config),
        "environment": _environment(),
        "initial_parameters": _tensor_to_json(target.initial_parameters),
        "runtime_seconds": elapsed,
        "status": status,
        "result": result,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.write_text(_markdown(payload, args.output) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

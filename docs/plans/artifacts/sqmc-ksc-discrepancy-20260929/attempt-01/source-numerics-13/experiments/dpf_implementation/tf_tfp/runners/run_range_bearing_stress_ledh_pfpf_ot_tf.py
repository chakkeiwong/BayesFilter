"""Run bounded range-bearing stress diagnostics for LEDH-PF-PF-OT."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import json
import sys
from pathlib import Path

from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    stable_digest,
    utc_now,
    wall_time_call,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.run_range_bearing_ledh_pfpf_ot_tf import (
    _run as run_range_bearing,
)


JSON_PATH = OUTPUT_DIR / "dpf_nonlinear_ssm_range_bearing_stress_2026-05-29.json"
REPORT_PATH = REPORT_DIR / "dpf-nonlinear-ssm-range-bearing-stress-result-2026-05-29.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--check-reproducibility", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate(_load_json_or_fail(JSON_PATH))
        return 0
    payload, runtime = wall_time_call(_run)
    payload["run_manifest"]["wall_time_seconds"] = runtime
    payload["reproducibility_digest"] = _digest(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    if args.check_reproducibility:
        second = _run()
        second["run_manifest"]["wall_time_seconds"] = payload["run_manifest"]["wall_time_seconds"]
        second["reproducibility_digest"] = _digest(second)
        if second["reproducibility_digest"] != payload["reproducibility_digest"]:
            raise RuntimeError("range-bearing stress reproducibility digest mismatch")
    _validate(payload)
    print(payload["decision"])
    return 0


def _run() -> dict:
    source = run_range_bearing()
    metrics = source["primary_metrics"]
    return {
        "decision": "DPF_NONLINEAR_SSM_RANGE_BEARING_STRESS_PASSED",
        "question": "range-bearing nonlinear local-linearization stress for LEDH-PF-PF-OT",
        "created_at_utc": utc_now(),
        "source_decision": source["decision"],
        "source_reproducibility_digest": _digest_source(source),
        "backend": "tensorflow_tensorflow_probability",
        "model_definition": source["model_definition"],
        "reference": source["reference"],
        "seed_list": source["seed_list"],
        "num_particles": source["num_particles"],
        "primary_metrics": metrics,
        "run_manifest": environment_manifest(
            command="CUDA_VISIBLE_DEVICES=-1 python -m experiments.dpf_implementation.tf_tfp.runners.run_range_bearing_stress_ledh_pfpf_ot_tf",
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _validate(payload: dict) -> None:
    if payload["decision"] != "DPF_NONLINEAR_SSM_RANGE_BEARING_STRESS_PASSED":
        raise RuntimeError(payload["decision"])
    if payload["source_decision"] != "DPF_LEDH_PFPF_OT_TF_TFP_RANGE_BEARING_PASSED":
        raise RuntimeError("source range-bearing runner did not pass")
    if "UKF is approximate" not in payload["reference"]["reference_status"]:
        raise RuntimeError("missing UKF caveat")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only pre-import manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing reproducibility digest")


def _markdown(payload: dict) -> str:
    metrics = payload["primary_metrics"]
    return f"""# DPF Nonlinear-SSM Range-Bearing Stress Result

## Decision

`{payload['decision']}`

| Check | Status | Evidence |
| --- | --- | --- |
| source nonlinear runner | pass | `{payload['source_decision']}` |
| median LEDH state RMSE to UKF | proxy | `{metrics['median_ledh_state_rmse_to_ukf']:.6f}` |
| max LEDH Sinkhorn residual | veto | `{metrics['max_ledh_sinkhorn_residual']:.3e}` |
| min LEDH Jacobian singular value | veto | `{metrics['min_ledh_jacobian_singular_value']:.3e}` |

UKF is approximate, not ground truth; proxy RMSE is diagnostic only.
"""


def _non_implications() -> list[str]:
    return [
        "No UKF ground truth is concluded.",
        "No posterior correctness is concluded.",
        "No production readiness is concluded.",
        "No DSGE or NAWM validation is concluded.",
    ]


def _digest(payload: dict) -> str:
    stable = {k: v for k, v in payload.items() if k not in {"created_at_utc", "run_manifest", "reproducibility_digest"}}
    return stable_digest(stable)


def _digest_source(payload: dict) -> str:
    stable = {k: v for k, v in payload.items() if k not in {"created_at_utc", "run_manifest", "reproducibility_digest"}}
    return stable_digest(stable)


def _load_json_or_fail(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

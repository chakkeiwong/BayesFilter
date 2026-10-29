"""Bounded nonlinear evidence ladder for annealed transport default."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import time
from typing import Any

from experiments.dpf_implementation.tf_tfp.runners import (
    run_range_bearing_stress_ledh_pfpf_ot_tf as range_bearing,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_structural_interface_nonlinear_ar1_tf as structural,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_sv_cut4_ledh_gradient_mle_tf as sv,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-annealed-transport-reference-alignment-plan-2026-05-31.md"
JSON_PATH = OUTPUT_DIR / "dpf_nonlinear_ladder_annealed_transport_2026-05-31.json"
REPORT_PATH = REPORT_DIR / "dpf-nonlinear-ladder-annealed-transport-2026-05-31.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        _validate_payload(load_json(JSON_PATH))
        return 0
    start = time.perf_counter()
    payload = _run()
    payload["run_manifest"]["wall_time_seconds"] = time.perf_counter() - start
    payload["reproducibility_digest"] = _digest_payload(payload)
    write_json(JSON_PATH, payload)
    write_text(REPORT_PATH, _markdown(payload))
    _validate_payload(payload)
    print(payload["decision"])
    return 0


def _run() -> dict[str, Any]:
    range_payload = range_bearing._run()
    sv_payload = sv._run()
    structural_payload = structural._run()
    rows = [
        _row("range_bearing", range_payload["decision"], "UKF approximate diagnostic; range-bearing stress"),
        _row("stochastic_volatility", sv_payload["decision"], "CUT4 differentiable comparator; scalar/gradient/MLE smoke"),
        _row("structural_ar1", structural_payload["decision"], "CUT4 comparator and deterministic residual contract"),
    ]
    decision = "nonlinear_ladder_annealed_transport_executed_with_caveats"
    if any("STRUCTURED_BLOCKER" in row["source_decision"] for row in rows):
        decision = "nonlinear_ladder_annealed_transport_structured_blocker_recorded"
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "Bounded nonlinear evidence ladder after annealed transport default wiring.",
        "transport_default_policy": "annealed_transport_default_for_new experimental DPF paths; legacy source runners may retain recorded fixed-target smoke caveats",
        "rows": rows,
        "range_bearing": range_payload,
        "stochastic_volatility": sv_payload,
        "structural_ar1": structural_payload,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_nonlinear_ladder_annealed_transport_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _row(model_id: str, source_decision: str, interpretation: str) -> dict[str, str]:
    return {
        "model_id": model_id,
        "source_decision": source_decision,
        "interpretation": interpretation,
        "promotion_status": "bounded_diagnostic_not_general_validity",
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] not in {
        "nonlinear_ladder_annealed_transport_executed_with_caveats",
        "nonlinear_ladder_annealed_transport_structured_blocker_recorded",
    }:
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")
    if not payload["rows"]:
        raise RuntimeError("missing nonlinear ladder rows")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Nonlinear Ladder Annealed Transport

## Decision

`{payload['decision']}`

## Rows

{_rows_table(payload['rows'])}

## Non-Implications

{_bullets(payload['non_implications'])}
"""


def _rows_table(rows: list[dict[str, str]]) -> str:
    lines = ["| Model | Source decision | Promotion status | Interpretation |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| `{row['model_id']}` | `{row['source_decision']}` | `{row['promotion_status']}` | {row['interpretation']} |"
        )
    return "\n".join(lines)


def _non_implications() -> list[str]:
    return [
        "No general nonlinear-SSM validity is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No production readiness is concluded.",
        "No DSGE or NAWM validation is concluded.",
    ]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    return stable_digest(comparable)


if __name__ == "__main__":
    raise SystemExit(main())

"""Matched LGSSM LEDH-PF-PF with annealed transport diagnostics."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import time
from typing import Any

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_matched_ledh_pfpf_ot_tf as ledh_matched,
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
JSON_PATH = OUTPUT_DIR / "dpf_ledh_pfpf_annealed_transport_lgssm_2026-05-31.json"
REPORT_PATH = REPORT_DIR / "dpf-ledh-pfpf-annealed-transport-lgssm-2026-05-31.md"


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
    source = ledh_matched._run()
    rows = source["rows"]
    summary = _summary(rows, source["summary"])
    decision = "ledh_pfpf_annealed_transport_lgssm_finite_diagnostics"
    if summary["nonfinite_row_count"] > 0:
        decision = "ledh_pfpf_annealed_transport_lgssm_nonfinite_veto"
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "LEDH-PF-PF with reusable annealed transport on matched filterflow LGSSM protocol.",
        "source_decision": source["decision"],
        "settings": {
            **source["settings"],
            "transport_method": "annealed_transport",
            "fixed_target_sinkhorn_status": "local comparator only",
        },
        "rows": rows,
        "summary": summary,
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_ledh_pfpf_annealed_transport_lgssm_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _summary(rows: list[dict[str, Any]], source_summary: dict[str, Any]) -> dict[str, Any]:
    annealed_rows = [
        row for row in rows if row.get("row_status") == "executed" and row.get("finite")
    ]
    return {
        **source_summary,
        "annealed_transport_row_count": len(annealed_rows),
        "transport_method": "filterflow_style_annealed_transport_tf",
        "pfpf_correction_status": "recorded_in_rows",
        "fixed_target_sinkhorn_status": "not_default_local_comparator_only",
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] != "ledh_pfpf_annealed_transport_lgssm_finite_diagnostics":
        raise RuntimeError(payload["decision"])
    if payload["summary"]["nonfinite_row_count"] != 0:
        raise RuntimeError("non-finite LEDH annealed rows")
    if payload["settings"]["transition_covariance"] != "I_2 executable filterflow convention":
        raise RuntimeError("wrong covariance convention")
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# LEDH-PF-PF Annealed Transport LGSSM

## Decision

`{payload['decision']}`

## Summary

{_key_value_table(payload['summary'])}

## Non-Implications

{_bullets(payload['non_implications'])}
"""


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No claim that LEDH must equal filterflow RegularisedTransform is concluded.",
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

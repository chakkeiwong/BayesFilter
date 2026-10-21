"""Run the reviewed filterflow OT-DPF gap-closure program."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
PRE_IMPORT_CUDA_VISIBLE_DEVICES = os.environ["CUDA_VISIBLE_DEVICES"]

import argparse
import subprocess
import time
from typing import Any

from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_full_comparison_tf as full_comparison,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_matched_ledh_pfpf_ot_tf as matched_ledh,
)
from experiments.dpf_implementation.tf_tfp.runners import (
    run_filterflow_smoothness_gradient_audit_tf as gradient_audit,
)
from experiments.dpf_implementation.tf_tfp.runners.common_tf import (
    OUTPUT_DIR,
    REPORT_DIR,
    REPO_ROOT,
    environment_manifest,
    load_json,
    stable_digest,
    utc_now,
    write_json,
    write_text,
)
from experiments.dpf_implementation.tf_tfp.runners.filterflow_reference_policy import (
    FILTERFLOW_REFERENCE_BRANCH,
    FILTERFLOW_REFERENCE_COMMIT,
    FILTERFLOW_UPSTREAM_BASE_COMMIT,
    validate_filterflow_reference_status,
)


PLAN_PATH = "docs/plans/bayesfilter-dpf-filterflow-gap-closure-program-plan-2026-05-31.md"
JSON_PATH = OUTPUT_DIR / "dpf_filterflow_gap_closure_program_2026-05-31.json"
REPORT_PATH = REPORT_DIR / "dpf-filterflow-gap-closure-program-2026-05-31.md"
FILTERFLOW_PATH = REPO_ROOT / ".localsource" / "filterflow"
FILTERFLOW_ENV_PYTHON = REPO_ROOT / ".localenv" / "filterflow-py311" / "bin" / "python"
FILTERFLOW_BRANCH = FILTERFLOW_REFERENCE_BRANCH
FILTERFLOW_COMMIT = FILTERFLOW_REFERENCE_COMMIT
FILTERFLOW_UPSTREAM_BASE = FILTERFLOW_UPSTREAM_BASE_COMMIT


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
    full_payload = full_comparison._run()
    ledh_payload = matched_ledh._run()
    gradient_payload = gradient_audit._run()
    env_freeze = _filterflow_environment_freeze()
    covariance_ledger = _covariance_ambiguity_ledger()
    gap_status = _gap_status(full_payload, ledh_payload, gradient_payload, env_freeze)
    decision = _decision(gap_status)
    return {
        "decision": decision,
        "created_at_utc": utc_now(),
        "plan_path": PLAN_PATH,
        "question": "Close or narrow six reviewed filterflow comparison gaps.",
        "gap_status": gap_status,
        "full_comparison": {
            "decision": full_payload["decision"],
            "method_status_summary": full_payload["method_status_summary"],
            "comparison_matrix": full_payload["comparison_matrix"],
            "discrepancy_ledger": full_payload["discrepancy_ledger"],
        },
        "matched_ledh": {
            "decision": ledh_payload["decision"],
            "summary": ledh_payload["summary"],
            "settings": ledh_payload["settings"],
        },
        "smoothness_gradient_audit": {
            "decision": gradient_payload["decision"],
            "scalar_normalization_ledger": gradient_payload["scalar_normalization_ledger"],
            "gradient_claim_status": gradient_payload["gradient_claim_status"],
            "bayesfilter_gradient_status": gradient_payload["bayesfilter_gradient_status"],
        },
        "filterflow_environment_freeze": env_freeze,
        "covariance_ambiguity_ledger": covariance_ledger,
        "claude_review": {
            "status": "prior_gap_closure_review_reconciled_no_pending_placeholder",
            "protocol": "claude -p --model claude-opus-4-7 --effort max",
            "max_iterations": 5,
            "reconciled_by": "annealed transport reference-alignment result review round 5",
        },
        "run_manifest": environment_manifest(
            command=(
                "CUDA_VISIBLE_DEVICES=-1 python -m "
                "experiments.dpf_implementation.tf_tfp.runners.run_filterflow_gap_closure_program_tf"
            ),
            pre_import_cuda_visible_devices=PRE_IMPORT_CUDA_VISIBLE_DEVICES,
        ),
        "non_implications": _non_implications(),
    }


def _gap_status(
    full_payload: dict[str, Any],
    ledh_payload: dict[str, Any],
    gradient_payload: dict[str, Any],
    env_freeze: dict[str, Any],
) -> list[dict[str, Any]]:
    fixed_status = full_payload["method_status_summary"]["bayesfilter_fixed_target_sinkhorn_status"]
    filterflow_style_status = full_payload["method_status_summary"][
        "bayesfilter_filterflow_style_transport_status"
    ]
    return [
        {
            "id": "fixed_target_sinkhorn_ess_gating",
            "status": (
                "closed_nontriggered_veto_removed"
                if fixed_status == "within_filterflow_mc_band"
                else fixed_status
            ),
            "detail": (
                "fixed-target Sinkhorn is now computed only on ESS-triggered rows in the "
                "matched audit path; unconditional residual ladder remains a component diagnostic"
            ),
        },
        {
            "id": "filterflow_style_annealed_transport_reference",
            "status": filterflow_style_status,
            "detail": "paper-style annealed transport mirror preserved",
        },
        {
            "id": "matched_ledh_pfpf_ot",
            "status": ledh_payload["decision"],
            "detail": ledh_payload["summary"]["interpretation"],
        },
        {
            "id": "gradient_smoothness_audit",
            "status": gradient_payload["decision"],
            "detail": gradient_payload["gradient_claim_status"],
        },
        {
            "id": "filterflow_environment_freeze",
            "status": env_freeze["status"],
            "detail": env_freeze["smoke_status"],
        },
        {
            "id": "covariance_ambiguity_ledger",
            "status": "recorded_permanent_note",
            "detail": "paper/supplement 0.5 I_2 versus executable I_2 recorded",
        },
    ]


def _decision(gap_status: list[dict[str, Any]]) -> str:
    status_by_id = {row["id"]: row["status"] for row in gap_status}
    required = {
        "fixed_target_sinkhorn_ess_gating": "closed_nontriggered_veto_removed",
        "filterflow_style_annealed_transport_reference": "within_filterflow_mc_band",
        "matched_ledh_pfpf_ot": "matched_ledh_pfpf_ot_finite_diagnostics",
        "gradient_smoothness_audit": "smoothness_gradient_severe_unreconciled_magnitude_risk_recorded",
        "filterflow_environment_freeze": "recorded_and_smoke_passed",
        "covariance_ambiguity_ledger": "recorded_permanent_note",
    }
    if all(status_by_id.get(key) == value for key, value in required.items()):
        return "filterflow_gap_closure_program_completed_with_severe_unreconciled_gradient_mismatch_risk"
    return "filterflow_gap_closure_program_has_blocker_or_open_gap"


def _filterflow_environment_freeze() -> dict[str, Any]:
    branch = _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _git(["git", "-C", str(FILTERFLOW_PATH), "rev-parse", "HEAD"])
    status = _git(["git", "-C", str(FILTERFLOW_PATH), "status", "--short", "--branch"])
    diff_summary = _git(["git", "-C", str(FILTERFLOW_PATH), "diff", "--stat"]) or "clean"
    smoke = _filterflow_smoke()
    freeze_status = "recorded_and_smoke_passed" if smoke["status"] == "pass" else "recorded_smoke_failed"
    return {
        "status": freeze_status,
        "branch": branch,
        "commit": commit,
        "upstream_base": FILTERFLOW_UPSTREAM_BASE,
        "source_status": status,
        "diff_summary": diff_summary,
        "python": smoke.get("python"),
        "tensorflow": smoke.get("tensorflow"),
        "numpy": smoke.get("numpy"),
        "command": smoke.get("command"),
        "smoke_status": smoke["status"],
        "stderr_excerpt": smoke.get("stderr_excerpt", ""),
    }


def _filterflow_smoke() -> dict[str, Any]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["MPLCONFIGDIR"] = str(REPO_ROOT / ".cache" / "filterflow-mpl")
    env["PYTHONPATH"] = str(FILTERFLOW_PATH)
    script = (
        "import json, tensorflow as tf\n"
        "np = __import__('numpy')\n"
        "from filterflow.resampling import RegularisedTransform\n"
        "payload={'python': __import__('sys').version.split()[0], "
        "'tensorflow': tf.__version__, 'numpy': np.__version__, "
        "'regularized_transform': RegularisedTransform.__name__}\n"
        "print('FILTERFLOW_SMOKE_JSON_BEGIN')\n"
        "print(json.dumps(payload, sort_keys=True))\n"
        "print('FILTERFLOW_SMOKE_JSON_END')\n"
    )
    completed = subprocess.run(
        [str(FILTERFLOW_ENV_PYTHON), "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    command = (
        "CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=.cache/filterflow-mpl "
        "PYTHONPATH=.localsource/filterflow "
        f"{FILTERFLOW_ENV_PYTHON} -c <filterflow smoke>"
    )
    if completed.returncode != 0:
        return {
            "status": "fail",
            "command": command,
            "stdout_excerpt": completed.stdout[-1000:],
            "stderr_excerpt": completed.stderr[-1000:],
        }
    start = completed.stdout.rfind("FILTERFLOW_SMOKE_JSON_BEGIN")
    end = completed.stdout.rfind("FILTERFLOW_SMOKE_JSON_END")
    if start < 0 or end < 0:
        return {"status": "fail", "command": command, "stdout_excerpt": completed.stdout[-1000:]}
    import json

    payload = json.loads(completed.stdout[start + len("FILTERFLOW_SMOKE_JSON_BEGIN"):end].strip())
    payload["status"] = "pass"
    payload["command"] = command
    payload["stderr_excerpt"] = completed.stderr[-1000:]
    return payload


def _covariance_ambiguity_ledger() -> dict[str, Any]:
    return {
        "status": "recorded_permanent_note",
        "paper_supplement_statement": "transition covariance 0.5 I_2",
        "executable_filterflow_setting": "transition covariance I_2",
        "current_reproduction_policy": (
            "Use executable filterflow I_2 for code comparisons because prior bounded reruns "
            "indicate Table 1 scale matches executable I_2."
        ),
        "future_reversal_condition": "separate paper-notation audit overturns the executable-code interpretation",
    }


def _validate_payload(payload: dict[str, Any]) -> None:
    if payload["decision"] != "filterflow_gap_closure_program_completed_with_severe_unreconciled_gradient_mismatch_risk":
        raise RuntimeError(payload["decision"])
    if payload["run_manifest"]["pre_import_cuda_visible_devices"] != "-1":
        raise RuntimeError("missing CPU-only manifest")
    validate_filterflow_reference_status(payload["filterflow_environment_freeze"])
    if payload["claude_review"]["protocol"] != "claude -p --model claude-opus-4-7 --effort max":
        raise RuntimeError("wrong Claude protocol")
    if "reproducibility_digest" not in payload:
        raise RuntimeError("missing digest")


def _markdown(payload: dict[str, Any]) -> str:
    return f"""# Filterflow Gap-Closure Program

## Decision

`{payload['decision']}`

## Gap Status

{_gap_table(payload['gap_status'])}

## Filterflow Environment Freeze

{_key_value_table(payload['filterflow_environment_freeze'])}

## Covariance Ambiguity Ledger

{_key_value_table(payload['covariance_ambiguity_ledger'])}

## Matched LEDH Summary

{_key_value_table(payload['matched_ledh']['summary'])}

## Gradient Audit Summary

{_key_value_table(payload['smoothness_gradient_audit']['scalar_normalization_ledger'])}

## Non-Implications

{_non_implications_markdown()}
"""


def _gap_table(rows: list[dict[str, Any]]) -> str:
    lines = ["| Gap | Status | Detail |", "| --- | --- | --- |"]
    for row in rows:
        lines.append(f"| `{row['id']}` | `{row['status']}` | {row['detail']} |")
    return "\n".join(lines)


def _key_value_table(values: dict[str, Any]) -> str:
    lines = ["| Key | Value |", "| --- | --- |"]
    for key, value in values.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines)


def _digest_payload(payload: dict[str, Any]) -> str:
    comparable = dict(payload)
    comparable["created_at_utc"] = "TIMESTAMP"
    comparable["run_manifest"] = dict(comparable["run_manifest"])
    comparable["run_manifest"]["wall_time_seconds"] = "WALL_TIME"
    comparable["run_manifest"]["dirty_state_summary"] = "DIRTY_STATE"
    comparable["claude_review"] = dict(comparable["claude_review"])
    comparable["claude_review"]["status"] = "REVIEW_STATUS"
    return stable_digest(comparable)


def _git(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _non_implications() -> list[str]:
    return [
        "No production readiness is concluded.",
        "No public API readiness is concluded.",
        "No posterior correctness is concluded.",
        "No HMC readiness is concluded.",
        "No general nonlinear-SSM validity is concluded.",
        "No external macro-model validation is concluded.",
        "No banking/model-risk claim is concluded.",
        "No monograph claim is concluded.",
        "No claim that fixed-target Sinkhorn is filterflow-equivalent is concluded.",
        "No claim that finite gradients establish gradient correctness is concluded.",
    ]


def _non_implications_markdown() -> str:
    return "\n".join(f"- {item}" for item in _non_implications())


if __name__ == "__main__":
    raise SystemExit(main())

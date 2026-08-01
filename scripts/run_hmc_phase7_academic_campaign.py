"""Launch and supervise one versioned Phase 7 academic serious-HMC attempt."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence


sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name, _value in {
    "TF_NUM_INTRAOP_THREADS": "8",
    "TF_NUM_INTEROP_THREADS": "1",
    "OMP_NUM_THREADS": "8",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}.items():
    os.environ[_name] = _value
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter-phase7-academic")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_academic_campaign import (  # noqa: E402
    AcademicCampaignError,
    finalize_academic_attempt,
    prepare_academic_launch,
    release_academic_launch,
    write_infrastructure_failure,
)
from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DeterministicLGSSMPhase7Config,
    DeterministicLGSSMPhase7Error,
    run_phase7,
    validate_phase7_inputs,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--campaign-root", type=Path)
    return parser.parse_args(argv)


def classify_controller_result(result: dict) -> str:
    if result.get("passed") is True:
        return "strict_pass"
    classification = result.get("failure_classification")
    if classification in {
        "diagnostic_cap_failure",
        "infrastructure_failure",
        "continuation_veto",
    }:
        return str(classification)
    reason = str(result.get("reason", ""))
    if reason in {
        "burnin_diagnostics_failed_at_cap",
        "retained_diagnostics_failed_at_cap",
    }:
        return "diagnostic_cap_failure"
    return "continuation_veto"


def classify_uncaught_error(error: Exception) -> str:
    if isinstance(error, (AcademicCampaignError, DeterministicLGSSMPhase7Error)):
        return "continuation_veto"
    if isinstance(error, (TimeoutError, ValueError)):
        return "continuation_veto"
    return "infrastructure_failure"


def _append_log(path: Path, message: str) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _invocation() -> tuple[str, ...]:
    return (str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *sys.argv[1:])


def main(argv: Sequence[str] | None = None) -> int:
    invocation_started = time.monotonic()
    args = parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise AcademicCampaignError("academic launcher requires CUDA_VISIBLE_DEVICES=-1")
    config = DeterministicLGSSMPhase7Config.load(args.config)
    preflight = validate_phase7_inputs(config)
    context = prepare_academic_launch(
        config=config,
        preflight=preflight,
        command=_invocation(),
        campaign_root=args.campaign_root,
        invocation_started_monotonic=invocation_started,
    )
    _append_log(
        context.paths["log_path"],
        (
            f"campaign={context.campaign_id} attempt={context.attempt_number} "
            f"status=launching remaining_seconds={context.remaining_wall_time_seconds:.6f}"
        ),
    )
    try:
        result = dict(
            run_phase7(
                config,
                smoke=False,
                academic_launch_context=context,
            )
        )
        elapsed = time.monotonic() - invocation_started
        classification = classify_controller_result(result)
        terminal_path = context.paths["public_result_path"]
        _append_log(
            context.paths["log_path"],
            f"status=controller_terminal classification={classification} elapsed_seconds={elapsed:.6f}",
        )
        summary = finalize_academic_attempt(
            context,
            elapsed_seconds=elapsed,
            terminal_path=terminal_path,
        )
        return int(summary["exit_code"])
    except BaseException as error:
        elapsed = time.monotonic() - invocation_started
        if not isinstance(error, Exception):
            release_academic_launch(context)
            raise
        _append_log(
            context.paths["log_path"],
            f"status=uncaught_failure type={type(error).__name__} elapsed_seconds={elapsed:.6f}",
        )
        classification = classify_uncaught_error(error)
        terminal = context.paths["failure_path"]
        write_infrastructure_failure(
            context,
            stage="launcher_supervision",
            error=error,
            elapsed_seconds=elapsed,
            classification=classification,
        )
        try:
            summary = finalize_academic_attempt(
                context,
                elapsed_seconds=time.monotonic() - invocation_started,
                terminal_path=terminal,
            )
        except BaseException:
            release_academic_launch(context)
            raise
        return int(summary["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())

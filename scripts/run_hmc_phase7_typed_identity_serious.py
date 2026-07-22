"""Launch one manifest-bound serious typed-identity Phase 7 HMC run."""

from __future__ import annotations

import argparse
import os
import sys
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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter-phase7-serious")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_serious_authority import (  # noqa: E402
    AUTHORITY_PATH,
    SeriousInheritedEvidenceDriftError,
    SeriousPostClaimPreparationError,
    SecureSeriousOutputSession,
    attach_serious_output_session,
    build_serious_output_manifest,
    discard_prepared_serious_launch_context,
    expected_launcher_command,
    parse_serious_output_manifest,
    prepare_serious_launch,
    verify_serious_output_manifest,
    write_serious_infrastructure_terminal,
)
from bayesfilter.inference.hmc_smoke_authority import (  # noqa: E402
    SmokeOutputReservationError,
)
from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (  # noqa: E402
    run_phase7,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", choices=("burnin_sampling",), required=True)
    parser.add_argument("--phase7-serious-authority", type=Path, required=True)
    return parser.parse_args(argv)


def _redirect(session: SecureSeriousOutputSession) -> None:
    fd = session.fd("log_path")
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())


def _seal_log(session: SecureSeriousOutputSession) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    session.finish_binary_write("log_path")
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, sys.stdout.fileno())
        os.dup2(devnull, sys.stderr.fileno())
    finally:
        os.close(devnull)


def _supervise(context) -> int:
    session = None
    stage = "secure_output_reservation"
    try:
        try:
            session = SecureSeriousOutputSession.reserve_from_context(context)
        except SmokeOutputReservationError as error:
            session = error.session
            # Once the permanent claim exists, terminal infrastructure evidence
            # must not depend on the verifier whose failure is being recorded.
            session.consumed_evidence_session = None
            if session.has_role("infrastructure_failure_path") and session.has_role(
                "infrastructure_manifest_path"
            ):
                write_serious_infrastructure_terminal(
                    context=context,
                    session=session,
                    stage=f"{stage}:{error.role}",
                    error=error.cause,
                )
            if not isinstance(error.cause, Exception):
                raise error.cause.with_traceback(error.cause.__traceback__)
            return 2
        context = attach_serious_output_session(context, session)
        stage = "log_redirection"
        _redirect(session)
        stage = "controller_runtime"
        result = run_phase7(
            context.config,
            smoke=False,
            serious_launch_context=context,
        )
        stage = "log_sealing"
        _seal_log(session)
        stage = "output_manifest"
        manifest = build_serious_output_manifest(context=context, session=session)
        session.write_json(
            "output_manifest_path",
            manifest,
            parser=parse_serious_output_manifest,
        )
        verify_serious_output_manifest(
            session.read_json("output_manifest_path"),
            context=context,
            session=session,
        )
        return 0 if result.get("passed") is True else 1
    except BaseException as error:
        if (
            session is not None
            and session.has_role("infrastructure_failure_path")
        ):
            try:
                session.consumed_evidence_session = None
                write_serious_infrastructure_terminal(
                    context=context,
                    session=session,
                    stage=stage,
                    error=error,
                )
            except BaseException:
                pass
        if not isinstance(error, Exception):
            raise
        return 2
    finally:
        discard_prepared_serious_launch_context(context)
        if session is not None:
            session.close()
        else:
            try:
                os.close(context.claim_fd)
            except OSError:
                pass
            context.output_directories.close()
        context.consumed_evidence_session.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase7_serious_authority.resolve() != AUTHORITY_PATH:
        raise ValueError("serious authority path mismatch")
    command = expected_launcher_command(Path(sys.executable).resolve())
    observed = (str(Path(sys.executable).resolve()), *sys.argv)
    if observed != command:
        raise ValueError("serious invocation differs from reviewed command")
    try:
        context = prepare_serious_launch(
            authority_path=args.phase7_serious_authority,
            current_command=command,
        )
    except SeriousPostClaimPreparationError as error:
        context = error.context
        session = None
        try:
            session = SecureSeriousOutputSession.reserve_emergency_from_context(
                context
            )
            write_serious_infrastructure_terminal(
                context=context,
                session=session,
                stage="post_claim_preparation",
                error=error.cause,
            )
        finally:
            discard_prepared_serious_launch_context(context)
            if session is not None:
                session.close()
            else:
                try:
                    os.close(context.claim_fd)
                except OSError:
                    pass
                context.output_directories.close()
            context.consumed_evidence_session.close()
        if not isinstance(error.cause, Exception):
            raise error.cause.with_traceback(error.cause.__traceback__)
        return 2
    return _supervise(context)


if __name__ == "__main__":
    raise SystemExit(main())

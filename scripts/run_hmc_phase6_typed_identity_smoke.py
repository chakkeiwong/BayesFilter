"""Launch the one-use Phase 6 typed-identity HMC mechanics smoke."""

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
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter-phase6-smoke")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.hmc_smoke_authority import (  # noqa: E402
    AUTHORITY_PATH,
    ConsumedAttempt1EvidenceDriftError,
    PROPOSAL_MANIFEST_PATH,
    PROPOSAL_PATH,
    SecureSmokeOutputSession,
    SmokeOutputReservationError,
    attach_prepared_output_session,
    build_smoke_output_manifest,
    discard_prepared_smoke_launch_context,
    expected_launcher_command,
    parse_smoke_output_manifest,
    prepare_smoke_launch,
    write_smoke_infrastructure_terminal,
)
from bayesfilter.testing.deterministic_lgssm_hmc_phase7_tf import (  # noqa: E402
    run_phase7,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", choices=("burnin_sampling",), required=True)
    parser.add_argument("--phase7-smoke", action="store_true", required=True)
    parser.add_argument(
        "--phase7-smoke-authority",
        type=Path,
        required=True,
    )
    return parser.parse_args(argv)


def _redirect_after_claim(session: SecureSmokeOutputSession) -> None:
    descriptor = session.fd("log_path")
    os.dup2(descriptor, sys.stdout.fileno())
    os.dup2(descriptor, sys.stderr.fileno())


def _seal_log_before_manifest(session: SecureSmokeOutputSession) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    session.finish_binary_write("log_path")
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(descriptor, sys.stdout.fileno())
        os.dup2(descriptor, sys.stderr.fileno())
    finally:
        os.close(descriptor)


def _stabilize_redirected_log_best_effort(session: SecureSmokeOutputSession) -> None:
    if not session.has_role("log_path"):
        return
    reserved = session.fds["log_path"]
    try:
        reserved_identity = os.fstat(reserved)
    except OSError:
        return
    redirected_stream_fds: list[int] = []
    for stream in (sys.stdout, sys.stderr):
        try:
            stream_fd = stream.fileno()
            stream_identity = os.fstat(stream_fd)
            if (stream_identity.st_dev, stream_identity.st_ino) == (
                reserved_identity.st_dev,
                reserved_identity.st_ino,
            ):
                redirected_stream_fds.append(stream_fd)
                try:
                    stream.flush()
                except (OSError, ValueError):
                    pass
        except (OSError, ValueError):
            continue
    if redirected_stream_fds:
        try:
            descriptor = os.open(os.devnull, os.O_WRONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                for stream_fd in redirected_stream_fds:
                    try:
                        os.dup2(descriptor, stream_fd)
                    except OSError:
                        try:
                            os.close(stream_fd)
                        except OSError:
                            pass
            finally:
                os.close(descriptor)
        else:
            for stream_fd in redirected_stream_fds:
                try:
                    os.close(stream_fd)
                except OSError:
                    pass
    for _attempt in range(3):
        try:
            session.finish_binary_write("log_path")
            break
        except (OSError, RuntimeError, ValueError):
            continue


def _seal_infrastructure_best_effort(*, context, session, stage, error) -> None:
    if not session.has_role("infrastructure_failure_path") or not session.has_role(
        "infrastructure_manifest_path"
    ):
        return
    try:
        write_smoke_infrastructure_terminal(
            context=context,
            session=session,
            stage=stage,
            error=error,
        )
    except BaseException as sealing_error:
        # The consumed claim and any descriptor bytes remain the primary evidence.
        # Preserve an original control-flow cause, but do not swallow a new user
        # interrupt that arrives while sealing an ordinary failure.
        if isinstance(error, Exception) and not isinstance(sealing_error, Exception):
            raise
        return


def _supervise_after_claim(context) -> int:
    stage = "secure_output_reservation"
    session: SecureSmokeOutputSession | None = None
    infrastructure_sealing_attempted = False
    try:
        try:
            session = SecureSmokeOutputSession.reserve(
                directories=context.output_directories,
                claim_fd=context.claim_fd,
                consumed_evidence_session=context.consumed_evidence_session,
            )
        except SmokeOutputReservationError as error:
            session = error.session
            if isinstance(error.cause, ConsumedAttempt1EvidenceDriftError):
                infrastructure_sealing_attempted = True
                raise error.cause
            infrastructure_sealing_attempted = True
            _seal_infrastructure_best_effort(
                context=context,
                session=session,
                stage=f"{stage}:{error.role}",
                error=error.cause,
            )
            if not isinstance(error.cause, Exception):
                raise error.cause.with_traceback(error.cause.__traceback__)
            return 2
        context = attach_prepared_output_session(context, session)
        stage = "log_redirection"
        _redirect_after_claim(session)
        stage = "controller_runtime"
        result = run_phase7(
            context.config,
            smoke=True,
            smoke_launch_context=context,
        )
        stage = "log_sealing"
        _seal_log_before_manifest(session)
        stage = "output_manifest_construction"
        manifest = build_smoke_output_manifest(
            proposal_path=PROPOSAL_PATH,
            proposal_manifest_path=PROPOSAL_MANIFEST_PATH,
            authority_path=AUTHORITY_PATH,
            claim_path=context.paths["claim_path"],
            progress_path=context.paths["public_progress_path"],
            result_path=context.paths["public_result_path"],
            log_path=context.paths["log_path"],
            private_samples_path=context.paths["private_samples_path"],
            infrastructure_failure_path=context.paths[
                "infrastructure_failure_path"
            ],
            infrastructure_manifest_path=context.paths[
                "infrastructure_manifest_path"
            ],
            output_session=session,
            launch_context=context,
        )
        stage = "output_manifest_write"
        session.write_json(
            "output_manifest_path",
            manifest,
            parser=parse_smoke_output_manifest,
        )
        return 0 if result.get("passed") is True else 1
    except BaseException as error:
        if isinstance(error, ConsumedAttempt1EvidenceDriftError):
            infrastructure_sealing_attempted = True
        if session is not None and not infrastructure_sealing_attempted:
            infrastructure_sealing_attempted = True
            try:
                _stabilize_redirected_log_best_effort(session)
            except BaseException as stabilization_error:
                if isinstance(error, Exception) and not isinstance(
                    stabilization_error, Exception
                ):
                    raise
            _seal_infrastructure_best_effort(
                context=context,
                session=session,
                stage=stage,
                error=error,
            )
        if not isinstance(error, Exception):
            raise
        return 2
    finally:
        discard_prepared_smoke_launch_context(context)
        if session is not None:
            session.close()
        else:
            try:
                os.close(context.claim_fd)
            finally:
                context.output_directories.close()
        if context.consumed_evidence_session is not None:
            context.consumed_evidence_session.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase7_smoke_authority.resolve() != AUTHORITY_PATH:
        raise ValueError("Phase 6 smoke authority path mismatch")
    command = expected_launcher_command(Path(sys.executable).resolve())
    observed = (str(Path(sys.executable).resolve()), *sys.argv)
    if tuple(observed) != command:
        raise ValueError("Phase 6 smoke invocation differs from the reviewed command")
    context = prepare_smoke_launch(
        authority_path=args.phase7_smoke_authority,
        current_command=command,
    )
    return _supervise_after_claim(context)


if __name__ == "__main__":
    raise SystemExit(main())

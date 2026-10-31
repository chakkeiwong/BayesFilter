"""Resume an interrupted public-tuner sequence without repeating its numerical work.

The public tune_hmc_kernel dispatch owns this route. Old archived transitions
are replayed as evidence, not simulated again. Recognized source migrations
change orchestration only; numerical-kernel source drift remains forbidden.
"""
from dataclasses import replace
import json
from pathlib import Path
import shutil
import time

import tensorflow as tf

from . import hmc_tensorflow_verified as verified
from . import hmc_tensorflow_tuning as stage
from .hmc_sequential_verification import runtime_budget_guard
from .neutra_hmc import run_sequential_neutra_hmc

# Exact pre-repair implementation preserved in the October 1 development root.
_PRIOR_VERIFIED_SHA256 = "46687b1a00bce5c5c515bbc73c0b59dd8a3831748bf5d077b194917c04754f4c"
_PRIOR_RESUME_SHA256 = "9e74ac1298f9c8e4cd8f94873bfbb4f9d540110101be529e94afab8a70b7d16c"
_LEGACY_EVENT_VERIFIED_SHA256 = "e52f576eed0e84354fa1abc79303a69fd8f0ecd649170fe7f47f2e67cffdf2c1"
_PRE_REPLAY_REPAIR_RESUME_SHA256 = "59a5472c360be93e95f802c34a289b97058b9ce3aa0377a846765ab93a94947b"
_PRE_ENERGY_WARNING_VERIFIED_SHA256 = "d1e05e5cfa537e5f26f5386de545a21af654b7f40e8bc3a826b9145614e3071b"
_PRE_ENERGY_WARNING_RESUME_SHA256 = "776e8b9a7a703067aef9b53ea4363a92d85027107781d786f5a0a8ec48d0a1fc"


def validate_source_migration(saved):
    current = verified._source_hashes()
    if saved == current:
        return dict(kind="unchanged", old=saved, current=current)
    expected = dict(current)
    expected["hmc_tensorflow_verified.py"] = _PRE_ENERGY_WARNING_VERIFIED_SHA256
    expected["hmc_tensorflow_resume.py"] = _PRE_ENERGY_WARNING_RESUME_SHA256
    if saved == expected:
        return dict(kind="owner_finite_energy_warning_20261019",
            old=saved, current=current, numerical_kernel_changed=False,
            preserved_source="docs/plans/artifacts/surrogate-hmc-energy-warning-repair-20261019-01/source-before",
            plan="docs/plans/surrogate-hmc-energy-warning-repair-2026-10-19.md")
    expected = dict(current)
    expected["hmc_tensorflow_resume.py"] = _PRIOR_RESUME_SHA256
    if saved == expected:
        return dict(kind="reviewed_unfinished_warmup_resume_20261018",
                    old=saved, current=current, numerical_kernel_changed=False,
                    preserved_source="docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01/source-before",
                    plan="docs/plans/surrogate-hmc-48h-extension-2026-10-18.md")
    for prior_resume, kind in (
            (_PRIOR_RESUME_SHA256, "reviewed_healthy_legacy_event_replay_20261018"),
            (_PRE_REPLAY_REPAIR_RESUME_SHA256, "reviewed_pre_replay_repair_resume_20261018")):
        expected = dict(current)
        expected["hmc_tensorflow_verified.py"] = _LEGACY_EVENT_VERIFIED_SHA256
        expected["hmc_tensorflow_resume.py"] = prior_resume
        if saved == expected:
            return dict(
                kind=kind, old=saved, current=current, numerical_kernel_changed=False,
                preserved_source="docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01/source-before-replay-repair",
                plan="docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01/resume-repair-note.md")
    expected = dict(current)
    expected.pop("hmc_tensorflow_resume.py")
    expected["hmc_tensorflow_verified.py"] = _PRIOR_VERIFIED_SHA256
    if saved != expected:
        raise ValueError("resume numerical source drift")
    return dict(kind="reviewed_orchestration_and_count_extension_20261001",
                old=saved, current=current,
                preserved_source="docs/plans/artifacts/surrogate-hmc-m13-resume-20261001-development-01/before",
                numerical_kernel_changed=False)


def validate_resume_payload(payload, config, adapter, binding):
    if (payload.get("schema") != verified.VERIFIED_TENSORFLOW_HMC_SCHEMA
            or payload.get("artifact_authority") is not True
            or payload.get("passed") is not False or payload.get("handoff_eligible") is not False
            or payload.get("rhat_threshold") != verified.HMC_TUNING_ORDINARY_RHAT_THRESHOLD):
        raise ValueError("resume requires an interrupted non-admitted verified-tuner artifact")
    sequence = payload["sequence"]
    energy_only_stop = (sequence.get("status") == "HEALTH_VETO"
        and any(event.get("checks", {}).get("energy_valid") is False
                for event in sequence.get("events", [])))
    if (sequence.get("status") != "BUDGET_EXHAUSTED" and not energy_only_stop
            or type(sequence.get("warmup_ready")) is not bool
            or not sequence.get("chunks") or not sequence.get("events")
            or any(not event["checks"] or not all(
                       value for key, value in event["checks"].items()
                       if energy_only_stop and key != "energy_valid" or not energy_only_stop)
                   for event in sequence["events"])):
        raise ValueError("resume requires healthy budget-interrupted chains or an energy-only stop")
    if (not sequence["warmup_ready"]
            and sequence["counts"]["warmup"] >= config.warmup_max_results):
        raise ValueError("resume cannot extend an exhausted warmup maximum")
    previous = verified.VerifiedTensorFlowHMCConfig.from_payload(payload["config"])
    comparable = replace(config, verification_max_results=previous.verification_max_results,
                         resume_manifest=previous.resume_manifest)
    if comparable.payload() != previous.payload():
        raise ValueError("resume changed frozen tuning settings")
    if config.verification_max_results <= sequence["counts"]["verification"]:
        raise ValueError("resume maximum leaves no transitions")
    if payload.get("adapter_signature") != stage._adapter_signature(adapter):
        raise ValueError("resume adapter drift")
    if payload.get("runner_binding") != json.loads(json.dumps(binding.payload())):
        raise ValueError("resume binding drift")
    if tuple(payload.get("endpoint_status_fields", ())) != stage.ENDPOINT_STATUS_TRACE_FIELDS[:-3]:
        raise ValueError("resume endpoint fields drift")
    return previous, validate_source_migration(payload.get("source_hashes"))


def replay_then_continue(*, config, nomination, binding, source_root, records,
                         output_root=None, permit_new=None, run_new=None, progress_callback=None,
                         legacy_energy_gate=False):
    """Use the unchanged sequential controller for both archive replay and extension."""
    source_root = Path(source_root)
    if legacy_energy_gate and output_root is not None:
        raise ValueError("legacy energy gate is archive-validation only")
    cursor, graphs, timing = [0], {}, {}
    affine = verified._affine(nomination)
    initial = affine.position_to_latent(tf.constant(config.initial_chain_positions, tf.float64))

    def permit(phase, count):
        if cursor[0] < len(records):
            return True
        return output_root is not None and (permit_new is None or permit_new(phase, count))

    def run(state, seed, count):
        if cursor[0] < len(records):
            record = records[cursor[0]]
            if record["seed"] != seed.numpy().tolist() or record["count"] != count:
                raise ValueError("resume seed/count mismatch")
            saved = stage._read_tensor_record(source_root, record["archive"]["initial"], tf.float64)
            tf.debugging.assert_equal(saved, state, message="resume state handoff mismatch")
            return verified._read_chunk(source_root, record["archive"]["tensors"])
        started = time.monotonic()
        if run_new is not None:
            chunk = run_new(state, seed, count)
        else:
            if count not in graphs:
                graphs[count] = verified._chunk_graph(nomination, binding, count)
            chunk = stage._call_numerical_graph(graphs[count], state, seed,
                numerical_device=nomination.config.numerical_device)
        # Materialization synchronizes the GPU before recording a chunk forecast.
        chunk.final_state.numpy()
        timing["last_chunk_wall_seconds"] = time.monotonic() - started
        return chunk

    def archive(phase, index, chunk, state, seed):
        if cursor[0] < len(records):
            record = records[cursor[0]]
            if record["phase"] != phase or record["index"] != index:
                raise ValueError("resume phase/index mismatch")
            cursor[0] += 1
            if output_root is None:
                return record["archive"]
        prefix = f"sequence-{phase}-{index:03d}"
        return dict(tensors=verified._save_chunk(output_root, prefix, chunk),
                    initial=stage._write_tensor(output_root / f"{prefix}-initial.tfbin", state))

    def persist(value):
        if output_root is not None:
            stage._write_json(output_root / "fresh-progress.json", value)
            if progress_callback is not None:
                progress_callback(value)
            if cursor[0] == len(records):
                last = value["events"][-1] if value["events"] else {}
                print("HMC resumed sequence", value["counts"], value["status"],
                      last.get("rhat"), flush=True)

    result = run_sequential_neutra_hmc(config=config, initial_state=initial,
        run_chunk=run, health=lambda chunk, start: verified._health(
            chunk, start, config.energy_error_max, binding, **(
                {"legacy_energy_gate": True} if legacy_energy_gate else {})),
        diagnose_chunk=lambda chunk: (verified._force_diagnostics(chunk, binding)
            if legacy_energy_gate else verified._chunk_diagnostics(chunk, binding, config.energy_error_max)),
        diagnose=verified._diagnostic_rhat,
        transform=lambda states: verified._diagnostic_samples(states, affine, binding),
        archive=archive, persist=persist,
        verification_threshold=verified.HMC_TUNING_ORDINARY_RHAT_THRESHOLD, permit_chunk=permit)
    if cursor[0] != len(records):
        raise ValueError("resume archive has chunks after its replay stopping point")
    result["resume_timing"] = timing
    return result


def prepare_resume(*, adapter, config, runner_binding):
    """Validate all prior transitions without launching a sampling kernel."""
    path = Path(config.resume_manifest).resolve()
    payload = json.loads(path.read_text())
    previous, migration = validate_resume_payload(payload, config, adapter, runner_binding)
    nomination_path = path.parent / "nomination" / "manifest.json"
    if stage._sha256(nomination_path) != payload["nomination_manifest_sha256"]:
        raise ValueError("resume nomination checksum mismatch")
    nomination = stage.load_tensorflow_hmc_tuning_result(
        nomination_path, adapter=adapter, runner_binding=runner_binding)
    verified._validate_nomination(nomination)
    if nomination.config.payload() != config.nomination.payload():
        raise ValueError("resume nomination configuration mismatch")
    replay = replay_then_continue(config=previous, nomination=nomination,
        binding=runner_binding, source_root=path.parent, records=payload["sequence"]["chunks"],
        legacy_energy_gate=migration["kind"] != "unchanged")
    verified._check_sequence_replay(replay, payload["sequence"])
    from .hmc_sequential_verification import combined_chunks
    for phase in ("warmup", "verification"):
        actual = combined_chunks(replay["phases"][phase])
        # The writer represents an absent phase as {}, not null.
        if actual is None or not payload[phase]:
            if actual is not None or bool(payload[phase]):
                raise ValueError("resume aggregate/chunk presence mismatch")
            continue
        aggregate = verified._read_chunk(path.parent, payload[phase])
        for name in actual._fields:
            tf.debugging.assert_equal(getattr(actual, name), getattr(aggregate, name),
                                      message="resume aggregate/chunk mismatch")
    return path, payload, nomination, migration


def resume_verified_tuning(*, adapter, config, output_dir, runner_binding):
    if output_dir is None:
        raise ValueError("resume requires a new durable output directory")
    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError("resume output directory must be new")
    path, payload, nomination, migration = prepare_resume(
        adapter=adapter, config=config, runner_binding=runner_binding)
    root.mkdir(parents=True)
    shutil.copytree(path.parent / "nomination", root / "nomination")
    nomination = stage.load_tensorflow_hmc_tuning_result(
        root / "nomination/manifest.json", adapter=adapter, runner_binding=runner_binding)
    continuation = dict(parent_manifest=str(path), parent_sha256=stage._sha256(path),
        source_migration=migration, prior_counts=payload["sequence"]["counts"],
        repeats_tuning=False, repeats_warmup=False,
        extends_warmup=not payload["sequence"]["warmup_ready"], scientific_attempt=2)
    stage._write_json(root / "continuation.json", continuation)
    def progress(value):
        counts = value["counts"]
        stage._write_json(root.parent / "next-phase-plan.json", dict(
            part_of_existing_master=True, phase=value["next_phase_plan"]["phase"],
            action=value["next_phase_plan"]["action"], counts=counts,
            post_warmup_maximum=config.verification_max_results,
            retained_after_verification=True, independent_replication=False))
    sequence = replay_then_continue(config=config, nomination=nomination,
        binding=runner_binding, source_root=path.parent, records=payload["sequence"]["chunks"],
        output_root=root, permit_new=runtime_budget_guard(), progress_callback=progress)
    result = verified._finish_verified_sequence(config, nomination, runner_binding, root, sequence)
    manifest_path = Path(result.artifact_manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["continuation"] = continuation
    stage._write_json(manifest_path, manifest)
    return result

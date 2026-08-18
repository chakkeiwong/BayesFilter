#!/usr/bin/env python3
"""Run searched NeuTra curriculum selection and fresh audit for one control target."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-neutra-curriculum-control-campaign-plan-2026-08-15.md"
BASE_RUNNER = ROOT / "docs/benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py"


def _base_runner() -> Any:
    spec = importlib.util.spec_from_file_location("curriculum_control_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the known-law base runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target", choices=("gaussian", "banana"), required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rates", default="2e-4,5e-4,1e-3")
    parser.add_argument("--probe-updates", type=int, default=100)
    parser.add_argument("--probe-replicates", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--maximum-depth", type=int, default=3)
    parser.add_argument("--maximum-probe-calls", type=int, default=80)
    parser.add_argument("--total-updates", type=int, default=3000)
    parser.add_argument("--warmup-updates-per-group", type=int, default=100)
    parser.add_argument("--tournament-replicates", type=int, default=4)
    parser.add_argument("--final-seeds", type=int, default=2)
    parser.add_argument("--selection-count", type=int, default=65536)
    parser.add_argument("--calibration-replicates", type=int, default=16)
    parser.add_argument("--proposal-audit-count", type=int, default=131072)
    return parser.parse_args()


def _rates(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(raw).split(","))
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("learning rates must be finite and positive")
    return values


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_ready(payload), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state(transport: Any) -> tuple[Any, ...]:
    return tuple(variable.read_value() for variable in transport.trainable_variables)


def _restore(transport: Any, state: Sequence[Any]) -> None:
    variables = tuple(transport.trainable_variables)
    if len(variables) != len(state):
        raise RuntimeError("transport state count mismatch")
    for variable, value in zip(variables, state, strict=True):
        variable.assign(value)


def _state_hash(tf: Any, state: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    for value in state:
        digest.update(bytes(tf.io.serialize_tensor(value).numpy()))
    return digest.hexdigest()


def _sequence_code(sequence: Sequence[str]) -> int:
    digest = hashlib.sha256("\x1f".join(sequence).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 900_000_000


def _protocol_name(sequence: Sequence[str], rate: float) -> str:
    prefix = "cold" if not sequence else "then".join(sequence)
    return f"{prefix}__lr_{format(float(rate), '.8g')}"


def _calibrate_exact_loss_repeatability(
    tf: Any,
    *,
    dimension: int,
    sample_count: int,
    replicates: int,
    target_offset: float,
) -> Mapping[str, Any]:
    means = []
    for replicate in range(int(replicates)):
        latent = tf.random.stateless_normal(
            (int(sample_count), int(dimension)),
            seed=(20260815, 61000 + replicate),
            dtype=tf.float64,
        )
        value = tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(latent), axis=1)) + tf.constant(
            float(target_offset), tf.float64
        )
        means.append(float(value.numpy()))
    standard_deviation = statistics.stdev(means)
    margin = 2.0 * standard_deviation
    return {
        "schema": "bayesfilter.neutra.curriculum.exact_loss_repeatability.v1",
        "sample_count": int(sample_count),
        "replicates": int(replicates),
        "batch_means": means,
        "sample_standard_deviation": standard_deviation,
        "two_sd_margin": margin,
    }


def _probe_payload(candidate: Any) -> Mapping[str, Any]:
    return {
        "parent_sequence": candidate.parent_sequence,
        "candidate_group": candidate.candidate_group,
        "improvements_per_update": candidate.improvements_per_update,
        "mean_improvement_per_update": candidate.mean_improvement_per_update,
        "standard_deviation": candidate.standard_deviation,
        "lower_confidence_bound": candidate.lower_confidence_bound,
        "passed": candidate.passed,
        "rejection_reason": candidate.rejection_reason,
        "observations": [observation.__dict__ for observation in candidate.observations],
    }


def _phase_payload(phase: Any) -> Mapping[str, Any]:
    return {
        "name": phase.name,
        "active_groups": phase.active_groups,
        "first_global_update": phase.first_global_update,
        "last_global_update": phase.last_global_update,
        "updates": phase.updates,
        "clipped_updates": phase.clipped_updates,
        "terminal_gradient_norm": phase.terminal_gradient_norm,
    }


def main() -> int:
    args = _args()
    rates = _rates(args.learning_rates)
    if not PLAN.is_file() or not BASE_RUNNER.is_file():
        raise FileNotFoundError("reviewed plan or known-law base runner is missing")
    if int(args.batch_size) <= 1 or int(args.selection_count) < 32768:
        raise ValueError("batch size must exceed one and selection count must be at least 32768")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()

    import tensorflow as tf

    from bayesfilter.inference.neutra_curriculum_search import (
        NeuTraCurriculumGroup,
        NeuTraCurriculumProbe,
        NeuTraCurriculumSearchConfig,
        NeuTraProtocolObservation,
        NeuTraProtocolSelectionConfig,
        search_neutra_curriculum,
        select_neutra_protocol,
    )
    from bayesfilter.inference.neutra_curriculum_training import (
        train_neutra_curriculum_protocol,
        tune_neutra_curriculum_probe,
    )
    from bayesfilter.inference.neutra_staged_training import (
        dense_iaf_five_stage_variable_groups,
    )
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical_gpus) != 1:
        raise RuntimeError(f"expected one visible GPU, found {logical_gpus}")

    base = _base_runner()
    model = base._model(tf, str(args.target))
    dimension = int(model["dimension"])
    exact_offset = 0.0
    if args.target == "gaussian":
        factor = tf.constant(model["manifest"]["factor"], tf.float64)
        exact_offset = -float(tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor))).numpy())
    calibration = _calibrate_exact_loss_repeatability(
        tf,
        dimension=dimension,
        sample_count=int(args.selection_count),
        replicates=int(args.calibration_replicates),
        target_offset=exact_offset,
    )
    probe_threshold = float(calibration["two_sd_margin"]) / float(args.probe_updates)
    protocol_tolerance = float(calibration["two_sd_margin"])

    template = base._transport(tf, model, 1000)
    group_names = tuple(
        group.name for group in dense_iaf_five_stage_variable_groups(template)
    )
    search_groups = tuple(NeuTraCurriculumGroup(name) for name in group_names)
    del template

    parent_states: dict[tuple[tuple[str, ...], int], tuple[Any, ...]] = {}
    search_tuning_updates = 0
    probe_details = []
    progress = {
        "schema": "bayesfilter.neutra.curriculum_control_progress.v1",
        "target": str(args.target),
        "phase": "calibration_complete",
        "completed_probes": 0,
        "completed_tournament_arms": 0,
        "completed_final_runs": 0,
        "calibration": calibration,
    }
    _write(output / "progress.json", progress)

    def root_state(replicate: int) -> tuple[Any, ...]:
        key = ((), int(replicate))
        if key not in parent_states:
            transport = base._transport(tf, model, 1000 + int(replicate))
            parent_states[key] = _state(transport)
        return parent_states[key]

    def probe_fn(sequence: tuple[str, ...], candidate: str, replicate: int):
        nonlocal search_tuning_updates
        parent_key = (tuple(sequence), int(replicate))
        parent = parent_states.get(parent_key)
        if parent is None:
            if sequence:
                raise RuntimeError(f"missing parent state for {parent_key}")
            parent = root_state(replicate)
        transport = base._transport(tf, model, 1000 + int(replicate))
        _restore(transport, parent)
        selection_latent = tf.random.stateless_normal(
            (int(args.selection_count), dimension),
            seed=(20260815 + int(replicate), 62001),
            dtype=tf.float64,
        )
        node_code = _sequence_code(sequence)
        result = tune_neutra_curriculum_probe(
            transport=transport,
            target_log_prob_fn=model["target_log_prob"],
            variable_groups=dense_iaf_five_stage_variable_groups(transport),
            active_groups=sequence + (candidate,),
            learning_rates=rates,
            updates=int(args.probe_updates),
            latent_batch_fn=lambda update: tf.random.stateless_normal(
                (int(args.batch_size), dimension),
                seed=(20260815 + int(replicate), 63000 + node_code + int(update)),
                dtype=tf.float64,
            ),
            selection_loss_fn=lambda active: base._selection_loss(
                tf, active, model["target_log_prob"], selection_latent
            ),
            jit_compile=True,
        )
        child_key = (sequence + (candidate,), int(replicate))
        parent_states[child_key] = result.selected_state
        search_tuning_updates += int(result.tuning_optimizer_updates)
        detail = {
            "parent_sequence": sequence,
            "candidate_group": candidate,
            "replicate": int(replicate),
            "incoming_loss": result.incoming_loss,
            "selected_learning_rate": result.selected_learning_rate,
            "selected_loss": result.selected_loss,
            "parent_state_hash": _state_hash(tf, parent),
            "child_state_hash": _state_hash(tf, result.selected_state),
            "tuning_optimizer_updates": result.tuning_optimizer_updates,
            "rate_candidates": [
                {
                    "learning_rate": row.learning_rate,
                    "terminal_loss": row.terminal_loss,
                    "clipped_updates": row.clipped_updates,
                    "terminal_gradient_norm": row.terminal_gradient_norm,
                }
                for row in result.candidates
            ],
        }
        probe_details.append(detail)
        progress.update(
            {
                "phase": "search",
                "completed_probes": len(probe_details),
                "latest_probe": {
                    "parent_sequence": sequence,
                    "candidate_group": candidate,
                    "replicate": int(replicate),
                    "selected_learning_rate": result.selected_learning_rate,
                    "selected_loss": result.selected_loss,
                },
            }
        )
        _write(output / "progress.json", progress)
        return NeuTraCurriculumProbe(
            parent_sequence=sequence,
            candidate_group=candidate,
            replicate=int(replicate),
            incoming_loss=result.incoming_loss,
            best_loss=result.selected_loss,
            executed_updates=int(args.probe_updates),
            probe_updates=int(args.probe_updates),
            finite=True,
            parent_state_hash=detail["parent_state_hash"],
        )

    search = search_neutra_curriculum(
        groups=search_groups,
        probe_fn=probe_fn,
        config=NeuTraCurriculumSearchConfig(
            probe_updates=int(args.probe_updates),
            probe_replicates=int(args.probe_replicates),
            beam_width=int(args.beam_width),
            maximum_depth=int(args.maximum_depth),
            maximum_probe_calls=int(args.maximum_probe_calls),
            critical_value=2.0,
            minimum_improvement_per_update=probe_threshold,
        ),
    )
    tournament_sequences = tuple(dict.fromkeys(((), *search.final_beam_sequences)))
    tournament_observations = []
    tournament_rows = []
    for replicate in range(int(args.tournament_replicates)):
        selection_latent = tf.random.stateless_normal(
            (int(args.selection_count), dimension),
            seed=(20260815 + replicate, 64001),
            dtype=tf.float64,
        )
        for sequence in tournament_sequences:
            for rate in rates:
                transport = base._transport(tf, model, 2000 + replicate)
                result = train_neutra_curriculum_protocol(
                    transport=transport,
                    target_log_prob_fn=model["target_log_prob"],
                    variable_groups=dense_iaf_five_stage_variable_groups(transport),
                    sequence=sequence,
                    learning_rate=rate,
                    total_updates=int(args.total_updates),
                    warmup_updates_per_group=int(args.warmup_updates_per_group),
                    latent_batch_fn=lambda update, rep=replicate: tf.random.stateless_normal(
                        (int(args.batch_size), dimension),
                        seed=(20260815 + rep, 65000 + int(update)),
                        dtype=tf.float64,
                    ),
                    selection_loss_fn=lambda active: base._selection_loss(
                        tf, active, model["target_log_prob"], selection_latent
                    ),
                    jit_compile=True,
                )
                name = _protocol_name(sequence, rate)
                observation = NeuTraProtocolObservation(
                    name=name,
                    sequence=sequence,
                    replicate=replicate,
                    terminal_loss=result.terminal_loss,
                    executed_updates=result.executed_updates,
                    update_budget=int(args.total_updates),
                    finite=True,
                    selection_partition_id=f"{args.target}-tournament-selection-{replicate}",
                )
                tournament_observations.append(observation)
                tournament_rows.append(
                    {
                        **observation.__dict__,
                        "learning_rate": rate,
                        "phases": [_phase_payload(phase) for phase in result.phases],
                        "final_state_hash": _state_hash(tf, result.final_state),
                    }
                )
                progress.update(
                    {
                        "phase": "tournament",
                        "completed_tournament_arms": len(tournament_rows),
                        "latest_tournament_arm": {
                            "name": name,
                            "replicate": replicate,
                            "terminal_loss": result.terminal_loss,
                        },
                    }
                )
                _write(output / "progress.json", progress)
    selection = select_neutra_protocol(
        observations=tournament_observations,
        config=NeuTraProtocolSelectionConfig(
            replicates=int(args.tournament_replicates),
            critical_value=2.0,
            practical_loss_tolerance=protocol_tolerance,
        ),
    )
    rate_by_name = {row["name"]: float(row["learning_rate"]) for row in tournament_rows}
    selected_rate = rate_by_name[selection.selected_name]
    cold_names = tuple(name for name in rate_by_name if name.startswith("cold__"))
    mean_loss_by_name = {
        name: statistics.mean(
            row["terminal_loss"] for row in tournament_rows if row["name"] == name
        )
        for name in rate_by_name
    }
    cold_name = min(cold_names, key=lambda name: (mean_loss_by_name[name], name))
    final_specs = [(selection.selected_name, selection.selected_sequence, selected_rate)]
    if cold_name != selection.selected_name:
        final_specs.append((cold_name, (), rate_by_name[cold_name]))
    final_rows = []
    for name, sequence, rate in final_specs:
        for seed_index in range(int(args.final_seeds)):
            transport = base._transport(tf, model, 3000 + seed_index)
            selection_latent = tf.random.stateless_normal(
                (int(args.selection_count), dimension),
                seed=(20260815 + seed_index, 66001),
                dtype=tf.float64,
            )
            trained = train_neutra_curriculum_protocol(
                transport=transport,
                target_log_prob_fn=model["target_log_prob"],
                variable_groups=dense_iaf_five_stage_variable_groups(transport),
                sequence=sequence,
                learning_rate=rate,
                total_updates=int(args.total_updates),
                warmup_updates_per_group=int(args.warmup_updates_per_group),
                latent_batch_fn=lambda update, seed=seed_index: tf.random.stateless_normal(
                    (int(args.batch_size), dimension),
                    seed=(20260815 + seed, 67000 + int(update)),
                    dtype=tf.float64,
                ),
                selection_loss_fn=lambda active: base._selection_loss(
                    tf, active, model["target_log_prob"], selection_latent
                ),
                jit_compile=True,
            )
            validation = base._proposal_audit(
                tf,
                transport,
                model,
                sample_count=int(args.proposal_audit_count),
                seed=(20260815, 68001 + seed_index),
            )
            final_rows.append(
                {
                    "name": name,
                    "sequence": sequence,
                    "learning_rate": rate,
                    "seed_index": seed_index,
                    "terminal_selection_loss": trained.terminal_loss,
                    "executed_updates": trained.executed_updates,
                    "phases": [_phase_payload(phase) for phase in trained.phases],
                    "state_hash": _state_hash(tf, trained.final_state),
                    "validation": validation,
                    "known_law_gate_passed": bool(tf.convert_to_tensor(validation["passed"]).numpy()),
                }
            )
            progress.update(
                {
                    "phase": "final",
                    "completed_final_runs": len(final_rows),
                    "latest_final_run": {
                        "name": name,
                        "seed_index": seed_index,
                        "known_law_gate_passed": final_rows[-1][
                            "known_law_gate_passed"
                        ],
                    },
                }
            )
            _write(output / "progress.json", progress)

    allocator = tf.config.experimental.get_memory_info("GPU:0")
    manifest = {
        "schema": "bayesfilter.neutra.curriculum_control_manifest.v1",
        "plan": PLAN.as_posix(),
        "target": model["manifest"],
        "target_name": str(args.target),
        "group_names": group_names,
        "learning_rates": rates,
        "batch_size": int(args.batch_size),
        "selection_count": int(args.selection_count),
        "probe_updates": int(args.probe_updates),
        "probe_replicates": int(args.probe_replicates),
        "beam_width": int(args.beam_width),
        "maximum_depth": int(args.maximum_depth),
        "maximum_probe_calls": int(args.maximum_probe_calls),
        "total_updates": int(args.total_updates),
        "warmup_updates_per_group": int(args.warmup_updates_per_group),
        "tournament_replicates": int(args.tournament_replicates),
        "final_seeds": int(args.final_seeds),
        "proposal_audit_count": int(args.proposal_audit_count),
        "probe_threshold": probe_threshold,
        "protocol_tolerance": protocol_tolerance,
        "jit_compile": True,
        "dtype": "float64",
        "tf32_enabled": False,
        "sample_wise_loop_or_scalar_fallback": False,
        "memory_policy": memory_policy,
        "allocator_bytes": allocator,
        "gpu": str(logical_gpus[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "tensorflow_version": tf.__version__,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, text=True, capture_output=True
        ).stdout.strip(),
        "command": " ".join(sys.argv),
    }
    result = {
        "schema": "bayesfilter.neutra.curriculum_control_result.v1",
        "manifest": manifest,
        "calibration": calibration,
        "search": {
            "representative_sequence": search.representative_sequence,
            "final_beam_sequences": search.final_beam_sequences,
            "viable_sequences": search.viable_sequences,
            "terminal_sequences": search.terminal_sequences,
            "probe_calls": search.probe_calls,
            "tuning_optimizer_updates": search_tuning_updates,
            "stop_reason": search.stop_reason,
            "candidates": [_probe_payload(candidate) for candidate in search.candidates],
            "details": probe_details,
            "nonclaims": search.nonclaims,
        },
        "tournament": {
            "sequences": tournament_sequences,
            "observations": tournament_rows,
            "selection": {
                "selected_name": selection.selected_name,
                "selected_sequence": selection.selected_sequence,
                "reference_name": selection.reference_name,
                "uncertainty_set": selection.uncertainty_set,
                "comparisons": [comparison.__dict__ for comparison in selection.comparisons],
                "nonclaims": selection.nonclaims,
            },
            "cold_comparator_name": cold_name,
        },
        "final": {
            "rows": final_rows,
            "selected_protocol_passed_both_seeds": all(
                row["known_law_gate_passed"]
                for row in final_rows
                if row["name"] == selection.selected_name
            ),
            "cold_comparator_passed_both_seeds": all(
                row["known_law_gate_passed"]
                for row in final_rows
                if row["name"] == cold_name
            ),
        },
        "decision": {
            "promotion": False,
            "no_hmc": True,
            "status": "control_campaign_complete",
            "nonclaims": [
                "no SSL-LSTM transfer claim",
                "no HMC or posterior-correctness claim",
                "no universal curriculum or default-readiness claim",
            ],
        },
        "wall_seconds": time.perf_counter() - started,
    }
    progress.update(
        {
            "phase": "complete",
            "selected_name": selection.selected_name,
            "selected_protocol_passed_both_seeds": result["final"][
                "selected_protocol_passed_both_seeds"
            ],
        }
    )
    _write(output / "progress.json", progress)
    _write(output / "run_manifest.json", manifest)
    _write(output / "result.json", result)
    _write(
        output / "artifact_hashes.json",
        {
            "schema": "bayesfilter.neutra.curriculum_control_hashes.v1",
            "artifacts": {
                path.relative_to(output).as_posix(): _sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "artifact_hashes.json"
            },
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output),
                "target": str(args.target),
                "selected_name": selection.selected_name,
                "selected_passed": result["final"]["selected_protocol_passed_both_seeds"],
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

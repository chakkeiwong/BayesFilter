"""Single shared top-level procedure for frozen-transport NeuTra broad-grid runs.

Every common claim-bearing NeuTra broad-grid campaign should route through the
shared repaired state-continuing epsilon-repair procedure. The historical
operational route is retained only as an explicit legacy/reference mode.

- ``state_continuing_epsilon_repair_v1``: the common/default NeuTra-whitened
  HMC tuning procedure. It uses dual averaging followed by a bounded,
  bracketed epsilon-repair loop with state continuation, then fresh final
  screens.
- ``operational_broad_grid_v1``: independent dual-averaging epsilon per primary
  ``L`` plus same-epsilon one-hop coverage probes; no state continuation and no
  epsilon-repair loop (legacy/reference only).

The procedure validates the frozen transport and target identity, dispatches
the selected variant, emits normalized artifacts that name the variant, and
owns the only sequential-handoff parser. Campaign scripts must select a
variant explicitly when they truly need the legacy/reference route; otherwise
this module should run the repaired common route.

This module is import-light (no TensorFlow at import time) so its contracts
can be unit-tested cheaply; heavy dependencies are imported inside the run
function.
"""

from __future__ import annotations

import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROCEDURE_FAMILY = "shared_neutra_broad_grid"
OPERATIONAL_BROAD_GRID_V1 = "operational_broad_grid_v1"
STATE_CONTINUING_EPSILON_REPAIR_V1 = "state_continuing_epsilon_repair_v1"
DEFAULT_COMMON_VARIANT = STATE_CONTINUING_EPSILON_REPAIR_V1
PROCEDURE_VARIANTS = (
    OPERATIONAL_BROAD_GRID_V1,
    STATE_CONTINUING_EPSILON_REPAIR_V1,
)

BASE_REQUIRED_STATUS_KEYS = ("status_code", "valid_pre_regularized_score")
DEFAULT_REQUIRED_STATUS_KEYS = (
    "status_code",
    "valid_pre_regularized_score",
    "floor_count_value",
    "min_innovation_eigenvalue",
    "innovation_condition_estimate",
)

EXPECTED_BROAD_GRID_ROUTE = "operational_broad_fixed_mass_l_epsilon_grid_v1"
LEGACY_UNLABELED_VARIANT = "legacy_unlabeled_pre_variant_artifact"


def _normalized_status_keys(required_status_keys: Any) -> tuple[str, ...]:
    keys = tuple(dict.fromkeys(str(item) for item in required_status_keys))
    if not keys or any(not item for item in keys):
        raise ValueError("required_status_keys must contain nonempty names")
    if any(base not in keys for base in BASE_REQUIRED_STATUS_KEYS):
        raise ValueError(
            "required_status_keys must include status_code and "
            "valid_pre_regularized_score"
        )
    return keys


def procedure_metadata(variant: str) -> Mapping[str, Any]:
    """Normalized variant flags embedded in every broad-grid artifact."""

    name = str(variant)
    if name not in PROCEDURE_VARIANTS:
        raise ValueError(f"unknown NeuTra procedure variant: {name}")
    repaired = name == STATE_CONTINUING_EPSILON_REPAIR_V1
    return {
        "procedure_family": PROCEDURE_FAMILY,
        "procedure_variant": name,
        "state_continuation_performed": repaired,
        "epsilon_repair_performed": repaired,
        "directional_refinement_performed": False,
        "same_epsilon_neighbor_guards_performed": True,
        "independent_primary_dual_averaging_performed": True,
        "fresh_final_screens_performed": True,
    }


def operational_procedure_metadata() -> Mapping[str, Any]:
    return procedure_metadata(OPERATIONAL_BROAD_GRID_V1)


def state_continuing_procedure_metadata() -> Mapping[str, Any]:
    return procedure_metadata(STATE_CONTINUING_EPSILON_REPAIR_V1)


@dataclass(frozen=True)
class SharedNeuTraProcedureConfig:
    """Inputs for one shared frozen-transport broad-grid procedure run."""

    output_root: Path
    frozen_transport_path: Path
    expected_frozen_transport_sha256: str
    root_seed: tuple[int, int]
    variant: str = DEFAULT_COMMON_VARIANT
    launch_sequential: bool = False
    initial_step_size: float | None = None
    screen_results: int = 128
    sequential_chunk_results: int = 65
    require_gpu: bool = True
    jit_compile: bool = True
    required_status_keys: tuple[str, ...] = DEFAULT_REQUIRED_STATUS_KEYS
    initial_epsilon_by_l: Mapping[int, float] | None = None
    adaptation_steps: int = 96
    post_adaptation_results: int = 32
    calibration_results: int = 32
    calibration_burnin: int = 1
    calibration_region: tuple[float, float] = (0.68, 0.72)
    epsilon_repair_factor: float = 1.20
    max_epsilon_repairs: int = 3
    final_screen_results: int = 96
    final_screen_burnin: int = 8

    def __post_init__(self) -> None:
        digest = str(self.expected_frozen_transport_sha256).lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("expected_frozen_transport_sha256 must be SHA-256 hex")
        seed = tuple(int(item) for item in self.root_seed)
        if len(seed) != 2 or any(item < 0 for item in seed):
            raise ValueError("root_seed must contain two nonnegative integers")
        if str(self.variant) not in PROCEDURE_VARIANTS:
            raise ValueError(f"unknown NeuTra procedure variant: {self.variant}")
        step = self.initial_step_size
        if step is not None and (not math.isfinite(float(step)) or float(step) <= 0.0):
            raise ValueError("initial_step_size must be positive and finite")
        if bool(self.require_gpu) and not bool(self.jit_compile):
            raise ValueError("serious broad-grid tuning requires GPU/XLA")
        if int(self.screen_results) <= 64:
            raise ValueError("broad-grid screen_results must exceed 64")
        if int(self.final_screen_results) <= 64:
            raise ValueError("final_screen_results must exceed 64")
        if int(self.sequential_chunk_results) <= 1:
            raise ValueError("sequential_chunk_results must exceed one")
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(
            self, "frozen_transport_path", Path(self.frozen_transport_path)
        )
        object.__setattr__(self, "expected_frozen_transport_sha256", digest)
        object.__setattr__(self, "root_seed", seed)
        object.__setattr__(self, "variant", str(self.variant))
        object.__setattr__(
            self, "initial_step_size", None if step is None else float(step)
        )
        object.__setattr__(self, "screen_results", int(self.screen_results))
        object.__setattr__(
            self, "sequential_chunk_results", int(self.sequential_chunk_results)
        )
        object.__setattr__(
            self,
            "required_status_keys",
            _normalized_status_keys(self.required_status_keys),
        )


def extract_sequential_handoff_from_broad_grid_result(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Parse the unique admissible pair from one broad-grid private artifact.

    This is the only reviewed sequential-handoff parser. It enforces the
    complete unranked viable-set semantics and refuses anything other than
    exactly one independently tuned primary in the primary-plus-coverage
    union. Artifacts produced before the variant metadata existed are accepted
    and labeled ``legacy_unlabeled_pre_variant_artifact``.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("broad-grid result payload must be a mapping")
    if (
        payload.get("route") != EXPECTED_BROAD_GRID_ROUTE
        or payload.get("disposition") != "viable_pair_set"
        or payload.get("stochastic_ranking_performed") is not False
        or payload.get("all_viable_pairs_preserved") is not True
    ):
        raise ValueError("broad-grid result is not a complete viable set")
    variant = str(payload.get("procedure_variant", LEGACY_UNLABELED_VARIANT))
    if variant != LEGACY_UNLABELED_VARIANT and variant not in PROCEDURE_VARIANTS:
        raise ValueError(f"broad-grid result names an unknown variant: {variant}")
    candidates = payload.get("next_round_candidates")
    if not isinstance(candidates, Sequence) or len(candidates) != 1:
        raise ValueError("sequential handoff requires exactly one unranked viable pair")
    candidate = candidates[0]
    evidence = candidate.get("evidence", {})
    if (
        candidate.get("viable") is not True
        or evidence.get("disposition") != "provisional_viable"
    ):
        raise ValueError("broad-grid candidate is not viable")
    request = candidate.get("request", {})
    if request.get("role") != "independently_tuned_primary":
        raise ValueError("unique survivor is not an independently tuned primary")
    step_size = float(candidate.get("tuned_step_size"))
    leapfrog = int(request.get("num_leapfrog_steps"))
    if not math.isfinite(step_size) or step_size <= 0.0 or leapfrog <= 0:
        raise ValueError("broad-grid kernel mechanics are invalid")
    return {
        "step_size": step_size,
        "num_leapfrog_steps": leapfrog,
        "candidate": candidate,
        "procedure_variant": variant,
    }


def run_shared_neutra_procedure(
    *,
    spec: Any,
    config: SharedNeuTraProcedureConfig,
) -> Mapping[str, Any]:
    """Run one explicit-variant broad-grid tuning campaign for one cell."""

    import tensorflow as tf

    from bayesfilter.inference.neutra_end_to_end import (
        BatchNativeBoundAdapter,
        NeuTraEndToEndError,
        _RunState,
        _base_result,
        _file_sha256,
        _fixed_transport_adapter,
        _read_mapping,
        _target_signature,
    )
    from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
    from bayesfilter.runtime import atomic_write_json
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    root = config.output_root / spec.cell_id
    if root.exists():
        raise NeuTraEndToEndError(f"cell output root must be fresh: {root}")
    root.mkdir(parents=True)
    run_state = _RunState(root, cell_id=spec.cell_id, output_root=config.output_root)
    run_state.update("launch", status="started")
    started = time.monotonic()
    memory_policy = configure_tensorflow_gpu_memory_growth(
        tf, require_gpu=config.require_gpu
    )
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    frozen_path = config.frozen_transport_path
    if not frozen_path.is_file():
        raise NeuTraEndToEndError(f"frozen transport does not exist: {frozen_path}")
    observed_sha256 = _file_sha256(frozen_path)
    if observed_sha256 != config.expected_frozen_transport_sha256:
        raise NeuTraEndToEndError(
            "frozen transport SHA-256 mismatch: "
            f"{observed_sha256} != {config.expected_frozen_transport_sha256}"
        )
    adapter = spec.adapter_factory()
    observed_signature = _target_signature(adapter)
    if observed_signature != spec.target_signature:
        raise NeuTraEndToEndError(
            f"target signature mismatch for {spec.cell_id}: "
            f"{observed_signature} != {spec.target_signature}"
        )
    bound_adapter = BatchNativeBoundAdapter(
        adapter, target_signature=spec.target_signature
    )
    loaded = load_frozen_neutra_artifact(
        _read_mapping(frozen_path),
        expected_target_signature=spec.target_signature,
    )
    transport_input = {
        "path": str(frozen_path),
        "sha256": observed_sha256,
        "target_signature": loaded.manifest.target_signature,
        "training_state_hash": loaded.manifest.training_state_hash,
        "retrained": False,
    }
    atomic_write_json(root / "frozen_transport_input.json", transport_input)
    metadata = procedure_metadata(config.variant)
    tuned_adapter = _fixed_transport_adapter(
        bound_adapter,
        loaded.transport,
        f"{spec.cell_id}:{config.variant}_fixed_identity_grid",
    )
    initial_step_size = (
        spec.initial_step_size
        if config.initial_step_size is None
        else config.initial_step_size
    )
    run_state.update(f"broad_grid_tuning:{config.variant}")
    if config.variant == OPERATIONAL_BROAD_GRID_V1:
        from bayesfilter.inference.neutra_broad_grid import (
            NeuTraBroadGridTuningConfig,
            run_neutra_operational_broad_grid_tuning,
        )

        broad = run_neutra_operational_broad_grid_tuning(
            adapter=tuned_adapter,
            target_signature=spec.target_signature,
            config=NeuTraBroadGridTuningConfig(
                initial_step_size=initial_step_size,
                root_seed=config.root_seed,
                screen_results=config.screen_results,
                use_xla=config.jit_compile,
                required_status_keys=config.required_status_keys,
                evidence_path=spec.plan_path,
            ),
            output_dir=root / "broad-grid-tuning",
            procedure_metadata=metadata,
        )
    else:
        from bayesfilter.inference.neutra_state_continuing_broad_grid import (
            NeuTraStateContinuingBroadGridConfig,
            run_neutra_state_continuing_broad_grid_tuning,
        )

        broad = run_neutra_state_continuing_broad_grid_tuning(
            adapter=tuned_adapter,
            target_signature=spec.target_signature,
            config=NeuTraStateContinuingBroadGridConfig(
                initial_step_size=initial_step_size,
                root_seed=config.root_seed,
                evidence_path=spec.plan_path,
                initial_epsilon_by_l=config.initial_epsilon_by_l,
                adaptation_steps=config.adaptation_steps,
                post_adaptation_results=config.post_adaptation_results,
                calibration_results=config.calibration_results,
                calibration_burnin=config.calibration_burnin,
                calibration_region=config.calibration_region,
                epsilon_repair_factor=config.epsilon_repair_factor,
                max_epsilon_repairs=config.max_epsilon_repairs,
                final_screen_results=config.final_screen_results,
                final_screen_burnin=config.final_screen_burnin,
                use_xla=config.jit_compile,
                required_status_keys=config.required_status_keys,
            ),
            output_dir=root / "broad-grid-tuning",
            procedure_metadata=metadata,
        )
    public = broad["public"]
    passed = public.get("disposition") == "viable_pair_set"

    sequential_result: Mapping[str, Any] | None = None
    sequential_block_reason: str | None = None
    if config.launch_sequential and passed:
        private_result_path = Path(broad["private_result_path"])
        try:
            extract_sequential_handoff_from_broad_grid_result(broad["private"])
        except ValueError as error:
            sequential_block_reason = str(error)
        else:
            from bayesfilter.inference.neutra_end_to_end import (
                BroadGridSequentialConfig,
                run_neutra_broad_grid_sequential_cell,
            )

            run_state.update("sequential_from_unique_pair")
            sequential_result = run_neutra_broad_grid_sequential_cell(
                spec=spec,
                config=BroadGridSequentialConfig(
                    output_root=root / "sequential",
                    frozen_transport_path=frozen_path,
                    expected_frozen_transport_sha256=observed_sha256,
                    broad_grid_result_path=private_result_path,
                    expected_broad_grid_result_sha256=_file_sha256(private_result_path),
                    chunk_results=config.sequential_chunk_results,
                ),
            )

    result = _base_result(
        spec,
        root,
        {
            "passed": passed,
            "decision": (
                "BROAD_GRID_TUNING_VIABLE_PAIR_SET"
                if passed
                else "BROAD_GRID_TUNING_NO_HANDOFF"
            ),
            "procedure_family": PROCEDURE_FAMILY,
            "procedure_variant": config.variant,
            "frozen_transport_input": transport_input,
            "broad_grid": public,
            "private_tuning_result_path": broad["private_result_path"],
            "public_tuning_result_path": broad["public_result_path"],
            "sampling_launched": sequential_result is not None,
            "retained_sampling_authorized": False,
            "sequential_block_reason": sequential_block_reason,
            "sequential_result": sequential_result,
            "nonclaims": (
                "discarded broad-grid tuning evidence only",
                "complete candidate set is unranked",
                "no convergence, posterior, or default-readiness claim",
            ),
        },
        memory_policy,
        started,
    )
    atomic_write_json(root / "result.json", result)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    import os as _os
    import platform as _platform

    atomic_write_json(
        root / "run_manifest.json",
        {
            "schema": "bayesfilter.neutra.broad_grid_tuning_manifest.v1",
            "cell_id": spec.cell_id,
            "target_signature": spec.target_signature,
            "procedure_family": PROCEDURE_FAMILY,
            "procedure_variant": config.variant,
            "procedure_metadata": dict(metadata),
            "git_commit": commit,
            "command": tuple(sys.argv),
            "environment": _os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "python_version": _platform.python_version(),
            "tensorflow_version": tf.__version__,
            "gpu_memory_policy": memory_policy,
            "device": "/GPU:0" if config.require_gpu else "explicit_cpu_exception",
            "jit_compile": config.jit_compile,
            "tf32_execution_enabled": True,
            "root_seed": config.root_seed,
            "initial_step_size_role": "target_specific_warm_start_hypothesis",
            "required_status_keys": config.required_status_keys,
            "screen_results": (
                config.screen_results
                if config.variant == OPERATIONAL_BROAD_GRID_V1
                else config.final_screen_results
            ),
            "output_root": str(root),
            "wall_time_seconds": time.monotonic() - started,
            "frozen_transport_input": transport_input,
            "plan_path": spec.plan_path,
            "result_path": str(root / "result.json"),
            "private_tuning_result_path": broad["private_result_path"],
            "public_tuning_result_path": broad["public_result_path"],
            "sampling_launched": sequential_result is not None,
        },
    )
    run_state.complete(result)
    return result

"""Analyze Phase 8E C2 replication records with fixture-level uncertainty.

This is diagnostic/reporting code.  It deliberately uses only the Python
standard library: TensorFlow proposal values are read from JSON sidecars, and
fixture IDs are the resampling unit.  The analyzer never selects a schedule,
changes a proposal, or turns an invalid branch into a finite result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "c2_phase8e_statistical_replication_analysis_v1"
DEFAULT_PHASE_ID = "c2_exact_likelihood_laplace_phase8e_statistical_replication_v1"
CANDIDATE_LABEL = "exact_likelihood_laplace_k1"
DEFAULT_HEURISTICS = (
    "ukf_apf_k1",
    "bootstrap_conditional",
    "transformed_student_nu8",
    "stationary_independence",
)
DEFAULT_ACTIVE_START = 1
DEFAULT_EPSILON = 1.0e-12
DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260907
DEFAULT_SCHEDULE_ID = "quarter_long_12"
DEFAULT_SCHEDULE_LADDER_ID = "baseline_plus_reviewed_repair_hypotheses"
DEFAULT_PARTICLE_COUNT = 4096
DEFAULT_REPLICATION_SET_ID = "c2_phase8e_primary_n4096_v1"
DEFAULT_MODEL_SEED = 20260912
DEFAULT_OBSERVATION_SEEDS = tuple(range(424300, 424312))
BRANCH_SEED_BASE = 98200
BRANCH_SEED_PARTICLE_MULTIPLIER = 1
BRANCH_SEED_STRIDE = 1009
BRANCH_SEED_EXPRESSION = (
    "98200 + particle_count + 1009 * branch_index"
)
SALIENT_TIMES = (5, 14)


class AnalysisError(ValueError):
    """Raised when a replication artifact cannot answer the declared question."""


@dataclass(frozen=True)
class BinBoundaries:
    """Frozen boundaries for mean absolute observation magnitude per time."""

    near_zero_upper: float
    ordinary_upper: float
    source_path: str | None = None
    source_sha256: str | None = None
    calibration_observation_seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        near = float(self.near_zero_upper)
        ordinary = float(self.ordinary_upper)
        if not math.isfinite(near) or not math.isfinite(ordinary):
            raise AnalysisError("observation-bin boundaries must be finite")
        if near < 0.0 or ordinary <= near:
            raise AnalysisError(
                "observation-bin boundaries must satisfy 0 <= near < ordinary"
            )
        object.__setattr__(self, "near_zero_upper", near)
        object.__setattr__(self, "ordinary_upper", ordinary)

    def classify(self, mean_absolute_observation: float) -> str:
        value = float(mean_absolute_observation)
        if not math.isfinite(value) or value < 0.0:
            raise AnalysisError("observation magnitude must be finite and nonnegative")
        if value <= self.near_zero_upper:
            return "near_zero"
        if value <= self.ordinary_upper:
            return "ordinary"
        return "upper_tail"

    def payload(self) -> dict[str, object]:
        return {
            "statistic": "mean_absolute_observation_per_time",
            "near_zero_upper": self.near_zero_upper,
            "ordinary_upper": self.ordinary_upper,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "calibration_observation_seeds": list(
                self.calibration_observation_seeds
            ),
        }


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON artifact {path} must contain an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_sha256_file(path: Path, *, label: str) -> str:
    try:
        return _sha256_file(path)
    except OSError as exc:
        raise AnalysisError(f"{label}: cannot read {path}: {exc}") from exc


def _finite_float(value: object, *, label: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} is not numeric") from exc
    if not math.isfinite(result):
        raise AnalysisError(f"{label} is not finite")
    return result


def _integer(value: object, *, label: str) -> int:
    """Parse an integer field while turning malformed artifacts into AnalysisError."""
    if isinstance(value, bool):
        raise AnalysisError(f"{label} must be an integer")
    try:
        result = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise AnalysisError(f"{label} is not an integer") from exc
    return result


def _finite_vector(value: object, *, label: str) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AnalysisError(f"{label} must be a numeric sequence")
    result = [_finite_float(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    if not result:
        raise AnalysisError(f"{label} must be nonempty")
    return result


def _percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise AnalysisError("cannot compute a percentile of an empty sequence")
    ordered = sorted(float(value) for value in values)
    if probability <= 0.0:
        return ordered[0]
    if probability >= 1.0:
        return ordered[-1]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _bootstrap_mean(
    values: Sequence[float], *, resamples: int, seed: int
) -> dict[str, object]:
    observations = [float(value) for value in values]
    if not observations:
        raise AnalysisError("bootstrap requires at least one fixture")
    count = int(resamples)
    if count < 1:
        raise AnalysisError("bootstrap resamples must be positive")
    generator = random.Random(int(seed))
    size = len(observations)
    draws: list[float] = []
    for _ in range(count):
        total = 0.0
        for _ in range(size):
            total += observations[generator.randrange(size)]
        draws.append(total / float(size))
    estimate = statistics.fmean(observations)
    return {
        "estimate_mean": estimate,
        "percentile_95": {
            "lower": _percentile(draws, 0.025),
            "upper": _percentile(draws, 0.975),
        },
        "resamples": count,
        "seed": int(seed),
        "resampling_unit": "independent_observation_fixture",
    }


def _sign_test(values: Sequence[float]) -> dict[str, object]:
    positive = sum(float(value) > 0.0 for value in values)
    negative = sum(float(value) < 0.0 for value in values)
    ties = len(values) - positive - negative
    effective = positive + negative
    if effective == 0:
        p_value: float | None = None
    else:
        tail = sum(math.comb(effective, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / float(2**effective))
    return {
        "positive_count": positive,
        "negative_count": negative,
        "tie_count": ties,
        "effective_count": effective,
        "two_sided_exact_p_value": p_value,
        "null": "fixture-level paired contrast has probability one-half of either sign",
    }


def _resolve_path(value: object, *, base: Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    candidates = (base / path, ROOT / path, Path.cwd() / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return (base / path).resolve()


def _fixture_identity_and_bins(
    result: Mapping[str, Any],
    *,
    run_path: Path,
    boundaries: BinBoundaries | None,
) -> tuple[dict[str, object], dict[int, str]]:
    sources = result.get("sources")
    if not isinstance(sources, Mapping):
        raise AnalysisError(f"{run_path}: missing sources object")
    fixture_info = sources.get("fixture")
    if not isinstance(fixture_info, Mapping):
        raise AnalysisError(f"{run_path}: fixture source must expose path and SHA-256")
    fixture_value = fixture_info.get("path")
    declared_sha256 = str(fixture_info.get("sha256", ""))
    if not fixture_value or len(declared_sha256) != 64:
        raise AnalysisError(f"{run_path}: fixture source identity is incomplete")
    fixture_path = _resolve_path(fixture_value, base=run_path)
    actual_sha256 = _checked_sha256_file(
        fixture_path, label=f"{run_path}: fixture source"
    )
    if actual_sha256 != declared_sha256:
        raise AnalysisError(
            f"{run_path}: fixture SHA-256 mismatch: declared {declared_sha256}, "
            f"actual {actual_sha256}"
        )
    fixture = _load_json(fixture_path)
    observations = fixture.get("observations")
    if not isinstance(observations, Sequence) or isinstance(observations, (str, bytes)):
        raise AnalysisError(f"{fixture_path}: observations must be a matrix")
    bins: dict[int, str] = {}
    for time_index, row in enumerate(observations):
        values = _finite_vector(row, label=f"{fixture_path}:observations[{time_index}]")
        if boundaries is not None:
            mean_absolute = statistics.fmean(abs(value) for value in values)
            bins[time_index] = boundaries.classify(mean_absolute)
    return (
        {
            "path": str(fixture_path),
            "sha256": actual_sha256,
            "model_seed": fixture.get("model_seed"),
            "observation_seed": fixture.get("observation_seed"),
            "horizon": len(observations),
        },
        bins,
    )


def _record_index(result: Mapping[str, Any], *, run_path: Path) -> dict[int, dict[str, Mapping[str, Any]]]:
    records = result.get("records")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise AnalysisError(f"{run_path}: records must be a sequence")
    indexed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for raw in records:
        if not isinstance(raw, Mapping):
            raise AnalysisError(f"{run_path}: record is not an object")
        try:
            if "branch_index" not in raw or "label" not in raw:
                raise KeyError("branch_index/label")
            branch = int(raw["branch_index"])
            label = str(raw["label"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError(f"{run_path}: malformed record identity") from exc
        branch_rows = indexed.setdefault(branch, {})
        if label in branch_rows:
            raise AnalysisError(f"{run_path}: duplicate branch/label {branch}/{label}")
        branch_rows[label] = raw
    if not indexed:
        raise AnalysisError(f"{run_path}: no branch records")
    if sorted(indexed) != list(range(len(indexed))):
        raise AnalysisError(f"{run_path}: branch indices must be contiguous from zero")
    return indexed


def _active_log_ess_differences(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    active_start: int,
    epsilon: float,
    expected_horizon: int,
) -> list[float]:
    candidate_ess = _finite_vector(candidate.get("ess_by_time"), label="candidate ess_by_time")
    baseline_ess = _finite_vector(baseline.get("ess_by_time"), label="baseline ess_by_time")
    if len(candidate_ess) != len(baseline_ess):
        raise AnalysisError("candidate and baseline ESS vectors have different lengths")
    if len(candidate_ess) != int(expected_horizon):
        raise AnalysisError(
            f"ESS horizon {len(candidate_ess)} does not match fixture horizon "
            f"{int(expected_horizon)}"
        )
    if active_start < 0 or active_start >= len(candidate_ess):
        raise AnalysisError("active transition range is empty")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise AnalysisError("epsilon must be finite and positive")
    return [
        math.log(candidate_ess[index] + epsilon) - math.log(baseline_ess[index] + epsilon)
        for index in range(active_start, len(candidate_ess))
    ]


def _fixture_row(
    run_path: Path,
    result: Mapping[str, Any],
    *,
    heuristics: Sequence[str],
    active_start: int,
    epsilon: float,
    boundaries: BinBoundaries | None,
    expected_schedule_id: str,
    expected_schedule_ladder_id: str,
    expected_particle_count: int | None,
    expected_replication_set_id: str | None,
    expected_phase_id: str,
    expected_analysis_seed: int,
    expected_model_seed: int,
) -> dict[str, Any]:
    indexed = _record_index(result, run_path=run_path)
    if str(result.get("phase")) != str(expected_phase_id):
        raise AnalysisError(f"{run_path}: Phase 8E phase identity mismatch")
    if _integer(result.get("branch_count", -1), label=f"{run_path}: branch_count") != len(indexed):
        raise AnalysisError(f"{run_path}: declared branch count does not match records")
    if _integer(
        result.get("proposal_active_start_time", -1),
        label=f"{run_path}: proposal_active_start_time",
    ) != int(active_start):
        raise AnalysisError(f"{run_path}: active-time contract mismatch")
    selected_schedule = result.get("selected_schedule")
    if (
        not isinstance(selected_schedule, Mapping)
        or str(selected_schedule.get("config_id")) != str(expected_schedule_id)
        or str(result.get("schedule_ladder_id"))
        != str(expected_schedule_ladder_id)
        or not bool(result.get("include_repair_schedules", False))
        or str(result.get("schedule_selection_mode"))
        != "fixed_requested_config_after_independent_validity_check"
    ):
        raise AnalysisError(f"{run_path}: fixed schedule contract mismatch")
    particle_count = _integer(
        result.get("particle_count", -1), label=f"{run_path}: particle_count"
    )
    if expected_particle_count is not None and particle_count != int(expected_particle_count):
        raise AnalysisError(f"{run_path}: particle-count contract mismatch")
    fixture_identity, fixture_bins = _fixture_identity_and_bins(
        result, run_path=run_path, boundaries=boundaries
    )
    fixture_model_seed = _integer(
        fixture_identity.get("model_seed"), label=f"{run_path}: fixture model seed"
    )
    if fixture_model_seed != int(expected_model_seed):
        raise AnalysisError(f"{run_path}: fixture model seed mismatch")
    fixture_observation_seed = _integer(
        fixture_identity.get("observation_seed"),
        label=f"{run_path}: fixture observation seed",
    )
    result_horizon = _integer(
        result.get("horizon", -1), label=f"{run_path}: result horizon"
    )
    if result_horizon != int(fixture_identity["horizon"]):
        raise AnalysisError(f"{run_path}: result and fixture horizons differ")
    replication = result.get("replication")
    if not isinstance(replication, Mapping):
        raise AnalysisError(f"{run_path}: missing replication metadata")
    replication_set_id = str(replication.get("set_id", ""))
    if not replication_set_id:
        raise AnalysisError(f"{run_path}: replication set ID is empty")
    if (
        expected_replication_set_id is not None
        and replication_set_id != str(expected_replication_set_id)
    ):
        raise AnalysisError(
            f"{run_path}: replication set ID {replication_set_id!r} does not match "
            f"{str(expected_replication_set_id)!r}"
        )
    try:
        analysis_seed = int(replication["analysis_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"{run_path}: replication analysis seed is missing") from exc
    if analysis_seed != int(expected_analysis_seed):
        raise AnalysisError(f"{run_path}: replication analysis seed mismatch")
    try:
        model_seed = int(replication["model_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"{run_path}: replication model seed is missing") from exc
    if model_seed != int(expected_model_seed):
        raise AnalysisError(f"{run_path}: replication model seed mismatch")
    branch_seed_formula = replication.get("branch_seed_formula")
    expected_branch_seed_formula = {
        "base": BRANCH_SEED_BASE,
        "particle_count_multiplier": BRANCH_SEED_PARTICLE_MULTIPLIER,
        "branch_index_stride": BRANCH_SEED_STRIDE,
        "expression": BRANCH_SEED_EXPRESSION,
    }
    if branch_seed_formula != expected_branch_seed_formula:
        raise AnalysisError(f"{run_path}: branch seed formula mismatch")
    replication_observation_seed = _integer(
        replication.get("observation_seed"),
        label=f"{run_path}: replication observation seed",
    )
    if fixture_observation_seed != replication_observation_seed:
        raise AnalysisError(f"{run_path}: fixture and replication observation seeds differ")
    checks = result.get("checks")
    if not isinstance(checks, Mapping):
        raise AnalysisError(f"{run_path}: missing validity checks")
    infrastructure_valid = all(
        bool(checks.get(key, False))
        for key in (
            "records_complete",
            "source_files_present",
            "snapshot_bank_complete",
            "gpu_memory_growth_verified",
        )
    ) and not bool(result.get("continuation_veto", True))
    environment = result.get("environment")
    infrastructure_valid = bool(
        infrastructure_valid
        and isinstance(environment, Mapping)
        and bool(environment.get("jit_compile", False))
    )
    workspace = result.get("workspace")
    workspace_valid = bool(
        isinstance(workspace, Mapping)
        and str(workspace.get("git_commit", ""))
        and str(workspace.get("git_status", "")) == ""
    )
    infrastructure_valid = infrastructure_valid and workspace_valid
    branch_rows: list[dict[str, Any]] = []
    candidate_failure = False
    comparator_failure = False
    promotion_losses: list[dict[str, Any]] = []
    bin_losses: list[dict[str, Any]] = []
    contrasts_by_heuristic: dict[str, list[float]] = {label: [] for label in heuristics}
    for branch_index, labels in sorted(indexed.items()):
        expected_labels = {CANDIDATE_LABEL, *heuristics}
        if set(labels) != expected_labels:
            raise AnalysisError(
                f"{run_path}: branch {branch_index} labels {sorted(labels)} do not "
                f"match {sorted(expected_labels)}"
            )
        expected_seed = (
            BRANCH_SEED_BASE
            + BRANCH_SEED_PARTICLE_MULTIPLIER * particle_count
            + BRANCH_SEED_STRIDE * branch_index
        )
        seeds = {row.get("seed") for row in labels.values()}
        if len(seeds) != 1 or None in seeds:
            raise AnalysisError(
                f"{run_path}: branch {branch_index} does not use one paired seed"
            )
        try:
            actual_seed = int(next(iter(seeds)))
        except (TypeError, ValueError) as exc:
            raise AnalysisError(
                f"{run_path}: branch {branch_index} seed is not an integer"
            ) from exc
        if actual_seed != expected_seed:
            raise AnalysisError(
                f"{run_path}: branch {branch_index} seed {actual_seed} does not "
                f"match deterministic seed {expected_seed}"
            )
        candidate = labels.get(CANDIDATE_LABEL)
        if candidate is None:
            candidate_failure = True
            branch_rows.append({"branch_index": branch_index, "valid": False, "error": "missing candidate"})
            continue
        if not bool(candidate.get("all_checks_pass", False)):
            candidate_failure = True
            branch_rows.append(
                {
                    "branch_index": branch_index,
                    "valid": False,
                    "error": candidate.get("error", candidate.get("failure_class", "candidate invalid")),
                }
            )
            continue
        branch_summary: dict[str, Any] = {"branch_index": branch_index, "valid": True, "contrasts": {}}
        for heuristic in heuristics:
            baseline = labels.get(heuristic)
            if baseline is None or not bool(baseline.get("all_checks_pass", False)):
                branch_summary["valid"] = False
                comparator_failure = True
                branch_summary.setdefault("errors", []).append(f"invalid or missing comparator: {heuristic}")
                continue
            differences = _active_log_ess_differences(
                candidate,
                baseline,
                active_start=active_start,
                epsilon=epsilon,
                expected_horizon=int(fixture_identity["horizon"]),
            )
            contrasts_by_heuristic[heuristic].append(statistics.median(differences))
            branch_summary["contrasts"][heuristic] = {
                "active_log_ess_differences": differences,
                "median_active_log_ess_difference": statistics.median(differences),
                "minimum_active_log_ess_difference": min(differences),
                "positive_active_count": sum(value > 0.0 for value in differences),
                "tie_active_count": sum(value == 0.0 for value in differences),
                "active_comparison_count": len(differences),
            }
            candidate_ess = _finite_vector(candidate["ess_by_time"], label="candidate ess_by_time")
            baseline_ess = _finite_vector(baseline["ess_by_time"], label=f"{heuristic} ess_by_time")
            for offset, (candidate_value, baseline_value) in enumerate(zip(candidate_ess[active_start:], baseline_ess[active_start:]), start=active_start):
                difference = math.log(candidate_value + epsilon) - math.log(baseline_value + epsilon)
                comparison = {
                    "branch_index": branch_index,
                    "heuristic": heuristic,
                    "time_index": offset,
                    "log_ess_difference": difference,
                    "candidate_loses": difference < 0.0,
                    "decision_role": "promotion_veto" if difference < 0.0 else "diagnostic",
                }
                if difference < 0.0:
                    promotion_losses.append(comparison)
                if boundaries is not None:
                    observation_bin = fixture_bins.get(offset)
                    comparison["observation_bin"] = observation_bin
                    if difference < 0.0:
                        bin_losses.append(comparison)
        branch_rows.append(branch_summary)

    fixture_contrasts: dict[str, Any] = {}
    for heuristic, values in contrasts_by_heuristic.items():
        if values:
            fixture_contrasts[heuristic] = {
                "branch_median_log_ess_differences": values,
                "mean_branch_median_log_ess_difference": statistics.fmean(values),
                "median_branch_median_log_ess_difference": statistics.median(values),
            }
    all_branch_validity = bool(checks.get("all_branch_validity", False))
    valid = infrastructure_valid and all_branch_validity and not candidate_failure and not comparator_failure and all(
        len(values) == len(indexed) for values in contrasts_by_heuristic.values()
    )
    return {
        "path": str(run_path),
        "fixture_sha256": fixture_identity["sha256"],
        "fixture": fixture_identity,
        "replication": {
            "set_id": replication_set_id,
            "analysis_seed": analysis_seed,
            "model_seed": model_seed,
            "observation_seed": replication_observation_seed,
            "branch_seed_formula": dict(branch_seed_formula),
        },
        "status": result.get("status"),
        "valid": valid,
        "candidate_failure": candidate_failure,
        "candidate_errors": [
            row for row in branch_rows if not row.get("valid", False) and row.get("error")
        ],
        "comparator_failure": comparator_failure,
        "infrastructure_valid": infrastructure_valid,
        "workspace": workspace,
        "branch_errors": [row for row in branch_rows if row.get("errors")],
        "branch_count": len(indexed),
        "active_start": active_start,
        "fixture_contrasts": fixture_contrasts,
        "branch_summaries": branch_rows,
        "promotion_veto": bool(promotion_losses),
        "promotion_losses": promotion_losses,
        "observation_bins_checked": boundaries is not None,
        "calibration_observation_seeds": (
            list(boundaries.calibration_observation_seeds)
            if boundaries is not None
            else []
        ),
        "observation_bin_losses": bin_losses,
        "observation_bins": fixture_bins,
    }


def analyze(
    paths: Sequence[Path],
    *,
    heuristics: Sequence[str] = DEFAULT_HEURISTICS,
    expected_branch_count: int | None = 2,
    active_start: int = DEFAULT_ACTIVE_START,
    epsilon: float = DEFAULT_EPSILON,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    boundaries: BinBoundaries | None = None,
    expected_schedule_id: str = DEFAULT_SCHEDULE_ID,
    expected_schedule_ladder_id: str = DEFAULT_SCHEDULE_LADDER_ID,
    expected_particle_count: int | None = DEFAULT_PARTICLE_COUNT,
    expected_replication_set_id: str | None = DEFAULT_REPLICATION_SET_ID,
    expected_phase_id: str = DEFAULT_PHASE_ID,
    expected_analysis_seed: int = DEFAULT_BOOTSTRAP_SEED,
    expected_model_seed: int = DEFAULT_MODEL_SEED,
    expected_observation_seeds: Sequence[int] | None = DEFAULT_OBSERVATION_SEEDS,
    require_observation_bins: bool = True,
) -> dict[str, Any]:
    if not paths:
        raise AnalysisError("at least one replication path is required")
    if not heuristics or CANDIDATE_LABEL in heuristics:
        raise AnalysisError("heuristics must be nonempty and exclude the candidate")
    if expected_branch_count is not None and int(expected_branch_count) < 1:
        raise AnalysisError("expected_branch_count must be positive or None")
    if require_observation_bins and boundaries is None:
        raise AnalysisError(
            "Phase 8E analysis requires frozen observation-bin boundaries"
        )
    fixture_rows = []
    parse_errors = []
    fixture_hashes: set[str] = set()
    for raw_path in paths:
        run_path = Path(raw_path).resolve()
        result_path = run_path / "result.json"
        try:
            result = _load_json(result_path)
            fixture_row = _fixture_row(
                    run_path,
                    result,
                    heuristics=tuple(heuristics),
                    active_start=int(active_start),
                    epsilon=float(epsilon),
                    boundaries=boundaries,
                    expected_schedule_id=str(expected_schedule_id),
                    expected_schedule_ladder_id=str(expected_schedule_ladder_id),
                    expected_particle_count=expected_particle_count,
                    expected_replication_set_id=expected_replication_set_id,
                    expected_phase_id=str(expected_phase_id),
                    expected_analysis_seed=int(expected_analysis_seed),
                    expected_model_seed=int(expected_model_seed),
                )
            if (
                expected_branch_count is not None
                and int(fixture_row["branch_count"]) != int(expected_branch_count)
            ):
                raise AnalysisError(
                    f"expected {int(expected_branch_count)} branches, found "
                    f"{fixture_row['branch_count']}"
                )
            fixture_hash = fixture_row.get("fixture_sha256")
            if fixture_hash is not None:
                if str(fixture_hash) in fixture_hashes:
                    raise AnalysisError(f"duplicate fixture hash {fixture_hash}")
                fixture_hashes.add(str(fixture_hash))
            fixture_rows.append(fixture_row)
        except AnalysisError as exc:
            parse_errors.append({"path": str(run_path), "error": str(exc)})

    observation_seeds = [
        int(row["fixture"]["observation_seed"])
        for row in fixture_rows
        if row["fixture"].get("observation_seed") is not None
    ]
    if len(observation_seeds) != len(set(observation_seeds)):
        parse_errors.append(
            {"path": "<replication-set>", "error": "duplicate observation seed"}
        )
    if expected_observation_seeds is not None:
        expected_seed_set = {int(value) for value in expected_observation_seeds}
        observed_seed_set = set(observation_seeds)
        if observed_seed_set != expected_seed_set:
            parse_errors.append(
                {
                    "path": "<replication-set>",
                    "error": (
                        "observation seed set mismatch: expected "
                        f"{sorted(expected_seed_set)}, observed {sorted(observed_seed_set)}"
                    ),
                }
            )
    if boundaries is not None and expected_observation_seeds is not None:
        overlap = sorted(
            set(int(value) for value in boundaries.calibration_observation_seeds)
            & {int(value) for value in expected_observation_seeds}
        )
        if overlap:
            parse_errors.append(
                {
                    "path": "<replication-set>",
                    "error": (
                        "calibration and claim observation seeds overlap: "
                        f"{overlap}"
                    ),
                }
            )

    valid_rows = [row for row in fixture_rows if bool(row["valid"])]
    statistical: dict[str, Any] = {}
    for heuristic in heuristics:
        values = [
            float(row["fixture_contrasts"][heuristic]["mean_branch_median_log_ess_difference"])
            for row in valid_rows
            if heuristic in row["fixture_contrasts"]
        ]
        complete = len(values) == len(valid_rows) and len(valid_rows) == len(fixture_rows) and not parse_errors
        entry: dict[str, Any] = {
            "fixture_values": values,
            "complete_valid_fixture_set": complete,
            "heuristic": heuristic,
        }
        if values:
            entry["descriptive"] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        if complete:
            entry["bootstrap"] = _bootstrap_mean(
                values, resamples=bootstrap_resamples, seed=int(bootstrap_seed)
            )
            entry["sign_test"] = _sign_test(values)
            entry["statistically_nominated"] = bool(
                entry["bootstrap"]["percentile_95"]["lower"] > 0.0
            )
        else:
            entry["bootstrap"] = None
            entry["sign_test"] = None
            entry["statistically_nominated"] = False
        statistical[heuristic] = entry

    all_valid = len(fixture_rows) == len(paths) and bool(fixture_rows) and all(
        bool(row["valid"]) for row in fixture_rows
    ) and not parse_errors
    promotion_veto = any(bool(row["promotion_veto"]) for row in fixture_rows)
    bin_veto = any(bool(row["observation_bin_losses"]) for row in fixture_rows)
    statistical_nomination = bool(
        all_valid
        and not promotion_veto
        and not bin_veto
        and all(bool(entry["statistically_nominated"]) for entry in statistical.values())
    )
    if parse_errors:
        status = "BLOCKED_ARTIFACT_PARSE"
    elif not all_valid:
        status = "VETO_PHASE8E_VALIDITY"
    elif promotion_veto or bin_veto:
        status = "PASS_VALIDITY_WITH_PROMOTION_VETO"
    elif statistical_nomination:
        status = "NOMINATED_STATISTICAL_CONTRAST_DIAGNOSTIC"
    else:
        status = "PASS_VALIDITY_DESCRIPTIVE_ONLY"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "paths": [str(Path(path).resolve()) for path in paths],
        "fixture_count": len(paths),
        "parsed_fixture_count": len(fixture_rows),
        "valid_fixture_count": len(valid_rows),
        "parse_errors": parse_errors,
        "active_start": int(active_start),
        "epsilon": float(epsilon),
        "heuristics": list(heuristics),
        "expected_branch_count": expected_branch_count,
        "expected_schedule_id": str(expected_schedule_id),
        "expected_schedule_ladder_id": str(expected_schedule_ladder_id),
        "expected_particle_count": expected_particle_count,
        "expected_replication_set_id": expected_replication_set_id,
        "expected_phase_id": str(expected_phase_id),
        "expected_analysis_seed": int(expected_analysis_seed),
        "expected_model_seed": int(expected_model_seed),
        "expected_observation_seeds": (
            None
            if expected_observation_seeds is None
            else [int(value) for value in expected_observation_seeds]
        ),
        "analyzer_path": str(Path(__file__).resolve().relative_to(ROOT)),
        "analyzer_sha256": _sha256_file(Path(__file__).resolve()),
        "observation_bin_boundaries": boundaries.payload() if boundaries is not None else None,
        "salient_times": list(SALIENT_TIMES),
        "fixture_rows": fixture_rows,
        "statistical": statistical,
        "heuristic_promotion_veto": promotion_veto,
        "observation_bin_promotion_veto": bin_veto,
        "statistical_nomination": statistical_nomination,
        "evidence_contract": {
            "primary_estimand": "fixture-cluster mean of branch-averaged median active-time log-ESS contrast",
            "resampling_unit": "independent observation fixture",
            "promotion_veto": "any active-time or conditional-observation-bin candidate loss",
            "nonclaims": [
                "no posterior correctness",
                "no unbiased likelihood",
                "no global Newton convergence",
                "no general-model, HMC, production, or default claim",
            ],
        },
    }


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# C2 Phase 8E Statistical Replication Analysis",
        "",
        f"Status: `{result['status']}`",
        "",
        f"Fixtures: `{result['valid_fixture_count']}/{result['fixture_count']}` valid",
        f"Promotion veto: `{bool(result['heuristic_promotion_veto'] or result['observation_bin_promotion_veto'])}`",
        "",
        "## Inference status",
        "",
        "| Comparator | Fixtures | Mean contrast | Bootstrap 95% interval | Sign p-value | Nomination |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for heuristic, entry in result["statistical"].items():
        descriptive = entry.get("descriptive", {})
        bootstrap = entry.get("bootstrap")
        sign_test = entry.get("sign_test")
        interval = "n/a"
        if bootstrap is not None:
            interval = f"[{bootstrap['percentile_95']['lower']:.6g}, {bootstrap['percentile_95']['upper']:.6g}]"
        p_value = "n/a" if sign_test is None or sign_test["two_sided_exact_p_value"] is None else f"{sign_test['two_sided_exact_p_value']:.6g}"
        mean = "n/a" if "mean" not in descriptive else f"{descriptive['mean']:.6g}"
        lines.append(
            f"| {heuristic} | {len(entry['fixture_values'])} | {mean} | {interval} | {p_value} | {bool(entry['statistically_nominated'])} |"
        )
    lines += [
        "",
        "The resampling unit is the independent observation fixture.  ESS and",
        "all differences are descriptive diagnostics; the route remains",
        "candidate-only unless the declared validity, heuristic, and statistical",
        "conditions are all satisfied.",
        "",
        "## Fixture rows",
        "",
        "| Fixture | Valid | Branches | Promotion veto | Status |",
        "| --- | :---: | ---: | :---: | --- |",
    ]
    for row in result["fixture_rows"]:
        lines.append(
            f"| `{row['fixture_sha256']}` | {bool(row['valid'])} | {row['branch_count']} | "
            f"{bool(row['promotion_veto'] or row['observation_bin_losses'])} | {row['status']} |"
        )
    if result["parse_errors"]:
        lines += ["", "## Parse errors", ""]
        lines.extend(f"- `{row['path']}`: {row['error']}" for row in result["parse_errors"])
    return "\n".join(lines) + "\n"


def _read_boundaries(
    path: Path, *, expected_model_seed: int | None = None
) -> BinBoundaries:
    payload = _load_json(path)
    if payload.get("schema_version") != "c2_phase8e_observation_bin_boundaries_v1":
        raise AnalysisError("unexpected observation-bin boundary schema")
    if payload.get("statistic") != "mean_absolute_observation_per_time":
        raise AnalysisError("unexpected observation-bin statistic")
    if payload.get("quantile_method") != "linear":
        raise AnalysisError("observation-bin boundaries must declare linear quantiles")
    probabilities = payload.get("quantile_probabilities")
    if (
        not isinstance(probabilities, Sequence)
        or isinstance(probabilities, (str, bytes))
        or len(probabilities) != 2
    ):
        raise AnalysisError(
            "observation-bin boundaries must declare two quantile probabilities"
        )
    probability_values = [
        _finite_float(value, label=f"quantile_probabilities[{index}]")
        for index, value in enumerate(probabilities)
    ]
    expected_probabilities = (1.0 / 3.0, 2.0 / 3.0)
    if any(
        not math.isclose(value, expected, rel_tol=0.0, abs_tol=1.0e-15)
        for value, expected in zip(probability_values, expected_probabilities)
    ):
        raise AnalysisError(
            "observation-bin boundaries must use one-third and two-thirds quantiles"
        )
    try:
        declared_model_seed = int(payload["model_seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError("observation-bin boundary model seed is missing") from exc
    if expected_model_seed is not None and declared_model_seed != int(expected_model_seed):
        raise AnalysisError("observation-bin boundary model seed mismatch")
    declared_seeds = payload.get("calibration_observation_seeds")
    if (
        not isinstance(declared_seeds, Sequence)
        or isinstance(declared_seeds, (str, bytes))
        or not declared_seeds
    ):
        raise AnalysisError(
            "observation-bin boundaries must declare calibration observation seeds"
        )
    calibration_seeds = []
    for index, value in enumerate(declared_seeds):
        try:
            calibration_seeds.append(int(value))
        except (TypeError, ValueError) as exc:
            raise AnalysisError(
                f"calibration_observation_seeds[{index}] is not an integer"
            ) from exc
    if len(set(calibration_seeds)) != len(calibration_seeds):
        raise AnalysisError("calibration observation seeds must be unique")
    partition = payload.get("calibration_partition")
    if (
        not isinstance(partition, Sequence)
        or isinstance(partition, (str, bytes))
        or len(partition) != len(calibration_seeds)
    ):
        raise AnalysisError(
            "observation-bin boundaries must list one source per calibration seed"
        )
    observed_seeds = []
    source_hashes: set[str] = set()
    all_values: list[float] = []
    common_horizon: int | None = None
    common_dimension: int | None = None
    for index, entry in enumerate(partition):
        if not isinstance(entry, Mapping):
            raise AnalysisError(f"calibration_partition[{index}] must be an object")
        fixture_value = entry.get("path")
        declared_sha256 = str(entry.get("sha256", ""))
        if not fixture_value or len(declared_sha256) != 64:
            raise AnalysisError(
                f"calibration_partition[{index}] must declare path and SHA-256"
            )
        fixture_path = _resolve_path(fixture_value, base=path.parent)
        actual_sha256 = _checked_sha256_file(
            fixture_path, label=f"calibration fixture source {fixture_path}"
        )
        if actual_sha256 != declared_sha256:
            raise AnalysisError(
                f"calibration fixture SHA-256 mismatch for {fixture_path}: "
                f"declared {declared_sha256}, actual {actual_sha256}"
            )
        if actual_sha256 in source_hashes:
            raise AnalysisError("calibration partition contains duplicate fixtures")
        source_hashes.add(actual_sha256)
        fixture = _load_json(fixture_path)
        try:
            fixture_model_seed = int(fixture["model_seed"])
            fixture_observation_seed = int(fixture["observation_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError(
                f"calibration fixture {fixture_path} lacks integer seeds"
            ) from exc
        if fixture_model_seed != declared_model_seed:
            raise AnalysisError(
                f"calibration fixture {fixture_path} model seed mismatch"
            )
        try:
            declared_entry_seed = int(entry["observation_seed"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AnalysisError(
                f"calibration_partition[{index}] observation seed is missing"
            ) from exc
        if fixture_observation_seed != declared_entry_seed:
            raise AnalysisError(
                f"calibration fixture {fixture_path} observation seed mismatch"
            )
        observed_seeds.append(fixture_observation_seed)
        observations = fixture.get("observations")
        if not isinstance(observations, Sequence) or isinstance(
            observations, (str, bytes)
        ) or not observations:
            raise AnalysisError(f"calibration fixture {fixture_path} has no observations")
        horizon = len(observations)
        if common_horizon is None:
            common_horizon = horizon
        elif horizon != common_horizon:
            raise AnalysisError("calibration fixtures have different horizons")
        for time_index, row in enumerate(observations):
            values = _finite_vector(
                row, label=f"{fixture_path}:observations[{time_index}]"
            )
            if common_dimension is None:
                common_dimension = len(values)
            elif len(values) != common_dimension:
                raise AnalysisError("calibration fixtures have different dimensions")
            all_values.append(statistics.fmean(abs(value) for value in values))
    if tuple(observed_seeds) != tuple(calibration_seeds):
        raise AnalysisError(
            "calibration partition order does not match calibration observation seeds"
        )
    expected_near = _percentile(all_values, probability_values[0])
    expected_ordinary = _percentile(all_values, probability_values[1])
    declared_near = _finite_float(
        payload.get("near_zero_upper"), label="near_zero_upper"
    )
    declared_ordinary = _finite_float(
        payload.get("ordinary_upper"), label="ordinary_upper"
    )
    if not math.isclose(
        declared_near, expected_near, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise AnalysisError(
            "near_zero_upper does not match the frozen calibration partition"
        )
    if not math.isclose(
        declared_ordinary, expected_ordinary, rel_tol=1.0e-12, abs_tol=1.0e-12
    ):
        raise AnalysisError(
            "ordinary_upper does not match the frozen calibration partition"
        )
    return BinBoundaries(
        near_zero_upper=declared_near,
        ordinary_upper=declared_ordinary,
        source_path=str(path),
        source_sha256=_checked_sha256_file(path, label="boundary artifact"),
        calibration_observation_seeds=tuple(calibration_seeds),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--active-start", type=int, default=DEFAULT_ACTIVE_START)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--expected-analysis-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="analysis seed recorded by each Phase 8E runner",
    )
    parser.add_argument("--expected-branch-count", type=int, default=2)
    parser.add_argument("--expected-schedule-id", default=DEFAULT_SCHEDULE_ID)
    parser.add_argument(
        "--expected-schedule-ladder-id", default=DEFAULT_SCHEDULE_LADDER_ID
    )
    parser.add_argument("--expected-particle-count", type=int, default=DEFAULT_PARTICLE_COUNT)
    parser.add_argument(
        "--expected-replication-set-id", default=DEFAULT_REPLICATION_SET_ID
    )
    parser.add_argument("--expected-phase-id", default=DEFAULT_PHASE_ID)
    parser.add_argument("--expected-model-seed", type=int, default=DEFAULT_MODEL_SEED)
    parser.add_argument(
        "--expected-observation-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_OBSERVATION_SEEDS),
    )
    parser.add_argument(
        "--allow-variable-branch-count",
        action="store_true",
        help="diagnostic exception: do not require the Phase 8E two-branch design",
    )
    parser.add_argument("--observation-bin-boundaries")
    parser.add_argument(
        "--allow-missing-observation-bins",
        action="store_true",
        help="diagnostic exception: disable the conditional observation-bin gate",
    )
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    boundaries = (
        _read_boundaries(
            Path(args.observation_bin_boundaries).resolve(),
            expected_model_seed=int(args.expected_model_seed),
        )
        if args.observation_bin_boundaries
        else None
    )
    result = analyze(
        [Path(value).resolve() for value in args.paths],
        expected_branch_count=(None if args.allow_variable_branch_count else int(args.expected_branch_count)),
        expected_schedule_id=str(args.expected_schedule_id),
        expected_schedule_ladder_id=str(args.expected_schedule_ladder_id),
        expected_particle_count=int(args.expected_particle_count),
        expected_replication_set_id=(
            None
            if args.expected_replication_set_id == ""
            else str(args.expected_replication_set_id)
        ),
        expected_phase_id=str(args.expected_phase_id),
        expected_analysis_seed=int(args.expected_analysis_seed),
        expected_model_seed=int(args.expected_model_seed),
        expected_observation_seeds=tuple(args.expected_observation_seeds),
        active_start=int(args.active_start),
        epsilon=float(args.epsilon),
        bootstrap_resamples=int(args.bootstrap_resamples),
        bootstrap_seed=int(args.bootstrap_seed),
        boundaries=boundaries,
        require_observation_bins=not bool(args.allow_missing_observation_bins),
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = (ROOT / output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({"output": str(output), "status": result["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

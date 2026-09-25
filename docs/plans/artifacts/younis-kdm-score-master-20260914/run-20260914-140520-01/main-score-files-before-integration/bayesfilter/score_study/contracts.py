"""Typed, backend-free scientific contracts for the score master.

Registry entries describe actual providers. A planned provider remains blocked
until its producer phase supplies the endpoint and its identity checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Model:
    id: str
    support: str
    measure: str
    initial_parameter_dependence: bool
    oracle: str


@dataclass(frozen=True)
class Target:
    id: str
    support: tuple[str, ...]
    normalization: str


@dataclass(frozen=True)
class Proposal:
    id: str
    support: tuple[str, ...]
    law: str
    status: str = "active"
    prerequisite: str = ""


@dataclass(frozen=True)
class Estimator:
    id: str
    target: str
    proposals: tuple[str, ...]
    endpoint: str | None
    source_paths: tuple[str, ...]
    includes_initial_terms: bool
    derivative: str = "analytical_recursion"
    prerequisite: str = ""


@dataclass
class Registry:
    models: dict[str, Model]
    targets: dict[str, Target]
    proposals: dict[str, Proposal]
    estimators: dict[str, Estimator]
    tuning: dict[str, str] = field(default_factory=dict)


def seed_pair(*, master_seed: int, model: str, dataset: int, replicate: int,
              stream: str, coupling_group: str, perturbation: str = "shared") -> list[int]:
    """Stateless TF seed; scheduling and method order cannot change a stream.

    Coupled perturbations deliberately use the same perturbation identifier;
    independent nodes must supply distinct identifiers. Both have correct
    marginal stateless random laws. Data streams use a separate stream name.
    """
    key = [master_seed, model, dataset, replicate, stream, coupling_group, perturbation]
    raw = hashlib.sha256(json.dumps(key, separators=(",", ":")).encode()).digest()
    return [int.from_bytes(raw[k:k+4], "big") & 0x7fffffff for k in (0, 4)]


def validate_result(result: dict, row: dict, registry: Registry) -> None:
    required = {"value", "score", "oracle_value", "oracle_score", "value_target",
                "derivative_target", "comparison_target", "derivative_id",
                "proposal_law", "initial_terms", "numerical_validity", "inference_status",
                "runtime", "diagnostics"}
    if missing := required - result.keys():
        raise ValueError(f"incomplete numerical result: {sorted(missing)}")
    estimator = registry.estimators[row["estimator"]]
    if result["derivative_target"] != estimator.target:
        raise ValueError("computed derivative target differs from registered target")
    if result["comparison_target"] != row["comparison_target"]:
        raise ValueError("comparison target changed")
    if result["derivative_id"] != estimator.derivative:
        raise ValueError("derivative implementation differs from registry")
    if result["proposal_law"] != registry.proposals[row["proposal"]].law:
        raise ValueError("sampler/denominator law differs from registry")
    if registry.models[row["model"]].initial_parameter_dependence and not result["initial_terms"]:
        raise ValueError("parameter-dependent initial law is missing from the derivative")
    if result["numerical_validity"] != "pass":
        raise ValueError("numerical validity veto")
    if not isinstance(result["score"], list) or not result["score"]:
        raise ValueError("score must be a nonempty vector")
    if len(result["score"]) != len(result["oracle_score"]):
        raise ValueError("oracle/score dimensions differ")
    for v in [result["value"], result["oracle_value"], *result["score"], *result["oracle_score"]]:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError("non-finite or nonnumeric value/score")
    if row.get("role", "mechanics") == "mechanics" and result["inference_status"] != "mechanics_only":
        raise ValueError("mechanics run cannot issue a scientific inference")
    json.dumps(result, allow_nan=False)


def validate_study(study: dict, registry: Registry) -> list[dict]:
    if study.get("schema") != "younis_score_study_v1":
        raise ValueError("unknown study schema")
    if not study.get("phase") or not study.get("plan") or not study.get("version"):
        raise ValueError("phase, plan and version are required")
    budget = study.get("budget", {})
    for key in ("wall_seconds", "max_attempts", "max_attempts_per_row"):
        if not isinstance(budget.get(key), (int, float)) or budget[key] <= 0:
            raise ValueError(f"positive budget required: {key}")
    if "settings" in study:
        settings = study["settings"]
        for key in ("dimension", "observation_dimension", "horizon", "particles"):
            if type(settings.get(key)) is not int or settings[key] <= 0:
                raise ValueError(f"positive integer setting required: {key}")
        if settings.get("device") not in ("CPU", "GPU") or settings.get("dtype") not in ("float32", "float64"):
            raise ValueError("unsupported execution scope")
        if type(settings.get("jit_compile")) is not bool or type(settings.get("tf32")) is not bool:
            raise ValueError("JIT/TF32 policy must be explicit")
        if settings["tf32"] and settings["dtype"] != "float32":
            raise ValueError("TF32 requires float32")
        if settings["device"] == "GPU" and not settings["jit_compile"] and not study.get("reference_exception"):
            raise ValueError("GPU candidate requires XLA or an explicit reference exception")
        for key in ("theta", "data_theta"):
            vector = settings.get(key, [])
            if len(vector) != 6 or not all(type(x) in (int, float) and math.isfinite(x) for x in vector):
                raise ValueError("Gaussian fixture requires six finite model parameters")
    partitions = study.get("partitions", {})
    groups = [set(partitions.get(k, [])) for k in ("calibration", "validation", "claim")]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
        raise ValueError("calibration, validation and claim data overlap")
    rows = study.get("rows", [])
    ids = [r["id"] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("rows must have distinct ids")
    for row_id in ids:
        if not row_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in row_id):
            raise ValueError("row id must be an ordinary filename component")
    missing = set(study.get("required_proposals", [])) - {r["proposal"] for r in rows}
    if missing:
        raise ValueError(f"missing requested methods: {sorted(missing)}")
    decisions = []
    for row in rows:
        for key, table in (("model", registry.models), ("proposal", registry.proposals),
                           ("estimator", registry.estimators), ("comparison_target", registry.targets)):
            if row.get(key) not in table:
                raise ValueError(f"unknown {key}: {row.get(key)}")
        model = registry.models[row["model"]]
        proposal = registry.proposals[row["proposal"]]
        estimator = registry.estimators[row["estimator"]]
        if estimator.target not in registry.targets or row["proposal"] not in estimator.proposals:
            raise ValueError("incompatible estimator/proposal")
        if any(model.support not in supports for supports in
               (proposal.support, registry.targets[estimator.target].support,
                registry.targets[row["comparison_target"]].support)):
            raise ValueError("incompatible support/target/reference measure")
        if model.initial_parameter_dependence and not estimator.includes_initial_terms:
            raise ValueError("estimator omits parameter-dependent initialization")
        if estimator.target != row["comparison_target"] and row.get("comparison") != "approximation_error":
            raise ValueError("different mathematical targets need explicit approximation-error comparison")
        if row.get("role", "mechanics") not in ("mechanics", "calibration", "validation", "claim"):
            raise ValueError("unknown row role")
        if row.get("role") == "claim" and not all(groups):
            raise ValueError("claim rows require disjoint nonempty tuning and claim partitions")
        role = row.get("role", "mechanics")
        if role != "mechanics" and row.get("dataset") not in partitions.get(role, []):
            raise ValueError("row dataset is outside its declared partition")
        if role == "claim" and row["proposal"] == "ledh" and not row.get("tuning_selection"):
            raise ValueError("claim LEDH requires a repository-issued tuning selection")
        dependencies = row.get("depends_on", [])
        if set(dependencies) - set(ids):
            raise ValueError("unknown prerequisite row")
        reasons = []
        status = "runnable"
        if proposal.status == "deferred":
            status = "deferred"
            reasons.append(proposal.prerequisite)
        elif proposal.status == "blocked" or estimator.endpoint is None:
            status = "blocked"
            reasons.append(proposal.prerequisite or estimator.prerequisite or "endpoint not implemented")
        decisions.append({"id": row["id"], "status": status, "reasons": reasons})
    pending = {r["id"]: set(r.get("depends_on", [])) for r in rows}
    while pending:
        ready = {k for k, v in pending.items() if not v & pending.keys()}
        if not ready:
            raise ValueError("cyclic row dependencies")
        pending = {k: v for k, v in pending.items() if k not in ready}
    return decisions

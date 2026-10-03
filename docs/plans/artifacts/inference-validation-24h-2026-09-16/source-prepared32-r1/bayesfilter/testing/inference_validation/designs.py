"""Explicit fixed experimental designs; no implicit numerical execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any

from .catalog import get_target

ENGINES = ("mechanics", "invariance", "search", "sbc", "accuracy", "stopping", "power")
ROUTES = ("frozen", "ordinary", "prepared", "fixed_transport", "reference", "controller", "external")
CONTROLS = ("baseline", "noop", "wrong_score", "wrong_metric", "omit_jacobian", "ignore_data",
            "identity", "two_cycle", "wrong_energy", "duplicate_stream", "warmup_leak",
            "drop_candidate", "cross_l_epsilon", "lost_chunk")


def digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seed_for(root: int, *parts: Any) -> tuple[int, int]:
    raw = hashlib.sha256(json.dumps([root, *parts], separators=(",", ":")).encode()).digest()
    # Leave room for the native controller's bounded per-chunk offsets.
    return (int.from_bytes(raw[:4], "big") % (2**30), int.from_bytes(raw[4:8], "big") % (2**30))


@dataclass(frozen=True)
class ScenarioSpec:
    target: str
    route: str
    control: str = "baseline"
    start: str = "dispersed"
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        get_target(self.target)
        if self.route not in ROUTES or self.control not in CONTROLS:
            raise ValueError("unknown procedure or control")
        if self.start not in {"dispersed", "remote", "single_mode", "reference"}:
            raise ValueError("unknown start regime")
        digest(self.parameters)  # finite, serializable, stable experiment identity


@dataclass(frozen=True)
class ValidationDesign:
    design_id: str
    engine: str
    scenario: ScenarioSpec
    replications: int
    draws: int
    seed: int
    budget_seconds: float
    purpose: str
    numerical_provenance: str
    device: str = "gpu"
    alpha: float = 0.05
    null_draws: int = 1999
    rank_draws: int = 7
    step_size: float = 0.4
    leapfrog_steps: int = 5
    member_l: int = 3
    l_grid: tuple[int, ...] = (3, 5, 9, 13, 18, 25)
    measurement_draws: int = 128
    posterior_cap: int = 1024
    mcse_tolerance: float = 0.1
    accuracy_tolerance: float = 0.25
    multiplicity: int = 1
    phase: str = "development"
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.engine not in ENGINES or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*",self.design_id):
            raise ValueError("invalid design identity/engine")
        if self.device not in {"gpu", "cpu_reference"} or self.phase not in {"development", "confirmation"}:
            raise ValueError("explicit supported device and phase required")
        for name in ("replications", "draws", "null_draws", "rank_draws", "leapfrog_steps",
                     "member_l", "measurement_draws", "posterior_cap", "multiplicity"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("budget_seconds", "step_size", "mcse_tolerance", "accuracy_tolerance"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0 < self.alpha < 1 or not self.purpose or not self.numerical_provenance:
            raise ValueError("test size, question and numerical provenance required")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not self.l_grid or len(set(self.l_grid)) != len(self.l_grid) or any(type(x) is not int or not 1 <= x <= 25 for x in self.l_grid):
            raise ValueError("distinct L values in [1,25] required")
        if self.measurement_draws < 64:
            raise ValueError("candidate evidence requires at least 64 draws")
        if self.options.get("posterior_members", "all") not in {"all", "selected"}:
            raise ValueError("posterior_members must be all or selected")
        if self.options.get("member_rule", "declared_l_first") not in {"declared_l_first", "first_verified"}:
            raise ValueError("unsupported predeclared member rule")
        native=self.options.get("native_search",False)
        if type(native) is not bool or (native and (self.scenario.route!="ordinary" or "search" in self.options)):
            raise ValueError("native_search requires ordinary preparation without a search override")
        if native and self.l_grid!=(3,5,9,13,18,25):
            raise ValueError("native_search records the native broad L grid")
        count_options = {"warmup_chunk_results", "warmup_min_results", "warmup_check_window_results",
                         "warmup_max_results", "retained_chunk_results", "retained_min_results", "retained_max_results"}
        posterior = self.options.get("posterior_settings", {})
        if not isinstance(posterior, dict) or set(posterior) - count_options:
            raise ValueError("posterior_settings supports explicit count controls only")
        if any(type(v) is not int or not 4 <= v <= 10000 for v in posterior.values()):
            raise ValueError("posterior counts must be integers in [4,10000]")
        fixed = self.options.get("fixed_comparator")
        if fixed is not None:
            if (self.engine != "stopping" or self.scenario.route not in {"ordinary", "prepared", "fixed_transport"}
                    or not isinstance(fixed, dict) or set(fixed) != {"warmup_results", "retained_results"}
                    or any(type(v) is not int or not 4 <= v <= 10000 for v in fixed.values())):
                raise ValueError("fixed comparator requires numerical stopping and declared counts in [4,10000]")
        if self.engine == "sbc":
            if not get_target(self.scenario.target).generative:
                raise ValueError("SBC requires a proper generative model")
            if self.scenario.route not in {"ordinary", "prepared", "fixed_transport", "reference"}:
                raise ValueError("SBC requires complete fits or an explicit reference control")
            if self.scenario.start == "reference":
                raise ValueError("generating truth/reference starts forbidden for full SBC")
            if self.options.get("member_rule", "declared_l_first") == "declared_l_first" and self.member_l not in self.l_grid:
                raise ValueError("SBC member group must be declared in the L grid")
        if self.engine == "invariance" and self.scenario.route != "frozen":
            raise ValueError("invariance requires a frozen kernel, not an adapting procedure")
        if self.engine == "mechanics" and self.scenario.route != "frozen":
            raise ValueError("mechanics covers a frozen numerical component only")
        if self.engine == "power":
            child = self.options.get("calibration_design")
            if child is None:
                if self.scenario.route != "reference" or self.scenario.target != "normal_conjugate":
                    raise ValueError("primitive power calibration requires the normal_conjugate reference")
            else:
                if child.get("engine") == "power":
                    raise ValueError("nested power engines are forbidden")
                nested = ValidationDesign.from_payload(child)
                if (nested.scenario.target,nested.scenario.route,nested.device) != (self.scenario.target,self.scenario.route,self.device):
                    raise ValueError("power scenario must match the experiment it repeats")
        supported_controls = {
            "mechanics": {"baseline", "noop", "wrong_score", "wrong_metric", "omit_jacobian"},
            "invariance": {"baseline", "noop", "wrong_score", "omit_jacobian", "identity", "two_cycle", "wrong_energy", "duplicate_stream"},
            "search": {"baseline", "noop", "drop_candidate", "cross_l_epsilon", "lost_chunk"},
            "accuracy": {"baseline", "noop", "ignore_data", "omit_jacobian", "warmup_leak", "lost_chunk", "duplicate_stream"},
            "stopping": {"baseline", "noop", "warmup_leak", "lost_chunk"},
            "sbc": {"baseline", "noop", "ignore_data", "omit_jacobian"},
            "power": {"baseline"},
        }
        if self.scenario.control not in supported_controls[self.engine]:
            raise ValueError("control is not implemented by this experiment engine")
        if self.scenario.control == "omit_jacobian" and get_target(self.scenario.target).support not in {"positive", "unit_interval", "simplex3"}:
            raise ValueError("omitted Jacobian requires constrained coordinates")
        if self.scenario.control == "ignore_data" and not get_target(self.scenario.target).generative:
            raise ValueError("ignored-data control requires a generative target")
        if self.scenario.control == "two_cycle" and self.scenario.target != "gaussian":
            raise ValueError("two-cycle control requires the symmetric Gaussian law")
        if self.scenario.control == "wrong_metric" and self.scenario.target != "gaussian":
            raise ValueError("metric oracle currently uses the Gaussian target")
        if self.scenario.route == "reference" and self.engine == "sbc" and self.scenario.control not in {"baseline", "noop", "ignore_data"}:
            raise ValueError("unsupported reference SBC control")
        if self.scenario.route == "reference" and self.engine == "stopping":
            expected = "mixture" if self.options.get("array_regime") == "missed_mode" else "gaussian"
            if self.scenario.target != expected or self.scenario.control not in {"baseline", "noop"}:
                raise ValueError("reference diagnostic arrays require their declared Gaussian or mixture target")
        if self.scenario.route == "controller" and self.scenario.target != "gaussian":
            raise ValueError("scripted controller fixture has no numerical target coverage")
        if self.scenario.route == "external" and (self.engine != "accuracy" or self.scenario.control != "baseline"):
            raise ValueError("external observations support reference accuracy only")
        if self.engine in {"search", "accuracy", "stopping"} and self.scenario.route not in {"ordinary", "prepared", "fixed_transport", "controller", "external"} and not (self.engine=="stopping" and self.scenario.route=="reference"):
            raise ValueError("pipeline experiment requires an explicit numerical/contract route")
        if self.scenario.route == "controller" and self.engine != "search":
            raise ValueError("controller double cannot establish posterior/numerical coverage")
        if self.engine in {"sbc", "invariance", "power"} and 1 / (self.null_draws + 1) > self.alpha / self.multiplicity:
            raise ValueError("Monte Carlo null resolution exceeds adjusted test threshold")
        digest(self.options)

        # Primitive null resolution includes every declared test quantity.
        width = len(get_target(self.scenario.target).parameters)
        quantities = width + 1 + int(width > 1) + int(self.engine == "sbc")
        required_family = 2 * quantities if self.engine == "invariance" else quantities
        if self.engine in {"sbc", "invariance"} and 1 / (self.null_draws + 1) > self.alpha / max(self.multiplicity, required_family):
            raise ValueError("null resolution insufficient for all declared quantities")

    def payload(self):
        return asdict(self)

    @property
    def identity(self):
        return digest(self.payload())

    @classmethod
    def from_payload(cls, payload):
        data = dict(payload)
        data["scenario"] = ScenarioSpec(**data["scenario"])
        if "l_grid" in data:
            data["l_grid"] = tuple(data["l_grid"])
        return cls(**data)

    def coverage_key(self):
        return {"engine": self.engine, "target": self.scenario.target,
                "family": get_target(self.scenario.target).family, "route": self.scenario.route,
                "control": self.scenario.control, "start": self.scenario.start,
                "device": self.device, "phase": self.phase,
                "transport": (self.options.get("transport_payload", {}).get("schema",
                              "bayesfilter.neutra.frozen_affine_diag.v1")
                              if self.scenario.route=="fixed_transport" else "none")}


def resolve_suite(payload):
    if payload.get("schema") != "bayesfilter.inference_validation_suite.v1":
        raise ValueError("unsupported suite schema")
    designs = [ValidationDesign.from_payload(p) for p in payload["designs"]]
    if len({d.design_id for d in designs}) != len(designs):
        raise ValueError("duplicate design ids")
    allowed = payload.get("profiles", {}).get(payload.get("profile"))
    if allowed is None:
        raise ValueError("suite profile must explicitly list permitted engines")
    if any(d.engine not in allowed for d in designs):
        raise ValueError("design outside the selected profile")
    return designs

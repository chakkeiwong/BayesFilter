"""Explicit q20 development hypotheses, validated before numerical work.

Numerical provenance and failure diagnostics are in the 2026-09-15 parameter
ledger and mathematical audit. These values define an experiment, not a claim
that the posterior has been whitened or that its accuracy has been established.
This module deliberately imports only the standard library.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


SCHEMA = "bayesfilter.q20.production_protocol.v1"
PARAMETERS = ("latent_mean_weight.0.0", "latent_mean_bias.0", "observation_weight.0.0", "observation_bias.0")
METHODS = ("identity", "classical", "neutra", "ensemble")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def frozen_scope_hash(config):
    """Confirmation reuses frozen development choices with fresh role seeds."""
    return digest({**config, "role": "development"})


def write_json(path, payload, *, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def protocol_template():
    """All counts are ledger hypotheses unless explicitly labeled otherwise."""
    return {
        "schema": SCHEMA, "role": "development", "cpu_reference": False,
        "jit_compile": True, "seed": [20260915, 150001],
        "provenance": "docs/plans/bayesfilter-ssl-lstm-q20-production-parameter-ledger-2026-09-15.md",
        "target": {"q": 20, "horizon": 30, "dimension": 4, "dtype": "float64",
                   "principal_sqrt_backend": "tensorflow_eigh_strict",
                   "prior_center": [.35, -.08, .65, .05], "prior_sd": 4.0,
                   "ukf_alpha": 1.0, "ukf_beta": 2.0, "ukf_kappa": 0.0},
        "training": {"widths": [16, 32], "learning_rates": [.0005, .001],
                     "roots": [0, 1, 2], "stages": 2, "activation": "tanh",
                     "batch_size": 32, "pricing_batches": [8, 32, 128],
                     "s_max": 2.0, "initialization_scale": .02,
                     "beta1": .9, "beta2": .999, "epsilon": 1e-7,
                     "gradient_clip_norm": 10.0, "betas": [0.0, .5, 1.0],
                     "rungs": [128, 512, 2048, 8192], "cohort_min_updates": 512,
                     "checkpoint_every": 128, "carry_optimizer_across_beta": True,
                     "beta_zero_updates": 0},
        "validation": {"bank_sizes": [768, 3072, 12288], "minimum_improvement": .04,
                       "maximum_half_width": .02, "plateau_comparisons": 2,
                       "normal_interval_multiplier": 1.959963984540054,
                       "reliability_rows": 32, "reliability_rtol": 1e-9,
                       "reliability_atol": 1e-10},
        "tuning": {"l_grid": [3, 5, 9, 13, 18, 25],
                   "initial_epsilon": .01, "epsilon_domain": [1e-6, 2.0],
                   "repair_factor": 1.5, "max_repairs_per_family": 12,
                   "target_acceptance": .7, "practical_region": [.65, .75],
                   "repair_region": [.55, .85], "startup": 32, "pilot": 64,
                   "measurement": 64, "verification": 64, "evidence_rungs": [1, 2, 4],
                   "total_budget_units": 200, "repair_reserve_units": 50,
                   "max_candidates": 100, "chunk_max_results": 64,
                   "max_wall_seconds": 28800.0},
        "posterior": {"chains": 4, "warmup_min": 2000, "warmup_window": 1000,
                      "warmup_chunk": 1000, "warmup_max": 10000, "warmup_rhat": 1.05,
                      "retained_min": 1000, "retained_chunk": 1000,
                      "retained_max": 10000, "retained_rhat": 1.01,
                      "bulk_ess": 400.0, "tail_ess": 400.0, "mean_mcse_sd": .02,
                      "quantile_mcse_sd": .05, "event_mcse": .01,
                      "quantiles": [.025, .5, .975], "event_coordinate": 2,
                      "max_energy_error": 1000.0},
        "starts": {"max_proposals": 128, "per_sign": 2},
        "ensemble": {"charts": 2, "conditional_charts": 4,
                     "alternative_betas": [0.0, .25, .5, .75, 1.0]},
        "comparison": {"methods": list(METHODS), "mean_margin_sd": .10,
                       "quantile_margin_sd": .20, "event_margin": .05,
                       "reference_error_fraction": 1.0 / 3.0},
        "budget": {"campaign_remaining_seconds": 135275.83289109988,
                   "diagnostic_remaining_seconds": 54299.37991617,
                   "repair_allocation_seconds": 7200.0, "arm_cap_seconds": 28800.0,
                   "pricing_updates": 2, "forecast_safety_factor": 2.0},
    }


def _same_keys(actual, expected, where):
    if not isinstance(actual, dict) or set(actual) != set(expected):
        raise ValueError(f"{where}: missing or unknown fields; require {sorted(expected)}")
    for key, value in expected.items():
        if isinstance(value, dict):
            _same_keys(actual[key], value, f"{where}.{key}")


def validate_protocol(config):
    template = protocol_template()
    _same_keys(config, template, "protocol")
    digest(config)  # Reject non-finite JSON, including nested values.
    if config["schema"] != SCHEMA or config["role"] not in {"smoke", "development", "confirmation"}:
        raise ValueError("unsupported protocol schema or role")
    for name in ("cpu_reference", "jit_compile"):
        if type(config[name]) is not bool:
            raise ValueError(f"{name} must be boolean")
    if config["role"] != "smoke" and (config["cpu_reference"] or not config["jit_compile"]):
        raise ValueError("serious q20 training requires GPU/XLA")
    if config["target"] != template["target"]:
        raise ValueError("this route binds the q20 T30 strict float64 target; target changes need a new scope")
    seed = config["seed"]
    if not isinstance(seed, list) or len(seed) != 2 or any(type(x) is not int or not 0 <= x < 2**31 for x in seed):
        raise ValueError("seed must contain two nonnegative int32 values")
    t, v, h, p = (config[k] for k in ("training", "validation", "tuning", "posterior"))
    for name in ("widths", "roots", "rungs", "pricing_batches"):
        values = t[name]
        if not values or len(set(values)) != len(values) or any(type(x) is not int or x < (0 if name == "roots" else 1) for x in values):
            raise ValueError(f"invalid training {name}")
    if sorted(t["rungs"]) != t["rungs"] or t["cohort_min_updates"] not in t["rungs"]:
        raise ValueError("rungs must increase and include the funded cohort floor")
    if t["beta_zero_updates"] != 0 or t["carry_optimizer_across_beta"] is not True:
        raise ValueError("protocol uses analytic beta zero and carries Adam across beta")
    if t["activation"] != "tanh" or t["stages"] != 2:
        raise ValueError("q20 architecture search is the declared two-stage tanh family")
    if t["betas"][0] != 0 or t["betas"][-1] != 1 or sorted(set(t["betas"])) != t["betas"]:
        raise ValueError("betas must strictly increase from zero to one")
    for name in ("batch_size", "checkpoint_every", "cohort_min_updates"):
        if type(t[name]) is not int or t[name] <= (1 if name == "batch_size" else 0):
            raise ValueError(f"invalid {name}")
    if any(x <= 1 for x in t["pricing_batches"]):
        raise ValueError("pricing must also use batch-native training")
    if not t["learning_rates"] or len(set(t["learning_rates"])) != len(t["learning_rates"]):
        raise ValueError("learning rates must be distinct")
    for x in [*t["learning_rates"], t["s_max"], t["initialization_scale"], t["epsilon"], t["gradient_clip_norm"], v["maximum_half_width"], v["reliability_rtol"], v["reliability_atol"]]:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise ValueError("positive finite numerical controls required")
    if not 0 < t["beta1"] < 1 or not 0 < t["beta2"] < 1:
        raise ValueError("invalid Adam moments")
    if not v["bank_sizes"] or sorted(set(v["bank_sizes"])) != v["bank_sizes"] or any(type(x) is not int or x <= 1 for x in v["bank_sizes"]):
        raise ValueError("validation banks must increase and contain multiple rows")
    if p["chains"] != 4 or config["starts"]["per_sign"] * 2 != p["chains"]:
        raise ValueError("four independent chains with two starts per sign are required")
    for key in ("warmup_min", "warmup_window", "warmup_chunk", "warmup_max", "retained_min", "retained_chunk", "retained_max"):
        if type(p[key]) is not int or p[key] < 4:
            raise ValueError(f"invalid posterior {key}")
    if not p["warmup_window"] <= p["warmup_min"] <= p["warmup_max"] or p["retained_min"] > p["retained_max"]:
        raise ValueError("invalid sequential caps")
    if config["role"] != "smoke" and (t["rungs"] != template["training"]["rungs"] or t["cohort_min_updates"] < 512 or len(t["roots"]) < 3 or p["warmup_min"] < 2000 or p["warmup_window"] < 1000 or p["retained_min"] < 1000):
        raise ValueError("shortened protocols must be explicitly classified smoke")
    if h["epsilon_domain"][0] <= 0 or not h["epsilon_domain"][0] <= h["initial_epsilon"] <= h["epsilon_domain"][1]:
        raise ValueError("initial epsilon is a bounded warm-start hypothesis")
    if config["comparison"]["methods"] != list(METHODS):
        raise ValueError("comparison inventory must retain every baseline and proposed method")
    if config["ensemble"]["charts"] < 2 or config["ensemble"]["charts"] > len(t["roots"]):
        raise ValueError("ensemble needs at least two independently rooted charts")
    if not 0 < config["comparison"]["reference_error_fraction"] < 1:
        raise ValueError("reference error allocation must lie in (0,1)")
    for key, value in config["budget"].items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"invalid budget {key}")
    return config


def load_protocol(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate configuration key: {key}")
            result[key] = value
        return result
    return validate_protocol(json.loads(Path(path).read_text(), object_pairs_hook=pairs))


def scoped_seed(config, *labels):
    """Role-separated deterministic stateless seed; persisted with each call."""
    raw = bytes.fromhex(digest([config["seed"], config["role"], *labels]))
    return tuple(int.from_bytes(raw[i:i+4], "big") & 0x7fffffff for i in (0, 4))


def training_cohort(config):
    t = config["training"]
    return [{"id": f"w{width}-lr{lr:g}-r{root}", "width": width,
             "learning_rate": lr, "root": root}
            for width in t["widths"] for lr in t["learning_rates"] for root in t["roots"]]

"""Repository-issued materiality selection for an imperfect M13 proposal field.

This deliberately does not issue score admission or posterior qualification.
The scalar endpoint stays fixed when numerical force controls change.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .surrogate_hmc_dctr_selection import MULTIPLIERS, protection_controls as original_controls
from .surrogate_hmc_program import ROOT, file_hash, write_json

SCHEMA = "bayesfilter.m13_mh_materiality_control_selection.v1"
ALLOWANCE = .01
PARTITIONS = {"calibration": (1705, 12), "validation": (2705, 13)}
ORIGINAL_PROTOCOL = "geometric_materiality_v1"
REPAIR_PROTOCOL = "weaker_geometric_materiality_v1"
REPAIR_MULTIPLIERS = (.001, .0001, .00001, .000001)
REPAIR_PARTITIONS = {"calibration": (1706, 15), "validation": (2706, 16)}


def protocol(scope):
    identifier = scope.get("materiality_protocol", ORIGINAL_PROTOCOL)
    if identifier == ORIGINAL_PROTOCOL:
        return MULTIPLIERS, PARTITIONS, 5
    if identifier == REPAIR_PROTOCOL:
        return REPAIR_MULTIPLIERS, REPAIR_PARTITIONS, 6
    raise ValueError("unknown reviewed materiality protocol")


def protection_controls(multiplier):
    if multiplier in MULTIPLIERS:
        return original_controls(multiplier)
    if multiplier not in REPAIR_MULTIPLIERS:
        raise ValueError("unreviewed materiality protection multiplier")
    settings = original_controls(.001)
    settings.update(reset_ridge=1e-5*multiplier, correction_lm_damping=1e-3*multiplier,
        correction_lm_scale_floor=1e-6*multiplier, correction_trust_radius=.1/multiplier,
        pairwise_rms_cap=2./multiplier, coordinate_cap=16./multiplier)
    return settings


def pair_decision(stronger, weaker):
    """Recompute from archived probabilities and health; never trust a pass flag."""
    def valid_arm(arm):
        probabilities = arm.get("acceptance_probabilities", [])
        return (isinstance(probabilities, list) and len(probabilities) == 16
                and all(isinstance(x, (float, int)) and not isinstance(x, bool)
                        and math.isfinite(x) and 0 <= x <= 1 for x in probabilities))
    if not all(valid_arm(arm) for arm in (stronger, weaker)):
        raise ValueError("sixteen finite matched acceptance probabilities required")
    difference = max(abs(a-b) for a, b in zip(stronger["acceptance_probabilities"],
                                            weaker["acceptance_probabilities"]))
    healthy = all(arm.get("nominal_healthy") is True and
                  arm.get("proposal_healthy") is True for arm in (stronger, weaker))
    stress = all(arm.get("stress_failures_flagged") is True for arm in (stronger, weaker))
    old, new = weaker.get("stress_healthy_rows"), stronger.get("stress_healthy_rows")
    if not all(isinstance(x, list) and len(x) == 4 and all(type(v) is bool for v in x)
               for x in (old, new)):
        raise ValueError("four stress-row health flags required")
    no_new_failure = all(not before or after for before, after in zip(old, new))
    return dict(maximum_acceptance_probability_difference=difference,
                acceptance_allowance=ALLOWANCE, healthy_nominal=healthy,
                stress_safe=stress, no_new_stress_failure=no_new_failure,
                non_harm_passed=healthy and stress and no_new_failure and difference <= ALLOWANCE)


def choose(curves, *, scope=None):
    multipliers, partitions, _ = protocol(scope or {})
    if set(curves) != set(partitions):
        raise ValueError("independent calibration and validation required")
    decisions = {}
    for partition, curve in curves.items():
        arms = curve["arms"]
        if len(arms) != 4 or tuple(a.get("multiplier") for a in arms) != multipliers:
            raise ValueError("wrong prospective control family/order")
        if curve.get("probe_steps") != [.05, .2] or curve.get("probe_leapfrogs") != 2:
            raise ValueError("wrong prospective proposal probes")
        decisions[partition] = [pair_decision(arms[i], arms[i+1]) for i in range(3)]
    index = next((i for i in range(3) if all(decisions[p][i]["non_harm_passed"]
                                           for p in partitions)), None)
    return index, decisions


def _evidence(scope, source_hashes, curves, data):
    _, partitions, _ = protocol(scope)
    if set(curves) != set(partitions) or set(data) != set(partitions):
        raise ValueError("wrong evidence partitions")
    loaded = {}
    for name, (seed, partition) in partitions.items():
        datum = json.loads(Path(data[name]).read_text())
        curve = json.loads(Path(curves[name]).read_text())
        if (datum.get("partition") != name or datum.get("data_seed") != seed
                or datum.get("design_partition") != partition
                or datum.get("scope") != scope or curve.get("scope") != scope
                or curve.get("source_hashes") != source_hashes
                or curve.get("data_sha256") != file_hash(Path(data[name]))):
            raise ValueError("control evidence data/scope/source mismatch")
        loaded[name] = curve
    return choose(loaded, scope=scope)


def issue_selection(output, *, scope, source_hashes, curves, data):
    multipliers, _, _ = protocol(scope)
    index, decisions = _evidence(scope, source_hashes, curves, data)
    if index is None:
        raise ValueError("NO_MATERIAL_CONTROL_PAIR")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    value = dict(schema=SCHEMA, scope=scope, source_hashes=source_hashes,
                 issuer_sha256=file_hash(Path(__file__)), selected_index=index,
                 multiplier=multipliers[index], controls=protection_controls(multipliers[index]),
                 decisions=decisions, exact_score_admission=False, posterior_qualified=False,
                 selection_rule="first adjacent materiality pass on both fresh partitions",
                 evidence={p: dict(path=str(Path(v).resolve()), sha256=file_hash(Path(v))) for p,v in curves.items()},
                 data={p: dict(path=str(Path(v).resolve()), sha256=file_hash(Path(v))) for p,v in data.items()})
    write_json(output, value)
    return value


def load_selection(path, *, expected_scope, expected_source_hashes):
    multipliers, partitions, _ = protocol(expected_scope)
    value = json.loads(Path(path).read_text())
    if (value.get("schema") != SCHEMA or value.get("scope") != expected_scope
            or value.get("source_hashes") != expected_source_hashes
            or value.get("issuer_sha256") != file_hash(Path(__file__))
            or value.get("exact_score_admission") is not False
            or value.get("posterior_qualified") is not False):
        raise ValueError("wrong or stale materiality selection")
    for name, digest in expected_source_hashes.items():
        if file_hash(ROOT/name) != digest:
            raise ValueError("selection source changed: "+name)
    for group in ("evidence", "data"):
        if set(value[group]) != set(partitions):
            raise ValueError("missing selection partition")
        for row in value[group].values():
            if file_hash(Path(row["path"])) != row["sha256"]:
                raise ValueError("selection evidence checksum mismatch")
    index, decisions = _evidence(expected_scope, expected_source_hashes,
        {p: row["path"] for p,row in value["evidence"].items()},
        {p: row["path"] for p,row in value["data"].items()})
    if (index is None or index != value.get("selected_index")
            or value.get("multiplier") != multipliers[index]
            or value.get("controls") != protection_controls(multipliers[index])
            or value.get("decisions") != decisions):
        raise ValueError("selection differs from measured decision")
    return value

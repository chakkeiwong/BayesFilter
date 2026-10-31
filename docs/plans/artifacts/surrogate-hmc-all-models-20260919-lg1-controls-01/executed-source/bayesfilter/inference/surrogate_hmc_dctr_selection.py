"""Offline full-DCTR controls for imperfect proposal-field experiments only.

This authority is deliberately separate from exact-score admission. Numerical
curves and original status must be evaluated on independent calibration and
validation partitions before issuing the scope-bound artifact.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = "bayesfilter.surrogate_hmc_dctr_control_selection.v1"
USE = "imperfect_proposal_field_experiment"
MULTIPLIERS = (1., .1, .01, .001)


def protection_controls(multiplier):
    from bayesfilter.testing.surrogate_hmc_lg1_tf import warm_start_controls
    if multiplier not in MULTIPLIERS:
        raise ValueError("unreviewed protection multiplier")
    settings = warm_start_controls()
    settings.update(reset_ridge=1e-5*multiplier, correction_lm_damping=1e-3*multiplier,
                    correction_lm_scale_floor=1e-6*multiplier, correction_trust_radius=.1/multiplier,
                    pairwise_rms_cap=2./multiplier, coordinate_cap=16./multiplier)
    return settings


def first_stable_pair(evidence):
    if set(evidence) != {"calibration", "validation"}:
        raise ValueError("separate calibration and validation evidence required")
    for rows in evidence.values():
        if len(rows) != 3:
            raise ValueError("three adjacent-pair decisions required")
        for row in rows:
            if type(row.get("non_harm_passed")) is not bool:
                raise ValueError("explicit non-harm decision required")
            change = row.get("normalized_maximum_change")
            recomputed = (all(row.get(k) is True for k in
                             ("healthy_nominal", "stress_safe", "no_new_stress_failure"))
                          and isinstance(change, (int, float)) and math.isfinite(change)
                          and 0 <= change <= 1e-3 and row.get("tolerance") == 1e-3)
            if row["non_harm_passed"] != recomputed:
                raise ValueError("non-harm decision does not match measured checks")
    return next((i for i in range(3) if all(evidence[p][i]["non_harm_passed"] for p in evidence)), None)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def issue_selection(*, output, scope, evidence_paths, partition_data_paths, source_hashes):
    """Read saved measured curves; callers cannot stamp a selected index."""
    if set(evidence_paths) != {"calibration", "validation"} or set(partition_data_paths) != set(evidence_paths):
        raise ValueError("selection cannot consume claim data")
    data = {p: json.loads(Path(path).read_text()) for p, path in partition_data_paths.items()}
    if any(d.get("partition") != p or d.get("attempt") != 4 for p, d in data.items()):
        raise ValueError("wrong tuning partition or attempt")
    curves = {p: json.loads(Path(path).read_text()) for p, path in evidence_paths.items()}
    if any(c.get("scope") != scope or c.get("source_hashes") != source_hashes for c in curves.values()):
        raise ValueError("curve scope/source mismatch")
    index = first_stable_pair({p: c["pairs"] for p, c in curves.items()})
    if index is None:
        raise ValueError("NO_STABLE_CONTROL_PAIR")
    path = Path(output)
    if path.exists():
        raise FileExistsError("selection must not overwrite evidence")
    payload = dict(schema=SCHEMA, use=USE, exact_score_admission=False, default_readiness=False,
                   scope=scope, source_hashes=source_hashes, selected_index=index,
                   selected_multiplier=MULTIPLIERS[index], controls=protection_controls(MULTIPLIERS[index]),
                   evidence={p:dict(path=str(Path(v).resolve()),sha256=_sha(v)) for p,v in evidence_paths.items()},
                   partition_data={p:dict(path=str(Path(v).resolve()),sha256=_sha(v)) for p,v in partition_data_paths.items()},
                   issuer_sha256=_sha(__file__), selection_rule="first adjacent non-harm pass on both independent partitions")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
    return payload


def load_selection(path, *, expected_scope, expected_source_hashes):
    payload = json.loads(Path(path).read_text())
    if (payload.get("schema") != SCHEMA or payload.get("use") != USE
            or payload.get("exact_score_admission") is not False or payload.get("default_readiness") is not False):
        raise ValueError("not an imperfect proposal-field selection")
    if (payload.get("scope") != expected_scope or payload.get("source_hashes") != expected_source_hashes
            or payload.get("issuer_sha256") != _sha(__file__)):
        raise ValueError("selection scope or source mismatch")
    for group in ("evidence", "partition_data"):
        if set(payload[group]) != {"calibration", "validation"}:
            raise ValueError("invalid tuning partitions")
        for record in payload[group].values():
            if _sha(record["path"]) != record["sha256"]:
                raise ValueError("selection evidence checksum mismatch")
    curves = {p:json.loads(Path(v["path"]).read_text()) for p,v in payload["evidence"].items()}
    index = first_stable_pair({p:c["pairs"] for p,c in curves.items()})
    if (index is None or payload.get("selected_index") != index
            or payload.get("controls") != protection_controls(MULTIPLIERS[index])):
        raise ValueError("selection does not follow the measured rule")
    return payload

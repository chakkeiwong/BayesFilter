"""Frozen FD design issuance from validated independent calibration records."""
import json
from pathlib import Path

from .contracts import digest
from .coordinator import write_json
from .fd_reporting import checked_fd_records, fd_signature


def fd_scope(study,row):
    return {"settings":study["settings"],"model":row["model"],"proposal":row["proposal"],
            "estimator":row["estimator"],"comparison_target":row["comparison_target"],
            "candidate_family":study["fd_candidate_family"]}


def derive_fd_selection(root):
    state,records = checked_fd_records(root)
    study = state["study"]
    rows = [(r,v) for r,v in records if r["role"] in ("calibration","validation")]
    if not rows: raise ValueError("missing FD calibration/validation")
    from .runtime import configure_runtime
    configure_runtime(device=study["settings"]["device"],tf32=study["settings"]["tf32"],
                      jit_compile=study["settings"]["jit_compile"])
    import tensorflow as tf
    ref_scope = fd_scope(study,rows[0][0])
    groups,signatures,evidence = {},{},[]
    for row,value in rows:
        if fd_scope(study,row)!=ref_scope: raise ValueError("mixed FD selection scopes")
        signature = fd_signature(row)
        candidate = digest(signature)
        signatures[candidate] = signature
        d = value["diagnostics"]
        for k in ("fd_stencil","fd_coupling","fd_directions"):
            if d[k]!=row[k]: raise ValueError("executed FD design differs from row")
        if abs(d["fd_h"]-row["fd_h"]) > 2e-7*abs(row["fd_h"]):
            raise ValueError("executed FD step differs from row")
        if d.get("controls")!=row.get("controls"): raise ValueError("executed LEDH controls differ")
        error = tf.constant(value["score"],tf.float64)-tf.constant(value["oracle_score"],tf.float64)
        key = (row["dataset"],row["replicate"])
        group = groups.setdefault(candidate,{"calibration":{},"validation":{}})[row["role"]]
        if key in group: raise ValueError("duplicate selection replicate")
        group[key] = tf.reduce_sum(error**2)
        evidence.append({"row":row["id"],"digest":digest(value)})
    if set(signatures)!={digest(x) for x in study["fd_candidate_family"]}:
        raise ValueError("FD candidate family incomplete or changed")
    expected_keys = None
    candidates = []
    for candidate,roles in sorted(groups.items()):
        keys = tuple(tuple(sorted(roles[role])) for role in ("calibration","validation"))
        if not all(keys) or (expected_keys is not None and keys!=expected_keys):
            raise ValueError("unmatched FD calibration/validation coverage")
        expected_keys = keys
        losses = {}
        for role,values in roles.items():
            means = [tf.reduce_mean(tf.stack([v for (ds,_),v in values.items() if ds==dataset]))
                     for dataset in sorted({ds for ds,_ in values})]
            losses[role] = float(tf.reduce_mean(tf.stack(means)).numpy())
        candidates.append({"candidate":candidate,"signature":signatures[candidate],**losses})
    choice = min(candidates,key=lambda c:(c["calibration"],c["candidate"]))
    return {"schema":"younis_score_fd_selection_v1","issuer":"bayesfilter.score_study.fd_selection",
        "source_run":str(Path(root).resolve()),"source_fingerprint":digest(state["fingerprint"]),
        "scope":ref_scope,"evidence":evidence,"candidates":candidates,
        "selected_signature":choice["signature"],"criterion":"equal_dataset_weight_calibration_vector_MSE",
        "partitions":study["partitions"],"evidence_class":study["evidence_class"],
        "validation_status":"finite_independent_descriptive_only","default_ready":False}


def issue_fd_selection(root,destination):
    destination = Path(destination)
    if destination.exists(): raise ValueError("preserve existing FD selection")
    result = derive_fd_selection(root)
    write_json(destination,result)
    return result


def consume_fd_selection(row,context):
    if "fd_selection" not in row: raise ValueError("missing repository-issued FD selection")
    supplied = json.loads(Path(row["fd_selection"]).read_text())
    derived = derive_fd_selection(supplied["source_run"])
    if supplied!=derived: raise ValueError("modified or caller-stamped FD selection")
    if fd_scope(context["study"],row)!=derived["scope"]: raise ValueError("FD selection scope mismatch")
    if fd_signature(row)!=derived["selected_signature"]: raise ValueError("claim changes frozen FD design")
    if row["dataset"] in derived["partitions"]["calibration"]+derived["partitions"]["validation"]:
        raise ValueError("FD claim data leaked into calibration")
    if derived["evidence_class"]=="mechanics" and context["study"]["evidence_class"]!="mechanics":
        raise ValueError("mechanics FD selection cannot support scientific claim")
    return derived

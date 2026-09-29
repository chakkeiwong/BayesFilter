"""Post-run TF covariance/error attribution with independent h calibration."""
import json
from pathlib import Path

from .contracts import digest, validate_result, validate_study
from .coordinator import fingerprint, write_json
from .registry import default_registry


def fd_signature(row):
    return {k: row[k] for k in ("model", "proposal", "fd_stencil", "fd_h", "fd_coupling",
                                "fd_directions", "fd_error_budget", "fd_lower", "fd_upper")} | {
        k: row[k] for k in ("controls", "coupling_group") if k in row}


def checked_fd_records(root):
    root = Path(root)
    state = json.loads((root/"state.json").read_text())
    study,registry = state["study"],default_registry()
    validate_study(study,registry)
    if state["fingerprint"] != fingerprint(study,registry):
        raise ValueError("stale FD source")
    records = []
    for row in study["rows"]:
        if row["estimator"] != "symmetric_fd": continue
        saved = state["rows"].get(row["id"],{})
        if saved.get("execution_status") != "complete":
            raise ValueError("FD report requires all requested FD rows; repair failed rows first")
        result = json.loads((root/saved["result_path"]).read_text())
        validate_result(result,row,registry)
        if digest(result) != saved["result_digest"]: raise ValueError("corrupt FD result")
        records.append((row,result))
    return state,records


def assemble_fd_diagnostics(root):
    state,records = checked_fd_records(root)
    from .runtime import configure_runtime
    configure_runtime(device="CPU",tf32=False,jit_compile=True)
    import tensorflow as tf
    from .finite_difference_tf import replicated_stencil_diagnostics
    groups, signatures = {},{}
    for row,result in records:
        candidate = digest(fd_signature(row))[:16]
        signatures[candidate] = fd_signature(row)
        key = (row["role"],row["dataset"],candidate)
        group = groups.setdefault(key,{})
        if row["replicate"] in group: raise ValueError("duplicate FD replicate")
        group[row["replicate"]] = result
    summaries = []
    for (role,dataset,candidate),group in sorted(groups.items()):
        results = [group[r] for r in sorted(group)]
        if len(results)<2: raise ValueError("FD conditional covariance needs independent replicates")
        first = results[0]
        if any(r["diagnostics"]["data_version"] != first["diagnostics"]["data_version"] for r in results):
            raise ValueError("FD replicates do not condition on the same observations")
        diag = first["diagnostics"]
        values = tf.constant([r["diagnostics"]["fd_values"] for r in results],tf.float64)
        exact = tf.constant(diag["fd_exact_values"],tf.float64)
        coef,h = tf.constant(diag["fd_coefficients"],tf.float64),tf.constant(diag["fd_h"],tf.float64)
        exact_direction = tf.constant(diag["fd_oracle_directional_scores"],tf.float64)
        directional = []
        for j in range(values.shape[1]):
            out = replicated_stencil_diagnostics(values[:,j,:],coef,h,exact[j],exact_direction[j])
            directional.append({k:(v.numpy().tolist() if hasattr(v,"numpy") else v) for k,v in out.items()})
        # Preserve the full covariance across directions and nodes. Its off-
        # diagonal blocks are lost by independent directional error bars.
        flat = tf.reshape(values,[len(results),-1])
        centered = flat-tf.reduce_mean(flat,axis=0)
        node_covariance = tf.transpose(centered)@centered/(len(results)-1)
        b = tf.einsum("rmj,j->rm",values,coef)/h
        bc = b-tf.reduce_mean(b,axis=0)
        b_covariance = tf.transpose(bc)@bc/(len(results)-1)
        V = tf.constant(diag["fd_directions"],tf.float64)
        singular,u,v = tf.linalg.svd(V,full_matrices=False)
        inverse = (u/singular[None,:])@tf.transpose(v)
        propagated = inverse@b_covariance@tf.transpose(inverse)
        scores = tf.constant([r["score"] for r in results],tf.float64)
        sc = scores-tf.reduce_mean(scores,axis=0)
        observed = tf.transpose(sc)@sc/(len(results)-1)
        error = tf.linalg.norm(propagated-observed)
        margin = tf.cast(1e-10,tf.float64)*tf.maximum(tf.linalg.norm(observed),tf.cast(1e-14,tf.float64))
        if float(error.numpy())>float(margin.numpy()): raise ValueError("FD covariance propagation mismatch")
        oracle = tf.constant(first["oracle_score"],tf.float64)
        analytic = tf.constant([r["diagnostics"]["fd_analytic_score"] for r in results],tf.float64)
        summaries.append({"role":role,"dataset":dataset,"candidate":candidate,
            "replicates":sorted(group),"directional":directional,
            "full_node_covariance":node_covariance.numpy().tolist(),
            "directional_covariance":b_covariance.numpy().tolist(),
            "score_covariance":propagated.numpy().tolist(),"propagation_error":float(error.numpy()),
            "vector_mse":float(tf.reduce_mean(tf.reduce_sum((scores-oracle)**2,axis=1)).numpy()),
            "analytic_vector_mse":float(tf.reduce_mean(tf.reduce_sum((analytic-oracle)**2,axis=1)).numpy()),
            "value_calls":sum(r["runtime"]["finite_difference_value_calls"] for r in results),
            "kernel_seconds":sum(r["runtime"]["kernel_wall_seconds"] for r in results)})
    joint_groups = {}
    for (role,dataset,candidate),group in groups.items():
        key = (role,dataset,signatures[candidate]["proposal"])
        joint_groups.setdefault(key,{})[candidate] = group
    joint_results = []
    for (role,dataset,proposal),candidates in sorted(joint_groups.items()):
        ordered = sorted(candidates)
        replica_sets = {tuple(sorted(g)) for g in candidates.values()}
        if len(replica_sets)!=1: raise ValueError("unpaired cross-stencil replicate coverage")
        reps = next(iter(replica_sets))
        joint = tf.constant([[x for c in ordered for x in candidates[c][r]["score"]]
                             for r in reps],tf.float64)
        centered = joint-tf.reduce_mean(joint,axis=0)
        cov = tf.transpose(centered)@centered/(len(reps)-1)
        joint_results.append({"role":role,"dataset":dataset,"proposal":proposal,
            "candidate_order":ordered,"replicates":list(reps),
            "coordinate_order":"six score coordinates per candidate in candidate_order",
            "cross_stencil_score_covariance":cov.numpy().tolist(),
            "limitation":"sample covariance rank at most replication count minus one"})
    calibration = {}
    for s in summaries:
        if s["role"] == "calibration": calibration.setdefault(s["candidate"],[]).append(s["vector_mse"])
    selection = None
    if calibration:
        # Equal weight per independent dataset; particle count per dataset
        # cannot silently alter the calibration objective.
        keys = {c:{s["dataset"] for s in summaries if s["candidate"]==c and s["role"]=="calibration"} for c in calibration}
        if len({tuple(sorted(x)) for x in keys.values()})!=1: raise ValueError("unmatched FD calibration data")
        means = {c:sum(v)/len(v) for c,v in calibration.items()}
        winner = min(means,key=lambda c:(means[c],c))
        heldout = [s for s in summaries if s["role"]=="validation" and s["candidate"]==winner]
        if not heldout: raise ValueError("selected FD candidate lacks independent validation")
        selection = {"candidate":winner,"signature":signatures[winner],"criterion":"dataset_mean_vector_MSE",
            "fit_partition":"calibration","evaluation_partition":"validation",
            "calibration_mse":means[winner],"validation_mse":sum(s["vector_mse"] for s in heldout)/len(heldout),
            "selected_from_claim":False,"scientific_claim_admitted":False,
            "reason":"engineering calibration; final scope-matched untouched replication remains required"}
    report = {"schema":"younis_score_fd_diagnostics_v1","source_fingerprint":digest(state["fingerprint"]),
        "signatures":signatures,"conditional_results":summaries,"calibrated_choice":selection,
        "cross_stencil_results":joint_results,
        "aggregation_backend":"TensorFlow CPU diagnostic; GPU intentionally hidden",
        "error_identity":"empirical MSE=(T_h+B_hat_N_h)^2+(R-1)/R sample_variance; no omitted cross-term",
        "particle_bias_status":"estimated conditional on dataset; squared bias corrected for mean MC variance",
        "stochastic_curve_shape":"measured; no universal U-curve or 1/h^2 variance assumption",
        "statistically_supported_ranking":False,"default_ready":False,
        "remaining":"final selection consumption and conditional nonlinear comparisons"}
    write_json(Path(root)/"finite-differences.json",report)
    return report

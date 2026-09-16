"""Independent-calibration control variates and biased-score blends.

This command executes a mechanics/diagnostic study, not a default selector.
The same numerical controls used by the rows remain frozen on validation.
"""
import json
from pathlib import Path

from .contracts import digest, validate_result, validate_study
from .coordinator import fingerprint, write_json
from .registry import default_registry


def combination_records(run_root):
    root = Path(run_root)
    state = json.loads((root / "state.json").read_text())
    study, registry = state["study"], default_registry()
    validate_study(study, registry)
    if state["fingerprint"] != fingerprint(study, registry):
        raise ValueError("combination source is stale")
    records = {"calibration": {}, "validation": {}}
    signatures = {}
    for row in study["rows"]:
        role = row.get("role")
        if role not in records or row["proposal"] not in ("ledh", "integrated_kdm"):
            continue
        saved = state["rows"].get(row["id"], {})
        if saved.get("execution_status") != "complete":
            raise ValueError("failed combination constituent")
        result = json.loads((root / saved["result_path"]).read_text())
        validate_result(result, row, registry)
        if digest(result) != saved["result_digest"]:
            raise ValueError("corrupt combination constituent")
        key = (row["dataset"], row["replicate"])
        pair = records[role].setdefault(key, {})
        if row["proposal"] in pair:
            raise ValueError("duplicate combination constituent")
        signature = {k: row.get(k) for k in ("model", "controls", "bandwidth_scale", "coupling_group")}
        if row["proposal"] in signatures and signatures[row["proposal"]] != signature:
            raise ValueError("combination constituents changed between partitions")
        signatures[row["proposal"]] = signature
        pair[row["proposal"]] = result
    for role, rows in records.items():
        if len(rows) < 2:
            raise ValueError("independent calibration and validation rows required")
        for pair in rows.values():
            if set(pair) != {"ledh", "integrated_kdm"}:
                raise ValueError("unpaired combination coverage")
            a, b = pair["ledh"], pair["integrated_kdm"]
            if a["diagnostics"]["data_version"] != b["diagnostics"]["data_version"] or a["oracle_score"] != b["oracle_score"]:
                raise ValueError("combination observations/oracles differ")
            if "kdm_zero_control" not in a["diagnostics"]:
                raise ValueError("missing exactly centered KDM control")
    return state, records


def assemble_combinations(run_root):
    state, records = combination_records(run_root)
    from .runtime import configure_runtime
    # Post-run fitting/reporting is an explicit CPU diagnostic. No new filter
    # kernel or GPU evidence is manufactured by this aggregation command.
    configure_runtime(device="CPU", tf32=False, jit_compile=True)
    import tensorflow as tf
    from .combinations_tf import make_combination_kernels
    from .reporting import paired_dataset_summary
    d = state["study"]["settings"]["dimension"]
    fit_control, apply_control, fit_blend, apply_blend = make_combination_kernels(6, d+1)
    def tensors(rows):
        pairs = [rows[k] for k in sorted(rows)]
        return tuple(tf.constant(values, tf.float64) for values in (
            [x["ledh"]["score"] for x in pairs],
            [x["integrated_kdm"]["score"] for x in pairs],
            [x["ledh"]["oracle_score"] for x in pairs],
            [x["ledh"]["diagnostics"]["kdm_zero_control"] for x in pairs]))
    a, b, oracle, c = tensors(records["calibration"])
    beta, rank, valid = fit_control(a-oracle, c)
    if not bool(valid.numpy()) or int(rank.numpy()) < d+1:
        raise ValueError("calibration control design rank/validity veto")
    alpha, calibration_mse = fit_blend(a, b, oracle)
    a, b, oracle, c = tensors(records["validation"])
    cv, blend = apply_control(a, c, beta), apply_blend(a, b, alpha)
    scores = {"ledh": a, "integrated_kdm": b, "known_center_kdm_cv": cv,
              "oracle_calibrated_blend": blend}
    squared_errors = {name: dict(zip(sorted(records["validation"]),
                                     tf.reduce_sum((s-oracle)**2, axis=1).numpy().tolist()))
                      for name, s in scores.items()}
    report = {"schema": "younis_score_combination_diagnostic_v1",
              "source_fingerprint": digest(state["fingerprint"]),
              "device": "CPU_reference_GPU_intentionally_hidden", "jit_compile": True,
              "coefficient_fit_partition": "calibration", "evaluation_partition": "validation",
              "coefficient": beta.numpy().tolist(), "control_design_rank": int(rank.numpy()),
              "blend_alpha": float(alpha.numpy()), "calibration_blend_mse": float(calibration_mse.numpy()),
              "control_center": [0.]*(d+1), "center_estimation_cost": 0,
              "center_identity": "integral of normalized fixed-mixture density derivative is zero",
              "baseline_bias_removed": False, "controls_fitted_on_evaluation": False,
              "row_counts": {role: len(rows) for role, rows in records.items()},
              "validation_scores": {name: value.numpy().tolist() for name, value in scores.items()},
              "validation_oracle_scores": oracle.numpy().tolist(),
              "comparisons": {name: paired_dataset_summary(errors, squared_errors["ledh"])
                              for name, errors in squared_errors.items() if name != "ledh"},
              "inference_status": "mechanics_only" if state["study"]["evidence_class"] == "mechanics" else "descriptive_only",
              "statistically_supported_ranking": False, "default_ready": False,
              "limitations": ["tiny fixture; no ranking", "finite-particle baseline bias is unchanged by the CV",
                              "integrated KDM and its blend are not derivatives of the canonical finite likelihood",
                              "pilot and coefficients are unpromoted hypotheses; full cost/replication study remains"]}
    write_json(Path(run_root) / "combinations.json", report)
    return report

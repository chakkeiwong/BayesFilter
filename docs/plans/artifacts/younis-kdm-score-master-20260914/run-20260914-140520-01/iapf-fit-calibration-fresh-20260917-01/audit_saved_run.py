"""Read-only diagnostic of completed campaign artifacts; no TensorFlow work."""
import hashlib
import json
from pathlib import Path

from bayesfilter.score_study.contracts import digest, validate_result
from bayesfilter.score_study.coordinator import fingerprint
from bayesfilter.score_study.registry import default_registry

root = Path(__file__).resolve().parent / "launch01"
read = lambda path: json.loads(path.read_text())
manifest = read(root / "run-manifest.json")
registry = default_registry()
rows = read(root / "consumer-evidence.json")
failures = read(root / "invalid-fits.json")
charges = read(root / "attempt-accounting.json")
assert manifest["status"] == "completed_with_numerical_vetoes"
assert len(rows) == 54 and len(failures) == 14 and len(charges) == 68
assert sum(c["charges"] for c in charges) == manifest["charged_attempts"] == 171
assert len({(c["stage"], c["row"]) for c in charges}) == 68

specs = {}
for stage in manifest["studies"]:
    spec = read(root / (stage["name"] + "-study.json"))
    state = read(Path(stage["path"]) / "state.json")
    assert fingerprint(spec, registry) == state["fingerprint"] == stage["fingerprint"]
    specs[stage["name"]] = {r["id"]: r for r in spec["rows"]}

references, refinements, shapes = {}, [], []
for row in rows:
    result = read(Path(row["path"]))
    assert digest(result) == row["digest"]
    validate_result(result, specs[row["stage"]][row["row"]], registry)
    diag = result["diagnostics"]
    key = row["regime"], row["dataset"]
    reference = result["oracle_value"], result["oracle_score"], diag["executed_observation_digest"]
    if key in references:
        assert references[key] == reference
    references[key] = reference
    refinements.append([diag[name] for name in (
        "reference_mesh_relative_error", "reference_domain_relative_error", "reference_max_tail_mass")])
    assert refinements[-1][0] <= 1e-7 and refinements[-1][1] <= 1e-7 and refinements[-1][2] <= 1e-9
    if row["proposal"] == "iapf":
        assert row["fit_summary"]["underflow_steps"] == 0
        shapes.append(row["fit_summary"]["max_shape_residual"])

failed_fit_details = []
for failure in failures:
    diag = failure["diagnostics"]["details"]
    records = diag["fit_iterations"]
    steps = [s for rec in records for s in rec.get("density_fit_diagnostics", [])]
    underflow = [s for s in steps if s[9]]
    assert underflow and any(not rec["density_fit_valid"] for rec in records)
    assert all(rec["density_fit_converged"] and rec["coefficient_cast_valid"] for rec in records)
    failed_fit_details.append({"stage": failure["stage"], "row": failure["row"]["id"],
        "underflow_steps": len(underflow), "underflow_shape_residuals": [s[1] for s in underflow],
        "underflow_log_amplitudes": [s[8] for s in underflow],
        "solver_converged_despite_invalid_fit": True})

tables = read(root / "conditional-heuristics.json")
for regime in tables:
    for entry in regime["datasets"]:
        assert entry["heuristic_table_complete"] and entry["baseline"]["complete"]
        assert not entry["selected"]["complete"]
        assert {"ekf", "ukf"} <= set(entry["baseline"]["observed_losses_to"])
        assert entry["baseline"]["heuristic_dominance_verdict"] == "promotion_veto_observed_loss"
        pair = [r for r in rows if r["regime"] == regime["regime"]
                and r["dataset"] == entry["dataset"] and r["arm"] == "baseline"]
        assert len(pair) == 2 and pair[0]["fit_digest"] == pair[1]["fit_digest"]
        assert pair[0]["final_seeds"] != pair[1]["final_seeds"]

result = {"status": "pass", "numerical_work": False, "gpu_intentionally_hidden": True,
    "checked_source_studies": len(specs), "checked_result_digests": len(rows),
    "reference_maxima": {name: max(x[i] for x in refinements) for i, name in enumerate((
        "mesh_relative_error", "domain_relative_error", "tail_mass"))},
    "underflow_failed_rows": len(failed_fit_details),
    "underflow_failed_steps": sum(x["underflow_steps"] for x in failed_fit_details),
    "failed_fits": failed_fit_details,
    "complete_heuristic_datasets": 6, "complete_baseline_claim_rows": 12,
    "blocked_selected_claim_rows": 12, "charged_attempts": 171,
    "boundary_affected_baseline_claim_rows": sum(
        bool(r["fit_summary"]["boundary_steps"]) for r in rows if r["arm"] == "baseline"),
    "max_accepted_shape_residual": max(shapes),
    "heuristic_dominance_verdict": "promotion_veto_observed_loss_in_every_claim_dataset",
    "statistically_supported_ranking": False, "default_ready": False,
    "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(root.parent / "saved-run-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: result[k] for k in ("status", "checked_source_studies", "checked_result_digests",
    "underflow_failed_rows", "underflow_failed_steps", "reference_maxima")}))

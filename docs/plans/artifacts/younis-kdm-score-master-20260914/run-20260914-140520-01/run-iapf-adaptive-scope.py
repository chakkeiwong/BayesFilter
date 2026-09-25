"""Bounded GPU/XLA mechanics: adaptive iAPF selection and actual final consumer."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path.cwd()))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    if root.exists():
        raise ValueError("fresh versioned output required")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != args.source_revision:
        raise ValueError("source revision mismatch")
    from bayesfilter.score_study.coordinator import execute, fingerprint, report, write_json
    from bayesfilter.score_study.registry import default_registry
    from bayesfilter.score_study.runtime import configure_runtime
    from bayesfilter.score_study.tuning import issue_selection
    registry = default_registry()
    root.mkdir(parents=True)
    started = time.monotonic()
    manifest = {"schema": "iapf_adaptive_scope_mechanics_v1", "status": "running",
        "git_commit": commit, "source_root": str(Path.cwd()), "command": sys.argv,
        "interpreter": sys.executable, "plan": "docs/plans/younis-score-iapf-adaptive-scope-2026-09-16.md",
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": {k: os.environ.get(k) for k in
            ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "BAYESFILTER_PRELOAD_CUSTOM_OP",
             "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS")},
        "studies": [], "statistically_supported_ranking": False, "default_ready": False,
        "runtime_comparison": "forbidden_concurrent_workloads",
        "score_semantics": "finite_program_fixed_fit_realized_N_and_labels"}
    write_json(root / "run-manifest.json", manifest)
    try:
        manifest["hardware"] = configure_runtime(device="GPU", tf32=True, jit_compile=True)
        settings = dict(dimension=1, observation_dimension=1, horizon=2, particles=16,
            dtype="float32", device="GPU", tf32=True, jit_compile=True,
            theta=[.62,-.8,-.6,.9,.25,-.3], data_theta=[.62,-.8,-.6,.9,.25,-.3],
            prepared_data_regime="scalar_gaussian_all_parameter_mechanics")
        config = dict(k=1, tau=100., max_iterations=4, max_particles=128,
            mean_bound=4., sd_lower=.2, sd_upper=4., max_fit_steps=2000,
            max_backtracks=30, fit_tolerance=1e-7, floor_ratio=.01,
            fit_theta=settings["theta"], fit_dtype="float64")
        family = [{"iapf": config}, {"iapf": {**config, "k": 2, "max_iterations": 5}}]
        base_row = dict(model="gaussian_all_parameters", proposal="iapf",
            estimator="analytical_filter", comparison_target="model_score",
            comparison="approximation_error", replicate=0, coupling_group="baseline")
        study = dict(schema="younis_score_study_v1", phase="0E", version=1,
            plan=manifest["plan"], evidence_class="mechanics", seed=841,
            settings=settings, required_proposals=["iapf"], tuning_candidate_family=family,
            budget=dict(wall_seconds=900, max_attempts=4, max_attempts_per_row=1),
            partitions=dict(calibration=[800], validation=[810], claim=[820]),
            rows=[dict(base_row, **candidate, id=f"candidate{index}-{role}", role=role, dataset=dataset)
                  for index, candidate in enumerate(family)
                  for role, dataset in (("calibration",800),("validation",810))])
        source_before = fingerprint(study, registry)["sources"]
        numerical = []

        def stage(name, spec):
            write_json(root / f"{name}-study.json", spec)
            state = execute(spec, registry, root / name)
            summary = report(root / name, registry)
            manifest["studies"].append({"name": name, "output": str(root/name),
                "rows": len(spec["rows"]), "status": summary["execution_status"],
                "wall_seconds": summary["wall_seconds"]})
            manifest["wall_seconds"] = time.monotonic() - started
            write_json(root / "run-manifest.json", manifest)
            if summary["execution_status"] != "complete":
                raise ValueError(f"{name} has incomplete rows; retain failure for repair")
            rows = []
            for row in spec["rows"]:
                saved = state["rows"][row["id"]]
                result = json.loads((root/name/saved["result_path"]).read_text())
                runtime = result["runtime"]
                if not (runtime["device"] == "GPU" and runtime["tf32"] and runtime["jit_compile"]
                        and runtime["memory_policy"]["all_physical_devices_memory_growth"]
                        and runtime["traces"] == 1):
                    raise ValueError("GPU/XLA/memory/trace evidence mismatch")
                record = {"stage": name, "row": row, "result": result,
                          "path": str(root/name/saved["result_path"]), "digest": saved["result_digest"]}
                rows.append(record)
            numerical.extend(rows)
            return rows

        stage("calibration", study)
        selection = issue_selection(root/"calibration", root/"selection.json")
        claim_study = deepcopy(study)
        claim_study["rows"] = [dict(base_row, id=f"claim-{replicate}", role="claim", dataset=820,
            replicate=replicate, tuning_selection=str(root/"selection.json")) for replicate in range(2)]
        claims = stage("evaluation", claim_study)
        a, b = (r["result"]["diagnostics"] for r in claims)
        if a["fit_digest"] != b["fit_digest"] or a["fit_seed_records"]["iapf_final_process"] == b["fit_seed_records"]["iapf_final_process"]:
            raise ValueError("frozen shared fit or independent final sample check failed")
        comparator_study = deepcopy(study)
        comparator_study["settings"]["particles"] = a["actual_particle_count"]
        comparators = [("kalman",{}),("bootstrap",{}),("twist",{"twist_power":1.}),
            ("fitted_twist",dict(fit_theta=settings["theta"],fit_iterations=2,
                                 fit_initial_variance=4.,fit_floor_ratio=.01))]
        comparator_study["required_proposals"] = [name for name,_ in comparators]
        comparator_study["rows"] = [{**base_row,"id":name,"proposal":name,"role":"mechanics",
            "estimator":"exact_gaussian" if name == "kalman" else "analytical_filter",
            "dataset":820,**extra} for name,extra in comparators]
        stage("comparators", comparator_study)
        if fingerprint(study, registry)["sources"] != source_before:
            raise ValueError("source changed during run")
        fit_calls = sum(r["result"]["diagnostics"].get("work_accounting",{}).get("offline_fit_calls",0)
                        for r in numerical)
        # The log-quadratic comparator performs its declared two recursive fits.
        attempts = len(numerical) + fit_calls + 2
        if attempts > 60:
            raise ValueError("row/recursive-fit attempt budget exceeded")
        manifest.update(status="complete", numerical_rows=len(numerical),
            iapf_recursive_fit_calls=fit_calls, charged_row_or_recursive_fit_attempts=attempts,
            selected_controls=selection["selected_controls"],
            evaluation_realized_particles=a["actual_particle_count"],
            source_fingerprint=source_before,
            evidence_class="mechanics_only_no_matched_cost_or_quality_claim")
        write_json(root/"consumer-evidence.json", {"rows":numerical,
            "same_fit_across_final_replicates":True,"final_streams_independent":True,
            "no_default_or_ranking_claim":True})
        print(json.dumps({k:manifest[k] for k in ("status","numerical_rows",
            "charged_row_or_recursive_fit_attempts","evaluation_realized_particles")}))
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        write_json(root / "run-manifest.json", manifest)


if __name__ == "__main__":
    run()

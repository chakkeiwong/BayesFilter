"""CPU-only independent validation of every saved Lorenz refinement.

NumPy/SciPy are independent-reference-only. This module is never imported by a
BayesFilter runtime. GPU outputs are decoded with devices deliberately hidden.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import tensorflow as tf
import reference_lorenz_stable_fit as reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    checks, fits, fitted_files = [], [], []
    summary = dict(
        status="FAIL", scope="LORENZ_IDENTICAL_INPUT_ALL_REFINEMENTS",
        cpu_only=True, gpu_intentionally_hidden=True,
        numpy_role="INDEPENDENT_REFERENCE_ONLY", replication="INCOMPLETE",
        checks=checks, fits=fits,
        original_independent_trajectory_coefficient_verdict="FAIL_RETAINED",
        nonclaims=["Figure6 replication", "raw coefficients on separate trajectories",
                   "performance ranking", "default readiness"])
    def check(name, passed, **detail):
        checks.append(dict(name=name, passed=bool(passed), **detail))
    def require(name, passed):
        check(name, passed)
        if not passed:
            raise ValueError(name)
    def tensor(name):
        payload = (args.candidate / (name + ".tensor")).read_bytes()
        return tf.io.parse_tensor(payload, tf.float64).numpy()

    try:
        index = json.loads((args.reference / "fixture.json").read_text())
        original_path = args.candidate / "summary.json"
        original = json.loads(original_path.read_text())
        summary["original_summary_sha256"] = hashlib.sha256(original_path.read_bytes()).hexdigest()
        require("original_failure_preserved", original["status"] == "FAIL")
        require("three_cases", len(index["cases"]) == len(original["cases"]) == 3)
        retained = [item for item in original["checks"]
                    if not re.fullmatch(r"case[0-2]_stage[1-4]_[Abc]", item["name"])]
        require("all_120_original_forward_apf_controller_checks_present", len(retained) == 120)
        require("all_original_forward_apf_controller_checks_passed",
                all(item["passed"] for item in retained))
        summary["retained_original_check_names"] = [item["name"] for item in retained]
        summary["original_failed_check_names"] = [item["name"] for item in original["checks"]
                                                  if not item["passed"]]
        for case_id, case in enumerate(index["cases"]):
            iterations = len(case["stages"]) - 1
            require("iteration_count_case%d" % case_id,
                    iterations == original["cases"][case_id]["iterations"] == (4, 1, 3)[case_id])
            local = dict(index["metadata"], forcing=case["forcing"])
            gpu = {key: tensor("case%d_%s" % (case_id, key))[:, :, 0]
                   for key in ("A", "b", "c", "x", "log_incremental_weight")}
            for key, value in gpu.items():
                require("finite_case%d_%s" % (case_id, key), np.all(np.isfinite(value)))
                require("history_shape_case%d_%s" % (case_id, key),
                        value.shape[0] == 5 and value.shape[1] == 101)
            for stage in range(1, iterations + 1):
                source = {key: value[stage - 1] for key, value in gpu.items()}
                fitted, rss, ratios, margins = reference.fitted_policy(local, source)
                require("valid_case%d_stage%d" % (case_id, stage),
                        np.all(np.isfinite(rss)) and np.min(ratios) > 64*np.finfo(float).eps
                        and np.min(margins) > 64*np.finfo(float).eps)
                errors, comparisons = {}, {}
                for key in ("A", "b", "c"):
                    errors[key] = gpu[key][stage] - fitted[key]
                    absolute = np.abs(errors[key])
                    scaled = absolute / (1e-7 + 1e-9*np.abs(fitted[key]))
                    detail = dict(max_error=float(absolute.max()),
                                  max_scaled_error=float(scaled.max()))
                    comparisons[key] = detail
                    check("case%d_stage%d_%s" % (case_id, stage, key),
                          np.all(scaled <= 1), **detail)
                    name = "case%d_stage%d_%s.bin" % (case_id, stage, key)
                    payload = np.asarray(fitted[key], dtype="<f8").tobytes()
                    (args.output / name).write_bytes(payload)
                    fitted_files.append(dict(file=name, shape=list(fitted[key].shape),
                        dtype="float64", sha256=hashlib.sha256(payload).hexdigest()))
                cloud = source["x"]
                policy_error = (np.einsum("hni,hij,hnj->hn", cloud, errors["A"], cloud)
                                + np.einsum("hni,hi->hn", cloud, errors["b"])
                                + errors["c"][:, None])
                fits.append(dict(case=case_id, forcing=case["forcing"], stage=stage,
                                 coefficients=comparisons,
                                 max_policy_error=float(np.abs(policy_error).max()),
                                 minimum_svd_ratio=float(ratios.min()),
                                 minimum_precision_margin=float(margins.min())))
        require("all_eight_refinements_checked", len(fits) == 8)
        summary["status"] = "PASS" if all(item["passed"] for item in checks) else "FAIL"
        summary["interpretation"] = (
            "Scoped identical-input fitting and original forward/APF/controller validation; "
            "attempt70 remains failed under its original coefficient comparison.")
    except Exception as error:
        summary["error"] = repr(error)
    summary["wall_seconds"] = time.monotonic() - started
    (args.output / "fitted-policies.json").write_text(json.dumps(fitted_files, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=summary["status"], fits=len(fits),
                         failed=[item for item in checks if not item["passed"]],
                         wall_seconds=summary["wall_seconds"], error=summary.get("error"))))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

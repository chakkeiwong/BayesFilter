"""Saved-fit CPU reference diagnosis; never tunes or replaces an LEDH route."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads((args.source / "manifest.json").read_text())
    manifest = {"schema": "ksc_saved_information_diagnostic_v1",
                "plan": "docs/plans/ledh-ksc-information-diagnosis-20261017.md",
                "command": [sys.executable, *sys.argv], "cwd": str(Path.cwd()),
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                "python": sys.executable, "cpu_only": True, "cuda_visible_devices": "-1",
                "jit_compile": False, "dtype": "float64", "purpose": "reference/debug exception",
                "source_manifest": str(args.source / "manifest.json"),
                "source_hashes": source["source_hashes"], "input_hashes": {},
                "source_data_version": source["data_version"], "controls": source["controls"],
                "status": "running", "seeds": [], "budget_seconds": 90}
    for relative in ("scripts/diagnose_ledh_ksc_information.py",
                     "bayesfilter/score_study/ksc_information_diagnostic_tf.py"):
        manifest["source_hashes"][relative] = digest(ROOT / relative)
    write_json(args.output / "manifest.json", manifest)
    try:
        for relative, expected in manifest["source_hashes"].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError("source drift: " + relative)
        import tensorflow as tf
        from bayesfilter.highdim import covariance_proposal_tf as core
        from bayesfilter.score_study.ksc_guidance_diagnostic_tf import future_guides, make_clouds
        from bayesfilter.score_study.ksc_information_diagnostic_tf import make_information_probe

        manifest["tensorflow"] = tf.__version__
        manifest["input_hashes"][str(args.source / "manifest.json")] = digest(args.source / "manifest.json")
        theta = tf.constant([1.5, 0.0], tf.float64)
        controls = core.ProposalControls(**source["controls"])
        probes = {"fitted_kernel": make_information_probe(256, controls),
                  "exact_future": make_information_probe(7, controls)}
        rows = []
        for condition, data in source["observations"].items():
            observations = tf.constant(data, tf.float64)
            exact = future_guides(theta, observations)["mixture"]
            for cluster in range(8):
                fit_path = args.source / f"fit-{condition}-kernel-{cluster}.json"
                manifest["input_hashes"][str(fit_path)] = digest(fit_path)
                saved = json.loads(fit_path.read_text())
                fitted = saved["fit"]
                if not fitted["guide_eligible"] or not fitted["fit_valid"]:
                    raise RuntimeError("saved fit is not eligible")
                seed = source["seed_design"]["final_clouds"][condition][cluster][0]
                manifest["seeds"].append({"condition": condition, "cluster": cluster,
                                          "fit": saved["seed"], "cloud": seed})
                ancestors = make_clouds(32, seed)[0]
                parameters = tuple(tf.constant(fitted[key], tf.bool if key == "active" else tf.float64)
                                   for key in ("logits", "centers", "covariances", "active"))
                for family, guide in (("fitted_kernel", parameters), ("exact_future", exact)):
                    result = probes[family](theta, observations, ancestors, *guide)
                    row = {key: value.numpy().tolist() for key, value in result.items()}
                    row.update({"condition": condition, "cluster": cluster,
                                "family": family, "cloud_seed": seed})
                    scalars = [value for value in row.values() if isinstance(value, float)]
                    if not row["valid"] or not all(math.isfinite(value) for value in scalars):
                        raise RuntimeError("invalid accepted diagnostic row")
                    if row["identity_relative_error"] > 1e-9:
                        raise RuntimeError("consumer identity mismatch")
                    if family == "fitted_kernel":
                        gap_error = abs(row["product_variance"] - row["consumer_variance"]
                                        - row["product_between"])
                        if row["kernel_variance_spread"] != 0.0 or gap_error > 1e-9 * max(1.0, row["product_variance"]):
                            raise RuntimeError("homoscedastic covariance identity mismatch")
                        row["gap_identity_error"] = gap_error
                    rows.append(row)
                if time.monotonic() - started > 85:
                    raise RuntimeError("diagnostic budget nearly exhausted")
        if len(rows) != 48 or any(probe.experimental_get_tracing_count() != 1 for probe in probes.values()):
            raise RuntimeError("missing rows or unexpected retracing")
        summaries = []
        for condition in source["observations"]:
            for family in probes:
                selected = [row for row in rows if row["condition"] == condition and row["family"] == family]
                ratios = [row["variance_ratio"] for row in selected]
                omitted_fraction = [row["omitted_score_variance"][0] / row["precision"][0] for row in selected]
                summaries.append({"condition": condition, "family": family, "count": len(selected),
                                  "variance_ratio_min": min(ratios), "variance_ratio_median": statistics.median(ratios),
                                  "variance_ratio_max": max(ratios),
                                  "global_omitted_fraction_min": min(omitted_fraction),
                                  "global_omitted_fraction_max": max(omitted_fraction),
                                  "negative_hessian_anchors": sum(value < 0.0 for row in selected for value in row["negative_hessian"]),
                                  "anchors": sum(len(row["anchors"]) for row in selected)})
        for relative, expected in manifest["source_hashes"].items():
            if digest(ROOT / relative) != expected:
                raise RuntimeError("source changed during diagnosis: " + relative)
        for path, expected in manifest["input_hashes"].items():
            if digest(Path(path)) != expected:
                raise RuntimeError("saved input changed during diagnosis")
        write_json(args.output / "rows.json", rows)
        result = {"status": "pass", "rows": len(rows), "summaries": summaries,
                  "max_identity_relative_error": max(row["identity_relative_error"] for row in rows),
                  "max_gap_identity_error": max(row.get("gap_identity_error", 0.0) for row in rows),
                  "decision": "local covariance diagnosis only; no repair or score promotion"}
        write_json(args.output / "result.json", result)
        manifest["status"] = "pass"
        print(json.dumps(result, allow_nan=False))
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        write_json(args.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()

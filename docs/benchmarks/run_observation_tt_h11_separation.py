#!/usr/bin/env python3
"""H11 fresh guide/fitting separation diagnostic.

The SGQF guide and the existing A06 representation selector are run on fresh
stateless sequences. This is a diagnostic campaign, not a filtering or
promotion route. TensorFlow is the numerical backend; host Python only writes
structured records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PLAN = ROOT / "docs/plans/observation-aware-tt-master-amendment-07-guide-fit-separation-20260915.md"
FIXTURE = ROOT / "docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/diagnostic-02/downstream_fixture.json"


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def write(path: Path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_failed_update(tf, lib, model, predictive, observation, clouds):
    """Preserve structured per-level evidence when sgqf_update rejects all levels."""
    records = []
    for level, cloud in clouds:
        x = predictive.forward(cloud.points)
        logg = model.observation_log_prob(x, observation)
        shift = tf.reduce_max(logg)
        signed = cloud.weights * tf.exp(logg - shift)
        mass = tf.reduce_sum(signed)
        record = {
            "level": level,
            "points": int(cloud.points.shape[0]),
            "negative_rule_weights": int(tf.reduce_sum(tf.cast(cloud.weights < 0, tf.int32)).numpy()),
            "scaled_mass": float(mass.numpy()),
        }
        try:
            lib.finite(mass, "SGQF mass")
            if float(mass.numpy()) <= 0.0:
                raise ValueError("nonpositive SGQF likelihood integral")
            weights = signed / mass
            mean, covariance = lib.gaussian_moments(x, weights)
            eigenvalues = tf.linalg.eigvalsh((covariance + tf.transpose(covariance)) * 0.5)
            maximum = float(tf.reduce_max(tf.abs(eigenvalues)).numpy())
            minimum = float(tf.reduce_min(eigenvalues).numpy())
            bound = 64.0 * lib.EPS * int(covariance.shape[0]) * maximum
            record.update(min_covariance_eigenvalue=minimum,
                          covariance_spd_bound=bound)
            if minimum <= bound:
                raise ValueError(f"chart covariance is not numerically SPD: {minimum} <= {bound}")
            record["status"] = "valid"
        except (ValueError, tf.errors.InvalidArgumentError) as exc:
            record.update(status="invalid", reason=str(exc))
        records.append(record)
    return records


def run_guide(tf, lib, model, observations):
    """Call the production SGQF update while retaining partial failure evidence."""
    clouds = [(level, lib.tf_fixed_sgqf_cloud(model.dimension, level))
              for level in (2, 3, 4, 5)]
    mean, covariance = tf.zeros([model.dimension], lib.D), model.covariance0
    path, records = [], []
    for t, observation in enumerate(tf.unstack(observations)):
        if t:
            mean = tf.linalg.matvec(model.transition, mean)
            covariance = (model.transition @ covariance @ tf.transpose(model.transition)
                          + model.sigma ** 2 * tf.eye(model.dimension, dtype=lib.D))
        predictive = lib.Chart.from_moments(mean, covariance)
        try:
            posterior, info = lib.sgqf_update(model, predictive, observation, clouds)
        except (ValueError, tf.errors.InvalidArgumentError) as exc:
            records.append({"time": t, "status": "failed", "error": repr(exc),
                            "rules": scan_failed_update(tf, lib, model, predictive,
                                                         observation, clouds)})
            return None, records, {"time": t, "error": repr(exc)}
        records.append({"time": t, "status": "complete", "info": info})
        path.append((predictive, posterior))
        mean, covariance = posterior.mean, posterior.factor @ tf.transpose(posterior.factor)
    return path, records, None


def annotate_rules(tf, records):
    """Add scale-aware covariance margins without altering the guide."""
    annotated = []
    for record in records:
        record = dict(record)
        info = record.get("info")
        if info is not None:
            rules = []
            for rule in info.get("rules", []):
                rule = dict(rule)
                covariance = rule.get("covariance")
                if covariance is not None:
                    covariance = tf.convert_to_tensor(covariance)
                    eigenvalues = tf.linalg.eigvalsh(
                        (covariance + tf.transpose(covariance)) * 0.5)
                    maximum = float(tf.reduce_max(tf.abs(eigenvalues)).numpy())
                    rule["min_covariance_eigenvalue"] = float(
                        tf.reduce_min(eigenvalues).numpy())
                    rule["covariance_spd_bound"] = (
                        64.0 * 2.220446049250313e-16
                        * int(covariance.shape[0]) * maximum)
                    rule.pop("covariance", None)
                rules.append(rule)
            record["info"] = dict(info, rules=rules)
        annotated.append(record)
    return annotated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wall-budget-seconds", type=float, default=10800.0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise RuntimeError("H11 requires trusted GPU access")
    growth = []
    for device in physical:
        tf.config.experimental.set_memory_growth(device, True)
        actual = tf.config.experimental.get_memory_growth(device)
        if not actual:
            raise RuntimeError("GPU memory growth policy failed")
        growth.append(dict(name=device.name, memory_growth=actual,
                           details=tf.config.experimental.get_device_details(device)))

    lib = importlib.import_module("bayesfilter.highdim.observation_guided_tt_tf")
    pair = importlib.import_module("bayesfilter.highdim.pair_block_tt_tf")
    joint = importlib.import_module("bayesfilter.highdim.sgqf_joint_consumer_tf")
    projection = importlib.import_module("docs.benchmarks.observation_tt_sgqf_projection_diagnostic")
    base = importlib.import_module("docs.benchmarks.run_observation_aware_tt_complete")
    a06 = importlib.import_module("docs.benchmarks.run_observation_tt_independent_filtering")
    D = tf.float64
    a06.tf, a06.D, a06.lib = tf, D, lib
    a06.pair, a06.joint, a06.projection = pair, joint, projection
    a06.base, a06.logrho = base, importlib.import_module(
        "bayesfilter.highdim.c2_gaussian_hermite_proposal_tf")._log_standard_normal
    dependencies = [Path(__file__).resolve(), PLAN, FIXTURE, Path(a06.__file__),
                    Path(lib.__file__), Path(pair.__file__), Path(joint.__file__),
                    Path(projection.__file__), Path(base.__file__)]
    manifest = dict(
        status="RUNNING", started_utc=datetime.now(timezone.utc).isoformat(),
        command=[sys.executable, *sys.argv], plan=str(PLAN), result=str(out / "result.json"),
        budget_ledger=str(ROOT / "docs/plans/artifacts/observation-tt-continuation-24h-20260915-01/budget.json"),
        proposal_review=str(ROOT / "docs/plans/artifacts/observation-tt-h11-separation-20260915-01/review-proposal-01.txt"),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_status=subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True),
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in dependencies},
        input_hashes={str(FIXTURE.relative_to(ROOT)): sha(FIXTURE)},
        environment=sys.prefix, python=sys.version, tensorflow=tf.__version__,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_only=False, gpu_intentionally_hidden=False, jit_compile=True,
        numerical_dtype="float64", tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        gpu_memory_policy=dict(schema="h11_verified_growth_v1", mode="memory_growth", devices=growth),
        trust_basis="escalated_gpu_access", classification="h11_fresh_guide_fit_diagnostic",
        data=dict(dimensions=[1, 4], sequences_per_dimension=1 if args.smoke else 6,
                  horizon=20, data_seed="926000+100*d+i", fit_seed="927000+10000*d+100*i",
                  A06_observations_reused=False),
        controls=dict(levels=[2, 3, 4, 5], degree=3, rank=3, sweeps=4,
                      proximal_steps=128, l1_grid=[0.0, 1e-5, 1e-3],
                      rows=dict(train=1024, validation=4096, audit=8192),
                      validation_selects=True, audit_used_for_selection=False),
        wall_budget_seconds=args.wall_budget_seconds)
    write(out / "run_manifest.json", manifest)
    result = dict(status="RUNNING", sequences=[], guide_failure_count=0,
                  fit_failure_count=0, audit_loss_count=0,
                  heuristic_dominance_verdict="NOT_EVALUATED",
                  default_ready=False, filter_promotion=False)
    write(out / "result.json", result)

    def budget():
        if time.monotonic() - start > args.wall_budget_seconds:
            raise TimeoutError("H11 per-launch wall budget exhausted")

    fixture = json.loads(FIXTURE.read_text())
    try:
        dimensions = [1] if args.smoke else [1, 4]
        sequence_count = 1 if args.smoke else 6
        for dimension in dimensions:
            data = fixture["dimensions"][str(dimension)]
            model = lib.SVModel(tf.constant(data["A"], D), tf.constant(data["P0"], D),
                                beta=fixture["beta"], sigma=fixture["sigma"])
            generate = a06.data_generator(model, 20)
            for sequence in range(sequence_count):
                budget()
                tic = time.monotonic()
                data_seed = 926000 + 100 * dimension + sequence
                fit_seed = 927000 + 10000 * dimension + 100 * sequence
                dest = out / f"d{dimension}" / f"sequence-{sequence:02d}"
                dest.mkdir(parents=True)
                states, observations = generate(tf.constant(data_seed, tf.int32))
                write(dest / "data.json", dict(seed=data_seed, states=states,
                    observations=observations, A=model.transition, P0=model.covariance0,
                    beta=model.beta, sigma=model.sigma))
                guide, guide_records, guide_failure = run_guide(tf, lib, model, observations)
                guide_records = annotate_rules(tf, guide_records)
                write(dest / "guide.json", dict(seed=data_seed, records=guide_records,
                    failure=guide_failure, complete=guide is not None))
                entry = dict(dimension=dimension, sequence=sequence, data_seed=data_seed,
                             fit_seed=fit_seed, guide_complete=guide is not None,
                             guide_failure=guide_failure, fit_complete=False,
                             audit_loss_count=0, wall_seconds=None)
                if guide is None:
                    result["guide_failure_count"] += 1
                else:
                    fit_dir = dest / "representation"
                    fit_dir.mkdir()
                    try:
                        a06.enhanced_path(model, observations, guide, fit_seed, fit_dir, budget)
                        fit_records = []
                        for fit_path in sorted(fit_dir.glob("fit-t*.json")):
                            record = json.loads(fit_path.read_text())
                            audit_loss = bool(record["audit_loss_vs_sgqf"])
                            fit_records.append(dict(time=record["selection"]["time"],
                                selected=record["selection"]["selected"],
                                validation_h2=record["selection"]["validation_h2"],
                                audit_h2=record["audit_h2"],
                                audit_loss_vs_sgqf=audit_loss))
                        entry["fit_complete"] = True
                        entry["fit_records"] = fit_records
                        entry["audit_loss_count"] = sum(r["audit_loss_vs_sgqf"] for r in fit_records)
                        result["audit_loss_count"] += entry["audit_loss_count"]
                    except (ValueError, tf.errors.OpError) as exc:
                        entry["fit_failure"] = repr(exc)
                        write(dest / "fit-failure.json", dict(error=repr(exc),
                            traceback=traceback.format_exc()))
                        result["fit_failure_count"] += 1
                entry["wall_seconds"] = time.monotonic() - tic
                result["sequences"].append(entry)
                write(out / "result.json", result)
        result["status"] = "COMPLETE"
        result["heuristic_dominance_verdict"] = (
            "PROMOTION_VETO" if result["guide_failure_count"] or result["fit_failure_count"]
            or result["audit_loss_count"] else "SCREEN_PASSED_NO_PROMOTION")
        result["decision"] = (
            "Guide or representation veto recorded; no downstream filtering phase admitted."
            if result["heuristic_dominance_verdict"] == "PROMOTION_VETO" else
            "Fresh guide and audit screens passed; write a separate reviewed filtering amendment.")
        result["decision_table"] = [
            dict(decision="guide robustness", status="VETO" if result["guide_failure_count"] else "SCREEN_PASSED",
                 criterion="zero fresh complete-path guide failures", veto="any guide failure"),
            dict(decision="TT representation", status="VETO" if result["fit_failure_count"] or result["audit_loss_count"] else "SCREEN_PASSED",
                 criterion="finite fits and no audit loss versus exact SGQF joint",
                 veto="fit failure or audit loss"),
            dict(decision="promotion/default", status="NOT_EVALUATED",
                 criterion="requires a later independent filtering contract", veto="this phase cannot promote"),
        ]
        result["inference_status"] = [
            dict(item="hard veto screen", status="FAILED" if result["guide_failure_count"] or result["fit_failure_count"] or result["audit_loss_count"] else "PASSED"),
            dict(item="statistically supported ranking", status="NOT_TESTED"),
            dict(item="descriptive differences", status="RECORDED_ONLY"),
            dict(item="default readiness", status="NOT_ESTABLISHED"),
            dict(item="next evidence needed", status="reviewed conditional filtering phase"),
        ]
        manifest["status"] = "COMPLETE"
    except Exception as exc:
        result.update(status="FAILED", failure_type=type(exc).__name__, failure=repr(exc))
        manifest.update(status="FAILED", failure_type=type(exc).__name__, failure=repr(exc))
        (out / "traceback.txt").write_text(traceback.format_exc())
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - start
        manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["source_unchanged_at_close"] = {
            str(p.relative_to(ROOT)): sha(p) for p in dependencies
        }
        manifest["gpu_allocator"] = {d.name: tf.config.experimental.get_memory_info(
            "GPU:" + d.name.split(":")[-1]) for d in physical}
        write(out / "result.json", result)
        write(out / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()

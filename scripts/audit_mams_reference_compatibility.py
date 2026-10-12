#!/usr/bin/env python3
"""Independent diagnostic bridge from archived finite-filter references.

NumPy/SciPy are used only for independent reference parity, never by a sampler.
No tuning, transport, posterior convergence or method-ranking authority is issued.
Reviewed exclusions are exact old/new source pairs, not a skip-source option.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

REFERENCES = {
    "NL-PP-SGQF": "predator-prey-reference-04/nl-pp-sgqf",
    "NL-PP-UKF": "predator-prey-reference-05/nl-pp-ukf",
    "NL-SIR": "sir-reference-02/reference",
    "NL-STR": "structural-influence-reference-01/reference",
}
SOURCE_NOTE = "docs/plans/bayesfilter-mams-reference-source-addendum-2026-10-12.md"
PLAN = "docs/plans/bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md"
# These exact diffs were inspected against the archived PP/SIR/structural
# references. Any new difference needs its own dependency/mathematical audit.
REVIEWED_PAIRS = {
    "bayesfilter/inference/hmc_acceptance_statistics.py": (
        "c0a4499ff35d244e0eb6e1b08f5476c692a070ce4d013cf7a1403e4c7cf81dcd",
        "81bd3bb8fcc14d932a494d1d90cf01688f2f3618d6c5bf660821c2030c67eb69",
        "adds only a target-independent preflight function; existing executable AST unchanged"),
    "bayesfilter/inference/__init__.py": (
        "e5345996987a055188c10718e274ed490356cf05302b1395f20eccd5a0c96020",
        "1408b983565723153e48cf344c5571aad06a09ce23dff81729414adf58122b14", "lazy exports only"),
    "bayesfilter/inference/hmc_candidate_set_artifacts.py": (
        "d3d0595d4daf8df7aba1a91287ba4240337b361432b040bc3d0bf02f142dcd9f",
        "579df6152d5284677f9fafcb89fc6b2651845a6ec8bda8ff502add0636e04d06", "tuning artifact validation only"),
    "bayesfilter/inference/hmc_candidate_set_execution.py": (
        "9a82c18a421bb81a4800688598b2e5d51eaffec19f356d7df9b5ffe4378a6e14",
        "a963d51c00762fbc42d75030d382cc74691b54eb83890d19cd299ef5fed8ba6f", "tuning source inventory only"),
    "bayesfilter/inference/hmc_candidate_set_tuning.py": (
        "2fcf14414fd7c2ed448fe519aebd6713980b284fa800b79c394700e9bb850d3e",
        "6ba7504dbf3336958d5452cb41dc0efc30e60b0a80fc5a48b676de2cfdb06dff", "tuner repair scheduling only"),
    "bayesfilter/inference/neutra_transport.py": (
        "1609d8c4df6d04cf470950db7bbf94ee1de473f7733fad4fe51f3f8be7b87a84",
        "d688b2b720dc9e30a69601df2d6fda647564b098481f84bdcc6371ad56e48ab2", "unused learned transport configuration/training"),
    "bayesfilter/inference/neutra_transport_core.py": (
        "48d1bc6bc2bc293476462dc932d2e6259838089e9f311cddf8e9c7e552ab491f",
        "e1bd6f40f0b7041e7844ace8b8e404ab812cc518719c447d8cd9bd5275d70e2f", "unused learned transport operations"),
    "bayesfilter/inference/neutra_weighted_training.py": (
        "0a3937fd152c352fe878d927f8feaa54ecfc20b24c51be01725bdf66646c1cb9",
        "1a38cb01572a815c0fc95c70bc3237fffe4d26a2db5f9b25d9926b554f0c306d", "module documentation only"),
    "bayesfilter/inference/tuning_contract.py": (
        "9e24bf0b60a384a1a05003fa536461af0eb43b66bfd2061d32bc1c58973734bc",
        "61893a24a757bffa36f4f55c0ba0adb360d2783a99f77ed74cd82542071448bd", "capability description strings only"),
    "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py": (
        "1072bce8d077b3d0cabba90cc215667365e0329a22d9ca470b8bde8bfe7bb4b6",
        "cd5d0341fc2f64b4eb304c01f99fcb7d0e56763cc17af727977d700def7964c0", "unchanged True compilation default on actual UKF call path"),
}
ARCHIVED_ONLY = {
    "gaussian_mixture_reference.py", "modern_predator_prey.py", "modern_sir.py",
    "modern_structural.py", "modern_structural_mixture.py",
}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def relative_source(name):
    parts = Path(name).parts
    for index, part in enumerate(parts):
        if part in ("bayesfilter", "scripts"):
            return Path(*parts[index:])
    raise ValueError("unrecognized archived source path: " + str(name))


def quantity_names(target):
    if target.startswith("NL-PP") and target in REFERENCES:
        return ["r", "capacity", "half_saturation", "s", "u", "v"] + [
            f"{prefix}_{i}" for prefix in ("second", "upper") for i in range(6)]
    names = {"NL-SIR": ("kappa", "nu", "observation_sd"),
             "NL-STR": ("rho", "sigma", "phi", "gamma", "R")}[target]
    return [f"{prefix}_{name}" for prefix in ("physical", "source_square", "source_tail") for name in names]


def model_quantities(source, *, target):
    """TF diagnostic coordinates; source is native theta, including SIR SD .5."""
    import tensorflow as tf
    if target not in REFERENCES:
        raise ValueError("unsupported reference target")
    q = tf.convert_to_tensor(source, tf.float64)
    tf.debugging.assert_equal(tf.shape(q)[-1], len(quantity_names(target)) // 3)
    if target == "NL-SIR":
        physical = tf.constant([.1, 18., 10.], tf.float64) * tf.exp(q)
        z = q / tf.constant(.5, tf.float64)
    else:
        lower, upper = (([.05, .05, .05, .02, .02], [.98, 1.25, .98, 1., 1.])
                        if target == "NL-STR" else
                        ([.1, 110., 20., .1, 0., 0.], [1.1, 130., 30., 1.1, 1., 1.]))
        lo, hi = tf.constant(lower, tf.float64), tf.constant(upper, tf.float64)
        z = q
        physical = lo + (hi-lo) * .5 * tf.math.erfc(-z/tf.sqrt(tf.constant(2., tf.float64)))
    return tf.concat((physical, z*z, tf.sigmoid(2*(z-1))), axis=-1)


def audit_sources(reference, current, repository):
    reference, current, repository = map(Path, (reference, current, repository))
    design = read(reference / "design.json")
    archived, current_hashes, changes = {}, {}, []
    if not design["source_closure"]:
        raise ValueError("empty archived source closure")
    for recorded, expected in design["source_closure"].items():
        relative = relative_source(recorded)
        old, new = reference / "source" / relative, current / relative
        if not old.is_file() or digest(old) != expected:
            raise ValueError("missing or altered archived source: " + str(relative))
        archived[str(old.resolve())] = expected
        observed = digest(new) if new.is_file() else None
        if observed is not None:
            current_hashes[str(relative)] = observed
        if observed == expected:
            continue
        approved = REVIEWED_PAIRS.get(str(relative))
        if approved is not None and (expected, observed) == approved[:2]:
            reason = approved[2]
        elif observed is None and (
                (relative.parent == Path("bayesfilter/testing/inference_validation/references")
                 and relative.name in ARCHIVED_ONLY)
                or str(relative) in {
                    "scripts/diagnose_modern_hmc_predator_prey_reference.py",
                    "scripts/diagnose_modern_hmc_sir_reference.py",
                    "scripts/diagnose_modern_hmc_structural_reference.py"}):
            reason = "independent reference source retained and checked in original archive"
        else:
            raise ValueError("unreviewed source difference: " + str(relative))
        changes.append(dict(path=str(relative), archived_sha256=expected,
                            current_sha256=observed, reason=reason))
    inputs = {}
    for name, expected in design["inputs"].items():
        path = Path(name)
        if not path.is_absolute():
            path = repository / path
        if not path.is_file() or digest(path) != expected:
            raise ValueError("reference input changed: " + str(path))
        inputs[str(path.resolve())] = expected
    return dict(archived_source_sha256=archived, current_source_sha256=current_hashes,
                input_sha256=inputs, reviewed_differences=changes)


def finite_vector(values, size, name, *, positive=False):
    if (not isinstance(values, list) or len(values) != size
            or any(type(x) not in (int, float) or not math.isfinite(x)
                   or (positive and x <= 0) for x in values)):
        raise ValueError("invalid reference vector: " + name)


def reference_summary(directory, *, target, data_sha256, data):
    """Recheck every scramble and the final two orders, never a success flag alone."""
    directory = Path(directory)
    design, result, parity = (read(directory / name) for name in ("design.json", "result.json", "parity.json"))
    metadata = design["target"]
    dimension = 3 if target == "NL-SIR" else 5 if target == "NL-STR" else 6
    horizon = 100 if target == "NL-STR" else 20
    actual_target = result["target"].get("target_id") if isinstance(result["target"], dict) else result["target"]
    if (metadata["target_id"] != target or actual_target != target
            or metadata["horizon"] != horizon or metadata["full_horizon"] != horizon
            or metadata["parameter_dim"] != dimension or len(design["data"]) != horizon):
        raise ValueError("reference target, dimension or horizon mismatch")
    if metadata["data_sha256"] != data_sha256 or design["data"] != data:
        raise ValueError("reference observations or hash mismatch")
    names = quantity_names(target)
    if target in ("NL-SIR", "NL-STR") and design.get("quantity_names") != names:
        raise ValueError("reference quantity chart mismatch")
    if target == "NL-SIR" and design.get("standardization") != "theta=0.5*z; log_jacobian=3*log(0.5)":
        raise ValueError("SIR reference standardization mismatch")
    finite_vector([parity["value_max_absolute_error"], *parity["score_max_scaled_errors"]], 3, "original parity")
    if (result["status"] != "empirically_refined" or parity.get("passed") is not True
            or min([parity["value_max_absolute_error"], *parity["score_max_scaled_errors"]]) < 0
            or parity["value_max_absolute_error"] > 1e-8
            or max(parity["score_max_scaled_errors"]) > 2e-5):
        raise ValueError("reference refinement or original parity failed")
    if design.get("reference_tolerance", design.get("relative_tolerance")) != .002 or design["replications"] != 8:
        raise ValueError("reference precision design changed")
    powers = design["powers"]
    if (len(powers) < 2 or any(type(p) is not int or not 3 <= p <= 30 for p in powers)
            or sorted(set(powers)) != powers):
        raise ValueError("reference requires distinct increasing integration orders")
    if [r["power"] for r in result["rungs"]] != powers:
        raise ValueError("reference result and declared orders differ")
    paths = [directory / f"rung-{p:02d}.json" for p in powers[-2:]]
    previous, final = map(read, paths)
    for power, rung in zip(powers[-2:], (previous, final)):
        if (rung["power"] != power or rung["points_per_replication"] != 2**power
                or rung["replications"] != 8 or len(rung["rows"]) != 8
                or any(type(r["replication"]) is not int for r in rung["rows"])
                or {r["replication"] for r in rung["rows"]} != set(range(8))):
            raise ValueError("incomplete or duplicate reference scramble")
        for row in [rung, *rung["rows"]]:
            finite_vector(row["mean"], len(names), "mean")
            finite_vector(row["sd"], len(names), "sd", positive=True)
        finite_vector(rung["pointwise_half_width"], len(names), "pointwise_half_width")
        finite_vector(rung["randomization_mcse"], len(names), "randomization_mcse")
        if min(rung["pointwise_half_width"] + rung["randomization_mcse"]) < 0:
            raise ValueError("negative reference uncertainty")
        # The original independent drivers use the same eight-scramble mean,
        # RMS within-scramble SD and t(.975, 7) interval. 1e-12 is a diagnostic
        # FP64 aggregation tolerance, not a posterior precision threshold.
        for i in range(len(names)):
            values = [r["mean"][i] for r in rung["rows"]]
            mean = math.fsum(values)/8
            sd = math.sqrt(math.fsum(r["sd"][i]**2 for r in rung["rows"])/8)
            se = math.sqrt(math.fsum((v-mean)**2 for v in values)/(8*7))
            half = 2.3646242515927844 * se  # scipy.stats.t.ppf(.975, 7)
            for key, expected in (("mean", mean), ("sd", sd), ("randomization_mcse", se), ("pointwise_half_width", half)):
                if not math.isclose(rung[key][i], expected, rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("reference summary differs from scramble rows: " + key)
    if target == "NL-STR" and result["final_rung"] != paths[-1].name:
        raise ValueError("wrong final reference order")
    change = [abs(a-b) for a, b in zip(final["mean"], previous["mean"])]
    if any(delta/sd > .002 or half/sd > .002 for delta, half, sd in
           zip(change, final["pointwise_half_width"], final["sd"])):
        raise ValueError("reference has not met unchanged empirical precision")
    files = [directory / name for name in ("design.json", "result.json", "parity.json")] + paths
    if (directory / "frozen-proposal.json").exists():
        files.append(directory / "frozen-proposal.json")
    return dict(target=target, data_sha256=data_sha256, quantities=names,
                mean=final["mean"], sd=final["sd"],
                uncertainty=[a+b for a,b in zip(change, final["pointwise_half_width"])],
                reference_files_sha256={str(p.resolve()): digest(p) for p in files},
                exact_nonlinear_posterior=False, rigorous_integration_error_bound=False)


def archived_reference_module(directory, target):
    suffix = hashlib.sha256(str(Path(directory).resolve()).encode()).hexdigest()[:12]
    package_name = "_mams_archived_reference_" + target.replace("-", "_") + suffix
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path(directory) / "source/bayesfilter/testing/inference_validation/references")]
    sys.modules[package_name] = package
    name = "modern_sir" if target == "NL-SIR" else "modern_structural" if target == "NL-STR" else "modern_predator_prey"
    return importlib.import_module(package_name + "." + name)


def original_points(directory, target):
    parity = read(Path(directory) / "parity.json")
    if target != "NL-SIR":
        return parity["source_points" if target == "NL-STR" else "points"]
    center = [x*.5 for x in read(Path(directory) / "design.json")["proposal"]["centers"][1]]
    offset = [.01, -.01, .01]  # original archived SIR driver's parity points
    return [[0.]*3, center, [c+x for c,x in zip(center, offset)], [c-x for c,x in zip(center, offset)]]


def check_parity(directory, target, fixture, prepared_starts):
    import numpy as np  # independent diagnostic only
    import tensorflow as tf
    reference = archived_reference_module(directory, target)
    q = np.asarray(original_points(directory, target) + prepared_starts, dtype=float)
    data = fixture.observations.numpy()
    native, score, status = fixture.adapter.log_prob_and_grad_status(tf.constant(q, tf.float64))
    valid = ((status["status_code"] == 0) & status["valid_pre_regularized_score"] & (status["floor_count_value"] == 0))
    if not bool(tf.reduce_all(valid)):
        raise ValueError("current native target invalid at reference/prepared points")
    def value(points):
        return reference.log_posterior(points, data, **({"method": target} if target.startswith("NL-PP") else {}))
    independent = value(q)
    errors = []
    for step in (1e-4, 1e-5):
        gradient = np.stack([(value(q+step*np.eye(q.shape[1])[j])-value(q-step*np.eye(q.shape[1])[j]))/(2*step)
                             for j in range(q.shape[1])], axis=1)
        errors.append(float(np.max(np.abs(gradient-score.numpy())/(1+np.abs(score.numpy())))))
    if target == "NL-SIR":
        # Execute only the archived quantity definition, not its campaign driver.
        path = Path(directory) / "source/scripts/diagnose_modern_hmc_sir_reference.py"
        definition = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "quantities")
        namespace = {}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
        independent_quantities = namespace["quantities"](q/.5)
    else:
        independent_quantities = reference.quantities(q)
    quantities = model_quantities(q, target=target).numpy()
    value_error = float(np.max(np.abs(independent-native.numpy())))
    quantity_error = float(np.max(np.abs(independent_quantities-quantities)/(1+np.abs(quantities))))
    finite = all(np.isfinite(v).all() for v in (independent, native.numpy(), score.numpy(), independent_quantities, quantities))
    passed = bool(finite and value_error <= 1e-8 and max(errors) <= 2e-5 and quantity_error <= 1e-12)
    return dict(passed=passed, points=q.tolist(), original_point_count=4,
                prepared_point_count=len(prepared_starts), native_values=native.numpy().tolist(),
                independent_values=independent.tolist(), value_max_absolute_error=value_error,
                score_max_scaled_errors=errors, finite_difference_steps=[1e-4, 1e-5],
                quantity_max_scaled_error=quantity_error, all_finite=finite,
                native_status_valid=True, tolerances=dict(value=1e-8, score=2e-5, quantity=1e-12))


def load_compatible_reference(report_path, *, current_source, target, data_sha256, data):
    """Diagnostic-only consumer; reject changed current or archived evidence."""
    report = read(report_path)
    if (report.get("status") != "compatible" or report.get("target") != target
            or report.get("schema") != "bayesfilter.mams_reference_compatibility.v1"):
        raise ValueError("wrong or incomplete reference compatibility report")
    if any(not report["sources"][group] for group in ("current_source_sha256", "archived_source_sha256", "input_sha256")):
        raise ValueError("incomplete reference provenance")
    for name, expected in report["sources"]["current_source_sha256"].items():
        if digest(Path(current_source) / name) != expected:
            raise ValueError("current source changed since compatibility audit: " + name)
    for group in (report["sources"]["archived_source_sha256"], report["sources"]["input_sha256"], report["audit_files_sha256"],
                  report["summary"]["reference_files_sha256"]):
        for name, expected in group.items():
            if digest(name) != expected:
                raise ValueError("archived compatibility evidence changed: " + name)
    parity = report["parity"]
    finite_vector([parity["value_max_absolute_error"], *parity["score_max_scaled_errors"],
                   parity["quantity_max_scaled_error"]], 4, "current parity")
    if (parity["passed"] is not True or parity["all_finite"] is not True
            or parity["native_status_valid"] is not True or parity["original_point_count"] != 4
            or parity["prepared_point_count"] != 4 or len(parity["points"]) != 8
            or min([parity["value_max_absolute_error"], *parity["score_max_scaled_errors"], parity["quantity_max_scaled_error"]]) < 0
            or parity["value_max_absolute_error"] > 1e-8 or max(parity["score_max_scaled_errors"]) > 2e-5
            or parity["quantity_max_scaled_error"] > 1e-12):
        raise ValueError("current reference parity failed")
    summary = reference_summary(report["reference_directory"], target=target, data_sha256=data_sha256, data=data)
    if summary != report["summary"]:
        raise ValueError("reference compatibility identity/summary mismatch")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-source", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", nargs="+", choices=tuple(REFERENCES), default=list(REFERENCES))
    args = parser.parse_args()
    args.current_source, args.repository, args.geometry_root, args.output = (
        p.resolve() for p in (args.current_source, args.repository, args.geometry_root, args.output))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("independent reference check must hide GPUs before import")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(command=[sys.executable, *sys.argv], plan=PLAN,
        current_source=str(args.current_source.resolve()), **read(args.current_source / ".validation-source.json"),
        diagnostic_driver_sha256=digest(__file__), source_audit_note_sha256=digest(args.repository / SOURCE_NOTE),
        environment=sys.executable, started_utc=datetime.now(timezone.utc).isoformat(),
        gpu_intentionally_hidden=True, role="independent_reference_compatibility_only", status="running",
        rows=[dict(target=t, status="not_started") for t in args.targets])
    write(args.output / "manifest.json", manifest)
    sys.path.insert(0, str(args.current_source.resolve()))
    import tensorflow as tf
    import numpy as np
    import scipy
    from scripts.price_mams_adaptation_full_models import prepare
    from bayesfilter.testing.inference_validation.modern_bridge_audit import source_dependencies
    manifest["versions"] = dict(tensorflow=tf.__version__, numpy=np.__version__, scipy=scipy.__version__)
    for row in manifest["rows"]:
        tick = time.monotonic()
        target = row["target"]
        directory = args.geometry_root / REFERENCES[target]
        row.update(status="running", stage="source_and_native_preparation")
        write(args.output / "manifest.json", manifest)
        try:
            sources = audit_sources(directory, args.current_source, args.repository)
            seed = 5400 + 10000*list(REFERENCES).index(target)
            _, fixture, metadata = prepare(target, args.geometry_root, seed, use_xla=False)
            # Include the current target closure, including additions since the
            # historical run, rather than just its old subset.
            paths = source_dependencies([*fixture.source_paths,
                args.current_source / "scripts/price_mams_adaptation_full_models.py",
                args.current_source / "bayesfilter/testing/inference_validation/modern_nonlinear.py",
                args.current_source / "bayesfilter/testing/inference_validation/modern_prepared_nonlinear.py"],
                repository_root=args.current_source)
            sources["current_source_sha256"].update({str(Path(p).relative_to(args.current_source)): digest(p) for p in paths})
            operator = Path("bayesfilter/ops/_symmetric_sylvester_ops.so")
            if digest(args.current_source / operator) != manifest["compiled_operator_sha256"][str(operator)]:
                raise ValueError("current native operator differs from committed snapshot record")
            sources["current_source_sha256"].update({str(p): digest(args.current_source / p) for p in (
                operator, Path("bayesfilter/ops/symmetric_sylvester_op.cc"))})
            summary = reference_summary(directory, target=target,
                data_sha256=fixture.metadata["data_sha256"], data=fixture.observations.numpy().tolist())
            row["stage"] = "native_reference_parity"
            write(args.output / "manifest.json", manifest)
            parity = check_parity(directory, target, fixture, metadata["start_positions"].numpy().tolist())
            report = dict(schema="bayesfilter.mams_reference_compatibility.v1", target=target,
                status="compatible" if parity["passed"] else "parity_failed",
                reference_directory=str(directory.resolve()), current_source=str(args.current_source.resolve()),
                audit_files_sha256={str(Path(__file__).resolve()): digest(__file__),
                    str(args.repository / SOURCE_NOTE): digest(args.repository / SOURCE_NOTE)},
                sources=sources, summary=summary, parity=parity, prepared_seed=seed,
                tuning_artifact_authority=False, posterior_convergence_authority=False,
                exact_nonlinear_posterior=False, source_audit_note=SOURCE_NOTE)
            report_path = args.output / (target + ".json")
            write(report_path, report)
            if parity["passed"]:
                load_compatible_reference(report_path, current_source=args.current_source, target=target,
                    data_sha256=fixture.metadata["data_sha256"], data=fixture.observations.numpy().tolist())
            row.update(status=report["status"], report=str(report_path), parity=parity)
        except Exception as error:
            row.update(status="repair_required", error_type=type(error).__name__, error=str(error))
        row["elapsed_seconds"] = time.monotonic()-tick
        write(args.output / "manifest.json", manifest)
    manifest.update(status="compatible" if all(r["status"] == "compatible" for r in manifest["rows"]) else "repair_required",
        elapsed_seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "elapsed_seconds")}))
    return 0 if manifest["status"] == "compatible" else 1


if __name__ == "__main__":
    sys.exit(main())

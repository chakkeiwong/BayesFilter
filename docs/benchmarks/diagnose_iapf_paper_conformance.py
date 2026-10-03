"""CPU-only independent paper/reference audit; never an algorithm runtime.

The oracle uses scalar/diagonal Gaussian algebra from GJL (2017), equations
(5)--(6), Proposition 1, Algorithms 3--5, and equations (15)--(16).
Source mismatches are results, not successful paper-reference admission.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ".localresources/code/sempreteamo-iapf-a8811439/iapf.R"
PAPER = ROOT / ".localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf"
PROBE = ROOT / "docs/benchmarks/diagnose_iapf_public_reference.R"
PLAN = "docs/plans/iapf-public-reference-paper-conformance-2026-09-19.md"
SOURCE_SHA256 = "bf6f283b3eac0c3ef0a390af6be522af04026d47dcd6b1f52e606db6c5719851"
SCHEMA = "iapf_public_reference_paper_conformance_v1"
CORE = frozenset({"source_identity", "initial_proposal", "transition_proposal",
    "backward_normalizers", "incremental_weights", "path_density_identity",
    "recursive_targets", "apf_retain", "apf_resample", "exact_twist_likelihood",
    "likelihood_cv", "fresh_final_run", "particle_schedule"})
REQUIRED = CORE | {"outer_stopping_index", "paper_fit_objective", "positive_floor",
                   "single_observation_domain"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_reference(path):
    result = {}
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            values = result.setdefault(row["case"], {}).setdefault(row["field"], [])
            if int(row["index"]) != len(values) + 1:
                raise ValueError("duplicate or unordered reference field")
            value = float(row["value"])
            if not math.isfinite(value):
                raise ValueError("nonfinite reference observation")
            values.append(value)
    return result


def run_reference(output, source=SOURCE):
    """Run actual pinned function bodies with recorded diagnostic substitutions."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1")
    with (output / "r-reference.log").open("w") as log:
        completed = subprocess.run(["Rscript", str(PROBE), str(source),
            str(output / "r-reference.csv")], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False)
    if completed.returncode:
        raise RuntimeError(f"R source probe failed ({completed.returncode}); see r-reference.log")
    return read_reference(output / "r-reference.csv")


def normal(x, mean, variance):
    return math.exp(-.5 * ((x - mean)**2 / variance + math.log(2 * math.pi * variance)))


def logmeanexp(values):
    maximum = max(values)
    return maximum + math.log(sum(math.exp(x - maximum) for x in values) / len(values))


def product_normal(mean, twist_mean, twist_variance):
    """N(mean,1) times N(twist_mean,twist_variance), normalized."""
    variance = twist_variance / (1 + twist_variance)
    return variance * (mean + twist_mean / twist_variance), variance


def gaussian_filter_oracle(model, resample):
    """Independent Algorithm-5 calculation with supplied innovations/ancestors."""
    observations, twists, noise = (model[k] for k in ("observations", "twists", "noise"))
    transition = model["transition"][0]
    particles = len(noise) // len(observations)
    previous, weights, clouds, all_weights, probabilities = [], [], [], [], []
    log_likelihood = 0.
    for t, observation in enumerate(observations):
        if t and resample:
            log_likelihood += logmeanexp(weights)
            mass = sum(math.exp(w - max(weights)) for w in weights)
            probabilities.append([math.exp(w - max(weights)) / mass for w in weights])
            ancestors = ([2, 0, 1], [1, 2, 0])[t - 1]
        else:
            ancestors = list(range(particles))
        cloud, updated = [], []
        center, variance = twists[2*t:2*t+2]
        for i in range(particles):
            prior_mean = transition * previous[ancestors[i]] if t else 0.
            mean, var = product_normal(prior_mean, center, variance)
            x = mean + math.sqrt(var) * noise[t*particles+i]
            future = normal(transition*x, twists[2*t+2], 1+twists[2*t+3]) if t+1 < len(observations) else 1.
            potential = normal(observation, x, 1.) * future / normal(x, center, variance)
            if t == 0:
                potential *= normal(0., center, 1+variance)
            log_weight = math.log(potential)
            if t and not resample:
                log_weight += weights[i]
            cloud.append(x)
            updated.append(log_weight)
        clouds.extend(cloud)
        all_weights.extend(updated)
        previous, weights = cloud, updated
    return dict(clouds=clouds, log_weights=all_weights,
                log_likelihood=[log_likelihood+logmeanexp(weights)],
                probabilities=probabilities)


def assess_reference(reference, source=SOURCE):
    checks = []

    def compare(name, layer, anchor, actual, expected, note="", tolerance=2e-10):
        actual, expected = list(actual), list(expected)
        finite = all(math.isfinite(x) for x in actual + expected)
        delta = max((abs(a-b) for a,b in zip(actual, expected)), default=0.)
        passed = finite and len(actual) == len(expected) and all(
            abs(a-b) <= tolerance*(1+abs(b)) for a,b in zip(actual, expected))
        checks.append(dict(name=name, layer=layer, paper_anchor=anchor,
            passed=passed, maximum_absolute_error=delta, actual=actual,
            expected=expected, tolerance=tolerance, note=note))

    compare("source_identity", "provenance", "pinned public source; author identity unverified",
            [int(sha(source) == SOURCE_SHA256)], [1], tolerance=0.)
    comp = reference["components"]
    centers = [comp["psi"][4*t:4*t+2] for t in range(3)]
    variances = [comp["psi"][4*t+2:4*t+4] for t in range(3)]
    matrix, x = comp["A"], comp["x"]
    ax = [sum(matrix[2*i+j]*x[j] for j in range(2)) for i in range(2)]
    for label, t, prior in (("initial", 0, [0., 0.]), ("transition", 1, ax)):
        values = [product_normal(prior[j], centers[t][j], variances[t][j]) for j in range(2)]
        expected = [v[0] for v in values] + [values[0][1], 0., 0., values[1][1]]
        compare(label+"_proposal", "core", "equation (5)",
            reference[label]["mean"]+reference[label]["covariance"], expected)
    future_actual, future_expected, weight_actual, weight_expected = [], [], [], []
    observations = reference["fit"]["observations"]
    initial_z = math.prod(normal(0., centers[0][j], 1+variances[0][j]) for j in range(2))
    for t in range(3):
        value = reference[f"weight{t+1}"]
        future = math.prod(normal(ax[j], centers[t+1][j], 1+variances[t+1][j]) for j in range(2)) if t < 2 else 1.
        twist = math.prod(normal(x[j], centers[t][j], variances[t][j]) for j in range(2))
        obs = math.prod(normal(observations[2*t+j], x[j], 1.) for j in range(2))
        future_actual.append(value["log_future"][0]); future_expected.append(math.log(future))
        weight_actual.extend(value["log_psi"]+value["log_weight"])
        weight_expected.extend([math.log(twist), math.log(obs*future/twist*(initial_z if t == 0 else 1.))])
    compare("backward_normalizers", "core", "definitions before (5)", future_actual, future_expected)
    compare("incremental_weights", "core", "equation (6), including initial normalizer", weight_actual, weight_expected)

    model, path = reference["paper_model"], reference["paper_path"]
    twisted_joint, original_joint = 1., 1.
    for t, state in enumerate(path["states"]):
        mean, var = path[f"proposal{t+1}"]
        twisted_joint *= normal(state, mean, var)*path[f"weight{t+1}"][0]
        prior_mean = model["transition"][0]*path["states"][t-1] if t else 0.
        original_joint *= normal(state, prior_mean, 1.)*normal(model["observations"][t], state, 1.)
    compare("path_density_identity", "core", "Proposition 1", [math.log(twisted_joint)], [math.log(original_joint)])
    targets, expected = [], []
    for t in range(3):
        targets.extend(reference["paper_backward"][f"target{t+1}"])
        for state in reference["paper_backward"]["cloud"]:
            future = normal(model["transition"][0]*state, model["twists"][2*t+2],
                            1+model["twists"][2*t+3]) if t < 2 else 1.
            expected.append(normal(model["observations"][t], state, 1.)*future)
    compare("recursive_targets", "core", "Algorithm 3 step 1", targets, expected,
            "Actual Psi targets captured at optimizer boundary; prescribed next-twist parameters.")
    for resample in (False, True):
        label = "apf_resample" if resample else "apf_retain"
        observed = reference["paper_resample" if resample else "paper_retain"]
        expected = gaussian_filter_oracle(model, resample)
        actual_values, expected_values = [], []
        for key in ("clouds", "log_weights", "log_likelihood"):
            actual_values.extend(observed[key]); expected_values.extend(expected[key])
        actual_values += observed["sample_calls"]+observed["noise_consumed"]
        expected_values += [2 if resample else 0, 9]
        if resample:
            for i, probabilities in enumerate(expected["probabilities"], 1):
                actual_values.extend(observed[f"ancestor_probabilities{i}"])
                expected_values.extend(probabilities)
                actual_values.extend(observed[f"ancestors{i}"])
                expected_values.extend(([3,1,2], [2,3,1])[i-1])
        compare(label, "core", "Algorithm 5 and following likelihood formula", actual_values, expected_values)
    exact = sum(math.log(normal(y, 0., 2.)) for y in reference["full_filter"]["observations"])
    compare("exact_twist_likelihood", "core", "Proposition 2, independent-state Gaussian fixture",
            reference["full_filter"]["log_likelihood"], [exact])

    cvs, expected_cvs = [], []
    for name in ("partial", "first_complete", "next_complete", "oscillating", "shifted"):
        row = reference[name]; logs = row["history"][-3:]
        values = [math.exp(v-max(logs)) for v in logs]
        cvs.append(row["controller"][0]); expected_cvs.append(statistics.stdev(values)/statistics.mean(values))
    compare("likelihood_cv", "core", "Algorithm 4 step 2(b)", cvs, expected_cvs)
    stops, expected_stops, final_checks, schedule_checks = [], [], [], []
    for name in ("stable", "oscillating"):
        row = reference["paper_outer_"+name]
        history = row["history"]
        expected_stop = next(l for l in range(3, len(history))
            if statistics.stdev([math.exp(z-max(history[l-2:l+1])) for z in history[l-2:l+1]]) /
               statistics.mean([math.exp(z-max(history[l-2:l+1])) for z in history[l-2:l+1]]) < .1)
        stop = int(row["stop_index_zero_based"][0])
        stops.append(stop); expected_stops.append(expected_stop)
        events = [row["events"][i:i+4] for i in range(0,len(row["events"]),4)]
        final_checks += [int(events[-1][3] == 1 and events[-2][3] == 0),
                         int(events[-1][:3] == events[-2][:3]),
                         int(abs(row["final_value"][0] - history[stop] - .123) < 1e-12)]
        counts = [8]
        for l in range(stop):
            window = history[max(0,l-2):l+1]
            double = l >= 2 and counts[l-2] == counts[l] and any(a>b for a,b in zip(window,window[1:]))
            counts.append(counts[-1]*(2 if double else 1))
        schedule_checks += [int(event[1] == counts[int(event[0])-1]) for event in events]
    compare("outer_stopping_index", "outer", "Algorithm 4: l starts at zero, stop only when l>k", stops, expected_stops,
            "Execute the source while loop; convert its one-based index to the paper's zero-based index.", tolerance=0.)
    compare("fresh_final_run", "outer", "Algorithm 4 step 3", final_checks, [1]*len(final_checks), tolerance=0.)
    compare("particle_schedule", "outer", "Algorithm 4 step 2(d), once a complete window exists", schedule_checks, [1]*len(schedule_checks), tolerance=0.)

    points, target = reference["fit"]["points"], reference["objective"]["targets"]
    losses, paper_losses, relations = [], [], []
    for i in (1, 2, 3):
        values = reference[f"objective{i}"]; par = values["parameters"]
        density = [math.prod(normal(points[2*j+k], par[k], math.exp(2*par[2+k])) for k in range(2)) for j in range(len(target))]
        a = sum(x*x for x in density); b = sum(x*y for x,y in zip(density,target)); c = sum(y*y for y in target)
        paper_loss = sum((x-b/c*y)**2 for x,y in zip(density,target))
        losses.extend(values["loss"]); paper_losses.append(paper_loss)
        relations.append(dict(relative_shape=paper_loss/a,
            public_objective_from_shape=c*(a*c/(b*b)-1), paper_profiled_loss=paper_loss))
    compare("paper_fit_objective", "example_scheme", "equation (15)", losses, paper_losses,
            "Public F=c*R/(1-R) differs from paper L=a-b^2/c. This can be a different Algorithm-3 approximation, not equation-(15) fidelity.")
    floor = reference["paper_floor"]
    expected_floor = [normal(x, model["twists"][0], model["twists"][1])+floor["positive_floor"][0] for x in floor["x"]]
    compare("positive_floor", "example_scheme", "equation (16)", floor["twist"], expected_floor,
            "Positive test floor .01 is explicit, not a tuned default. Public interface executes a pure Gaussian with no floor input; general psi-APF permits pure Gaussians.")
    single = reference["paper_single_time"]
    single_actual = single["executed"] + single.get("log_likelihood", [])
    compare("single_observation_domain", "domain", "Algorithms 2 and 5 at T=1",
            single_actual, [1., math.log(normal(.3,0.,2.))], "R for(t in 2:Time) has no empty-range handling at T=1.")
    report = dict(schema=SCHEMA, execution_status="complete", source_sha256=sha(source),
        source_commit="a88114395f6c11075fedc653db9480791b43391a",
        original_author_provenance="not_verified", paper_sha256=sha(PAPER),
        checks=checks, objective_relations=relations,
        coverage="Gaussian identity-noise model; scalar three-step finite filter and two-dimensional primitives; controlled outer histories",
        unchecked=["original authors' code", "RNG distribution/bitwise parity", "top-level data generation and initial bootstrap experiment", "nonlinear model 1900", "general dimensions and horizons"],
        reference_scope="matched_gaussian_operations_only",
        cpu_only=True, gpu_intentionally_hidden=True)
    report["deviations"] = [row["name"] for row in checks if not row["passed"]]
    report["paper_reference_eligible"] = not report["deviations"]
    return report


def require_reference(report, *, scope="paper"):
    """Reject missing, partial, stale, failed or inappropriate reference evidence."""
    required = REQUIRED if scope == "paper" else CORE if scope == "matched_gaussian" else None
    if required is None:
        raise ValueError("unknown reference scope")
    checks = report.get("checks", [])
    names = [row.get("name") for row in checks]
    if (report.get("schema") != SCHEMA or report.get("execution_status") != "complete"
            or report.get("source_sha256") != SOURCE_SHA256 or len(names) != len(set(names))
            or not required.issubset(names)):
        raise ValueError("incomplete or unrecognized iAPF reference evidence")
    failures = [row["name"] for row in checks if row["name"] in required and row.get("passed") is not True]
    if failures:
        raise ValueError("iAPF reference does not meet "+scope+": "+", ".join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    command = [sys.executable, *sys.argv]
    manifest = dict(plan=PLAN, command=command, cpu_only=True, gpu_intentionally_hidden=True,
        cuda_visible_devices="-1", environment=sys.executable, python=sys.version,
        R=subprocess.check_output(["Rscript", "--version"], text=True, stderr=subprocess.STDOUT).strip(),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        data_version="paper_equation_fixtures_v1", seeds=[91801],
        sources={str(p.relative_to(ROOT)):sha(p) for p in (SOURCE,PAPER,PROBE,Path(__file__))},
        result_file=str(output/"report.json"))
    try:
        report = assess_reference(run_reference(output))
        require_reference(report, scope="matched_gaussian")
        report["matched_gaussian_operations_passed"] = True
        exit_code = 0 if report["paper_reference_eligible"] else 2
    except Exception as exc:
        report = dict(schema=SCHEMA, execution_status="invalid", paper_reference_eligible=False,
                      error=f"{type(exc).__name__}: {exc}")
        exit_code = 1
    manifest["wall_seconds"] = time.monotonic()-started
    for name, data in (("report.json",report),("manifest.json",manifest)):
        (output/name).write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"execution_status":report["execution_status"],
        "paper_reference_eligible":report["paper_reference_eligible"],
        "deviations":report.get("deviations"), "error":report.get("error"),
        "wall_seconds":manifest["wall_seconds"]}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

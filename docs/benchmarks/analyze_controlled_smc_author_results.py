"""Post-run diagnostic analysis of unchanged author-source matrices (C2).

NumPy is used only for independent, CPU-only statistical reporting. This module
is not imported by a filter, training, admission, or candidate runtime path.
The completed source attempt remains read-only. A PASS means analysis completed;
it never admits the TensorFlow port or establishes full paper replication.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # Explicit post-run diagnostic role; never filter execution.

BOOTSTRAPS = 5000
SEED = 202610035
AUTHOR_COMMIT = "2696f6d611aae172d10102cb7bfbfd314bb6b88f"
SOURCE_LOCK_SHA = "f8534fddd2163877a4d6d8c6e2aa542b0c4f8afe9d30f0019bcc276e0917ff45"
FILES = {'lorenz_csmc':'cSMC_VaryF.m','lorenz_apf':'APF_VaryF.m',
         'neuroscience_csmc':'NormConst_cSMC.m','neuroscience_bpf':'NormConst_BPF.m'}
DRIVERS = {
    "lorenz_csmc": ("Lorenz", 40, 100, 512, 4, 202610131),
    "lorenz_apf": ("Lorenz", 40, 100, None, 0, 202610132),
    "neuroscience_csmc": ("NeuroScience", 20, 2999, 128, 3, 202610133),
    "neuroscience_bpf": ("NeuroScience", 20, 2999, 5529, 0, 202610134),
}
LORENZ_SHIFT = -101 * 6 / 2 * math.log(2 * math.pi * 1e-4)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_cases(path):
    rows = []
    with path.open() as stream:
        for values in csv.reader(stream):
            require(len(values) == 11, "partial or malformed source CSV")
            require(values[4] in ("0", "1"), "invalid finite-real flag")
            value, ess = float(values[3]), float(values[8])
            rows.append(dict(grid_index=int(values[0]), repetition=int(values[1]),
                seconds=float(values[2]), logZ=value if math.isfinite(value) else None,
                finite_real=values[4] == "1", N=int(values[5]), I=int(values[6]),
                T=int(values[7]), ESS=ess if math.isfinite(ess) else None,
                refinements=int(values[9]), cpu_seconds=float(values[10])))
    return rows


def validate_driver(key, driver):
    lane, grids, horizon, particles, iterations, seed = DRIVERS[key]
    require(driver["status"] == "COMPLETE" and driver["mode"] == "full", "incomplete source driver")
    require(driver["lane"] == lane and driver["seed"] == seed, "wrong lane or seed")
    require(driver["returncode"] == 0, "failed source process")
    rows = driver["cases"]
    keys = [(r["grid_index"], r["repetition"]) for r in rows]
    expected = {(g, r) for g in range(1, grids + 1) for r in range(1, 101)}
    require(len(keys) == len(set(keys)) and set(keys) == expected, "missing or duplicate source cases")
    require(driver["expected_cases"] == grids * 100 and driver["exact_case_coverage"], "wrong source coverage")
    settings = driver["source_settings"]
    require((settings["nrep"], settings["ngrid"], settings["T"], settings["I"])
            == (100, grids, horizon, iterations), "wrong source settings")
    if particles is not None:
        require(settings["N"] == particles, "wrong source particle count")
    native = driver["native_matrix_validation"]
    require(native == dict(shape=[100, grids], checked_cases=grids * 100,
                           csv_values_match=True), "native matrix not verified")
    for row in rows:
        require(row["T"] == horizon and row["I"] == iterations, "wrong case horizon/refinements")
        expected_n = particles if particles is not None else {1:1382, 2:2867, 3:4505, 4:6144}.get(row["refinements"])
        require(row["N"] == expected_n and expected_n is not None, "wrong case particle count")
        for clock in ("seconds", "cpu_seconds"):
            require(row[clock] is not None and math.isfinite(row[clock]) and row[clock] >= 0, "invalid source timer")
        require(not row["finite_real"] or (row["logZ"] is not None and math.isfinite(row["logZ"])), "invalid finite source logZ")
    require(driver["source_numerical_failure"] == any(not r["finite_real"] for r in rows), "wrong failure count")
    return {(r["grid_index"], r["repetition"]): r for r in rows}


def load_source(source):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    require(manifest["phase"] == "c1_source_full" and manifest["status"] == "COMPLETE"
            and manifest["decision"]["status"] == "PASS", "source attempt is not complete")
    require(manifest["author_commit"] == AUTHOR_COMMIT and manifest["cpu_only"], "wrong source provenance")
    hashes = manifest["output_sha256"]

    def checked(relative):
        path = source / relative
        require(relative in hashes and path.is_file() and sha(path) == hashes[relative], "source hash mismatch: " + relative)
        return path

    summary = json.loads(checked("summary.json").read_text())
    require(summary["status"] == "PASS" and summary["scope"] == "AUTHOR_SOURCE_MATRIX"
            and summary["author_algorithm_modified"] is False, "wrong source scope")
    require(summary["source_lock_sha256"] == SOURCE_LOCK_SHA, "wrong source lock")
    lock_path = Path(summary["source_lock"]) / "source-lock.json"
    require(lock_path.is_file() and sha(lock_path) == SOURCE_LOCK_SHA, "source lock file changed")
    items = summary["drivers"]
    require(len(items) == 4 and {d["driver"] for d in items} == set(DRIVERS), "wrong driver matrix")
    indices = {}
    for driver in items:
        key = driver["driver"]
        indices[key] = validate_driver(key, driver)
        prefix = "reference-work/" + driver["lane"] + "/"
        from scipy.io import loadmat  # Independent post-run native-result verification.
        native = loadmat(checked(prefix + key + '-result.mat'))
        variable = ('cSMC' if key.endswith('csmc') else 'APF' if key.endswith('apf') else 'BPF') + '_LogNormConst'
        values = native[variable]
        require(values.shape == (100, DRIVERS[key][1]), 'wrong native matrix shape')
        require(all(int(native[k].item()) == v for k,v in driver['source_settings'].items()), 'native settings differ')
        for row in driver['cases']:
            value = complex(values[row['repetition']-1,row['grid_index']-1])
            require(row['finite_real'] == (value.imag == 0 and math.isfinite(value.real)), 'native validity differs')
            require(row['logZ'] == (value.real if math.isfinite(value.real) else None), 'native values differ')
        rows = read_cases(checked(prefix + key + "-cases.csv"))
        require(rows == driver["cases"], "source CSV/summary mismatch")
        algorithm = checked(prefix + "cSMC_Resample.m")
        require(sha(algorithm) == driver["source_algorithm_sha256"], "source algorithm changed")
        original = checked(prefix + FILES[key])
        instrumented = checked(prefix + Path(FILES[key]).stem + '_full.m')
        locked_source = lock_path.parent / 'author-source/StateSpaceModels' / driver['lane']
        require(sha(original) == driver['source_driver_sha256'] == sha(locked_source / FILES[key]), 'source driver changed')
        require(sha(instrumented) == driver['instrumented_sha256'], 'instrumented driver changed')
        require(sha(algorithm) == sha(locked_source / 'cSMC_Resample.m'), 'locked algorithm differs')
        data_file = 'SimulatedData.mat' if driver['lane']=='Lorenz' else 'Data_NeuroScience.mat'
        require(sha(checked(prefix + data_file)) == sha(locked_source / data_file), 'source data changed')
    for key, row in indices["lorenz_apf"].items():
        require(row["refinements"] == indices["lorenz_csmc"][key]["refinements"], "APF/cSMC refinement pairing broken")
    return summary, indices, {"manifest.json": sha(manifest_path), "summary.json": hashes["summary.json"]}


def descriptive(values):
    data = np.asarray(values, dtype=np.float64)
    count = len(data)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        mean = float(np.mean(data)) if count else None
        variance = float(np.var(data, ddof=1)) if count > 1 else None
        rv = float(np.divide(variance, np.square(mean))) if mean and variance is not None else None
    finite_or_none = lambda value: value if value is not None and math.isfinite(value) else None
    return dict(n=count, mean_logZ=finite_or_none(mean), sample_variance_logZ=finite_or_none(variance),
                relative_variance=finite_or_none(rv), conditional_on_finite_only=count != 100)


def bootstrap_cell(values, *, paired, cell_seed, draws=BOOTSTRAPS):
    """Return sample RV and paired/independent bootstrap RV ratios, no selection."""
    data = np.asarray(values, dtype=np.float64)
    require(data.ndim == 2 and data.shape[0] == 2 and data.shape[1] >= 2, "invalid bootstrap data")
    require(np.isfinite(data).all(), "nonfinite bootstrap input")
    rng = np.random.default_rng(cell_seed)
    first = rng.integers(data.shape[1], size=(draws, data.shape[1]))
    second = first if paired else rng.integers(data.shape[1], size=first.shape)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        rvs = np.var(data, axis=1, ddof=1) / np.mean(data, axis=1)**2
        samples = np.stack((data[0, first], data[1, second]))
        boot_rvs = np.var(samples, axis=2, ddof=1) / np.mean(samples, axis=2)**2
        ratio = rvs[0] / rvs[1]
        ratios = boot_rvs[0] / boot_rvs[1]
    valid = bool(np.isfinite(rvs).all() and np.all(rvs > 0) and np.isfinite(ratio)
                 and np.isfinite(ratios).all() and np.all(ratios > 0))
    result = dict(valid=valid, invalid_bootstrap_draws=int(np.count_nonzero(~np.isfinite(ratios) | (ratios <= 0))))
    if not valid:
        return result, None
    result.update(relative_variance_ratio=float(ratio),
        pointwise_95_interval=[float(x) for x in np.quantile(ratios, [.025, .975], method="linear")],
        relative_variance_pointwise_95_intervals=[[float(x) for x in np.quantile(row, [.025, .975], method="linear")] for row in boot_rvs])
    return result, np.log(ratios)


def simultaneous(cells, bootstrap_logs):
    if len(cells) != len(bootstrap_logs) or any(x is None for x in bootstrap_logs):
        return dict(verdict="VETO", reason="Source or bootstrap validity failed; no unconditional grid inference")
    points = np.log([c["comparison"]["relative_variance_ratio"] for c in cells])
    deviations = np.max(np.abs(np.stack(bootstrap_logs) - points[:, None]), axis=0)
    width = float(np.quantile(deviations, .95, method="linear"))
    with np.errstate(over="ignore"):
        bands = np.exp(np.stack((points - width, points + width), axis=1))
    if not np.isfinite(bands).all():
        return dict(verdict="VETO", reason="Nonfinite simultaneous interval")
    for cell, band in zip(cells, bands):
        cell["comparison"]["simultaneous_95_interval"] = [float(x) for x in band]
    supported = bool(np.all(bands[:, 1] < 1))
    return dict(verdict="SUPPORTED_LOWER_RV_THROUGHOUT_GRID" if supported else "INCONCLUSIVE",
                simultaneous_log_halfwidth=width,
                criterion="All simultaneous 95% upper limits below one")


def analyze_indices(indices):
    experiments = []
    for experiment_id, (name, other, grids, paired) in enumerate((
        ("lorenz", "apf", 40, True), ("neuroscience", "bpf", 20, False))):
        for view, shift in (("source", 0.0), ("normalized", LORENZ_SHIFT)) if name == "lorenz" else (("source", 0.0),):
            cells, logs = [], []
            for grid in range(1, grids + 1):
                rows = [[indices[name + "_" + method][grid, rep] for rep in range(1, 101)] for method in ("csmc", other)]
                values = [[r["logZ"] + shift for r in arm if r["finite_real"]] for arm in rows]
                failures = [100 - len(v) for v in values]
                stats = [descriptive(v) for v in values]
                cell = dict(grid_index=grid, grid_value=(2.5 + 6 * (grid - 1)/39 if name == "lorenz" else .01 * grid),
                            failures=failures, unconditional_failure_rates=[n/100 for n in failures],
                            csmc=stats[0], comparator=stats[1],
                            refinement_counts=[dict(sorted(Counter(str(r['refinements']) for r in arm).items())) for arm in rows],
                            particle_count_distribution=[dict(sorted(Counter(str(r['N']) for r in arm).items())) for arm in rows],
                            mean_case_cpu_seconds=[sum(r['cpu_seconds'] for r in arm)/100 for arm in rows],
                            mean_case_seconds=[sum(r["seconds"] for r in arm)/100 for arm in rows])
                if any(failures):
                    result, draws = dict(valid=False, reason="Nonfinite or complex source logZ"), None
                else:
                    result, draws = bootstrap_cell(values, paired=paired,
                        cell_seed=np.random.SeedSequence([SEED, experiment_id, grid]))
                cell["comparison"] = result
                cells.append(cell)
                logs.append(draws)
            experiments.append(dict(experiment=name, view=view, primary=view == "source", comparator=other,
                paired_bootstrap=paired, normalization_shift=shift, cells=cells,
                grid_inference=simultaneous(cells, logs)))
    return experiments


def result_note(summary):
    lines = ["# Original-source C2 result", "", "Analysis complete; TensorFlow and full paper replication remain incomplete.", "",
             "| Experiment | Convention | Statistical verdict |", "|---|---|---|"]
    for experiment in summary["experiments"]:
        lines.append(f"| {experiment['experiment']} | {experiment['view']} | {experiment['grid_inference']['verdict']} |")
    lines += ["", "| Decision | Status |", "|---|---|",
        "| Primary completion criterion | All 12,000 source cases validated and accounted for |",
        "| Veto diagnostics | Per-cell source and bootstrap validity in summary.json; VETO blocks grid inference |",
        "| Main uncertainty | 100 replications per cell; exchangeable-replication bootstrap; MATLAB/Octave equivalence unresolved |",
        "| Next justified action | Source-valid P0 consumer parity and frozen P1 no-worse criteria; recover KDM training specification |",
        "| What is not concluded | Full replication, TensorFlow parity, exact published-plot equality, default/HMC readiness |", "",
        "| Inference status | Evidence |", "|---|---|",
        "| Hard veto screen | Nonfinite/complex source outputs or undefined bootstrap statistics veto unconditional grid inference |",
        "| Statistically supported ranking | Only the explicit simultaneous-grid verdicts above |",
        "| Descriptive-only differences | Timing, mean logZ, ESS/refinements and conditional finite-only statistics |",
        "| Default-readiness | Not evaluated |",
        "| Next evidence needed | Matched downstream port comparison and complete KDM author training |", "",
        "The bootstrap uses 5,000 draws and seed 202610035; Lorenz pairs repetition indices because APF cost uses the corresponding cSMC refinement count. Neuroscience resamples independently. The normalized Lorenz view is supplementary. Source logZ finiteness does not certify every internal covariance/domain operation.", "",
        "Post-run red team: solver/runtime differences and unspecified original random seeds can explain differences from the publication. A MATLAB run could overturn an Octave-specific conclusion. The weakest evidence is runtime equivalence and finite-sample bootstrap coverage; no exact figure claim is made.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-attempt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    require(not (args.output / "summary.json").exists(), "refuse to overwrite analysis")
    source, indices, hashes = load_source(args.source_attempt)
    experiments = analyze_indices(indices)
    summary = dict(status="PASS", scope="AUTHOR_SOURCE_STATISTICAL_ANALYSIS", replication="INCOMPLETE",
        scientific_verdict="Source-only comparative evidence; see per-grid inference; port and full replication incomplete",
        cpu_only=True, gpu_intentionally_hidden=True, jit_compile=False,
        backend="NumPy CPU post-run diagnostic reporting only; no filter execution", numpy_version=np.__version__,
        source_attempt=str(args.source_attempt), source_artifacts_sha256=hashes,
        bootstrap=dict(draws=BOOTSTRAPS, seed=SEED, interval=.95, quantile_method="linear",
                       generator="PCG64; SeedSequence([seed, experiment_index, grid_index]); same draws for normalized view"),
        source_runtime_equivalence=source["runtime_equivalence"], experiments=experiments,
        heuristic_dominance_verdict="Original APF/BPF comparisons only, as scoped for author-source replication; no new method promotion",
        wall_seconds=time.monotonic() - started)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (args.output / "result.md").write_text(result_note(summary))
    print(json.dumps({"status": "PASS", "scope": summary["scope"], "replication": "INCOMPLETE",
                      "verdicts": [e["grid_inference"]["verdict"] for e in experiments]}))


if __name__ == "__main__":
    main()

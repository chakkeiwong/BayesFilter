"""Lock the author-source experiment matrix before any replication run.

This is a metadata/provenance phase.  It does not run a translated filter and
does not reinterpret a missing author driver as permission to reconstruct one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUTHOR_ROOT = ROOT / ".localresources/code/heng-controlled-smc"
AUTHOR_COMMIT = "2696f6d611aae172d10102cb7bfbfd314bb6b88f"
PAPER = ROOT / ".localresources/papers/heng-controlled-smc-author-manuscript-and-supplement.pdf"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(AUTHOR_ROOT), "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_entry(relative: str) -> dict:
    path = AUTHOR_ROOT / relative
    entry = {"path": str(path.relative_to(ROOT)), "exists": path.is_file()}
    if path.is_file():
        entry.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    return entry


def check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def main() -> None:
    global ROOT, AUTHOR_ROOT, PAPER
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="repository root when executed from a campaign snapshot")
    args = parser.parse_args()
    ROOT = args.root.resolve()
    AUTHOR_ROOT = ROOT / ".localresources/code/heng-controlled-smc"
    PAPER = ROOT / ".localresources/papers/heng-controlled-smc-author-manuscript-and-supplement.pdf"
    args.output.mkdir(parents=True, exist_ok=True)

    required = {
        "lorenz": [
            "StateSpaceModels/Lorenz/cSMC_VaryF.m",
            "StateSpaceModels/Lorenz/APF_VaryF.m",
            "StateSpaceModels/Lorenz/cSMC_Resample.m",
            "StateSpaceModels/Lorenz/RungeKutta4.m",
            "StateSpaceModels/Lorenz/Systematic_Resampling.m",
            "StateSpaceModels/Lorenz/SimulatedData.mat",
        ],
        "neuroscience": [
            "StateSpaceModels/NeuroScience/NormConst_cSMC.m",
            "StateSpaceModels/NeuroScience/NormConst_BPF.m",
            "StateSpaceModels/NeuroScience/cSMC_Resample.m",
            "StateSpaceModels/NeuroScience/ConstructTransition.m",
            "StateSpaceModels/NeuroScience/Model_NeuroScience.m",
            "StateSpaceModels/NeuroScience/Systematic_Resampling.m",
            "StateSpaceModels/NeuroScience/Data_NeuroScience.mat",
        ],
        "multimodal": [
            "StateSpaceModels/ToyMultimodal/Demo.m",
            "StateSpaceModels/ToyMultimodal/Model_ToyMultimodal.m",
            "StateSpaceModels/ToyMultimodal/cSMC_Resample.m",
            "StateSpaceModels/ToyMultimodal/Systematic_Resampling.m",
            "StateSpaceModels/ToyMultimodal/SimulatedData_SigmaSq_point1.mat",
            "StateSpaceModels/ToyMultimodal/SimulatedData_SigmaSq_point5.mat",
            "StateSpaceModels/ToyMultimodal/SimulatedData_SigmaSq_1.mat",
        ],
    }
    files = {lane: [source_entry(item) for item in paths] for lane, paths in required.items()}
    all_required = [item for rows in files.values() for item in rows]
    required_present = all(item["exists"] for item in all_required)

    repeat_driver = AUTHOR_ROOT / "StateSpaceModels/ToyMultimodal/Repeat.m"
    lorenz_pilot = ROOT / "docs/plans/controlled-smc-lorenz-author-native-result-20260930.md"
    paper_present = PAPER.is_file()
    octave = shutil.which("octave")
    octave_version = None
    if octave:
        try:
            first_line = subprocess.check_output(
                [octave, "--version"], cwd=ROOT, text=True, stderr=subprocess.STDOUT
            ).splitlines()
            octave_version = first_line[0] if first_line else None
        except (OSError, subprocess.CalledProcessError):
            octave_version = None

    matrices = [
        {
            "name": "lorenz_figure6",
            "status": "READY_FOR_SOURCE_RUN",
            "source_driver": "StateSpaceModels/Lorenz/cSMC_VaryF.m",
            "comparator_driver": "StateSpaceModels/Lorenz/APF_VaryF.m",
            "settings": {
                "dimension": 8,
                "horizon_T": 100,
                "observations": 101,
                "forcing_grid": "linspace(2.5,8.5,40)",
                "repetitions": 100,
                "csmc_particles": 512,
                "maximum_refinements": 4,
                "observation_variance": 1e-4,
                "apf_particle_counts_by_refinement": [1382, 2867, 4505, 6144],
            },
            "data": "StateSpaceModels/Lorenz/SimulatedData.mat",
            "existing_pilot": str(lorenz_pilot.relative_to(ROOT)) if lorenz_pilot.is_file() else None,
            "source_result_claim": "native source pilot only until the full 40x100 matrix is run",
        },
        {
            "name": "neuroscience_figure4_right",
            "status": "READY_FOR_SOURCE_RUN",
            "source_driver": "StateSpaceModels/NeuroScience/NormConst_cSMC.m",
            "comparator_driver": "StateSpaceModels/NeuroScience/NormConst_BPF.m",
            "settings": {
                "dimension": 1,
                "horizon_from_data": True,
                "csmc_particles": 128,
                "csmc_iterations": 3,
                "parameter_grid": "linspace(0.01,0.2,20)",
                "repetitions": 100,
                "bpf_particles": 5529,
                "bpf_iterations": 0,
                "alpha": 0.99,
            },
            "data": "StateSpaceModels/NeuroScience/Data_NeuroScience.mat",
            "source_result_claim": "source matrix is locked; no result has been run by this phase",
        },
        {
            "name": "multimodal_table2",
            "status": "BLOCKED_MISSING_AUTHOR_DRIVER",
            "source_driver": "StateSpaceModels/ToyMultimodal/Repeat.m",
            "available_algorithm": "StateSpaceModels/ToyMultimodal/cSMC_Resample.m",
            "data": [
                "StateSpaceModels/ToyMultimodal/SimulatedData_SigmaSq_point1.mat",
                "StateSpaceModels/ToyMultimodal/SimulatedData_SigmaSq_point5.mat",
                "StateSpaceModels/ToyMultimodal/SimulatedData_SigmaSq_1.mat",
            ],
            "settings": {
                "particles": 512,
                "iterations": 1,
                "knots": 16,
                "repetitions": 100,
                "observation_variances": [0.1, 0.5, 1.0],
                "bpf_particles": [2252, 4710, 6553],
                "supplied_observation_rows": 101,
                "textual_horizon_discrepancy": "Appendix F.2 says T+1=100; Demo/Model/data use T=100 and 101 rows",
            },
            "source_result_claim": "no exact source replication until Repeat.m or an author-equivalent driver is recovered",
        },
    ]

    observed_commit = git_commit()
    # HEAD alone does not prove that the executed source is unmodified.
    tree = subprocess.check_output(
        ["git", "-C", str(AUTHOR_ROOT), "ls-tree", "-r", "-z", AUTHOR_COMMIT],
        text=True,
    )
    pinned_blobs = {}
    for record in tree.split("\0"):
        if record:
            metadata, filename = record.split("\t", 1)
            pinned_blobs[filename] = metadata.split()[2]
    # Include the complete local MATLAB dependency set for the selected lanes,
    # including analysis/driver files. Keep path names to avoid collisions.
    snapshot = args.output / "author-source"
    verified_files = []
    for lane in ("Lorenz", "NeuroScience", "ToyMultimodal"):
        folder = AUTHOR_ROOT / "StateSpaceModels" / lane
        for path in sorted(folder.glob("*.m")):
            if not any(item["path"] == str(path.relative_to(ROOT)) for item in all_required):
                all_required.append(source_entry(str(path.relative_to(AUTHOR_ROOT))))
    for entry in all_required:
        if not entry["exists"]:
            continue
        path = ROOT / entry["path"]
        relative = str(path.relative_to(AUTHOR_ROOT))
        data = path.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        entry["matches_pinned_blob"] = blob == pinned_blobs.get(relative)
        dest = snapshot / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        verified_files.append(entry)
    matrices.insert(1, {
        "name": "lorenz_table1",
        "status": "READY_FOR_GENERATIVE_DESIGN_COST_PILOT",
        "source_drivers": ["StateSpaceModels/Lorenz/Repeat_Dim%d.m" % d for d in (8,16,32,64)],
        "settings": {"dimensions": [8,16,32,64], "observation_variances": [0.0001,0.001,0.01],
                     "csmc_particles": [512,512,1024,4096], "apf_particles": [1382,2027,4034,11468],
                     "refinements": 1, "repetitions": 100},
        "data_provenance": "Author driver generates data; historical unseeded data are unavailable",
        "source_result_claim": "Generative-design replication is possible; exact published dataset/point-value reproduction is not established",
    })
    for matrix in matrices:
        if matrix["status"] == "READY_FOR_SOURCE_RUN":
            matrix["status"] = "READY_FOR_SOURCE_COST_PILOT"
    runtime_root = ROOT / ".localresources/code/octave-statistics-1.5.0/inst"
    runtime_files = [runtime_root / "distributions/mvnrnd.m", runtime_root / "x2fx.m", runtime_root / "squareform.m"]
    runtime_dependencies = [{"path": str(p.relative_to(ROOT)), "exists": p.is_file(),
                             "sha256": sha256(p) if p.is_file() else None} for p in runtime_files]
    checks = [
        check("author_repository_present", AUTHOR_ROOT.is_dir(), str(AUTHOR_ROOT)),
        check("author_commit_matches_pinned", observed_commit == AUTHOR_COMMIT, f"observed={observed_commit} expected={AUTHOR_COMMIT}"),
        check("required_source_and_data_files_present", required_present, "all selected matrix inputs"),
        check("source_files_match_pinned_commit", all(e["matches_pinned_blob"] for e in verified_files), "working files checked against pinned git blobs"),
        check("octave_statistics_dependencies_present", all(e["exists"] for e in runtime_dependencies), "mvnrnd, x2fx, squareform"),
        check("paper_copy_present", paper_present, str(PAPER.relative_to(ROOT))),
        check("octave_runtime_available", octave is not None and octave_version is not None, octave_version or "octave not available"),
        check("lorenz_native_pilot_recorded", lorenz_pilot.is_file(), str(lorenz_pilot.relative_to(ROOT))),
        check("multimodal_missing_driver_is_explicit", not repeat_driver.is_file(), str(repeat_driver.relative_to(ROOT))),
    ]
    # A source lock may pass with a blocked matrix.  It fails only if the
    # provenance needed to explain that block is itself missing.
    status = "PASS" if all(row["passed"] for row in checks) else "FAIL"
    summary = {
        "schema": "controlled_smc_kdm_c0_source_lock_v1",
        "status": status,
        "scope": "SOURCE_MATRIX_PROVENANCE_AND_EXECUTABILITY_LOCK",
        "source_repository": str(AUTHOR_ROOT.relative_to(ROOT)),
        "author_commit": AUTHOR_COMMIT,
        "observed_author_commit": observed_commit,
        "paper_copy": str(PAPER.relative_to(ROOT)),
        "paper_sha256": sha256(PAPER) if paper_present else None,
        "octave": {"path": octave, "version": octave_version},
        "required_files": files,
        "verified_snapshot_files": verified_files,
        "runtime_dependencies": runtime_dependencies,
        "cpu_only": True,
        "gpu_intentionally_hidden": True,
        "matrices": matrices,
        "checks": checks,
        "interpretation": {
            "source_baseline_first": True,
            "tensorflow_no_worse_is_separate_lane": True,
            "new_numerical_safeguards_are_E0": True,
            "blocked_multimodal_is_not_silently_reconstructed": True,
        },
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "source-lock.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"status": status, "matrices": {row["name"]: row["status"] for row in matrices}}))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

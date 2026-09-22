"""CPU-only executable paper conformance for the independent R reference."""
import csv
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_gaussian_limit_tail_check_distinguishes_finite_and_infinite_moments(tmp_path):
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable; no independent tail evidence")
    # With Q=R=C=1 and T=1, the second-moment precision is 3-1/V.
    # V=.5 is integrable with relative margin 1/5; V=.2 diverges with -2/8.
    program = tmp_path / "tail-check.R"
    program.write_text(
        'args <- commandArgs(trailingOnly=TRUE)\n'
        'source(file.path(args[1],"docs/benchmarks/reference_iapf_paper.R"))\n'
        'source(file.path(args[1],"docs/benchmarks/diagnose_iapf_r_validation_tails.R"))\n'
        'model <- iapf_linear_model(matrix(0,1,1),matrix(0,1,1))\n'
        'good <- iapf_validation_tail_rows(model,list(iapf_gaussian_twist(0,matrix(.5,1,1))),1)\n'
        'bad <- iapf_validation_tail_rows(model,list(iapf_gaussian_twist(0,matrix(.2,1,1))),2)\n'
        'stopifnot(good$passed,!bad$passed,abs(good$relative_margin-.2)<1e-12,\n'
        'abs(bad$relative_margin+.25)<1e-12)\n'
    )
    result = subprocess.run(["Rscript", "--vanilla", str(program), str(ROOT)],
        cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1",
                          OMP_NUM_THREADS="1"), text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_worker_timeout_stops_nested_phase(tmp_path):
    path = ROOT / "docs/benchmarks/run_iapf_r_replication.py"
    spec = importlib.util.spec_from_file_location("r_reference_driver_timeout", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    pid_file = tmp_path / "child.pid"
    program = (
        "import subprocess,sys,time,pathlib; "
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    with (tmp_path / "worker.log").open("w") as log:
        with pytest.raises(subprocess.TimeoutExpired):
            driver.run_bounded_worker([sys.executable, "-c", program, str(pid_file)],
                cwd=tmp_path, env=dict(os.environ), log=log, timeout=1)
    pid = int(pid_file.read_text())
    status = Path(f"/proc/{pid}/stat")
    # A terminated child can await collection by the system's init process.
    if status.exists():
        assert status.read_text().split()[2] == "Z"


def test_floor_repair_healthy_parity_and_phase_exit_status(tmp_path):
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable; no phase execution evidence")
    output = tmp_path / "phase-mechanics"
    result = subprocess.run(
        ["Rscript", "--vanilla", str(ROOT / "docs/benchmarks/diagnose_iapf_r_positive_floor.R"),
         str(ROOT), str(output), "80", "1", "1", "72000080", "floor_repair",
         "log_quadratic", "8", "first_full_window", "200", "mechanics_only"],
        cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1",
                          OMP_NUM_THREADS="1"), capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    with (output / "checks.csv").open() as stream:
        checks = list(csv.DictReader(stream))
    assert len(checks) == 7 and all(row["passed"] == "TRUE" for row in checks)
    for status in (0, 2):
        log = (output / f"status-{status}.log").read_text()
        assert "stdout sentinel" in log and "stderr sentinel" in log


def test_snapshot_preserves_source_and_test_with_same_basename(tmp_path):
    path = ROOT / "docs/benchmarks/run_iapf_r_replication.py"
    spec = importlib.util.spec_from_file_location("r_reference_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    names = ["docs/benchmarks/reference_iapf_paper.R", "tests/reference_iapf_paper.R"]
    hashes = driver.snapshot_sources(ROOT, names, tmp_path)
    assert len(set(hashes.values())) == 2
    for name in names:
        assert (tmp_path / name).read_bytes() == (ROOT / name).read_bytes()


def test_worker_executes_captured_dependencies_after_live_edit(tmp_path):
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable; no source execution evidence")
    path = ROOT / "docs/benchmarks/run_iapf_r_replication.py"
    spec = importlib.util.spec_from_file_location("r_reference_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    live = tmp_path / "live"
    live.mkdir()
    (live / "core.R").write_text("value <- 17\n")
    (live / "runner.R").write_text(
        'args <- commandArgs(TRUE)\nsource(file.path(args[1],"core.R"))\n'
        'writeLines(as.character(value),args[2])\n')
    snapshots = tmp_path / "sources"
    driver.snapshot_sources(live, ["core.R", "runner.R"], snapshots)
    (live / "core.R").write_text("value <- 999\n")
    (live / "runner.R").write_text('stop("live runner executed")\n')
    output = tmp_path / "value.txt"
    command = driver.r_worker_command(snapshots, "runner.R", output)
    subprocess.run(command, check=True, capture_output=True, timeout=15,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"))
    assert output.read_text().strip() == "17"


def test_independent_r_reference_matches_paper_identities(tmp_path):
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable; no R reference admission")
    result = subprocess.run(
        ["Rscript", "--vanilla", str(ROOT / "tests/reference_iapf_paper.R"),
         str(ROOT), str(tmp_path / "checks.csv")], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1",
                 OMP_NUM_THREADS="1"), text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    with (tmp_path / "checks.csv").open() as stream:
        checks = list(csv.DictReader(stream))
    assert len(checks) >= 50
    assert all(row["passed"] == "TRUE" for row in checks)
    assert len({row["name"] for row in checks}) == len(checks)


def test_explicit_relative_fit_math_and_consumer_wiring(tmp_path):
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable; no R reference admission")
    result = subprocess.run(
        ["Rscript", "--vanilla", str(ROOT / "tests/reference_iapf_relative_fit.R"),
         str(ROOT), str(tmp_path / "relative-checks.csv")], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1",
                 OMP_NUM_THREADS="1"), text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    with (tmp_path / "relative-checks.csv").open() as stream:
        checks = list(csv.DictReader(stream))
    assert len(checks) >= 16
    assert all(row["passed"] == "TRUE" for row in checks)


@pytest.mark.parametrize("old,new", [
    ("floor_probability=exp(twist$log_floor-log_integral)",
     "floor_probability=rep(0,nrow(means))"),
    ("if (iteration>k &&", "if (iteration>=k &&"),
    ("loss <- mean(residual^2)", "loss <- mean(residual^2)/lambda^2"),
    ("status=\"complete\",final=final,twists=twists",
     "status=\"complete\",final=result,twists=twists"),
])
def test_r_reference_checks_detect_wrong_executed_object(tmp_path, old, new):
    if not shutil.which("Rscript"):
        pytest.skip("Rscript unavailable; no mutation evidence")
    source = (ROOT / "docs/benchmarks/reference_iapf_paper.R").read_text()
    assert source.count(old) == 1
    changed = tmp_path / "changed-reference.R"
    changed.write_text(source.replace(old, new))
    result = subprocess.run(
        ["Rscript", "--vanilla", str(ROOT / "tests/reference_iapf_paper.R"),
         str(ROOT), str(tmp_path / "checks.csv"), str(changed)], cwd=ROOT,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1",
                 OMP_NUM_THREADS="1"), text=True, capture_output=True, timeout=120)
    assert result.returncode != 0
    assert "FAILED" in result.stderr, result.stdout + result.stderr

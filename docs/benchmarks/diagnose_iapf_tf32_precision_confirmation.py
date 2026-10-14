"""Diagnostic phase21 supervisor; numerical work uses the frozen shared worker.

The detached supervisor survives loss of its editor/terminal. Ordinary PID and
file-lock tracking prevent an accidental duplicate launch; they are not approval
or scientific-evidence mechanisms. CPU wall accounting includes reporting.
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = "docs/plans/artifacts/iapf-tf32-precision-confirmation-20260923-01"
PLAN = "docs/plans/iapf-tf32-precision-confirmation-2026-09-23.md"
SCRIPT = "docs/benchmarks/diagnose_iapf_tf32_precision_confirmation.py"
BASE = "docs/benchmarks/diagnose_iapf_tf32_revalidation.py"
REPORT = "docs/benchmarks/summarize_iapf_tf32_revalidation.R"
REVIEW = "docs/benchmarks/review_iapf_tf32_recorded_paths.py"
MASTER = "docs/plans/younis-kdm-score-master-program-2026-09-14.md"
CHECKPOINTS = (".localresources/iapf-execution-checkpoint.md",
    "docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/checkpoint.md")
PYTHON = "/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
GPU = "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"
BEGIN, END = "<!-- IAPF_PHASE21_STATUS -->", "<!-- END_IAPF_PHASE21_STATUS -->"


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def publish_status(repo, state, detail):
    root = repo / ROOT
    budget = json.loads((root / "budget.json").read_text()) if (root / "budget.json").exists() else None
    stamp = dt.datetime.now(dt.timezone.utc).isoformat()
    dump(root / "status.json", dict(phase=21, state=state, detail=detail,
         supervisor_pid=os.getpid(), updated_utc=stamp, budget=budget))
    status = (f"Phase21 {state.upper()}: {detail}\n"
              f"Updated {stamp}. [Plan](iapf-tf32-precision-confirmation-2026-09-23.md); "
              f"[live status](artifacts/iapf-tf32-precision-confirmation-20260923-01/status.json).\n"
              "Phase20 is complete: 18/24 precision comparisons pass; six tail vetoes remain.\n"
              "Phase21 is one fixed 4096-stream confirmation, with unchanged criteria.\n"
              "Runtime/defaults and all earlier vetoes remain unchanged.\n")
    master = repo / MASTER
    text = master.read_text()
    if text.count(BEGIN) != 1 or text.count(END) != 1:
        raise RuntimeError("Missing unique phase21 master status block; preserve other edits")
    before, rest = text.split(BEGIN)
    _, after = rest.split(END)
    master.write_text(before + BEGIN + "\n" + status + END + after)
    balance = "See rolling budget.json."
    if budget:
        balance = (f"Remaining campaign CPU {budget['remaining_campaign_cpu_seconds']:.6f}s; "
                   f"GPU {budget['remaining_campaign_gpu_seconds']:.6f}s. "
                   f"Current GPU reservation {budget['gpu_reserved_seconds']:.0f}s.\n")
    checkpoint = (f"# Active iAPF checkpoint\n\nPhase21 {state.upper()}: {detail}\n"
        f"Checkout {repo}\nBranch ledh-refactor-with-policy-fix; numerical base89065bc6.\n"
        f"Master {MASTER}\nPlan {PLAN}\nArtifacts {ROOT}/\n"
        f"Supervisor PID {os.getpid()}; updated {stamp}.\n"
        "Read status.json, completed.json, budget.json and current attempt command.log before resuming.\n"
        "Do not launch a second supervisor while its process or a worker remains alive.\n"
        "Three sequential GPU workers, then R summary and CPU terminal review execute automatically.\n"
        "Scope: d2/N64/T5, 3 fixed datasets, 4 frozen guides, 4096 fresh streams per cell/mode;\n"
        "base seeds+20000000+replicate[0,4095], 147456 evaluations with every path recorded.\n"
        "GPU RTX4080 GPU-68251639-fe82-8f81-3ccc-2953c32e805b; growth/XLA, escalated.\n"
        "Phase cap7200 GPU/240 CPU seconds,6 attempts,2100s per worker.\n"
        + balance + "\nExisting campaign authorization persists without a wall-clock expiry.\n"
        "Phase20 complete: all36864 traces valid/neutral;18/24 passes,6 tail vetoes,0 interval violations.\n"
        "Phase18/19 broader vetoes persist; small CDF differences change ancestors and large likelihoods.\n"
        "No paper/default/HMC or marginal-score promotion. Stress guide is an adverse diagnostic.\n"
        "On completion inspect result.md and decision.json; no automatic further sample enlargement.\n")
    for name in CHECKPOINTS:
        (repo / name).write_text(checkpoint)


def supervise(repo):
    started = time.monotonic()
    root = repo / ROOT
    prior = json.loads((repo / "docs/plans/artifacts/iapf-tf32-revalidation-20260923-01/completed.json").read_text())
    opening = json.loads((repo / "docs/plans/artifacts/iapf-tf32-recorded-paths-20260923-01/budget.json").read_text())
    ledger = root / "budget.json"
    budget = json.loads(ledger.read_text()) if ledger.exists() else dict(
        cpu_charged_seconds=0., gpu_charged_seconds=0., cpu_reserved_seconds=0., gpu_reserved_seconds=0., attempts=0,
        phase_cpu_cap_seconds=240., phase_gpu_cap_seconds=7200.,
        **{k: v for k, v in opening.items() if k.startswith("remaining_campaign_")})
    assert budget["cpu_reserved_seconds"] == budget["gpu_reserved_seconds"] == 0, "Reconcile interrupted attempt before resuming"
    assert "post_run_review_seconds" not in budget, "Terminal review already attempted; inspect its result"
    preflight = json.loads((root / "preflight.json").read_text())
    assert preflight["status"] == "pass" and preflight["report_source_sha256"] == sha(repo / REVIEW)
    assert preflight["shared_worker_bitwise_identical_to_phase20"]
    assert all(preflight["phase20_reports_bitwise_identical"].values())
    assert sha(repo / BASE) == sha(repo / "docs/plans/artifacts/iapf-tf32-recorded-paths-20260923-01/source-01" / BASE)
    assert shutil.disk_usage(root).free > 8 * 1024**3, "Less than8GiB free for diagnostic traces"
    fixture = Path(prior["fixtures"])
    for name, digest in prior["fixture_hashes"].items():
        assert sha(fixture / name) == digest
    source = root / f"source-{len(list(root.glob('source-*'))) + 1:02d}"
    source.mkdir()
    files = [SCRIPT, BASE, REPORT, REVIEW, PLAN, "bayesfilter/__init__.py",
             "bayesfilter/runtime/__init__.py", "bayesfilter/runtime/gpu_memory_policy.py"]
    files += [str(x.relative_to(repo)) for x in (repo / "bayesfilter/score_study").glob("*.py")]
    old = json.loads((Path(prior["fp64"]).parent / "manifest.json").read_text())
    for name in files:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, path)
        if name.startswith("bayesfilter/"):
            assert sha(path) == old["source_sha256"][name]
    hashes = {name: sha(source / name) for name in files}
    git = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    env = dict(os.environ, PYTHONPATH=str(source), TF_FORCE_GPU_ALLOW_GROWTH="true", BAYESFILTER_PRELOAD_CUSTOM_OP="0",
        OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2")
    if env.get("XLA_FLAGS", ""):
        raise RuntimeError("Unexpected compiler override")
    completed_path = root / "completed.json"
    completed = json.loads(completed_path.read_text()) if completed_path.exists() else dict(fixtures=str(fixture))
    gpu_open, cpu_open = budget["gpu_charged_seconds"], budget["cpu_charged_seconds"]
    dump(ledger, budget)

    def run(label, command, gpu, cap):
        kind = "gpu" if gpu else "cpu"
        assert budget["attempts"] < 6
        assert budget[f"{kind}_charged_seconds"] + cap <= budget[f"phase_{kind}_cap_seconds"]
        assert cap <= budget[f"remaining_campaign_{kind}_seconds"]
        budget["attempts"] += 1
        budget[f"{kind}_reserved_seconds"] = cap
        dump(ledger, budget)
        job = root / f"attempt{budget['attempts']:02d}-{label}"
        job.mkdir()
        command = [str(x).replace("{OUT}", str(job / "outputs")) for x in command]
        runenv = dict(env, CUDA_VISIBLE_DEVICES=GPU if gpu else "-1")
        manifest = dict(status="running", command=command, git_commit=git, source=str(source), source_sha256=hashes,
            plan=PLAN, result_file=str(root / "result.md"), fixture=str(fixture), fixture_sha256=prior["fixture_hashes"],
            data_version="phase18 fixed d2 datasets, fresh4096 particle streams",
            seeds="cells.csv particle_seed+20000000+replicate[0,4095]", cpu_only=not gpu, gpu_intentionally_hidden=not gpu,
            environment={k: runenv.get(k, "") for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "XLA_FLAGS")},
            started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), cap_seconds=cap)
        dump(job / "manifest.json", manifest)
        publish_status(repo, "running", f"Executing {label}; terminal reporting follows automatically.")
        t = time.monotonic()
        try:
            with (job / "command.log").open("w") as log:
                process = subprocess.Popen(command, cwd=source, env=runenv, stdout=log, stderr=subprocess.STDOUT)
                manifest["pid"] = process.pid
                dump(job / "manifest.json", manifest)
                try:
                    code = process.wait(timeout=cap)
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
            manifest.update(status="complete" if code == 0 else "failed", returncode=code)
        except BaseException as error:
            manifest.update(status="failed", error=repr(error))
        finally:
            seconds = time.monotonic() - t
            manifest["wall_seconds"] = seconds
            budget[f"{kind}_reserved_seconds"] = 0.
            budget[f"{kind}_charged_seconds"] += seconds
            budget[f"remaining_campaign_{kind}_seconds"] -= seconds
            manifest["output_sha256"] = {str(x.relative_to(job)): sha(x) for x in job.rglob("*") if x.is_file() and x.name != "manifest.json"}
            dump(job / "manifest.json", manifest)
            dump(ledger, budget)
        print(json.dumps(dict(label=label, status=manifest["status"], seconds=seconds)), flush=True)
        if manifest["status"] != "complete":
            raise RuntimeError(f"Execution repair required at {job}; this is not a scientific candidate verdict")
        return str(job / "outputs")

    try:
        for mode in ("fp64", "fp32", "tf32"):
            if mode not in completed:
                completed[mode] = run(mode, [PYTHON, source / SCRIPT, "--mode", mode, "--fixture", fixture, "--output", "{OUT}"], True, 2100)
                dump(completed_path, completed)
        if "summary" not in completed:
            completed["summary"] = run("summary", ["/usr/bin/Rscript", "--vanilla", source / REPORT, fixture,
                completed["fp64"], completed["fp32"], completed["tf32"], "{OUT}", "12", "4096"], False, 60)
            dump(completed_path, completed)
    finally:
        overhead = max(0., time.monotonic() - started - (budget["gpu_charged_seconds"] - gpu_open) - (budget["cpu_charged_seconds"] - cpu_open))
        budget["cpu_charged_seconds"] += overhead
        budget["remaining_campaign_cpu_seconds"] -= overhead
        dump(ledger, budget)
    return source


def finish(repo, source):
    root = repo / ROOT
    command = [PYTHON, str(source / REVIEW), "--repository", str(repo), "--root", ROOT, "--plan", PLAN,
               "--replicates", "4096", "--particle-seed-offset", "20000000"]
    publish_status(repo, "reviewing", "All three modes and R summary completed; CPU integrity and trace review running.")
    review_started = time.monotonic()
    try:
        with (root / "review-command.log").open("w") as log:
            subprocess.run(command, cwd=repo, env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"),
                           stdout=log, stderr=subprocess.STDOUT, timeout=150, check=True)
    finally:
        measured_review = time.monotonic() - review_started
        review_budget = json.loads((root / "budget.json").read_text())
        charged_review = review_budget.get("post_run_review_seconds", 0.)
        extra = max(0., measured_review - charged_review)
        if "post_run_review_seconds" not in review_budget:
            # A killed reviewer cannot settle its own finally block.
            extra += json.loads((root / "preflight.json").read_text())["wall_seconds"] + 2.
            review_budget["interrupted_review_seconds"] = measured_review
        review_budget["cpu_charged_seconds"] += extra
        review_budget["remaining_campaign_cpu_seconds"] -= extra
        review_budget["review_process_overhead_seconds"] = extra
        dump(root / "budget.json", review_budget)
    started = time.monotonic()
    decision = json.loads((root / "review-01/decision.json").read_text())
    verification = json.loads((root / "review-01/verification.json").read_text())
    budget = json.loads((root / "budget.json").read_text())
    assert verification["status"] == "pass" and verification["records"] == verification["neutral_valid_traces"] == 147456
    detail = (f"{decision['passed_contrasts']}/{decision['total_contrasts']} precision comparisons pass; "
              f"{decision['tail_veto_contrasts']} tail vetoes and {decision['interval_violations']} interval violations. "
              "All147456 traces verified. See result.md; no further automatic sample enlargement.")
    result = f"""# Phase21 fixed precision confirmation

{detail}

This is a conditional comparison on three fixed d2/N64/T5 datasets, four
frozen guides and4096 fresh particle streams per cell and precision. Every
actual FP64, ordinary-FP32 and TF32 path was recorded. The shared numerical
runtime was frozen; all source, data, output, tracing-neutrality and seed checks
passed. Streams are disjoint from phases18 and20. Earlier results remain separate.

The primary criterion is all24 simultaneous approximate99% paired Student
intervals strictly within[-.02,.02], with maximum centered squared contribution
at most.2. These are engineering criteria for mean normalized likelihood;
they do not establish distributional equivalence. Outcome-selected path
localizations explain mechanisms and cannot estimate population frequencies.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Fixed confirmation complete | {decision['passed_contrasts']}/24 pass | {decision['tail_veto_contrasts']} tail; {decision['interval_violations']} interval violations | Rare events and fixed datasets/compilations | Interpret conditional precision and retained failures; no repeated enlargement | Paper replication or default promotion |

| Inference status | Finding |
|---|---|
| Hard veto screen | {decision['tail_veto_contrasts']} current tail vetoes; prior strict TF32 and broader-scope vetoes retained |
| Statistically supported ranking | None; interval equivalence is a different question |
| Descriptive-only differences | Conditional guide MSE and selected extreme paths |
| Default-readiness | Not established; no runtime or default change |
| Next evidence needed | Resolve the retained scope-specific failures with a distinct discriminating question |

The deliberately random stress guide is a diagnostic adversary. Its heuristic
table reports {decision['heuristic_dominance']['observed_losses']} observed losses
among {decision['heuristic_dominance']['comparisons']} conditional comparisons.
These are quality-promotion vetoes where present, not a ranking of trained iAPF.
Engineering integrity passed; numerical precision is reported above; scientific
paper, model-score, KDM, LEDH and HMC claims remain unestablished.

Post-run review: the strongest alternative explanation of small mean differences
is insufficient observation of rare weight events. New datasets/compilations
could overturn conditional agreement. The weakest evidence is tail coverage;
do not pool prior studies or keep increasing replication until a pass.

Actual commands, Git, source hashes, environment, device/growth/XLA settings,
seeds and wall times are in attempt*/manifest.json and worker metadata.
review-01 contains decision.json, verification.json, tail-influence.json,
path-localization.json and its CPU review manifest. budget.json is authoritative.
Charged GPU time: {budget['gpu_charged_seconds']:.3f}s; CPU before final report:
{budget['cpu_charged_seconds']:.3f}s. Plan: {PLAN}.
"""
    (root / "result.md").write_text(result)
    # The reviewer settles its own measured CPU time; reload before this charge.
    overhead = time.monotonic() - started
    budget["cpu_charged_seconds"] += overhead
    budget["remaining_campaign_cpu_seconds"] -= overhead
    budget["final_reporting_seconds"] = overhead
    dump(root / "budget.json", budget)
    assert budget["cpu_charged_seconds"] <= budget["phase_cpu_cap_seconds"]
    dump(root / "terminal.json", dict(status="complete", result="result.md", decision="review-01/decision.json",
         result_sha256=sha(root / "result.md"), completed_utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    publish_status(repo, "complete", detail)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--mode", choices=("fp64", "fp32", "tf32"))
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.detach:
        root = repo / ROOT
        with (root / "supervisor.log").open("a") as log:
            process = subprocess.Popen([PYTHON, str(repo / SCRIPT), "--execute"], cwd=repo,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="-1"), stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(supervisor_pid=process.pid, log=str(root / "supervisor.log"))), flush=True)
    elif args.execute:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        with (repo / ROOT / "supervisor.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                source = supervise(repo)
                finish(repo, source)
            except BaseException as error:
                publish_status(repo, "repair_required", repr(error))
                raise
    else:
        if not args.mode or not args.fixture or not args.output:
            parser.error("worker requires mode, fixture and output")
        spec = importlib.util.spec_from_file_location("phase18_diagnostic", Path(__file__).with_name(Path(BASE).name))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.worker(args.mode, args.fixture, args.output, replicates=4096, dimensions=(2,),
                      particle_seed_offset=20000000, trace_all=True, compress_traces=True)


if __name__ == "__main__":
    main()

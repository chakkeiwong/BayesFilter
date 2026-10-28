"""Bounded, diagnostic-only orchestration of the authorized M13 particle ladder.

Invoked through run_range_bearing_covariance_comparison.py's approved run/worker
commands. Numerical authority remains in the shared TensorFlow implementation.
"""
import contextlib
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/plans/artifacts/ledh-range-bearing-covariance-20261009-07"
PLAN = ROOT / "docs/plans/ledh-range-bearing-particle-ladder-2026-10-10.md"
RESULT = ROOT / "docs/plans/ledh-range-bearing-particle-ladder-results-2026-10-10.md"
CHECKPOINT = ROOT / "docs/plans/ledh-main-merge-active-checkpoint.md"


def driver():
    import run_range_bearing_covariance_comparison as module
    return module


def repair_checks(args, tf, dtype):
    """GPU/XLA parity, finite differences, and unchanged N1008 replay."""
    m = driver()
    from bayesfilter.highdim.covariance_proposal_tf import (
        ProposalControls, _dense_reset_transport, make_filter, reset_cloud)
    from bayesfilter.highdim.covariance_proposal_transport_tf import tiled_sinkhorn_barycentres
    from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
    if dtype != tf.float64:
        raise ValueError("Repair checks require the explicit FP64 reference scope")
    result = {"valid": True, "checks": {}, "role": "engineering checks, not model-score accuracy"}
    ctl = ProposalControls(proposal_policy="original_three_branch", reset_steps=5, reset_epsilon=.7)

    def check_error(name, actual, expected, tolerance):
        error = float(tf.reduce_max(tf.abs(actual-expected)/(1+tf.abs(expected))))
        result["checks"][name] = dict(max_scaled_error=error, tolerance=tolerance)
        result["valid"] &= math.isfinite(error) and error <= tolerance
        m.save(args.output / "ladder-check.json", result)

    sig = [tf.TensorSpec([32, 3], dtype), tf.TensorSpec([32, 3], dtype),
           tf.TensorSpec([32], dtype), tf.TensorSpec([32], dtype)]
    @tf.function(input_signature=sig, jit_compile=True, autograph=False)
    def small(x, dx, lw, dlw):
        return (_dense_reset_transport(x, dx, lw, dlw, ctl),
                tiled_sinkhorn_barycentres(x, dx, lw, dlw, ctl))
    x = tf.random.stateless_normal([32, 3], [707, 1], dtype=dtype)*.3
    dx = tf.random.stateless_normal([32, 3], [707, 2], dtype=dtype)*.1
    lw = tf.nn.log_softmax(tf.random.stateless_normal([32], [707, 3], dtype=dtype))
    dlw = tf.random.stateless_normal([32], [707, 4], dtype=dtype)
    dlw -= tf.reduce_sum(tf.exp(lw)*dlw)
    dense, tiled = small(x, dx, lw, dlw)
    for i, label in enumerate(("barycentre", "tangent", "row_error", "column_error")):
        check_error("single_block_"+label, tiled[i], dense[i], 2e-11)

    @tf.function(input_signature=[tf.TensorSpec([], dtype)], jit_compile=True, autograph=False)
    def grouped(h):
        x = tf.constant([[-1.], [-.4], [.3], [1.2]], dtype)
        dx = tf.constant([[-.2], [.1], [.4], [-.1]], dtype)
        raw = tf.math.log(tf.constant([.1, .2, .25, .45], dtype))
        draw = tf.constant([.2, -.1, .3, -.4], dtype)
        lw = tf.nn.log_softmax(raw+h*draw)
        dlw = draw-tf.reduce_sum(tf.exp(lw)*draw)
        x += h*dx
        small_result = reset_cloud(x, dx, lw, dlw, ctl)
        large_result = reset_cloud(tf.repeat(x, 1008, axis=0), tf.repeat(dx, 1008, axis=0),
            tf.repeat(lw-tf.math.log(tf.constant(1008., dtype)), 1008), tf.repeat(dlw, 1008), ctl)
        return small_result, large_result
    small_result, large_result = grouped(tf.constant(0., dtype))
    result["checks"]["chunk_policy"] = {str(n):select_transport_chunks(n).row_chunk_size for n in (1008, 2016, 4032)}
    result["valid"] &= bool(small_result[6]) and bool(large_result[6])
    for i, label in enumerate(("reset_cloud", "reset_tangent")):
        check_error("multiblock_replicated_"+label, large_result[i], tf.repeat(small_result[i], 1008, axis=0), 2e-10)
    for i, label in ((2,"mean"),(3,"mean_tangent"),(4,"covariance"),(5,"covariance_tangent")):
        check_error("multiblock_replicated_"+label, large_result[i], small_result[i], 2e-10)

    # A different full-rank cloud checks total state/weight derivatives through
    # both axes of the real N4032 block grid, without a forbidden dense N4032 path.
    n = 4032
    cloud = tf.random.stateless_normal([n, 3], [708, 1], dtype=dtype)*.2
    tangent = tf.random.stateless_normal([n, 3], [708, 2], dtype=dtype)*.1
    raw = tf.random.stateless_normal([n], [708, 3], dtype=dtype)*.2
    draw = tf.random.stateless_normal([n], [708, 4], dtype=dtype)*.1
    probe = tf.random.stateless_normal([n, 3], [708, 5], dtype=dtype)
    @tf.function(input_signature=[tf.TensorSpec([], dtype)], jit_compile=True, autograph=False)
    def objective(h):
        lw = tf.nn.log_softmax(raw+h*draw)
        dlw = draw-tf.reduce_sum(tf.exp(lw)*draw)
        r = reset_cloud(cloud+h*tangent, tangent, lw, dlw, ctl)
        return tf.reduce_sum(r[0]*probe)/n, tf.reduce_sum(r[1]*probe)/n, r[6], r[7]
    base = objective(tf.constant(0., dtype))
    result["valid"] &= bool(base[2])
    for step in (1e-4, 5e-5):
        plus, minus = objective(tf.constant(step, dtype)), objective(tf.constant(-step, dtype))
        result["valid"] &= bool(plus[2]) and bool(minus[2])
        check_error("multiblock_fd_"+str(step), base[1], (plus[0]-minus[0])/(2*step), 2e-7)
    result["checks"]["multiblock_reset_diagnostics"] = m.clean(base[3])

    # Baseline source reuse is deliberate: only the exact-tiling extension is
    # permitted to differ among previously recorded numerical dependencies.
    old_manifest = m.load(BASE / "compare-manifest.json")
    allowed_change = "bayesfilter/highdim/covariance_proposal_tf.py"
    changed = []
    for path, expected in old_manifest["source_sha256"].items():
        if path.startswith(("bayesfilter/", "experiments/")):
            actual = hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
            if actual != expected:
                changed.append(path)
    if set(changed) - {allowed_change}:
        raise ValueError("Unreviewed numerical changes since baseline: "+str(changed))
    result["checks"]["changed_baseline_numerical_dependencies"] = changed
    frozen = m.load(BASE / "tuning.json")
    baseline_args = SimpleNamespace(**vars(args))
    baseline_args.output, baseline_args.particles = BASE, 1008
    if m.scope(baseline_args) != frozen["scope"]:
        raise ValueError("Baseline scope does not match FP64 ladder")
    spec, binding = m.specs(tf, dtype)
    selected = frozen["selected"]["new"]
    program = make_filter(binding, 1008, 20, ProposalControls(**{"proposal_policy": "original_three_branch", **selected["controls"]}), dtype)
    saved = [json.loads(line) for line in (BASE/"compare.jsonl").read_text().splitlines()]
    replay = []
    for ds, seed in enumerate(m.HELDOUT):
        obs = m.observations(baseline_args, tf, seed, dtype)
        fixed = m.inputs(tf, spec, 1008, 31001+100*ds, dtype)
        for point, theta in enumerate(m.POINTS):
            expected = next(r for r in saved if (r["data_seed"],r["point"],r["seed"],r["arm"])==(seed,point,31001+100*ds,"new"))
            actual = m.new_result(tf, program, tf.constant(theta,dtype), obs, fixed, tf.constant(selected["beta"],dtype))
            result["valid"] &= actual["valid"] == expected["valid"]
            error = max(abs(a-b)/(1+abs(b)) for a,b in zip([actual["value"],*actual["score"]],[expected["value"],*expected["score"]]))
            replay.append(dict(data_seed=seed,point=point,max_scaled_error=error))
            result["valid"] &= math.isfinite(error) and error <= 1e-10
    result["checks"]["N1008_six_cell_replay"] = replay
    m.save(args.output/"ladder-check.json",result)
    print(json.dumps(m.clean(result)),flush=True)
    if not result["valid"]:
        raise ValueError("Particle-ladder engineering checks failed")


def summarize_group(rows, reference):
    if len(rows)!=8 or len({r["seed"] for r in rows})!=8 or not all(r["valid"] for r in rows):
        return dict(valid=False, rows=len(rows), valid_rows=sum(bool(r["valid"]) for r in rows))
    errors=[math.dist(r["score"],reference["score"]) for r in rows]
    centroid=[statistics.mean(r["score"][j] for r in rows) for j in range(2)]
    spread=math.sqrt(sum(statistics.variance(r["score"][j] for r in rows) for j in range(2)))
    centre_error=math.dist(centroid,reference["score"])
    rmse=math.sqrt(statistics.mean(e*e for e in errors))
    if not math.isclose(rmse*rmse,centre_error**2+7/8*spread**2,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError("Variance decomposition failed")
    return dict(valid=True, rows=8, score_error_mean=statistics.mean(errors),
        score_error_sd=statistics.stdev(errors), score_spread=spread, score_centroid_error=centre_error,
        score_error_max=max(errors), score_rmse=rmse,
        loglik_mean_offset=statistics.mean(r["value"] for r in rows)-reference["value"],
        loglik_mean_absolute_error=statistics.mean(abs(r["value"]-reference["value"]) for r in rows),
        loglik_sd=statistics.stdev(r["value"] for r in rows),
        seconds_median=statistics.median(r["wall_seconds"] for r in rows))


def assemble(root, execution):
    m=driver()
    directories={1008:BASE,2016:root/"N2016",4032:root/"N4032"}
    rows={}
    for n,directory in directories.items():
        path=directory/"compare.jsonl"
        rows[n]=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    original=m.load(BASE/"report.json")
    cells=[]
    for old_cell in original["cells"]:
        seed,point=old_cell["data_seed"],old_cell["point"]
        ref=old_cell["reference"]
        summary={}
        baseline={r["seed"]:r for r in rows[1008] if (r["data_seed"],r["point"],r["arm"])==(seed,point,"new")}
        for n in directories:
            groups={arm:[r for r in rows[n] if (r["data_seed"],r["point"],r["arm"])==(seed,point,arm)] for arm in ("new","old","prior","global_guide")}
            arms={arm:summarize_group(group,ref) for arm,group in groups.items()}
            contrast=None
            if n!=1008 and arms["new"]["valid"]:
                differences=[math.dist(r["score"],ref["score"])-math.dist(baseline[r["seed"]]["score"],ref["score"]) for r in groups["new"]]
                centre=statistics.mean(differences)
                half=2.364624252*statistics.stdev(differences)/math.sqrt(8)
                envelope=[centre-half-old_cell["reference_sensitivity"],centre+half+old_cell["reference_sensitivity"]]
                contrast=dict(mean_difference=centre,paired_95_interval=[centre-half,centre+half],
                    reference_sensitive_interval=envelope,
                    verdict="conditional_improvement" if envelope[1]<0 else "conditional_deterioration" if envelope[0]>0 else "no_supported_ranking",
                    score_spread_ratio=arms["new"]["score_spread"]/summarize_group(list(baseline.values()),ref)["score_spread"])
            bootstrap_path=directories[n]/"references"/f"{seed}-{point}-{n}.json"
            if bootstrap_path.exists():
                b=m.checked_reference(bootstrap_path)
                arms["bootstrap_fisher"]=dict(valid=True,rows=16,score_error_mean=statistics.mean(math.dist(s,ref["score"]) for s in b["replicate_scores"]))
            losses=[arm for arm in ("prior","global_guide","bootstrap_fisher")
                    if arms["new"]["valid"] and arms.get(arm,{}).get("valid") and arms[arm]["score_error_mean"]<arms["new"]["score_error_mean"]]
            summary[str(n)]=dict(arms=arms,new_vs_N1008=contrast,heuristic_losses_descriptive=losses,
                heuristic_complete=all(arms.get(a,{}).get("valid") for a in ("prior","global_guide","bootstrap_fisher")))
        cells.append(dict(data_seed=seed,point=point,reference=ref,counts=summary))
    complete=all(c["counts"][str(n)]["arms"][arm]["valid"] for c in cells for n in (2016,4032) for arm in ("new","old"))
    result=dict(complete=complete,cells=cells,execution=execution,
        uncertainty="Eight particle designs, three inspected datasets; intervals exploratory and not multiplicity-adjusted; finite-particle reference bias unbounded.",
        role="particle-count diagnostic; no HMC/default/all-model claim")
    m.save(root/"ladder-report.json",result)
    text="# Range-bearing particle-count ladder\n\n"
    text+=("Both requested counts completed.\n\n" if complete else "The ladder is incomplete; preserved valid results are reported below.\n\n")
    text+="FP64/TF32-off, GPU/XLA, T20, state4/observation2/parameter2; eight paired designs per cell. Each N was tuned separately on the same disjoint calibration and validation protocol. These are filter score diagnostics preparing surrogate HMC; no HMC chain was run.\n\n"
    text+="| Data | Point | N1008 mean score error / spread | N2016 mean score error / spread | N4032 mean score error / spread |\n|---|---|---:|---:|---:|\n"
    def format_metric(s):
        return f'{s["score_error_mean"]:.3f} / {s["score_spread"]:.3f}' if s["valid"] else f'invalid/incomplete ({s["valid_rows"]}/{s["rows"]})'
    for c in cells:
        text+=f'| {c["data_seed"]} | {"truth" if c["point"]==0 else "difficult"} | '+ " | ".join(format_metric(c["counts"][str(n)]["arms"]["new"]) for n in (1008,2016,4032))+" |\n"
    text+="\nSpread is sqrt(sum of sample coordinate variances). It is across particle designs, not HMC MCSE. Centroid discrepancies, likelihood offsets/SD, every comparator, and full intervals are in ladder-report.json.\n\n"
    text+="| Data | Point | N | New-minus-N1008 score error | Reference-sensitive interval | Verdict | Heuristics with lower descriptive mean |\n|---|---|---:|---:|---|---|---|\n"
    for c in cells:
        for n in (2016,4032):
            row=c["counts"][str(n)];contrast=row["new_vs_N1008"]
            if contrast:
                lo,hi=contrast["reference_sensitive_interval"]
                text+=f'| {c["data_seed"]} | {c["point"]} | {n} | {contrast["mean_difference"]:.3f} | [{lo:.3f}, {hi:.3f}] | {contrast["verdict"]} | {", ".join(row["heuristic_losses_descriptive"]) or "none among valid arms"} |\n'
    text+="\nAn interval spanning zero supports no ranking. These intervals are nominal, conditional and exploratory, following inspection of N1008 and with no multiple-comparison adjustment. Smaller point-estimate spread is descriptive only. Identical design-seed indices across N are coupled labels, not nested Gaussian draws. Retuning also changes settings, so this compares tuned particle budgets, not N with all controls held fixed.\n\n"
    text+="| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |\n|---|---|---|---|---|---|\n"
    text+=f'| Requested extension {"complete" if complete else "incomplete"} | Per-cell errors and spreads above | Invalid/missing cells retained | Eight designs and reference bias | Assess materiality for the downstream target | Necessary final N |\n'
    text+="| Default/HMC status unchanged | No default-precision/posterior test | Heuristic losses/incomplete screens above | Precision and posterior consequences | A separate tuned FP32 scope and actual HMC would answer those questions | Default readiness, posterior convergence, all-model improvement |\n\n"
    text+="| Inference status | Assessment |\n|---|---|\n| Hard veto screen | See per-arm valid counts in JSON; no invalid-subset ranking |\n| Statistically supported ranking | Only conditional exploratory interval verdicts shown above |\n| Descriptive differences | Spread ratios, log-likelihood SD, centroids, runtime and heuristic means |\n| Default readiness | Not established: explicit FP64 diagnostic |\n| Next evidence needed | Independent replication/reference refinement and downstream HMC materiality |\n\n"
    text+="The strongest alternative explanation is that a larger particle count selects a different calibrated mixture, or that a few baseline outliers drive the contrast. Reference bias, few designs and previously inspected datasets limit the inference. Fresh data/particle designs reversing the result, or a refined reference revealing a persistent shift, would overturn an improvement interpretation. More particles do not automatically establish a correct posterior.\n\n"
    text+=f'Artifacts: {root.relative_to(ROOT)}. Plan: {PLAN.relative_to(ROOT)}. Numerical commands, source hashes, environment, verified GPU memory growth, allocator usage, seeds and elapsed times are in execution.json and per-worker manifests. Aggregate worker wall time: {execution["worker_seconds"]:.1f}s of {execution["budget_seconds"]}s; {len(execution["attempts"])} of 20 workers.\n'
    RESULT.write_text(text)
    return result


def checkpoint(root, stage, execution):
    CHECKPOINT.write_text(
        "# Active checkpoint: authorized M13 particle ladder\n"
        "Question: do separately tuned N2016/N4032 reduce score error and design variability versus N1008?\n"
        "Checkout/branch: BayesFilter-ledh-graceful-failure-execution-20260917 / ledh-graceful-failure-execution-20260917.\n"
        "Preserve unrelated staged/dirty work. Same M13 T20/data/points; FP64/TF32-off isolates N. No HMC run.\n"
        f"Stage: {stage}.\nPlan: {PLAN.relative_to(ROOT)}.\n"
        f"Evidence: {root.relative_to(ROOT)}.\nResult: {RESULT.relative_to(ROOT)}.\n"
        f'Budget: {execution["worker_seconds"]:.1f}s/{execution["budget_seconds"]}s; {len(execution["attempts"])}/20 workers.\n'
        "Continuation: finite blocked transport parity and derivatives must pass; candidate failures alone do not cancel the other count.\n"
        "Run through scripts/run_range_bearing_covariance_comparison.py run --particle-ladder; GPU escalation and memory growth required.\n")


def run_ladder(args):
    m=driver()
    if args.dtype!="float64" or not args.no_tf32 or args.replicates!=8:
        raise ValueError("Ladder must match the FP64/TF32-off eight-design baseline")
    if not 0 < args.budget_seconds <= 14400:
        raise ValueError("Authorized particle ladder is bounded by four hours")
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    execution=dict(budget_seconds=args.budget_seconds,worker_seconds=0.,attempts=[],
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        command=[sys.executable,*sys.argv],plan=str(PLAN),result=str(RESULT),
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline=str(BASE))
    if args.retry_from:
        previous=m.load(args.retry_from / "execution.json")
        if previous["baseline"]!=str(BASE) or previous["plan"]!=str(PLAN) or previous["budget_seconds"]!=args.budget_seconds:
            raise ValueError("Retry changes the authorized contract or budget")
        execution["attempts"]=previous["attempts"]
        execution["worker_seconds"]=previous["worker_seconds"]
        execution["retry"]=dict(source=str(args.retry_from),failure=previous.get("continuation_veto"),repair="Explicit static shapes after TensorArray concatenation; no mathematics changed")
    def persist(stage):
        m.save(root/"execution.json",execution)
        checkpoint(root,stage,execution)
    def job(stage,n,directory,cpu=False):
        remaining=args.budget_seconds-execution["worker_seconds"]
        if remaining<=0 or len(execution["attempts"])>=20:
            return False
        directory.mkdir(parents=True,exist_ok=True)
        if cpu:
            command=[sys.executable,"-B","-m","pytest","-q","tests/highdim/test_covariance_proposal.py","tests/diagnostics/test_range_bearing_covariance_report.py"]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",TF_CPP_MIN_LOG_LEVEL="2",TF_FORCE_GPU_ALLOW_GROWTH="true")
        else:
            command=[sys.executable,"-B",str(ROOT/"scripts/run_range_bearing_covariance_comparison.py"),
                "worker","--stage",stage,"--particles",str(n),"--dtype","float64","--no-tf32",
                "--replicates","8","--gpu",str(args.gpu),"--output",str(directory),"--plan-file",str(PLAN)]
            env=None
        m.save(root/"next-phase.json",dict(stage=stage,particles=n,command=command,remaining_seconds=remaining,
            reason="Authorized plan phase; valid earlier phases permit automatic continuation"))
        persist("running "+stage+" N"+str(n))
        began=time.monotonic()
        with (directory/(stage+".log")).open("x") as stream:
            try:
                run=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=remaining)
                code=run.returncode
            except subprocess.TimeoutExpired:
                code="budget_timeout"
        elapsed=time.monotonic()-began
        execution["worker_seconds"]+=elapsed
        attempt=dict(stage=stage,particles=n,command=command,exit_code=code,wall_seconds=elapsed,
                     cpu_only=cpu,gpu_intentionally_hidden=cpu,output=str(directory))
        execution["attempts"].append(attempt)
        persist("completed "+stage+" N"+str(n))
        print(json.dumps(attempt),flush=True)
        return code==0
    persist("preflight; plan audited, blocked-reset checks pending")
    if not any(a["stage"]=="healthy-checks" and a["exit_code"]==0 for a in execution["attempts"]) and not job("healthy-checks",0,root/"checks",cpu=True):
        execution["continuation_veto"]="healthy regression failed"
        persist("continuation veto; healthy regression")
        return
    if not job("ladder-check",4032,root/"checks"):
        execution["continuation_veto"]="exact block parity, derivative or baseline reuse failed"
        persist("continuation veto; repair checks")
        return
    execution["repair_checks"]=m.load(root/"checks/ladder-check.json")
    for n in (2016,4032):
        directory=root/("N"+str(n));directory.mkdir(exist_ok=False)
        reuse={}
        for name in ("references","data"):
            shutil.copytree(BASE/name,directory/name)
            for path in sorted((BASE/name).glob("*.json")):
                reuse[str(path.relative_to(BASE))]=hashlib.sha256(path.read_bytes()).hexdigest()
        m.save(directory/"reuse.json",dict(source=str(BASE),sha256=reuse,
            role="identical observations and independent references; tuning is NOT copied"))
        for stage in ("check-fp64","references","tune","compare"):
            if not job(stage,n,directory):
                break
            if stage=="check-fp64" and not m.load(directory/"check-fp64.json")["valid"]:
                execution.setdefault("rejected_counts",{})[str(n)]="default preflight invalid"
                break
            if stage=="tune" and not m.load(directory/"tuning.json").get("valid"):
                execution.setdefault("rejected_counts",{})[str(n)]="no valid frozen scope-specific tuning"
                break
        child=SimpleNamespace(**vars(args));child.output,child.particles=directory,n
        capture=io.StringIO()
        with contextlib.redirect_stdout(capture):
            m.report(child)
        (directory/"report.log").write_text(capture.getvalue())
        persist("N"+str(n)+" assessed; continuing remaining planned count")
    result=assemble(root,execution)
    execution["complete"]=result["complete"]
    persist("terminal: both counts assessed" if result["complete"] else "terminal: incomplete; see preserved failures")
    m.save(root/"next-phase.json",dict(stage="terminal",complete=result["complete"],result=str(RESULT),
        reason="Requested particle-count extension is assessed; no implicit HMC or precision-scope launch"))
    print(json.dumps(dict(complete=result["complete"],result=str(RESULT),artifact=str(root))),flush=True)


def progress(root):
    """Read-only bounded progress, including only complete new-method cells."""
    m=driver()
    root=Path(root)
    phase=m.load(root/"next-phase.json")
    execution=m.load(root/"execution.json")
    result=dict(stage=phase["stage"],particles=phase.get("particles"),
        completed_worker_seconds=execution["worker_seconds"],budget_seconds=execution["budget_seconds"],
        completed_workers=len(execution["attempts"]),counts={})
    baseline=m.load(BASE/"report.json")
    for n in (2016,4032):
        directory=root/("N"+str(n))
        rowfile=directory/"compare.jsonl"
        rows=[json.loads(line) for line in rowfile.read_text().splitlines()] if rowfile.exists() else []
        item=dict(rows=len(rows),valid_by_arm={arm:sum(r["valid"] for r in rows if r["arm"]==arm) for arm in ("new","old","prior","global_guide")})
        tuning=directory/"tuning.json"
        if tuning.exists():
            ledger=m.load(tuning)
            item["tuning_candidates"]=len(ledger["candidates"])
            item["frozen"]=ledger.get("valid",False)
            item["selected"]=ledger["selected"]
        complete_cells=[]
        for cell in baseline["cells"]:
            group=[r for r in rows if (r["arm"],r["data_seed"],r["point"])==("new",cell["data_seed"],cell["point"])]
            if len(group)==8:
                metrics=summarize_group(group,cell["reference"])
                complete_cells.append(dict(data=cell["data_seed"],point=cell["point"],
                    **{k:round(v,4) if isinstance(v,(int,float)) else v for k,v in metrics.items() if k in ("valid","score_error_mean","score_spread","score_centroid_error","loglik_sd")}))
        item["completed_cells"]=complete_cells
        result["counts"][str(n)]=item
    print(json.dumps(result,allow_nan=False))

"""Terminal read-only numerical audit of the independent R reference campaign."""
import csv
import datetime
import hashlib
import json
import math
import os
import pathlib
import shutil
import statistics
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01"
PLAN = ROOT / "docs/plans/iapf-scale-aware-reconstruction-20260926.md"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))

R_AUDIT = r'''
args <- commandArgs(TRUE); campaign <- args[1]; out <- args[2]
source(file.path(campaign,"attempt04/sources/reference_iapf_paper.R"))
source(file.path(campaign,"attempt04/sources/reference_iapf_scale_aware_fit.R"))
write_rows <- function(x,name) write.csv(x,file.path(out,name),row.names=FALSE)
checks <- list(); failures <- list()
for(seed in c(9262001,9262002)) {
  saved <- readRDS(file.path(campaign,"attempt03",paste0("data-",seed,".rds")))
  data <- iapf_paper_data(d=5,horizon=100,seed=seed)
  data_error <- max(abs(data$observations-saved$bundle$data$observations))
  model <- saved$bundle$model
  set.seed(9264000+seed-9262000)
  replay <- iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),1000)
  cloud_error <- max(vapply(seq_len(100),function(t)
    max(abs(replay$clouds[[t]]-saved$baseline$clouds[[t]])),0))
  for(power in c(2,1)) {
    payload <- readRDS(file.path(campaign,"attempt03",paste0("min_mass_Nminus",power,"-",seed,".rds")))
    input <- payload$inputs[[payload$time]]; fit <- payload$fit
    target <- iapf_backward_log_target(model,input$points,payload$time,input$next_twist)
    target_error <- max(abs(target-input$log_targets))
    problem <- sa_problem(input$points,input$log_targets,fit$initial)
    final <- problem$evaluate(fit$z); initial <- problem$evaluate(rep(0,length(fit$z)))
    gradient_error <- abs(final$relative_gradient-fit$relative_gradient)/max(1,fit$relative_gradient)
    checks[[length(checks)+1]] <- data.frame(seed=seed,power=power,data_error=data_error,
      cloud_error=cloud_error,target_error=target_error,gradient_error=gradient_error,
      pass=data_error==0 && cloud_error==0 && target_error<=1e-12 && gradient_error<=1e-12)
    d <- ncol(input$points); final_v <- exp(fit$theta[d+seq_len(d)])
    failures[[length(failures)+1]] <- data.frame(seed=seed,power=power,time=payload$time,
      start_shape=initial$shape,final_shape=final$shape,
      initial_relative_gradient=initial$relative_gradient,final_relative_gradient=final$relative_gradient,
      start_min_variance=min(exp(fit$initial[d+seq_len(d)])),start_max_variance=max(exp(fit$initial[d+seq_len(d)])),
      final_min_variance=min(final_v),final_max_variance=max(final_v),
      mean_movement=max(abs(fit$theta[seq_len(d)]-fit$initial[seq_len(d)])),
      max_final_log_density=max(sa_log_density(input$points,fit$theta)),
      initial_log_loss=initial$log_raw_loss,final_log_loss=final$log_raw_loss,
      initial_log_lambda=initial$log_lambda,final_log_lambda=final$log_lambda)
  }
}
checks <- do.call(rbind,checks); write_rows(checks,"wiring_checks.csv")
write_rows(do.call(rbind,failures),"failure_anatomy.csv")
stopifnot(all(checks$pass))
cat("Terminal wiring and numerical replay PASS\n")
'''

def main():
    manifests = [json.loads(p.read_text()) for p in sorted(CAMPAIGN.glob("attempt*/manifest.json"))]
    if len(manifests) != 5 or any(m["status"] == "RUNNING" for m in manifests):
        raise SystemExit("Expected five terminal attempts before this single terminal audit")
    prior = sum(m["wall_seconds"] for m in manifests)
    timeout = min(60, 1500-prior)
    if timeout < 1:
        raise SystemExit("No remaining CPU budget")
    out = CAMPAIGN / "attempt06"
    out.mkdir(exist_ok=False)
    sources = out / "sources"
    sources.mkdir()
    shutil.copy2(pathlib.Path(__file__).resolve(), sources / pathlib.Path(__file__).name)
    shutil.copy2(PLAN, sources / PLAN.name)
    audit_script = sources / "terminal_reference_audit.R"
    audit_script.write_text(R_AUDIT)
    command = ["/usr/bin/Rscript", "--vanilla", str(audit_script), str(CAMPAIGN), str(out)]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = dict(phase="terminal_audit", status="RUNNING", start_utc=stamp(), command=command,
        git_commit=manifests[-1]["git_commit"], branch=manifests[-1]["branch"],
        reference_git_commit=manifests[-1]["reference_git_commit"], reference_branch=manifests[-1]["reference_branch"],
        source_sha256={str(p): digest(p) for p in sources.iterdir()},
        input_sha256={str(p): digest(p) for a in CAMPAIGN.glob("attempt0[1-5]") for p in a.iterdir() if p.is_file()},
        environment={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        interpreter=manifests[-1]["interpreter"], hardware="Independent FP64 CPU reference; GPU deliberately hidden",
        seeds={"replayed_data": [9262001, 9262002], "replayed_filter": [9264001, 9264002]},
        data="Frozen campaign inputs; no new scientific evaluation data", GPU_seconds=0,
        CPU_wall_limit=timeout, total_CPU_wall_limit=1500, previous_CPU_wall_seconds=prior,
        prior_combined_CPU_wall_seconds=89.936265, plan=str(PLAN), result=str(out/"result.md"), output_root=str(out))
    write(out/"manifest.json", manifest)
    started = time.monotonic()
    verified_sources = verified_outputs = verified_inputs = 0
    try:
        for m in manifests:
            attempt = pathlib.Path(m["output_root"])
            for original, expected in m["source_sha256"].items():
                assert digest(attempt/"sources"/pathlib.Path(original).name) == expected, original
                verified_sources += 1
            for name, expected in m["output_sha256"].items():
                assert digest(attempt/name) == expected, name
                verified_outputs += 1
            for name, expected in m["input_sha256"].items():
                assert digest(pathlib.Path(name)) == expected, name
                verified_inputs += 1
        with (out/"worker.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=True)
        evaluation = rows(CAMPAIGN/"attempt05/evaluation.csv")
        assert len(evaluation) == 24
        summaries = []
        for method in ("BPF", "observation_only", "fully_adapted"):
            arm = [r for r in evaluation if r["method"] == method]
            assert len(arm) == 8 and {r["N"] for r in arm} == {"1000"}
            metrics = {k: [float(r[k]) for r in arm] for k in ("total_error", "ordinary_mse", "large_mse", "seconds")}
            assert all(math.isfinite(v) for values in metrics.values() for v in values)
            summaries.append(dict(method=method,draws=8,N=1000,
                mean_total_log_error=statistics.mean(metrics["total_error"]),
                total_log_error_rmse=math.sqrt(statistics.mean(v*v for v in metrics["total_error"])),
                mean_ordinary_increment_mse=statistics.mean(metrics["ordinary_mse"]),
                mean_large_increment_mse=statistics.mean(metrics["large_mse"]),
                mean_seconds=statistics.mean(metrics["seconds"]),
                n_large=int(arm[0]["n_large"]),n_ordinary=int(arm[0]["n_ordinary"])))
        result = dict(status="PASS",verified_sources=verified_sources,verified_outputs=verified_outputs,
            verified_inputs=verified_inputs,baseline_summary=summaries,
            fitted_adaptive_pilots=0,heuristic_dominance="NOT_EVALUABLE_FOR_IAPF_NO_ELIGIBLE_FITTED_ARM",
            statistical_ranking="NOT_ESTABLISHED",replication="INCOMPLETE")
        write(out/"audit.json", result)
        manifest.update(status="COMPLETED", exit_code=0)
    except Exception as exc:
        manifest.update(status="FAILED", exit_code=1, error=str(exc))
        raise
    finally:
        manifest.update(end_utc=stamp(), wall_seconds=time.monotonic()-started)
        manifest["output_sha256"] = {p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!="manifest.json"}
        write(out/"manifest.json", manifest)
        with (CAMPAIGN/"attempts.jsonl").open("a") as stream:
            stream.write(json.dumps(manifest)+"\n")
        (out/"result.md").write_text(f"# Terminal audit\n\n{manifest['status']}; CPU wall {manifest['wall_seconds']:.6f}s; GPU 0. See audit.json and wiring_checks.csv. No scientific promotion.\n")
    print(json.dumps(dict(status=manifest["status"],wall_seconds=manifest["wall_seconds"],
        verified_sources=verified_sources,verified_outputs=verified_outputs,verified_inputs=verified_inputs,
        output_root=str(out))), flush=True)

if __name__ == "__main__":
    main()

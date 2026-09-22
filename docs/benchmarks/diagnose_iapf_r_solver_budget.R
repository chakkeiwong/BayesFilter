# Calibration of an optimizer resource limit on preserved failed pilot inputs.
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1],"docs/benchmarks/reference_iapf_paper.R"))
old <- file.path(args[1],"docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01/attempt01-sensitivity/results")
output <- args[2]; stopifnot(!dir.exists(output)); dir.create(output,recursive=TRUE)
rows <- list()
for(limit in c(1000,5000)) {
  pass <- TRUE
  for(d in c(5,10)) {
    saved <- readRDS(file.path(old,paste0("d",d,"-baseline.rds")))
    started <- proc.time()[[3]]
    fit <- tryCatch(iapf_fit_gaussian(saved$points,saved$log_targets,
      maxit=limit,fit_mode="relative_l2"),error=function(e)e)
    saveRDS(fit,file.path(output,paste0("d",d,"-maxit",limit,".rds")))
    ok <- !inherits(fit,"error"); pass <- pass && ok
    rows[[length(rows)+1]] <- data.frame(dimension=d,maxit=limit,passed=ok,
      elapsed=proc.time()[[3]]-started,
      residual=if(ok)fit$diagnostics$relative_residual else fit$evaluation$relative_residual,
      gradient=if(ok)fit$diagnostics$gradient_max else max(abs(fit$evaluation$gradient)),
      evaluations=if(ok)fit$diagnostics$evaluations else NA,
      change_from_capped=if(ok)max(abs(c(fit$twist$mean,log(diag(fit$twist$covariance)))-saved$parameters)) else NA)
    write.csv(do.call(rbind,rows),file.path(output,"solver-budget.csv"),row.names=FALSE)
  }
  if(pass)break
}
print(do.call(rbind,rows))
if(!pass)quit(status=2)

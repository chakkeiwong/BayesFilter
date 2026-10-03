# Saved-input allowance calibration; no likelihood ranking or silent fallback.
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1],"docs/benchmarks/reference_iapf_paper.R"))
output <- args[2]; stopifnot(!dir.exists(output));dir.create(output,recursive=TRUE)
root <- "docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01"
saved <- readRDS(file.path(args[1],root,
  "attempt09-pilot-d20-nlminb/results/iapf-1-failure.rds"))
rows <- list()
for(budget in c(5000L,10000L)) {
  started <- proc.time()[[3]]
  fit <- tryCatch(iapf_fit_gaussian(saved$points,saved$log_targets,
    maxit=budget,fit_mode="relative_l2_nlminb"),error=function(e)e)
  saveRDS(fit,file.path(output,paste0("budget",budget,".rds")))
  ok <- !inherits(fit,"error")
  rows[[length(rows)+1]] <- data.frame(maxit=budget,passed=ok,
    message=if(ok)"converged" else fit$optimizer_message,
    residual=if(ok)fit$diagnostics$relative_residual else fit$evaluation$relative_residual,
    gradient=if(ok)fit$diagnostics$gradient_max else max(abs(fit$evaluation$gradient)),
    elapsed=proc.time()[[3]]-started)
  write.csv(do.call(rbind,rows),file.path(output,"solver-budget.csv"),row.names=FALSE)
}
print(do.call(rbind,rows))
if(!tail(rows,1)[[1]]$passed)quit(status=2)

# Diagnostic replay of all saved failure classes; never a runtime fallback.
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1],"docs/benchmarks/reference_iapf_paper.R"))
output <- args[2]; stopifnot(!dir.exists(output));dir.create(output,recursive=TRUE)
old <- "docs/plans/artifacts/iapf-r-paper-replication-20260920-01/mechanics02"
new <- "docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01"
paths <- c(file.path(old,paste0(c("failed-fit","boundary-d5","boundary-d10"),".rds")),
  file.path(new,"attempt01-sensitivity/results",paste0(c("d5-baseline","d10-baseline"),".rds")),
  file.path(new,"attempt02-sensitivity/results/d10-baseline.rds"),
  file.path(new,"attempt08-pilot-d20/results/iapf-1-failure.rds"))
rows <- list()
for(i in seq_along(paths)) {
  saved <- readRDS(file.path(args[1],paths[i])); started <- proc.time()[[3]]
  fit <- tryCatch(iapf_fit_gaussian(saved$points,saved$log_targets,
    maxit=5000,fit_mode="relative_l2_nlminb"),error=function(e)e)
  saveRDS(fit,file.path(output,paste0("case",i,".rds")))
  ok <- !inherits(fit,"error")
  rows[[i]] <- data.frame(case=i,path=paths[i],dimension=ncol(saved$points),
    passed=ok,elapsed=proc.time()[[3]]-started,
    residual=if(ok)fit$diagnostics$relative_residual else fit$evaluation$relative_residual,
    gradient=if(ok)fit$diagnostics$gradient_max else max(abs(fit$evaluation$gradient)),
    evaluations=if(ok)fit$diagnostics$evaluations else NA)
  write.csv(do.call(rbind,rows),file.path(output,"solver-replay.csv"),row.names=FALSE)
}
print(do.call(rbind,rows)[,c("case","dimension","passed","elapsed","residual","gradient","evaluations")])
if(!all(vapply(rows,function(x)x$passed,TRUE)))quit(status=2)

# Focused diagnostic replay, not an empirical replication sample.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
stopifnot(!dir.exists(output)); dir.create(output,recursive=TRUE)
old <- file.path(root,"docs/plans/artifacts/iapf-r-paper-replication-20260920-01/mechanics02")
rows <- list()
for(name in c("failed-fit","boundary-d5","boundary-d10")) {
  saved <- readRDS(file.path(old,paste0(name,".rds")))
  initial <- iapf_fit_initial(saved$points,saved$log_targets)
  before <- iapf_relative_fit_objective(initial$parameters,saved$points,saved$log_targets)
  start <- proc.time()[[3]]
  fit <- tryCatch(iapf_fit_gaussian(saved$points,saved$log_targets,fit_mode="relative_l2"),
    error=function(e)e)
  saveRDS(fit,file.path(output,paste0(name,"-relative.rds")))
  failed <- inherits(fit,"error")
  row <- data.frame(case=name,dimension=ncol(saved$points),N=nrow(saved$points),
    elapsed=proc.time()[[3]]-start,passed=!failed,error=if(failed)conditionMessage(fit) else "",
    initial_residual=before$value,final_residual=if(failed)NA else fit$diagnostics$relative_residual,
    max_log_variance=if(failed)NA else fit$diagnostics$log_variance_max,
    evaluations=if(failed)NA else fit$diagnostics$evaluations,
    gradient_max=if(failed)NA else fit$diagnostics$gradient_max)
  rows[[length(rows)+1]] <- row
  write.csv(do.call(rbind,rows),file.path(output,"replay.csv"),row.names=FALSE)
  print(row)
}
if(!all(vapply(rows,function(x)x$passed,TRUE)))quit(status=2)

# Independent CPU diagnostic: exact replay of the bounded d80 failure.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
stopifnot(as.integer(args[3])==80,as.integer(args[4])==1,
  as.integer(args[5])==1,as.integer(args[6])==72000080,
  args[7]=="controller_diagnostic",args[8]=="log_quadratic",!dir.exists(output))
dir.create(output,recursive=TRUE)
data <- iapf_paper_data(80,100,seed=72000080)
model <- iapf_linear_model(data$observations,data$A)
kalman <- iapf_kalman(model)
write.csv(data$observations,file.path(output,"observations.csv"),row.names=FALSE)
saveRDS(list(model=model,kalman=kalman),file.path(output,"model.rds"))
old <- readRDS(file.path(root,paste0("docs/plans/artifacts/",
  "iapf-r-log-fit-validation-20260920-01/attempt04-d80-pilot/results/iapf-1.rds")))
iteration_call <- 0L; controller <- list(); prefixes <- list(); fitted <- list()
filter <- function(model,twists,N,kappa,store_clouds) {
  iteration_call <<- iteration_call+1L
  i <- iteration_call
  saveRDS(list(twists=twists,N=N,kappa=kappa,rng=.Random.seed),
    file.path(output,paste0("filter-input-",i,".rds")))
  value <- iapf_apf(model,twists,N,kappa,store_clouds)
  errors <- value$prefix-kalman$prefix
  controller[[i]] <<- data.frame(iteration=i-1L,N=N,logZ=value$log_likelihood,
    log_error=value$log_likelihood-kalman$log_likelihood,
    min_ess=min(value$ess),min_ess_fraction=min(value$ess)/N,
    min_ess_time=which.min(value$ess),low_ess_times=sum(value$ess/N<.01),
    resampling_count=value$resampling_count,floor_probability_max=value$floor_probability_max,
    worst_prefix_error=min(errors),worst_prefix_time=which.min(errors))
  prefixes[[i]] <<- data.frame(iteration=i-1L,N=N,time=1:100,
    log_prefix_error=errors,ess=value$ess,ess_fraction=value$ess/N,resampled=value$resampled)
  write.csv(do.call(rbind,controller),file.path(output,"controller.csv"),row.names=FALSE)
  write.csv(do.call(rbind,prefixes),file.path(output,"prefixes.csv"),row.names=FALSE)
  cat("PASS",i,"N",N,"log_error",tail(errors,1),"min_ess",min(value$ess),"\n")
  flush.console(); value
}
fit <- function(...) {
  value <- iapf_fit_backward(...)
  i <- length(fitted)+1L
  fitted[[i]] <<- do.call(rbind,lapply(seq_along(value$diagnostics),function(t)
    data.frame(iteration=i-1L,time=t,as.data.frame(value$diagnostics[[t]]))))
  value
}
set.seed(61000011)
value <- iapf_iterate(model,.filter=filter,.fit=fit,fit_mode="log_quadratic")
saveRDS(value,file.path(output,"replayed-result.rds"))
write.csv(do.call(rbind,fitted),file.path(output,"fits.csv"),row.names=FALSE)
stopifnot(identical(value$status,old$status),identical(value$history,old$history),
  identical(value$counts,old$counts),identical(value$fits,old$fits))
exact <- iapf_exact_twists(model)
set.seed(72000980)
oracle <- iapf_apf(model,exact,16,store_clouds=FALSE)
oracle_error <- oracle$log_likelihood-kalman$log_likelihood
stopifnot(is.finite(oracle_error),abs(oracle_error)<=1e-7)
checks <- data.frame(check=c("same_status","identical_history","identical_counts",
  "identical_fits","exact_twist_Kalman"),passed=TRUE)
write.csv(checks,file.path(output,"replay-checks.csv"),row.names=FALSE)
write.csv(data.frame(status=value$status,iterations=length(value$history),
  final_training_N=tail(value$counts,1),last6_cv=iapf_likelihood_cv(tail(value$history,6)),
  kalman_log_likelihood=kalman$log_likelihood,oracle_log_error=oracle_error,
  oracle_min_ess=min(oracle$ess),oracle_N=16),
  file.path(output,"diagnostic-summary.csv"),row.names=FALSE)
cat("COMPLETE exact replay; observed",value$status,"oracle_error",oracle_error,"\n")

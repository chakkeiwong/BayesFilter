# Bounded complete learning through the shared controller, independent R only.
stopifnot(length(args)==5)
d <- as.integer(args[3]);data_seed <- as.integer(args[4]);arm <- args[5]
stopifnot(d %in% c(5L,20L,80L),arm %in% c("qr","ridge_bounded1"))
dat <- iapf_paper_data(d,100,seed=data_seed)
model <- iapf_linear_model(dat$observations,dat$A)
truth <- iapf_kalman(model);filter_seed <- 92240000L+d+1009L
set.seed(filter_seed);start <- proc.time()[[3]]
ans <- tryCatch(if(arm=="qr")iapf_choice_run(model,arm="qr",floor_rule="tail8",
    max_iterations=12,max_particles=4000) else iapf_constrained_diagnostic_run(model),
  error=function(e)e)
elapsed <- proc.time()[[3]]-start
failed <- inherits(ans,"error")
status <- if(failed)"candidate_failed" else ans$status
saveRDS(ans,file.path(out,"learning.rds"))
record <- data.frame(dimension=d,data_seed=data_seed,filter_seed=filter_seed,method=arm,
  status=status,message=if(failed)conditionMessage(ans) else "",wall_seconds=elapsed,
  failure_iteration=if(failed)v(ans,"iteration") else NA_real_,
  failure_time=if(failed)v(ans,"backward_time") else NA_real_,
  stop_iteration=if(failed)NA_real_ else v(ans,"stop_iteration"),
  final_particles=if(failed)NA_real_ else v(ans,"final_particles"),
  log_likelihood=if(status=="complete")ans$final$log_likelihood else NA_real_,
  kalman=truth$log_likelihood,
  terminal_error=if(status=="complete")ans$final$log_likelihood-truth$log_likelihood else NA_real_,
  prefix_error_max=if(status=="complete")max(abs(ans$final$prefix-truth$prefix)) else NA_real_)
write.csv(record,file.path(out,"learning.csv"),row.names=FALSE)
fit_rows <- list()
if(!failed && length(ans$fits))for(iteration in seq_along(ans$fits)) {
  for(time in seq_along(ans$fits[[iteration]])) {
    diag <- ans$fits[[iteration]][[time]]
    fit_rows[[length(fit_rows)+1L]] <- data.frame(iteration=iteration-1,time=time,
      kkt=v(diag,"kkt_residual"),lambda=v(diag,"lambda"),
      active_constraints=v(diag,"active_constraints"),weight_ess=v(diag,"weight_ess"),
      condition=v(diag,"design_condition",v(diag,"weighted_design_condition")),
      shape_training_rms=v(diag,"shape_training_rms"))
  }
} else if(failed) {
  diag <- ans$diagnostics
  fit_rows[[1L]] <- data.frame(iteration=v(ans,"iteration"),time=v(ans,"backward_time"),
    kkt=v(diag,"kkt_residual"),lambda=v(diag,"lambda"),
    active_constraints=v(diag,"active_constraints"),weight_ess=v(diag,"weight_ess"),
    condition=v(diag,"design_condition",v(diag,"weighted_design_condition")),
    shape_training_rms=v(diag,"shape_training_rms"))
}
write_table(fit_rows,"fit-diagnostics.csv")
cat("learning",d,data_seed,arm,status,if(failed)conditionMessage(ans) else record$terminal_error,"\n")

# Paired replay of the original failed consumer, followed by the frozen repair.
stopifnot(length(args)==4)
data_seed <- as.integer(args[3]);replication <- as.integer(args[4])
stopifnot(data_seed %in% c(92100180L,92100280L),replication %in% 1:4)
data_index <- match(data_seed,c(92100180L,92100280L))
path <- file.path(old,paste0("runs/controller_probe-d80-j",data_index,
  "-wlog1-tail8-population/results/wlog1-",replication,"-failure.rds"))
writeLines(path,file.path(out,"input-paths.txt"));saved <- readRDS(path)
dat <- iapf_paper_data(80,100,seed=data_seed)
model <- iapf_linear_model(dat$observations,dat$A);truth <- iapf_kalman(model)
filter_seed <- data_seed+1009L*replication+100003L
set.seed(filter_seed)
original <- tryCatch(iapf_choice_run(model,"wlog1",floor_rule="tail8",sd_mode="population",
  max_iterations=12,max_particles=4000),error=function(e)e)
check("original_failure_reproduced",as.numeric(!inherits(original,"iapf_fit_error")),0)
check("original_failure_time",abs(original$backward_time-saved$backward_time),0)
check("original_failure_cloud",max(abs(original$points-saved$points)),0)
check("original_failure_target",max(abs(original$log_targets-saved$log_targets)),0)
saveRDS(original,file.path(out,"original-failure.rds"))
original_filter <- iapf_apf;filter_calls <- 0L
iapf_apf <- function(...) {
  ans <- original_filter(...);filter_calls <<- filter_calls+1L
  if(filter_calls==1)check("repaired_consumer_same_initial_cloud",
    max(abs(ans$clouds[[saved$backward_time]]-saved$points)),0)
  ans
}
set.seed(filter_seed);start <- proc.time()[[3]]
repaired <- tryCatch(iapf_constrained_diagnostic_run(model,sd_mode="population"),error=function(e)e)
elapsed <- proc.time()[[3]]-start;failed <- inherits(repaired,"error")
status <- if(failed)"candidate_failed" else repaired$status
saveRDS(repaired,file.path(out,"repaired-learning.rds"))
record <- data.frame(dimension=80,data_seed=data_seed,replication=replication,
  filter_seed=filter_seed,method="ridge_bounded1",original_status="candidate_failed",
  original_message=conditionMessage(original),original_time=original$backward_time,
  status=status,message=if(failed)conditionMessage(repaired) else "",
  failure_iteration=if(failed)v(repaired,"iteration") else NA_real_,
  failure_time=if(failed)v(repaired,"backward_time") else NA_real_,
  wall_seconds=elapsed,filter_calls=filter_calls,
  stop_iteration=if(failed)NA_real_ else v(repaired,"stop_iteration"),
  final_particles=if(failed)NA_real_ else v(repaired,"final_particles"),
  kalman=truth$log_likelihood,
  log_likelihood=if(status=="complete")repaired$final$log_likelihood else NA_real_,
  terminal_error=if(status=="complete")repaired$final$log_likelihood-truth$log_likelihood else NA_real_)
write.csv(record,file.path(out,"bridge.csv"),row.names=FALSE)
cat("bridge",data_seed,replication,status,record$terminal_error,"\n")

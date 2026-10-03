# Calibration only: expose unspecified floor and early-controller choices.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]; data_seed <- as.integer(args[6])
fit_maxit <- as.integer(args[11])
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
stopifnot(!dir.exists(output)); dir.create(output,recursive=TRUE)
settings <- data.frame(name=c("baseline","floor1","floor3","after_k"),
  floor_power=c(2,1,3,2),doubling=c(rep("first_full_window",3),"after_k"))
rows <- list()
for(d in c(5,10)) {
  data <- iapf_paper_data(d,100,seed=data_seed+d)
  write.csv(data$observations,file.path(output,paste0("observations-d",d,".csv")),row.names=FALSE)
  model <- iapf_linear_model(data$observations,data$A)
  truth <- iapf_kalman(model)$log_likelihood
  for(i in seq_len(nrow(settings))) {
    setting <- settings[i,]; set.seed(63000000+d)
    start <- proc.time()[[3]]
    answer <- tryCatch(iapf_iterate(model,fit_mode="relative_l2",fit_maxit=fit_maxit,
      floor_tail_power=setting$floor_power,doubling_mode=setting$doubling),error=function(e)e)
    saveRDS(answer,file.path(output,paste0("d",d,"-",setting$name,".rds")))
    success <- !inherits(answer,"error") && answer$status=="complete"
    fits <- if(success)unlist(answer$fits,recursive=FALSE) else list()
    row <- data.frame(dimension=d,setting=setting$name,fit_mode="relative_l2",fit_maxit=fit_maxit,
      floor_power=setting$floor_power,doubling_mode=setting$doubling,
      data_seed=data_seed+d,particle_seed=63000000+d,
      status=if(success)"complete" else if(inherits(answer,"error"))conditionMessage(answer) else answer$status,
      wall_seconds=proc.time()[[3]]-start,ratio=if(success)exp(answer$final$log_likelihood-truth) else NA,
      particles=if(success)answer$final_particles else NA,
      iterations=if(success)length(answer$history) else NA,
      floor_probability_max=if(success)answer$final$floor_probability_max else NA,
      residual_max=if(success)max(vapply(fits,function(x)x$relative_residual,0.0)) else NA,
      max_log_variance=if(success)max(vapply(fits,function(x)x$log_variance_max,0.0)) else NA)
    rows[[length(rows)+1]] <- row
    write.csv(do.call(rbind,rows),file.path(output,"sensitivity.csv"),row.names=FALSE)
    print(row); flush.console()
  }
}
if(any(vapply(rows,function(x)x$status!="complete",TRUE)))quit(status=2)

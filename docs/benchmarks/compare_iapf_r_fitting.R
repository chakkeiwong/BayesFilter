# Independent CPU R full-filter fitting comparison. No production entry point.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]; arm <- args[3]
d <- as.integer(args[4]); repeats <- as.integer(args[5])
first <- as.integer(args[6]); data_seed <- as.integer(args[7])
fit_maxit <- as.integer(args[8])
completed_keys <- if(length(args)>=9)read.csv(args[9],stringsAsFactors=FALSE) else
  data.frame(replication=integer(),method=character())
skip_keys <- paste(completed_keys$replication,completed_keys$method,sep=":")
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_plausible_choices.R"))
source(file.path(root,"docs/benchmarks/diagnose_iapf_r_validation_tails.R"))
stopifnot(!dir.exists(output),Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",
  arm %in% c("qr","box0.5","box1","box2","baselines","controller"),fit_maxit %in% c(200,800))
dir.create(output,recursive=TRUE)
data <- iapf_paper_data(d,100,seed=data_seed)
model <- iapf_linear_model(data$observations,data$A); kalman <- iapf_kalman(model)
write.csv(data$observations,file.path(output,"observations.csv"),row.names=FALSE)
write.csv(data.frame(time=1:100,prefix=kalman$prefix,innovation=kalman$innovation),
  file.path(output,"kalman.csv"),row.names=FALSE)
dput(list(arm=arm,dimension=d,horizon=100,repeats=repeats,first=first,
  data_seed=data_seed,cpu_only=TRUE,noncanonical_reference=TRUE,original_author_data=FALSE,
  N0=1000,k=5,tau=.5,kappa=.5,max_iterations=20,max_particles=16000,
  fit_maxit=fit_maxit,floor_tail_power=8,doubling_mode="after_k",
  stopping_windows=if(arm=="controller")c(qr=6,short_qr=5) else 6,
  completed_keys_from_prior_attempt=skip_keys,
  box_multiplier=if(startsWith(arm,"box"))as.numeric(sub("box","",arm)) else NA,
  total_cost_includes_all_learning=TRUE),file=file.path(output,"settings.R"))
append_csv <- function(x,name) {
  path <- file.path(output,name); exists <- file.exists(path)
  write.table(x,path,append=exists,col.names=!exists,row.names=FALSE,sep=",",qmethod="double")
}
methods <- if(arm=="controller")c("qr","short_qr","bpf","fully_adapted","sis") else
  if(arm=="baselines")c("qr","bpf","fully_adapted","sis") else arm
failed <- FALSE
for(replication in first:(first+repeats-1)) for(method in methods) {
  # Each method resets its seed below. Completing missing pairs therefore does
  # not depend on executing already completed methods or reusing their state.
  if(paste(replication,method,sep=":") %in% skip_keys) next
  index <- if(method %in% c("bpf","fully_adapted","sis"))
    match(method,c("qr","bpf","fully_adapted","sis")) else 1
  seed <- 53000000+d*100000+replication*10+index
  set.seed(seed); started <- proc.time()[[3]]
  algorithm_seconds <- NA_real_
  cat("BEGIN",arm,d,replication,method,"seed",seed,"\n"); flush.console()
  answer <- tryCatch({
    if(method %in% c("qr","short_qr") || startsWith(method,"box")) {
      value <- if(method=="qr")iapf_reconstruction(model,"delayed") else
        if(method=="short_qr")iapf_window_reconstruction(model,5) else
        iapf_constrained_reconstruction(model,as.numeric(sub("box","",method)),fit_maxit)
      algorithm_seconds <- proc.time()[[3]]-started
      if(value$status!="complete") {
        saveRDS(value,file.path(output,paste0(method,"-incomplete-",replication,".rds")))
        stop(paste("incomplete",value$status))
      }
      fit_rows <- list()
      for(iteration in seq_along(value$fits)) for(time in seq_along(value$fits[[iteration]]))
        fit_rows[[length(fit_rows)+1]] <- data.frame(replication=replication,method=method,
          iteration=iteration,time=time,value$fits[[iteration]][[time]])
      append_csv(do.call(rbind,fit_rows),"fits.csv")
      tails <- iapf_validation_tail_rows(model,value$twists,replication)
      tails$method <- method
      append_csv(tails,"tails.csv")
      value$final$clouds <- value$final$log_weights <- value$final$ancestors <- NULL
      saveRDS(value,file.path(output,paste0(method,"-",replication,".rds")))
      list(final=value$final,particles=value$final_particles,iterations=length(value$history),
        tail_pass=all(tails$passed))
    } else if(method=="fully_adapted") {
      final <- iapf_fully_adapted(model,5000)
      algorithm_seconds <- proc.time()[[3]]-started
      list(final=final,particles=5000,iterations=1,tail_pass=TRUE)
    } else {
      final <- iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),
        10000,if(method=="sis")0 else .5,FALSE)
      algorithm_seconds <- proc.time()[[3]]-started
      list(final=final,particles=10000,iterations=1,tail_pass=TRUE)
    }
  },error=function(e) {
    saveRDS(e,file.path(output,paste0(method,"-",replication,"-failure.rds")))
    fields <- c("message","iteration","backward_time","optimizer_code","optimizer_message",
      "relative_bound_margin","evaluation","initial_evaluation")
    dput(e[intersect(names(e),fields)],file=file.path(output,paste0("failure-",method,"-",replication,".R")))
    list(error=conditionMessage(e))
  })
  elapsed <- proc.time()[[3]]-started
  if(!is.finite(algorithm_seconds))algorithm_seconds <- elapsed
  if(!is.null(answer$error)) {
    failed <- TRUE
    append_csv(data.frame(dimension=d,replication=replication,method=method,seed=seed,
      status="error",error=answer$error,log_ratio=NA,ratio=NA,wall_seconds=elapsed,
      algorithm_seconds=algorithm_seconds,diagnostic_io_seconds=elapsed-algorithm_seconds,
      particles=NA,iterations=NA,resampling_count=NA,floor_probability_max=NA,
      tail_pass=FALSE),"replicates.csv")
    cat("ERROR",answer$error,"elapsed",elapsed,"\n"); flush.console()
    next
  }
  log_ratio <- answer$final$log_likelihood-kalman$log_likelihood
  stopifnot(is.finite(log_ratio),is.finite(exp(log_ratio)))
  # Write mandatory diagnostics before the completion row used by resume logic.
  # A timeout may leave uncommitted diagnostics, which are retained but excluded.
  append_csv(data.frame(replication=replication,method=method,time=1:100,
    log_prefix_error=answer$final$prefix-kalman$prefix,
    situation=ifelse(kalman$innovation>qchisq(.9,d),"large_innovation","ordinary")),"prefixes.csv")
  append_csv(data.frame(dimension=d,replication=replication,method=method,seed=seed,
    status="complete",error="",log_ratio=log_ratio,ratio=exp(log_ratio),wall_seconds=elapsed,
    algorithm_seconds=algorithm_seconds,diagnostic_io_seconds=elapsed-algorithm_seconds,
    particles=answer$particles,iterations=answer$iterations,
    resampling_count=answer$final$resampling_count,
    floor_probability_max=if(is.null(answer$final$floor_probability_max))0 else answer$final$floor_probability_max,
    tail_pass=answer$tail_pass),"replicates.csv")
  cat("END",arm,d,replication,method,"elapsed",elapsed,"\n"); flush.console()
}
cat("COMPLETE",arm,d,repeats,"\n")
quit(status=if(failed)2 else 0)

# Independent CPU R reference validation; no production/default implementation.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)>=2,Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(args[2]))
stage <- args[1];out <- args[2];dir.create(out,recursive=TRUE)
source("docs/benchmarks/reference_iapf_paper.R")
source("docs/benchmarks/reference_iapf_author_choices.R")
source("docs/benchmarks/reference_iapf_constrained_diagnostic.R")
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
value <- function(x,key,default=NA_real_)if(is.null(x[[key]]))default else x[[key]]
write_rows <- function(rows,name)if(length(rows))
  write.csv(do.call(rbind,rows),file.path(out,name),row.names=FALSE)

# Measure the actual shared consumer without changing its RNG or numerical path.
measure_learning <- function(model,method,N0=1000,max_particles=4000) {
  original <- get("iapf_apf",envir=.GlobalEnv)
  calls <- list()
  wrapper <- function(model,twists,N,kappa=.5,store_clouds=TRUE) {
    start <- proc.time()[[3]]
    ans <- original(model,twists,N,kappa,store_clouds)
    calls[[length(calls)+1L]] <<- data.frame(call=length(calls)+1L,N=N,
      wall_seconds=proc.time()[[3]]-start,log_likelihood=ans$log_likelihood,
      min_ess=min(ans$ess),resampling_count=ans$resampling_count,
      floor_probability_max=ans$floor_probability_max)
    ans
  }
  assign("iapf_apf",wrapper,envir=.GlobalEnv)
  on.exit(assign("iapf_apf",original,envir=.GlobalEnv))
  start <- proc.time()[[3]]
  ans <- tryCatch(if(method=="qr")iapf_choice_run(model,arm="qr",floor_rule="tail8",
      sd_mode="sample",N0=N0,max_iterations=12,max_particles=max_particles)
    else iapf_constrained_diagnostic_run(model,N0=N0,max_iterations=12,
      max_particles=max_particles,sd_mode="sample"),error=function(e)e)
  elapsed <- proc.time()[[3]]-start
  list(answer=ans,wall_seconds=elapsed,calls=calls)
}

if(stage=="checks") {
  checks <- list()
  check <- function(name,ok) {
    checks[[length(checks)+1L]] <<- data.frame(check=name,pass=isTRUE(ok))
    write_rows(checks,"checks.csv");stopifnot(isTRUE(ok))
  }
  dat <- iapf_paper_data(2,4,seed=99900001)
  model <- iapf_linear_model(dat$observations,dat$A)
  for(method in c("qr","ridge_bounded1")) {
    set.seed(99900002)
    raw <- if(method=="qr")iapf_choice_run(model,"qr",N0=128,
      max_iterations=12,max_particles=512) else
      iapf_constrained_diagnostic_run(model,N0=128,max_particles=512)
    rng <- .Random.seed
    set.seed(99900002);timed <- measure_learning(model,method,128,512)
    check(paste0(method,"_complete"),raw$status=="complete" && timed$answer$status=="complete")
    check(paste0(method,"_values_unchanged"),identical(raw,timed$answer))
    check(paste0(method,"_RNG_unchanged"),identical(rng,.Random.seed))
    check(paste0(method,"_actual_final_call_measured"),
      length(timed$calls)==length(raw$history)+1 &&
      identical(tail(timed$calls,1)[[1]]$log_likelihood,raw$final$log_likelihood))
    check(paste0(method,"_timing_nonnegative"),
      timed$wall_seconds>=tail(timed$calls,1)[[1]]$wall_seconds)
  }
  truth <- iapf_kalman(model)
  for(N in c(250,1000,4000)) {
    set.seed(99900003);a <- iapf_apf(model,iapf_exact_twists(model),N,store_clouds=FALSE)
    check(paste0("exact_oracle_N",N),abs(a$log_likelihood-truth$log_likelihood)<=1e-7)
  }
  cat("Independent validation checks passed:",length(checks),"\n")
} else if(stage=="pair") {
  stopifnot(length(args)==5)
  d <- as.integer(args[3]);j <- as.integer(args[4]);r <- as.integer(args[5])
  stopifnot(d %in% c(5,10,20,40,80),j %in% 1:2,r %in% 1:8)
  data_seed <- 92800000L+10000L*j+d
  learning_seed <- 92840000L+10000L*j+101L*r+d
  dat <- iapf_paper_data(d,100,seed=data_seed)
  model <- iapf_linear_model(dat$observations,dat$A);truth <- iapf_kalman(model)
  saveRDS(list(data=dat,model=model,kalman=truth),file.path(out,"data.rds"))
  learning_rows <- probe_rows <- fit_rows <- call_rows <- list()
  methods <- c("qr","ridge_bounded1")
  if((j+r)%%2==1)methods <- rev(methods)
  for(method in methods) {
    set.seed(learning_seed);run <- measure_learning(model,method)
    ans <- run$answer;failed <- inherits(ans,"error")
    status <- if(failed)"candidate_failed" else ans$status
    complete <- identical(status,"complete")
    delta <- if(complete)ans$final$log_likelihood-truth$log_likelihood else NA_real_
    if(complete && !all(is.finite(c(delta,ans$final$prefix)))) {
      status <- "nonfinite";complete <- FALSE
    }
    final_time <- if(complete)tail(run$calls,1)[[1]]$wall_seconds else NA_real_
    train_time <- if(complete)max(0,run$wall_seconds-final_time) else NA_real_
    learning_rows[[length(learning_rows)+1L]] <- data.frame(dimension=d,data_index=j,
      data_seed=data_seed,replication=r,filter_seed=learning_seed,method=method,
      status=status,message=if(failed)conditionMessage(ans) else "",
      wall_seconds=run$wall_seconds,training_seconds=train_time,
      original_final_seconds=final_time,stop_iteration=value(ans,"stop_iteration"),
      final_particles=value(ans,"final_particles"),failure_iteration=value(ans,"iteration"),
      failure_time=value(ans,"backward_time"),terminal_error=delta,
      prefix_error_max=if(complete)max(abs(ans$final$prefix-truth$prefix)) else NA_real_)
    write_rows(learning_rows,"learning.csv")
    for(call in run$calls)call_rows[[length(call_rows)+1L]] <- cbind(method=method,call)
    write_rows(call_rows,"internal-filter-timing.csv")
    if(!failed && length(ans$fits))for(ii in seq_along(ans$fits))
      for(tt in seq_along(ans$fits[[ii]])) {
        z <- ans$fits[[ii]][[tt]]
        fit_rows[[length(fit_rows)+1L]] <- data.frame(method=method,iteration=ii-1,time=tt,
          kkt=value(z,"kkt_residual"),lambda=value(z,"lambda"),
          active_constraints=value(z,"active_constraints"),
          active_set_iterations=value(z,"active_set_iterations",0),
          condition=value(z,"weighted_design_condition",value(z,"design_condition")),
          weight_ess=value(z,"weight_ess"))
      }
    write_rows(fit_rows,"fit-diagnostics.csv")
    if(complete) {
      for(h in 1:3) {
        N <- c(250,1000,4000)[h]
        filter_seed <- 92900000L+100000L*h+10000L*j+1000L*r+d
        set.seed(filter_seed);start <- proc.time()[[3]]
        probe <- tryCatch(iapf_apf(model,ans$twists,N,store_clouds=FALSE),error=function(e)e)
        seconds <- proc.time()[[3]]-start
        ok <- !inherits(probe,"error") && all(is.finite(c(probe$log_likelihood,probe$prefix)))
        error <- if(ok)probe$log_likelihood-truth$log_likelihood else NA_real_
        probe_rows[[length(probe_rows)+1L]] <- data.frame(dimension=d,data_index=j,
          data_seed=data_seed,replication=r,filter_seed=filter_seed,method=method,N=N,
          status=if(ok)"complete" else "candidate_failed",terminal_error=error,
          likelihood_ratio=if(ok)exp(error) else NA_real_,
          relative_error=if(ok)expm1(error) else NA_real_,
          filter_seconds=seconds,guide_seconds=train_time,total_seconds=train_time+seconds,
          min_ess=if(ok)min(probe$ess) else NA_real_,
          prefix_error_max=if(ok)max(abs(probe$prefix-truth$prefix)) else NA_real_)
        write_rows(probe_rows,"probes.csv")
        saveRDS(probe,file.path(out,paste0(method,"-N",N,"-probe.rds")))
      }
      # Preserve every fit diagnostic and guide; omit large final particle arrays.
      ans$final <- ans$final[c("log_likelihood","prefix","ess","resampled",
        "resampling_count","floor_probability_max","floor_probability_quantiles")]
    }
    saveRDS(ans,file.path(out,paste0(method,"-learning.rds")))
    cat("learner",d,j,r,method,status,"seconds",run$wall_seconds,"\n")
  }
  start <- proc.time()[[3]];full <- iapf_exact_twists(model)
  full_seconds <- proc.time()[[3]]-start
  for(method in c("constant","observation","moment_diagonal","precision_diagonal","exact_full")) {
    start <- proc.time()[[3]]
    guide <- switch(method,constant=replicate(100,iapf_constant_twist(),simplify=FALSE),
      observation=iapf_observation_twists(model),
      moment_diagonal=lapply(full,function(g)iapf_gaussian_twist(g$mean,diag(diag(g$covariance)))),
      precision_diagonal=lapply(full,function(g)iapf_gaussian_twist(g$mean,diag(1/diag(solve(g$covariance))))),
      exact_full=full)
    guide_seconds <- proc.time()[[3]]-start+
      if(method %in% c("moment_diagonal","precision_diagonal","exact_full"))full_seconds else 0
    filter_seed <- 92900000L+200000L+10000L*j+1000L*r+d
    set.seed(filter_seed);start <- proc.time()[[3]]
    probe <- iapf_apf(model,guide,1000,store_clouds=FALSE)
    seconds <- proc.time()[[3]]-start;error <- probe$log_likelihood-truth$log_likelihood
    stopifnot(all(is.finite(c(error,probe$prefix))))
    if(method=="exact_full")stopifnot(abs(error)<=1e-7)
    probe_rows[[length(probe_rows)+1L]] <- data.frame(dimension=d,data_index=j,
      data_seed=data_seed,replication=r,filter_seed=filter_seed,method=method,N=1000,
      status="complete",terminal_error=error,likelihood_ratio=exp(error),relative_error=expm1(error),
      filter_seconds=seconds,guide_seconds=guide_seconds,total_seconds=guide_seconds+seconds,
      min_ess=min(probe$ess),prefix_error_max=max(abs(probe$prefix-truth$prefix)))
    write_rows(probe_rows,"probes.csv")
    saveRDS(probe,file.path(out,paste0(method,"-probe.rds")))
  }
  cat("paired cell complete",d,j,r,"\n")
} else stop("unknown stage")

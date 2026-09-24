# Bounded independent CPU R experiment worker. Uses shared filter/controller.
a <- commandArgs(trailingOnly=TRUE)
root <- a[1]; output <- a[2]; mode <- a[3]; d <- as.integer(a[4])
data_seed <- as.integer(a[5]); arm <- a[6]; floor_rule <- a[7]; sd_mode <- a[8]
repeats <- as.integer(a[9]); first <- as.integer(a[10])
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(output))
dir.create(output,recursive=TRUE)
for(f in c("reference_iapf_paper.R","reference_iapf_author_choices.R",
  "diagnose_iapf_r_validation_tails.R"))source(file.path(root,"docs/benchmarks",f))
append_rows <- function(x,name) {
  p <- file.path(output,name); exists <- file.exists(p)
  write.table(x,p,sep=",",append=exists,col.names=!exists,row.names=FALSE,qmethod="double")
}
dat <- iapf_paper_data(d,100,seed=data_seed)
model <- iapf_linear_model(dat$observations,dat$A); kalman <- iapf_kalman(model)
write.csv(dat$observations,file.path(output,"observations.csv"),row.names=FALSE)
write.csv(data.frame(time=1:100,prefix=kalman$prefix,innovation=kalman$innovation),
  file.path(output,"kalman.csv"),row.names=FALSE)
dput(list(mode=mode,dimension=d,data_seed=data_seed,arm=arm,floor_rule=floor_rule,
  sd_mode=sd_mode,repeats=repeats,first=first,cpu_only=TRUE,CUDA_VISIBLE_DEVICES="-1",
  method_status="independent_R_hypothesis_unknown_author_choices",N0=1000,k=5,
  tau=.5,kappa=.5,stopping_window=6,max_iterations=20,max_particles=16000),
  file=file.path(output,"settings.R"))
val <- function(x,key,default=NA)if(is.null(x[[key]]))default else x[[key]]

if(mode=="fixtures") {
  exact <- iapf_exact_twists(model)
  set.seed(data_seed+10001); training <- iapf_apf(model,exact,1000)
  set.seed(data_seed+20001); smoothing <- iapf_apf(model,exact,2000)
  set.seed(data_seed+30001); filtering <- iapf_apf(model,
    replicate(100,iapf_constant_twist(),simplify=FALSE),2000)
  for(time in c(100,99,50)) {
    x <- training$clouds[[time]]
    log_target <- iapf_backward_log_target(model,x,time,if(time<100)exact[[time+1]] else NULL)
    prior_means <- if(time==1)matrix(rep(model$initial_mean,each=2000),2000,d) else {
      ancestors <- sample.int(2000,2000,replace=TRUE,
        prob=exp(filtering$log_weights[[time-1]]-max(filtering$log_weights[[time-1]])))
      model$transition_mean(filtering$clouds[[time-1]][ancestors,,drop=FALSE],time)
    }
    predictive <- iapf_draw_proposal(iapf_proposal(prior_means,
      if(time==1)model$initial_covariance else model$transition_covariance,iapf_constant_twist()))
    saveRDS(list(points=x,log_targets=log_target,exact=exact[[time]],
      predictive=predictive,smoothing=smoothing$clouds[[time]]),
      file.path(output,paste0("fixture-",time,".rds")))
    for(method in c("qr","f1_strict","f1_loose","f2_strict","f2_loose","wlog1","wlog2")) {
      started <- proc.time()[[3]]
      fit <- tryCatch(iapf_choice_fit(x,log_target,arm=method),error=function(e)e)
      failed <- inherits(fit,"error"); dg <- fit$diagnostics
      saveRDS(fit,file.path(output,paste0("fit-",time,"-",method,".rds")))
      append_rows(data.frame(dimension=d,data_seed=data_seed,time=time,method=method,
        status=if(failed)"candidate_failed" else "complete",
        message=if(failed)conditionMessage(fit) else val(dg,"optimizer_message",""),
        convergence=val(dg,"convergence"),relative_residual=val(dg,"relative_residual"),
        gradient_max=val(dg,"gradient_max"),weight_ess=val(dg,"weight_ess"),
        log_density_shift=val(dg,"log_density_shift"),seconds=proc.time()[[3]]-started),"fixtures.csv")
      if(!failed)for(distribution in c("predictive","smoothing")) {
        points <- if(distribution=="predictive")predictive else smoothing$clouds[[time]]
        target <- iapf_backward_log_target(model,points,time,if(time<100)exact[[time+1]] else NULL)
        density <- iapf_log_twist(points,fit$twist)
        p <- exp(density-max(density)); b <- exp(target-max(target))
        lambda <- sum(p*b)/sum(b^2); logs <- density-target
        append_rows(data.frame(dimension=d,data_seed=data_seed,time=time,method=method,
          distribution=distribution,relative_density_residual=sum((p-lambda*b)^2)/sum(p^2),
          centered_log_rmse=sqrt(mean((logs-mean(logs))^2))),"heldout.csv")
      }
    }
  }
} else {
  stopifnot(mode=="run")
  methods <- if(arm=="baselines")c("qr","bpf","fully_adapted","sis") else arm
  for(replication in first+seq_len(repeats)-1L)for(method in methods) {
    seed_index <- if(method %in% c("bpf","fully_adapted","sis"))
      match(method,c("qr","bpf","fully_adapted","sis")) else 1L
    seed <- as.integer(data_seed+replication*1009+seed_index*100003)
    set.seed(seed); started <- proc.time()[[3]]
    cat("BEGIN",d,data_seed,method,floor_rule,sd_mode,replication,"\n");flush.console()
    answer <- tryCatch({
      if(method %in% c("bpf","fully_adapted","sis")) {
        final <- if(method=="fully_adapted")iapf_fully_adapted(model,5000) else
          iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),10000,
            if(method=="sis")0 else .5,FALSE)
        list(status="complete",final=final,final_particles=if(method=="fully_adapted")5000 else 10000,
          history=final$log_likelihood)
      } else iapf_choice_run(model,method,floor_rule,sd_mode)
    },error=function(e)e)
    algorithm_seconds <- proc.time()[[3]]-started
    failed <- inherits(answer,"error") || answer$status!="complete"
    error <- if(inherits(answer,"error"))conditionMessage(answer) else
      if(failed)answer$status else ""
    tail_pass <- FALSE; log_ratio <- ratio <- particles <- iterations <- resampling <- max_floor <- NA_real_
    if(!failed) {
      log_ratio <- answer$final$log_likelihood-kalman$log_likelihood; ratio <- exp(log_ratio)
      if(!is.finite(ratio) || !is.finite(log_ratio)) {failed <- TRUE;error <- "nonfinite terminal ratio"}
    }
    if(!failed) {
      particles <- answer$final_particles; iterations <- length(answer$history)
      resampling <- answer$final$resampling_count
      max_floor <- answer$final$floor_probability_max
      if(is.null(max_floor))max_floor <- 0
      tail_pass <- TRUE
      if(!is.null(answer$twists)) {
        tails <- iapf_validation_tail_rows(model,answer$twists,replication)
        tails$method <- method; append_rows(tails,"tails.csv"); tail_pass <- all(tails$passed)
        for(i in seq_along(answer$fits)) for(t in seq_along(answer$fits[[i]])) {
          dg <- answer$fits[[i]][[t]]
          append_rows(data.frame(replication=replication,method=method,iteration=i,time=t,
            start_source=dg$start_source,relative_residual=dg$relative_residual,
            gradient_max=dg$gradient_max,log_density_shift=dg$log_density_shift,
            weight_ess=dg$weight_ess,convergence=dg$convergence),"fits.csv")
        }
        for(i in seq_along(answer$history))if(i>=7) {
          h <- answer$history[(i-5):i]
          append_rows(data.frame(replication=replication,method=method,iteration=i-1,
            cv_sample=iapf_likelihood_cv(h),cv_population=iapf_likelihood_cv(h,"population"),
            particles=answer$counts[i]),"history.csv")
        }
      }
      append_rows(data.frame(replication=replication,method=method,time=1:100,
        log_prefix_error=answer$final$prefix-kalman$prefix,
        situation=ifelse(kalman$innovation>qchisq(.9,d),"large_innovation","ordinary")),"prefixes.csv")
      if(!is.null(answer$final$floor_probability_quantiles))append_rows(data.frame(
        replication=replication,method=method,time=1:100,
        answer$final$floor_probability_quantiles),"floor-probabilities.csv")
      answer$final$clouds <- answer$final$log_weights <- answer$final$ancestors <- NULL
    }
    saveRDS(answer,file.path(output,paste0(method,"-",replication,if(failed)"-failure" else "",".rds")))
    total <- proc.time()[[3]]-started
    append_rows(data.frame(dimension=d,data_seed=data_seed,replication=replication,
      method=method,floor_rule=if(method %in% c("bpf","fully_adapted","sis"))"none" else floor_rule,
      sd_mode=sd_mode,seed=seed,status=if(failed)"candidate_failed" else "complete",message=error,
      log_ratio=log_ratio,ratio=ratio,particles=particles,iterations=iterations,resampling_count=resampling,
      floor_probability_max=max_floor,tail_pass=tail_pass,algorithm_seconds=algorithm_seconds,
      diagnostic_io_seconds=total-algorithm_seconds,wall_seconds=total,
      failure_iteration=val(answer,"iteration"),failure_time=val(answer,"backward_time")),"replicates.csv")
    cat(if(failed)"FAILED" else "COMPLETE",method,replication,error,"seconds",total,"\n");flush.console()
  }
}
cat("WORKER COMPLETE",mode,d,data_seed,arm,"\n")

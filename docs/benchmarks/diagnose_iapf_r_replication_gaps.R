# Independent CPU diagnostics; no production or original-author implementation claim.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]; mode <- args[3]
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_plausible_choices.R"))
source(file.path(root,"docs/benchmarks/diagnose_iapf_r_validation_tails.R"))
stopifnot(!dir.exists(output),Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",
  mode %in% c("fit","fixed","pilot","pre"))
dir.create(output,recursive=TRUE)
append_csv <- function(x,name) {
  path <- file.path(output,name); exists <- file.exists(path)
  write.table(x,path,append=exists,col.names=!exists,row.names=FALSE,sep=",",qmethod="double")
}
settings <- list(mode=mode,cpu_only=TRUE,gpu_intentionally_hidden=TRUE,
  independent_noncanonical_reference=TRUE,R=R.version.string,T=100,alpha=.42)
dput(settings,file.path(output,"settings.R"))

state_distributions <- function(model) {
  T <- model$horizon; d <- model$dimension
  predicted_mean <- filtered_mean <- matrix(0,T,d)
  predicted_covariance <- filtered_covariance <- vector("list",T)
  m <- model$initial_mean; P <- model$initial_covariance
  for(t in seq_len(T)) {
    if(t>1) {m <- as.vector(model$A %*% m); P <- model$A %*% P %*% t(model$A)+model$transition_covariance}
    predicted_mean[t,] <- m; predicted_covariance[[t]] <- P
    S <- model$C %*% P %*% t(model$C)+model$R
    K <- P %*% t(model$C) %*% chol2inv(iapf_chol(S))
    m <- m+as.vector(K %*% (model$observations[t,]-model$C %*% m))
    D <- diag(d)-K %*% model$C
    P <- D %*% P %*% t(D)+K %*% model$R %*% t(K)
    filtered_mean[t,] <- m; filtered_covariance[[t]] <- P
  }
  smooth_mean <- filtered_mean; smooth_covariance <- filtered_covariance
  for(t in rev(seq_len(T-1))) {
    J <- filtered_covariance[[t]] %*% t(model$A) %*%
      chol2inv(iapf_chol(predicted_covariance[[t+1]]))
    smooth_mean[t,] <- filtered_mean[t,]+J %*% (smooth_mean[t+1,]-predicted_mean[t+1,])
    P <- filtered_covariance[[t]]+J %*%
      (smooth_covariance[[t+1]]-predicted_covariance[[t+1]]) %*% t(J)
    smooth_covariance[[t]] <- (P+t(P))/2
    iapf_chol(smooth_covariance[[t]])
  }
  list(predictive=list(mean=predicted_mean,covariance=predicted_covariance),
    smoothing=list(mean=smooth_mean,covariance=smooth_covariance))
}

if(mode=="fit") {
  for(d in c(10,20)) {
    data_seed <- 87010000+d; seed <- 87200000+d
    data <- iapf_paper_data(d,100,seed=data_seed)
    model <- iapf_linear_model(data$observations,data$A)
    set.seed(seed)
    training <- iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),1000)
    next_fit <- iapf_fit_gaussian(training$clouds[[100]],
      model$log_observation(training$clouds[[100]],model$observations[100,],100),
      1000,floor_tail_power=8,fit_mode="log_quadratic")
    points <- training$clouds[[99]]
    target <- iapf_backward_log_target(model,points,99,next_fit$twist)
    qr <- iapf_fit_gaussian(points,target,1000,floor_tail_power=8,fit_mode="log_quadratic")
    distributions <- state_distributions(model)
    # Independent RTS-vs-Gaussian-conditioning identity for the actual heldout law.
    oracle <- iapf_proposal(matrix(distributions$predictive$mean[99,],1),
      distributions$predictive$covariance[[99]],iapf_exact_twists(model)[[99]])
    stopifnot(max(abs(oracle$means-distributions$smoothing$mean[99,]))<1e-10,
      max(abs(oracle$covariance-distributions$smoothing$covariance[[99]]))<1e-10)
    set.seed(87300000+d)
    heldout <- lapply(distributions,function(s)iapf_draw_normal(
      matrix(rep(s$mean[99,],each=2000),2000,d),s$covariance[[99]]))
    heldout$training <- points
    answers <- list()
    for(arm in c("qr","box_0.5","box_1","box_2")) {
      start <- proc.time()[[3]]
      fit <- tryCatch(if(arm=="qr")qr else iapf_local_eq15_fit(points,target,
        floor_tail_power=8,box_multiplier=as.numeric(sub("box_","",arm)),
        allow_active_bounds=TRUE),iapf_fit_error=identity)
      elapsed <- proc.time()[[3]]-start
      answers[[arm]] <- fit
      if(inherits(fit,"iapf_fit_error")) {
        append_csv(data.frame(dimension=d,arm=arm,status="optimizer_failure",
          reason=conditionMessage(fit),seconds=elapsed),"fit-status.csv")
        next
      }
      append_csv(data.frame(dimension=d,arm=arm,status="converged",reason="",seconds=elapsed),"fit-status.csv")
      fixed_log_scale <- max(iapf_log_normal(points,qr$twist$mean,qr$twist$covariance))
      paper_objective <- iapf_fit_objective(c(fit$twist$mean,log(diag(fit$twist$covariance))),
        points,target,fixed_log_scale)$value
      initial_objective <- iapf_fit_objective(c(qr$twist$mean,log(diag(qr$twist$covariance))),
        points,target,fixed_log_scale)$value
      for(situation in names(heldout)) {
        x <- heldout[[situation]]
        y <- iapf_backward_log_target(model,x,99,next_fit$twist)
        p <- iapf_log_normal(x,fit$twist$mean,fit$twist$covariance)
        a <- exp(p-max(p)); b <- exp(y-max(y)); lambda <- sum(a*b)/sum(b*b)
        log_residual <- y-p
        append_csv(data.frame(dimension=d,data_seed=data_seed,training_seed=seed,
          heldout_seed=87300000+d,arm=arm,situation=situation,
          target_effective_points=sum(b)^2/sum(b*b),
          density_loss_effective_points=sum(b*b)^2/sum(b^4),
          paper_objective=paper_objective,initial_paper_objective=initial_objective,
          relative_density_residual=sum((a-lambda*b)^2)/sum(a*a),
          centered_log_rmse=sqrt(mean((log_residual-mean(log_residual))^2)),
          mean_shift_qr_sd=sqrt(sum((fit$twist$mean-qr$twist$mean)^2/diag(qr$twist$covariance))),
          variance_ratio_min=min(diag(fit$twist$covariance)/diag(qr$twist$covariance)),
          variance_ratio_max=max(diag(fit$twist$covariance)/diag(qr$twist$covariance)),
          active_bound_coordinates=if(arm=="qr")0 else fit$diagnostics$active_bound_coordinates,
          projected_gradient_max=if(arm=="qr")NA_real_ else fit$diagnostics$projected_gradient_max),
          "fit-sensitivity.csv")
      }
    }
    saveRDS(list(model=model,points=points,target=target,next_twist=next_fit$twist,
      fits=answers,heldout=heldout,distributions=distributions),file.path(output,paste0("fit-d",d,".rds")))
  }
} else if(mode=="fixed") {
  saved <- file.path(root,"docs/plans/artifacts/iapf-r-plausible-reconstruction-20260921-01/attempt08-validation-delayed-d80/results")
  data <- iapf_paper_data(80,100,seed=86000080)
  stopifnot(max(abs(data$observations-as.matrix(read.csv(file.path(saved,"observations.csv")))))<1e-12)
  model <- iapf_linear_model(data$observations,data$A); kalman <- iapf_kalman(model)
  for(id in 1401:1404) {
    saved_run <- readRDS(file.path(saved,paste0("iapf-",id,".rds")))
    stopifnot(saved_run$status=="complete")
    twists <- saved_run$twists
    append_csv(iapf_validation_tail_rows(model,twists,id),"tails.csv")
    for(N in c(1000,2000)) for(replication in 1:16) {
      seed <- 87600000+(id-1400)*100+replication
      set.seed(seed); start <- proc.time()[[3]]
      run <- iapf_apf(model,twists,N,.5,FALSE)
      append_csv(data.frame(guide=id,arm="frozen_qr",N=N,replication=replication,seed=seed,
        log_ratio=run$log_likelihood-kalman$log_likelihood,
        ratio=exp(run$log_likelihood-kalman$log_likelihood),
        resampling_count=run$resampling_count,floor_probability_max=run$floor_probability_max,
        seconds=proc.time()[[3]]-start),"fixed-guides.csv")
    }
    cat("COMPLETE fixed guide",id,"\n"); flush.console()
  }
  for(arm in c("exact_future","observation_only")) for(replication in 1:4) {
    N <- if(arm=="exact_future")32 else 1000
    twists <- if(arm=="exact_future")iapf_exact_twists(model) else iapf_observation_twists(model)
    seed <- 87700000+replication
    set.seed(seed); start <- proc.time()[[3]]
    run <- iapf_apf(model,twists,N,.5,FALSE)
    if(arm=="exact_future")stopifnot(abs(run$log_likelihood-kalman$log_likelihood)<1e-8)
    append_csv(data.frame(guide=0,arm=arm,N=N,replication=replication,seed=seed,
      log_ratio=run$log_likelihood-kalman$log_likelihood,
      ratio=exp(run$log_likelihood-kalman$log_likelihood),resampling_count=run$resampling_count,
      floor_probability_max=run$floor_probability_max,seconds=proc.time()[[3]]-start),"fixed-guides.csv")
  }
} else if(mode=="pre") {
  saved <- file.path(root,"docs/plans/artifacts/iapf-r-plausible-reconstruction-20260921-01/attempt08-validation-delayed-d80/results")
  data <- iapf_paper_data(80,100,seed=86000080)
  stopifnot(max(abs(data$observations-as.matrix(read.csv(file.path(saved,"observations.csv")))))<1e-12)
  model <- iapf_linear_model(data$observations,data$A); kalman <- iapf_kalman(model)
  for(id in 1401:1404) {
    original <- readRDS(file.path(saved,paste0("iapf-",id,".rds")))
    calls <- 0L; retained <- NULL
    capture <- function(model,twists,N,kappa,store_clouds) {
      calls <<- calls+1L
      if(calls==7L) {stopifnot(N==1000); retained <<- twists}
      iapf_apf(model,twists,N,kappa,store_clouds)
    }
    seed <- 53000000+80*100000+id*10+1
    set.seed(seed)
    reproduced <- do.call(iapf_iterate,c(list(model=model,.filter=capture),
      iapf_reconstruction_settings("delayed")))
    stopifnot(reproduced$status=="complete",!is.null(retained),
      identical(reproduced$counts,original$counts),
      length(reproduced$history)==length(original$history))
    history_error <- max(abs(reproduced$history-original$history))
    final_error <- abs(reproduced$final$log_likelihood-original$final$log_likelihood)
    stopifnot(history_error<1e-10,final_error<1e-10)
    append_csv(data.frame(guide=id,learning_seed=seed,history_error=history_error,
      final_error=final_error,pre_doubling_N=1000,
      six_estimate_cv=iapf_likelihood_cv(reproduced$history[2:7]),
      last_five_cv=iapf_likelihood_cv(reproduced$history[3:7]),
      final_N=reproduced$final_particles),"history-parity.csv")
    saveRDS(list(twists=retained,history=reproduced$history,counts=reproduced$counts,
      seed=seed,iteration=6),file.path(output,paste0("pre-guide-",id,".rds")))
    append_csv(iapf_validation_tail_rows(model,retained,id),"tails.csv")
    for(replication in 1:16) {
      seed <- 87800000+(id-1400)*100+replication
      set.seed(seed); start <- proc.time()[[3]]
      run <- iapf_apf(model,retained,1000,.5,FALSE)
      append_csv(data.frame(guide=id,arm="pre_doubling_qr",N=1000,replication=replication,seed=seed,
        log_ratio=run$log_likelihood-kalman$log_likelihood,
        ratio=exp(run$log_likelihood-kalman$log_likelihood),
        resampling_count=run$resampling_count,floor_probability_max=run$floor_probability_max,
        seconds=proc.time()[[3]]-start),"fixed-guides.csv")
    }
    cat("COMPLETE pre-doubling guide",id,"\n"); flush.console()
  }
} else {
  data <- iapf_paper_data(5,100,seed=87100005)
  model <- iapf_linear_model(data$observations,data$A); kalman <- iapf_kalman(model)
  set.seed(87400005)
  run <- tryCatch(iapf_constrained_reconstruction(model),iapf_fit_error=identity)
  saveRDS(run,file.path(output,"constrained-pilot.rds"))
  if(inherits(run,"iapf_fit_error")) {
    append_csv(data.frame(status="optimizer_failure",reason=conditionMessage(run),
      iteration=run$iteration,time=run$backward_time),"pilot-status.csv")
  } else if(run$status!="complete") {
    append_csv(data.frame(status=run$status,reason="resource cap",iteration=length(run$history),time=NA),"pilot-status.csv")
  } else {
    append_csv(data.frame(status="complete",reason="",iteration=run$stop_iteration,time=NA),"pilot-status.csv")
    append_csv(data.frame(ratio=exp(run$final$log_likelihood-kalman$log_likelihood),
      N=run$final_particles,resampling_count=run$final$resampling_count,
      bound_fits=sum(vapply(unlist(run$fits,recursive=FALSE),function(f)f$boundary,logical(1)))),"pilot-result.csv")
    append_csv(iapf_validation_tail_rows(model,run$twists,1),"tails.csv")
  }
}
cat("COMPLETE diagnostic",mode,"\n")

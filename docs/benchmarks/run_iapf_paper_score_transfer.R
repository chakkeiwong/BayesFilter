# Independent-reference campaign, paper model/timing but reconstructed fitting.
args <- commandArgs(trailingOnly=TRUE);mode <- args[1];out <- args[2]
source('docs/benchmarks/reference_iapf_paper.R')
source('docs/benchmarks/reference_iapf_author_choices.R')
source('docs/benchmarks/diagnostic_iapf_paper_score.R')
options(digits=17)
mat <- function(x) do.call(rbind,lapply(x,unlist))
write_rows <- function(rows,name) write.csv(do.call(rbind,rows),file.path(out,name),row.names=FALSE)

if (mode=='preflight') {
  source(args[3]);rows <- list()
  for (entry in input) {
    model <- iapf_linear_model(mat(entry$y),mat(entry$model[[1]]),unlist(entry$model[[3]]),
      mat(entry$model[[4]]),mat(entry$model[[5]]),mat(entry$model[[2]]),mat(entry$model[[6]]))
    clouds <- lapply(entry$clouds,mat)
    for (name in names(entry$coefficients)) {
      pieces <- strsplit(name,'_')[[1]]
      fit <- iapf_score_backward(model,clouds,nrow(clouds[[1]]),pieces[1],pieces[2])
      coeff <- entry$coefficients[[name]];cs <- mat(coeff[[1]]);vs <- lapply(coeff[[2]],mat);fs <- unlist(coeff[[3]])
      error <- if(fit$valid) max(vapply(seq_len(model$horizon),function(t)
        max(abs(fit$twists[[t]]$mean-cs[t,]),abs(fit$twists[[t]]$covariance-vs[[t]]),
            abs(fit$twists[[t]]$log_floor-fs[t])),numeric(1))) else 0
      stopifnot(fit$valid==entry$accepted[[name]],error<=1e-8)
      rows[[length(rows)+1]] <- data.frame(case=entry$case,method=name,error=error)
    }
  }
  # Independent finite differences and rejection controls on a correlated model.
  d <- 5;set.seed(95701);x <- matrix(rnorm(30*d),30,d)
  dat <- iapf_paper_data(d,3,seed=95702);model <- iapf_linear_model(dat$observations,dat$A)
  next_twist <- iapf_gaussian_twist(seq(-.2,.3,length.out=d),diag(d)+matrix(.04,d,d),-5)
  h <- iapf_score_target(model,x,1,next_twist);fd <- h*0;epsilon <- 1e-5
  for (j in seq_len(d)) {
    plus <- minus <- x;plus[,j] <- plus[,j]+epsilon;minus[,j] <- minus[,j]-epsilon
    fd[,j] <- (iapf_backward_log_target(model,plus,1,next_twist)-
      iapf_backward_log_target(model,minus,1,next_twist))/(2*epsilon)
  }
  stopifnot(max(abs(h-fd))<=1e-6,!iapf_score_fit_cloud(x,x)$valid,
            !iapf_score_fit_cloud(matrix(1,30,d),-matrix(1,30,d))$valid)
  write_rows(rows,'checks.csv')
  write.csv(data.frame(fd_error=max(abs(h-fd)),rank_rejection=TRUE,precision_rejection=TRUE),
            file.path(out,'controls.csv'),row.names=FALSE)
  cat('Preflight passed:',length(rows),'parity fits\n')
} else if (mode=='case') {
  d <- as.integer(args[3]);j <- as.integer(args[4]);N <- 1000;T <- 100
  seeds <- list(data=951000+10*d+j,pilot=952000+10*d+j,final=953000+100*d+10*j+seq_len(4))
  dat <- iapf_paper_data(d,T,seed=seeds$data);model <- iapf_linear_model(dat$observations,dat$A)
  saveRDS(list(data=dat,seeds=seeds,N=N,T=T,model_matrices=list(A=model$A,H=model$C,Q=model$transition_covariance,
    R=model$R,m1=model$initial_mean,P1=model$initial_covariance)),file.path(out,'inputs.rds'))
  constant <- replicate(T,iapf_constant_twist(),simplify=FALSE)
  exact <- iapf_exact_twists(model);kalman <- iapf_kalman(model)$log_likelihood
  set.seed(seeds$pilot);pilot <- iapf_apf(model,constant,N,.5,TRUE)
  saveRDS(pilot,file.path(out,'pilot.rds'))
  fits <- list();fit_rows <- list();started <- proc.time()[3]
  limit <- iapf_score_backward(model,pilot$clouds,N,'full','negligible')
  stopifnot(limit$valid)
  oracle_error <- max(vapply(seq_len(T),function(t) max(abs(limit$twists[[t]]$mean-exact[[t]]$mean),
      abs(limit$twists[[t]]$covariance-exact[[t]]$covariance)),numeric(1)))
  stopifnot(oracle_error<=1e-8)
  fits$full_negligible <- limit
  for (covariance_mode in c('diagonal','full')) for (floor_mode in c('positive','tail8')) {
    name <- paste(covariance_mode,floor_mode,sep='_');start <- proc.time()[3]
    fit <- iapf_score_backward(model,pilot$clouds,N,covariance_mode,floor_mode);fits[[name]] <- fit
    fit_rows[[length(fit_rows)+1]] <- data.frame(method=name,valid=fit$valid,seconds=proc.time()[3]-start,
      reason=if(fit$valid)'' else fit$reason,backward_time=if(fit$valid)NA_integer_ else fit$backward_time)
  }
  start <- proc.time()[3]
  qr <- tryCatch(iapf_choice_backward(model,pilot$clouds,N,'qr',floor_rule='tail8'),iapf_fit_error=function(e)e)
  qr_valid <- !inherits(qr,'iapf_fit_error');fits$diagonal_QR_tail8 <- if(qr_valid)c(list(valid=TRUE),qr) else
    list(valid=FALSE,reason=conditionMessage(qr),backward_time=qr$backward_time)
  fit_rows[[length(fit_rows)+1]] <- data.frame(method='diagonal_QR_tail8',valid=qr_valid,seconds=proc.time()[3]-start,
    reason=if(qr_valid)'' else conditionMessage(qr),backward_time=if(qr_valid)NA_integer_ else qr$backward_time)
  saveRDS(fits,file.path(out,'fits.rds'));write_rows(fit_rows,'fit-status.csv')
  # Non-harm comparison to the Gaussian limit: retain perturbations without tuning.
  perturbations <- list()
  for (covariance_mode in c('diagonal','full')) {
    base <- if(covariance_mode=='full')limit else iapf_score_backward(model,pilot$clouds,N,'diagonal','negligible')
    for (floor_mode in c('positive','tail8')) {
      name <- paste(covariance_mode,floor_mode,sep='_');fit <- fits[[name]]
      error <- if(base$valid && fit$valid)max(vapply(seq_len(T),function(t)
        max(abs(base$twists[[t]]$mean-fit$twists[[t]]$mean),
            abs(base$twists[[t]]$covariance-fit$twists[[t]]$covariance)),numeric(1))) else NA_real_
      perturbations[[length(perturbations)+1]] <- data.frame(method=name,coefficient_perturbation=error)
    }
  }
  write_rows(perturbations,'floor-perturbation.csv')
  methods <- c('bootstrap','fully_adapted','current_observation','full_oracle',names(fits))
  observations <- iapf_observation_twists(model);rows <- list();details <- list()
  for (replicate in seq_len(4)) for (name in methods) {
    set.seed(seeds$final[replicate]);start <- proc.time()[3]
    count <- if(name=='bootstrap')10000 else if(name=='fully_adapted')5000 else N
    fit_valid <- if(name%in%names(fits))fits[[name]]$valid else TRUE
    if (!fit_valid) {
      rows[[length(rows)+1]] <- data.frame(d=d,data=j,method=name,replicate=replicate,N=count,
        status='fit_rejected',log_error=NA_real_,ratio=NA_real_,relative_error=NA_real_,seconds=0,
        resampling_count=NA_integer_,floor_max=NA_real_,floor_median_mean=NA_real_,minimum_ess=NA_real_)
      next
    }
    result <- if(name=='fully_adapted')iapf_fully_adapted(model,count,.5) else {
      twists <- if(name=='bootstrap')constant else if(name=='current_observation')observations else
        if(name=='full_oracle')exact else fits[[name]]$twists
      iapf_apf(model,twists,count,.5,FALSE)
    }
    error <- result$log_likelihood-kalman;ratio <- exp(error)
    stopifnot(is.finite(error),is.finite(ratio))
    if(name%in%c('full_oracle','full_negligible'))stopifnot(abs(error)<=1e-8)
    rows[[length(rows)+1]] <- data.frame(d=d,data=j,method=name,replicate=replicate,N=count,status='complete',
      log_error=error,ratio=ratio,relative_error=ratio-1,seconds=proc.time()[3]-start,
      resampling_count=result$resampling_count,floor_max=if(name=='fully_adapted')0 else result$floor_probability_max,
      floor_median_mean=if(name=='fully_adapted')0 else mean(result$floor_probability_quantiles[,'median']),
      minimum_ess=if(name=='fully_adapted')NA_real_ else min(result$ess))
    details[[paste(name,replicate,sep='_')]] <- result
    write_rows(rows,'partial-records.csv')
  }
  write_rows(rows,'records.csv');saveRDS(details,file.path(out,'filter-details.rds'))
  write.csv(data.frame(d=d,data=j,N=N,T=T,kalman=kalman,oracle_coefficient_error=oracle_error),
            file.path(out,'checks.csv'),row.names=FALSE)
  cat('Completed paper-model case',d,j,'with',length(rows),'final evaluations\n')
} else stop('Unknown mode')

# CPU-only independent R diagnostic. Neither Equation 15 nor a runtime backend.
# Uses the existing APF, controller and fitters; the hybrid is explicitly named.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==6, Sys.getenv('CUDA_VISIBLE_DEVICES')=='-1')
out <- args[1]; mode <- args[2]; data_seed <- as.integer(args[3])
seed <- as.integer(args[4]); label <- as.integer(args[5]); method <- args[6]
source('docs/benchmarks/reference_iapf_paper.R')
source('docs/benchmarks/reference_iapf_author_choices.R')
source('docs/benchmarks/diagnostic_iapf_paper_score.R')
options(digits=17)
write_table <- function(x,name) write.csv(x,file.path(out,name),row.names=FALSE)
if(mode=='verify') {
  pairs <- read.csv(file.path(out,'pairs.csv'),stringsAsFactors=FALSE)
  rows <- lapply(seq_len(nrow(pairs)),function(i) {
    base <- readRDS(pairs$score[i]); hybrid <- readRDS(pairs$rescue[i])
    healthy <- length(hybrid$events)==0L
    exact <- if(healthy)identical(base$result,hybrid$result) &&
      identical(base$fits,hybrid$fits) && identical(base$filter_calls,hybrid$filter_calls) else NA
    if(healthy)stopifnot(exact)
    data.frame(data_seed=pairs$data_seed[i],label=pairs$label[i],
      no_rescue=healthy,bitwise_identical=exact)
  })
  write_table(do.call(rbind,rows),'healthy-identity.csv')
  quit(status=0)
}
dat <- iapf_paper_data(80,100,seed=data_seed)
model <- iapf_linear_model(dat$observations,dat$A)
kalman <- iapf_kalman(model)$log_likelihood
saveRDS(list(data=dat,data_seed=data_seed,run_seed=seed,label=label,method=method,
  cpu_only=TRUE,diagnostic_only=TRUE),file.path(out,'inputs.rds'))
old_root <- '/home/chakwong/BayesFilter/docs/plans/artifacts/iapf-adaptive-replication-ladder-20260922-01'
old_learner <- function() {
  first <- ((label-1)%/%10)*10+1
  readRDS(file.path(old_root,sprintf('d80_r%04d-%04d_attempt01',first,first+9),
    sprintf('learner_score_after_k_r%d.rds',label)))
}
record <- function(result,status,particles,stop_iteration=NA_integer_,rescues=0L) {
  complete <- status=='complete'
  error <- if(complete)result$log_likelihood-kalman else NA_real_
  data.frame(data_seed=data_seed,run_seed=seed,label=label,method=method,status=status,
    particles=particles,stop_iteration=stop_iteration,rescues=rescues,
    log_likelihood=if(complete)result$log_likelihood else NA_real_,kalman=kalman,
    log_error=error,ratio=if(complete)exp(error) else NA_real_)
}

if(mode=='mechanism') {
  # Observe actual calls without replacing the numerical implementation.
  original_target <- iapf_score_target
  original_fit <- iapf_score_fit_cloud
  context <- NULL; failure <- NULL
  iapf_score_target <- function(model,x,time,next_twist=NULL) {
    h <- original_target(model,x,time,next_twist)
    context <<- list(x=x,h=h,time=time,next_twist=next_twist)
    h
  }
  iapf_score_fit_cloud <- function(x,h) {
    fit <- original_fit(x,h)
    if(!fit$valid) failure <<- c(context,list(fit=fit))
    fit
  }
  set.seed(seed)
  bootstrap <- iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),1000,.5,TRUE)
  actual <- iapf_score_backward(model,bootstrap$clouds,1000,'diagonal','tail8')
  iapf_score_target <- original_target; iapf_score_fit_cloud <- original_fit
  old <- old_learner()
  stopifnot(!actual$valid,actual$reason=='nonpositive_precision',
    identical(bootstrap$log_likelihood,old$result$likelihood_history[1]),
    identical(actual,old$fit_history[[1]]$actual),!is.null(failure))
  saveRDS(list(model=model,failure=failure,actual=actual),file.path(out,'failure-context.rds'))
  x <- failure$x; h <- failure$h; t <- failure$time; next_twist <- failure$next_twist
  n <- nrow(x); d <- ncol(x); m <- colMeans(x); xc <- sweep(x,2,m)
  C <- crossprod(xc)/n; L <- t(chol(C)); z <- t(solve(L,t(xc)))
  g <- h%*%L; a <- colMeans(g); raw <- -crossprod(sweep(g,2,a),z)/n
  P <- (raw+t(raw))/2; pe <- eigen(P,symmetric=TRUE)
  S <- model$transition_covariance+next_twist$covariance
  B <- t(model$A)%*%solve(S,model$A)
  future <- sweep(-x%*%t(model$A),2,next_twist$mean,'+')
  u <- future%*%solve(S,model$A)
  be <- eigen(B,symmetric=TRUE); rotated_u <- u%*%be$vectors
  # Here H=R=I. Matrix determinant lemma gives an exact PD test for
  # I+r B-r(1-r)u u': 1-r(1-r)u'(I+r B)^(-1)u > 0.
  curvature_margin <- function(r) 1-r*(1-r)*rowSums(rotated_u^2/
    (1+outer(r,be$values)))
  shifts <- c(0,-2,-4,-8,-16,-32,-64,-128,-1000)
  curves <- lapply(shifts,function(shift) {
    guide <- next_twist; guide$log_floor <- next_twist$log_floor+shift
    score <- iapf_score_target(model,x,t,guide); fit <- iapf_score_fit_cloud(x,score)
    r <- attr(score,'gaussian_probability'); margins <- curvature_margin(r)
    data.frame(log_floor_shift=shift,fit_valid=fit$valid,
      precision_margin=fit$diagnostics$precision_margin,
      r_min=min(r),r_mean=mean(r),negative_curvature_count=sum(margins<0),
      curvature_margin_min=min(margins))
  })
  write_table(do.call(rbind,curves),'floor-continuation.csv')
  r <- attr(h,'gaussian_probability'); margins <- curvature_margin(r)
  j <- which.min(margins); uj <- u[j,]
  negative_hessian <- diag(d)+r[j]*B-r[j]*(1-r[j])*tcrossprod(uj)
  ne <- eigen(negative_hessian,symmetric=TRUE)
  v <- ne$vectors[,d]; point <- x[j,,drop=FALSE]
  delta <- 1e-4 # Central differences: repeat at half step to check truncation.
  fd_rows <- lapply(c(delta,delta/2),function(eps) {
    plus <- point+eps*matrix(v,1); minus <- point-eps*matrix(v,1)
    fd_score <- as.vector((iapf_score_target(model,plus,t,next_twist)-
      iapf_score_target(model,minus,t,next_twist))/(2*eps))
    fd_log <- (iapf_backward_log_target(model,plus,t,next_twist)-
      iapf_backward_log_target(model,minus,t,next_twist))/(2*eps)
    data.frame(step=eps,score_direction_error=abs(fd_log-sum(h[j,]*v)),
      hessian_vector_error=max(abs(fd_score+as.vector(negative_hessian%*%v))))
  })
  fd <- do.call(rbind,fd_rows); write_table(fd,'finite-differences.csv')
  stopifnot(max(fd$score_direction_error)<1e-6,max(fd$hessian_vector_error)<1e-6,
    (min(ne$values)<0)==(margins[j]<0))
  # Affine Gaussian score on the SAME failing cloud, with the SAME next mean/V.
  J <- diag(d)+B
  linear <- as.vector(model$observations[t,]+t(model$A)%*%solve(S,next_twist$mean))
  affine_score <- sweep(-x%*%J,2,linear,'+')
  gaussian_fit <- iapf_score_fit_cloud(x,affine_score)
  recovery_error <- max(abs(gaussian_fit$mean-solve(J,linear)),
    abs(gaussian_fit$covariance-solve(J)))
  stopifnot(gaussian_fit$valid,recovery_error<1e-9)
  # Strict QR alternative on the exact failed local target, plus a complete
  # independent backward QR refit (the latter is the declared rescue).
  local_qr <- tryCatch(iapf_choice_fit(x,iapf_backward_log_target(model,x,t,next_twist),
    1000,'qr',NULL,'tail8',200),iapf_fit_error=function(e)e)
  full_qr <- tryCatch(iapf_choice_backward(model,bootstrap$clouds,1000,'qr',NULL,'tail8',200),
    iapf_fit_error=function(e)e)
  summary <- data.frame(label=label,time=t,baseline_reproduced=TRUE,
    cloud_margin=actual$diagnostics[[t]]$cloud_margin,
    precision_min=min(pe$values),precision_max=max(pe$values),
    precision_margin=actual$diagnostics[[t]]$precision_margin,
    whitening_error=max(abs(crossprod(z)/n-diag(d))),skew_norm=norm(raw-t(raw),'F'),
    score_residual=mean(rowSums((sweep(g,2,a)+z%*%P)^2)),
    target_nonconcave_points=sum(margins<0),curvature_margin_min=min(margins),
    example_negative_hessian_min=min(ne$values),gaussian_recovery_error=recovery_error,
    same_target_qr_valid=!inherits(local_qr,'iapf_fit_error'),
    full_backward_qr_valid=!inherits(full_qr,'iapf_fit_error'))
  write_table(summary,'mechanism.csv'); print(summary)
} else if(mode=='learner') {
  stopifnot(method%in%c('score','score_qr_rescue','qr'))
  fits <- list(); filter_calls <- list(); events <- list()
  fitter <- function(model,clouds,N,maxit,floor_tail_power) {
    index <- length(fits)+1L
    if(method=='qr') {
      fit <- iapf_choice_backward(model,clouds,N,'qr',NULL,'tail8',maxit)
    } else {
      actual <- iapf_score_backward(model,clouds,N,'diagonal','tail8')
      if(!actual$valid) {
        events[[length(events)+1L]] <<- list(fit_call=index,time=actual$backward_time,
          reason=actual$reason,diagnostics=actual$diagnostics)
        if(method=='score') stop(structure(list(message=actual$reason,call=NULL,
          backward_time=actual$backward_time,diagnostics=actual$diagnostics),
          class=c('iapf_fit_error','error','condition')))
        fit <- iapf_choice_backward(model,clouds,N,'qr',NULL,'tail8',maxit)
      } else fit <- list(twists=actual$twists,diagnostics=actual$diagnostics)
    }
    fits[[index]] <<- fit
    fit
  }
  filter <- function(model,twists,N,kappa,store_clouds) {
    answer <- iapf_apf(model,twists,N,kappa,store_clouds)
    filter_calls[[length(filter_calls)+1L]] <<- list(N=N,log_likelihood=answer$log_likelihood)
    answer
  }
  set.seed(seed)
  answer <- tryCatch(iapf_iterate(model,N0=1000,k=5,tau=.5,kappa=.5,
    max_iterations=18,max_particles=4000,fit_maxit=200,floor_tail_power=8,
    .filter=filter,.fit=fitter,fit_mode='paper_eq15',doubling_mode='after_k',
    stopping_window=6,cv_sd_mode='sample'),iapf_fit_error=function(e)e)
  rejected <- inherits(answer,'iapf_fit_error')
  status <- if(rejected)'fit_rejection' else answer$status
  history <- if(rejected)answer$likelihood_history else answer$history
  counts <- if(rejected)answer$particle_history else answer$counts
  complete <- status=='complete'
  stopifnot(length(filter_calls)==length(history)+as.integer(complete),all(is.finite(history)))
  if(complete) {
    stopifnot(identical(answer$final$log_likelihood,tail(filter_calls,1)[[1]]$log_likelihood),
      length(fits)==answer$stop_iteration,all(is.finite(answer$final$log_likelihood)))
    answer$final$clouds <- answer$final$log_weights <- answer$final$ancestors <- NULL
  }
  old_prefix <- NA; old_guides <- NA
  if(data_seed==971080) {
    old <- old_learner(); old_answer <- old$result
    old_history <- if(inherits(old_answer,'iapf_fit_error'))old_answer$likelihood_history else old_answer$history
    old_counts <- if(inherits(old_answer,'iapf_fit_error'))old_answer$particle_history else old_answer$counts
    old_prefix <- identical(head(history,length(old_history)),old_history) &&
      identical(head(counts,length(old_counts)),old_counts)
    stopifnot(old_prefix)
    if(method=='score') {
      old_guides <- length(fits)>=length(old$fit_history) &&
        all(vapply(seq_along(old$fit_history),function(i)
          identical(fits[[i]]$twists,old$fit_history[[i]]$actual$twists),logical(1)))
      stopifnot(old_guides)
    }
  }
  saveRDS(list(result=answer,fits=fits,events=events,filter_calls=filter_calls,
    method=method,data_seed=data_seed,run_seed=seed),file.path(out,'learner.rds'))
  write_table(data.frame(iteration=seq_along(history)-1,N=counts,logZ=history,
    cv=vapply(seq_along(history),function(i)if(i>=6)iapf_likelihood_cv(history[(i-5):i])
      else NA_real_,numeric(1))),'history.csv')
  write_table(data.frame(old_history_exact=old_prefix,old_guides_exact=old_guides,
    controller_wiring=TRUE,rejection_events=length(events)),'checks.csv')
  row <- record(if(complete)answer$final else NULL,status,
    if(complete)answer$final_particles else tail(counts,1),
    if(complete)answer$stop_iteration else NA_integer_,if(method=='score_qr_rescue')length(events) else 0L)
  write_table(row,'records.csv'); print(row)
} else if(mode=='controls') {
  rows <- list()
  for(control in c('bootstrap','fully_adapted','current_observation','full_oracle')) {
    method <- control; set.seed(seed)
    N <- if(control=='bootstrap')10000 else if(control=='fully_adapted')5000 else 1000
    result <- if(control=='fully_adapted')iapf_fully_adapted(model,N,.5) else
      iapf_apf(model,switch(control,bootstrap=replicate(100,iapf_constant_twist(),simplify=FALSE),
        current_observation=iapf_observation_twists(model),full_oracle=iapf_exact_twists(model)),N,.5,FALSE)
    stopifnot(is.finite(result$log_likelihood))
    if(control=='full_oracle')stopifnot(abs(result$log_likelihood-kalman)<1e-8)
    saveRDS(result,file.path(out,paste0(control,'.rds')))
    rows[[length(rows)+1L]] <- record(result,'complete',N)
  }
  write_table(do.call(rbind,rows),'records.csv')
} else stop('unknown diagnostic mode')

# Independent CPU R reference hypotheses. Not an original-author implementation.
# F1/F2 use the printed Eq15 residual; weighted-log arms change that objective.
iapf_choice_floor <- function(mean,variance,N,rule="tail8") {
  stopifnot(rule %in% c("tail8","peak2","peak4","peak8"),N>=2,
    all(is.finite(variance)),all(variance>0))
  peak <- -.5*(length(mean)*log(2*pi)+sum(log(variance)))
  peak-if(rule=="tail8") .5*qchisq(N^-8,length(mean),lower.tail=FALSE) else
    as.numeric(sub("peak","",rule))*log(N)
}

iapf_choice_fit <- function(points,log_targets,N=nrow(points),arm="qr",
                           previous=NULL,floor_rule="tail8",maxit=200) {
  stopifnot(arm %in% c("qr","f1_strict","f1_loose","f2_strict","f2_loose",
    "f2_independent_strict","f2_independent_loose","wlog1","wlog2"))
  points <- as.matrix(points); d <- ncol(points)
  reject <- function(message,diagnostics=list(),parameters=NULL) {
    stop(structure(list(message=message,call=NULL,points=points,log_targets=log_targets,
      diagnostics=diagnostics,parameters=parameters,arm=arm),
      class=c("iapf_fit_error","error","condition")))
  }
  independent_previous <- startsWith(arm,"f2_independent") &&
    !is.null(previous) && !previous$constant
  if(independent_previous) {
    if(length(previous$mean)!=d || !all(is.finite(previous$mean)) ||
       !all(is.finite(previous$covariance)) || any(diag(previous$covariance)<=0))
      reject("invalid previous guide")
    qr_start <- list(parameters=c(previous$mean,log(diag(previous$covariance))),
      log_fit=list(design_condition=NA_real_))
  } else qr_start <- iapf_fit_initial(points,log_targets,TRUE)
  anchor <- qr_start$parameters
  log_anchor <- iapf_log_normal(points,anchor[seq_len(d)],diag(exp(anchor[d+seq_len(d)]),d))
  fixed_log_scale <- max(log_anchor)
  fixed_loss_scale <- mean(exp(log_anchor-fixed_log_scale)^2)
  coordinate_scale <- c(sqrt(exp(anchor[d+seq_len(d)])),rep(1,d))
  evaluate <- function(p)iapf_fit_objective(p,points,log_targets,fixed_log_scale)
  start <- anchor; start_source <- "qr"
  if(startsWith(arm,"f2") && !is.null(previous) && !previous$constant) {
    start <- c(previous$mean,log(diag(previous$covariance)))
    start_source <- "previous_iteration"
  }
  if(independent_previous) start_source <- "previous_iteration_independent_scales"
  diagnostics <- list(arm=arm,objective="log_quadratic",start_source=start_source,
    convergence=0L,optimizer_message="QR",evaluations=1L,
    initial_relative_residual=evaluate(start)$relative_residual,
    initial_profiled_loss=evaluate(start)$value,
    start_distance_to_qr=if(independent_previous)NA_real_ else max(abs((start-anchor)/coordinate_scale)),
    start_distance_to_anchor=max(abs((start-anchor)/coordinate_scale)),
    initialization_requires_qr=!independent_previous,
    start_parameters=start,coordinate_scale=coordinate_scale,
    fixed_log_scale=fixed_log_scale,fixed_loss_scale=fixed_loss_scale,
    weight_ess=nrow(points),weighted_design_condition=qr_start$log_fit$design_condition,
    boundary=FALSE,loss_underflow=FALSE,factr=NA_real_)
  p <- anchor
  if(startsWith(arm,"wlog")) {
    exponent <- as.integer(sub("wlog","",arm))
    weights <- exp(exponent*(log_targets-max(log_targets)))
    center <- colMeans(points); scale <- sqrt(colMeans(sweep(points,2,center)^2))
    z <- sweep(sweep(points,2,center),2,scale,"/")
    design <- cbind(1,z,z^2)
    reg <- lm.wfit(design,log_targets-max(log_targets),w=weights)
    diagnostics$objective <- paste0("weighted_log_quadratic_power",exponent)
    diagnostics$weight_ess <- sum(weights)^2/sum(weights^2)
    diagnostics$weighted_design_condition <- kappa(design*sqrt(weights),exact=TRUE)
    quadratic <- reg$coefficients[1+d+seq_len(d)]
    linear <- reg$coefficients[1+seq_len(d)]
    if(reg$rank!=ncol(design) || !all(is.finite(reg$coefficients)))
      reject("rank-deficient weighted log fit",diagnostics)
    if(any(quadratic>=0)) reject("nonconcave weighted log fit",diagnostics)
    p <- c(center-scale*linear/(2*quadratic),log(-scale^2/(2*quadratic)))
  } else if(arm!="qr") {
    diagnostics$objective <- "printed_equation15_profiled_density_residual"
    diagnostics$factr <- if(endsWith(arm,"loose"))1e7 else 10
    fn <- function(u)evaluate(anchor+coordinate_scale*u)$value/fixed_loss_scale
    gr <- function(u)evaluate(anchor+coordinate_scale*u)$gradient*coordinate_scale/fixed_loss_scale
    limit <- -log(.Machine$double.eps)
    lower <- c(rep(-Inf,d),rep(-limit,d))
    upper <- c(rep(Inf,d),rep(limit,d))
    u0 <- (start-anchor)/coordinate_scale
    if(any(start<lower | start>upper))reject("initial floating domain boundary",diagnostics,start)
    fit <- if(evaluate(start)$relative_residual<=1e-26)
      list(par=u0,convergence=0L,message="exact initial fit",counts=c("function"=1L)) else
      optim(u0,fn,gr,method="L-BFGS-B",lower=(lower-anchor)/coordinate_scale,
        upper=(upper-anchor)/coordinate_scale,
        control=list(maxit=maxit,factr=diagnostics$factr,pgtol=1e-8,lmm=5))
    p <- anchor+coordinate_scale*fit$par
    diagnostics$convergence <- fit$convergence
    diagnostics$optimizer_message <- fit$message
    diagnostics$evaluations <- unname(fit$counts[1])
    diagnostics$boundary <- any(p[d+seq_len(d)]<=-limit+1e-6 |
      p[d+seq_len(d)]>=limit-1e-6)
  }
  ev <- evaluate(p)
  diagnostics$relative_residual <- ev$relative_residual
  diagnostics$profiled_loss <- ev$value
  diagnostics$gradient_max <- max(abs(ev$gradient*coordinate_scale/fixed_loss_scale))
  diagnostics$loss_underflow <- ev$loss_underflow
  diagnostics$logdet_change <- sum(p[d+seq_len(d)]-anchor[d+seq_len(d)])
  diagnostics$log_density_shift <- ev$log_density_max-max(log_anchor)
  failure <- if(diagnostics$boundary)"floating domain boundary" else
    if(diagnostics$convergence!=0)"Eq15 optimizer did not converge" else
    if((startsWith(arm,"f1") || startsWith(arm,"f2")) &&
      (ev$loss_underflow || !is.finite(ev$relative_residual)))"density loss underflow" else NULL
  if(!is.null(failure))reject(failure,diagnostics,p)
  variance <- exp(p[d+seq_len(d)]); mean <- p[seq_len(d)]
  log_floor <- iapf_choice_floor(mean,variance,N,floor_rule)
  diagnostics$floor_rule <- floor_rule; diagnostics$log_floor <- log_floor
  list(twist=iapf_gaussian_twist(mean,diag(variance,d),log_floor),
    diagnostics=diagnostics,parameters=p)
}

iapf_choice_backward <- function(model,clouds,N,arm="qr",previous=NULL,
                                 floor_rule="tail8",maxit=200) {
  twists <- diagnostics <- vector("list",model$horizon)
  for(time in rev(seq_len(model$horizon))) {
    target <- iapf_backward_log_target(model,clouds[[time]],time,
      if(time<model$horizon)twists[[time+1]] else NULL)
    fitted <- tryCatch(iapf_choice_fit(clouds[[time]],target,N,arm,
      if(is.null(previous))NULL else previous[[time]],floor_rule,maxit),
      iapf_fit_error=function(e) {e$backward_time <- time; stop(e)})
    twists[[time]] <- fitted$twist; diagnostics[[time]] <- fitted$diagnostics
  }
  list(twists=twists,diagnostics=diagnostics)
}

iapf_choice_run <- function(model,arm="qr",floor_rule="tail8",sd_mode="sample",
    N0=1000,k=5,tau=.5,max_iterations=20,max_particles=16000) {
  previous <- NULL
  fitter <- function(model,clouds,N,maxit,floor_tail_power) {
    fitted <- iapf_choice_backward(model,clouds,N,arm,previous,floor_rule,maxit)
    previous <<- fitted$twists
    fitted
  }
  result <- iapf_iterate(model,N0=N0,k=k,tau=tau,kappa=.5,
    max_iterations=max_iterations,max_particles=max_particles,fit_maxit=200,
    floor_tail_power=8,doubling_mode="after_k",stopping_window=k+1,
    cv_sd_mode=sd_mode,.fit=fitter)
  result$hypothesis <- list(fit=arm,floor=floor_rule,sd_mode=sd_mode,
    status="independent_R_numerical_hypothesis_not_author_identity")
  result
}

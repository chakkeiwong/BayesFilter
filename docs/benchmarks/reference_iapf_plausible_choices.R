# Independent CPU R reconstruction hypotheses, not original-author settings.
# Consumers use the shared independent-reference R core without monkey-patching.
iapf_local_eq15_fit <- function(points,log_targets,N=nrow(points),maxit=200,
                                floor_tail_power=8,box_multiplier=1,
                                allow_active_bounds=FALSE) {
  points <- as.matrix(points); d <- ncol(points)
  iapf_assert(length(box_multiplier)==1 && is.finite(box_multiplier) &&
    box_multiplier>0 && is.logical(allow_active_bounds) &&
    length(allow_active_bounds)==1 && !is.na(allow_active_bounds),"invalid local box")
  initial <- iapf_fit_initial(points,log_targets,TRUE)
  p0 <- initial$parameters
  steps <- box_multiplier*c(sqrt(exp(p0[d+seq_len(d)])),rep(log(2),d))
  lower <- p0-steps; upper <- p0+steps
  log_scale <- max(iapf_log_normal(points,p0[seq_len(d)],
    diag(exp(p0[d+seq_len(d)]),d)))
  evaluate <- function(p) iapf_fit_objective(p,points,log_targets,log_scale)
  fit <- optim(p0,function(p)evaluate(p)$value,
    function(p)evaluate(p)$gradient,method="L-BFGS-B",lower=lower,upper=upper,
    control=list(maxit=maxit,factr=10,pgtol=1e-8,lmm=5,
      # Hold optimizer scaling fixed while changing only the constraint width.
      parscale=c(sqrt(exp(p0[d+seq_len(d)])),rep(1,d))))
  ev <- evaluate(fit$par); initial_ev <- evaluate(p0)
  bound_margin <- min((fit$par-lower)/(2*steps),(upper-fit$par)/(2*steps))
  active_lower <- (fit$par-lower)/(2*steps)<=1e-6
  active_upper <- (upper-fit$par)/(2*steps)<=1e-6
  projected_gradient <- ev$gradient
  projected_gradient[(active_lower & ev$gradient>0) |
    (active_upper & ev$gradient<0)] <- 0
  reason <- if(fit$convergence!=0) "local Eq15 optimizer did not converge" else
    if(ev$loss_underflow || !is.finite(ev$relative_residual)) "invalid local Eq15 objective" else
    if(bound_margin<=1e-6 && !allow_active_bounds)
      "local Eq15 solution depends on an arbitrary active bound" else NULL
  if(!is.null(reason)) stop(structure(list(message=reason,call=NULL,
    parameters=fit$par,initial_parameters=p0,lower=lower,upper=upper,
    points=points,log_targets=log_targets,evaluation=ev,initial_evaluation=initial_ev,
    optimizer_code=fit$convergence,optimizer_message=fit$message,box_multiplier=box_multiplier,
    relative_bound_margin=bound_margin),class=c("iapf_fit_error","error","condition")))
  v <- exp(fit$par[d+seq_len(d)])
  log_floor <- -.5*(d*log(2*pi)+sum(log(v)))-
    .5*qchisq(N^(-floor_tail_power),df=d,lower.tail=FALSE)
  list(twist=iapf_gaussian_twist(fit$par[seq_len(d)],diag(v,d),log_floor),
    diagnostics=list(convergence=fit$convergence,optimizer="L-BFGS-B local box",
      evaluations=unname(fit$counts[1]),loss=ev$value,gradient_max=max(abs(ev$gradient)),
      relative_residual=ev$relative_residual,initial_relative_residual=initial_ev$relative_residual,
      relative_bound_margin=bound_margin,boundary=bound_margin<=1e-6,log_floor=log_floor,
      box_multiplier=box_multiplier,active_bound_coordinates=sum(active_lower|active_upper),
      active_bounds_explicitly_allowed=allow_active_bounds,
      projected_gradient_max=max(abs(projected_gradient)),
      initial_loss=initial_ev$value,
      fit_mode=if(allow_active_bounds)"constrained_local_equation15_diagnostic" else
        "local_box_equation15",log_variance_min=min(log(v)),log_variance_max=max(log(v))))
}

iapf_local_eq15_backward <- function(model,clouds,N,maxit=200,floor_tail_power=8,
                                    box_multiplier=1,allow_active_bounds=FALSE) {
  twists <- diagnostics <- vector("list",model$horizon)
  for(time in rev(seq_len(model$horizon))) {
    next_twist <- if(time<model$horizon)twists[[time+1]] else NULL
    log_target <- iapf_backward_log_target(model,clouds[[time]],time,next_twist)
    fit <- tryCatch(iapf_local_eq15_fit(clouds[[time]],log_target,N,maxit,floor_tail_power,
      box_multiplier,allow_active_bounds),
      iapf_fit_error=function(e) {e$backward_time <- time; stop(e)})
    twists[[time]] <- fit$twist; diagnostics[[time]] <- fit$diagnostics
  }
  list(twists=twists,diagnostics=diagnostics)
}

iapf_reconstruction_settings <- function(arm) {
  stopifnot(arm %in% c("current","delayed","floor4","local_eq15"))
  list(fit_mode=if(arm=="local_eq15")"paper_eq15" else "log_quadratic",
    floor_tail_power=if(arm=="floor4")4 else 8,
    doubling_mode=if(arm=="current")"first_full_window" else "after_k")
}

iapf_reconstruction <- function(model,arm) {
  settings <- iapf_reconstruction_settings(arm)
  do.call(iapf_iterate,c(list(model=model),settings,
    if(arm=="local_eq15")list(.fit=iapf_local_eq15_backward) else list()))
}

iapf_constrained_reconstruction <- function(model,box_multiplier=1,fit_maxit=200) {
  # Explicit diagnostic extension. An active constraint is emitted, never hidden.
  fitter <- function(model,clouds,N,maxit,floor_tail_power)
    iapf_local_eq15_backward(model,clouds,N,maxit,floor_tail_power,
      box_multiplier=box_multiplier,allow_active_bounds=TRUE)
  iapf_iterate(model,floor_tail_power=8,doubling_mode="after_k",.fit=fitter,
    fit_maxit=fit_maxit)
}

iapf_window_reconstruction <- function(model,stopping_window=5) {
  # Explicit controller extension, distinct from the paper reconstruction.
  iapf_iterate(model,fit_mode="log_quadratic",floor_tail_power=8,
    doubling_mode="after_k",stopping_window=stopping_window)
}

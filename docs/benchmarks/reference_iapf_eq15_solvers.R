# Independent CPU R reference diagnostics; unspecified author solver choices.
# Both parameterizations minimize the printed density residual, not log/relative LS.
iapf_eq15_joint_objective <- function(parameters,points,log_targets,log_density_scale) {
  d <- ncol(points); mean <- parameters[seq_len(d)]
  lv <- parameters[d+seq_len(d)]; variance <- exp(lv)
  centered <- sweep(points,2,mean)
  square <- sweep(centered^2,2,variance,"/")
  log_p <- -.5*(d*log(2*pi)+sum(lv)+rowSums(square))
  p <- exp(log_p-log_density_scale)
  b <- exp(log_targets-max(log_targets))
  scaled_target <- exp(parameters[2*d+1])*b
  residual <- p-scaled_target
  derivatives <- cbind(sweep(centered,2,variance,"/"),.5*(square-1))
  gradient <- c(colSums(derivatives*(2*residual*p/nrow(points))),
    -2*mean(residual*scaled_target))
  value <- mean(residual^2)
  iapf_assert(is.finite(value) && all(is.finite(gradient)),"nonfinite joint Eq15 residual")
  list(value=value,gradient=gradient)
}

iapf_eq15_solver <- function(points,log_targets,N=nrow(points),maxit=200,
    floor_tail_power=8,parameterization="joint",initialization="qr") {
  points <- as.matrix(points); d <- ncol(points)
  stopifnot(parameterization %in% c("joint","profiled"),
    initialization %in% c("qr","moments"),maxit %in% c(200,600))
  initial <- if(initialization=="qr")iapf_fit_initial(points,log_targets,TRUE) else list()
  if(initialization=="moments") {
    w <- exp(log_targets-max(log_targets)); w <- w/sum(w)
    m <- colSums(points*w); v <- colSums(sweep(points,2,m)^2*w)
    iapf_assert(all(is.finite(v)) && all(v>0),"degenerate moment start")
    initial$parameters <- c(m,log(v))
  }
  p0 <- initial$parameters
  log_p0 <- iapf_log_normal(points,p0[seq_len(d)],diag(exp(p0[d+seq_len(d)]),d))
  log_scale <- max(log_p0)
  profile <- function(p) iapf_fit_objective(p,points,log_targets,log_scale)
  initial_ev <- profile(p0)
  fixed_loss_scale <- mean(exp(log_p0-log_scale)^2)
  coord_scale <- c(sqrt(exp(p0[d+seq_len(d)])),rep(1,d))
  offset <- if(parameterization=="joint")c(p0,log(initial_ev$lambda_scaled)) else p0
  if(parameterization=="joint")coord_scale <- c(coord_scale,1)
  evaluate <- function(u) {
    p <- offset+coord_scale*u
    ev <- if(parameterization=="joint")
      iapf_eq15_joint_objective(p,points,log_targets,log_scale) else profile(p)
    list(value=ev$value/fixed_loss_scale,
      gradient=ev$gradient*coord_scale/fixed_loss_scale)
  }
  limit <- -log(.Machine$double.eps)
  lower <- c(rep(-Inf,d),rep(-limit,d)); upper <- c(rep(Inf,d),rep(limit,d))
  if(parameterization=="joint") {lower <- c(lower,-600); upper <- c(upper,600)}
  start <- rep(0,length(offset))
  # Exact representability is checked using a scale-free roundoff criterion.
  if(initial_ev$relative_residual<=1e-26) {
    fit <- list(par=start,convergence=0L,iterations=0L,message="exact initial fit",
      evaluations=c("function"=1L,gradient=1L))
  } else fit <- nlminb(start,function(u)evaluate(u)$value,
      gradient=function(u)evaluate(u)$gradient,
      lower=(lower-offset)/coord_scale,upper=(upper-offset)/coord_scale,
      control=list(iter.max=maxit,eval.max=4L*maxit,rel.tol=1e-10,x.tol=1e-10))
  params <- offset+coord_scale*fit$par
  p <- params[seq_len(2*d)]; ev <- profile(p); final <- evaluate(fit$par)
  variance <- exp(p[d+seq_len(d)])
  boundary <- any(p[d+seq_len(d)]<=-limit+1e-6 | p[d+seq_len(d)]>=limit-1e-6) ||
    (parameterization=="joint" && abs(params[2*d+1])>=600-1e-6)
  diagnostics <- list(parameterization=parameterization,initialization=initialization,
    convergence=fit$convergence,iterations=fit$iterations,optimizer="nlminb",
    optimizer_message=fit$message,optimizer_maxit=maxit,
    objective="printed_equation15_density_residual",raw_scaled_loss=final$value*fixed_loss_scale,
    profiled_loss=ev$value,initial_profiled_loss=initial_ev$value,
    relative_residual=ev$relative_residual,initial_relative_residual=initial_ev$relative_residual,
    gradient_max=max(abs(final$gradient)),boundary=boundary,loss_underflow=ev$loss_underflow,
    logdet_change=sum(p[d+seq_len(d)]-p0[d+seq_len(d)]),
    log_density_max=ev$log_density_max,initial_log_density_max=max(log_p0),
    log_density_scale=log_scale,fixed_loss_scale=fixed_loss_scale)
  failure <- if(boundary)"floating-point domain boundary" else
    if(ev$loss_underflow)"density loss underflow" else
    if(fit$convergence!=0)"Eq15 local optimizer did not converge" else NULL
  if(!is.null(failure))stop(structure(list(message=failure,call=NULL,
    parameters=p,initial_parameters=p0,points=points,log_targets=log_targets,
    diagnostics=diagnostics),class=c("iapf_fit_error","error","condition")))
  log_floor <- -.5*(d*log(2*pi)+sum(log(variance)))-
    .5*qchisq(N^(-floor_tail_power),df=d,lower.tail=FALSE)
  list(twist=iapf_gaussian_twist(p[seq_len(d)],diag(variance,d),log_floor),
       diagnostics=diagnostics,parameters=p)
}

iapf_eq15_backward <- function(model,clouds,N,maxit=200,floor_tail_power=8,
    parameterization="joint",initialization="qr") {
  twists <- diagnostics <- vector("list",model$horizon)
  for(time in rev(seq_len(model$horizon))) {
    target <- iapf_backward_log_target(model,clouds[[time]],time,
      if(time<model$horizon)twists[[time+1]] else NULL)
    fit <- tryCatch(iapf_eq15_solver(clouds[[time]],target,N,maxit,floor_tail_power,
      parameterization,initialization),iapf_fit_error=function(e) {
        e$backward_time <- time; stop(e)
      })
    twists[[time]] <- fit$twist; diagnostics[[time]] <- fit$diagnostics
  }
  list(twists=twists,diagnostics=diagnostics)
}

iapf_eq15_reconstruction <- function(model,parameterization="joint",initialization="qr",maxit=200) {
  fitter <- function(model,clouds,N,maxit,floor_tail_power)
    iapf_eq15_backward(model,clouds,N,maxit,floor_tail_power,parameterization,initialization)
  iapf_iterate(model,N0=1000,k=5,tau=.5,kappa=.5,max_iterations=20,
    max_particles=16000,fit_maxit=maxit,floor_tail_power=8,
    doubling_mode="after_k",stopping_window=6,.fit=fitter)
}

# Independent CPU R fitting extensions; not the paper's Equation15 objective.
# Evidence contract: iapf-r-targeted-fitting-repair-2026-09-22.md.
iapf_diagnostic_design <- function(points,log_targets,exponent=1) {
  points <- as.matrix(points); d <- ncol(points)
  center <- colMeans(points)
  scale <- sqrt(colMeans(sweep(points,2,center)^2))
  iapf_assert(all(is.finite(scale)) && all(scale>0),"degenerate diagnostic design")
  z <- sweep(sweep(points,2,center),2,scale,"/")
  D <- cbind(1,z,z^2); y <- log_targets-max(log_targets)
  weights <- exp(exponent*y); weights <- weights/sum(weights)
  A <- D*sqrt(weights); b <- y*sqrt(weights)
  list(D=D,y=y,A=A,b=b,center=center,scale=scale,d=d,weights=weights)
}

iapf_diagnostic_svd <- function(A,b) {
  s <- svd(A); threshold <- .Machine$double.eps*max(dim(A))*max(s$d)
  keep <- s$d>threshold
  coefficient <- as.vector(s$v[,keep,drop=FALSE] %*%
    (crossprod(s$u[,keep,drop=FALSE],b)/s$d[keep]))
  list(coefficients=coefficient,rank=sum(keep),threshold=threshold,
    condition=max(s$d)/min(s$d),decomposition=s)
}

iapf_diagnostic_active_set <- function(A,b,anchor,lambda,upper,initial,
    maxit=500,tolerance=1e-9) {
  p <- ncol(A)
  B <- if(lambda>0)rbind(A,sqrt(lambda)*diag(p)) else A
  response <- if(lambda>0)c(b,sqrt(lambda)*anchor) else b
  scale_loss <- max(1,sum((as.vector(A %*% anchor)-b)^2))
  beta <- pmin(initial,upper)
  bounded <- which(is.finite(upper));active <- bounded[beta[bounded]>=upper[bounded]]
  beta[active] <- upper[active]
  for(iteration in seq_len(maxit)) {
    free <- setdiff(seq_len(p),active)
    residual <- response-if(length(active))as.vector(B[,active,drop=FALSE] %*% beta[active]) else 0
    proposal <- beta
    if(length(free)) {
      solution <- iapf_diagnostic_svd(B[,free,drop=FALSE],residual)
      if(solution$rank<length(free))return(list(coefficients=beta,iterations=iteration,
        convergence=2L,message="active-set free design rank failure"))
      proposal[free] <- solution$coefficients
    }
    violated <- bounded[proposal[bounded]>upper[bounded]]
    if(length(violated)) {
      step <- proposal-beta
      ratios <- (upper[violated]-beta[violated])/step[violated]
      first <- which.min(ratios);alpha <- max(0,min(1,ratios[first]))
      beta <- beta+alpha*step
      active <- unique(c(active,violated[first]));beta[active] <- upper[active]
      next
    }
    beta <- proposal
    gradient <- as.vector(crossprod(B,as.vector(B %*% beta)-response))/scale_loss
    projected <- gradient;projected[active] <- pmax(gradient[active],0)
    if(max(abs(projected))<=tolerance)return(list(coefficients=beta,
      iterations=iteration,convergence=0L,message="active-set KKT satisfied"))
    release <- active[gradient[active]>tolerance]
    if(!length(release))return(list(coefficients=beta,iterations=iteration,
      convergence=3L,message="free-gradient residual remains"))
    active <- setdiff(active,release[which.max(gradient[release])])
  }
  list(coefficients=beta,iterations=maxit,convergence=1L,message="active-set iteration cap")
}

iapf_diagnostic_qp <- function(design,constrained=TRUE,ridge=FALSE,maxit=1000) {
  D <- design$D; A <- design$A; b <- design$b; d <- design$d
  raw <- iapf_diagnostic_svd(A,b)
  base <- iapf_diagnostic_svd(D/sqrt(nrow(D)),design$y/sqrt(nrow(D)))
  iapf_assert(base$rank==ncol(D),"unweighted diagnostic anchor rank failure")
  anchor <- base$coefficients; qi <- 1+d+seq_len(d)
  precision_reference <- median(abs(2*anchor[qi]))
  iapf_assert(is.finite(precision_reference) && precision_reference>0,
    "invalid precision reference")
  q_upper <- -.5*sqrt(.Machine$double.eps)*precision_reference
  s <- raw$decomposition; C <- .Machine$double.eps^(-.5)
  lambda <- if(ridge)max(0,(max(s$d)^2-C*min(s$d)^2)/(C-1)) else 0
  residual0 <- as.vector(A %*% anchor-b)
  scale_loss <- max(1,sum(residual0^2))
  fn <- function(delta) {
    r <- as.vector(A %*% delta)+residual0
    .5*(sum(r^2)+lambda*sum(delta^2))/scale_loss
  }
  gr <- function(delta) {
    r <- as.vector(A %*% delta)+residual0
    as.vector(crossprod(A,r)+lambda*delta)/scale_loss
  }
  coefficients <- if(lambda==0)raw$coefficients else anchor-as.vector(s$v %*%
    ((s$d*as.vector(crossprod(s$u,residual0)))/(s$d^2+lambda)))
  initial <- coefficients
  code <- 0L; message <- "unconstrained solution feasible"; iterations <- 0L
  if(constrained && any(coefficients[qi]>q_upper)) {
    upper <- rep(Inf,ncol(D)); upper[qi] <- q_upper-anchor[qi]
    delta0 <- pmin(coefficients-anchor,upper)
    fit <- optim(delta0,fn,gr,method="L-BFGS-B",upper=upper,
      control=list(maxit=maxit,factr=10,pgtol=1e-9,lmm=10))
    coefficients <- anchor+fit$par; code <- fit$convergence
    message <- fit$message; iterations <- unname(fit$counts[1])
  }
  original_code <- code;original_message <- message
  gradient <- gr(coefficients-anchor); projected_gradient <- gradient
  active <- if(constrained)coefficients[qi]>=q_upper-1e-10*max(1,abs(q_upper)) else rep(FALSE,d)
  # At an upper bound, a negative gradient points outside the feasible set.
  projected_gradient[qi[active]] <- pmax(gradient[qi[active]],0)
  kkt <- max(abs(projected_gradient))
  original_kkt <- kkt;active_set_iterations <- 0L
  if(constrained && kkt>1e-6) {
    upper <- rep(Inf,ncol(D));upper[qi] <- q_upper
    rescued <- iapf_diagnostic_active_set(A,b,anchor,lambda,upper,coefficients)
    coefficients <- rescued$coefficients;active_set_iterations <- rescued$iterations
    code <- rescued$convergence;message <- rescued$message
    gradient <- gr(coefficients-anchor);projected_gradient <- gradient
    active <- coefficients[qi]>=q_upper-1e-10*max(1,abs(q_upper))
    projected_gradient[qi[active]] <- pmax(gradient[qi[active]],0)
    kkt <- max(abs(projected_gradient))
  }
  valid_curvature <- all(is.finite(coefficients)) && all(coefficients[qi]<0)
  valid_constraint <- !constrained || all(coefficients[qi]<=q_upper+
    .Machine$double.eps*max(1,abs(q_upper)))
  valid_rank <- raw$rank==ncol(D) || lambda>0
  valid_solver <- kkt<=1e-6
  list(coefficients=coefficients,unconstrained=initial,
    accepted=valid_curvature && valid_constraint && valid_rank && valid_solver,
    diagnostics=list(objective="weighted_log_quadratic_extension",exponent=NA_real_,
      svd_rank=raw$rank,design_columns=ncol(D),svd_threshold=raw$threshold,
      design_condition=raw$condition,weight_ess=1/sum(design$weights^2),
      lambda=lambda,hessian_condition=(max(s$d)^2+lambda)/(min(s$d)^2+lambda),
      ridge_condition_target=C,precision_reference=precision_reference,q_upper=q_upper,
      active_constraints=sum(active),positive_curvatures=sum(coefficients[qi]>=0),
      kkt_residual=kkt,optimizer_convergence=code,optimizer_message=message,
      original_optimizer_convergence=original_code,original_optimizer_message=original_message,
      original_kkt_residual=original_kkt,active_set_iterations=active_set_iterations,
      evaluations=iterations,valid_rank=valid_rank,valid_solver=valid_solver,
      valid_constraint=valid_constraint,
      shape_training_rms=sd(as.vector(D %*% coefficients-design$y))),
    objective=fn,gradient=gr,anchor=anchor)
}

iapf_constrained_diagnostic_fit <- function(points,log_targets,N=nrow(points),
    exponent=1,constrained=TRUE,ridge=TRUE,floor_rule="tail8",maxit=1000) {
  design <- iapf_diagnostic_design(points,log_targets,exponent)
  fit <- iapf_diagnostic_qp(design,constrained,ridge,maxit)
  fit$diagnostics$exponent <- exponent
  if(!fit$accepted) stop(structure(list(message=paste("diagnostic fit rejected:",
      if(!fit$diagnostics$valid_rank)"rank" else if(fit$diagnostics$positive_curvatures>0)
        "curvature" else if(!fit$diagnostics$valid_constraint)"constraint" else "KKT"),
      call=NULL,points=points,log_targets=log_targets,
      diagnostics=fit$diagnostics,coefficients=fit$coefficients),
    class=c("iapf_fit_error","error","condition")))
  d <- design$d; q <- fit$coefficients[1+d+seq_len(d)]
  l <- fit$coefficients[1+seq_len(d)]
  variance <- -design$scale^2/(2*q)
  mean <- design$center-design$scale*l/(2*q)
  floor <- iapf_choice_floor(mean,variance,N,floor_rule)
  fit$diagnostics$floor_rule <- floor_rule; fit$diagnostics$log_floor <- floor
  list(twist=iapf_gaussian_twist(mean,diag(variance,d),floor),
    parameters=c(mean,log(variance)),coefficients=fit$coefficients,
    diagnostics=fit$diagnostics)
}

iapf_constrained_diagnostic_run <- function(model,N0=1000,max_iterations=12,
    max_particles=4000,sd_mode="sample") {
  fitter <- function(model,clouds,N,maxit,floor_tail_power) {
    twists <- diagnostics <- vector("list",model$horizon)
    for(time in rev(seq_len(model$horizon))) {
      target <- iapf_backward_log_target(model,clouds[[time]],time,
        if(time<model$horizon)twists[[time+1]] else NULL)
      fit <- tryCatch(iapf_constrained_diagnostic_fit(clouds[[time]],target,N),
        iapf_fit_error=function(e) {e$backward_time <- time;stop(e)})
      twists[[time]] <- fit$twist; diagnostics[[time]] <- fit$diagnostics
    }
    list(twists=twists,diagnostics=diagnostics)
  }
  ans <- iapf_iterate(model,N0=N0,k=5,tau=.5,kappa=.5,
    max_iterations=max_iterations,max_particles=max_particles,fit_maxit=1000,
    floor_tail_power=8,doubling_mode="after_k",stopping_window=6,
    cv_sd_mode=sd_mode,.fit=fitter)
  ans$hypothesis <- "independent_R_ridge_bounded_weighted_log_extension"
  ans
}

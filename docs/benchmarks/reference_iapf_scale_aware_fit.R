# Independent R/FP64 diagnostic reference. Not an admitted TF/GPU runtime.
# Equation (15), profiled lambda; positive residual scaling is frozen per step.
sa_log_density <- function(x,theta) {
  d <- ncol(x); v <- exp(theta[d+seq_len(d)])
  -.5*(d*log(2*pi)+sum(log(v))+rowSums(sweep(sweep(x,2,theta[seq_len(d)])^2,2,v,"/")))
}
sa_problem <- function(x,ly,theta0) {
  d <- ncol(x); n <- nrow(x); b <- exp(ly-max(ly)); b2 <- sum(b*b)
  units <- c(exp(theta0[d+seq_len(d)]/2),rep(1,d))
  evaluate <- function(z,scale=NULL) {
    theta <- theta0+units*z
    if(any(!is.finite(theta)) || any(abs(theta[d+seq_len(d)])>600)) return(NULL)
    lp <- sa_log_density(x,theta)
    if(any(!is.finite(lp))) return(NULL)
    if(is.null(scale)) scale <- max(lp)
    if(max(lp-scale)>300) return(NULL)
    p <- exp(lp-scale); p2 <- mean(p*p)
    if(!is.finite(p2) || p2<=0) return(NULL)
    lambda <- sum(p*b)/b2; r <- p-lambda*b
    v <- exp(theta[d+seq_len(d)]); dx <- sweep(x,2,theta[seq_len(d)])
    score <- cbind(sweep(dx,2,v,"/"),.5*(sweep(dx^2,2,v,"/")-1))
    Jp <- sweep(score,2,units,"*")*p
    J <- Jp-outer(b,colSums(Jp*b)/b2)
    loss <- mean(r*r); gradient <- as.vector(2*crossprod(J,r)/n)
    list(theta=theta,scale=scale,p=p,r=r,J=J,loss=loss,gradient=gradient,
      relative_gradient=max(abs(gradient))/p2,shape=loss/p2,p2=p2,
      log_raw_loss=2*scale+log(loss),log_density_norm=2*scale+log(p2),
      log_lambda=log(lambda)+scale-max(ly))
  }
  list(evaluate=evaluate,units=units,theta0=theta0)
}
sa_curvature <- function(problem,z,ev) {
  h <- .Machine$double.eps^(1/3); k <- length(z)
  H <- matrix(0,k,k)
  for(j in seq_len(k)) {
    dz <- rep(0,k); dz[j] <- h*max(1,abs(z[j]))
    ep <- problem$evaluate(z+dz,ev$scale); em <- problem$evaluate(z-dz,ev$scale)
    if(is.null(ep) || is.null(em)) return(c(min=NA,max=NA))
    H[,j] <- (ep$gradient-em$gradient)/(2*dz[j])/ev$p2
  }
  eig <- eigen((H+t(H))/2,symmetric=TRUE,only.values=TRUE)$values
  c(min=min(eig),max=max(eig))
}
sa_fit <- function(x,ly,theta0=NULL,maxit=400,tol=1e-6,scale_offset=0,curvature=TRUE,initial_damping=1e-3) {
  if(is.null(theta0)) theta0 <- iapf_fit_initial(x,ly,TRUE)$parameters
  problem <- sa_problem(x,ly,theta0); z <- rep(0,length(theta0))
  damping <- initial_damping; nu <- 2; rejected <- 0L; trace <- list(); status <- "iteration_cap"
  initial <- problem$evaluate(z)
  if(is.null(initial)) stop("invalid initial density")
  for(iteration in seq_len(maxit)) {
    ev <- problem$evaluate(z)
    ev <- problem$evaluate(z,ev$scale+scale_offset)
    trace[[length(trace)+1L]] <- data.frame(iteration=iteration,
      log_raw_loss=ev$log_raw_loss,shape=ev$shape,relative_gradient=ev$relative_gradient,
      log_density_norm=ev$log_density_norm,damping=damping,rejections=rejected)
    if(ev$relative_gradient<=tol) {status <- "stationary"; break}
    H <- crossprod(ev$J)/nrow(x); g <- ev$gradient/2
    peak <- max(diag(H))
    if(!is.finite(peak) || peak<=0) {status <- "singular_jacobian"; break}
    D <- pmax(diag(H),sqrt(.Machine$double.eps)*peak)
    accepted <- FALSE
    for(trial in seq_len(32)) {
      step <- tryCatch(as.vector(solve(H+diag(damping*D,length(D)),-g)),error=function(e)NULL)
      candidate <- if(is.null(step)) NULL else problem$evaluate(z+step,ev$scale)
      predicted <- if(is.null(step)) -Inf else -2*sum(g*step)-sum(step*(H%*%step))
      rho <- if(is.null(candidate) || predicted<=0) -Inf else (ev$loss-candidate$loss)/predicted
      if(is.finite(rho) && rho>0 && candidate$loss<ev$loss) {
        z <- z+step; damping <- max(.Machine$double.eps,damping*max(1/3,1-(2*rho-1)^3))
        nu <- 2; accepted <- TRUE; break
      }
      rejected <- rejected+1L; damping <- damping*nu; nu <- min(2*nu,1e6)
      if(!is.finite(damping) || damping>1/.Machine$double.eps) break
    }
    if(!accepted) {status <- "no_descent_step"; break}
  }
  ev <- problem$evaluate(z); trace <- do.call(rbind,trace)
  # Include the last accepted endpoint even when the iteration budget is reached.
  log_losses <- c(trace$log_raw_loss,ev$log_raw_loss)
  descent <- all(diff(log_losses)[is.finite(diff(log_losses))]<=1e-10)
  if(!descent) status <- "descent_check_failed"
  eig <- if(status=="stationary" && curvature) sa_curvature(problem,z,ev) else c(min=NA,max=NA)
  if(status=="stationary" && curvature && (!all(is.finite(eig)) || eig[1]< -1e-5*max(1,abs(eig[2]))))
    status <- "curvature_failed"
  list(status=status,theta=ev$theta,z=z,initial=theta0,units=problem$units,
    relative_gradient=ev$relative_gradient,shape=ev$shape,
    log_raw_loss=ev$log_raw_loss,log_lambda=ev$log_lambda,
    log_density_change=ev$log_density_norm-initial$log_density_norm,
    log_absolute_gradient=log(max(abs(ev$gradient)))+2*ev$scale,
    eigen_min=unname(eig[1]),eigen_max=unname(eig[2]),
    original_loss_descent=descent,iterations=iteration,rejections=rejected,trace=trace)
}
sa_twist <- function(theta,N,power,Q) {
  d <- length(theta)/2; variance <- exp(theta[d+seq_len(d)]); delta <- N^(-power)
  log_peak <- -.5*(d*log(2*pi)+as.numeric(determinant(Q+diag(variance,d),logarithm=TRUE)$modulus))
  iapf_gaussian_twist(theta[seq_len(d)],diag(variance,d),log(delta)-log1p(-delta)+log_peak)
}
sa_backward <- function(model,clouds,N,power,save_path,row_callback=NULL,maxit=400,initial_damping=1e-3) {
  twists <- vector("list",model$horizon); diagnostics <- vector("list",model$horizon)
  inputs <- vector("list",model$horizon)
  for(t in rev(seq_len(model$horizon))) {
    x <- clouds[[t]]; next_twist <- if(t<model$horizon)twists[[t+1]] else NULL
    ly <- iapf_backward_log_target(model,x,t,next_twist)
    inputs[[t]] <- list(points=x,log_targets=ly,next_twist=next_twist)
    fit <- tryCatch(sa_fit(x,ly,maxit=maxit,initial_damping=initial_damping),error=function(e)list(status="initialization_failed",message=conditionMessage(e)))
    first_fit <- fit
    if(identical(fit$status,"iteration_cap")) {
      fit <- sa_fit(x,ly,theta0=fit$initial,maxit=800,initial_damping=initial_damping)
      fit$first_attempt <- first_fit
    }
    diagnostics[[t]] <- fit
    if(!is.null(row_callback)) row_callback(t,fit)
    if(!identical(fit$status,"stationary")) {
      payload <- list(time=t,fit=fit,inputs=inputs,twists=twists,diagnostics=diagnostics,N=N,power=power)
      saveRDS(payload,save_path)
      stop(structure(list(message=paste("scale-aware fit",fit$status,"at time",t),call=NULL,
        backward_time=t,points=x,log_targets=ly,fit=fit,snapshot=save_path),
        class=c("iapf_fit_error","error","condition")))
    }
    twists[[t]] <- sa_twist(fit$theta,N,power,model$transition_covariance)
  }
  result <- list(twists=twists,diagnostics=diagnostics,inputs=inputs,N=N,power=power)
  saveRDS(result,save_path); result
}

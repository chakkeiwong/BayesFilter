# Independent diagnostic reconstruction of the local TF adaptive consumer.
# This is not the paper-study reference and does not change that frozen source.
source("docs/benchmarks/reference_iapf_paper.R")

tfref_f32 <- function(x) readBin(writeBin(x, raw(), size=4), "numeric", n=length(x), size=4)
tfref_matrix <- function(x, nr, nc) matrix(unlist(x), nr, nc, byrow=TRUE)
tfref_model <- function(theta, d, o) {
  A <- diag(theta[1], d)
  if(d>1) A[cbind(1:(d-1), 2:d)] <- tfref_f32(.08)
  list(A=A, Q=exp(2*theta[2])*diag(seq(1,tfref_f32(1.3),length.out=d),d),
    R=exp(2*theta[3])*diag(o), H=theta[4]*(diag(1,o,d)+matrix(tfref_f32(.15),o,d)),
    mean=theta[5]*seq(tfref_f32(.7),tfref_f32(1.1),length.out=d),
    P=exp(2*theta[6])*(diag(d)+matrix(tfref_f32(.15),d,d)))
}

tfref_profile <- function(z, log_target, par) {
  d <- ncol(z); mu <- par[1:d]; sd <- exp(par[d+1:d])
  e <- sweep(sweep(z,2,mu,"-"),2,sd,"/")
  # Direct density algebra; independent of the TF max-amplitude scaling.
  p <- exp(-sum(log(sd))-.5*rowSums(e^2)); b <- exp(log_target)
  lambda <- sum(p*b)/sum(b*b); residual <- p-lambda*b
  dp <- cbind(sweep(e,2,sd,"/"),e^2-1)*p
  list(loss=mean(residual^2), gradient=2*colMeans(dp*residual),
    shape=sum(residual^2)/sum(p^2), lambda=lambda)
}

tfref_fit <- function(points, log_targets, config) {
  d <- ncol(points); mu <- colMeans(points)
  sd <- sqrt(colMeans(sweep(points,2,mu,"-")^2))
  z <- sweep(sweep(points,2,mu,"-"),2,sd,"/")
  target_max <- max(log_targets); target <- log_targets-target_max
  lower <- c(rep(tfref_f32(-config$mean_bound),d),rep(tfref_f32(log(config$sd_lower)),d))
  upper <- c(rep(tfref_f32(config$mean_bound),d),rep(tfref_f32(log(config$sd_upper)),d))
  clip <- function(x) pmax(lower,pmin(upper,x))
  par <- clip(rep(0,2*d)); f <- tfref_profile(z,target,par)
  tolerance <- tfref_f32(config$fit_tolerance)
  norm <- function(p,g) max(abs(p-clip(p-g)))
  step <- 1; iteration <- 0L; healthy <- all(is.finite(z)) && all(is.finite(target)) && all(sd>0)
  while(iteration<config$max_fit_steps && healthy && norm(par,f$gradient)>tolerance) {
    accepted <- FALSE
    for(j in seq_len(config$max_backtracks)) {
      trial <- clip(par-step*f$gradient); trial_f <- tfref_profile(z,target,trial)
      accepted <- is.finite(trial_f$loss) &&
        trial_f$loss<=f$loss+tfref_f32(1e-4)*sum(f$gradient*(trial-par))
      if(accepted) break
      step <- step*.5
    }
    if(accepted) par <- trial
    f <- tfref_profile(z,target,par)
    healthy <- healthy && accepted && all(is.finite(f$gradient))
    step <- min(step*2,max(upper-lower)/tolerance); iteration <- iteration+1L
  }
  center <- mu+sd*par[1:d]; fitted_sd <- sd*exp(par[d+1:d])
  log_constant <- -.5*d*log(2*pi)-sum(log(sd))
  log_floor <- log(tfref_f32(config$floor_ratio))+log_constant-sum(par[d+1:d])
  projected_gradient <- norm(par,f$gradient)
  list(center=center,covariance=diag(fitted_sd^2,d),log_floor=log_floor,
    valid=healthy && is.finite(f$loss) && is.finite(f$shape) && is.finite(log(f$lambda)) && all(fitted_sd>0),
    converged=healthy && projected_gradient<=tolerance, iterations=iteration,
    projected_gradient=projected_gradient,shape=f$shape,loss=f$loss,
    log_lambda=log(f$lambda)+log_constant-target_max,
    boundary=any(pmin(par-lower,upper-par)<=32*.Machine$double.eps*(1+abs(par))),
    parameters=par)
}

tfref_backward <- function(theta, observations, clouds, config, d, o) {
  m <- tfref_model(theta,d,o); T <- nrow(observations); result <- vector("list",T)
  guides <- vector("list",T)
  for(t in T:1) {
    x <- clouds[[t]]; N <- nrow(x)
    target <- iapf_log_normal(matrix(rep(observations[t,],each=N),N,o),x%*%t(m$H),m$R)
    if(t<T) target <- target+iapf_log_integral(x%*%t(m$A),m$Q,guides[[t+1]])
    result[[t]] <- tfref_fit(x,target,config)
    f <- result[[t]]; guides[[t]] <- iapf_gaussian_twist(f$center,f$covariance,f$log_floor)
  }
  list(guides=guides,details=result,valid=all(vapply(result,function(x)x$valid,TRUE)),
    converged=all(vapply(result,function(x)x$converged,TRUE)))
}

tfref_filter <- function(theta, observations, draws, guides, d, o, constant=FALSE) {
  T <- nrow(observations); N <- length(draws$initial); m <- tfref_model(theta,d,o)
  initial <- tfref_matrix(draws$initial,N,d)
  noise <- lapply(draws$process,tfref_matrix,nr=N,nc=d)
  uniforms <- tfref_matrix(draws$ancestors,T+1,N)
  mixture <- tfref_matrix(draws$mixture,T,N)
  x <- sweep(initial%*%chol(m$P),2,m$mean,"+")
  clouds <- list(); labels <- branches <- integer(); cdfs <- list(); probabilities <- list()
  margin <- Inf
  future <- function(x,t) if(constant) rep(0,N) else iapf_log_integral(x%*%t(m$A),m$Q,guides[[t]])
  resample <- function(logw,u) {
    cumulative <- cumsum(exp(logw-iapf_logsum(logw)))
    index <- findInterval(u,cumulative)+1L
    stopifnot(all(index<=N))
    labels <<- c(labels,index-1L); cdfs[[length(cdfs)+1]] <<- cumulative
    margin <<- min(margin,abs(outer(u,cumulative,"-")))
    list(index=index,term=iapf_logmean(logw))
  }
  r <- resample(future(x,1),uniforms[1,]); total <- r$term; x <- x[r$index,,drop=FALSE]
  for(t in seq_len(T)) {
    mean <- x%*%t(m$A)
    if(constant) {
      x <- mean+noise[[t]]%*%chol(m$Q); lp <- rep(0,N); probability <- rep(0,N)
    } else {
      guide <- guides[[t]]
      # Independent precision-form product, not the TF gain formula.
      qi <- solve(m$Q); vi <- solve(guide$covariance); covariance <- solve(qi+vi)
      rhs <- mean%*%qi+matrix(rep(as.vector(vi%*%guide$mean),each=N),N,d)
      adapted_mean <- rhs%*%covariance
      integral <- iapf_log_integral(mean,m$Q,guide)
      probability <- exp(iapf_log_normal(mean,guide$mean,m$Q+guide$covariance)-integral)
      choose <- mixture[t,]<probability; branches <- c(branches,as.integer(choose))
      margin <- min(margin,abs(mixture[t,]-probability))
      x <- mean+noise[[t]]%*%chol(m$Q)
      adapted <- adapted_mean+noise[[t]]%*%chol(covariance)
      x[choose,] <- adapted[choose,,drop=FALSE]
      lp <- iapf_log_twist(x,guide)
    }
    probabilities[[t]] <- probability; clouds[[t]] <- x
    lg <- iapf_log_normal(matrix(rep(observations[t,],each=N),N,o),x%*%t(m$H),m$R)
    lf <- if(t<T) future(x,t+1) else rep(0,N)
    r <- resample(lg+lf-lp,uniforms[t+1,]); total <- total+r$term; x <- x[r$index,,drop=FALSE]
  }
  list(value=total,clouds=clouds,labels=labels,branches=branches,cdfs=cdfs,
    probabilities=probabilities,min_margin=margin)
}

tfref_decision <- function(values,counts,config) {
  iteration <- length(values)-1L; N <- tail(counts,1); k <- config$k
  if(iteration<k) return(list(action="fit",next_particles=N,cv=NA_real_))
  window <- tail(values,k+1); likelihoods <- exp(window-max(window))
  cv <- sd(likelihoods)/mean(likelihoods)
  if(iteration>k && cv<config$tau) return(list(action="final",next_particles=N,cv=cv))
  next_N <- if(all(tail(counts,k+1)==N) && any(diff(window)<0)) 2*N else N
  list(action=if(next_N>config$max_particles) "capacity_veto" else "fit",next_particles=next_N,cv=cv)
}

tfref_flat_cloud <- function(xs) unlist(lapply(xs,function(x)as.vector(t(x))))
tfref_flat_guides <- function(gs,field) unlist(lapply(gs,function(g) {
  x <- g[[field]]; if(is.matrix(x)) as.vector(t(x)) else x
}))

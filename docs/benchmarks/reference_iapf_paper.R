# Independent, noncanonical R reference for GJL (2017), equations (5)--(6),
# (15)--(16), Algorithms 3--5. No copied public iAPF code. CPU diagnostic only.
# This implements Gaussian transitions with fixed covariance and callable mean,
# and arbitrary observation log density. Fits use diagonal Gaussian twists plus
# a positive constant. An exact Gaussian twist is available as a test oracle.

iapf_logsum <- function(x) {
  m <- max(x)
  if (is.infinite(m) && m < 0) return(-Inf)
  m + log(sum(exp(x-m)))
}
iapf_logmean <- function(x) iapf_logsum(x)-log(length(x))
iapf_logadd <- function(a,b) {
  m <- pmax(a,b)
  ans <- m + log(exp(a-m)+exp(b-m))
  ans[is.infinite(m) & m < 0] <- -Inf
  ans
}
iapf_assert <- function(condition, message) {
  if (!isTRUE(condition)) stop(message, call.=FALSE)
}
iapf_chol <- function(covariance) {
  iapf_assert(is.matrix(covariance) && all(is.finite(covariance)) &&
    max(abs(covariance-t(covariance))) < 1e-10*(1+max(abs(covariance))),
    "invalid covariance")
  chol(covariance)
}
iapf_log_normal <- function(x, mean, covariance) {
  x <- as.matrix(x)
  centered <- if (is.matrix(mean)) x-mean else sweep(x,2,mean)
  factor <- iapf_chol(covariance)
  standardized <- t(backsolve(factor,t(centered),transpose=TRUE))
  -.5*(ncol(x)*log(2*pi)+rowSums(standardized^2))-sum(log(diag(factor)))
}
iapf_draw_normal <- function(means, covariance) {
  means <- as.matrix(means)
  means + matrix(rnorm(length(means)),nrow=nrow(means)) %*% iapf_chol(covariance)
}
iapf_constant_twist <- function() list(constant=TRUE)
iapf_gaussian_twist <- function(mean,covariance,log_floor=-Inf) {
  iapf_chol(covariance)
  iapf_assert(length(mean)==nrow(covariance) && all(is.finite(mean)) &&
    length(log_floor)==1 && !is.na(log_floor) && log_floor < Inf,"invalid twist")
  list(constant=FALSE,mean=as.numeric(mean),covariance=covariance,log_floor=log_floor)
}
iapf_log_twist <- function(x,twist) {
  if (twist$constant) return(rep(0,nrow(as.matrix(x))))
  iapf_logadd(iapf_log_normal(x,twist$mean,twist$covariance),twist$log_floor)
}
iapf_log_integral <- function(means,covariance,twist) {
  if (twist$constant) return(rep(0,nrow(as.matrix(means))))
  iapf_logadd(iapf_log_normal(means,twist$mean,covariance+twist$covariance),twist$log_floor)
}
iapf_proposal <- function(means,covariance,twist) {
  # Multiplying N(x;mu,Q) by N(x;m,V)+c yields exactly two components.
  means <- as.matrix(means)
  if (twist$constant) return(list(means=means,covariance=covariance,
      original_means=means,original_covariance=covariance,
      floor_probability=rep(0,nrow(means)),log_integral=rep(0,nrow(means)),constant=TRUE))
  sum_covariance <- covariance+twist$covariance
  gain <- covariance %*% chol2inv(iapf_chol(sum_covariance))
  shifted <- sweep(-means,2,twist$mean,"+")
  posterior_means <- means + shifted %*% t(gain)
  posterior_covariance <- covariance-gain %*% covariance
  posterior_covariance <- (posterior_covariance+t(posterior_covariance))/2
  iapf_chol(posterior_covariance)
  log_gaussian_mass <- iapf_log_normal(means,twist$mean,sum_covariance)
  log_integral <- iapf_logadd(log_gaussian_mass,twist$log_floor)
  list(means=posterior_means,covariance=posterior_covariance,
       original_means=means,original_covariance=covariance,
       floor_probability=exp(twist$log_floor-log_integral),
       log_integral=log_integral,constant=FALSE)
}
iapf_draw_proposal <- function(proposal) {
  n <- nrow(proposal$means)
  if (proposal$constant || all(proposal$floor_probability==0))
    return(iapf_draw_normal(proposal$means,proposal$covariance))
  original <- runif(n) < proposal$floor_probability
  draws <- matrix(NA_real_,n,ncol(proposal$means))
  if (any(!original)) draws[!original,] <- iapf_draw_normal(
    proposal$means[!original,,drop=FALSE],proposal$covariance)
  if (any(original)) draws[original,] <- iapf_draw_normal(
    proposal$original_means[original,,drop=FALSE],proposal$original_covariance)
  draws
}
iapf_log_proposal <- function(x,proposal) {
  if (proposal$constant) return(iapf_log_normal(x,proposal$means,proposal$covariance))
  probability <- proposal$floor_probability
  iapf_logadd(log1p(-probability)+iapf_log_normal(x,proposal$means,proposal$covariance),
    log(probability)+iapf_log_normal(x,proposal$original_means,proposal$original_covariance))
}
iapf_ess <- function(log_weights) {
  weights <- exp(log_weights-max(log_weights))
  sum(weights)^2/sum(weights^2)
}

iapf_model <- function(initial_mean,initial_covariance,transition_mean,
                       transition_covariance,log_observation,observations) {
  iapf_chol(initial_covariance); iapf_chol(transition_covariance)
  observations <- as.matrix(observations)
  iapf_assert(nrow(observations)>=1 && length(initial_mean)==nrow(initial_covariance),
              "invalid model dimensions")
  list(initial_mean=initial_mean,initial_covariance=initial_covariance,
       transition_mean=transition_mean,transition_covariance=transition_covariance,
       log_observation=log_observation,observations=observations,
       dimension=length(initial_mean),horizon=nrow(observations))
}
iapf_apf <- function(model,twists,N,kappa=.5,store_clouds=TRUE) {
  # Algorithm 5, including retained weights and likelihood factors at boundaries.
  iapf_assert(length(twists)==model$horizon && N>=2 && N==as.integer(N) &&
    is.finite(kappa) && kappa>=0 && kappa<=1,"invalid APF configuration")
  horizon <- model$horizon; d <- model$dimension
  clouds <- if (store_clouds) vector("list",horizon) else NULL
  weight_history <- if (store_clouds) vector("list",horizon) else NULL
  ancestors <- if (store_clouds) vector("list",horizon) else NULL
  prefixes <- original_prefix <- ess <- numeric(horizon)
  resampled <- logical(horizon)
  floor_max <- 0; accumulated <- 0
  floor_quantiles <- matrix(0,horizon,5,dimnames=list(NULL,c("min","q25","median","q75","max")))
  for (time in seq_len(horizon)) {
    if (time==1) {
      means <- matrix(rep(model$initial_mean,each=N),N,d)
      covariance <- model$initial_covariance
      previous <- rep(0,N)
      ancestor <- seq_len(N)
    } else {
      resampled[time] <- iapf_ess(weights) <= kappa*N
      if (resampled[time]) {
        accumulated <- accumulated + iapf_logmean(weights)
        ancestor <- sample.int(N,N,replace=TRUE,prob=exp(weights-max(weights)))
        previous <- rep(0,N)
      } else {
        ancestor <- seq_len(N)
        previous <- weights
      }
      means <- as.matrix(model$transition_mean(x[ancestor,,drop=FALSE],time))
      covariance <- model$transition_covariance
    }
    proposal <- iapf_proposal(means,covariance,twists[[time]])
    floor_max <- max(floor_max,proposal$floor_probability)
    floor_quantiles[time,] <- quantile(proposal$floor_probability,c(0,.25,.5,.75,1),names=FALSE)
    x <- iapf_draw_proposal(proposal)
    future <- if (time<horizon) iapf_log_integral(
      model$transition_mean(x,time+1),model$transition_covariance,twists[[time+1]]) else rep(0,N)
    weights <- previous + model$log_observation(x,model$observations[time,],time) +
      future - iapf_log_twist(x,twists[[time]])
    if (time==1) weights <- weights + proposal$log_integral
    iapf_assert(all(is.finite(weights)) && all(is.finite(x)),"nonfinite filter state/weight")
    # With future twists this is the twisted-model prefix, NOT the original
    # model's prefix likelihood. Only the terminal quantity targets its Z.
    prefixes[time] <- accumulated+iapf_logmean(weights)
    # Proposition 1's terminal test function 1/future untwists this prefix.
    original_prefix[time] <- accumulated+iapf_logmean(weights-future)
    ess[time] <- iapf_ess(weights)
    if (store_clouds) {
      clouds[[time]] <- x; weight_history[[time]] <- weights; ancestors[[time]] <- ancestor
    }
  }
  list(log_likelihood=accumulated+iapf_logmean(weights),clouds=clouds,
       log_weights=weight_history,ancestors=ancestors,ess=ess,resampled=resampled,
       resampling_count=sum(resampled),floor_probability_max=floor_max,
       floor_probability_quantiles=floor_quantiles,
       twisted_prefix=prefixes,prefix=original_prefix)
}

iapf_fit_objective <- function(parameters,points,log_targets,log_density_scale) {
  # Profile lambda analytically in equation (15). All scaling is FIXED per fit.
  d <- ncol(points); mean <- parameters[seq_len(d)]
  log_variance <- parameters[d+seq_len(d)]; variance <- exp(log_variance)
  residual_x <- sweep(points,2,mean)
  standardized_square <- sweep(residual_x^2,2,variance,"/")
  log_density <- -.5*(d*log(2*pi)+sum(log_variance)+rowSums(standardized_square))
  density <- exp(log_density-log_density_scale)
  targets <- exp(log_targets-max(log_targets))
  lambda <- sum(density*targets)/sum(targets^2)
  residual <- density-lambda*targets
  loss <- mean(residual^2)
  derivatives_mean <- sweep(residual_x,2,variance,"/")
  derivatives_variance <- .5*(standardized_square-1)
  multiplier <- 2*residual*density/nrow(points)
  gradient <- colSums(cbind(derivatives_mean,derivatives_variance)*multiplier)
  iapf_assert(is.finite(loss) && all(is.finite(gradient)),"nonfinite equation-15 objective")
  # Normalize ONLY the diagnostic, never the objective or its derivative.
  # Squared densities can underflow even for a valid Gaussian proposal.
  diagnostic_density <- exp(log_density-max(log_density))
  diagnostic_lambda <- sum(diagnostic_density*targets)/sum(targets^2)
  diagnostic_residual <- diagnostic_density-diagnostic_lambda*targets
  diagnostic_loss <- mean(diagnostic_residual^2)
  log_loss <- if(diagnostic_loss==0)-Inf else
    log(diagnostic_loss)+2*(max(log_density)-log_density_scale)
  list(value=loss,gradient=gradient,
       relative_residual=sum(diagnostic_residual^2)/sum(diagnostic_density^2),
       log_loss=log_loss,loss_underflow=loss==0 && diagnostic_loss>0,
       log_density_max=max(log_density),lambda_scaled=lambda)
}
iapf_fit_initial <- function(points,log_targets,strict_log_fit=FALSE) {
  reject <- function(message) stop(structure(list(message=message,call=NULL,
    points=points,log_targets=log_targets),class=c("iapf_fit_error","error","condition")))
  center <- colMeans(points)
  scale <- sqrt(colMeans(sweep(points,2,center)^2))
  if (strict_log_fit && (!all(is.finite(scale)) || any(scale<=0)))
    reject("degenerate log-quadratic fitting cloud")
  iapf_assert(all(is.finite(scale)) && all(scale>0),"degenerate fitting cloud")
  standardized <- sweep(sweep(points,2,center),2,scale,"/")
  design <- cbind(1,standardized,standardized^2)
  regression <- lm.fit(design,log_targets-max(log_targets))
  coefficients <- regression$coefficients
  d <- ncol(points); linear <- coefficients[1+seq_len(d)]
  quadratic <- coefficients[1+d+seq_len(d)]
  if (strict_log_fit && (regression$rank!=ncol(design) || !all(is.finite(coefficients))))
    reject("rank-deficient log-quadratic regression")
  if (strict_log_fit && any(quadratic>=0))
    reject("nonconcave log-quadratic regression")
  if (all(is.finite(coefficients)) &&
      all(quadratic < if(strict_log_fit)0 else -sqrt(.Machine$double.eps))) {
    variance <- -scale^2/(2*quadratic)
    mean <- center-scale*linear/(2*quadratic)
    method <- "diagonal_log_quadratic"
  } else {
    w <- exp(log_targets-max(log_targets)); w <- w/sum(w)
    mean <- colSums(points*w)
    variance <- colSums(sweep(points,2,mean)^2*w)
    iapf_assert(all(variance>0),"degenerate fallback initialization")
    method <- "target_weighted_moments"
  }
  log_fit <- if(strict_log_fit)list(loss=mean(regression$residuals^2),
    gradient_max=max(abs(2*crossprod(design,regression$residuals)/nrow(points))),
    design_rank=regression$rank,design_columns=ncol(design),
    design_condition=kappa(design,exact=TRUE)) else NULL
  list(parameters=c(mean,log(variance)),method=method,log_fit=log_fit)
}
iapf_relative_fit_objective <- function(parameters,points,log_targets) {
  # Explicit alternative to equation (15): remove the density amplitude escape.
  d <- ncol(points); mean <- parameters[seq_len(d)]
  log_variance <- parameters[d+seq_len(d)]; variance <- exp(log_variance)
  centered <- sweep(points,2,mean)
  squares <- sweep(centered^2,2,variance,"/")
  log_density <- -.5*(d*log(2*pi)+sum(log_variance)+rowSums(squares))
  density <- exp(log_density-max(log_density))
  targets <- exp(log_targets-max(log_targets))
  lambda <- sum(density*targets)/sum(targets^2)
  residual <- density-lambda*targets
  norm <- sum(density^2)
  loss <- sum(residual^2)/norm
  derivatives <- cbind(sweep(centered,2,variance,"/"),.5*(squares-1))
  gradient <- colSums(derivatives*(2*density*(residual-loss*density)/norm))
  iapf_assert(is.finite(loss) && all(is.finite(gradient)),"nonfinite relative-L2 objective")
  list(value=loss,gradient=gradient,relative_residual=loss,
    log_loss=log(loss),loss_underflow=FALSE,log_density_max=max(log_density),
    lambda_scaled=lambda)
}
iapf_fit_gaussian <- function(points,log_targets,N=nrow(points),maxit=200,
                              floor_tail_power=2,fit_mode="paper_eq15") {
  points <- as.matrix(points); d <- ncol(points)
  iapf_assert(length(log_targets)==nrow(points) && all(is.finite(log_targets)) &&
    N>=2 && is.finite(floor_tail_power) && floor_tail_power>0 &&
    all(is.finite(points)) &&
    fit_mode %in% c("paper_eq15","relative_l2","relative_l2_nlminb","log_quadratic"),"invalid fitting input")
  initial <- tryCatch(iapf_fit_initial(points,log_targets,fit_mode=="log_quadratic"),
    iapf_fit_error=function(e) { e$N <- N; e$fit_mode <- fit_mode; stop(e) })
  log_density_scale <- max(iapf_log_normal(points,initial$parameters[seq_len(d)],
    diag(exp(initial$parameters[d+seq_len(d)]),d)))
  # Avoid recomputing the same value/gradient pair for optim's two callbacks.
  previous_parameters <- NULL; previous_evaluation <- NULL
  evaluate <- function(parameters) {
    if (!identical(parameters,previous_parameters)) {
      previous_evaluation <<- if (fit_mode=="paper_eq15")
        iapf_fit_objective(parameters,points,log_targets,log_density_scale) else
        iapf_relative_fit_objective(parameters,points,log_targets)
      previous_parameters <<- parameters
    }
    previous_evaluation
  }
  initial_residual <- evaluate(initial$parameters)$relative_residual
  limit <- -log(.Machine$double.eps)
  optimizer_memory <- if(fit_mode=="relative_l2")max(5L,2L*d) else 5L
  if (fit_mode=="log_quadratic") {
    optimizer_memory <- NA_integer_
    fit <- list(par=initial$parameters,convergence=0L,counts=1L)
  } else if (fit_mode=="relative_l2_nlminb") {
    optimizer_memory <- NA_integer_
    fit <- nlminb(initial$parameters,function(p) evaluate(p)$value,
      gradient=function(p) evaluate(p)$gradient,
      lower=c(rep(-Inf,d),rep(-limit,d)),upper=c(rep(Inf,d),rep(limit,d)),
      control=list(iter.max=maxit,eval.max=2L*maxit,rel.tol=1e-10))
    fit$counts <- fit$evaluations
  } else fit <- optim(initial$parameters,function(p) evaluate(p)$value,
      function(p) evaluate(p)$gradient,method="L-BFGS-B",
      lower=c(rep(-Inf,d),rep(-limit,d)),upper=c(rep(Inf,d),rep(limit,d)),
      control=list(maxit=maxit,factr=10,pgtol=1e-8,lmm=optimizer_memory,
                   parscale=c(sqrt(exp(initial$parameters[d+seq_len(d)])),rep(1,d))))
  result <- evaluate(fit$par)
  iapf_assert(is.finite(result$relative_residual),"vanishing-density fit")
  mean <- fit$par[seq_len(d)]; log_variance <- fit$par[d+seq_len(d)]
  boundary <- any(abs(log_variance)>=limit-1e-6)
  failure <- if (boundary) "fit reached floating-point variance boundary" else
    if (fit$convergence!=0) paste(fit_mode,"fit did not converge") else
    if (result$loss_underflow) "equation-15 objective underflowed" else
    if (fit_mode!="paper_eq15" && result$relative_residual>initial_residual+1e-10)
      "relative-L2 fit worsened initial residual" else NULL
  if (!is.null(failure)) stop(structure(list(message=failure,
      call=NULL,parameters=fit$par,points=points,log_targets=log_targets,N=N,
      initial_parameters=initial$parameters,log_density_scale=log_density_scale,
      optimizer_code=fit$convergence,optimizer_message=fit$message,
      optimizer_evaluations=fit$counts,optimizer_iterations=fit$iterations,
      evaluation=result,fit_mode=fit_mode),
      class=c("iapf_fit_error","error","condition")))
  log_floor <- -.5*(d*log(2*pi)+sum(log_variance))-
    .5*qchisq(N^(-floor_tail_power),df=d,lower.tail=FALSE)
  twist <- iapf_gaussian_twist(mean,diag(exp(log_variance),d),log_floor)
  diagnostics <- list(convergence=fit$convergence,
      evaluations=unname(fit$counts[1]),loss=result$value,
      relative_residual=result$relative_residual,gradient_max=max(abs(result$gradient)),
      log_loss=result$log_loss,loss_underflow=result$loss_underflow,
      log_density_scale=log_density_scale,log_density_max=result$log_density_max,
      log_variance_min=min(log_variance),log_variance_max=max(log_variance),
      log_floor=log_floor,initialization=initial$method,boundary=boundary,
      fit_mode=fit_mode,initial_relative_residual=initial_residual,
      optimizer=if(fit_mode=="log_quadratic")"QR" else
        if(fit_mode=="relative_l2_nlminb")"nlminb" else "L-BFGS-B",
      optimizer_memory=optimizer_memory,optimizer_maxit=maxit)
  if (fit_mode=="log_quadratic") {
    diagnostics$relative_gradient_max <- diagnostics$gradient_max
    diagnostics$loss <- initial$log_fit$loss
    diagnostics$log_loss <- log(initial$log_fit$loss)
    diagnostics$gradient_max <- initial$log_fit$gradient_max
    diagnostics$design_rank <- initial$log_fit$design_rank
    diagnostics$design_columns <- initial$log_fit$design_columns
    diagnostics$design_condition <- initial$log_fit$design_condition
    diagnostics$optimizer_maxit <- NA_integer_
  }
  list(twist=twist,diagnostics=diagnostics)
}
iapf_backward_log_target <- function(model,points,time,next_twist=NULL) {
  target <- model$log_observation(points,model$observations[time,],time)
  if (time < model$horizon) {
    iapf_assert(!is.null(next_twist),"missing next-time twist")
    target <- target+iapf_log_integral(model$transition_mean(points,time+1),
      model$transition_covariance,next_twist)
  }
  target
}
iapf_fit_backward <- function(model,clouds,N,maxit=200,floor_tail_power=2,
                              fit_mode="paper_eq15") {
  twists <- vector("list",model$horizon); diagnostics <- vector("list",model$horizon)
  for (time in rev(seq_len(model$horizon))) {
    x <- clouds[[time]]
    next_twist <- if(time < model$horizon)twists[[time+1]] else NULL
    log_target <- iapf_backward_log_target(model,x,time,next_twist)
    fit <- tryCatch(if (fit_mode=="paper_eq15")
      iapf_fit_gaussian(x,log_target,N,maxit,floor_tail_power) else
      iapf_fit_gaussian(x,log_target,N,maxit,floor_tail_power,fit_mode),
      iapf_fit_error=function(e) {
        e$backward_time <- time; e$next_twist <- next_twist; e$model <- model
        e$floor_tail_power <- floor_tail_power; stop(e)
      })
    twists[[time]] <- fit$twist; diagnostics[[time]] <- fit$diagnostics
  }
  list(twists=twists,diagnostics=diagnostics)
}
iapf_likelihood_cv <- function(log_likelihoods,sd_mode="sample") {
  iapf_assert(sd_mode %in% c("sample","population"),"invalid SD convention")
  values <- exp(log_likelihoods-max(log_likelihoods))
  sd(values)/mean(values)*if(sd_mode=="population")sqrt((length(values)-1)/length(values)) else 1
}
iapf_iterate <- function(model,N0=1000,k=5,tau=.5,kappa=.5,max_iterations=20,
                         max_particles=16000,fit_maxit=200,floor_tail_power=2,
                         .filter=iapf_apf,.fit=iapf_fit_backward,
                         fit_mode="paper_eq15",doubling_mode="first_full_window",
                         stopping_window=NULL,cv_sd_mode="sample") {
  iapf_assert(k>=1 && k==as.integer(k) && tau>0 && max_iterations>k+1,
              "invalid iAPF controller")
  iapf_assert(fit_mode %in% c("paper_eq15","relative_l2","relative_l2_nlminb","log_quadratic") &&
    doubling_mode %in% c("first_full_window","after_k"),"invalid iAPF method identity")
  # Default is the paper's k+1 history. Other lengths are explicit diagnostics;
  # the earliest stopping iteration, doubling window and fresh final run stay fixed.
  if(is.null(stopping_window)) stopping_window <- k+1
  iapf_assert(length(stopping_window)==1 && is.finite(stopping_window) &&
    stopping_window==as.integer(stopping_window) && stopping_window>=2 &&
    stopping_window<=k+1,"invalid stopping window")
  twists <- replicate(model$horizon,iapf_constant_twist(),simplify=FALSE)
  history <- numeric(); counts <- numeric(); fits <- list(); filter_diagnostics <- list(); N <- N0
  for (iteration in 0:(max_iterations-1)) {
    if (N>max_particles) return(list(status="particle_cap",history=history,counts=counts,
      fits=fits,filter_diagnostics=filter_diagnostics))
    result <- .filter(model,twists,N,kappa,TRUE)
    filter_diagnostics[[length(filter_diagnostics)+1L]] <- list(iteration=iteration,
      particles=N,resampling_count=result$resampling_count,
      floor_probability_max=result$floor_probability_max,
      floor_probability_quantiles=result$floor_probability_quantiles)
    history <- c(history,result$log_likelihood); counts <- c(counts,N)
    # R positions are iteration+1; the paper's iteration is explicitly zero-based.
    if (iteration>k && iapf_likelihood_cv(tail(history,stopping_window),cv_sd_mode) < tau) {
      final <- .filter(model,twists,N,kappa,TRUE)
      return(list(status="complete",final=final,twists=twists,history=history,
                  counts=counts,fits=fits,filter_diagnostics=filter_diagnostics,
                  stop_iteration=iteration,final_particles=N))
    }
    fitted <- tryCatch(if (fit_mode=="paper_eq15")
      .fit(model,result$clouds,N,fit_maxit,floor_tail_power) else
      .fit(model,result$clouds,N,fit_maxit,floor_tail_power,fit_mode=fit_mode),
      iapf_fit_error=function(e) {
        e$iteration <- iteration; e$filter_twists <- twists
        e$particle_history <- counts; e$likelihood_history <- history
        e$filter_diagnostics <- filter_diagnostics
        e$kappa <- kappa; e$fit_maxit <- fit_maxit; stop(e)
      })
    twists <- fitted$twists; fits[[length(fits)+1]] <- fitted$diagnostics
    doubling_eligible <- if(doubling_mode=="first_full_window")iteration>=k else iteration>k
    if (doubling_eligible && counts[iteration-k+1]==N && any(diff(tail(history,k+1))<0)) N <- 2*N
  }
  list(status="iteration_cap",history=history,counts=counts,fits=fits,
    filter_diagnostics=filter_diagnostics)
}

iapf_linear_model <- function(observations,A,initial_mean=rep(0,ncol(A)),
    initial_covariance=diag(ncol(A)),Q=diag(ncol(A)),C=diag(ncol(A)),R=diag(nrow(C))) {
  force(A); force(C); force(R)
  model <- iapf_model(initial_mean,initial_covariance,
    function(x,time) x %*% t(A),Q,
    function(x,y,time) iapf_log_normal(matrix(rep(y,each=nrow(x)),nrow(x)),x %*% t(C),R),
    observations)
  model$A <- A; model$C <- C; model$R <- R
  model
}
iapf_paper_data <- function(d=5,horizon=100,alpha=.42,seed=2026092005) {
  set.seed(seed)
  A <- outer(seq_len(d),seq_len(d),function(i,j) alpha^(abs(i-j)+1))
  x <- rnorm(d); y <- matrix(NA_real_,horizon,d)
  for (time in seq_len(horizon)) {
    if (time>1) x <- as.vector(A %*% x)+rnorm(d)
    y[time,] <- x+rnorm(d)
  }
  list(observations=y,A=A,seed=seed)
}
iapf_kalman <- function(model) {
  mean <- model$initial_mean; covariance <- model$initial_covariance
  prefix <- innovation <- numeric(model$horizon); total <- 0
  for (time in seq_len(model$horizon)) {
    if (time>1) {
      mean <- as.vector(model$A %*% mean)
      covariance <- model$A %*% covariance %*% t(model$A)+model$transition_covariance
    }
    predictive_covariance <- model$C %*% covariance %*% t(model$C)+model$R
    residual <- model$observations[time,]-as.vector(model$C %*% mean)
    total <- total+iapf_log_normal(matrix(model$observations[time,],1),
      as.vector(model$C %*% mean),predictive_covariance)
    inverse <- chol2inv(iapf_chol(predictive_covariance))
    innovation[time] <- sum(residual*(inverse %*% residual))
    gain <- covariance %*% t(model$C) %*% inverse
    mean <- mean+as.vector(gain %*% residual)
    identity <- diag(model$dimension)-gain %*% model$C
    covariance <- identity %*% covariance %*% t(identity)+gain %*% model$R %*% t(gain)
    prefix[time] <- total
  }
  list(log_likelihood=total,prefix=prefix,innovation=innovation)
}
iapf_exact_twists <- function(model) {
  twists <- vector("list",model$horizon)
  observation_precision <- t(model$C) %*% solve(model$R,model$C)
  for (time in rev(seq_len(model$horizon))) {
    precision <- observation_precision
    natural_mean <- t(model$C) %*% solve(model$R,model$observations[time,])
    if (time < model$horizon) {
      next_twist <- twists[[time+1]]
      inverse <- solve(model$transition_covariance+next_twist$covariance)
      precision <- precision+t(model$A) %*% inverse %*% model$A
      natural_mean <- natural_mean+t(model$A) %*% inverse %*% next_twist$mean
    }
    covariance <- solve(precision)
    twists[[time]] <- iapf_gaussian_twist(as.vector(covariance %*% natural_mean),covariance)
  }
  twists
}
iapf_observation_twists <- function(model) {
  covariance <- solve(t(model$C) %*% solve(model$R,model$C))
  lapply(seq_len(model$horizon),function(time) iapf_gaussian_twist(
    as.vector(covariance %*% t(model$C) %*% solve(model$R,model$observations[time,])),covariance))
}
iapf_fully_adapted <- function(model,N=5000,kappa=.5) {
  # Fully adapted auxiliary filter: sample ancestors using predictive likelihood,
  # then sample the exact conditional Gaussian. Carry weights if no resampling.
  x <- NULL; total <- 0; prefix <- numeric(model$horizon)
  weights <- rep(0,N); resampling_count <- 0
  for (time in seq_len(model$horizon)) {
    means <- if (time==1) matrix(rep(model$initial_mean,each=N),N) else model$transition_mean(x,time)
    covariance <- if (time==1) model$initial_covariance else model$transition_covariance
    predictive_covariance <- model$C %*% covariance %*% t(model$C)+model$R
    observation <- matrix(rep(model$observations[time,],each=N),N)
    predicted <- means %*% t(model$C)
    log_predictive <- iapf_log_normal(observation,predicted,predictive_covariance)
    if (time==1) {
      total <- log_predictive[1]
    } else {
      weights <- weights+log_predictive
    }
    if (time>1 && iapf_ess(weights)<=kappa*N) {
      total <- total+iapf_logmean(weights)
      ancestors <- sample.int(N,N,replace=TRUE,prob=exp(weights-max(weights)))
      means <- means[ancestors,,drop=FALSE]; predicted <- predicted[ancestors,,drop=FALSE]
      weights <- rep(0,N); resampling_count <- resampling_count+1
    }
    gain <- covariance %*% t(model$C) %*% solve(predictive_covariance)
    posterior_means <- means+(observation-predicted) %*% t(gain)
    posterior_covariance <- covariance-gain %*% model$C %*% covariance
    x <- iapf_draw_normal(posterior_means,(posterior_covariance+t(posterior_covariance))/2)
    prefix[time] <- total+iapf_logmean(weights)
  }
  list(log_likelihood=total+iapf_logmean(weights),resampling_count=resampling_count,prefix=prefix,
       resampling_policy="fully_adapted_auxiliary_ESS")
}

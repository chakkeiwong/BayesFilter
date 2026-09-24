# Linear-Gaussian diagnostic oracles; never a fitting or runtime fallback.
iapf_linear_predictive_marginals <- function(model) {
  mean <- model$initial_mean; covariance <- model$initial_covariance
  answer <- vector("list",model$horizon)
  for(time in seq_len(model$horizon)) {
    if(time>1) {
      mean <- as.vector(model$A %*% mean)
      covariance <- model$A %*% covariance %*% t(model$A)+model$transition_covariance
    }
    answer[[time]] <- list(mean=mean,covariance=covariance)
    precision <- chol2inv(iapf_chol(covariance))+t(model$C) %*% solve(model$R,model$C)
    natural <- solve(covariance,mean)+t(model$C) %*% solve(model$R,model$observations[time,])
    covariance <- chol2inv(iapf_chol(precision))
    mean <- as.vector(covariance %*% natural)
  }
  answer
}
iapf_linear_smoothing_marginal <- function(model,time,predictions=NULL,exact=NULL) {
  if(is.null(predictions))predictions <- iapf_linear_predictive_marginals(model)
  if(is.null(exact))exact <- iapf_exact_twists(model)
  prior <- predictions[[time]]
  proposal <- iapf_proposal(matrix(prior$mean,1),prior$covariance,exact[[time]])
  list(mean=as.vector(proposal$means),covariance=proposal$covariance)
}
iapf_backward_target_mixture <- function(model,time,next_twist=NULL) {
  # g(y|x) is proportional to this Gaussian when C has full column rank.
  covariance <- chol2inv(iapf_chol(t(model$C) %*% solve(model$R,model$C)))
  mean <- as.vector(covariance %*% t(model$C) %*% solve(model$R,model$observations[time,]))
  if(time==model$horizon || (!is.null(next_twist) && next_twist$constant))
    return(list(means=matrix(mean,1),covariances=list(covariance),probabilities=1,
                mean=mean,covariance=covariance))
  iapf_assert(!is.null(next_twist),"missing next-time twist")
  noise <- model$transition_covariance+next_twist$covariance
  predictive <- noise+model$A %*% covariance %*% t(model$A)
  gain <- covariance %*% t(model$A) %*% chol2inv(iapf_chol(predictive))
  gaussian_mean <- as.vector(mean+gain %*% (next_twist$mean-model$A %*% mean))
  identity <- diag(model$dimension)-gain %*% model$A
  gaussian_covariance <- identity %*% covariance %*% t(identity)+gain %*% noise %*% t(gain)
  log_mass <- iapf_log_normal(matrix(next_twist$mean,1),as.vector(model$A %*% mean),predictive)
  log_total <- iapf_logadd(log_mass,next_twist$log_floor)
  probabilities <- exp(c(log_mass,next_twist$log_floor)-log_total)
  means <- rbind(gaussian_mean,mean)
  mixture_mean <- colSums(means*probabilities)
  covariances <- list(gaussian_covariance,covariance)
  mixture_covariance <- matrix(0,model$dimension,model$dimension)
  for(i in 1:2) {
    delta <- means[i,]-mixture_mean
    mixture_covariance <- mixture_covariance+probabilities[i]*(covariances[[i]]+tcrossprod(delta))
  }
  list(means=means,covariances=covariances,probabilities=probabilities,
       mean=mixture_mean,covariance=mixture_covariance)
}
iapf_support_metrics <- function(log_fit,log_target) {
  iapf_assert(all(is.finite(log_fit)) && all(is.finite(log_target)),"invalid support diagnostic")
  p <- exp(log_fit-max(log_fit)); y <- exp(log_target-max(log_target))
  coefficient <- sum(p*y)/sum(y^2)
  list(relative_residual=sum((p-coefficient*y)^2)/sum(p^2),
       target_squared_effective_points=sum(y^2)^2/sum(y^4),
       fitted_squared_effective_points=sum(p^2)^2/sum(p^4),
       log_ratio_sd=sd(log_fit-log_target),
       log_ratio_range=diff(range(log_fit-log_target)))
}
iapf_log_gaussian_product_expectation <- function(reference,mean_a,covariance_a,mean_b,covariance_b) {
  product <- iapf_proposal(matrix(mean_a,1),covariance_a,
    iapf_gaussian_twist(mean_b,covariance_b))
  product$log_integral+iapf_log_normal(matrix(reference$mean,1),
    as.vector(product$means),reference$covariance+product$covariance)
}
iapf_exact_support_risk <- function(reference,candidate,target) {
  # Integrate p^2, p*y and y^2 against the exact Gaussian reference measure.
  k <- length(target$probabilities); log_mass <- log(target$probabilities)
  log_y <- vapply(seq_len(k),function(i)iapf_log_normal(matrix(reference$mean,1),
    target$means[i,],reference$covariance+target$covariances[[i]]),0.0)
  yy <- numeric(k*k); at <- 0
  for(i in seq_len(k))for(j in seq_len(k)) {
    at <- at+1
    yy[at] <- log_mass[i]+log_mass[j]+iapf_log_gaussian_product_expectation(reference,
      target$means[i,],target$covariances[[i]],target$means[j,],target$covariances[[j]])
  }
  log_yy <- iapf_logsum(yy)
  if(candidate$constant) {
    log_pp <- 0; log_py <- iapf_logsum(log_mass+log_y)
  } else {
    log_p <- iapf_log_normal(matrix(reference$mean,1),candidate$mean,
      reference$covariance+candidate$covariance)
    log_pp <- iapf_logsum(c(iapf_log_gaussian_product_expectation(reference,
      candidate$mean,candidate$covariance,candidate$mean,candidate$covariance),
      log(2)+candidate$log_floor+log_p,2*candidate$log_floor))
    py <- vapply(seq_len(k),function(i)iapf_log_gaussian_product_expectation(reference,
      candidate$mean,candidate$covariance,target$means[i,],target$covariances[[i]]),0.0)
    log_py <- iapf_logsum(c(log_mass+py,log_mass+candidate$log_floor+log_y))
  }
  log_cosine_squared <- 2*log_py-log_pp-log_yy
  tolerance <- 64*.Machine$double.eps*max(1,abs(c(log_py,log_pp,log_yy)))
  iapf_assert(is.finite(log_cosine_squared) && log_cosine_squared<=tolerance,
    "invalid continuous fitting-risk integral")
  list(relative_residual=-expm1(log_cosine_squared),log_pp=log_pp,log_py=log_py,log_yy=log_yy)
}

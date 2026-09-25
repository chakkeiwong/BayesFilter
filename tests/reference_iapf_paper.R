# Executable independent-reference conformance tests; base R, CPU only.
args <- commandArgs(trailingOnly=TRUE)
source(if(length(args)>=3) args[3] else file.path(args[1],"docs/benchmarks/reference_iapf_paper.R"))
checks <- list()
check <- function(name,condition) {
  checks[[length(checks)+1]] <<- data.frame(name=name,passed=isTRUE(condition))
  if (!isTRUE(condition)) stop(paste("FAILED",name),call.=FALSE)
}
close <- function(name,actual,expected,tolerance=1e-9) check(name,
  length(actual)==length(expected) && all(is.finite(actual)) &&
    all(abs(actual-expected)<=tolerance*(1+abs(expected))))
expect_error <- function(name,expression) check(name,
  inherits(tryCatch(force(expression),error=function(e)e),"error"))

set.seed(52001)
twist <- iapf_gaussian_twist(.4,matrix(.7,1,1),log(.03))
means <- matrix(c(-.7,0,.8),ncol=1); Q <- matrix(1.2,1,1)
proposal <- iapf_proposal(means,Q,twist)
normalizer <- dnorm(as.vector(means),.4,sqrt(1.9))+.03
close("floor integral",exp(proposal$log_integral),normalizer)
close("floor mixture probability",proposal$floor_probability,.03/normalizer)
close("Gaussian product mean",proposal$means,(as.vector(means)/1.2+.4/.7)/(1/1.2+1/.7))
close("Gaussian product variance",proposal$covariance,1/(1/1.2+1/.7))
grid <- matrix(c(-1,.2,2),ncol=1)
close("actual mixture density",exp(iapf_log_proposal(grid,proposal)),
  dnorm(as.vector(grid),as.vector(means),sqrt(1.2))*(dnorm(as.vector(grid),.4,sqrt(.7))+.03)/normalizer)
numerical_integral <- integrate(function(x) dnorm(x,-.7,sqrt(1.2))*(dnorm(x,.4,sqrt(.7))+.03),-Inf,Inf)$value
close("quadrature mixture normalization",exp(proposal$log_integral[1]),numerical_integral)
close("ESS invariant to shift",iapf_ess(log(c(.2,.3,.5))-1000),1/sum(c(.2,.3,.5)^2))
check("ESS equality resamples",iapf_ess(rep(-1000,4))<=4)

data <- iapf_paper_data(2,4,seed=52002)
model <- iapf_linear_model(data$observations,data$A)
kalman <- iapf_kalman(model)
# Independent joint-observation covariance assembled directly from latent states.
d <- 2; horizon <- 4; state_cov <- matrix(0,d*horizon,d*horizon)
indices <- function(t) ((t-1)*d+1):(t*d)
for (t in seq_len(horizon)) {
  it <- indices(t)
  state_cov[it,it] <- if(t==1) diag(d) else data$A %*% state_cov[indices(t-1),indices(t-1)] %*% t(data$A)+diag(d)
  if(t>1) for(s in seq_len(t-1)) {
    state_cov[it,indices(s)] <- data$A %*% state_cov[indices(t-1),indices(s)]
    state_cov[indices(s),it] <- t(state_cov[it,indices(s)])
  }
}
joint_cov <- state_cov+diag(d*horizon)
y <- as.vector(t(data$observations))
joint_log <- -.5*(length(y)*log(2*pi)+as.numeric(determinant(joint_cov,logarithm=TRUE)$modulus)+
                    sum(y*solve(joint_cov,y)))
close("Kalman versus joint Gaussian likelihood",kalman$log_likelihood,joint_log)
source(file.path(args[1],"docs/benchmarks/reference_iapf_support_diagnostics.R"))
for(time in seq_len(horizon)) {
  oracle <- iapf_linear_smoothing_marginal(model,time)
  cross <- state_cov[indices(time),,drop=FALSE]
  close(paste("smoothing mean versus joint Gaussian",time),oracle$mean,
    as.vector(cross %*% solve(joint_cov,y)))
  close(paste("smoothing covariance versus joint Gaussian",time),oracle$covariance,
    state_cov[indices(time),indices(time)]-cross %*% solve(joint_cov,t(cross)))
}
exact <- iapf_exact_twists(model)
for (kappa in c(0,.5,1)) {
  run <- iapf_apf(model,exact,17,kappa)
  close(paste("exact twist likelihood",kappa),run$log_likelihood,joint_log)
  check(paste("exact twist finite weights",kappa),all(vapply(run$log_weights,function(x)all(is.finite(x)),TRUE)))
}
single <- iapf_linear_model(matrix(c(.3,-.2),1),diag(2)*.4)
single_run <- iapf_apf(single,iapf_exact_twists(single),11)
close("one observation",single_run$log_likelihood,sum(dnorm(c(.3,-.2),0,sqrt(2),log=TRUE)))
for(kappa in c(0,.5,1)) {
  set.seed(52004)
  observation_only <- iapf_apf(model,iapf_observation_twists(model),51,kappa)
  set.seed(52004)
  fully_adapted <- iapf_fully_adapted(model,51,kappa)
  close(paste("fully adapted likelihood parity",kappa),fully_adapted$log_likelihood,
        observation_only$log_likelihood)
  close(paste("fully adapted prefix parity",kappa),fully_adapted$prefix,observation_only$prefix)
  close(paste("fully adapted resampling parity",kappa),fully_adapted$resampling_count,
        observation_only$resampling_count)
}

# Nonideal twists, positive floors, prescribed actual draws: independently
# accumulate density ratios through the actual APF for both resampling branches.
model1 <- iapf_linear_model(matrix(c(.1,.8,-.2),ncol=1),matrix(.42,1,1))
twists <- list(iapf_gaussian_twist(.3,matrix(.7,1,1),log(.01)),
  iapf_gaussian_twist(-.4,matrix(1.2,1,1),log(.02)),
  iapf_gaussian_twist(.2,matrix(.9,1,1),log(.03)))
path <- c(.2,-.5,.7); log_joint <- 0; log_twisted <- 0
for(t in 1:3) {
  mean <- if(t==1) 0 else .42*path[t-1]
  p <- iapf_proposal(matrix(mean,1,1),matrix(1,1,1),twists[[t]])
  future <- if(t==3) 0 else log(dnorm(.42*path[t],twists[[t+1]]$mean,sqrt(1+twists[[t+1]]$covariance[1]))+exp(twists[[t+1]]$log_floor))
  incremental <- dnorm(model1$observations[t],path[t],1,log=TRUE)+future-iapf_log_twist(matrix(path[t],1,1),twists[[t]])
  if(t==1) incremental <- incremental+p$log_integral
  log_twisted <- log_twisted+iapf_log_proposal(matrix(path[t],1,1),p)+incremental
  log_joint <- log_joint+dnorm(path[t],mean,1,log=TRUE)+dnorm(model1$observations[t],path[t],1,log=TRUE)
}
close("positive-floor path telescoping",log_twisted,log_joint)
for(kappa in c(0,1)) {
  set.seed(52003)
  run <- iapf_apf(model1,twists,5,kappa)
  independent_weights <- rep(0,5); accumulated <- 0
  for(t in 1:3) {
    if(t>1 && run$resampled[t]) {
      accumulated <- accumulated+log(mean(exp(independent_weights)))
      independent_weights <- rep(0,5)
    }
    x <- as.vector(run$clouds[[t]])
    future <- if(t==3) rep(1,5) else dnorm(.42*x,twists[[t+1]]$mean,
      sqrt(1+twists[[t+1]]$covariance[1]))+exp(twists[[t+1]]$log_floor)
    psi <- dnorm(x,twists[[t]]$mean,sqrt(twists[[t]]$covariance[1]))+exp(twists[[t]]$log_floor)
    incremental <- dnorm(model1$observations[t],x,1)*future/psi
    if(t==1) incremental <- incremental*(dnorm(0,twists[[1]]$mean,sqrt(1+twists[[1]]$covariance[1]))+exp(twists[[1]]$log_floor))
    independent_weights <- independent_weights+log(incremental)
    close(paste("actual APF weight",kappa,t),run$log_weights[[t]],independent_weights)
    close(paste("untwisted prefix",kappa,t),run$prefix[t],
      accumulated+log(mean(exp(independent_weights)/future)))
  }
  close(paste("nonideal likelihood",kappa),run$log_likelihood,accumulated+log(mean(exp(independent_weights))))
  check(paste("resampling branch",kappa),run$resampling_count==if(kappa==0)0 else 2)
}

points <- as.matrix(expand.grid(seq(-2,2,length.out=9),seq(-1.5,2.5,length.out=9)))
target <- -.5*((points[,1]-.3)^2/.8+(points[,2]+.2)^2/1.2)
parameters <- c(.1,-.4,log(c(.6,1.1)))
evaluation <- iapf_fit_objective(parameters,points,target,0)
p <- dnorm(points[,1],.1,sqrt(.6))*dnorm(points[,2],-.4,sqrt(1.1))
yt <- exp(target-max(target)); lambda <- sum(p*yt)/sum(yt^2)
close("equation 15 profiled loss",evaluation$value,mean((p-lambda*yt)^2))
fd <- sapply(seq_along(parameters),function(i) {
  plus <- minus <- parameters; plus[i] <- plus[i]+1e-5; minus[i] <- minus[i]-1e-5
  (iapf_fit_objective(plus,points,target,0)$value-iapf_fit_objective(minus,points,target,0)$value)/2e-5
})
close("equation 15 analytic gradient",evaluation$gradient,fd,1e-7)
shifted <- iapf_fit_objective(parameters,points,target-10000,0)
close("target scale invariance",shifted$value,evaluation$value)
scaled <- iapf_fit_objective(parameters,points,target,2)
close("fixed density scaling",scaled$value,evaluation$value*exp(-4))
close("healthy relative diagnostic",evaluation$relative_residual,
  sum((p-lambda*yt)^2)/sum(p^2))
check("healthy loss underflow flag",!evaluation$loss_underflow)
tiny <- iapf_fit_objective(c(40,-40,log(c(.6,1.1))),points,target,0)
check("vanishing density remains observable",is.finite(tiny$relative_residual) &&
  tiny$relative_residual>=0 && tiny$relative_residual<=1+1e-12 && is.finite(tiny$log_loss))
check("vanishing objective explicitly flagged",tiny$value==0 && tiny$loss_underflow)
fit <- iapf_fit_gaussian(points,target)
close("known Gaussian fitted mean",fit$twist$mean,c(.3,-.2),1e-6)
close("known Gaussian fitted variance",diag(fit$twist$covariance),c(.8,1.2),1e-6)
check("equation 16 positive floor",is.finite(fit$twist$log_floor) && fit$twist$log_floor<0)

# Error observability must preserve the actual failed fit and recursive time.
original_optim <- optim
optim <- function(par,...) list(par=c(par[1:2],rep(-log(.Machine$double.eps),2)),
  convergence=0,counts=setNames(1,"function"))
failure <- tryCatch(iapf_fit_gaussian(points,target),error=function(e)e)
check("boundary failure preserves fitting inputs",inherits(failure,"iapf_fit_error") &&
  identical(failure$points,points) && identical(failure$log_targets,target) &&
  length(failure$parameters)==4)
failure <- tryCatch(iapf_fit_backward(model,replicate(4,points,simplify=FALSE),nrow(points)),
  error=function(e)e)
check("recursive failure preserves backward time",inherits(failure,"iapf_fit_error") &&
  failure$backward_time==4 && failure$N==nrow(points))
optim <- function(par,...) list(par=par,convergence=1,counts=setNames(201,"function"))
failure <- tryCatch(iapf_fit_gaussian(points,target),error=function(e)e)
check("equation 15 rejects nonconvergence",inherits(failure,"iapf_fit_error") &&
  failure$optimizer_code==1 && identical(failure$points,points))
optim <- function(par,...) list(par=c(40,-40,log(c(.6,1.1))),
  convergence=0,counts=setNames(1,"function"))
failure <- tryCatch(iapf_fit_gaussian(points,target),error=function(e)e)
check("equation 15 rejects underflow false optimum",inherits(failure,"iapf_fit_error") &&
  failure$evaluation$loss_underflow)
optim <- original_optim

# Actual recursive fitter sees observation likelihood times the next integral.
original_fit <- iapf_fit_gaussian
captured <- list(); prescribed <- twists
iapf_fit_gaussian <- function(points,log_targets,N,maxit,floor_tail_power) {
  time <- 4-length(captured)-1
  captured[[length(captured)+1]] <<- log_targets
  list(twist=prescribed[[time]],diagnostics=list(convergence=0))
}
clouds <- replicate(3,matrix(c(-1,-.2,.4,1.3),ncol=1),simplify=FALSE)
backward <- iapf_fit_backward(model1,clouds,4)
iapf_fit_gaussian <- original_fit
for(t in 1:3) {
  x <- as.vector(clouds[[t]])
  expected <- dnorm(model1$observations[t],x,1,log=TRUE)
  if(t<3) expected <- expected+log(dnorm(.42*x,twists[[t+1]]$mean,sqrt(1+twists[[t+1]]$covariance[1]))+exp(twists[[t+1]]$log_floor))
  close(paste("Algorithm 3 target",t),captured[[4-t]],expected)
}

# Exercise the full failure call chain and evaluate its saved target elsewhere.
iapf_fit_gaussian <- function(points,log_targets,N,maxit,floor_tail_power) {
  if (identical(log_targets,model1$log_observation(points,model1$observations[3,],3)))
    return(list(twist=twists[[3]],diagnostics=list(convergence=0)))
  stop(structure(list(message="injected second backward fit failure",call=NULL,
    points=points,log_targets=log_targets,N=N),class=c("iapf_fit_error","error","condition")))
}
set.seed(731)
failure <- tryCatch(iapf_iterate(model1,N0=20,k=1,max_iterations=5),error=function(e)e)
iapf_fit_gaussian <- original_fit
check("failure records actual controller context",inherits(failure,"iapf_fit_error") &&
  failure$iteration==0 && failure$backward_time==2 && failure$particle_history[1]==20 &&
  length(failure$filter_twists)==3 && identical(failure$next_twist,twists[[3]]))
close("saved failure reconstructs training target",
  iapf_backward_log_target(failure$model,failure$points,2,failure$next_twist),failure$log_targets)
independent <- matrix(c(-3,-.7,.6,2.5),ncol=1)
expected <- dnorm(model1$observations[2],independent[,1],1,log=TRUE)+
  log(dnorm(.42*independent[,1],twists[[3]]$mean,sqrt(1+twists[[3]]$covariance[1]))+
    exp(twists[[3]]$log_floor))
close("saved failure target on independent points",
  iapf_backward_log_target(failure$model,independent,2,failure$next_twist),expected)
mixture <- iapf_backward_target_mixture(model1,2,twists[[3]])
grid <- matrix(seq(-15,15,length.out=10001),ncol=1)
target_density <- dnorm(model1$observations[2],grid[,1],1)*
  (dnorm(.42*grid[,1],twists[[3]]$mean,sqrt(1+twists[[3]]$covariance[1]))+
     exp(twists[[3]]$log_floor))
normalized <- target_density/sum(target_density)
mean_grid <- sum(grid[,1]*normalized)
close("backward mixture mean versus quadrature",mixture$mean,mean_grid,1e-7)
close("backward mixture covariance versus quadrature",mixture$covariance,
      sum((grid[,1]-mean_grid)^2*normalized),1e-7)
mixture_density <- rowSums(vapply(1:2,function(i)mixture$probabilities[i]*
  dnorm(grid[,1],mixture$means[i,1],sqrt(mixture$covariances[[i]][1])),numeric(nrow(grid))))
close("backward mixture density versus actual target",mixture_density,
  normalized/(grid[2,1]-grid[1,1]),1e-7)
reference <- list(mean=.6,covariance=matrix(.9,1,1))
nu <- dnorm(grid[,1],reference$mean,sqrt(reference$covariance[1]))
for(candidate in list(iapf_constant_twist(),twists[[1]],
                     iapf_gaussian_twist(.3,matrix(.7,1,1)))) {
  p <- exp(iapf_log_twist(grid,candidate)); y <- mixture_density
  expected <- 1-sum(nu*p*y)^2/(sum(nu*p^2)*sum(nu*y^2))
  risk <- iapf_exact_support_risk(reference,candidate,mixture)
  close(paste("continuous risk versus quadrature",candidate$constant,
    if(is.null(candidate$log_floor))"constant" else candidate$log_floor),
    risk$relative_residual,expected,1e-7)
}
# A perfect finite-cloud residual can coexist with a poor continuous shape.
training <- c(0,50,100)
in_sample <- iapf_support_metrics(dnorm(training,0,.1,log=TRUE),dnorm(training,0,1,log=TRUE))
elsewhere <- seq(-2,2,length.out=501)
out_sample <- iapf_support_metrics(dnorm(elsewhere,0,.1,log=TRUE),dnorm(elsewhere,0,1,log=TRUE))
check("single-point fit cannot certify shape",in_sample$relative_residual==0 &&
  in_sample$target_squared_effective_points==1 && out_sample$relative_residual>.5)

# Dependency injection executes the actual controller with prescribed histories.
for(case in c("stable","oscillating")) {
  # The post-stop draw deliberately differs from the stopping estimate.
  history <- if(case=="stable")c(rep(0,4),log(7),rep(0,5)) else c(0,1,0,1,1,1,log(11),1)
  calls <- 0; fit_calls <- 0; seen_counts <- numeric()
  filter <- function(model,twists,N,kappa,store_clouds) {
    calls <<- calls+1; seen_counts <<- c(seen_counts,N)
    list(log_likelihood=history[calls],clouds=list())
  }
  fitter <- function(model,clouds,N,maxit,floor_tail_power) {
    fit_calls <<- fit_calls+1
    list(twists=list(iapf_constant_twist()),diagnostics=list())
  }
  iterated <- iapf_iterate(single,8,k=2,tau=.1,max_iterations=10,.filter=filter,.fit=fitter)
  check(paste("controller complete",case),iterated$status=="complete")
  check(paste("fresh final evaluation",case),calls==length(iterated$history)+1)
  close(paste("returned fresh estimate",case),iterated$final$log_likelihood,
        if(case=="stable")log(7) else log(11))
  check(paste("controller fit count",case),fit_calls==iterated$stop_iteration)
  if(case=="stable") check("paper zero-based stop",iterated$stop_iteration==3)
  else close("particle doubling schedule",seen_counts,c(8,8,8,16,16,16,16))
}
expect_error("negative covariance fails",iapf_gaussian_twist(0,matrix(-1,1,1)))
expect_error("invalid ESS threshold fails",iapf_apf(single,iapf_exact_twists(single),5,1.1))
write.csv(do.call(rbind,checks),args[2],row.names=FALSE)
cat(length(checks),"R conformance checks passed\n")

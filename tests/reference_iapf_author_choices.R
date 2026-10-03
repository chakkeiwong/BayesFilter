# Focused mathematical and consumer tests for independent CPU R hypotheses.
args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args))args[1] else getwd()
original_root <- if(length(args)>=2)args[2] else root
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_author_choices.R"))
near <- function(a,b,tol=1e-9)stopifnot(max(abs(a-b))<tol)
set.seed(9122001); x <- matrix(rnorm(1600),800,2)
target <- iapf_log_normal(x,c(.2,-.3),diag(c(.8,1.2)))+4
for(arm in c("qr","f1_strict","f1_loose","f2_strict","f2_loose","wlog1","wlog2")) {
  fit <- iapf_choice_fit(x,target,arm=arm)
  near(fit$twist$mean,c(.2,-.3)); near(diag(fit$twist$covariance),c(.8,1.2))
  shifted <- iapf_choice_fit(x,target+317,arm=arm)
  near(fit$parameters,shifted$parameters,1e-8)
}
cat("PASS exact Gaussian and target-scale invariance for all seven arms\n")
p <- c(.1,.2,log(.7),log(1.1)); scale <- max(iapf_log_normal(x,c(.1,.2),diag(c(.7,1.1))))
ev <- iapf_fit_objective(p,x,target,scale)
fd <- sapply(seq_along(p),function(j) {step <- rep(0,length(p));step[j] <- 1e-5
  (iapf_fit_objective(p+step,x,target,scale)$value-
   iapf_fit_objective(p-step,x,target,scale)$value)/2e-5})
near(ev$gradient,fd,1e-8)
cat("PASS Equation15 analytic gradient\n")
for(rule in c("tail8","peak2","peak4","peak8")) {
  fit <- iapf_choice_fit(x,target,arm="qr",floor_rule=rule)
  proposal <- iapf_proposal(matrix(c(0,0,.2,-.4),2,2,byrow=TRUE),diag(2),fit$twist)
  log_mass <- iapf_log_normal(matrix(c(0,0,.2,-.4),2,2,byrow=TRUE),
    fit$twist$mean,diag(2)+fit$twist$covariance)
  near(proposal$floor_probability,plogis(fit$twist$log_floor-log_mass),1e-12)
}
near(iapf_likelihood_cv(log(1:6),"population"),
  iapf_likelihood_cv(log(1:6),"sample")*sqrt(5/6),1e-14)
cat("PASS floor mixture identities and SD convention\n")
model <- iapf_linear_model(matrix(.3,1,1),matrix(.42,1,1))
set.seed(21); warm <- iapf_choice_run(model,"f2_loose",N0=40,k=1,max_iterations=5)
stopifnot(warm$status=="complete",length(warm$fits)>=2,
  warm$fits[[1]][[1]]$start_source=="qr",
  warm$fits[[2]][[1]]$start_source=="previous_iteration")
near(warm$fits[[2]][[1]]$start_parameters,c(.3,0))
for(arm in c("wlog1","wlog2")) {
  set.seed(22); value <- iapf_choice_run(model,arm,N0=40,k=1,max_iterations=5)
  stopifnot(value$status=="complete",
    startsWith(value$fits[[1]][[1]]$objective,"weighted_log_quadratic"))
}
cat("PASS actual shared-controller warm-start and weighted-fit wiring\n")
calls <- 0L
fake_filter <- function(model,twists,N,kappa,store_clouds) {
  calls <<- calls+1L
  list(log_likelihood=log(c(1,2,1.1,2.1,1.2,2.2,1.3,2.3,1.4,2.4)[calls]))
}
fake_fit <- function(model,clouds,N,maxit,floor_tail_power)
  list(twists=replicate(model$horizon,iapf_constant_twist(),simplify=FALSE),diagnostics=list())
# A threshold strictly between the two CVs makes the consumer branch observable.
v <- log(c(2,1.1,2.1,1.2,2.2,1.3)); threshold <-
  mean(c(iapf_likelihood_cv(v,"sample"),iapf_likelihood_cv(v,"population")))
calls <- 0L; pop <- iapf_iterate(model,k=5,tau=threshold,max_iterations=8,
  .filter=fake_filter,.fit=fake_fit,doubling_mode="after_k",cv_sd_mode="population")
calls <- 0L; sam <- iapf_iterate(model,k=5,tau=threshold,max_iterations=8,
  .filter=fake_filter,.fit=fake_fit,doubling_mode="after_k",cv_sd_mode="sample")
stopifnot(pop$status=="complete",pop$stop_iteration==6,
  sam$status!="complete" || sam$stop_iteration>6)
cat("PASS SD option reaches the controller stopping decision\n")
legacy <- new.env(parent=globalenv())
sys.source(file.path(original_root,"docs/plans/artifacts/iapf-r-equation15-resolution-20260921-01/source-v1/docs/benchmarks/reference_iapf_paper.R"),legacy)
dat <- iapf_paper_data(2,4,seed=9881002)
model <- iapf_linear_model(dat$observations,dat$A)
set.seed(77); old <- legacy$iapf_iterate(model,N0=100,k=1,max_iterations=5,
  fit_mode="log_quadratic",floor_tail_power=8,doubling_mode="after_k")
set.seed(77); new <- iapf_iterate(model,N0=100,k=1,max_iterations=5,
  fit_mode="log_quadratic",floor_tail_power=8,doubling_mode="after_k")
stopifnot(old$status=="complete",new$status=="complete",identical(old$history,new$history),
  identical(old$final$prefix,new$final$prefix),
  identical(old$final$clouds,new$final$clouds),
  all(new$final$floor_probability_quantiles>=0 & new$final$floor_probability_quantiles<=1))
cat("PASS unchanged sample-SD/default numerical output against preserved core\n")
stopifnot(length(new$filter_diagnostics)==length(new$history),
  all(vapply(new$filter_diagnostics,function(z)
    identical(dim(z$floor_probability_quantiles),c(4L,5L)),logical(1))))
fail_fit <- function(...)stop(structure(list(message="injected fit failure",call=NULL),
  class=c("iapf_fit_error","error","condition")))
set.seed(77); failure <- tryCatch(iapf_iterate(model,N0=100,k=1,max_iterations=5,
  .fit=fail_fit),iapf_fit_error=function(e)e)
stopifnot(inherits(failure,"iapf_fit_error"),failure$iteration==0,
  length(failure$filter_diagnostics)==1,
  identical(dim(failure$filter_diagnostics[[1]]$floor_probability_quantiles),c(4L,5L)),
  all(is.finite(failure$filter_diagnostics[[1]]$floor_probability_quantiles)))
cat("PASS learning diagnostics survive successful and failed controller exits\n")
fixture <- readRDS(file.path(original_root,
  "docs/plans/artifacts/iapf-r-author-choice-hypotheses-20260922-01/runs",
  "floor_probe-d20-j2-wlog1-peak4-sample/results/wlog1-3-failure.rds"))
corrected <- iapf_choice_fit(fixture$points,fixture$log_targets,
  N=nrow(fixture$points),arm="wlog1",floor_rule="peak4")
stopifnot(corrected$diagnostics$loss_underflow,
  corrected$diagnostics$objective=="weighted_log_quadratic_power1",
  all(is.finite(corrected$parameters)),all(diag(corrected$twist$covariance)>0),
  all(is.finite(iapf_log_normal(fixture$points,corrected$twist$mean,
    corrected$twist$covariance))))
near(corrected$parameters,fixture$parameters,1e-12)
cat("PASS explanatory density underflow does not reject a valid weighted-log fit\n")

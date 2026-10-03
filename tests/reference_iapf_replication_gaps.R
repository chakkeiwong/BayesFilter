# Independent identities, including actual public R consumers; CPU reference only.
args <- commandArgs(trailingOnly=TRUE); root <- if(length(args))args[1] else "."
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_plausible_choices.R"))
data <- iapf_paper_data(3,7,seed=87000001)
model <- iapf_linear_model(data$observations,data$A)
truth <- iapf_kalman(model)
for(kappa in c(0,.5,1)) {
  set.seed(123)
  exact <- iapf_apf(model,iapf_exact_twists(model),19,kappa)
  stopifnot(abs(exact$log_likelihood-truth$log_likelihood)<1e-10,
    max(vapply(exact$log_weights,function(w)diff(range(w)),numeric(1)))<1e-10)
  set.seed(124); a <- iapf_apf(model,iapf_observation_twists(model),47,kappa)
  set.seed(124); b <- iapf_fully_adapted(model,47,kappa)
  stopifnot(abs(a$log_likelihood-b$log_likelihood)<1e-10,
    max(abs(a$prefix-b$prefix))<1e-10,a$resampling_count==b$resampling_count)
}
twists <- lapply(1:7,function(t)iapf_gaussian_twist(rep(t/10,3),diag(c(.7,1.2,2)),log(.03)))
set.seed(125); draw <- iapf_apf(model,twists,31,0)
direct <- rep(0,31)
for(t in 1:7) {
  mu <- if(t==1)matrix(0,31,3) else model$transition_mean(draw$clouds[[t-1]],t)
  covariance <- if(t==1)model$initial_covariance else model$transition_covariance
  x <- draw$clouds[[t]]
  direct <- direct+iapf_log_normal(x,mu,covariance)+
    model$log_observation(x,model$observations[t,],t)-
    iapf_log_proposal(x,iapf_proposal(mu,covariance,twists[[t]]))
  future <- if(t==7)rep(0,31) else iapf_log_integral(
    model$transition_mean(x,t+1),model$transition_covariance,twists[[t+1]])
  stopifnot(max(abs(direct-(draw$log_weights[[t]]-future)))<1e-10,
    abs(iapf_logmean(direct)-draw$prefix[t])<1e-10)
}
points <- as.matrix(expand.grid(seq(-2,2,length.out=9),seq(-2,2,length.out=9)))
target <- iapf_log_normal(points,c(0,0),matrix(c(1,.6,.6,1),2))
old <- tryCatch(iapf_local_eq15_fit(points,target),iapf_fit_error=identity)
new <- iapf_local_eq15_fit(points,target,allow_active_bounds=TRUE)
stopifnot(inherits(old,"iapf_fit_error"),new$diagnostics$boundary,
  new$diagnostics$active_bounds_explicitly_allowed,new$diagnostics$active_bound_coordinates>0,
  new$diagnostics$loss<=new$diagnostics$initial_loss,
  max(abs(new$twist$mean-old$parameters[1:2]))<1e-12,
  max(abs(diag(new$twist$covariance)-exp(old$parameters[3:4])))<1e-12)
# Endpoint must really pass the diagnostic fitter into the common controller.
seen <- NULL
iapf_iterate <- function(...) {seen <<- list(...); seen}
invisible(iapf_constrained_reconstruction(model,.5,fit_maxit=800))
stopifnot(seen$floor_tail_power==8,seen$doubling_mode=="after_k",is.function(seen$.fit),
  seen$fit_maxit==800)
iapf_local_eq15_backward <- function(...)list(...)
wiring <- seen$.fit(model,list(),1000,200,8)
stopifnot(wiring$allow_active_bounds,identical(wiring$box_multiplier,.5))
cat("PASS exact future oracle, FA consumer equivalence, finite-floor path/prefix identities, explicit bound reporting and consumer wiring\n")

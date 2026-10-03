# Deterministic controller decision and fresh-final-call checks; CPU R reference.
args <- commandArgs(trailingOnly=TRUE); root <- if(length(args))args[1] else "."
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_plausible_choices.R"))
model <- list(horizon=1)
factory <- function() {
  calls <- 0L
  function(model,twists,N,kappa,store_clouds) {
    calls <<- calls+1L
    list(log_likelihood=log(c(2,10,rep(1,6),1.7)[calls]),
      clouds=list(matrix(0,1,1)),call=calls,N=N)
  }
}
fitter <- function(...)list(twists=list(iapf_constant_twist()),diagnostics=list(list(convergence=0)))
default <- iapf_iterate(model,.filter=factory(),.fit=fitter,doubling_mode="after_k")
explicit <- iapf_iterate(model,.filter=factory(),.fit=fitter,doubling_mode="after_k",stopping_window=6)
short <- iapf_iterate(model,.filter=factory(),.fit=fitter,doubling_mode="after_k",stopping_window=5)
stopifnot(identical(default,explicit),default$stop_iteration==7,short$stop_iteration==6,
  default$final_particles==2000,short$final_particles==1000,
  default$final$call==length(default$history)+1,
  short$final$call==length(short$history)+1,
  default$final$N==tail(default$counts,1),short$final$N==tail(short$counts,1),
  identical(head(default$history,7),short$history),
  identical(head(default$counts,7),short$counts))
for(window in c(1,7,5.5,NA)) {
  rejected <- tryCatch({iapf_iterate(model,stopping_window=window);FALSE},error=function(e)TRUE)
  stopifnot(rejected)
}
seen <- NULL
iapf_iterate <- function(...) {seen <<- list(...); seen}
invisible(iapf_window_reconstruction(model,5))
stopifnot(seen$stopping_window==5,seen$fit_mode=="log_quadratic",
  seen$floor_tail_power==8,seen$doubling_mode=="after_k")
cat("PASS default decision preserved, isolated window change, fresh final calls and shared-controller endpoint wiring\n")

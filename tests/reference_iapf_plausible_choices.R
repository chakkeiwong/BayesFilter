# Mathematical no-fire and actual-consumer wiring checks for optional R guesses.
args <- commandArgs(trailingOnly=TRUE); root <- if(length(args))args[1] else "."
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_plausible_choices.R"))
points <- as.matrix(expand.grid(seq(-2,2,length.out=9),seq(-2,2,length.out=9)))
m <- c(.2,-.3); V <- diag(c(.7,1.3))
target <- iapf_log_normal(points,m,V)
fit <- iapf_local_eq15_fit(points,target)
stopifnot(max(abs(fit$twist$mean-m))<1e-8,max(abs(fit$twist$covariance-V))<1e-8,
  fit$diagnostics$relative_bound_margin>.49,is.finite(fit$twist$log_floor))
# With correlation, bounds must be flagged if the absolute density loss escapes.
bad <- tryCatch(iapf_local_eq15_fit(points,
  iapf_log_normal(points,c(0,0),matrix(c(1,.6,.6,1),2))),iapf_fit_error=identity)
stopifnot(inherits(bad,"iapf_fit_error"),bad$relative_bound_margin<=1e-6,
  all(bad$parameters>=bad$lower),all(bad$parameters<=bad$upper))
seen <- NULL
iapf_iterate <- function(...) { seen <<- list(...); seen }
invisible(iapf_reconstruction(list(),"local_eq15"))
stopifnot(identical(seen$.fit,iapf_local_eq15_backward),seen$fit_mode=="paper_eq15",
  seen$doubling_mode=="after_k",seen$floor_tail_power==8)
invisible(iapf_reconstruction(list(),"floor4"))
stopifnot(is.null(seen$.fit),seen$fit_mode=="log_quadratic",seen$floor_tail_power==4)
cat("PASS exact Gaussian no-fire, bounded-escape guard, actual endpoint wiring\n")

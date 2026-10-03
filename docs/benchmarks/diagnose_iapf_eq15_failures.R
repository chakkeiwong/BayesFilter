# Post-run independent-reference inspection. This file cannot select a runtime.
args <- commandArgs(trailingOnly=TRUE)
campaign <- args[1]; output <- args[2]
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1", !file.exists(output))
paths <- list.files(file.path(campaign,"runs"),pattern="candidate-failure.rds$",
  recursive=TRUE,full.names=TRUE)
records <- lapply(paths,function(path) {
  x <- readRDS(path); d <- x$diagnostics
  stopifnot(inherits(x,"iapf_fit_error"),!is.null(d))
  data.frame(job_id=basename(dirname(dirname(path))), reason=x$message,
    backward_time=x$backward_time, as.data.frame(d),
    mean_displacement=sqrt(sum((x$parameters[seq_len(ncol(x$points))]-
      x$initial_parameters[seq_len(ncol(x$points))])^2)))
})
write.csv(do.call(rbind,records),output,row.names=FALSE)

# Independent quadrature checks of the Gaussian second-moment derivation,
# including shifted and unequal-variance cases. No stochastic ranking follows.
cases <- data.frame(P=c(1,1,.8), Q=c(1,2,.6), delta=c(.7,0,.2))
for(i in seq_len(nrow(cases))) {
  P <- cases$P[i]; Q <- cases$Q[i]; delta <- cases$delta[i]
  M <- 2/P-1/Q; h <- delta/Q
  formula <- sqrt(Q)/(P*sqrt(M))*exp(.5*delta^2/Q+.5*h^2/M)
  quadrature <- integrate(function(x)
    exp(2*dnorm(x,delta,sqrt(P),log=TRUE)-dnorm(x,0,sqrt(Q),log=TRUE)),
    -Inf,Inf,rel.tol=1e-10)$value
  stopifnot(abs(formula/quadrature-1)<1e-9)
}
cat("PASS: preserved",length(records),"fit failures; three independent Gaussian-integral checks\n")

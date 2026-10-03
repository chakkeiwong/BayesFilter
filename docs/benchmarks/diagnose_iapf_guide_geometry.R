# Independent diagnostic oracle. Uses the frozen R reference without changes.
args <- commandArgs(trailingOnly=TRUE)
source("docs/benchmarks/reference_iapf_paper.R")
source(args[1])
out <- args[2]
mat <- function(x) do.call(rbind,lapply(x,unlist))
metrics <- list(); coefficients <- list()
for (entry in input) {
  A <- mat(entry$A); H <- mat(entry$H); Q <- mat(entry$Q); R <- mat(entry$R)
  m0 <- unlist(entry$m0); P0 <- mat(entry$P0)
  noise <- mat(entry$initial_noise); N <- nrow(noise); d <- ncol(noise)
  x0 <- sweep(noise %*% chol(P0),2,m0,"+")
  for (regime in names(entry$observations)) {
    y <- mat(entry$observations[[regime]]); T <- nrow(y)
    model <- iapf_linear_model(y,A,as.vector(A %*% m0),A %*% P0 %*% t(A)+Q,Q,H,R)
    twists <- iapf_exact_twists(model)
    constants <- numeric(T); identity_error <- 0; minimum_eigenvalue <- Inf
    for (t in seq_len(T)) {
      twist <- twists[[t]]; c <- twist$mean; V <- twist$covariance
      # Include off-center points, not only the point defining each constant.
      directions <- diag(sqrt(diag(V)),d)
      points <- rbind(c,rep(0,d),sweep(directions,2,c,"+"),sweep(-directions,2,c,"+"))
      increment <- model$log_observation(points,y[t,],t)-iapf_log_twist(points,twist)
      if (t<T) increment <- increment+iapf_log_integral(points %*% t(A),Q,twists[[t+1]])
      constants[t] <- increment[1]
      identity_error <- max(identity_error,max(abs(increment-constants[t])))
      minimum_eigenvalue <- min(minimum_eigenvalue,eigen(V,symmetric=TRUE,only.values=TRUE)$values)
      for (j in seq_len(d)) {
        coefficients[[length(coefficients)+1]] <- data.frame(case=entry$case,regime=regime,
          time=t,row=j,column=0L,value=c[j])
        for (k in seq_len(d)) coefficients[[length(coefficients)+1]] <- data.frame(
          case=entry$case,regime=regime,time=t,row=j,column=k,value=V[j,k])
      }
    }
    first <- twists[[1]]
    initial_prediction <- iapf_logmean(iapf_log_integral(x0 %*% t(A),Q,first))+sum(constants)
    integrated_prediction <- iapf_log_integral(matrix(as.vector(A %*% m0),1),
      A %*% P0 %*% t(A)+Q,first)+sum(constants)
    kalman <- iapf_kalman(model)$log_likelihood
    metrics[[length(metrics)+1]] <- data.frame(case=entry$case,regime=regime,d=d,N=N,
      exact_kalman=kalman,initial_sample_prediction=initial_prediction,
      integrated_prediction=integrated_prediction,identity_error=identity_error,
      integration_identity_error=abs(kalman-integrated_prediction),
      minimum_eigenvalue=minimum_eigenvalue)
    stopifnot(identity_error<1e-9,abs(kalman-integrated_prediction)<1e-9,minimum_eigenvalue>0)
  }
}
options(digits=17)
write.csv(do.call(rbind,metrics),file.path(out,"metrics.csv"),row.names=FALSE)
write.csv(do.call(rbind,coefficients),file.path(out,"coefficients.csv"),row.names=FALSE)
cat(length(metrics),"exact-guide cases passed independent R identities\n")

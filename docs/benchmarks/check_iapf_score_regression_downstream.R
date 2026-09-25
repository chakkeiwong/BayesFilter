# Independent fresh-data target, recursive candidate and initial-integral checks.
args <- commandArgs(trailingOnly=TRUE)
source('docs/benchmarks/reference_iapf_paper.R');source(args[1]);out <- args[2]
mat <- function(x) do.call(rbind,lapply(x,unlist))
array_mats <- function(x) lapply(x,mat)
records <- list()
logn <- function(res,V) {
  z <- t(solve(t(chol(V)),t(res)))
  -.5*rowSums(z*z)-sum(log(diag(chol(V))))-.5*ncol(res)*log(2*pi)
}
for (entry in input) {
  A <- mat(entry$model[[1]]);H <- mat(entry$model[[2]]);m0 <- unlist(entry$model[[3]])
  P0 <- mat(entry$model[[4]]);Q <- mat(entry$model[[5]]);R <- mat(entry$model[[6]])
  y <- mat(entry$y);T <- nrow(y);d <- ncol(y);clouds <- array_mats(entry$clouds)
  model <- iapf_linear_model(y,A,as.vector(A%*%m0),A%*%P0%*%t(A)+Q,Q,H,R)
  exact <- iapf_exact_twists(model);kalman <- iapf_kalman(model)$log_likelihood
  center <- mat(entry$exact_coefficients[[1]]);covs <- array_mats(entry$exact_coefficients[[2]])
  oracle_error <- abs(kalman-entry$exact_value);constants <- numeric(T)
  for (t in seq_len(T)) {
    oracle_error <- max(oracle_error,abs(exact[[t]]$mean-center[t,]),abs(exact[[t]]$covariance-covs[[t]]))
    point <- matrix(exact[[t]]$mean,1)
    constants[t] <- model$log_observation(point,y[t,],t)-iapf_log_twist(point,exact[[t]])
    if (t<T) constants[t] <- constants[t]+iapf_log_integral(point%*%t(A),Q,exact[[t+1]])
  }
  initial_error <- 0
  for (r in seq_along(entry$initial_noises)) {
    x0 <- sweep(mat(entry$initial_noises[[r]])%*%chol(P0),2,m0,'+')
    predicted <- iapf_logmean(iapf_log_integral(x0%*%t(A),Q,exact[[1]]))+sum(constants)
    initial_error <- max(initial_error,abs(predicted-entry$initial_predictions[[r]]))
  }
  candidate_c <- mat(entry$score_coefficients[[1]]);candidate_v <- array_mats(entry$score_coefficients[[2]])
  candidate_f <- unlist(entry$score_coefficients[[3]])
  candidate_error <- 0;candidate_valid <- TRUE;c <- V <- f <- NULL
  for (t in rev(seq_len(T))) {
    x <- clouds[[t]];N <- nrow(x)
    res <- sweep(-x%*%t(H),2,y[t,],'+');h <- res%*%solve(R,H)
    if (t<T) {
      future <- sweep(-x%*%t(A),2,c,'+');lg <- logn(future,Q+V);mx <- pmax(lg,f)
      lm <- mx+log(exp(lg-mx)+exp(f-mx));h <- h+exp(lg-lm)*(future%*%solve(Q+V,A))
    }
    m <- colMeans(x);xc <- sweep(x,2,m);C <- crossprod(xc)/N
    eig <- eigen(C,symmetric=TRUE,only.values=TRUE)$values;tol <- 64*2^-52*max(N,d)
    if (min(eig)<=tol*max(abs(eig))) {candidate_valid <- FALSE;break}
    L <- t(chol(C));z <- t(solve(L,t(xc)));g <- h%*%L;a <- colMeans(g)
    raw <- -crossprod(sweep(g,2,a),z)/N;P <- (raw+t(raw))/2
    eig <- eigen(P,symmetric=TRUE,only.values=TRUE)$values
    if (min(eig)<=tol*max(abs(eig))) {candidate_valid <- FALSE;break}
    full <- L%*%solve(P,t(L));c <- as.vector(m+L%*%solve(P,a));V <- diag(diag(full),d)
    # Match the repository's recorded float32 scalar cast before FP64 log.
    ratio <- 0.009999999776482582
    f <- log(ratio)-.5*d*log(2*pi)-.5*sum(log(diag(V)))
    candidate_error <- max(candidate_error,abs(c-candidate_c[t,]),abs(V-candidate_v[[t]]),abs(f-candidate_f[t]))
  }
  current_cov <- solve(t(H)%*%solve(R,H));current_c <- t(current_cov%*%t(H)%*%solve(R,t(y)))
  current_error <- max(abs(current_c-mat(entry$one_step[[1]])),sapply(array_mats(entry$one_step[[2]]),function(V)max(abs(V-current_cov))))
  stopifnot(oracle_error<=1e-9,initial_error<=1e-8,current_error<=1e-8,
            candidate_valid==entry$score_valid,candidate_error<=1e-8)
  records[[length(records)+1]] <- data.frame(case=entry$case,oracle_error=oracle_error,
    initial_error=initial_error,current_error=current_error,candidate_valid=candidate_valid,candidate_error=candidate_error)
}
options(digits=17)
write.csv(do.call(rbind,records),file.path(out,'reference-checks.csv'),row.names=FALSE)
cat(length(records),'fresh cases passed independent R checks\n')

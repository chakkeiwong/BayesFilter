# Independent R score recursion for full/diagonal covariance and positive/negligible floor.
args <- commandArgs(trailingOnly=TRUE)
source('docs/benchmarks/reference_iapf_paper.R');source(args[1]);out <- args[2]
mat <- function(x) do.call(rbind,lapply(x,unlist))
logn <- function(res,V) {
  z <- t(solve(t(chol(V)),t(res)))
  -.5*rowSums(z*z)-sum(log(diag(chol(V))))-.5*ncol(res)*log(2*pi)
}
records <- list()
for (entry in input) {
  A <- mat(entry$model[[1]]);H <- mat(entry$model[[2]]);m0 <- unlist(entry$model[[3]])
  P0 <- mat(entry$model[[4]]);Q <- mat(entry$model[[5]]);R <- mat(entry$model[[6]])
  y <- mat(entry$y);T <- nrow(y);d <- ncol(y);clouds <- lapply(entry$clouds,mat)
  model <- iapf_linear_model(y,A,as.vector(A%*%m0),A%*%P0%*%t(A)+Q,Q,H,R)
  exact <- iapf_exact_twists(model);kalman <- iapf_kalman(model)$log_likelihood
  oracle_error <- abs(kalman-entry$exact_value);constants <- numeric(T)
  for (t in seq_len(T)) {
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
  for (name in names(entry$coefficients)) {
    components <- strsplit(name,'_')[[1]];full <- components[1]=='full';positive <- components[2]=='positive'
    coeff <- entry$coefficients[[name]];candidate_c <- mat(coeff[[1]])
    candidate_v <- lapply(coeff[[2]],mat);candidate_f <- unlist(coeff[[3]])
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
      Vfull <- L%*%solve(P,t(L));c <- as.vector(m+L%*%solve(P,a))
      V <- if (full) Vfull else diag(diag(Vfull),d)
      ratio <- if (positive) log(0.009999999776482582) else -1000
      f <- ratio-.5*d*log(2*pi)-sum(log(diag(chol(V))))
      candidate_error <- max(candidate_error,abs(c-candidate_c[t,]),abs(V-candidate_v[[t]]),abs(f-candidate_f[t]))
      if (full && !positive) {
        oracle_error <- max(oracle_error,abs(c-exact[[t]]$mean),abs(V-exact[[t]]$covariance))
      }
    }
    stopifnot(oracle_error<=1e-9,initial_error<=1e-8,candidate_error<=1e-8,
              candidate_valid==entry$accepted[[name]])
    records[[length(records)+1]] <- data.frame(case=entry$case,method=name,candidate_valid=candidate_valid,
      candidate_error=candidate_error,oracle_error=oracle_error,initial_error=initial_error)
  }
}
options(digits=17)
write.csv(do.call(rbind,records),file.path(out,'reference-checks.csv'),row.names=FALSE)
cat(length(records),'factorial recursive fits passed independent R checks\n')

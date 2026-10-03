# Independent R verification of the explicitly non-paper score fitting rule.
args <- commandArgs(trailingOnly=TRUE);source(args[1]);out <- args[2]
mat <- function(x) do.call(rbind,lapply(x,unlist))
records <- list()
for (r in input$cases) {
  x <- mat(r$points);h <- mat(r$scores);n <- nrow(x);d <- ncol(x)
  m <- colMeans(x); xc <- sweep(x,2,m);C <- crossprod(xc)/n;L <- t(chol(C))
  z <- t(solve(L,t(xc)));g <- h %*% L;a <- colMeans(g)
  raw <- -crossprod(sweep(g,2,a),z)/n;P <- (raw+t(raw))/2
  V <- L %*% solve(P,t(L));c <- as.vector(m+L%*%solve(P,a))
  error <- max(abs(c-unlist(r$center)),abs(V-mat(r$info$full_covariance)),
               abs(diag(V)-diag(mat(r$covariance))),abs(c-unlist(r$exact_center)),
               abs(V-mat(r$exact_covariance)))
  stopifnot(error<1e-8,all(eigen(P,symmetric=TRUE,only.values=TRUE)$values>0))
  records[[length(records)+1]] <- data.frame(d=d,error=error,whitening=max(abs(crossprod(z)/n-diag(d))))
}
r <- input$target;x <- mat(r$points);A <- mat(r$A);H <- mat(r$H);Q <- mat(r$Q);R <- mat(r$R)
c <- unlist(r$next_center);V <- mat(r$next_cov);f <- r$next_floor;d <- ncol(x)
logn <- function(res,V) {
  z <- t(solve(t(chol(V)),t(res)))
  -.5*rowSums(z*z)-sum(log(diag(chol(V))))-.5*d*log(2*pi)
}
res <- sweep(-x%*%t(H),2,unlist(r$y),'+');future <- sweep(-x%*%t(A),2,c,'+')
lg <- logn(future,Q+V);mx <- pmax(lg,f);lm <- mx+log(exp(lg-mx)+exp(f-mx))
score <- res%*%solve(R,H)+exp(lg-lm)*(future%*%solve(Q+V,A))
value <- logn(res,R)+lm
target_error <- max(abs(value-unlist(r$value)),abs(score-mat(r$score)))
stopifnot(target_error<1e-8)
write.csv(do.call(rbind,records),file.path(out,'recovery.csv'),row.names=FALSE)
write.csv(data.frame(target_error=target_error),file.path(out,'target.csv'),row.names=FALSE)
cat('Five full Gaussian recoveries and positive-floor score parity passed\n')

source("docs/benchmarks/reference_iapf_paper.R")
p <- readRDS("docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01/attempt03/min_mass_Nminus1-9262001.rds")
data <- readRDS("docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01/attempt03/data-9262001.rds")
model <- data$bundle$model
for (tt in c(100,99,98)) {
  inp <- p$inputs[[tt]]
  cat("TIME",tt,"input",!is.null(inp),"\n")
  if (is.null(inp$next_twist)) next
  tw <- inp$next_twist
  A <- model$A; S <- model$transition_covariance + tw$covariance
  P <- diag(model$dimension) + t(A) %*% solve(S,A)
  V <- solve(P)
  m <- V %*% (model$observations[tt,] + t(A)%*%solve(S,tw$mean))
  corr <- V/sqrt(outer(diag(V),diag(V)))
  eig <- eigen(V,symmetric=TRUE,only.values=TRUE)$values
  cat("twist var:",paste(signif(diag(tw$covariance),5),collapse=" "),"\n")
  cat("target Gaussian mean:",paste(signif(as.vector(m),5),collapse=" "),"\n")
  cat("target Gaussian var:",paste(signif(diag(V),5),collapse=" "),"\n")
  cat("target corr offdiag max:",max(abs(corr-diag(nrow(corr)))) ,"cond:",max(eig)/min(eig),"\n")
  cat("cloud mean/sd:",paste(signif(colMeans(inp$points),4),collapse=" "),paste(signif(sqrt(colMeans(sweep(inp$points,2,colMeans(inp$points))^2)),4),collapse=" "),"\n")
  cat("target range:",paste(signif(range(inp$log_targets),5),collapse=" "),"\n")
}

# Deterministic independent-reference diagnostic; never a runtime fitter.
args <- commandArgs(TRUE)
stopifnot(length(args)==2)
root <- args[1]; output <- args[2]
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
stopifnot(!dir.exists(output)); dir.create(output,recursive=TRUE)

points <- as.matrix(expand.grid(seq(-2,2),seq(-2,2)))
rho <- .6
target <- -.5*(points[,1]^2-2*rho*points[,1]*points[,2]+points[,2]^2)/(1-rho^2)
targets <- exp(target-max(target))
scales <- c(1,2,4,16,256,4096,1e6)
rows <- lapply(scales,function(scale) {
  variance <- (1-rho^2)*scale^2
  actual <- iapf_fit_objective(c(0,0,rep(log(variance),2)),points,target,0)
  density <- dnorm(points[,1],0,sqrt(variance))*dnorm(points[,2],0,sqrt(variance))
  lambda <- sum(density*targets)/sum(targets^2)
  direct <- mean((density-lambda*targets)^2)
  direct_relative <- sum((density-lambda*targets)^2)/sum(density^2)
  stopifnot(abs(actual$value/direct-1)<1e-12,
    abs(actual$relative_residual-direct_relative)<1e-12,
    is.finite(actual$value),actual$value>0,!actual$loss_underflow)
  data.frame(scale=scale,variance=variance,loss=actual$value,
    direct_loss=direct,relative_residual=actual$relative_residual,
    direct_relative=direct_relative,loss_underflow=actual$loss_underflow,
    gaussian_gradient_max=max(abs(actual$gradient)))
})
rows <- do.call(rbind,rows)
n <- nrow(rows)
constant_limit <- 1-sum(targets)^2/(length(targets)*sum(targets^2))
stopifnot(rows$loss[n]<1e-12*rows$loss[1],
  rows$relative_residual[n]>rows$relative_residual[1],
  abs(rows$relative_residual[n]-constant_limit)<1e-10)

lookup <- function(x,y) target[points[,1]==x & points[,2]==y]
contrast <- lookup(1,1)-lookup(1,0)-lookup(0,1)+lookup(0,0)
stopifnot(abs(contrast-rho/(1-rho^2))<1e-12,contrast!=0)
diagonal_target <- -.5*rowSums(points^2)
control <- iapf_fit_objective(c(0,0,0,0),points,diagonal_target,0)
stopifnot(control$value<1e-28,control$relative_residual<1e-28,
  !control$loss_underflow)

write.csv(rows,file.path(output,"scale-path.csv"),row.names=FALSE)
write.csv(data.frame(target_log_mixed_contrast=contrast,
  diagonal_gaussian_log_mixed_contrast=0,
  limiting_relative_residual=constant_limit,
  loss_ratio_last_to_first=rows$loss[n]/rows$loss[1],
  exact_diagonal_control_loss=control$value,
  exact_diagonal_control_relative_residual=control$relative_residual,
  all_checks_passed=TRUE),file.path(output,"checks.csv"),row.names=FALSE)
cat("PASS: finite scale escape, direct objective parity, nonrepresentability, and exact control\n")

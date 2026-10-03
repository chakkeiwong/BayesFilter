# Independent CPU reference algebra, gradient, source-target and endpoint checks.
args <- commandArgs(trailingOnly=TRUE); root <- if(length(args))args[1] else "."
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_eq15_solvers.R"))
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
set.seed(921901); x <- matrix(rnorm(900),300,3)
target <- -.5*rowSums(sweep(x,2,c(.3,-.1,.2))^2)+.1*sin(x[,1]*x[,2])
p <- c(.2,.1,-.2,log(c(.8,1.2,.7)))
scale <- max(iapf_log_normal(x,p[1:3],diag(exp(p[4:6]))))
profile <- iapf_fit_objective(p,x,target,scale)
jointp <- c(p,log(profile$lambda_scaled))
joint <- iapf_eq15_joint_objective(jointp,x,target,scale)
stopifnot(abs(joint$value-profile$value)<1e-13,
 max(abs(joint$gradient[1:6]-profile$gradient))<1e-12,abs(joint$gradient[7])<1e-12)
jointp[7] <- jointp[7]+.2
for(j in seq_along(jointp)) {
 hi <- lo <- jointp; hi[j] <- hi[j]+1e-5; lo[j] <- lo[j]-1e-5
 fd <- (iapf_eq15_joint_objective(hi,x,target,scale)$value-
   iapf_eq15_joint_objective(lo,x,target,scale)$value)/2e-5
 stopifnot(abs(fd-iapf_eq15_joint_objective(jointp,x,target,scale)$gradient[j])<1e-7)
}
exact_mean <- c(.2,-.4,.1); exact_v <- c(.7,1.1,1.5)
exact <- iapf_log_normal(x,exact_mean,diag(exact_v))
for(kind in c("joint","profiled")) {
 f <- iapf_eq15_solver(x,exact,parameterization=kind)
 stopifnot(max(abs(f$parameters-c(exact_mean,log(exact_v))))<1e-10,
  f$diagnostics$relative_residual<1e-24)
}
# A fixed positive rescaling of both density and target residual scales the
# loss by its square; the shape and profiled optimum are unchanged.
other <- iapf_fit_objective(p,x,target,scale+2)
stopifnot(abs(other$value/profile$value-exp(-4))<1e-12)
# Correlated target cannot generally be represented by a diagonal Gaussian;
# spreading candidate variance nevertheless reduces absolute Eq15 loss.
target <- -.5*(x[,1]^2+x[,2]^2+x[,3]^2+1.2*x[,1]*x[,2])
loss <- vapply(c(1,10,100),function(v)
 iapf_fit_objective(c(rep(0,3),rep(log(v),3)),x,target,0)$value,numeric(1))
stopifnot(all(diff(loss)<0),loss[3]<loss[1]/10000)
# Execute the actual reconstruction endpoint on a one-time Gaussian model.
# Its fitted target is exactly diagonal, so both local routes must be callable.
model <- iapf_linear_model(matrix(c(.2,-.1),1,2),diag(2))
for(kind in c("joint","profiled")) {
 set.seed(921902); answer <- iapf_eq15_reconstruction(model,kind)
 stopifnot(answer$status=="complete",is.finite(answer$final$log_likelihood),
  answer$fits[[1]][[1]]$parameterization==kind,
  answer$fits[[1]][[1]]$objective=="printed_equation15_density_residual")
}
cat("PASS: Eq15 joint/profile identity, finite-difference gradients, exact Gaussian, scaling, escape, actual endpoints\n")

# Independent tests for the explicitly non-equation-(15) Gaussian shape fit.
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1],"docs/benchmarks/reference_iapf_paper.R"))
checks <- list()
check <- function(name,condition) {
  checks[[length(checks)+1]] <<- data.frame(name=name,passed=isTRUE(condition))
  if (!isTRUE(condition)) stop(paste("FAILED",name),call.=FALSE)
}
close <- function(name,a,b,tol=1e-8) check(name,
  length(a)==length(b) && all(is.finite(a)) && all(abs(a-b)<=tol*(1+abs(b))))
points <- as.matrix(expand.grid(seq(-3,3,length.out=13),seq(-2,2,length.out=11)))
target <- -.5*((points[,1]-.3)^2/.8+(points[,2]+.2)^2/1.2)
par <- c(.1,-.4,log(c(.6,1.1)))
objective <- iapf_relative_fit_objective(par,points,target)
p <- dnorm(points[,1],.1,sqrt(.6))*dnorm(points[,2],-.4,sqrt(1.1))
y <- exp(target); lambda <- sum(p*y)/sum(y^2)
close("relative loss independent formula",objective$value,sum((p-lambda*y)^2)/sum(p^2))
fd <- sapply(seq_along(par),function(i) {
  plus <- minus <- par; plus[i] <- plus[i]+1e-5; minus[i] <- minus[i]-1e-5
  (iapf_relative_fit_objective(plus,points,target)$value-
   iapf_relative_fit_objective(minus,points,target)$value)/2e-5
})
close("relative analytic gradient",objective$gradient,fd)
close("relative target shift",iapf_relative_fit_objective(par,points,target-10000)$value,objective$value)
# A coordinate rescaling changes the Gaussian density amplitude but not shape.
scaled <- iapf_relative_fit_objective(c(par[1:2]*1e6,par[3:4]+log(1e12)),points*1e6,target)
close("density amplitude invariance",scaled$value,objective$value)
diffuse <- iapf_relative_fit_objective(c(0,0,rep(log(1e15),2)),points,target)
check("diffuse Gaussian is not a zero loss",diffuse$value>.1)
fit <- iapf_fit_gaussian(points,target,fit_mode="relative_l2")
close("relative exact Gaussian mean",fit$twist$mean,c(.3,-.2),1e-6)
close("relative exact Gaussian variance",diag(fit$twist$covariance),c(.8,1.2),1e-6)
check("relative method identity",fit$diagnostics$fit_mode=="relative_l2")
check("strict fit remains default",iapf_fit_gaussian(points,target)$diagnostics$fit_mode=="paper_eq15")
original_optim <- optim
optim <- function(par,...) list(par=par,convergence=1,counts=setNames(201,"function"))
failed <- tryCatch(iapf_fit_gaussian(points,target,fit_mode="relative_l2"),error=function(e)e)
check("nonconvergence rejected with inputs",inherits(failed,"iapf_fit_error") &&
  failed$fit_mode=="relative_l2" && identical(failed$points,points))
optim <- original_optim

data <- iapf_paper_data(2,3,seed=61000002)
model <- iapf_linear_model(data$observations,data$A)
set.seed(61000102)
result <- iapf_iterate(model,N0=100,k=1,tau=1,max_iterations=8,
  fit_mode="relative_l2")
check("actual relative consumer completes",result$status=="complete")
check("actual backward call chain uses relative fit",all(vapply(
  unlist(result$fits,recursive=FALSE),function(x)x$fit_mode=="relative_l2",TRUE)))
port <- iapf_fit_gaussian(points,target,fit_mode="relative_l2_nlminb")
close("nlminb exact Gaussian mean",port$twist$mean,c(.3,-.2),1e-6)
close("nlminb exact Gaussian variance",diag(port$twist$covariance),c(.8,1.2),1e-6)
check("nlminb explicit solver identity",port$diagnostics$optimizer=="nlminb")
original_nlminb <- nlminb
nlminb <- function(start,...) list(par=start,convergence=1,evaluations=c(201,201))
failed <- tryCatch(iapf_fit_gaussian(points,target,fit_mode="relative_l2_nlminb"),error=function(e)e)
check("nlminb nonconvergence rejected with inputs",inherits(failed,"iapf_fit_error") &&
  failed$fit_mode=="relative_l2_nlminb" && identical(failed$points,points))
nlminb <- original_nlminb
set.seed(61000102)
port_result <- iapf_iterate(model,N0=100,k=1,tau=1,max_iterations=8,
  fit_mode="relative_l2_nlminb",fit_maxit=1000)
check("actual nlminb consumer completes",port_result$status=="complete")
check("actual backward call chain uses nlminb",all(vapply(
  unlist(port_result$fits,recursive=FALSE),function(x)
    x$fit_mode=="relative_l2_nlminb" && x$optimizer=="nlminb",TRUE)))
log_fit <- iapf_fit_gaussian(points,target,fit_mode="log_quadratic")
close("log fit recovers Gaussian mean",log_fit$twist$mean,c(.3,-.2))
close("log fit recovers Gaussian variance",diag(log_fit$twist$covariance),c(.8,1.2))
check("log fit full rank normal equations",log_fit$diagnostics$design_rank==5 &&
  log_fit$diagnostics$gradient_max<1e-12 && log_fit$diagnostics$loss<1e-25)
multiplier <- c(-7,.04); shift <- c(31,-3)
transformed <- sweep(sweep(points,2,multiplier,"*"),2,shift,"+")
affine_fit <- iapf_fit_gaussian(transformed,target+1700,fit_mode="log_quadratic")
close("log fit affine equivariance mean",affine_fit$twist$mean,c(.3,-.2)*multiplier+shift)
close("log fit affine equivariance variance",diag(affine_fit$twist$covariance),c(.8,1.2)*multiplier^2)
# A non-Gaussian target checks stationarity of the stated log objective.
non_gaussian_target <- target+.05*sin(points[,1])*cos(points[,2])
non_gaussian_fit <- iapf_fit_gaussian(points,non_gaussian_target,fit_mode="log_quadratic")
log_residual <- iapf_log_normal(points,non_gaussian_fit$twist$mean,
  non_gaussian_fit$twist$covariance)-non_gaussian_target
log_residual <- log_residual-mean(log_residual)
close("log fit normal equations for non-Gaussian target",
  as.vector(crossprod(cbind(1,points,points^2),log_residual)/nrow(points)),rep(0,5))
for(problem in c("rank","nonconcave","degenerate")) {
  bad_points <- points; bad_target <- target
  if(problem=="rank")bad_points[,2] <- bad_points[,1]
  if(problem=="nonconcave")bad_target <- points[,1]^2-points[,2]^2
  if(problem=="degenerate")bad_points[,2] <- 1
  rejected <- tryCatch(iapf_fit_gaussian(bad_points,bad_target,fit_mode="log_quadratic"),
    error=function(e)e)
  check(paste("log fit rejects",problem),inherits(rejected,"iapf_fit_error") &&
    rejected$fit_mode=="log_quadratic" && identical(rejected$points,bad_points))
}
set.seed(61000102)
log_result <- iapf_iterate(model,N0=100,k=1,tau=1,max_iterations=8,fit_mode="log_quadratic")
check("actual log fit consumer completes",log_result$status=="complete")
check("actual backward call chain uses log QR fit",all(vapply(
  unlist(log_result$fits,recursive=FALSE),function(x)
    x$fit_mode=="log_quadratic" && x$optimizer=="QR" &&
    x$design_rank==x$design_columns,TRUE)))
for(mode in c("first_full_window","after_k")) {
  seen <- numeric(); calls <- 0
  filter <- function(model,twists,N,kappa,store_clouds) {
    calls <<- calls+1; seen <<- c(seen,N)
    list(log_likelihood=c(0,.01,0,0,0)[calls],clouds=list())
  }
  fitter <- function(...) list(twists=list(iapf_constant_twist()),diagnostics=list())
  answer <- iapf_iterate(model,N0=8,k=2,tau=.1,max_iterations=8,
    .filter=filter,.fit=fitter,doubling_mode=mode)
  close(paste("doubling schedule",mode),seen,
    if(mode=="after_k")rep(8,5) else c(8,8,8,16,16))
  check(paste("fresh final remains",mode),calls==length(answer$history)+1)
}
write.csv(do.call(rbind,checks),args[2],row.names=FALSE)
cat(length(checks),"relative-fit checks passed\n")

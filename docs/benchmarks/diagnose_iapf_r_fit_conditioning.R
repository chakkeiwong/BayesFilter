# Explanatory local geometry and same-objective solver probe; no runtime repair.
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1],"docs/benchmarks/reference_iapf_paper.R"))
output <- args[2]; stopifnot(!dir.exists(output));dir.create(output,recursive=TRUE)
saved <- readRDS(file.path(args[1],
  "docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01/attempt08-pilot-d20/results/iapf-1-failure.rds"))
x <- saved$points; target <- saved$log_targets; d <- ncol(x)
geometry <- function(par) {
  mean <- par[seq_len(d)]; variance <- exp(par[d+seq_len(d)])
  dx <- sweep(x,2,mean); z2 <- sweep(dx^2,2,variance,"/")
  lp <- -.5*(d*log(2*pi)+sum(log(variance))+rowSums(z2))
  u <- exp(lp-max(lp));u <- u/sqrt(sum(u^2))
  v <- exp(target-max(target));v <- v/sqrt(sum(v^2))
  score <- cbind(sweep(dx,2,variance,"/"),.5*(z2-1))
  jacobian <- sweep(score,2,colSums(score*u^2))*u
  jacobian <- jacobian - v %o% as.vector(crossprod(v,jacobian))
  singular <- svd(jacobian,nu=0,nv=0)$d
  residual <- u-v*sum(u*v)
  list(loss=sum(residual^2),gradient=as.vector(2*crossprod(jacobian,residual)),
       singular=singular,target_squared_effective_points=1/sum(v^4),
       fitted_squared_effective_points=1/sum(u^4))
}
initial <- iapf_fit_initial(x,target)$parameters
before <- geometry(saved$parameters)
eval <- iapf_relative_fit_objective(saved$parameters,x,target)
stopifnot(max(abs(before$gradient-eval$gradient))<1e-10)
fd <- sapply(seq_along(initial),function(i) {
  a <- b <- saved$parameters;a[i]<-a[i]+1e-5;b[i]<-b[i]-1e-5
  (iapf_relative_fit_objective(a,x,target)$value-iapf_relative_fit_objective(b,x,target)$value)/2e-5
})
limit <- -log(.Machine$double.eps);started <- proc.time()[[3]]
candidate <- nlminb(initial,function(p)iapf_relative_fit_objective(p,x,target)$value,
  gradient=function(p)iapf_relative_fit_objective(p,x,target)$gradient,
  lower=c(rep(-Inf,d),rep(-limit,d)),upper=c(rep(Inf,d),rep(limit,d)),
  control=list(iter.max=5000,eval.max=10000,rel.tol=1e-10))
after <- geometry(candidate$par)
saveRDS(list(saved=saved,initial=initial,before=before,alternative=candidate,after=after),
  file.path(output,"conditioning.rds"))
row <- data.frame(dimension=d,N=nrow(x),backward_time=saved$backward_time,
  target_squared_effective_points=before$target_squared_effective_points,
  fitted_squared_effective_points=before$fitted_squared_effective_points,
  jacobian_largest=max(before$singular),jacobian_smallest=min(before$singular),
  jacobian_condition=max(before$singular)/min(before$singular),
  finite_difference_gradient_error=max(abs(fd-eval$gradient)),
  original_residual=before$loss,nlminb_residual=after$loss,
  nlminb_convergence=candidate$convergence,nlminb_iterations=candidate$iterations,
  nlminb_gradient=max(abs(after$gradient)),elapsed=proc.time()[[3]]-started)
write.csv(row,file.path(output,"conditioning.csv"),row.names=FALSE)
print(row)

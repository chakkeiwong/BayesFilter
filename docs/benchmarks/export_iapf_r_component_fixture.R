# Deterministic independent R reference fixture; no filtering/default claims.
args <- commandArgs(trailingOnly=TRUE)
source(file.path(args[1],"docs/benchmarks/reference_iapf_paper.R"))
out <- args[2]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
save_matrix <- function(name,value) write.table(as.matrix(value),
  file.path(out,paste0(name,".csv")),sep=",",row.names=FALSE,col.names=FALSE)
means <- matrix(c(-1,.2,0,0,.8,-.6,1.4,1.1,-.2,-1),ncol=2,byrow=TRUE)
Q <- matrix(c(1,.2,.2,.7),2,2)
V <- matrix(c(.6,-.1,-.1,1.3),2,2)
center <- c(.3,-.4)
log_floor <- log(.02)
noise <- matrix(c(.1,-.7,.3,.2,-.4,.8,1.2,-.3,-.6,-.2),ncol=2,byrow=TRUE)
uniform <- c(0,1,.4,.8,.99)
twist <- iapf_gaussian_twist(center,V,log_floor)
proposal <- iapf_proposal(means,Q,twist)
gaussian_probability <- 1-proposal$floor_probability
adapted <- proposal$means+noise%*%iapf_chol(proposal$covariance)
prior <- means+noise%*%iapf_chol(Q)
draws <- adapted
draws[uniform>=gaussian_probability,] <- prior[uniform>=gaussian_probability,]
points <- as.matrix(expand.grid(seq(-2,2,length.out=5),seq(-2,2,length.out=5)))
log_targets <- iapf_log_normal(points,c(.2,-.3),matrix(c(1,.6,.6,1),2,2))
parameters <- c(.1,-.2,log(c(.7,1.4)))
density <- iapf_fit_objective(parameters,points,log_targets,-log(2*pi))
relative <- iapf_relative_fit_objective(parameters,points,log_targets)
values <- list(means=means,Q=Q,V=V,center=center,log_floor=log_floor,
  noise=noise,uniform=uniform,points=points,log_targets=log_targets,
  parameters=parameters,log_integral=iapf_log_integral(means,Q,twist),
  gaussian_probability=gaussian_probability,draws=draws,
  density_loss=density$value,density_gradient=density$gradient,
  shape_error=density$relative_residual,relative_loss=relative$value,
  relative_gradient=relative$gradient)
invisible(lapply(names(values),function(name) save_matrix(name,values[[name]])))
stopifnot(any(uniform<gaussian_probability),any(uniform>=gaussian_probability))
cat("Exported deterministic two-dimensional R fixture.\n")

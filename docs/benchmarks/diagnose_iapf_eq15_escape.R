# Independent CPU reference diagnostic: printed Eq15 amplitude-escape rays.
# No optimizer, filter, candidate floor or alternative objective is promoted.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; failures <- args[2]; output <- args[3]
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",dir.exists(output))
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))

ray <- function(points,targets,initial,label) {
  d <- ncol(points); m <- initial[seq_len(d)]; lv <- initial[d+seq_len(d)]
  y <- exp(targets-max(targets))
  p0 <- iapf_log_normal(points,m,diag(exp(lv),d)); scale <- max(p0)
  do.call(rbind,lapply(c(0,.25,.5,1,2,4,8,16),function(t) {
    lp <- iapf_log_normal(points,m,diag(exp(lv+t),d))
    p <- exp(lp-scale)
    lambda <- sum(p*y)/sum(y*y)
    residual <- p-lambda*y
    data.frame(input=label,d=d,variance_log_multiplier=t,
      fixed_scale_loss=mean(residual^2),relative_shape_residual=sum(residual^2)/sum(p^2),
      log_profiled_lambda=log(lambda),log_density_max=max(lp),
      density_scale=scale,variance_ratio=exp(t))
  }))
}

# Exact Gaussian data admit a finite zero residual; inflating its covariance
# damages the shape even while eventually making the scalar loss tiny again.
set.seed(924121)
controls <- list()
for(d in c(1L,5L,20L)) {
  x <- matrix(rnorm(300*d),ncol=d)
  m <- rep(.2,d); lv <- rep(log(.8),d)
  target <- iapf_log_normal(x,m,diag(exp(lv),d))
  r <- ray(x,target,c(m,lv),paste0("exact_gaussian_d",d))
  stopifnot(r$relative_shape_residual[1]<1e-27,
            tail(r$relative_shape_residual,1)>1e-3,
            tail(r$fixed_scale_loss,1)<r$fixed_scale_loss[4])
  controls[[length(controls)+1L]] <- r
}
write.csv(do.call(rbind,controls),file.path(output,"gaussian-controls.csv"),row.names=FALSE)

paths <- sort(list.files(file.path(failures,"runs"),pattern="candidate-failure.rds",recursive=TRUE,full.names=TRUE))
unique_inputs <- list(); records <- list(); rays <- list()
for(path in paths) {
  old <- readRDS(path)
  key <- list(points=old$points,targets=old$log_targets,initial=old$initial_parameters)
  matched <- which(vapply(unique_inputs,function(previous) identical(previous,key),logical(1)))
  if(length(matched)) idx <- matched[1] else {
    unique_inputs[[length(unique_inputs)+1L]] <- key
    idx <- length(unique_inputs)
    rays[[idx]] <- ray(old$points,old$log_targets,old$initial_parameters,paste0("input",idx))
  }
  records[[length(records)+1L]] <- data.frame(path=path,input=paste0("input",idx),d=ncol(old$points))
}
write.csv(do.call(rbind,records),file.path(output,"input-identities.csv"),row.names=FALSE)
all_rays <- do.call(rbind,rays)
stopifnot(all(is.finite(all_rays$fixed_scale_loss)),all(is.finite(all_rays$relative_shape_residual)))
write.csv(all_rays,file.path(output,"variance-escape-rays.csv"),row.names=FALSE)
summary <- do.call(rbind,lapply(rays,function(r) data.frame(input=r$input[1],d=r$d[1],
  initial_shape=r$relative_shape_residual[1],expanded_shape=tail(r$relative_shape_residual,1),
  loss_ratio=tail(r$fixed_scale_loss,1)/r$fixed_scale_loss[1],
  log_density_change=tail(r$log_density_max,1)-r$log_density_max[1])))
write.csv(summary,file.path(output,"escape-summary.csv"),row.names=FALSE)
writeLines(sprintf(paste0('{"check":"PASS","records":%d,"unique_fit_inputs":%d,',
  '"gaussian_nonharm_control":"PASS","diagnostic":"unrestricted_equation15_amplitude_escape",',
  '"fit_policy":"UNRESOLVED","floor_policy":"UNRESOLVED",',
  '"scientific_replication":"INCOMPLETE","full_filter":"NOT_RUN",',
  '"cpu_only":true}'),length(paths),length(unique_inputs)),file.path(output,"result.json"))
cat("PASS: exact Gaussian controls and saved-input variance rays; fit/floor unresolved\n")

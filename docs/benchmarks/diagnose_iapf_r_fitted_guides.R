# Post-run independent-reference shape diagnostic, not a filter decision path.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]; campaign <- args[3]
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
stopifnot(!dir.exists(output),Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
dir.create(output,recursive=TRUE)
kl_components <- function(exact,fitted) {
  d <- length(exact$mean)
  inv <- solve(fitted$covariance)
  displacement <- fitted$mean-exact$mean
  mean_term <- as.numeric(crossprod(displacement,inv %*% displacement))/2
  inflation <- as.numeric(determinant(fitted$covariance,logarithm=TRUE)$modulus-
    determinant(exact$covariance,logarithm=TRUE)$modulus)
  covariance_term <- (inflation-d+sum(diag(inv %*% exact$covariance)))/2
  stopifnot(is.finite(mean_term),is.finite(covariance_term),covariance_term> -1e-10)
  c(kl=mean_term+covariance_term,mean_term=mean_term,covariance_term=covariance_term,
    log_determinant_inflation=inflation)
}
check <- iapf_gaussian_twist(c(1,2),matrix(c(2,.3,.3,1),2))
stopifnot(max(abs(kl_components(check,check)))<1e-12)
out <- list(); coverage <- list()
for(attempt in list.dirs(campaign,recursive=FALSE,full.names=TRUE)) {
  if(!grepl("-validation-",basename(attempt),fixed=TRUE)) next
  path <- file.path(attempt,"results")
  if(!file.exists(file.path(path,"replicates.csv"))) next
  settings <- dget(file.path(path,"settings.R"))
  observations <- as.matrix(read.csv(file.path(path,"observations.csv")))
  A <- outer(1:settings$dimension,1:settings$dimension,function(i,j).42^(abs(i-j)+1))
  model <- iapf_linear_model(observations,A)
  exact <- iapf_exact_twists(model)
  stopifnot(max(abs(exact[[100]]$mean-observations[100,]))<1e-10,
    max(abs(exact[[100]]$covariance-diag(settings$dimension)))<1e-10)
  records <- read.csv(file.path(path,"replicates.csv"),stringsAsFactors=FALSE)
  records <- records[records$status=="complete" &
    (records$method=="qr" | startsWith(records$method,"box")),]
  for(i in seq_len(nrow(records))) {
    row <- records[i,]
    file <- file.path(path,paste0(row$method,"-",row$replication,".rds"))
    value <- readRDS(file)
    stopifnot(value$status=="complete",length(value$twists)==100)
    for(t in 1:100) out[[length(out)+1]] <- data.frame(
      attempt=basename(attempt),data_seed=settings$data_seed,method=row$method,
      replication=row$replication,time=t,as.list(kl_components(exact[[t]],value$twists[[t]])))
    coverage[[length(coverage)+1]] <- data.frame(attempt=basename(attempt),
      replication=row$replication,method=row$method,times=100)
  }
}
stopifnot(length(out)>0)
write.csv(do.call(rbind,out),file.path(output,"guide-shapes.csv"),row.names=FALSE)
write.csv(do.call(rbind,coverage),file.path(output,"coverage.csv"),row.names=FALSE)
cat("PASS KL identities, exact terminal guide and",length(coverage),"saved full filters;",
  length(out),"shape rows. Positive floors excluded from this diagnostic.\n")

# Bounded independent CPU reference experiments; no production consumers.
a <- commandArgs(trailingOnly=TRUE)
root <- a[1]; output <- a[2]; mode <- a[3]; d <- as.integer(a[4])
data_seed <- as.integer(a[5]); arm <- a[6]; rep_id <- as.integer(a[7]); maxit <- as.integer(a[8])
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(output))
dir.create(output,recursive=TRUE)
for(f in c("reference_iapf_paper.R","reference_iapf_plausible_choices.R",
  "reference_iapf_eq15_solvers.R","diagnose_iapf_r_validation_tails.R"))
  source(file.path(root,"docs/benchmarks",f))
write_rows <- function(rows,name) {
 p <- file.path(output,name); exists <- file.exists(p)
 write.table(rows,p,sep=",",append=exists,col.names=!exists,row.names=FALSE,qmethod="double")
}
data <- iapf_paper_data(d,100,seed=data_seed)
model <- iapf_linear_model(data$observations,data$A); kalman <- iapf_kalman(model)
write.csv(data$observations,file.path(output,"observations.csv"),row.names=FALSE)
write.csv(data.frame(time=1:100,prefix=kalman$prefix,innovation=kalman$innovation),
 file.path(output,"kalman.csv"),row.names=FALSE)
dput(list(mode=mode,dimension=d,data_seed=data_seed,arm=arm,replication=rep_id,
 maxit=maxit,cpu_only=TRUE,CUDA_VISIBLE_DEVICES=Sys.getenv("CUDA_VISIBLE_DEVICES"),
 method_status="independent_reference_unknown_author_solver",floor_tail_power=8,
 stopping_window=6,k=5,N0=1000,tau=.5,kappa=.5),file=file.path(output,"settings.R"))

if(mode=="fixtures") {
 exact <- iapf_exact_twists(model)
 set.seed(62100000+d*1000+rep_id)
 training <- iapf_apf(model,exact,1000)
 for(time in c(1,50,100)) {
  x <- training$clouds[[time]]
  target <- iapf_backward_log_target(model,x,time,if(time<100)exact[[time+1]] else NULL)
  saveRDS(list(points=x,log_targets=target,exact=exact[[time]]),
    file.path(output,paste0("fixture-",time,".rds")))
  for(init in c("qr","moments")) for(kind in c("joint","profiled")) {
   started <- proc.time()[[3]]
   fit <- tryCatch(iapf_eq15_solver(x,target,parameterization=kind,
      initialization=init,maxit=maxit),iapf_fit_error=function(e)e)
   status <- if(inherits(fit,"iapf_fit_error"))"candidate_failed" else "complete"
   saveRDS(fit,file.path(output,paste0("fit-",time,"-",kind,"-",init,".rds")))
   diag <- fit$diagnostics; p <- fit$parameters
   v <- exp(p[d+seq_len(d)]); delta <- p[seq_len(d)]-exact[[time]]$mean
   reference_cov <- exact[[time]]$covariance
   kl <- .5*(sum(diag(reference_cov)/v)+sum(delta^2/v)-d+
       sum(log(v))-2*sum(log(diag(chol(reference_cov)))))
   write_rows(data.frame(dimension=d,data_seed=data_seed,time=time,kind=kind,
     initialization=init,status=status,gaussian_component_KL=kl,
     seconds=proc.time()[[3]]-started,as.data.frame(diag)),"fixtures.csv")
  }
 }
} else if(mode=="probe") {
 index <- if(arm %in% c("bpf","fully_adapted","sis"))match(arm,c("qr","bpf","fully_adapted","sis")) else 1
 seed <- 61000000+d*100000+rep_id*10+index
 set.seed(seed); started <- proc.time()[[3]]
 answer <- tryCatch({
  if(arm=="qr")iapf_reconstruction(model,"delayed") else
  if(arm %in% c("bpf","fully_adapted","sis")) {
   n <- if(arm=="fully_adapted")5000 else 10000
   final <- if(arm=="fully_adapted")iapf_fully_adapted(model,n) else
    iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),n,
     if(arm=="sis")0 else .5,FALSE)
   list(status="complete",final=final,final_particles=n,history=final$log_likelihood)
  } else {
   bits <- strsplit(arm,"_",fixed=TRUE)[[1]]
   iapf_eq15_reconstruction(model,bits[1],bits[2],maxit)
  }
 },iapf_fit_error=function(e)e)
 algorithm_seconds <- proc.time()[[3]]-started
 if(inherits(answer,"iapf_fit_error") || answer$status!="complete") {
  saveRDS(answer,file.path(output,"candidate-failure.rds"))
  dg <- answer$diagnostics
  write_rows(data.frame(dimension=d,data_seed=data_seed,replication=rep_id,method=arm,
    seed=seed,status="candidate_failed",reason=if(inherits(answer,"iapf_fit_error"))
    answer$message else answer$status,algorithm_seconds=algorithm_seconds,
    backward_time=if(is.null(answer$backward_time))NA else answer$backward_time,
    optimizer_code=if(is.null(dg))NA else dg$convergence,
    optimizer_message=if(is.null(dg))"" else dg$optimizer_message),"failure.csv")
 } else {
  answer$final$clouds <- answer$final$log_weights <- answer$final$ancestors <- NULL
  saveRDS(answer,file.path(output,"answer.rds"))
  if(!is.null(answer$fits))for(i in seq_along(answer$fits))for(t in seq_along(answer$fits[[i]]))
    write_rows(data.frame(iteration=i,time=t,as.data.frame(answer$fits[[i]][[t]])),"fits.csv")
  tail_pass <- TRUE
  if(!is.null(answer$twists)) {
   tails <- iapf_validation_tail_rows(model,answer$twists,rep_id)
   write_rows(tails,"tails.csv"); tail_pass <- all(tails$passed)
  }
  write_rows(data.frame(replication=rep_id,method=arm,time=1:100,
   log_prefix_error=answer$final$prefix-kalman$prefix,
   situation=ifelse(kalman$innovation>qchisq(.9,d),"large_innovation","ordinary")),"prefixes.csv")
  ratio <- answer$final$log_likelihood-kalman$log_likelihood
  stopifnot(is.finite(ratio),is.finite(exp(ratio)))
  total <- proc.time()[[3]]-started
  write_rows(data.frame(dimension=d,data_seed=data_seed,replication=rep_id,method=arm,
   seed=seed,status="complete",log_ratio=ratio,ratio=exp(ratio),particles=answer$final_particles,
   iterations=length(answer$history),resampling_count=answer$final$resampling_count,
   tail_pass=tail_pass,algorithm_seconds=algorithm_seconds,diagnostic_io_seconds=total-algorithm_seconds,
   wall_seconds=total),"replicates.csv")
 }
} else if(mode=="oracle") {
 exact <- iapf_exact_twists(model)
 moments <- vector("list",100); m <- model$initial_mean; P <- model$initial_covariance
 for(t in 1:100) {
  if(t>1) {m <- as.vector(model$A %*% m); P <- model$A %*% P %*% t(model$A)+model$transition_covariance}
  S <- model$C %*% P %*% t(model$C)+model$R
  K <- P %*% t(model$C) %*% solve(S)
  m <- m+as.vector(K %*% (model$observations[t,]-model$C %*% m))
  D <- diag(d)-K %*% model$C; P <- D %*% P %*% t(D)+K %*% model$R %*% t(K)
  Pinv <- solve(P)
  J <- matrix(0,d,d); j <- rep(0,d)
  if(t<100) {
   inverse <- solve(model$transition_covariance+exact[[t+1]]$covariance)
   J <- t(model$A) %*% inverse %*% model$A
   j <- as.vector(t(model$A) %*% inverse %*% exact[[t+1]]$mean)
  }
  Qinv <- Pinv+J; Q <- solve(Qinv); n <- as.vector(Q %*% (Pinv %*% m+j))
  M <- 2*Pinv-Qinv; margin <- min(eigen(M,symmetric=TRUE,only.values=TRUE)$values)
  delta <- m-n; h <- as.vector(Qinv %*% delta)
  log_moment <- if(margin<=0)Inf else .5*as.numeric(determinant(Q,logarithm=TRUE)$modulus)-
   as.numeric(determinant(P,logarithm=TRUE)$modulus)-
   .5*as.numeric(determinant(M,logarithm=TRUE)$modulus)+
   .5*sum(delta*(Qinv %*% delta))+.5*sum(h*solve(M,h))
  moments[[t]] <- list(m=m,P=P,n=n,Q=Q)
  write_rows(data.frame(time=t,situation=if(kalman$innovation[t]>qchisq(.9,d))
    "large_innovation" else "ordinary",second_moment_margin=margin,
    log_second_moment=log_moment,expected_prefix_MSE=expm1(log_moment)/1000),"oracle-theory.csv")
 }
 for(rep in 1:16) {
  set.seed(62200000+d*1000+rep); value <- iapf_apf(model,exact,1000)
  stopifnot(abs(value$log_likelihood-kalman$log_likelihood)<1e-8)
  for(t in 1:100) {
   f <- moments[[t]]
   density_ratio <- iapf_log_normal(value$clouds[[t]],f$m,f$P)-
    iapf_log_normal(value$clouds[[t]],f$n,f$Q)
   discrepancy <- iapf_logmean(density_ratio)-(value$prefix[t]-kalman$prefix[t])
   stopifnot(abs(discrepancy)<1e-8)
   write_rows(data.frame(replication=rep,time=t,log_prefix_error=value$prefix[t]-kalman$prefix[t],
      identity_error=discrepancy,terminal_error=value$log_likelihood-kalman$log_likelihood),"oracle-prefixes.csv")
  }
 }
} else stop("unknown mode")
cat("COMPLETE",mode,d,arm,data_seed,rep_id,"\n")

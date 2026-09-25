# Analytic guide-family controls on preassigned fresh data; no learned fitting.
rows <- list()
for(d in c(5L,20L,80L))for(data_index in 1:2) {
  data_seed <- 92200000L+10000L*data_index+d
  dat <- iapf_paper_data(d,100,seed=data_seed)
  model <- iapf_linear_model(dat$observations,dat$A)
  truth <- iapf_kalman(model)
  full <- iapf_exact_twists(model)
  guides <- list(constant=replicate(100,iapf_constant_twist(),simplify=FALSE),
    observation=iapf_observation_twists(model),
    moment_diagonal=lapply(full,function(g)iapf_gaussian_twist(g$mean,diag(diag(g$covariance)))),
    precision_diagonal=lapply(full,function(g)iapf_gaussian_twist(g$mean,diag(1/diag(solve(g$covariance))))),
    exact_full=full)
  for(replication in 1:2)for(method in names(guides)) {
    filter_seed <- 92240000L+d+1009L*replication
    set.seed(filter_seed);start <- proc.time()[[3]]
    ans <- iapf_apf(model,guides[[method]],1000,store_clouds=FALSE)
    elapsed <- proc.time()[[3]]-start
    error <- ans$log_likelihood-truth$log_likelihood
    check(paste0("finite_",d,"_",data_index,"_",replication,"_",method),
      as.numeric(!all(is.finite(c(ans$prefix,ans$log_likelihood)))),0)
    if(method=="exact_full")check(paste0("exact_Kalman_",d,"_",data_index,"_",replication),abs(error),1e-7)
    rows[[length(rows)+1L]] <- data.frame(dimension=d,data_seed=data_seed,
      replication=replication,filter_seed=filter_seed,method=method,N=1000,T=100,
      log_likelihood=ans$log_likelihood,kalman=truth$log_likelihood,
      terminal_error=error,prefix_error_max=max(abs(ans$prefix-truth$prefix)),
      min_ess=min(ans$ess),resampling_count=ans$resampling_count,wall_seconds=elapsed)
    saveRDS(list(model=model,guide=guides[[method]],filter=ans,kalman=truth),
      file.path(out,paste0("d",d,"-j",data_index,"-r",replication,"-",method,".rds")))
    write_table(rows,"analytic-controls.csv")
  }
  cat("analytic d",d,"data",data_index,"complete\n")
}

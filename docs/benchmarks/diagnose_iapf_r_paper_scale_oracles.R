# Paper-dimensional mathematical controls; never fitted-iAPF replication.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]; repeats <- as.integer(args[4])
first <- as.integer(args[5]); base_seed <- as.integer(args[6])
source(file.path(root,'docs/benchmarks/reference_iapf_paper.R'))
stopifnot(!dir.exists(output),args[7]=='oracle',repeats==3)
dir.create(output,recursive=TRUE)
rows <- list()
for(d in c(5,10,20,40,80)) {
  data <- iapf_paper_data(d,100,seed=base_seed+d)
  model <- iapf_linear_model(data$observations,data$A)
  truth <- iapf_kalman(model)$log_likelihood
  optimal <- iapf_exact_twists(model)
  one_step <- iapf_observation_twists(model)
  write.csv(data$observations,file.path(output,paste0('observations-d',d,'.csv')),row.names=FALSE)
  for(replication in first:(first+repeats-1)) {
    seed <- 59000000+d*1000+replication
    start <- proc.time()[[3]]
    cat('BEGIN',d,replication,'\n');flush.console()
    set.seed(seed)
    oracle <- iapf_apf(model,optimal,1000,.5,TRUE)
    log_error <- oracle$log_likelihood-truth
    weight_spread <- diff(range(oracle$log_weights[[100]]))
    set.seed(seed)
    adapted <- iapf_fully_adapted(model,5000)
    set.seed(seed)
    observation <- iapf_apf(model,one_step,5000,.5,FALSE)
    adapted_difference <- adapted$log_likelihood-observation$log_likelihood
    adapted_prefix_difference <- max(abs(adapted$prefix-observation$prefix))
    passed <- abs(log_error)<=1e-10*(1+abs(truth)) && weight_spread<=1e-9 &&
      abs(adapted_difference)<=1e-10*(1+abs(truth)) &&
      adapted_prefix_difference<=1e-10*(1+abs(truth)) &&
      adapted$resampling_count==observation$resampling_count
    rows[[length(rows)+1]] <- data.frame(dimension=d,replication=replication,
      data_seed=base_seed+d,particle_seed=seed,kalman=truth,optimal_log_error=log_error,
      terminal_weight_spread=weight_spread,optimal_resampling=oracle$resampling_count,
      fully_adapted_path_difference=adapted_difference,
      fully_adapted_prefix_difference=adapted_prefix_difference,
      fully_adapted_resampling=adapted$resampling_count,
      observation_twist_resampling=observation$resampling_count,
      passed=passed,wall_seconds=proc.time()[[3]]-start,cpu_only=TRUE,
      role='exact_twist_and_one_step_oracles_not_fitted_iapf')
    write.csv(do.call(rbind,rows),file.path(output,'oracle-checks.csv'),row.names=FALSE)
    cat('END',d,replication,'passed',passed,'\n');flush.console()
    if(!passed)quit(status=2)
  }
}
cat('COMPLETE all five dimensions\n')

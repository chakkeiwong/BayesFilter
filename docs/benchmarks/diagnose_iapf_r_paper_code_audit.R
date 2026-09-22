# Frozen-input replays and independent support checks, not method selection.
args <- commandArgs(TRUE); root <- args[1]; output <- args[2]
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_support_diagnostics.R"))
stopifnot(!dir.exists(output)); dir.create(output,recursive=TRUE)
cases <- data.frame(id=c("lbfgs_d20_t92","nlminb_d20_t68"),
  attempt=c("attempt08-pilot-d20","attempt09-pilot-d20-nlminb"),
  data_seed=c(66000020,67000020),fit_mode=c("relative_l2","relative_l2_nlminb"))
rows <- list(); replays <- list()
for(case in seq_len(nrow(cases))) {
  settings <- cases[case,]; cat("REPLAY",settings$id,"\n"); flush.console()
  old <- readRDS(file.path(root,"docs/plans/artifacts/iapf-r-reference-gap-repair-20260920-01",
    settings$attempt,"results/iapf-1-failure.rds"))
  data <- iapf_paper_data(20,100,seed=settings$data_seed)
  write.csv(data$observations,file.path(output,paste0("observations-",settings$id,".csv")),row.names=FALSE)
  model <- iapf_linear_model(data$observations,data$A)
  set.seed(55000011); started <- proc.time()[[3]]
  failed <- tryCatch(iapf_iterate(model,fit_mode=settings$fit_mode,fit_maxit=5000),
                    iapf_fit_error=function(e)e)
  stopifnot(inherits(failed,"iapf_fit_error"),failed$backward_time==old$backward_time,
    identical(failed$points,old$points),identical(failed$log_targets,old$log_targets),
    identical(failed$parameters,old$parameters))
  saveRDS(failed,file.path(output,paste0(settings$id,"-failure.rds")))
  time <- failed$backward_time; N <- failed$N; d <- ncol(failed$points)
  replays[[case]] <- data.frame(case=settings$id,data_seed=settings$data_seed,
    particle_seed=55000011,iteration=failed$iteration,time=time,N=N,
    training_inputs_match=TRUE,parameters_match=TRUE,
    optimizer_code=failed$optimizer_code,wall_seconds=proc.time()[[3]]-started)
  write.csv(do.call(rbind,replays),file.path(output,"replay.csv"),row.names=FALSE)
  predictions <- iapf_linear_predictive_marginals(model)
  smoothing <- iapf_linear_smoothing_marginal(model,time,predictions,iapf_exact_twists(model))
  mixture <- iapf_backward_target_mixture(model,time,failed$next_twist)
  candidates <- list(failed_fit=failed$parameters,initial_fit=failed$initial_parameters,
    diagonal_target_moments=c(mixture$mean,log(diag(mixture$covariance))),constant=NULL)
  evaluate <- function(points,support,seed) {
    target <- iapf_backward_log_target(model,points,time,failed$next_twist)
    for(name in names(candidates)) {
      parameters <- candidates[[name]]
      if(name=="constant") {
        gaussian <- floored <- rep(0,nrow(points)); log_floor <- -Inf
      } else {
        variance <- exp(parameters[d+seq_len(d)])
        gaussian <- iapf_log_normal(points,parameters[seq_len(d)],diag(variance,d))
        log_floor <- -.5*(d*log(2*pi)+sum(log(variance)))-
          .5*qchisq(N^(-failed$floor_tail_power),df=d,lower.tail=FALSE)
        floored <- iapf_logadd(gaussian,log_floor)
      }
      for(component in c("gaussian","gaussian_plus_floor")) {
        lp <- if(component=="gaussian")gaussian else floored
        rows[[length(rows)+1]] <<- data.frame(case=settings$id,iteration=failed$iteration,
          time=time,support=support,seed=seed,N=nrow(points),candidate=name,component=component,
          as.data.frame(iapf_support_metrics(lp,target)),
          floor_dominated_fraction=if(name=="constant")0 else mean(gaussian<log_floor))
      }
    }
  }
  evaluate(failed$points,"training",55000011)
  for(repetition in 1:6) {
    seed <- 68000000+case*1000+repetition
    set.seed(seed)
    independent <- iapf_apf(model,failed$filter_twists,N,failed$kappa,TRUE)$clouds[[time]]
    evaluate(independent,"independent_apf",seed)
    set.seed(seed+100)
    points <- iapf_draw_normal(matrix(rep(smoothing$mean,each=N),N,d),smoothing$covariance)
    evaluate(points,"exact_smoothing",seed+100)
  }
  write.csv(do.call(rbind,rows),file.path(output,"support.csv"),row.names=FALSE)
  cat("CHECKED",settings$id,"support rows",length(rows),"\n"); flush.console()
}
cat("COMPLETE two replays and 24 independent support clouds\n")

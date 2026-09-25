# CPU-only independent reference extensions, not Eq15 or original-author code.
args <- commandArgs(trailingOnly=TRUE)
out <- args[1]; d <- as.integer(args[2]); batch <- as.integer(args[3])
source('docs/benchmarks/reference_iapf_paper.R')
source('docs/benchmarks/reference_iapf_author_choices.R')
source('docs/benchmarks/diagnostic_iapf_paper_score.R')
options(digits=17)
stopifnot(d%in%c(5,10,20,40,80),batch>=1L,Sys.getenv('CUDA_VISIBLE_DEVICES')=='-1')
cpu_time <- function() sum(proc.time()[1:2])
write_rows <- function(rows,name) write.csv(do.call(rbind,rows),file.path(out,name),row.names=FALSE)
T <- 100L; N0 <- 1000L
labels <- if(length(args)>=5L)seq.int(as.integer(args[4]),as.integer(args[5])) else if(batch==1L)1:5 else 6:10
data_seed <- if(length(args)>=6L)as.integer(args[6]) else 961000+d
learner_seed_base <- if(length(args)>=7L)as.integer(args[7]) else 962000+100*d
heuristic_seed_base <- if(length(args)>=8L)as.integer(args[8]) else 963000+100*d
conventions <- if(length(args)>=9L)args[9] else c('after_k','first_full_window')
stopifnot(all(conventions%in%c('after_k','first_full_window')),all(labels>=1L),length(labels)<=10L)
seeds <- list(data=data_seed,learner=learner_seed_base+labels,heuristic=heuristic_seed_base+labels)
dat <- iapf_paper_data(d,T,seed=seeds$data)
model <- iapf_linear_model(dat$observations,dat$A)
saveRDS(list(data=dat,seeds=seeds,labels=labels,N0=N0,T=T,k=5,tau=.5,kappa=.5,
  max_iterations=12,max_particles=4000,stopping_window=6,cv_sd_mode='sample',conventions=conventions,
  model_matrices=list(A=model$A,H=model$C,Q=model$transition_covariance,R=model$R,
    m1=model$initial_mean,P1=model$initial_covariance)),file.path(out,'inputs.rds'))
kalman <- iapf_kalman(model)$log_likelihood
constant <- replicate(T,iapf_constant_twist(),simplify=FALSE)
oracle <- iapf_exact_twists(model); observation <- iapf_observation_twists(model)

run_learner <- function(fitter,convention,seed,label) {
  set.seed(seed); started <- cpu_time()
  fit_history <- list(); filter_calls <- list(); previous <- NULL
  # This wrapper records calls; the frozen consumer still performs every APF operation.
  recorded_filter <- function(model,twists,N,kappa,store_clouds) {
    filtered <- iapf_apf(model,twists,N,kappa,store_clouds)
    filter_calls[[length(filter_calls)+1L]] <<- list(N=N,
      log_likelihood=filtered$log_likelihood,resampling_count=filtered$resampling_count,
      min_ess=min(filtered$ess),floor_probability_max=filtered$floor_probability_max,
      floor_probability_quantiles=filtered$floor_probability_quantiles)
    filtered
  }
  recorded_fit <- function(model,clouds,N,maxit,floor_tail_power) {
    fit_started <- cpu_time()
    if(fitter=='score') {
      actual <- iapf_score_backward(model,clouds,N,'diagonal','tail8')
      limit <- iapf_score_backward(model,clouds,N,'diagonal','negligible')
      differences <- if(actual$valid && limit$valid) vapply(seq_len(T),function(t)
        max(abs(actual$twists[[t]]$mean-limit$twists[[t]]$mean),
          abs(actual$twists[[t]]$covariance-limit$twists[[t]]$covariance)),numeric(1)) else rep(NA_real_,T)
      fit_history[[length(fit_history)+1L]] <<- list(N=N,actual=actual,
        negligible_valid=limit$valid,negligible_reason=if(limit$valid)'' else limit$reason,
        negligible_diagnostics=limit$diagnostics,coefficient_differences=differences,
        max_coefficient_difference=if(all(is.finite(differences)))max(differences) else NA_real_,
        cpu_seconds=cpu_time()-fit_started)
      if(!actual$valid) stop(structure(list(message=paste('score fit rejected:',actual$reason),
        call=NULL,backward_time=actual$backward_time,diagnostics=actual$diagnostics),
        class=c('iapf_fit_error','error','condition')))
      # The negligible-floor recursion above is explanatory and never supplies a guide.
      return(list(twists=actual$twists,diagnostics=actual$diagnostics))
    }
    actual <- tryCatch(iapf_choice_backward(model,clouds,N,'qr',previous,'tail8',maxit),
      iapf_fit_error=function(e)e)
    fit_history[[length(fit_history)+1L]] <<- list(N=N,actual=actual,
      cpu_seconds=cpu_time()-fit_started)
    if(inherits(actual,'iapf_fit_error'))stop(actual)
    previous <<- actual$twists
    actual
  }
  # fit_mode selects the callback signature, not its scientific identity.
  answer <- tryCatch(iapf_iterate(model,N0=N0,k=5,tau=.5,kappa=.5,
    max_iterations=12,max_particles=4000,fit_maxit=200,floor_tail_power=8,
    .filter=recorded_filter,.fit=recorded_fit,fit_mode='paper_eq15',
    doubling_mode=convention,stopping_window=6,cv_sd_mode='sample'),
    iapf_fit_error=function(e)e)
  elapsed <- cpu_time()-started
  rejected <- inherits(answer,'iapf_fit_error')
  status <- if(rejected)'fit_rejection' else answer$status
  history <- if(rejected)answer$likelihood_history else answer$history
  counts <- if(rejected)answer$particle_history else answer$counts
  complete <- status=='complete'
  wiring <- length(filter_calls)==length(history)+as.integer(complete)
  if(complete) wiring <- wiring && answer$final$log_likelihood==tail(filter_calls,1)[[1]]$log_likelihood &&
    answer$stop_iteration+1==length(history) && length(fit_history)==answer$stop_iteration
  stopifnot(wiring,length(history)==length(counts),all(is.finite(history)),elapsed>=0)
  final <- if(complete)answer$final else NULL
  if(complete) {
    stopifnot(is.finite(final$log_likelihood),all(vapply(final$clouds,function(x)all(is.finite(x)),logical(1))))
    # Keep full diagnostics and all fitted guides; reproduce raw clouds from pinned seeds.
    final$clouds <- final$log_weights <- final$ancestors <- NULL
    answer$final <- final
  }
  identity <- paste(fitter,convention,sep='_')
  saveRDS(list(method=identity,seed=seed,replicate=label,result=answer,
    fit_history=fit_history,filter_calls=filter_calls,actual_controller='iapf_iterate',
    controller_wiring=wiring,cpu_seconds=elapsed,raw_cloud_policy='reproduce_from_seed'),
    file.path(out,sprintf('learner_%s_r%02d.rds',identity,label)))
  fits <- list()
  if(fitter=='score') for(i in seq_along(fit_history)) {
    fit <- fit_history[[i]]
    for(t in seq_len(T)) {
      info <- fit$actual$diagnostics[[t]]
      get <- function(key)if(is.null(info[[key]]))NA_real_ else info[[key]]
      fits[[length(fits)+1L]] <- data.frame(d=d,replicate=label,method=identity,
        fit_call=i,time=t,N=fit$N,actual_valid=fit$actual$valid,negligible_valid=fit$negligible_valid,
        coefficient_difference=fit$coefficient_differences[t],
        gaussian_probability_min=get('target_gaussian_probability_min'),
        gaussian_probability_mean=get('target_gaussian_probability_mean'),
        cloud_margin=get('cloud_margin'),precision_margin=get('precision_margin'),
        guard_tolerance=get('guard_tolerance'),score_residual=get('score_residual'))
    }
  }
  if(length(fits))write_rows(fits,sprintf('fit_diagnostics_%s_r%02d.csv',identity,label))
  log_error <- if(complete)final$log_likelihood-kalman else NA_real_
  data.frame(d=d,batch=batch,replicate=label,method=identity,status=status,
    data_seed=seeds$data,run_seed=seed,particles=if(complete)answer$final_particles else tail(counts,1),
    stop_iteration=if(complete)answer$stop_iteration else NA_integer_,
    filter_calls=length(filter_calls),fit_calls=length(fit_history),
    particle_filter_work=sum(vapply(filter_calls,function(x)x$N,numeric(1))),
    controller_wiring=wiring,log_likelihood=if(complete)final$log_likelihood else NA_real_,
    kalman=kalman,log_error=log_error,ratio=if(complete)exp(log_error) else NA_real_,
    resampling_count=if(complete)final$resampling_count else NA_integer_,
    min_ess=if(complete)min(final$ess) else NA_real_,
    floor_probability_max=if(complete)final$floor_probability_max else NA_real_,
    cpu_seconds=elapsed,reason=if(rejected)conditionMessage(answer)else'')
}

rows <- list()
for(label in labels) {
  for(fitter in c('score','qr')) for(convention in conventions) {
    row <- run_learner(fitter,convention,learner_seed_base+label,label)
    rows[[length(rows)+1L]] <- row;write_rows(rows,'partial-records.csv')
    cat('learner',d,label,row$method,row$status,'N',row$particles,'\n');flush.console()
  }
  for(method in c('bootstrap','fully_adapted','current_observation','full_oracle')) {
    seed <- heuristic_seed_base+label;set.seed(seed);started <- cpu_time()
    N <- if(method=='bootstrap')10000 else if(method=='fully_adapted')5000 else N0
    final <- if(method=='fully_adapted')iapf_fully_adapted(model,N,.5) else
      iapf_apf(model,switch(method,bootstrap=constant,current_observation=observation,full_oracle=oracle),N,.5,FALSE)
    elapsed <- cpu_time()-started;error <- final$log_likelihood-kalman
    stopifnot(is.finite(error),elapsed>=0)
    if(method=='full_oracle')stopifnot(abs(error)<=1e-8)
    saveRDS(final,file.path(out,sprintf('heuristic_%s_r%02d.rds',method,label)))
    rows[[length(rows)+1L]] <- data.frame(d=d,batch=batch,replicate=label,method=method,
      status='complete',data_seed=seeds$data,run_seed=seed,particles=N,stop_iteration=NA_integer_,
      filter_calls=1,fit_calls=0,particle_filter_work=N,controller_wiring=TRUE,
      log_likelihood=final$log_likelihood,kalman=kalman,log_error=error,ratio=exp(error),
      resampling_count=final$resampling_count,min_ess=if(is.null(final$ess))NA_real_ else min(final$ess),
      floor_probability_max=if(is.null(final$floor_probability_max))NA_real_ else final$floor_probability_max,
      cpu_seconds=elapsed,reason='')
    write_rows(rows,'partial-records.csv')
  }
}
write_rows(rows,'records.csv')
stopifnot(length(rows)==length(labels)*(2L*length(conventions)+4L))
write.csv(data.frame(d=d,batch=batch,data_seed=seeds$data,N0=N0,T=T,kalman=kalman,
  record_count=length(rows),actual_controller='iapf_iterate',cpu_only=TRUE),
  file.path(out,'checks.csv'),row.names=FALSE)
cat('Completed adaptive batch:',d,batch,'records',length(rows),'\n')

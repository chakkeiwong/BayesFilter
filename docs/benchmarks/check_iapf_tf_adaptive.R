# Entrypoint for independent reference comparison; no contributed R packages.
args <- commandArgs(trailingOnly=TRUE); stopifnot(length(args)==3)
source("docs/benchmarks/reference_iapf_tf_adaptive.R")
source(args[2]); out <- args[3]; checks <- list()
check <- function(name,actual,expected,atol=0,rtol=0,note="") {
  a <- as.numeric(unlist(actual)); b <- as.numeric(unlist(expected))
  same_length <- length(a)==length(b) && length(a)>0
  finite <- same_length && all(is.finite(a)) && all(is.finite(b))
  err <- if(finite) max(abs(a-b)) else Inf
  threshold <- if(finite) atol+rtol*abs(b) else 0
  pass <- finite && all(abs(a-b)<=threshold)
  checks[[length(checks)+1]] <<- data.frame(name=name,pass=pass,max_abs_error=err,
    max_allowed=if(finite)max(threshold)else 0,note=note)
  invisible(pass)
}
flag <- function(name,ok,note="") check(name,as.integer(isTRUE(ok)),1,note=note)
finish <- function(extra) {
  table <- do.call(rbind,checks)
  write.table(table,file.path(out,"R-checks.tsv"),sep="\t",row.names=FALSE,quote=TRUE)
  saveRDS(extra,file.path(out,"R-reference.rds"))
  writeLines(capture.output(sessionInfo()),file.path(out,"R-session.txt"))
  cat(sprintf("%s: %d/%d checks passed\n",args[1],sum(table$pass),nrow(table)))
  if(!all(table$pass)) print(table[!table$pass,])
  quit(status=if(all(table$pass))0L else 2L)
}

if(args[1]=="preflight") {
  reference <- list()
  for(f in input$fixtures) {
    d <- f$d; points <- tfref_matrix(f$points,length(f$points),d); targets <- unlist(f$targets)
    fit <- tfref_fit(points,targets,input$config)
    z <- sweep(sweep(points,2,colMeans(points),"-"),2,sqrt(colMeans(sweep(points,2,colMeans(points),"-")^2)),"/")
    target <- targets-max(targets); par <- unlist(f$parameters); profile <- tfref_profile(z,target,par)
    prefix <- paste0("d",d,"-")
    flag(paste0(prefix,"TF-valid-converged"),f$valid&&f$converged)
    flag(paste0(prefix,"R-valid-converged"),fit$valid&&fit$converged)
    check(paste0(prefix,"center"),f$center,fit$center,2e-5,1e-6)
    check(paste0(prefix,"covariance"),unlist(f$covariance),as.vector(t(fit$covariance)),2e-5,1e-6)
    check(paste0(prefix,"floor"),f$floor,fit$log_floor,2e-5,1e-6)
    check(paste0(prefix,"known-center"),fit$center,f$expected_center,2e-5)
    check(paste0(prefix,"known-variance"),diag(fit$covariance),f$expected_variance,2e-5)
    check(paste0(prefix,"profile-loss"),f$profile_loss,profile$loss,1e-12,1e-10)
    check(paste0(prefix,"profile-gradient"),f$profile_gradient,profile$gradient,1e-11,1e-9)
    for(h in c(1e-5,5e-6)) {
      fd <- vapply(seq_along(par),function(k) {
        delta <- rep(0,length(par)); delta[k] <- h
        (tfref_profile(z,target,par+delta)$loss-tfref_profile(z,target,par-delta)$loss)/(2*h)
      },0.)
      check(paste0(prefix,"independent-gradient-FD-",h),profile$gradient,fd,1e-9,1e-7)
    }
    rec <- f$recursive
    recursive <- tfref_backward(unlist(input$config$fit_theta),tfref_matrix(rec$observations,2,d),
      lapply(rec$clouds,tfref_matrix,nr=128,nc=d),input$config,d,d)
    flag(paste0(prefix,"recursive-TF-valid-converged"),rec$valid&&rec$converged)
    flag(paste0(prefix,"recursive-R-valid-converged"),recursive$valid&&recursive$converged)
    check(paste0(prefix,"recursive-centers"),rec$centers,tfref_flat_guides(recursive$guides,"mean"),2e-5,1e-6)
    check(paste0(prefix,"recursive-covariances"),rec$covariances,tfref_flat_guides(recursive$guides,"covariance"),2e-5,1e-6)
    check(paste0(prefix,"recursive-floors"),rec$floors,tfref_flat_guides(recursive$guides,"log_floor"),2e-5,1e-6)
    reference[[prefix]] <- fit
  }
  finish(reference)
}

stopifnot(args[1]=="consumer")
d <- input$d; o <- input$o; config <- input$config; theta <- unlist(config$fit_theta)
T <- input$settings$horizon; N <- input$settings$particles
histories <- counts <- numeric(); guides <- NULL; saved <- list(); decisions <- list(); all_fits <- list()
offline <- length(input$diagnostics$fit_iterations)
flag("actual-consumer-recorded",input$actual_consumer=="bayesfilter.score_study.iapf_adapter.execute_iapf")
flag("offline-calls-present",offline>0 && length(input$runs)>=offline)
compare_filter <- function(prefix,run,result) {
  check(paste0(prefix,"value"),run$value,result$value,1e-7,1e-8)
  check(paste0(prefix,"clouds"),run$clouds,tfref_flat_cloud(result$clouds),1e-7,1e-8)
  check(paste0(prefix,"ancestor-labels"),run$trace$ancestor_indices,result$labels)
  check(paste0(prefix,"ancestor-CDF"),run$trace$ancestor_cdf,unlist(result$cdfs),1e-7,1e-8)
  check(paste0(prefix,"mixture-probabilities"),run$trace$gaussian_probability,unlist(result$probabilities),1e-7,1e-8)
  flag(paste0(prefix,"trace-bitwise-neutral"),run$trace_bitwise_equal)
}
for(j in seq_len(offline)) {
  run <- input$runs[[j]]; prefix <- paste0("iteration",j-1,"-")
  check(paste0(prefix,"particle-count"),run$particles,N)
  observations <- tfref_matrix(run$input$observations,T,o)
  result <- tfref_filter(theta,observations,run$input,guides,d,o,constant=j==1)
  compare_filter(prefix,run,result)
  histories <- c(histories,result$value); counts <- c(counts,N)
  decision <- tfref_decision(histories,counts,config)
  tfdecision <- input$diagnostics$fit_iterations[[j]]
  flag(paste0(prefix,"decision"),decision$action==tfdecision$action)
  check(paste0(prefix,"next-particles"),decision$next_particles,tfdecision$next_particles)
  if(is.finite(decision$cv)) check(paste0(prefix,"CV"),decision$cv,tfdecision$cv,1e-7,1e-8)
  saved[[j]] <- result; decisions[[j]] <- decision
  if(j<=length(input$fits)) {
    fit <- tfref_backward(theta,observations,result$clouds,config,d,o)
    tffit <- input$fits[[j]]
    check(paste0(prefix,"fitted-centers"),tffit$centers,tfref_flat_guides(fit$guides,"mean"),2e-5,1e-6)
    check(paste0(prefix,"fitted-covariances"),tffit$covariances,tfref_flat_guides(fit$guides,"covariance"),2e-5,1e-6)
    check(paste0(prefix,"fitted-floors"),tffit$floors,tfref_flat_guides(fit$guides,"log_floor"),2e-5,1e-6)
    flag(paste0(prefix,"fit-valid-status"),identical(fit$valid,tffit$valid))
    flag(paste0(prefix,"fit-converged-status"),identical(fit$converged,tffit$converged))
    guides <- fit$guides; all_fits[[j]] <- fit
  }
  N <- decision$next_particles
}

fd <- NULL
if(input$status=="complete") {
  flag("R-stopped",tail(decisions,1)[[1]]$action=="final")
  flag("all-R-fits-valid-converged",all(vapply(all_fits,function(x)x$valid&&x$converged,TRUE)))
  flag("final-call-present",length(input$runs)==offline+1)
  run <- tail(input$runs,1)[[1]]; observations <- tfref_matrix(run$input$observations,T,o)
  result <- tfref_filter(unlist(run$input$theta),observations,run$input,guides,d,o)
  compare_filter("final-",run,result)
  fd_ladder <- list(); stable_pair <- FALSE
  for(base_h in c(1e-5,1e-6,1e-7,1e-8)) {
    fd <- matrix(NA_real_,2,6); stable_pair <- TRUE
    for(s in 1:2) for(k in 1:6) {
      h <- base_h/2^(s-1); delta <- rep(0,6); delta[k] <- h
      a <- tfref_filter(theta+delta,observations,run$input,guides,d,o)
      b <- tfref_filter(theta-delta,observations,run$input,guides,d,o)
      stable <- identical(a$labels,result$labels)&&identical(b$labels,result$labels)&&
        identical(a$branches,result$branches)&&identical(b$branches,result$branches)
      stable_pair <- stable_pair && stable
      fd[s,k] <- (a$value-b$value)/(2*h)
      fd_ladder[[length(fd_ladder)+1]] <- data.frame(base_step=base_h,step=h,direction=k,
        labels_stable=stable,derivative=fd[s,k],TF_score=unlist(run$score)[k],
        base_choice_margin=result$min_margin)
    }
    if(stable_pair) break
  }
  write.table(do.call(rbind,fd_ladder),file.path(out,"R-FD-ladder.tsv"),sep="\t",row.names=FALSE,quote=TRUE)
  flag("FD-label-stable-pair",stable_pair)
  if(stable_pair) {
    check("FD-two-step-consistency",fd[1,],fd[2,],2e-5,2e-5)
    check("final-frozen-guide-score",run$score,fd[2,],2e-5,2e-5)
  }
} else {
  flag("matched-failure-path",any(!vapply(all_fits,function(x)x$valid&&x$converged,TRUE)) ||
       tail(decisions,1)[[1]]$action=="capacity_veto" || offline==config$max_iterations,
       "Failure-path parity does not establish a successful adaptive consumer.")
}
finish(list(runs=saved,fits=all_fits,decisions=decisions,final=if(exists("result"))result else NULL,fd=fd))

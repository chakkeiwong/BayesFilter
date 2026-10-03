# Post-run independent-reference diagnostics; never used for filter decisions.
iapf_validation_tail_rows <- function(model,twists,replication) {
  stopifnot(length(twists)==model$horizon)
  observation <- t(model$C) %*% solve(model$R,model$C)
  rows <- vector("list",model$horizon)
  for (time in seq_len(model$horizon)) {
    Q <- if(time==1)model$initial_covariance else model$transition_covariance
    H <- observation
    if(time<model$horizon) H <- H+t(model$A) %*%
      solve(model$transition_covariance+twists[[time+1]]$covariance,model$A)
    Qinv <- solve(Q); Vinv <- solve(twists[[time]]$covariance)
    M <- Qinv+2*H-Vinv
    values <- eigen(M,symmetric=TRUE,only.values=TRUE)$values
    scale <- norm(Qinv,"2")+2*norm(H,"2")+norm(Vinv,"2")
    relative <- min(values)/scale
    threshold <- 100*model$dimension*.Machine$double.eps
    rows[[time]] <- data.frame(replication=replication,time=time,
      minimum_eigenvalue=min(values),scale=scale,relative_margin=relative,
      threshold=threshold,passed=is.finite(relative) && relative>threshold)
  }
  do.call(rbind,rows)
}

iapf_validation_diagnose <- function(root,results) {
  source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
  settings <- dget(file.path(results,"settings.R"))
  stopifnot(settings$fit_mode=="log_quadratic",settings$floor_tail_power==8,
    settings$mode=="replication",settings$repeats==32,settings$horizon==100,
    settings$N0==1000,settings$BPF_N==10000,settings$FA_N==5000,settings$SIS_N==10000,
    settings$k==5,settings$tau==.5,settings$kappa==.5,settings$fit_maxit==200,
    settings$doubling_mode=="first_full_window",!settings$equation_15_objective,
    settings$cpu_only,settings$noncanonical_reference,!settings$original_author_data)
  output <- file.path(results,"diagnostics")
  stopifnot(!dir.exists(output)); dir.create(output)
  write.csv(data.frame(name=names(settings),value=vapply(settings,as.character,"")),
    file.path(output,"settings.csv"),row.names=FALSE)
  data <- iapf_paper_data(settings$dimension,100,seed=settings$data_seed)
  observed <- as.matrix(read.csv(file.path(results,"observations.csv")))
  stopifnot(isTRUE(all.equal(unname(observed),unname(data$observations),tolerance=1e-12)))
  model <- iapf_linear_model(data$observations,data$A)
  summary <- list(); all_passed <- TRUE
  for (replication in settings$first_replication+seq_len(settings$repeats)-1L) {
    fitted <- readRDS(file.path(results,paste0("iapf-",replication,".rds")))
    stopifnot(fitted$status=="complete",length(fitted$history)<=20,
      fitted$final_particles<=16000,all(is.finite(fitted$history)),
      is.finite(fitted$final$log_likelihood))
    rows <- iapf_validation_tail_rows(model,fitted$twists,replication)
    path <- file.path(output,"tail-margins.csv"); exists <- file.exists(path)
    write.table(rows,path,append=exists,col.names=!exists,row.names=FALSE,sep=",")
    all_passed <- all_passed && all(rows$passed)
    summary[[length(summary)+1L]] <- data.frame(replication=replication,
      iterations=length(fitted$history),final_particles=fitted$final_particles,
      log_likelihood=fitted$final$log_likelihood,
      min_ess=min(fitted$final$ess),max_ess=max(fitted$final$ess),
      floor_probability_max=fitted$final$floor_probability_max,
      minimum_tail_margin=min(rows$relative_margin),tail_passed=all(rows$passed))
  }
  write.csv(do.call(rbind,summary),file.path(output,"final-guides.csv"),row.names=FALSE)
  cat("COMPLETE tail diagnostics",settings$repeats,"replications; all passed",all_passed,"\n")
  invisible(all_passed)
}

if(sys.nframe()==0L) {
  args <- commandArgs(trailingOnly=TRUE)
  stopifnot(length(args)==2)
  iapf_validation_diagnose(args[1],args[2])
}

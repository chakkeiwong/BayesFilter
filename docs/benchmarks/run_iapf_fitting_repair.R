# Independent CPU reference/diagnostic driver. No production/default claims.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)>=2,Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(args[2]))
stage <- args[1];out <- args[2];dir.create(out,recursive=TRUE)
source("docs/benchmarks/reference_iapf_paper.R")
source("docs/benchmarks/reference_iapf_author_choices.R")
source("docs/benchmarks/reference_iapf_constrained_diagnostic.R")
old <- "docs/plans/artifacts/iapf-r-author-choice-hypotheses-20260922-01"
write_table <- function(rows,name) {
  if(length(rows))write.csv(do.call(rbind,rows),file.path(out,name),row.names=FALSE)
}
checks <- list()
check <- function(name,error,tolerance=1e-6) {
  checks[[length(checks)+1L]] <<- data.frame(check=name,error=error,tolerance=tolerance,
    pass=is.finite(error) && error<=tolerance)
  write_table(checks,"checks.csv")
  stopifnot(tail(checks,1)[[1]]$pass)
}
v <- function(x,key,default=NA)if(is.null(x[[key]]))default else x[[key]]

if(stage=="checks") {
  set.seed(92250001);x <- matrix(rnorm(600),200,3)
  target <- iapf_log_normal(x,c(.2,-.3,.1),diag(c(.7,1.2,.9)))
  for(exponent in c(0,.5,1)) {
    design <- iapf_diagnostic_design(x,target,exponent)
    raw <- iapf_diagnostic_qp(design,FALSE,FALSE)
    bounded <- iapf_diagnostic_qp(design,TRUE,FALSE)
    ridged <- iapf_diagnostic_qp(design,TRUE,TRUE)
    check(paste0("healthy_constraint_no_change_",exponent),
      max(abs(raw$coefficients-bounded$coefficients)),0)
    for(mode in c("raw","bounded","ridged")) {
      f <- get(mode);check(paste0(mode,"_KKT_",exponent),f$diagnostics$kkt_residual)
      recovered <- as.vector(design$D %*% f$coefficients)
      check(paste0(mode,"_exact_diagonal_",exponent),max(abs(recovered-design$y)))
    }
    delta <- rep(.003,ncol(design$D));h <- 1e-6
    numeric_gradient <- sapply(seq_along(delta),function(j) {
      plus <- minus <- delta;plus[j] <- plus[j]+h;minus[j] <- minus[j]-h
      (ridged$objective(plus)-ridged$objective(minus))/(2*h)
    })
    check(paste0("analytic_gradient_",exponent),
      max(abs(numeric_gradient-ridged$gradient(delta))),1e-7)
  }
  # Compare unchanged scientific outputs against the preserved pre-repair source.
  env <- new.env(parent=globalenv())
  sys.source(paste0(old,"/source-v3/docs/benchmarks/reference_iapf_author_choices.R"),env)
  previous <- iapf_gaussian_twist(c(.2,-.3,.1),diag(c(.7,1.2,.9)))
  for(arm in c("qr","f1_strict","f1_loose","f2_strict","f2_loose","wlog1","wlog2")) {
    before <- env$iapf_choice_fit(x,target,arm=arm,previous=previous)
    after <- iapf_choice_fit(x,target,arm=arm,previous=previous)
    check(paste0("legacy_parameters_",arm),max(abs(before$parameters-after$parameters)),0)
  }
  original <- iapf_fit_initial
  iapf_fit_initial <- function(...)stop("forbidden QR call")
  independent <- iapf_choice_fit(x,target,arm="f2_independent_strict",previous=previous)
  check("previous_guide_bypasses_QR",as.numeric(independent$diagnostics$initialization_requires_qr),0)
  iapf_fit_initial <- original
  active_design <- iapf_diagnostic_design(x,.2*x[,1]^2-.5*x[,2]^2-.5*x[,3]^2,0)
  active_fit <- iapf_diagnostic_qp(active_design,TRUE,TRUE)
  check("active_constraint_KKT",active_fit$diagnostics$kkt_residual)
  check("active_constraint_feasible",as.numeric(!active_fit$diagnostics$valid_constraint),0)
  check("active_constraint_exercised",as.numeric(active_fit$diagnostics$active_constraints==0),0)
  p <- ncol(active_design$A);upper <- rep(Inf,p)
  upper[1+3+1:3] <- active_fit$diagnostics$q_upper
  direct <- iapf_diagnostic_active_set(active_design$A,active_design$b,
    active_fit$anchor,0,upper,active_fit$unconstrained)
  feasible <- list()
  for(mask in 0:7) {
    active <- (1+3+1:3)[as.logical(intToBits(mask)[1:3])]
    free <- setdiff(seq_len(p),active);beta <- numeric(p);beta[active] <- upper[active]
    response <- active_design$b-if(length(active))as.vector(active_design$A[,active,drop=FALSE] %*% beta[active]) else 0
    beta[free] <- qr.solve(active_design$A[,free,drop=FALSE],response,tol=1e-12)
    if(all(beta<=upper+1e-12))feasible[[length(feasible)+1L]] <- beta
  }
  losses <- vapply(feasible,function(beta)sum((as.vector(active_design$A %*% beta)-active_design$b)^2),numeric(1))
  check("active_set_vs_exhaustive_QR_oracle",max(abs(direct$coefficients-feasible[[which.min(losses)]])),1e-8)
  check("active_set_reports_KKT_completion",direct$convergence,0)
  failing_path <- "docs/plans/artifacts/iapf-r-targeted-fitting-repair-20260922-01/attempt22-bridge/results/repaired-learning.rds"
  if(file.exists(failing_path)) {
    failed <- readRDS(failing_path)
    rescued <- iapf_constrained_diagnostic_fit(failed$points,failed$log_targets)
    check("saved_QP_failure_rescued",rescued$diagnostics$kkt_residual,1e-6)
    check("saved_QP_rescue_exercised",as.numeric(rescued$diagnostics$active_set_iterations==0),0)
    check("saved_QP_original_failure_retained",as.numeric(rescued$diagnostics$original_kkt_residual<=1e-6),0)
    saveRDS(rescued,file.path(out,"rescued-QP.rds"))
    writeLines(failing_path,file.path(out,"input-paths.txt"))
  }
  one_model <- iapf_linear_model(matrix(.3,1,1),matrix(.42,1,1))
  set.seed(92250004)
  warm_run <- iapf_choice_run(one_model,"f2_independent_loose",N0=40,k=1,max_iterations=5)
  check("independent_warm_shared_controller",as.numeric(warm_run$status!="complete" ||
    warm_run$fits[[2]][[1]]$initialization_requires_qr ||
    warm_run$fits[[2]][[1]]$start_source!="previous_iteration_independent_scales"),0)
  # Exercise the real consumer, not just the isolated fitter.
  calls <- 0L;original_fit <- iapf_constrained_diagnostic_fit
  iapf_constrained_diagnostic_fit <- function(...) {calls <<- calls+1L;original_fit(...)}
  dat <- iapf_paper_data(2,3,seed=92250002);model <- iapf_linear_model(dat$observations,dat$A)
  set.seed(92250003);run <- iapf_constrained_diagnostic_run(model,N0=64,max_particles=256)
  check("shared_controller_calls_constrained_fitter",as.numeric(calls==0),0)
  check("tiny_controller_completes",as.numeric(run$status!="complete"),0)
  saveRDS(run,file.path(out,"tiny-controller.rds"))
  cat("PASS",length(checks),"focused checks\n")
} else if(stage=="warm") {
  summary <- read.csv("docs/plans/artifacts/iapf-r-root-cause-audit-20260922-01/attempt03/results/saved-optimizer-failures.csv")
  selected <- summary[grepl("^f2",summary$method) & summary$qr_rejection,,drop=FALSE]
  check("expected_25_saved_QR_rejections",abs(nrow(selected)-25),0)
  rows <- list();input_paths <- character()
  qr_original <- iapf_fit_initial
  iapf_fit_initial <- function(...) {qr_calls <<- qr_calls+1L;qr_original(...)}
  optim <- function(...) {optimizer_calls <<- optimizer_calls+1L;stats::optim(...)}
  for(j in seq_len(nrow(selected))) {
    saved <- readRDS(selected$source[j]);input_paths <- c(input_paths,selected$source[j])
    previous <- saved$filter_twists[[saved$backward_time]]
    for(tolerance in c("strict","loose")) {
      qr_calls <- optimizer_calls <- 0L;start <- proc.time()[[3]]
      fit <- tryCatch(iapf_choice_fit(saved$points,saved$log_targets,
        arm=paste0("f2_independent_",tolerance),previous=previous),error=function(e)e)
      elapsed <- proc.time()[[3]]-start;failed <- inherits(fit,"error")
      diag <- fit$diagnostics
      check(paste0("case",j,"_",tolerance,"_no_QR"),qr_calls,0)
      check(paste0("case",j,"_",tolerance,"_optimizer_reached"),as.numeric(optimizer_calls==0),0)
      rows[[length(rows)+1L]] <- data.frame(case=j,source=selected$source[j],
        dimension=selected$dimension[j],iteration=selected$iteration[j],
        time=selected$time[j],tolerance=tolerance,status=if(failed)"rejected" else "accepted",
        message=if(failed)conditionMessage(fit) else "",qr_calls=qr_calls,
        optimizer_calls=optimizer_calls,wall_seconds=elapsed,
        initial_shape=v(diag,"initial_relative_residual"),final_shape=v(diag,"relative_residual"),
        initial_loss=v(diag,"initial_profiled_loss"),final_loss=v(diag,"profiled_loss"),
        log_density_shift=v(diag,"log_density_shift"),convergence=v(diag,"convergence"))
      saveRDS(fit,file.path(out,paste0("case",j,"-",tolerance,".rds")))
      write_table(rows,"warm-start.csv")
    }
    cat("warm case",j,"of",nrow(selected),"complete\n")
  }
  writeLines(unique(input_paths),file.path(out,"input-paths.txt"))
} else {
  source(paste0("docs/benchmarks/iapf_fitting_repair_",stage,".R"))
}
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))

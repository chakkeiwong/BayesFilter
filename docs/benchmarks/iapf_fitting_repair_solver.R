# Saved-cloud regression and fresh-region diagnostics, independent R reference.
records <- list()
for(path in sort(Sys.glob(file.path(old,"runs/*/results/replicates.csv")))) {
  tab <- read.csv(path,stringsAsFactors=FALSE)
  tab <- tab[tab$dimension==80 & grepl("^wlog",tab$method) &
    tab$status=="candidate_failed",,drop=FALSE]
  if(nrow(tab))for(j in seq_len(nrow(tab))) {
    r <- tab[j,,drop=FALSE]
    r$source <- file.path(dirname(path),paste0(r$method,"-",r$replication,"-failure.rds"))
    records[[length(records)+1L]] <- r
  }
}
records <- do.call(rbind,records)
stopifnot(nrow(records)>0)
rows <- list();shape_rows <- list();input_paths <- records$source
configs <- list(svd0=c(0,FALSE,FALSE),svd05=c(.5,FALSE,FALSE),
  svd1=c(1,FALSE,FALSE),bounded1=c(1,TRUE,FALSE),ridge_bounded1=c(1,TRUE,TRUE))

predictive_at <- function(model,time) {
  m <- model$initial_mean;C <- model$initial_covariance
  for(t in seq_len(time)) {
    if(t>1) {m <- as.vector(model$A %*% m);C <- model$A %*% C %*% t(model$A)+diag(model$dimension)}
    if(t==time)break
    gain <- C %*% solve(C+diag(model$dimension))
    m <- m+as.vector(gain %*% (model$observations[t,]-m))
    J <- diag(model$dimension)-gain
    C <- J %*% C %*% t(J)+gain %*% t(gain)
  }
  list(mean=m,covariance=C)
}
gaussian_kl <- function(m,C,other_m,other_C) {
  delta <- other_m-m;inv <- solve(other_C)
  .5*(sum(diag(inv %*% C))+sum(delta*(inv %*% delta))-length(m)+
    as.numeric(determinant(other_C,logarithm=TRUE)$modulus)-
    as.numeric(determinant(C,logarithm=TRUE)$modulus))
}

for(j in seq_len(nrow(records))) {
  saved <- readRDS(records$source[j]);x <- saved$points;b <- saved$log_targets;d <- ncol(x)
  fitted <- list();designs <- list()
  for(name in names(configs)) {
    config <- configs[[name]]
    design <- iapf_diagnostic_design(x,b,config[1]);designs[[name]] <- design
    fit <- iapf_diagnostic_qp(design,as.logical(config[2]),as.logical(config[3]))
    fitted[[name]] <- fit;diag <- fit$diagnostics
    baseline <- iapf_diagnostic_svd(design$D/sqrt(nrow(x)),design$y/sqrt(nrow(x)))$coefficients
    qi <- 1+d+seq_len(d)
    control_beta <- baseline;control_beta[qi] <- -abs(baseline[qi])-.25
    control <- design;control$y <- as.vector(design$D %*% control_beta)
    control$b <- control$y*sqrt(control$weights)
    control_fit <- iapf_diagnostic_qp(control,as.logical(config[2]),as.logical(config[3]))
    control_error <- max(abs(control_fit$coefficients-control_beta)/(1+abs(control_beta)))
    if(name=="svd0")check(paste0("case",j,"_unweighted_exact_diagonal"),control_error)
    perturbation <- 1e-10*max(1,sqrt(mean(design$y^2)))*sin(seq_len(nrow(x))*sqrt(2))
    perturb <- design;perturb$y <- design$y+perturbation
    perturb$b <- perturb$y*sqrt(perturb$weights)
    perturbed <- iapf_diagnostic_qp(perturb,as.logical(config[2]),as.logical(config[3]))
    sensitivity <- max(abs(perturbed$coefficients-fit$coefficients)/(1+abs(fit$coefficients)))
    qr <- lm.wfit(design$D,design$y,design$weights)
    rows[[length(rows)+1L]] <- data.frame(case=j,source=records$source[j],
      data_seed=records$data_seed[j],replication=records$replication[j],
      time=saved$backward_time,method=name,status=if(fit$accepted)"accepted" else "rejected",
      qr_rank=qr$rank,svd_rank=diag$svd_rank,condition=diag$design_condition,
      ess=diag$weight_ess,positive_curvatures=diag$positive_curvatures,
      active_constraints=diag$active_constraints,lambda=diag$lambda,
      kkt=diag$kkt_residual,convergence=diag$optimizer_convergence,
      exact_diagonal_error=control_error,exact_diagonal_pass=control_error<=1e-6,
      perturbation_response=sensitivity,
      perturbation_curvature_flips=sum(sign(perturbed$coefficients[qi])!=sign(fit$coefficients[qi])),
      shape_training_rms=diag$shape_training_rms)
    write_table(rows,"solver-controls.csv")
  }
  # Only these two cases have independently replayed terminal-target algebra.
  shape_case <- grepl("controller_probe-d80-j1-wlog1-tail8-population",records$source[j]) &&
    records$replication[j] %in% c(1,3)
  if(shape_case) {
    dat <- iapf_paper_data(80,100,seed=records$data_seed[j])
    model <- iapf_linear_model(dat$observations,dat$A)
    set.seed(as.integer(records$data_seed[j]+records$replication[j]*1009+100003))
    initial <- iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),1000)
    terminal <- iapf_choice_fit(initial$clouds[[100]],
      iapf_backward_log_target(model,initial$clouds[[100]],100),arm="wlog1")$twist
    check(paste0("shape_case",j,"_saved_cloud"),max(abs(initial$clouds[[99]]-x)),0)
    target <- iapf_backward_log_target(model,x,99,terminal)
    check(paste0("shape_case",j,"_saved_target"),max(abs(target-b)),1e-9)
    inv <- solve(diag(d)+terminal$covariance)
    P <- diag(d)+t(dat$A) %*% inv %*% dat$A
    C <- solve(P);m <- as.vector(C %*% (dat$observations[99,]+t(dat$A) %*% inv %*% terminal$mean))
    pred <- predictive_at(model,99);Pinv <- solve(pred$covariance)
    smooth_C <- solve(Pinv+P)
    smooth_m <- as.vector(smooth_C %*% (Pinv %*% pred$mean+P %*% m))
    set.seed(92230000+records$replication[j])
    regions <- list(predictive=iapf_draw_normal(matrix(rep(pred$mean,each=1000),1000),pred$covariance),
      smoothing=iapf_draw_normal(matrix(rep(smooth_m,each=1000),1000),smooth_C))
    guides <- list(exact_full=list(mean=m,covariance=C),
      moment_diagonal=list(mean=m,covariance=diag(diag(C))),
      precision_diagonal=list(mean=m,covariance=diag(1/diag(P))),
      observation_only=list(mean=dat$observations[99,],covariance=diag(d)))
    for(name in names(fitted))if(fitted[[name]]$accepted) {
      design <- designs[[name]];beta <- fitted[[name]]$coefficients
      q <- beta[1+d+seq_len(d)];l <- beta[1+seq_len(d)]
      guides[[name]] <- list(mean=design$center-design$scale*l/(2*q),
        covariance=diag(-design$scale^2/(2*q)))
    }
    for(region in names(regions)) {
      points <- regions[[region]];reference <- iapf_log_normal(points,m,C)
      with_floor <- iapf_backward_log_target(model,points,99,terminal)
      without_floor <- iapf_log_normal(points,dat$observations[99,],diag(d))+
        iapf_log_normal(points %*% t(dat$A),terminal$mean,diag(d)+terminal$covariance)
      check(paste0("shape_case",j,"_",region,"_Gaussian_identity"),
        sd(reference-without_floor),1e-9)
      floor_difference <- max(abs(with_floor-without_floor))
      for(name in c("constant",names(guides))) {
        g <- guides[[name]]
        log_fit <- if(name=="constant")rep(0,nrow(points)) else iapf_log_normal(points,g$mean,g$covariance)
        shape_rows[[length(shape_rows)+1L]] <- data.frame(case=j,replication=records$replication[j],
          region=region,method=name,shape_rms=sd(log_fit-reference),
          KL_exact_to_fit=if(name=="constant")NA_real_ else gaussian_kl(m,C,g$mean,g$covariance),
          floor_log_target_change=floor_difference)
      }
    }
    write_table(shape_rows,"heldout-shape.csv")
  }
  saveRDS(list(record=records[j,],fits=lapply(fitted,function(f) {
    f$objective <- f$gradient <- NULL;f
  })),file.path(out,paste0("case",j,".rds")))
  cat("solver case",j,"of",nrow(records),"complete\n")
}
writeLines(unique(input_paths),file.path(out,"input-paths.txt"))

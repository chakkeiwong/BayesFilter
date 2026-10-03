# Independent CPU-only forensic checks of two saved failures, not a fit repair.
# Evidence contract: docs/plans/iapf-r-root-cause-audit-2026-09-22.md.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1L,Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(args[1]))
out <- args[1]; dir.create(out,recursive=TRUE)
source("docs/benchmarks/reference_iapf_paper.R")
source("docs/benchmarks/reference_iapf_author_choices.R")
old <- "docs/plans/artifacts/iapf-r-author-choice-hypotheses-20260922-01"
job <- file.path(old,"runs/controller_probe-d80-j1-wlog1-tail8-population/results")
checks <- rows <- components <- profiles <- list(); input_paths <- character()
add_check <- function(name,value,tolerance) {
  checks[[length(checks)+1L]] <<- data.frame(check=name,error=value,tolerance=tolerance,
    pass=is.finite(value) && value<=tolerance)
  if(!is.finite(value) || value>tolerance) {
    write.csv(do.call(rbind,checks),file.path(out,"checks.csv"),row.names=FALSE)
    stop(paste("identity check failed",name,value))
  }
}
dat <- iapf_paper_data(80,100,seed=92100180)
model <- iapf_linear_model(dat$observations,dat$A)
d <- model$dimension
for(replication in c(1L,3L)) {
  path <- file.path(job,paste0("wlog1-",replication,"-failure.rds"))
  input_paths <- c(input_paths,path)
  saved <- readRDS(path)
  stopifnot(saved$iteration==0,saved$backward_time==99)
  seed <- 92100180L+replication*1009L+100003L
  set.seed(seed)
  initial_pass <- iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),1000)
  x <- initial_pass$clouds[[99]]
  add_check(paste0("rep",replication,"_saved_cloud"),max(abs(x-saved$points)),0)
  terminal <- iapf_choice_fit(initial_pass$clouds[[100]],
    iapf_backward_log_target(model,initial_pass$clouds[[100]],100),arm="wlog1")
  nxt <- terminal$twist
  b <- iapf_backward_log_target(model,x,99,nxt)
  add_check(paste0("rep",replication,"_saved_target"),max(abs(b-saved$log_targets)),1e-9)

  # Independent Gaussian-product algebra for this specific I/I observation model.
  S <- diag(d)+nxt$covariance; Sinv <- solve(S)
  delta <- sweep(x %*% t(dat$A),2,nxt$mean)
  logF <- -.5*(d*log(2*pi)+as.numeric(determinant(S,logarithm=TRUE)$modulus)+
    rowSums((delta %*% Sinv)*delta))
  y <- dat$observations[99,]
  logg <- -.5*(d*log(2*pi)+rowSums(sweep(x,2,y)^2))
  no_floor <- logg+logF
  floor_delta <- log1p(exp(nxt$log_floor-logF))
  add_check(paste0("rep",replication,"_independent_backward_identity"),
    max(abs(b-(no_floor+floor_delta))),1e-9)
  P <- diag(d)+t(dat$A) %*% Sinv %*% dat$A
  h <- y+as.vector(t(dat$A) %*% Sinv %*% nxt$mean)
  center <- colMeans(x); scale <- sqrt(colMeans(sweep(x,2,center)^2))
  z <- sweep(sweep(x,2,center),2,scale,"/")
  D <- cbind(1,z,z^2)
  K <- sweep(sweep(P,1,scale,"*"),2,scale,"*")
  off <- K; diag(off) <- 0
  cross <- -.5*rowSums((z %*% off)*z)
  center_delta <- as.vector(dat$A %*% center)-nxt$mean
  constant <- -.5*(d*log(2*pi)+sum((center-y)^2))-
    .5*(d*log(2*pi)+as.numeric(determinant(S,logarithm=TRUE)$modulus)+
      sum(center_delta*as.vector(Sinv %*% center_delta)))
  diagonal_coef <- c(constant-max(b),scale*(h-as.vector(P %*% center)),-.5*diag(K))
  diagonal_response <- as.vector(D %*% diagonal_coef)
  response <- b-max(b)
  add_check(paste0("rep",replication,"_quadratic_decomposition"),
    max(abs(response-diagonal_response-cross-floor_delta)),1e-9)
  summary <- data.frame(replication=replication,seed=seed,time=99,
    terminal_mean_error=max(abs(nxt$mean-dat$observations[100,])),
    terminal_covariance_error=max(abs(nxt$covariance-diag(d))),
    floor_log_target_change=max(floor_delta),
    precision_min_eigenvalue=min(eigen(P,symmetric=TRUE,only.values=TRUE)$values),
    precision_max_eigenvalue=max(eigen(P,symmetric=TRUE,only.values=TRUE)$values),
    offdiagonal_frobenius_fraction=sqrt(sum((P-diag(diag(P)))^2)/sum(P^2)),
    cross_term_sd=sd(cross),target_log_range=diff(range(b)))
  write.csv(summary,file.path(out,paste0("replay-",replication,".csv")),row.names=FALSE)

  for(a in c(0,.25,.5,1,2)) {
    w <- exp(a*response); Dw <- D*sqrt(w)
    sv <- svd(Dw,nu=ncol(D),nv=ncol(D))
    tol <- .Machine$double.eps*max(dim(Dw))*sv$d[1]
    keep <- sv$d>tol
    solve_response <- function(v)as.vector(sv$v[,keep,drop=FALSE] %*%
      (crossprod(sv$u[,keep,drop=FALSE],v*sqrt(w))/sv$d[keep]))
    beta <- solve_response(response)
    beta_diag <- solve_response(diagonal_response)
    beta_cross <- solve_response(cross)
    beta_floor <- solve_response(floor_delta)
    quadratics <- 1+d+seq_len(d)
    worst <- quadratics[which.max(beta[quadratics])]
    qrfit <- lm.wfit(D,response,w=w)
    qdiag <- lm.wfit(D,diagonal_response,w=w)
    qr_q <- qrfit$coefficients[quadratics]
    rows[[length(rows)+1L]] <- data.frame(replication=replication,exponent=a,
      particles=nrow(x),columns=ncol(D),positive_weights=sum(w>0),
      ess=sum(w)^2/sum(w^2),max_weight=max(w)/sum(w),
      top10_share=sum(sort(w,decreasing=TRUE)[1:10])/sum(w),
      qr_rank=qrfit$rank,qr_diagonal_control_rank=qdiag$rank,
      svd_rank=sum(keep),svd_relative_tolerance=tol/sv$d[1],
      min_singular_relative=tail(sv$d,1)/sv$d[1],condition=sv$d[1]/tail(sv$d,1),
      qr_positive_curvatures=sum(qr_q>=0,na.rm=TRUE),qr_missing_curvatures=sum(is.na(qr_q)),
      svd_positive_curvatures=sum(beta[quadratics]>=0),
      svd_max_quadratic=beta[worst],true_diagonal_quadratic=diagonal_coef[worst],
      projected_cross_quadratic=beta_cross[worst],projected_floor_quadratic=beta_floor[worst],
      diagonal_control_positive_curvatures=sum(beta_diag[quadratics]>=0),
      diagonal_control_max_coefficient_relative_error=max(abs(beta_diag-diagonal_coef)/pmax(1,abs(diagonal_coef))),
      linear_decomposition_error=max(abs(beta-beta_diag-beta_cross-beta_floor)),
      qr_svd_max_difference=if(qrfit$rank==ncol(D))max(abs(qrfit$coefficients-beta)) else NA_real_,
      svd_weighted_residual=sqrt(sum(w*(response-as.vector(D %*% beta))^2)/sum(w)))
    components[[length(components)+1L]] <- data.frame(replication=replication,
      exponent=a,coordinate=seq_len(d),true_quadratic=diagonal_coef[quadratics],
      fitted_quadratic=beta[quadratics],diagonal_control=beta_diag[quadratics],
      cross_projection=beta_cross[quadratics],floor_projection=beta_floor[quadratics])
  }

  anchor <- iapf_fit_initial(x,b,TRUE)$parameters
  logscale <- max(iapf_log_normal(x,anchor[seq_len(d)],diag(exp(anchor[d+seq_len(d)]),d)))
  for(multiplier in c(1,1.2,2,4,16,64)) {
    par <- anchor; par[d+seq_len(d)] <- par[d+seq_len(d)]+log(multiplier)
    ev <- iapf_fit_objective(par,x,b,logscale)
    p <- exp(iapf_log_normal(x,par[seq_len(d)],diag(exp(par[d+seq_len(d)]),d))-logscale)
    target <- exp(b-max(b)); lam <- sum(p*target)/sum(target^2)
    direct <- mean((p-lam*target)^2)
    add_check(paste0("rep",replication,"_equation15_",multiplier),
      abs(ev$value-direct)/max(ev$value,direct,1e-300),1e-8)
    profiles[[length(profiles)+1L]] <- data.frame(replication=replication,
      variance_multiplier=multiplier,absolute_profiled_loss=ev$value,
      log_loss=ev$log_loss,relative_shape_residual=ev$relative_residual,
      loss_underflow=ev$loss_underflow)
  }
  cat("CHECKED failure replicate",replication,"\n"); flush.console()
}
exact <- iapf_exact_twists(model)
set.seed(92100181L)
oracle <- iapf_apf(model,exact,64)
kalman <- iapf_kalman(model)
add_check("full_gaussian_shared_filter_vs_kalman",abs(oracle$log_likelihood-kalman$log_likelihood),1e-8)
write.csv(do.call(rbind,checks),file.path(out,"checks.csv"),row.names=FALSE)
write.csv(do.call(rbind,rows),file.path(out,"conditioning.csv"),row.names=FALSE)
write.csv(do.call(rbind,components),file.path(out,"curvature-decomposition.csv"),row.names=FALSE)
write.csv(do.call(rbind,profiles),file.path(out,"equation15-profile.csv"),row.names=FALSE)
writeLines(input_paths,file.path(out,"input-paths.txt"))
writeLines(c(capture.output(sessionInfo()),"CPU only; GPU intentionally hidden.",
  "No runtime fit policy changed. Saved failures are purposive diagnostic cases."),file.path(out,"environment.txt"))
cat("PASS",length(checks),"identity checks; two saved clouds; no performance ranking.\n")

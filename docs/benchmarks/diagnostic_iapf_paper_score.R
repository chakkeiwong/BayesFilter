# Independent R diagnostic extension. Not Eq15, author code, or a runtime backend.
iapf_score_target <- function(model,x,time,next_twist=NULL) {
  H <- model$C;R <- model$R;A <- model$A
  residual <- sweep(-x%*%t(H),2,model$observations[time,],'+')
  h <- residual%*%solve(R,H)
  gaussian_probability <- rep(1,nrow(x))
  if (!is.null(next_twist)) {
    future <- sweep(-x%*%t(A),2,next_twist$mean,'+')
    covariance <- model$transition_covariance+next_twist$covariance
    lg <- iapf_log_normal(future,rep(0,ncol(x)),covariance)
    lm <- iapf_logadd(lg,next_twist$log_floor)
    gaussian_probability <- exp(lg-lm)
    h <- h+gaussian_probability*(future%*%solve(covariance,A))
  }
  attr(h,'gaussian_probability') <- gaussian_probability
  h
}

iapf_score_fit_cloud <- function(x,h) {
  N <- nrow(x);d <- ncol(x);tol <- 64*2^-52*max(N,d)
  rejected <- function(reason,diagnostics) list(valid=FALSE,reason=reason,diagnostics=diagnostics)
  if (N<=d || any(!is.finite(x)) || any(!is.finite(h))) return(rejected('cloud_or_score',list()))
  m <- colMeans(x);xc <- sweep(x,2,m);C <- crossprod(xc)/N
  eig <- eigen(C,symmetric=TRUE,only.values=TRUE)$values
  cloud_margin <- min(eig)/max(abs(eig))
  info <- list(cloud_margin=cloud_margin,guard_tolerance=tol)
  if (!is.finite(cloud_margin) || cloud_margin<=tol) return(rejected('cloud_rank',info))
  L <- t(chol(C));z <- t(solve(L,t(xc)));g <- h%*%L;a <- colMeans(g)
  raw <- -crossprod(sweep(g,2,a),z)/N;P <- (raw+t(raw))/2
  eig <- eigen(P,symmetric=TRUE,only.values=TRUE)$values
  info$precision_margin <- min(eig)/max(abs(eig))
  if (!is.finite(info$precision_margin) || info$precision_margin<=tol) return(rejected('nonpositive_precision',info))
  V <- L%*%solve(P,t(L));center <- as.vector(m+L%*%solve(P,a))
  info$whitening_error <- max(abs(crossprod(z)/N-diag(d)))
  info$skew_norm <- norm(raw-t(raw),'F')
  info$score_residual <- mean(rowSums((sweep(g,2,a)+z%*%P)^2))
  if (any(!is.finite(V)) || any(!is.finite(center))) return(rejected('nonfinite_fit',info))
  list(valid=TRUE,mean=center,covariance=V,diagnostics=info)
}

iapf_score_backward <- function(model,clouds,N,covariance_mode='diagonal',floor_mode='tail8') {
  stopifnot(covariance_mode%in%c('diagonal','full'),floor_mode%in%c('positive','tail8','negligible'))
  T <- model$horizon;d <- model$dimension;twists <- vector('list',T);diagnostics <- vector('list',T)
  for (time in rev(seq_len(T))) {
    h <- iapf_score_target(model,clouds[[time]],time,if (time<T) twists[[time+1]] else NULL)
    fit <- iapf_score_fit_cloud(clouds[[time]],h);diagnostics[[time]] <- fit$diagnostics
    diagnostics[[time]]$target_gaussian_probability_min <- min(attr(h,'gaussian_probability'))
    diagnostics[[time]]$target_gaussian_probability_mean <- mean(attr(h,'gaussian_probability'))
    if (!fit$valid) return(list(valid=FALSE,reason=fit$reason,backward_time=time,diagnostics=diagnostics))
    V <- if (covariance_mode=='full') fit$covariance else diag(diag(fit$covariance),d)
    log_ratio <- switch(floor_mode,positive=log(0.009999999776482582),negligible=-1000,
                        tail8=-.5*qchisq(N^-8,d,lower.tail=FALSE))
    floor <- log_ratio-.5*d*log(2*pi)-sum(log(diag(chol(V))))
    twists[[time]] <- iapf_gaussian_twist(fit$mean,V,floor)
  }
  list(valid=TRUE,twists=twists,diagnostics=diagnostics)
}

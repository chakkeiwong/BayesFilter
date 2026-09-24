# Independent, deterministic reference diagnosis; no fitting optimizer or runtime change.
args <- commandArgs(trailingOnly=TRUE)
source(args[1]); source(args[2]); previous <- read.csv(args[3]); out <- args[4]
writeLines(R.version.string,file.path(out,'R-version.txt'))
vec <- function(x) as.numeric(unlist(x))
mat <- function(x) do.call(rbind,lapply(x,vec))
scaled_error <- function(a,b) max(abs(a-b))/(1+max(abs(a),abs(b)))
records <- list(); details <- list(); curvature <- list()
for (k in seq_along(input)) {
  item <- input[[k]]; control <- controls[[k]]
  stopifnot(identical(item$key,control$key))
  if (control$objective!='density_l2') next
  z <- mat(item$z); logb <- vec(item$log_target); b <- exp(logb)
  n <- nrow(z); d <- ncol(z); lower <- vec(control$lower); upper <- vec(control$upper)
  phi <- cbind(z,z*z); D <- cbind(1,phi)
  pairs <- t(combn(seq_len(d),2))
  C <- vapply(seq_len(nrow(pairs)),function(j) z[,pairs[j,1]]*z[,pairs[j,2]],numeric(n))
  F <- cbind(D,C); p <- ncol(F); tol <- .Machine$double.eps*max(n,p)
  stopifnot(n>=p)
  full_qr <- qr(F,tol=tol); reduced_qr <- qr(D,tol=tol)
  margin <- min(abs(diag(qr.R(full_qr))))/max(abs(diag(qr.R(full_qr))))
  reduced_margin <- min(abs(diag(qr.R(reduced_qr))))/max(abs(diag(qr.R(reduced_qr))))
  rank_pass <- full_qr$rank==p && reduced_qr$rank==ncol(D) && min(margin,reduced_margin)>tol
  if (!rank_pass) stop('Insufficient numerical rank: ',item$key)
  coefficient <- qr.coef(full_qr,logb); reduced <- qr.coef(reduced_qr,logb)
  omitted <- as.vector(C%*%coefficient[ncol(D)+seq_len(ncol(C))])
  projected <- qr.coef(reduced_qr,omitted)
  projection_error <- max(abs(reduced-coefficient[seq_len(ncol(D))]-projected))
  full_residual <- max(abs(logb-as.vector(F%*%coefficient)))
  diagonal_residual <- max(abs(logb-as.vector(D%*%reduced)))
  cloud_mean <- vec(item$states$cloud$center)
  cloud_sd <- sqrt(diag(mat(item$states$cloud$covariance)))
  exact_mean <- vec(control$exact_center); exact_cov <- mat(control$exact_covariance)
  logdet <- function(x) 2*sum(log(diag(chol(x))))
  KL <- function(center,covariance) {
    delta <- center-exact_mean
    (sum(diag(solve(covariance,exact_cov)))+sum(delta*solve(covariance,delta))-d+
      logdet(covariance)-logdet(exact_cov))/2
  }
  shape <- function(logp) {
    pvalue <- exp(logp-max(logp))
    residual <- pvalue-sum(pvalue*b)/sum(b*b)*b
    sum(residual*residual)/sum(pvalue*pvalue)
  }
  gaussian_shape <- function(center,covariance) {
    m <- (center-cloud_mean)/cloud_sd
    V <- covariance/outer(cloud_sd,cloud_sd)
    u <- sweep(z,2,m,'-')
    shape(-rowSums((u%*%solve(V))*u)/2)
  }
  quad <- reduced[1+d+seq_len(d)]
  qr_valid <- all(quad < -sqrt(.Machine$double.eps))
  qr_parity <- NA_real_; qr_raw <- NULL
  if (qr_valid) {
    qr_raw <- c(-reduced[1+seq_len(d)]/(2*quad),-.5*log(-2*quad))
    qr_clipped <- pmin(upper,pmax(lower,qr_raw))
    qr_parity <- max(abs(qr_clipped-vec(item$states$QR$parameters)))
  }
  precision <- diag(-2*coefficient[1+d+seq_len(d)],d)
  for (j in seq_len(nrow(pairs))) {
    precision[pairs[j,1],pairs[j,2]] <- precision[pairs[j,2],pairs[j,1]] <- -coefficient[ncol(D)+j]
  }
  precision_eigenvalues <- eigen(precision,symmetric=TRUE,only.values=TRUE)$values
  pd_margin <- min(precision_eigenvalues)/max(abs(precision_eigenvalues))
  positive <- is.finite(pd_margin) && pd_margin > .Machine$double.eps*d
  full_mean <- full_cov <- NULL
  full_KL <- full_diag_KL <- full_shape <- full_diag_shape <- NA_real_
  center_error <- covariance_error <- NA_real_; inside <- NA
  if (positive) {
    full_cov_z <- chol2inv(chol(precision))
    full_mean_z <- as.vector(full_cov_z%*%coefficient[1+seq_len(d)])
    full_mean <- cloud_mean+cloud_sd*full_mean_z
    full_cov <- full_cov_z*outer(cloud_sd,cloud_sd)
    full_diag <- diag(diag(full_cov),d)
    par_diag <- c(full_mean_z,.5*log(diag(full_cov_z)))
    inside <- all(par_diag>=lower & par_diag<=upper)
    full_KL <- KL(full_mean,full_cov); full_diag_KL <- KL(full_mean,full_diag)
    full_shape <- gaussian_shape(full_mean,full_cov)
    full_diag_shape <- gaussian_shape(full_mean,full_diag)
    center_error <- max(abs(full_mean-exact_mean))
    covariance_error <- max(abs(full_cov-exact_cov))
    stopifnot(all(is.finite(c(full_mean,full_cov,full_KL,full_diag_KL,full_shape,full_diag_shape))))
  }
  centered <- sweep(phi,2,colMeans(phi),'-')
  unweighted <- crossprod(centered)/n
  whitening <- chol(unweighted)
  W <- centered%*%solve(whitening)
  whitening_error <- max(abs(crossprod(W)/n-diag(ncol(W))))
  pi <- b*b/sum(b*b); mu <- as.vector(crossprod(W,pi))
  Wc <- sweep(W,2,mu,'-')
  H <- 2*crossprod(Wc,pi*Wc)
  eig <- eigen(H,symmetric=TRUE)
  directional <- list(); curvature_error <- 0
  for (direction in c('largest','smallest')) {
    j <- if(direction=='largest') 1 else ncol(W)
    tilt <- as.vector(W%*%eig$vectors[,j]); predicted <- eig$values[j]
    for (h in c(1e-3,1e-4,1e-5)) {
      measured <- (shape(logb+h*tilt)+shape(logb-h*tilt)-2*shape(logb))/h^2
      discrepancy <- abs(measured-predicted)/(1+abs(predicted))
      if (h==1e-4) curvature_error <- max(curvature_error,discrepancy)
      row <- data.frame(key=item$key,direction=direction,h=h,predicted=predicted,
                        measured=measured,scaled_error=discrepancy)
      directional[[length(directional)+1L]] <- row
      curvature[[length(curvature)+1L]] <- row
    }
  }
  R_density <- previous[previous$key==item$key,]
  R_shape <- previous[previous$key==sub('/density_l2$','/relative_shape',item$key),]
  other <- input[[which(vapply(input,function(x) identical(x$key,sub('/density_l2$','/relative_shape',item$key)),logical(1)))]]
  stopifnot(nrow(R_density)==1,nrow(R_shape)==1)
  record <- data.frame(key=item$key,case=control$case,time=control$time,target=control$target,
    dimension=d,N=n,full_parameters=p,rank_pass=rank_pass,full_rank_margin=margin,
    diagonal_rank_margin=reduced_margin,rank_tolerance=tol,
    full_log_residual=full_residual,diagonal_log_residual=diagonal_residual,
    projection_identity_error=projection_error,omitted_projection_max=max(abs(projected[-1])),
    qr_valid=qr_valid,qr_parity_error=qr_parity,qr_KL=vec(item$states$QR$KL),qr_shape=vec(item$states$QR$shape),
    full_positive_definite=positive,precision_margin=pd_margin,full_diagonal_inside_box=inside,
    full_center_error=center_error,full_covariance_error=covariance_error,
    full_KL=full_KL,full_diag_KL=full_diag_KL,full_shape=full_shape,full_diag_shape=full_diag_shape,
    exact_diagonal_KL=vec(item$states$exact_diagonal$KL),
    pi_ESS=1/sum(pi*pi),pi_max=max(pi),whitening_error=whitening_error,
    hessian_min=min(eig$values),hessian_max=max(eig$values),
    hessian_condition=max(eig$values)/min(eig$values),curvature_check_error=curvature_error,
    TF_density_KL=vec(item$states$final$KL),TF_density_shape=vec(item$states$final$shape),
    TF_relative_KL=vec(other$states$final$KL),TF_relative_shape=vec(other$states$final$shape),
    R_density_KL=R_density$KL,R_density_shape=R_density$shape,R_density_accepted=R_density$accepted,
    R_relative_KL=R_shape$KL,R_relative_shape=R_shape$shape,R_relative_accepted=R_shape$accepted)
  records[[length(records)+1L]] <- record
  details[[item$key]] <- list(full_coefficients=coefficient,diagonal_coefficients=reduced,
    omitted_projection=projected,full_precision=precision,full_mean=full_mean,full_covariance=full_cov,
    QR_raw_parameters=qr_raw,whitening=whitening,weights=pi,eigenvalues=eig$values,
    eigenvectors=eig$vectors,directional=directional)
  cat(item$key,'full residual',full_residual,'pi ESS',1/sum(pi*pi),'condition',record$hessian_condition,'\n')
}
result <- do.call(rbind,records)
write.csv(result,file.path(out,'results.csv'),row.names=FALSE)
write.csv(do.call(rbind,curvature),file.path(out,'curvature.csv'),row.names=FALSE)
saveRDS(details,file.path(out,'details.rds'))
exact <- result[result$target=='exact',]
stopifnot(nrow(result)==32,nrow(exact)==16,all(result$rank_pass),all(exact$full_positive_definite),
  max(result$projection_identity_error)<=1e-8,max(result$qr_parity_error)<=1e-8,
  max(exact$full_log_residual)<=1e-9,max(exact$full_center_error)<=1e-8,
  max(exact$full_covariance_error)<=1e-8,max(result$curvature_check_error)<=1e-5,
  max(result$whitening_error)<=1e-8)

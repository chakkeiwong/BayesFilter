# Base-R independent oracle-start diagnosis. Never a runtime or author implementation.
args <- commandArgs(trailingOnly=TRUE)
source(args[1]); source(args[2]); previous <- read.csv(args[3]); out <- args[4]
writeLines(R.version.string,file.path(out,'R-version.txt'))
vec <- function(x) as.numeric(unlist(x))
mat <- function(x) do.call(rbind,lapply(x,vec))
err <- function(a,b) max(abs(a-b))/(1+max(abs(a),abs(b)))
records <- list(); fits <- list()
for (k in seq_along(input)) {
  item <- input[[k]]; control <- controls[[k]]
  stopifnot(identical(item$key,control$key))
  z <- mat(item$z); b <- exp(vec(item$log_target)); n <- nrow(z); d <- ncol(z)
  lower <- vec(control$lower); upper <- vec(control$upper)
  initial <- vec(item$states$exact_diagonal$parameters)
  stopifnot(all(initial>=lower & initial<=upper))
  scale <- control$objective_log_density_scale
  profile <- function(par) {
    lsd <- par[d+seq_len(d)]
    u <- sweep(sweep(z,2,par[seq_len(d)],'-'),2,exp(-lsd),'*')
    logp <- -sum(lsd)-rowSums(u*u)/2; a <- max(logp); p <- exp(logp-a)
    residual <- p-sum(p*b)/sum(b*b)*b
    energy <- exp(2*a)*mean(p*p); loss <- exp(2*a)*mean(residual*residual)
    shape <- sum(residual*residual)/sum(p*p)
    score <- cbind(sweep(u,2,exp(-lsd),'*'),u*u-1)
    gradient <- as.vector(2*exp(2*a)*crossprod(score,p*residual)/n)
    shape_gradient <- as.vector(2*crossprod(score,p*residual-shape*p*p)/sum(p*p))
    if (control$objective=='relative_shape') {
      value <- shape; grad <- shape_gradient
    } else {
      value <- exp(2*(a-scale))*mean(residual*residual)
      grad <- as.vector(2*exp(2*(a-scale))*crossprod(score,p*residual)/n)
    }
    list(value=value,grad=grad,loss=loss,energy=energy,shape=shape,
         gradient=gradient,shape_gradient=shape_gradient)
  }
  start <- profile(initial); expected <- item$states$exact_diagonal
  initial_error <- max(err(start$loss,vec(expected$loss)),err(start$energy,vec(expected$energy)),
    err(start$shape,vec(expected$shape)),err(start$gradient,vec(expected$gradient)),
    err(start$shape_gradient,vec(expected$shape_gradient)))
  if (!is.finite(initial_error) || initial_error>1e-8) stop('Oracle objective identity failed')
  fn_calls <- 0L; gr_calls <- 0L
  fn <- function(par) {fn_calls <<- fn_calls+1L; profile(par)$value}
  gr <- function(par) {gr_calls <<- gr_calls+1L; profile(par)$grad}
  started <- proc.time()[['elapsed']]
  answer <- optim(initial,fn,gr,method='L-BFGS-B',lower=lower,upper=upper,
    control=list(maxit=control$maxit,lmm=control$lmm,factr=control$factr,
                 pgtol=control$tolerance,parscale=rep(1,2*d)))
  elapsed <- proc.time()[['elapsed']]-started; final <- profile(answer$par)
  pg <- max(abs(answer$par-pmin(upper,pmax(lower,answer$par-final$grad))))
  finite <- all(is.finite(c(answer$par,unlist(final),pg)))
  accepted <- finite && answer$convergence==0L && pg<=control$tolerance
  cloud_mean <- vec(item$states$cloud$center)
  cloud_sd <- sqrt(diag(mat(item$states$cloud$covariance)))
  center <- cloud_mean+cloud_sd*answer$par[seq_len(d)]
  variance <- (cloud_sd*exp(answer$par[d+seq_len(d)]))^2
  exact_center <- vec(control$exact_center); exact_cov <- mat(control$exact_covariance)
  KL <- (sum(diag(exact_cov)/variance)+sum((center-exact_center)^2/variance)-d+
      sum(log(variance))-as.numeric(determinant(exact_cov,logarithm=TRUE)$modulus))/2
  old <- previous[previous$key==item$key,]; stopifnot(nrow(old)==1)
  records[[k]] <- data.frame(key=item$key,case=control$case,time=control$time,
    target=control$target,objective=control$objective,dimension=d,N=n,
    initial_identity_error=initial_error,finite=finite,solver_status=answer$convergence,
    solver_message=if(is.null(answer$message)) '' else answer$message,
    accepted=accepted,projected_gradient=pg,tolerance=control$tolerance,
    loss=final$loss,energy=final$energy,shape=final$shape,KL=KL,
    initial_loss=start$loss,initial_energy=start$energy,initial_shape=start$shape,
    initial_objective=start$value,final_objective=final$value,
    initial_KL=vec(expected$KL),delta_KL=KL-vec(expected$KL),
    prior_R_KL=old$KL,prior_R_shape=old$shape,prior_R_accepted=old$accepted,
    prior_TF_KL=control$baseline_KL,prior_TF_shape=control$baseline_shape,
    function_calls=fn_calls,gradient_calls=gr_calls,wall_seconds=elapsed)
  fits[[item$key]] <- list(optimizer=answer,profile=final,controls=control,center=center,variance=variance)
}
result <- do.call(rbind,records)
write.csv(result,file.path(out,'oracle-start.csv'),row.names=FALSE)
saveRDS(fits,file.path(out,'fits.rds'))
stopifnot(nrow(result)==64,all(result$finite),max(result$initial_identity_error)<=1e-8)

# Complete the square for E_{N(0,C)} exp[-(x'P x-2h'x+c)/2].
logdet <- function(M) 2*sum(log(diag(chol(M))))
log_product <- function(C,P,h,c) {
  G <- solve(C)+P
  (-logdet(C)-logdet(G)-c+sum(h*solve(G,h)))/2
}
formula_rows <- list(); sample_rows <- list(); narrow_rows <- list()
for (d in c(2,5,10,20,40,80)) {
  I <- diag(d); zero <- rep(0,d)
  for (law in c('matched','half','dimension_scaled')) {
    r <- switch(law,matched=1,half=.5,dimension_scaled=1/d)
    C <- r*I
    logE2 <- log_product(C,2*I,zero,0)
    logE4 <- log_product(C,4*I,zero,0)
    log_fraction <- 2*logE2-logE4
    predicted_log_fraction <- d*(.5*log1p(4*r)-log1p(2*r))
    for (delta in c(0,3)) {
      h <- c(delta,rep(0,d-1))
      logEp2 <- log_product(C,2*I,2*h,2*delta^2)
      logEpb <- log_product(C,2*I,h,delta^2)
      population_shape <- -expm1(2*logEpb-logEp2-logE2)
      predicted_shape <- -expm1(-delta^2*r/(1+2*r))
      formula_error <- max(abs(log_fraction-predicted_log_fraction),abs(population_shape-predicted_shape),
                           abs(logE2+d/2*log1p(2*r)),abs(logE4+d/2*log1p(4*r)))
      formula_rows[[length(formula_rows)+1L]] <- data.frame(dimension=d,law=law,r=r,delta=delta,
        KL=delta^2/2,population_shape=population_shape,asymptotic_ESS_fraction=exp(log_fraction),
        log_asymptotic_ESS_fraction=log_fraction,formula_error=formula_error)
    }
    for (seed in 9101:9132) {
      set.seed(seed); x <- sqrt(r)*matrix(rnorm(1000*d),nrow=1000,ncol=d)
      logb <- -rowSums(x*x)/2; target <- exp(logb-max(logb))
      w <- target*target/sum(target*target)
      for (delta in c(0,3)) {
        logp <- logb+delta*x[,1]-delta^2/2; p <- exp(logp-max(logp))
        residual <- p-sum(p*target)/sum(target*target)*target
        sampled_shape <- sum(residual*residual)/sum(p*p)
        values <- c(w,sampled_shape,1/sum(w*w))
        sample_rows[[length(sample_rows)+1L]] <- data.frame(dimension=d,law=law,r=r,delta=delta,
          N=1000,seed=seed,ESS=1/sum(w*w),max_weight=max(w),weight_sum=sum(w),
          sample_shape=sampled_shape,population_shape=-expm1(-delta^2*r/(1+2*r)),
          asymptotic_ESS_fraction=exp(log_fraction),KL=delta^2/2,finite=all(is.finite(values)))
      }
    }
  }
  for (r in c(1e-2,1e-4,1e-6)) {
    narrow_rows[[length(narrow_rows)+1L]] <- data.frame(dimension=d,r=r,delta=3,KL=4.5,
      population_shape=-expm1(-9*r/(1+2*r)),
      asymptotic_ESS_fraction=exp(d*(.5*log1p(4*r)-log1p(2*r))))
  }
}
formulas <- do.call(rbind,formula_rows); samples <- do.call(rbind,sample_rows)
write.csv(formulas,file.path(out,'population-formulas.csv'),row.names=FALSE)
write.csv(samples,file.path(out,'population-draws.csv'),row.names=FALSE)
write.csv(do.call(rbind,narrow_rows),file.path(out,'narrow-limit.csv'),row.names=FALSE)
groups <- split(samples,interaction(samples$dimension,samples$law,samples$delta,drop=TRUE))
summary <- do.call(rbind,lapply(groups,function(g) {
  z <- g[1,c('dimension','law','r','delta','N','population_shape','asymptotic_ESS_fraction','KL')]
  for (metric in c('ESS','max_weight','sample_shape')) {
    mu <- mean(g[[metric]]); se <- sd(g[[metric]])/sqrt(nrow(g)); radius <- qt(.995,nrow(g)-1)*se
    z[[paste0(metric,'_mean')]] <- mu; z[[paste0(metric,'_se')]] <- se
    z[[paste0(metric,'_lower99')]] <- mu-radius; z[[paste0(metric,'_upper99')]] <- mu+radius
    z[[paste0(metric,'_median')]] <- median(g[[metric]])
  }
  z$replicates <- nrow(g); z
}))
write.csv(summary,file.path(out,'population-summary.csv'),row.names=FALSE)
stopifnot(nrow(formulas)==36,nrow(samples)==1152,max(formulas$formula_error)<=1e-10,
  all(samples$finite),max(abs(samples$weight_sum-1))<=1e-12,
  all(samples$ESS>=1-1e-10 & samples$ESS<=1000+1e-10),
  max(samples$sample_shape[samples$delta==0])<=1e-12)
cat('64 oracle fits,36 Gaussian formulas,1152 sample records and18 narrow controls complete\n')

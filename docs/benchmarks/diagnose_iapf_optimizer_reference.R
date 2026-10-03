# Independent R diagnostic, not an original-author or deployable filter.
args <- commandArgs(trailingOnly=TRUE)
source(args[1]); source(args[2]); out <- args[3]
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
  initial <- vec(item$states$initial$parameters)
  scale <- control$objective_log_density_scale
  profile <- function(par) {
    lsd <- par[d+seq_len(d)]
    u <- sweep(sweep(z,2,par[seq_len(d)],'-'),2,exp(-lsd),'*')
    logp <- -sum(lsd)-rowSums(u*u)/2; a <- max(logp); p <- exp(logp-a)
    residual <- p-sum(p*b)/sum(b*b)*b
    energy <- exp(2*a)*mean(p*p)
    loss <- exp(2*a)*mean(residual*residual)
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
  start <- profile(initial); expected <- item$states$initial
  initial_error <- max(err(start$loss,vec(expected$loss)),err(start$energy,vec(expected$energy)),
    err(start$shape,vec(expected$shape)),err(start$gradient,vec(expected$gradient)),
    err(start$shape_gradient,vec(expected$shape_gradient)))
  if (!is.finite(initial_error) || initial_error>1e-8) stop('Initial objective identity failed')
  fn_calls <- 0L; gr_calls <- 0L
  fn <- function(par) {fn_calls <<- fn_calls+1L; profile(par)$value}
  gr <- function(par) {gr_calls <<- gr_calls+1L; profile(par)$grad}
  started <- proc.time()[['elapsed']]
  answer <- optim(initial,fn,gr,method='L-BFGS-B',lower=lower,upper=upper,
    control=list(maxit=control$maxit,lmm=control$lmm,factr=control$factr,
                 pgtol=control$tolerance,parscale=rep(1,2*d)))
  elapsed <- proc.time()[['elapsed']]-started
  final <- profile(answer$par)
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
  boundary <- any(pmin(answer$par-lower,upper-answer$par)<=32*.Machine$double.eps*(1+abs(answer$par)))
  records[[k]] <- data.frame(key=item$key,case=control$case,time=control$time,
      target=control$target,objective=control$objective,dimension=d,N=n,
      initial_identity_error=initial_error,finite=finite,solver_status=answer$convergence,
      solver_message=if(is.null(answer$message)) '' else answer$message,
      accepted=accepted,projected_gradient=pg,tolerance=control$tolerance,
      boundary=boundary,loss=final$loss,energy=final$energy,shape=final$shape,KL=KL,
      initial_loss=start$loss,initial_energy=start$energy,initial_shape=start$shape,
      baseline_converged=control$baseline_converged,baseline_projected_gradient=control$baseline_projected_gradient,
      baseline_shape=control$baseline_shape,baseline_KL=control$baseline_KL,
      function_calls=fn_calls,gradient_calls=gr_calls,wall_seconds=elapsed)
  fits[[item$key]] <- list(optimizer=answer,profile=final,controls=control,center=center,variance=variance)
  cat(item$key,'status',answer$convergence,'accepted',accepted,'PG',pg,'KL',KL,'\n')
}
result <- do.call(rbind,records)
write.csv(result,file.path(out,'results.csv'),row.names=FALSE)
saveRDS(fits,file.path(out,'fits.rds'))
if (nrow(result)!=64L || any(!result$finite) || max(result$initial_identity_error)>1e-8) stop('Required diagnostic check failed')

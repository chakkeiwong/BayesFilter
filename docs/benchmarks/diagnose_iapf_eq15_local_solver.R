# Independent CPU R diagnostic: local joint Equation (15) solver hypothesis.
# Damping changes the Gauss-Newton step, never the target residual.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; failures <- args[2]; output <- args[3]
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1", dir.exists(output))
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_eq15_solvers.R"))

residual_jacobian <- function(p,points,log_targets,scale) {
  d <- ncol(points); n <- nrow(points)
  m <- p[seq_len(d)]; lv <- p[d+seq_len(d)]
  centered <- sweep(points,2,m)
  q <- sweep(centered^2,2,exp(lv),"/")
  density <- exp(-.5*(d*log(2*pi)+sum(lv)+rowSums(q))-scale)
  target <- exp(p[2*d+1]+log_targets-max(log_targets))
  residual <- (density-target)/sqrt(n)
  J <- cbind(sweep(centered,2,exp(lv),"/")*density,
             .5*(q-1)*density,-target)/sqrt(n)
  stopifnot(all(is.finite(residual)), all(is.finite(J)))
  list(residual=residual,J=J,value=sum(residual^2))
}

solve_local <- function(points,targets,initial,maxit=150L) {
  d <- ncol(points)
  initial_density <- iapf_log_normal(points,initial[seq_len(d)],
                                    diag(exp(initial[d+seq_len(d)]),d))
  scale <- max(initial_density)
  prof <- iapf_fit_objective(initial,points,targets,scale)
  p <- c(initial,log(prof$lambda_scaled))
  coords <- c(sqrt(exp(initial[d+seq_len(d)])),rep(1,d),1)
  ev <- residual_jacobian(p,points,targets,scale)
  # Scale dimensionless coordinates once per fit. Fixed normalization is not
  # the alternative relative-residual objective.
  normalization <- mean(exp(initial_density-scale)^2)
  lambda <- 1e-3
  nu <- 2
  trajectory <- list()
  status <- "max_iterations"
  for(iter in seq_len(maxit)) {
    J <- sweep(ev$J,2,coords,"*")/sqrt(normalization)
    r <- ev$residual/sqrt(normalization)
    H <- crossprod(J); g <- as.vector(crossprod(J,r))
    curvature <- max(diag(H))
    D <- pmax(diag(H),.Machine$double.eps*max(curvature,1))
    shaped <- iapf_fit_objective(p[seq_len(2*d)],points,targets,scale)
    gradient <- max(abs(2*g))
    logdet_change <- sum(p[d+seq_len(d)]-initial[d+seq_len(d)])
    log_density_change <- shaped$log_density_max-max(initial_density)
    record <- data.frame(iteration=iter,loss=ev$value/normalization,
       relative_residual=shaped$relative_residual,gradient_max=gradient,
       logdet_change=logdet_change,log_density_change=log_density_change,
       log_lambda=p[2*d+1],damping=lambda,rho=NA_real_,accepted=FALSE)
    # A relative residual above roundoff plus a density collapse is a veto,
    # even when the raw gradient is small. Threshold is diagnostic, not a bound.
    collapse <- log_density_change < log(sqrt(.Machine$double.eps)) &&
                shaped$relative_residual > 1e-12
    if(collapse || shaped$loss_underflow) {
      status <- if(collapse)"amplitude_collapse" else "underflow"
      trajectory[[length(trajectory)+1]] <- record; break
    }
    if(shaped$relative_residual < 1e-24 && ev$value/normalization < 1e-24) {
      status <- "exact_fit"; trajectory[[length(trajectory)+1]] <- record; break
    }
    if(gradient <= 1e-8 && shaped$relative_residual <= 1e-6) {
      status <- "stationary_usable_candidate"
      trajectory[[length(trajectory)+1]] <- record; break
    }
    step <- tryCatch(as.vector(solve(H+diag(lambda*D,length(D)),-g)),
                     error=function(e)rep(NA_real_,length(g)))
    if(any(!is.finite(step))) {
      status <- "linear_solve_failure"; trajectory[[length(trajectory)+1]] <- record; break
    }
    proposed <- p+coords*step
    next_ev <- tryCatch(residual_jacobian(proposed,points,targets,scale),error=function(e)NULL)
    prediction <- -2*sum(g*step)-sum(step*as.vector(H%*%step))
    rho <- if(!is.null(next_ev) && prediction>0)
      (ev$value-next_ev$value)/normalization/prediction else -Inf
    record$rho <- rho
    if(is.finite(rho) && rho>0) {
      p <- proposed; ev <- next_ev; record$accepted <- TRUE
      lambda <- lambda*max(1/3,1-(2*rho-1)^3)
      nu <- 2
    } else {
      lambda <- lambda*nu; nu <- 2*nu
    }
    trajectory[[length(trajectory)+1]] <- record
    if(!is.finite(lambda) || lambda>1/.Machine$double.eps) {
      status <- "trust_model_failure"; break
    }
  }
  final <- iapf_fit_objective(p[seq_len(2*d)],points,targets,scale)
  list(status=status,parameters=p,trajectory=do.call(rbind,trajectory),
       relative_residual=final$relative_residual,loss=ev$value/normalization,
       logdet_change=sum(p[d+seq_len(d)]-initial[d+seq_len(d)]),
       density_change=final$log_density_max-max(initial_density),
       gradient_max=max(abs(2*as.vector(crossprod(sweep(ev$J,2,coords,"*"),ev$residual))/normalization)))
}

# Diagnostic seed only; no full filter is launched.
set.seed(924021)
points <- matrix(rnorm(200),ncol=2)
truth <- c(.3,-.2,log(c(.7,1.4)))
log_targets <- iapf_log_normal(points,truth[1:2],diag(exp(truth[3:4]),2))
p <- c(.1,-.1,log(c(.8,1.2)),.2)
scale <- 0
ev <- residual_jacobian(p,points,log_targets,scale)
fd <- sapply(seq_along(p),function(j) {
  h <- 1e-5*max(1,abs(p[j])); plus <- minus <- p
  plus[j] <- plus[j]+h; minus[j] <- minus[j]-h
  (residual_jacobian(plus,points,log_targets,scale)$residual-
   residual_jacobian(minus,points,log_targets,scale)$residual)/(2*h)
})
jacobian_error <- max(abs(ev$J-fd))/max(1,max(abs(fd)))
original <- iapf_eq15_joint_objective(p,points,log_targets,scale)
stopifnot(jacobian_error<1e-8,
  abs(ev$value-original$value)<1e-14,
  max(abs(2*as.vector(crossprod(ev$J,ev$residual))-original$gradient))<1e-12)
exact <- solve_local(points,log_targets,truth)
near <- solve_local(points,log_targets,c(.1,-.1,log(c(.8,1.2))))
stopifnot(exact$status=="exact_fit",max(abs(exact$parameters[1:4]-truth))==0,
          near$status %in% c("exact_fit","stationary_usable_candidate"),
          max(abs(near$parameters[1:4]-truth))<1e-4)
write.csv(data.frame(jacobian_relative_error=jacobian_error,exact_unchanged=TRUE,
                    near_parameter_error=max(abs(near$parameters[1:4]-truth))),
          file.path(output,"controls.csv"),row.names=FALSE)
paths <- sort(list.files(file.path(failures,"runs"),pattern="candidate-failure.rds$",
                         recursive=TRUE,full.names=TRUE))
stopifnot(length(paths)==32)
records <- list()
for(i in seq_along(paths)) {
  old <- readRDS(paths[i])
  fit <- solve_local(old$points,old$log_targets,old$initial_parameters)
  label <- basename(dirname(dirname(paths[i])))
  write.csv(fit$trajectory,file.path(output,paste0("trajectory-",label,".csv")),row.names=FALSE)
  records[[i]] <- data.frame(job_id=label,old_failure=old$message,status=fit$status,
      iterations=nrow(fit$trajectory),relative_residual=fit$relative_residual,
      loss=fit$loss,logdet_change=fit$logdet_change,density_change=fit$density_change,
      gradient_max=fit$gradient_max)
}
records <- do.call(rbind,records)
write.csv(records,file.path(output,"saved-cloud-reconstruction.csv"),row.names=FALSE)
nominated <- sum(records$status %in% c("exact_fit","stationary_usable_candidate"))
counts <- table(records$status)
cat("PASS: unchanged Eq15 residual/Jacobian and Gaussian non-harm controls.\n")
cat("Saved clouds:",nrow(records),"; candidates:",nominated,"\n")
print(counts)
# No R JSON dependency: all fields below are fixed strings or finite integers.
writeLines(sprintf(paste0('{"check":"PASS","equation15_identity":"PASS",',
  '"jacobian_and_exact_controls":"PASS","saved_clouds":%d,"candidate_clouds":%d,',
  '"all_clouds_usable":%s,"scientific_replication":"INCOMPLETE",',
  '"full_filter":"NOT_RUN","solver_status":"RECONSTRUCTION_HYPOTHESIS",',
  '"role":"CPU independent reference diagnostic"}'),nrow(records),nominated,
  if(nominated==nrow(records))"true" else "false"),file.path(output,"result.json"))

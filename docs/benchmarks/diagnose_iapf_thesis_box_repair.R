# Independent R/FP64 solver diagnostic. No TF/GPU or source-identity claim.
args <- commandArgs(TRUE)
stopifnot(length(args)==4, Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
out <- args[1]; old <- args[2]; src <- args[3]; previous <- args[4]
source(file.path(src,"reference_iapf_paper.R"))
source(file.path(src,"reference_iapf_scale_aware_fit.R"))
controls <- rows <- comparisons <- list()
put <- function(x,name) write.csv(do.call(rbind,x),file.path(out,name),row.names=FALSE,na="")
check <- function(name,error,tol) {
  controls[[length(controls)+1L]] <<- data.frame(check=name,error=error,
    tolerance=tol,pass=is.finite(error)&&error<=tol)
  put(controls,"controls.csv")
  if(!is.finite(error)||error>tol)stop(paste("continuation veto:",name,error))
}
projected <- function(z,ev,lower,upper) {
  g <- ev$gradient/ev$p2
  at_lower <- z<=lower+1e-7; at_upper <- z>=upper-1e-7
  g[at_lower & g>0] <- 0; g[at_upper & g<0] <- 0
  list(residual=max(abs(g)),active=sum(at_lower|at_upper))
}
repair <- function(p,z,lower,upper) {
  began <- proc.time()[3]; trace <- list(); start <- p$evaluate(z)
  if(is.null(start))stop("invalid starting density")
  status <- "segment_cap"; descent <- TRUE
  for(segment in seq_len(16)) {
    ev <- p$evaluate(z); before <- projected(z,ev,lower,upper)
    if(before$residual<=1e-6){status<-"constrained_stationary";break}
    # Both quantities are constants throughout this optim call.
    frozen_scale <- ev$scale; frozen_norm <- ev$p2
    scaled <- function(candidate) {
      value <- p$evaluate(candidate,frozen_scale)
      if(is.null(value))stop("invalid density in bounded optimizer")
      value
    }
    opt <- optim(z,function(q)scaled(q)$loss/frozen_norm,
      function(q)scaled(q)$gradient/frozen_norm,method="L-BFGS-B",
      lower=lower,upper=upper,
      control=list(factr=0,pgtol=1e-9,maxit=300,lmm=10))
    final <- p$evaluate(opt$par)
    if(is.null(final)||any(opt$par<lower-1e-9)||any(opt$par>upper+1e-9))
      stop("invalid or out-of-box optimizer endpoint")
    after <- projected(opt$par,final,lower,upper)
    increase <- final$log_raw_loss-ev$log_raw_loss
    step_descent <- (is.finite(increase)&&increase<=1e-8)||
      (is.infinite(final$log_raw_loss)&&final$log_raw_loss<0)
    trace[[length(trace)+1L]] <- data.frame(segment=segment,
      optimizer_code=opt$convergence,message=if(is.null(opt$message))"" else opt$message,
      fn_calls=unname(opt$counts[1]),gr_calls=unname(opt$counts[2]),
      projected_before=before$residual,projected_after=after$residual,
      active_boundaries=after$active,shape=final$shape,
      log_raw_loss=final$log_raw_loss,log_density_norm=final$log_density_norm,
      log_loss_change=increase,descent=step_descent)
    if(!step_descent){status<-"descent_veto";descent<-FALSE;break}
    z <- opt$par
    if(after$residual<=1e-6){status<-"constrained_stationary";break}
  }
  ev <- p$evaluate(z); pg <- projected(z,ev,lower,upper)
  list(status=status,theta=ev$theta,z=z,initial=p$theta0,
    projected_gradient=pg$residual,relative_gradient=ev$relative_gradient,
    active_boundaries=pg$active,shape=ev$shape,log_raw_loss=ev$log_raw_loss,
    log_density_change=ev$log_density_norm-start$log_density_norm,
    max_variance=max(exp(tail(ev$theta,length(ev$theta)/2))),
    log_lambda=ev$log_lambda,original_loss_descent=descent,segments=length(trace),
    trace=if(length(trace))do.call(rbind,trace) else data.frame(),
    seconds=unname(proc.time()[3]-began))
}
setTimeLimit(elapsed=240,transient=FALSE)
control <- readRDS(file.path(previous,"controls.rds"))
x <- control$x; ly <- control$ly; d <- ncol(x)
theta0 <- control$truth+c(rep(.03,d),rep(.04,d))
p <- sa_problem(x,ly,theta0); z <- rep(0,2*d); e <- p$evaluate(z)
e2 <- p$evaluate(z,e$scale+.7)
check("positive_rescaling_objective",abs(e$loss-e2$loss*exp(1.4)),1e-12)
check("positive_rescaling_gradient",max(abs(e$gradient-e2$gradient*exp(1.4))),1e-12)
b <- exp(ly-max(ly)); density <- exp(sa_log_density(x,theta0)-e$scale)
amplitude <- sum(density*b)/sum(b*b)
check("literal_profile_objective",abs(e$loss-mean((density-amplitude*b)^2)),1e-12)
analytic <- e$gradient/e$p2
fd <- vapply(seq_along(z),function(j){
  delta <- rep(0,length(z));delta[j]<-1e-6
  (p$evaluate(z+delta,e$scale)$loss-p$evaluate(z-delta,e$scale)$loss)/(2e-6*e$p2)
},numeric(1))
check("fixed_scale_gradient_finite_difference",max(abs(fd-analytic))/max(1,max(abs(analytic))),5e-7)
lower <- c(rep(-2,d),rep(-log(4),d));upper <- -lower
good <- repair(p,z,lower,upper)
check("exact_gaussian_recovery",max(abs(good$theta-control$truth)),1e-5)
check("exact_gaussian_projected_gradient",good$projected_gradient,1e-6)
known <- control$bounded
no_fire <- repair(sa_problem(x,ly,known$initial),known$z,lower,upper)
check("box_no_fire_at_known_minimum",max(abs(no_fire$theta-known$theta)),1e-8)
check("known_minimum_active_boundaries",no_fire$active_boundaries,0)
saveRDS(list(recovered=good,no_fire=no_fire),file.path(out,"controls.rds"))
starts <- read.csv(file.path(previous,"heuristic_starts.csv"))
for(seed in c(9262001,9262002))for(power in c(1,2))for(radius in c(2,4)) {
  label <- paste(seed,power,sep="_")
  payload <- readRDS(file.path(old,paste0("min_mass_Nminus",power,"-",seed,".rds")))
  stopifnot(payload$time==98L,payload$N==1000L)
  x <- payload$inputs[[98]]$points;ly <- payload$inputs[[98]]$log_targets
  saved <- readRDS(file.path(previous,paste(label,"log_quadratic","profiled",
    "scaled","tight",radius,"rds",sep=".")))
  p <- sa_problem(x,ly,saved$initial);d <- ncol(x)
  lower <- c(rep(-radius,d),rep(-log(4),d));upper <- -lower
  ev <- p$evaluate(saved$z);before <- projected(saved$z,ev,lower,upper)
  check(paste("previous_endpoint_gradient",label,radius,sep="_"),
    abs(before$residual-saved$diagnostics$projected_gradient)/max(1,before$residual),1e-8)
  result <- repair(p,saved$z,lower,upper)
  stem <- paste(label,radius,sep=".")
  saveRDS(result,file.path(out,paste0(stem,".rds")))
  write.csv(result$trace,file.path(out,paste0(stem,".trace.csv")),row.names=FALSE,na="")
  rows[[length(rows)+1L]] <- data.frame(case=label,box=radius,status=result$status,
    projected_before=before$residual,projected_after=result$projected_gradient,
    relative_gradient=result$relative_gradient,active_boundaries=result$active_boundaries,
    initial_shape=ev$shape,shape=result$shape,log_raw_loss=result$log_raw_loss,
    log_density_change=result$log_density_change,max_variance=result$max_variance,
    original_loss_descent=result$original_loss_descent,segments=result$segments,
    seconds=result$seconds)
  put(rows,"box_repairs.csv")
  for(j in which(starts$case==label)) {
    comparisons[[length(comparisons)+1L]] <- data.frame(case=label,box=radius,
      heuristic=starts$kind[j],heuristic_shape=starts$shape[j],fitted_shape=result$shape,
      diagnostic_role="EXPLANATORY_FIT_ONLY")
  }
  put(comparisons,"conditional_shape_diagnostics.csv")
  if(!result$original_loss_descent)stop("continuation veto: original objective increased")
}
setTimeLimit(cpu=Inf,elapsed=Inf,transient=FALSE)
allrows <- do.call(rbind,rows)
decision <- data.frame(status="COMPLETE",phase="box_repair",
  numerical_controls=all(vapply(controls,function(x)x$pass,logical(1))),
  bounded_fits=nrow(allrows),constrained_stationary=sum(allrows$status=="constrained_stationary"),
  nonstationary=sum(allrows$status!="constrained_stationary"),
  source_identity="CONSTRAINED_RECONSTRUCTION",
  original_loss_descent=all(allrows$original_loss_descent),
  scientific_replication="INCOMPLETE",filter_quality="NOT_EVALUATED",
  ranking="NOT_SUPPORTED",TF_GPU_status="NOT_CHECKED")
write.csv(decision,file.path(out,"decision.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_session.txt"))
print(decision)

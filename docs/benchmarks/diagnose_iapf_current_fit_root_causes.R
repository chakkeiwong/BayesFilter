# Independent R/FP64 CPU forensic diagnostic; not a runtime or fit promotion.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3L,Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
src <- args[1]; old <- args[2]; out <- args[3]
source(file.path(src,"reference_iapf_paper.R"))
source(file.path(src,"reference_iapf_scale_aware_fit.R"))
checks <- geometry <- trajectories <- fits <- cases <- list()
save_rows <- function(rows,name) write.csv(do.call(rbind,rows),file.path(out,name),row.names=FALSE)
check <- function(name,error,tol=1e-10) {
  checks[[length(checks)+1L]] <<- data.frame(check=name,error=error,tolerance=tol,
    pass=is.finite(error)&&error<=tol)
  save_rows(checks,"checks.csv")
  if(!is.finite(error)||error>tol)stop(paste("continuation veto",name,error))
}
independent_normal <- function(x,m,V) {
  dx <- sweep(x,2,as.numeric(m))
  -.5*(ncol(x)*log(2*pi)+as.numeric(determinant(V,logarithm=TRUE)$modulus)+
    rowSums((dx%*%solve(V))*dx))
}
for(seed in c(9262001,9262002)) for(power in c(1,2)) {
  label <- paste(seed,power,sep="_")
  p <- readRDS(file.path(old,paste0("min_mass_Nminus",power,"-",seed,".rds")))
  data <- readRDS(file.path(old,paste0("data-",seed,".rds")))
  model <- data$bundle$model; d <- model$dimension
  stopifnot(p$time==98L,d==5L,model$horizon==100L,p$N==1000L)
  for(tt in c(100,99,98)) {
    input <- p$inputs[[tt]]; x <- input$points; ly <- input$log_targets
    check(paste(label,tt,"saved_cloud"),max(abs(x-data$baseline$clouds[[tt]])),0)
    y <- model$observations[tt,]
    logg <- independent_normal(x,y,diag(d)); no_floor <- logg
    V <- diag(d); m <- y; floor_effect <- rep(0,nrow(x))
    if(tt<100) {
      tw <- input$next_twist; S <- model$transition_covariance+tw$covariance
      logF <- independent_normal(x%*%t(model$A),tw$mean,S)
      no_floor <- logg+logF
      floor_effect <- pmax(0,tw$log_floor-logF)+log1p(exp(-abs(tw$log_floor-logF)))
      P <- diag(d)+t(model$A)%*%solve(S,model$A)
      V <- solve(P); m <- as.vector(V%*%(y+t(model$A)%*%solve(S,tw$mean)))
    }
    check(paste(label,tt,"independent_target"),max(abs(ly-no_floor-floor_effect)))
    full_density <- independent_normal(x,m,V)
    check(paste(label,tt,"gaussian_product"),max(abs((no_floor-full_density)-mean(no_floor-full_density))))
    center <- colMeans(x); units <- sqrt(colMeans(sweep(x,2,center)^2))
    u <- sweep(sweep(x,2,center),2,units,"/"); design <- cbind(1,u,u^2)
    log_reg <- lm.fit(design,ly-max(ly)); nofloor_reg <- lm.fit(design,no_floor-max(no_floor))
    corr <- V/sqrt(outer(diag(V),diag(V)))
    w <- exp(ly-max(ly)); w <- w/sum(w)
    geometry[[length(geometry)+1L]] <- data.frame(seed=seed,power=power,time=tt,
      diag_log_target_rms=sqrt(mean(log_reg$residuals^2)),
      nofloor_diag_log_target_rms=sqrt(mean(nofloor_reg$residuals^2)),
      floor_log_target_max=max(floor_effect),floor_log_target_median=median(floor_effect),
      min_gaussian_branch_share=min(exp(-floor_effect)),
      floor_share_target_squared_weighted=sum(w*w*(1-exp(-floor_effect)))/sum(w*w),
      target_ess=1/sum(w*w),target_squared_ess=sum(w*w)^2/sum(w^4),target_max_share=max(w),
      target_mean_cloud_z_max=max(abs((m-center)/units)),
      target_correlation_max=max(abs(corr-diag(d))),covariance_condition=kappa(V,exact=TRUE),
      initial_shape=p$diagnostics[[tt]]$trace$shape[1],final_shape=p$diagnostics[[tt]]$shape)
    if(tt==98) {
      tr <- p$fit$trace; selected <- unique(c(1:6,which.min(tr$shape),100,400,nrow(tr)))
      trajectories[[length(trajectories)+1L]] <- cbind(seed=seed,power=power,tr[selected,])
      theta0 <- p$fit$initial; prob <- sa_problem(x,ly,theta0); ev <- prob$evaluate(rep(0,2*d))
      b <- exp(ly-max(ly)); dx <- sweep(x,2,theta0[1:d]); v <- exp(theta0[d+1:d])
      Jp <- cbind(sweep(dx,2,v,"/"),.5*(sweep(dx^2,2,v,"/")-1))
      Jp <- sweep(Jp,2,prob$units,"*")*ev$p
      Hschur <- crossprod(Jp)-tcrossprod(as.vector(crossprod(Jp,b)))/sum(b*b)
      check(paste(label,"joint_profiled_GN_Schur"),max(abs(Hschur-crossprod(ev$J)))/max(abs(Hschur)))
      h <- 1e-5; fd <- vapply(seq_len(2*d),function(j){
        z <- rep(0,2*d);z[j]<-h
        (prob$evaluate(z,ev$scale)$loss-prob$evaluate(-z,ev$scale)$loss)/(2*h)
      },0.0)
      check(paste(label,"start_gradient_fd"),max(abs(fd-ev$gradient))/max(abs(ev$gradient)),1e-5)
      cases[[label]] <- list(seed=seed,power=power,x=x,ly=ly,theta0=theta0,
        no_floor=no_floor,diagonal=independent_normal(x,m,diag(diag(V))),saved=p$fit)
    }
  }
}
save_rows(geometry,"geometry.csv");save_rows(trajectories,"saved_trajectory_excerpt.csv")
cat("Geometry and analytic checks PASS; starting bounded solver probes\n")

# Joint LM retains log(lambda) as an optimized coordinate, preserving Eq15.
joint_lm <- function(x,ly,theta0,maxit=800L) {
  prob <- sa_problem(x,ly,theta0); k <- length(theta0); d <- k/2
  first <- prob$evaluate(rep(0,k)); base_lambda <- first$log_lambda
  b <- exp(ly-max(ly)); n <- nrow(x)
  evaluate <- function(z,scale=NULL) {
    theta <- theta0+prob$units*z[seq_len(k)]
    if(any(!is.finite(theta))||any(abs(theta[d+1:d])>600))return(NULL)
    lp <- sa_log_density(x,theta); ll <- base_lambda+z[k+1]
    if(any(!is.finite(lp))||!is.finite(ll))return(NULL)
    if(is.null(scale))scale<-max(lp)
    if(max(lp-scale)>300||ll+max(ly)-scale>300)return(NULL)
    p <- exp(lp-scale); q <- exp(ll+max(ly)-scale); r<-p-q*b;p2<-mean(p*p)
    if(!is.finite(p2)||p2<=0)return(NULL)
    dx<-sweep(x,2,theta[1:d]);v<-exp(theta[d+1:d])
    Jp<-sweep(cbind(sweep(dx,2,v,"/"),.5*(sweep(dx^2,2,v,"/")-1)),2,prob$units,"*")*p
    J<-cbind(Jp,-q*b);loss<-mean(r*r);gradient<-as.vector(2*crossprod(J,r)/n)
    list(loss=loss,J=J,gradient=gradient,p2=p2,scale=scale,theta=theta,log_lambda=ll,
      log_raw_loss=2*scale+log(loss),relative_gradient=max(abs(gradient))/p2)
  }
  z<-rep(0,k+1);damping<-1e-3;nu<-2;status<-"iteration_cap";tr<-list()
  # Validate this extra-coordinate derivative before interpreting joint trajectories.
  ev0<-evaluate(z);h<-1e-5
  fd<-vapply(seq_along(z),function(j){zz<-z;zz[j]<-h
    (evaluate(zz,ev0$scale)$loss-evaluate(-zz,ev0$scale)$loss)/(2*h)},0.0)
  check(paste("joint_start_fd",length(checks)),max(abs(fd-ev0$gradient))/max(abs(ev0$gradient)),1e-5)
  for(iteration in seq_len(maxit)) {
    ev<-evaluate(z);tr[[length(tr)+1]]<-data.frame(iteration=iteration,
      log_raw_loss=ev$log_raw_loss,relative_gradient=ev$relative_gradient)
    if(ev$relative_gradient<=1e-6){status<-"stationary";break}
    H<-crossprod(ev$J)/n;g<-ev$gradient/2;peak<-max(diag(H));accepted<-FALSE
    if(!is.finite(peak)||peak<=0){status<-"singular_jacobian";break}
    D<-pmax(diag(H),sqrt(.Machine$double.eps)*peak)
    for(trial in seq_len(32)) {
      step<-tryCatch(as.vector(solve(H+diag(damping*D,length(D)),-g)),error=function(e)NULL)
      candidate<-if(is.null(step))NULL else evaluate(z+step,ev$scale)
      predicted<-if(is.null(step))-Inf else -2*sum(g*step)-sum(step*(H%*%step))
      rho<-if(is.null(candidate)||predicted<=0)-Inf else (ev$loss-candidate$loss)/predicted
      if(is.finite(rho)&&rho>0&&candidate$loss<ev$loss){
        z<-z+step;damping<-max(.Machine$double.eps,damping*max(1/3,1-(2*rho-1)^3))
        nu<-2;accepted<-TRUE;break
      }
      damping<-damping*nu;nu<-min(2*nu,1e6)
      if(!is.finite(damping)||damping>1/.Machine$double.eps)break
    }
    if(!accepted){status<-"no_descent_step";break}
  }
  final<-evaluate(z)
  list(theta=final$theta,status=status,iterations=iteration,log_lambda=final$log_lambda,
    joint_relative_gradient=final$relative_gradient,trace=do.call(rbind,tr))
}
record_fit <- function(case,variant,solver,target,fit,elapsed) {
  prob<-sa_problem(case$x,target,case$theta0); z<-(fit$theta-case$theta0)/prob$units
  ev<-prob$evaluate(z);stopifnot(!is.null(ev))
  candidate<-ev$relative_gradient<=1e-6
  eig<-if(candidate)sa_curvature(prob,z,ev) else c(NA_real_,NA_real_)
  curvature_pass<-candidate&&all(is.finite(eig))&&eig[1]>=-1e-5*max(1,abs(eig[2]))
  lambda_log_gap<-if(is.null(fit$log_lambda))0 else abs(fit$log_lambda-ev$log_lambda)
  fits[[length(fits)+1L]]<<-data.frame(seed=case$seed,power=case$power,variant=variant,
    solver=solver,status=fit$status,iterations=fit$iterations,relative_gradient=ev$relative_gradient,
    shape=ev$shape,log_raw_loss=ev$log_raw_loss,log_density_norm=ev$log_density_norm,
    max_variance=max(exp(fit$theta[length(fit$theta)/2+seq_len(length(fit$theta)/2)])),
    lambda_log_gap=lambda_log_gap,curvature_min=eig[1],
    numerical_screen_pass=curvature_pass&&lambda_log_gap<=1e-6,elapsed_seconds=elapsed)
  saveRDS(fit,file.path(out,paste(case$seed,case$power,variant,solver,"rds",sep=".")))
  save_rows(fits,"solver_probes.csv")
}
for(case in cases) {
  record_fit(case,"actual","saved_profiled_LM",case$ly,case$saved,0)
  for(variant in c("no_floor","diagonal")) {
    target<-case[[variant]];start<-proc.time()[3]
    fit<-sa_fit(case$x,target,theta0=case$theta0,maxit=800)
    record_fit(case,variant,"profiled_LM",target,fit,proc.time()[3]-start)
  }
  start<-proc.time()[3];prob<-sa_problem(case$x,case$ly,case$theta0)
  z0<-rep(0,length(case$theta0));scale<-prob$evaluate(z0)$scale
  bfgs<-optim(z0,function(z){ev<-prob$evaluate(z,scale);if(is.null(ev))1e100 else ev$loss},
    function(z){ev<-prob$evaluate(z,scale);if(is.null(ev))rep(0,length(z)) else ev$gradient},
    method="BFGS",control=list(maxit=800,reltol=1e-14))
  fit<-list(theta=case$theta0+prob$units*bfgs$par,status=paste0("optim_code_",bfgs$convergence),
    iterations=unname(bfgs$counts[1]),counts=bfgs$counts)
  record_fit(case,"actual","profiled_BFGS",case$ly,fit,proc.time()[3]-start)
  start<-proc.time()[3];fit<-joint_lm(case$x,case$ly,case$theta0)
  record_fit(case,"actual","joint_LM",case$ly,fit,proc.time()[3]-start)
}
writeLines(capture.output(sessionInfo()),file.path(out,"R_session.txt"))
cat("All planned diagnostics completed; results preserved in CSV/RDS\n")

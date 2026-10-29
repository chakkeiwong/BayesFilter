# Independent R/FP64 CPU diagnostic. Not a BayesFilter runtime or source identity.
args <- commandArgs(TRUE); stopifnot(length(args)==3,Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
out <- args[1]; old <- args[2]; src <- args[3]
source(file.path(src,"reference_iapf_paper.R"))
source(file.path(src,"reference_iapf_scale_aware_fit.R"))
rows <- controls <- starts <- sweep_rows <- list()
put <- function(x,file) write.csv(do.call(rbind,x),file.path(out,file),row.names=FALSE,na="")
check <- function(name,error,tol) {
  controls[[length(controls)+1L]] <<- data.frame(check=name,error=error,tolerance=tol,
    pass=is.finite(error)&&error<=tol)
  put(controls,"controls.csv")
  if(!is.finite(error)||error>tol)stop(paste("continuation veto",name,error))
}
# In joint mode lambda has a linear, dimensionless coordinate. The initial
# lambda is profiled, then it is independently optimized; it may cross zero.
problem <- function(x,ly,theta0,amplitude,scaling) {
  d <- ncol(x); k <- 2*d; units <- c(exp(theta0[d+seq_len(d)]/2),rep(1,d))
  b <- exp(ly-max(ly)); b2 <- sum(b*b)
  scale <- if(scaling=="scaled")max(sa_log_density(x,theta0)) else 0
  p0 <- exp(sa_log_density(x,theta0)-scale); a0 <- sum(p0*b)/b2
  evaluate <- function(z) {
    theta <- theta0+units*z[seq_len(k)]
    if(any(!is.finite(theta))||any(abs(theta[d+seq_len(d)])>600))return(NULL)
    lp <- sa_log_density(x,theta)
    if(any(!is.finite(lp))||max(lp-scale)>300)return(NULL)
    p <- exp(lp-scale); p2 <- mean(p*p)
    if(!is.finite(p2)||p2<=0)return(NULL)
    a <- if(amplitude=="joint")a0*(1+z[k+1]) else sum(p*b)/b2
    r <- p-a*b; dx <- sweep(x,2,theta[seq_len(d)]); v <- exp(theta[d+seq_len(d)])
    score <- cbind(sweep(dx,2,v,"/"),.5*(sweep(dx^2,2,v,"/")-1))
    Jp <- sweep(score,2,units,"*")*p
    J <- if(amplitude=="joint")cbind(Jp,-a0*b) else Jp-outer(b,colSums(Jp*b)/b2)
    loss <- sum(r*r); gradient <- as.vector(2*crossprod(J,r))
    if(!is.finite(loss)||any(!is.finite(gradient)))return(NULL)
    list(theta=theta,loss=loss,gradient=gradient,a=a,scale=scale)
  }
  list(evaluate=evaluate,z0=rep(0,k+as.integer(amplitude=="joint")),units=units,scale=scale)
}
initial <- function(x,ly,kind) {
  if(kind=="log_quadratic")return(iapf_fit_initial(x,ly,TRUE)$parameters)
  w <- if(kind=="weighted_moments")exp(ly-max(ly)) else rep(1,nrow(x))
  w <- w/sum(w); m <- colSums(x*w); v <- colSums(sweep(x,2,m)^2*w)
  if(any(!is.finite(v))||any(v<=0))stop("invalid moment start")
  c(m,log(v))
}
diagnostics <- function(x,ly,theta0,ev,amplitude) {
  d <- ncol(x); k <- 2*d; p <- sa_problem(x,ly,theta0)
  z <- (ev$theta-theta0)/p$units; e <- p$evaluate(z); e0 <- p$evaluate(rep(0,k))
  if(is.null(e))stop("invalid final density")
  b <- exp(ly-max(ly)); a <- ev$a*exp(ev$scale-e$scale); residual <- e$p-a*b
  dx <- sweep(x,2,ev$theta[seq_len(d)]); v <- exp(ev$theta[d+seq_len(d)])
  score <- cbind(sweep(dx,2,v,"/"),.5*(sweep(dx^2,2,v,"/")-1))
  Jp <- sweep(score,2,p$units,"*")*e$p
  g <- as.vector(2*crossprod(Jp,residual)/nrow(x))/e$p2
  ga <- -2*mean(b*residual)*sqrt(e$p2/mean(b*b))/e$p2
  relative_gradient <- max(abs(c(g,ga)))
  eig <- if(is.finite(relative_gradient)&&relative_gradient<=1e-6)
    sa_curvature(p,z,e) else c(min=NA_real_,max=NA_real_)
  stationary <- is.finite(relative_gradient)&&relative_gradient<=1e-6&&a>0&&
    all(is.finite(eig))&&eig[1]>=-1e-5*max(1,abs(eig[2]))
  list(relative_gradient=relative_gradient,profiled_relative_gradient=e$relative_gradient,
    shape=mean(residual*residual)/e$p2,log_raw_loss=log(mean(residual*residual))+2*e$scale,log_density_change=e$log_density_norm-e0$log_density_norm,
    max_variance=max(v),eigen_min=unname(eig[1]),eigen_max=unname(eig[2]),
    stationary=stationary,g=g,ga=ga)
}
fit <- function(x,ly,kind,amplitude,scaling,stopping,box=0) {
  theta0 <- initial(x,ly,kind); p <- problem(x,ly,theta0,amplitude,scaling)
  fn <- function(z) { e<-p$evaluate(z); if(is.null(e))1e100 else e$loss }
  gr <- function(z) { e<-p$evaluate(z); if(is.null(e))rep(0,length(z)) else e$gradient }
  began <- proc.time()[3]; d<-ncol(x); k<-2*d
  if(box>0) {
    stopifnot(amplitude=="profiled",scaling=="scaled")
    lower<-c(rep(-box,d),rep(-log(4),d)); upper<- -lower
    opt<-optim(p$z0,fn,gr,method="L-BFGS-B",lower=lower,upper=upper,
      control=list(maxit=800,factr=1,pgtol=1e-10))
  } else {
    ctl<-if(stopping=="default")list(maxit=100,reltol=sqrt(.Machine$double.eps)) else list(maxit=800,reltol=1e-14)
    opt<-optim(p$z0,fn,gr,method="BFGS",control=ctl)
  }
  ev<-p$evaluate(opt$par)
  if(is.null(ev))stop("invalid optimizer endpoint")
  dg<-diagnostics(x,ly,theta0,ev,amplitude)
  dg$projected_gradient<-NA_real_; dg$active_boundaries<-0L; dg$constrained_stationary<-FALSE
  if(box>0) {
    at_lower<-opt$par<=lower+1e-7;at_upper<-opt$par>=upper-1e-7
    pg<-dg$g;pg[at_lower & pg>0]<-0;pg[at_upper & pg<0]<-0
    dg$projected_gradient<-max(abs(pg));dg$active_boundaries<-sum(at_lower|at_upper)
    dg$constrained_stationary<-is.finite(dg$projected_gradient)&&dg$projected_gradient<=1e-6&&ev$a>0
  }
  list(theta=ev$theta,initial=theta0,z=opt$par,objective=opt$value,
    optimizer_code=opt$convergence,fn_calls=unname(opt$counts[1]),gr_calls=unname(opt$counts[2]),
    seconds=unname(proc.time()[3]-began),diagnostics=dg)
}
record <- function(f,case,kind,amplitude,scaling,stopping,box=0,error="") {
  values<-if(is.null(f))list() else f$diagnostics
  pick<-function(name,default=NA)if(is.null(values[[name]]))default else values[[name]]
  row<-data.frame(case=case,kind=kind,amplitude=amplitude,scaling=scaling,stopping=stopping,box=box,
    optimizer_code=if(is.null(f))NA else f$optimizer_code,fn_calls=if(is.null(f))NA else f$fn_calls,
    relative_gradient=pick("relative_gradient"),profiled_relative_gradient=pick("profiled_relative_gradient"),
    shape=pick("shape"),log_raw_loss=pick("log_raw_loss"),log_density_change=pick("log_density_change"),
    max_variance=pick("max_variance"),eigen_min=pick("eigen_min"),eigen_max=pick("eigen_max"),
    stationary=pick("stationary",FALSE),projected_gradient=pick("projected_gradient"),
    active_boundaries=pick("active_boundaries",0),constrained_stationary=pick("constrained_stationary",FALSE),
    seconds=if(is.null(f))NA else f$seconds,error=error)
  rows[[length(rows)+1L]]<<-row;put(rows,"fits.csv")
  if(!is.null(f))saveRDS(f,file.path(out,paste(case,kind,amplitude,scaling,stopping,box,"rds",sep=".")))
}
# Independent representable Gaussian, finite-difference gradients, and equality
# to the literal joint sum of squared errors at the profiled initial lambda.
set.seed(9271001);d<-5;x<-matrix(rnorm(1000*d),ncol=d)
truth<-c(seq(-.2,.2,length.out=d),seq(-.1,.1,length.out=d))
ly<-iapf_log_normal(x,truth[seq_len(d)],diag(exp(truth[d+seq_len(d)]),d))+1.7
theta0<-truth+c(rep(.03,d),rep(.04,d))
for(amplitude in c("joint","profiled"))for(scaling in c("scaled","raw")) {
  p<-problem(x,ly,theta0,amplitude,scaling);ev<-p$evaluate(p$z0)
  h<-.Machine$double.eps^(1/3);fd<-sapply(seq_along(p$z0),function(j){
    dz<-p$z0;dz[j]<-h;(p$evaluate(p$z0+dz)$loss-p$evaluate(p$z0-dz)$loss)/(2*h)})
  check(paste(amplitude,scaling,"gradient"),max(abs(fd-ev$gradient))/max(abs(ev$gradient)),1e-5)
  rawp<-exp(iapf_log_normal(x,theta0[seq_len(d)],diag(exp(theta0[d+seq_len(d)]),d)))
  rawy<-exp(ly);lambda<-sum(rawp*rawy)/sum(rawy^2)
  direct<-sum((rawp-lambda*rawy)^2)
  check(paste(amplitude,scaling,"literal_objective"),abs(ev$loss*exp(2*p$scale)-direct)/direct,1e-10)
}
# No-fire test starts at an exact known minimizer. It checks domain guards
# preserve a healthy solution; it does not establish trajectory equivalence.
good<-fit(x,ly,"log_quadratic","profiled","scaled","tight")
bounded_good<-fit(x,ly,"log_quadratic","profiled","scaled","tight",2)
check("exact_gaussian_recovery",max(abs(good$theta-truth)),1e-6)
check("box_no_fire_at_known_minimum",max(abs(good$theta-bounded_good$theta)),1e-8)
check("box_control_boundaries",bounded_good$diagnostics$active_boundaries,0)
saveRDS(list(x=x,ly=ly,truth=truth,unbounded=good,bounded=bounded_good),file.path(out,"controls.rds"))
cases<-list()
for(seed in c(9262001,9262002))for(power in c(1,2)){
  p<-readRDS(file.path(old,paste0("min_mass_Nminus",power,"-",seed,".rds")))
  stopifnot(p$time==98L,p$N==1000L)
  cases[[paste(seed,power,sep="_")]]<-list(x=p$inputs[[98]]$points,ly=p$inputs[[98]]$log_targets)
}
setTimeLimit(elapsed=240,transient=FALSE)
grid<-expand.grid(amplitude=c("joint","profiled"),kind=c("log_quadratic","weighted_moments","cloud_moments"),
  stopping=c("default","tight"),scaling=c("scaled","raw"),stringsAsFactors=FALSE)
for(label in names(cases)){
  case<-cases[[label]]
  for(kind in c("log_quadratic","weighted_moments","cloud_moments")){
    th<-initial(case$x,case$ly,kind);ev<-sa_problem(case$x,case$ly,th)$evaluate(rep(0,10))
    starts[[length(starts)+1L]]<-data.frame(case=label,kind=kind,shape=ev$shape,
      log_raw_loss=ev$log_raw_loss,relative_gradient=ev$relative_gradient)
  }
  put(starts,"heuristic_starts.csv")
  for(i in seq_len(nrow(grid))){
    a<-grid[i,];error<-""
    f<-tryCatch(fit(case$x,case$ly,a$kind,a$amplitude,a$scaling,a$stopping),error=function(e){error<<-conditionMessage(e);NULL})
    record(f,label,a$kind,a$amplitude,a$scaling,a$stopping,error=error)
  }
}
calibration<-do.call(rbind,rows)
nominees<-grid[vapply(seq_len(nrow(grid)),function(i){
  a<-grid[i,];r<-calibration[calibration$kind==a$kind&calibration$amplitude==a$amplitude&
    calibration$scaling==a$scaling&calibration$stopping==a$stopping,]
  nrow(r)==4&&all(r$stationary)
},logical(1)),,drop=FALSE]
# Fixed arm order avoids ranking on descriptive fitting loss.
if(nrow(nominees)>0){
  nominees<-nominees[order(match(nominees$amplitude,c("joint","profiled")),
    match(nominees$kind,c("log_quadratic","weighted_moments","cloud_moments")),
    match(nominees$stopping,c("default","tight")),match(nominees$scaling,c("scaled","raw"))),,drop=FALSE]
  nominees<-head(nominees,2)
}
write.csv(nominees,file.path(out,"nominees.csv"),row.names=FALSE)
if(nrow(nominees)==0)for(label in names(cases))for(radius in c(2,4)){
  case<-cases[[label]];error<-""
  f<-tryCatch(fit(case$x,case$ly,"log_quadratic","profiled","scaled","tight",radius),
    error=function(e){error<<-conditionMessage(e);NULL})
  record(f,label,"log_quadratic","profiled","scaled","tight",radius,error)
}
setTimeLimit(cpu=Inf,elapsed=Inf,transient=FALSE)
# These are fresh backward sweeps, not the adaptive filter or a quality study.
if(nrow(nominees)>0)for(i in seq_len(nrow(nominees)))for(seed in c(9274001,9274002))for(power in c(1,2)){
  a<-nominees[i,];data<-iapf_paper_data(d=5,horizon=100,seed=seed)
  model<-iapf_linear_model(data$observations,data$A)
  set.seed(seed+100);baseline<-iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),1000)
  twists<-vector("list",100);allfits<-list();status<-"complete";failed<-NA_integer_
  for(tt in 100:1){
    ly<-iapf_backward_log_target(model,baseline$clouds[[tt]],tt,if(tt==100)NULL else twists[[tt+1]])
    f<-tryCatch(fit(baseline$clouds[[tt]],ly,a$kind,a$amplitude,a$scaling,a$stopping),error=function(e)NULL)
    allfits[[as.character(tt)]]<-f
    if(is.null(f)||!isTRUE(f$diagnostics$stationary)){status<-"fit_failed";failed<-tt;break}
    twists[[tt]]<-sa_twist(f$theta,1000,power,model$transition_covariance)
  }
  sweep_rows[[length(sweep_rows)+1L]]<-data.frame(arm=i,seed=seed,power=power,status=status,failed_time=failed,
    fitted_times=sum(vapply(allfits,function(f)!is.null(f)&&isTRUE(f$diagnostics$stationary),logical(1))))
  put(sweep_rows,"backward_sweeps.csv")
  saveRDS(list(config=a,data=data,baseline=baseline,twists=twists,fits=allfits),
    file.path(out,paste("sweep",i,seed,power,"rds",sep=".")))
}
allrows<-do.call(rbind,rows)
decision<-data.frame(status="COMPLETE",source_identity="PARTIAL_RECONSTRUCTION",
  numerical_controls=all(vapply(controls,function(x)x$pass,logical(1))),
  unbounded_fits=sum(allrows$box==0),stationary_unbounded=sum(allrows$stationary&allrows$box==0),
  nominated_arms=nrow(nominees),bounded_fits=sum(allrows$box>0),
  constrained_stationary=sum(allrows$constrained_stationary&allrows$box>0),
  complete_sweeps=if(length(sweep_rows))sum(vapply(sweep_rows,function(x)x$status=="complete",logical(1))) else 0,
  scientific_replication="INCOMPLETE",TF_GPU_status="NOT_CHECKED",ranking="NOT_SUPPORTED")
write.csv(decision,file.path(out,"decision.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_session.txt"))
print(decision)

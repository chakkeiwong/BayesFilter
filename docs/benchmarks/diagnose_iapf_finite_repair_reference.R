# Independent R/FP64 CPU reference diagnostic; no production/default claim.
args <- commandArgs(TRUE)
stopifnot(length(args)==3, Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
phase <- args[1]; out <- args[2]; src <- args[3]
source(file.path(src,"reference_iapf_paper.R"))
source(file.path(src,"reference_iapf_scale_aware_fit.R"))
write_rows <- function(rows,name) {
  if(length(rows))write.csv(do.call(rbind,rows),file.path(out,name),row.names=FALSE,na="")
}
finite_problem <- function(x,ly,theta0,scaled) {
  d<-ncol(x); units<-c(exp(theta0[d+seq_len(d)]/2),rep(1,d))
  scale<-if(scaled)max(sa_log_density(x,theta0)) else 0
  b<-exp(ly-max(ly)); p0<-exp(sa_log_density(x,theta0)-scale)
  a0<-sum(p0*b)/sum(b*b)
  stopifnot(is.finite(a0),a0>0)
  evaluate<-function(z) {
    theta<-theta0+units*z[seq_len(2*d)]; v<-exp(theta[d+seq_len(d)])
    if(any(!is.finite(theta))||any(!is.finite(v))||any(v<=0))stop("invalid covariance iterate")
    dx<-sweep(x,2,theta[seq_len(d)])
    p<-exp(sa_log_density(x,theta)-scale); a<-a0*(1+z[2*d+1])
    residual<-p-a*b
    score<-cbind(sweep(dx,2,v,"/"),.5*(sweep(dx^2,2,v,"/")-1))
    J<-cbind(sweep(score,2,units,"*")*p,-a0*b)
    g<-as.vector(2*crossprod(J,residual)); loss<-sum(residual^2)
    if(!is.finite(loss)||any(!is.finite(g)))stop("nonfinite objective or derivative")
    list(theta=theta,a=a,p=p,residual=residual,gradient=g,loss=loss,scale=scale)
  }
  list(evaluate=evaluate,z0=rep(0,2*d+1),units=units,a0=a0,scale=scale)
}
finite_fit <- function(x,ly,N,scaled) {
  initial<-iapf_fit_initial(x,ly,FALSE)
  theta0<-initial$parameters; p<-finite_problem(x,ly,theta0,scaled)
  opt<-optim(p$z0,function(z)p$evaluate(z)$loss,function(z)p$evaluate(z)$gradient,
    method="BFGS",control=list(maxit=100,reltol=sqrt(.Machine$double.eps)))
  ev<-p$evaluate(opt$par)
  if(!is.finite(ev$a)||ev$a<=0)stop("nonpositive amplitude endpoint")
  if(!opt$convergence %in% c(0L,1L))stop("unsupported optimizer termination")
  d<-ncol(x); var<-exp(ev$theta[d+seq_len(d)])
  # Shape and escape are diagnostics, not stationarity-based vetoes.
  q<-sa_problem(x,ly,theta0); z<-(ev$theta-theta0)/q$units
  dg<-q$evaluate(z); di<-q$evaluate(rep(0,2*d))
  if(is.null(dg)||is.null(di))stop("invalid diagnostic density normalization")
  a_scaled<-ev$a*exp(ev$scale-dg$scale)
  shape<-mean((dg$p-a_scaled*exp(ly-max(ly)))^2)/dg$p2
  twist<-sa_twist(ev$theta,N,1,diag(d))
  list(twist=twist,diagnostics=data.frame(optimizer_code=opt$convergence,
    function_calls=unname(opt$counts[1]),gradient_calls=unname(opt$counts[2]),
    initializer=initial$method,shape=shape,initial_shape=di$shape,
    relative_profile_gradient=dg$relative_gradient,
    log_density_change=dg$log_density_norm-di$log_density_norm,
    minimum_variance=min(var),maximum_variance=max(var),
    amplitude_log=log(ev$a)+ev$scale-max(ly),log_floor=twist$log_floor))
}
controls<-list()
check<-function(name,error,tolerance) {
  ok<-is.finite(error)&&error<=tolerance
  controls[[length(controls)+1L]]<<-data.frame(check=name,error=error,tolerance=tolerance,pass=ok)
  write_rows(controls,"controls.csv")
  if(!ok)stop(paste("continuation veto:",name))
}
if(phase=="controls") {
  set.seed(9280001); x<-matrix(rnorm(120*2),120,2);theta0<-c(.2,-.3,log(.8),log(1.3))
  ly<-iapf_log_normal(x,c(.1,.3),diag(c(.7,1.1)))
  for(scaled in c(FALSE,TRUE)) {
    p<-finite_problem(x,ly,theta0,scaled);z<-c(.04,-.07,.02,.03,.05)
    g<-p$evaluate(z)$gradient;fd<-numeric(length(z));h<-1e-5
    for(j in seq_along(z)) {dz<-rep(0,length(z));dz[j]<-h
      fd[j]<-(p$evaluate(z+dz)$loss-p$evaluate(z-dz)$loss)/(2*h)}
    check(paste0("joint_gradient_",scaled),max(abs(g-fd))/(1e-8+max(abs(g))),3e-5)
    fit<-finite_fit(x,ly,1000,scaled)
    check(paste0("finite_recipe_covariance_",scaled),as.numeric(any(diag(fit$twist$covariance)<=0)),0)
  }
  means<-matrix(rnorm(120*2),120,2);Q<-diag(2);V<-diag(c(.7,1.1));m<-c(.1,.3)
  for(delta in c(0,1e-6,.001)) {
    floor<-if(delta==0)-Inf else log(delta)-log1p(-delta)+
      iapf_log_normal(matrix(m,1),m,Q+V)
    tw<-iapf_gaussian_twist(m,V,floor);pr<-iapf_proposal(means,Q,tw)
    rhs<-iapf_log_normal(x,means,Q)+iapf_log_twist(x,tw)-pr$log_integral
    check(paste0("proposal_density_identity_",delta),max(abs(rhs-iapf_log_proposal(x,pr))),1e-10)
    check(paste0("mixture_floor_lower_bound_",delta),max(0,delta-min(pr$floor_probability)),1e-12)
  }
  dd<-iapf_paper_data(2,8,.42,9280002);model<-iapf_linear_model(dd$observations,dd$A)
  kf<-iapf_kalman(model)
  for(kappa in c(0,.5,1)) {
    set.seed(9280003)
    v<-iapf_apf(model,iapf_exact_twists(model),32,kappa)
    check(paste0("exact_twist_Kalman_",kappa),abs(v$log_likelihood-kf$log_likelihood),1e-9)
    check(paste0("terminal_prefix_identity_",kappa),abs(tail(v$prefix,1)-v$log_likelihood),1e-10)
  }
  # Independent constant-twist implementation, matched random stream.
  bootstrap<-function(model,N,kappa) {
    w<-rep(0,N);total<-0;x<-NULL
    for(t in seq_len(model$horizon)) {
      if(t==1) means<-matrix(rep(model$initial_mean,each=N),N) else {
        if(iapf_ess(w)<=kappa*N){total<-total+iapf_logmean(w)
          x<-x[sample.int(N,N,TRUE,exp(w-max(w))),,drop=FALSE];w<-rep(0,N)}
        means<-model$transition_mean(x,t)
      }
      cov<-if(t==1)model$initial_covariance else model$transition_covariance
      x<-means+matrix(rnorm(length(means)),nrow=N)%*%chol(cov)
      w<-w+model$log_observation(x,model$observations[t,],t)
    };total+iapf_logmean(w)
  }
  for(kappa in c(0,.5,1)){
    set.seed(9280004);a<-iapf_apf(model,replicate(8,iapf_constant_twist(),FALSE),32,kappa)
    set.seed(9280004);b<-bootstrap(model,32,kappa)
    check(paste0("constant_BPF_callchain_",kappa),abs(a$log_likelihood-b),1e-10)
  }
  calls<-0;fits<-0
  fakefilter<-function(model,twists,N,kappa,store_clouds) {
    calls<<-calls+1
    list(log_likelihood=0,clouds=list(),resampling_count=0,floor_probability_max=0,
      floor_probability_quantiles=matrix(0,1,5),marker=calls)
  }
  fakefit<-function(model,clouds,N,maxit,floor_tail_power){
    fits<<-fits+1;list(twists=list(),diagnostics=list())}
  z<-iapf_iterate(model,.filter=fakefilter,.fit=fakefit)
  check("controller_l_gt_k_and_fresh_final",abs(calls-8)+abs(fits-6)+abs(z$final$marker-8),0)
  write.csv(data.frame(status="PASS",controls=length(controls),
    conclusion="CHECKED_CONTROLS_ONLY_REPLICATION_INCOMPLETE"),file.path(out,"decision.csv"),row.names=FALSE)
  quit(status=0)
}
stopifnot(phase=="pilot")
rows<-fitrows<-histrows<-prefixrows<-list()
RNGkind("L'Ecuyer-CMRG")
for(dataset in seq_len(2)) {
  data_seed<-9281000+dataset
  data<-iapf_paper_data(5,100,.42,data_seed);model<-iapf_linear_model(data$observations,data$A)
  kf<-iapf_kalman(model);saveRDS(list(data=data,kalman=kf),file.path(out,paste0("data",dataset,".rds")))
  for(rep in seq_len(4)) {
    comparator_seed<-9283000+100*dataset+rep
    methods<-list(BPF=function()iapf_apf(model,replicate(100,iapf_constant_twist(),FALSE),10000,.5,FALSE),
      FA_APF=function()iapf_fully_adapted(model,5000,.5),
      observation_twist=function()iapf_apf(model,iapf_observation_twists(model),1000,.5,FALSE))
    for(name in names(methods)) {
      set.seed(comparator_seed+match(name,names(methods))*10000);began<-proc.time()[3]
      v<-methods[[name]]()
      rows[[length(rows)+1L]]<-data.frame(dataset=dataset,data_seed=data_seed,replicate=rep,method=name,
        status="complete",log_likelihood=v$log_likelihood,log_ratio=v$log_likelihood-kf$log_likelihood,
        seconds=unname(proc.time()[3]-began),final_particles=if(name=="BPF")10000 else if(name=="FA_APF")5000 else 1000,
        adaptation_passes=0,error="")
      prefix<-v$prefix
      prefixrows[[length(prefixrows)+1L]]<-data.frame(dataset=dataset,replicate=rep,method=name,
        time=seq_len(100),log_ratio=prefix-kf$prefix,innovation=kf$innovation)
      write_rows(rows,"runs.csv");write_rows(prefixrows,"prefix.csv")
    }
    for(recipe in c("A_raw","B_scaled")) {
      seed<-9282000+dataset*100+rep*2+as.integer(recipe=="B_scaled")
      set.seed(seed);adapt_state<-.Random.seed;final_state<-parallel::nextRNGStream(adapt_state)
      .Random.seed<-adapt_state;fit_count<-0L;filter_calls<-0L
      myfit<-function(model,clouds,N,maxit,floor_tail_power) {
        fit_count<<-fit_count+1L;twists<-vector("list",model$horizon);ds<-vector("list",model$horizon)
        for(t in rev(seq_len(model$horizon))){
          ly<-iapf_backward_log_target(model,clouds[[t]],t,if(t<model$horizon)twists[[t+1]] else NULL)
          f<-finite_fit(clouds[[t]],ly,N,recipe=="B_scaled")
          twists[[t]]<-f$twist;ds[[t]]<-f$diagnostics
          fitrows[[length(fitrows)+1L]]<<-cbind(data.frame(dataset=dataset,replicate=rep,
            method=recipe,iteration=fit_count-1L,time=t,particles=N),f$diagnostics)
        }
        write_rows(fitrows,"fits.csv")
        list(twists=twists,diagnostics=ds)
      }
      myfilter<-function(model,twists,N,kappa,store_clouds){
        # Controller makes two filters without a fitting call only at final draw.
        is_final<-filter_calls>fit_count
        if(is_final)assign(".Random.seed",final_state,envir=.GlobalEnv)
        filter_calls<<-filter_calls+1L
        v<-iapf_apf(model,twists,N,kappa,store_clouds)
        histrows[[length(histrows)+1L]]<<-data.frame(dataset=dataset,replicate=rep,method=recipe,
          filter_call=filter_calls,final=is_final,seed=seed,particles=N,
          log_likelihood=v$log_likelihood,floor_probability_max=v$floor_probability_max,
          resampling_count=v$resampling_count)
        write_rows(histrows,"histories.csv");v
      }
      began<-proc.time()[3]
      # Per replicate cap is a resource status, never a convergence criterion.
      setTimeLimit(elapsed=100,transient=TRUE)
      v<-tryCatch(iapf_iterate(model,N0=1000,k=5,tau=.5,kappa=.5,max_iterations=20,
        max_particles=16000,.filter=myfilter,.fit=myfit),error=function(e)list(status="error",error=conditionMessage(e)))
      setTimeLimit(cpu=Inf,elapsed=Inf,transient=FALSE)
      done<-identical(v$status,"complete")
      rows[[length(rows)+1L]]<-data.frame(dataset=dataset,data_seed=data_seed,replicate=rep,method=recipe,
        status=v$status,log_likelihood=if(done)v$final$log_likelihood else NA_real_,
        log_ratio=if(done)v$final$log_likelihood-kf$log_likelihood else NA_real_,
        seconds=unname(proc.time()[3]-began),final_particles=if(done)v$final_particles else NA_real_,
        adaptation_passes=fit_count,error=if(is.null(v$error))"" else v$error)
      if(done)prefixrows[[length(prefixrows)+1L]]<-data.frame(dataset=dataset,replicate=rep,
        method=recipe,time=seq_len(100),log_ratio=v$final$prefix-kf$prefix,innovation=kf$innovation)
      write_rows(rows,"runs.csv");write_rows(prefixrows,"prefix.csv");write_rows(fitrows,"fits.csv")
      saveRDS(list(result=v,adaptation_rng=adapt_state,final_rng=final_state,seed=seed,recipe=recipe),
        file.path(out,paste0("dataset",dataset,"_rep",rep,"_",recipe,".rds")))
      cat(dataset,rep,recipe,v$status,unname(proc.time()[3]-began),"seconds\n")
    }
  }
}
rr<-do.call(rbind,rows);selected<-rr$method %in% c("A_raw","B_scaled")
write.csv(data.frame(status="COMPLETE",planned=16,completed=sum(selected & rr$status=="complete"),
  failed_or_incomplete=sum(selected & rr$status!="complete"),
  conclusion="DESCRIPTIVE_FINITE_RECONSTRUCTION_PILOT_ONLY"),file.path(out,"decision.csv"),row.names=FALSE)

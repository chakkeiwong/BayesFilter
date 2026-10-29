# Independent R/FP64 CPU reference for finite BFGS and adaptive controller.
args<-commandArgs(TRUE);out<-args[2];src<-args[3]
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
source(file.path(src,"reference_iapf_paper.R"))
source(file.path(src,"reference_iapf_scale_aware_fit.R"))
nodes<-parse(file.path(src,"diagnose_iapf_finite_repair_reference.R"))
for(node in nodes)if(is.call(node)&&identical(node[[1]],as.name("<-"))&&
  is.symbol(node[[2]]) && as.character(node[[2]]) %in% c("finite_problem","finite_fit"))eval(node)
options(digits=17)
roundtrip_checks<-0L
write_matrix<-function(x,path) {
  x<-as.matrix(x)
  encoded<-matrix(sprintf("%.17g",as.numeric(x)),nrow=nrow(x),ncol=ncol(x))
  write.table(encoded,path,sep=",",quote=FALSE,row.names=FALSE,col.names=FALSE)
  restored<-as.matrix(read.csv(path,header=FALSE))
  stopifnot(identical(as.numeric(restored),as.numeric(x)))
  roundtrip_checks<<-roundtrip_checks+1L
}
fit_one<-function(x,ly,scaled) {
  init<-iapf_fit_initial(x,ly,FALSE)
  problem<-finite_problem(x,ly,init$parameters,scaled)
  evaluated<-list();accepted<-list()
  fn<-function(z){ev<-problem$evaluate(z);evaluated[[length(evaluated)+1L]]<<-c(z,ev$loss);ev$loss}
  gr<-function(z){ev<-problem$evaluate(z);accepted[[length(accepted)+1L]]<<-c(z,ev$loss,ev$gradient);ev$gradient}
  opt<-optim(problem$z0,fn,gr,method="BFGS",control=list(maxit=100,reltol=sqrt(.Machine$double.eps)))
  ev<-problem$evaluate(opt$par)
  list(opt=opt,ev=ev,initial=init,problem=problem,evaluated=evaluated,accepted=accepted)
}
rows<-list()
export_fit<-function(name,x,ly,scaled) {
  folder<-file.path(out,name);dir.create(folder)
  f<-fit_one(x,ly,scaled);d<-ncol(x)
  probe<-seq(-.05,.05,length.out=2*d+1);ev<-f$problem$evaluate(probe)
  values<-list(points=x,targets=ly,initial=f$initial$parameters,probe=probe,
    probe_gradient=ev$gradient,theta=f$ev$theta,z=f$opt$par,gradient=f$ev$gradient,
    evaluations=do.call(rbind,f$evaluated),accepted=do.call(rbind,f$accepted))
  for(key in names(values))write_matrix(values[[key]],file.path(folder,paste0(key,".csv")))
  rows[[length(rows)+1L]]<<-data.frame(name=name,d=d,N=nrow(x),scaled=scaled,
    initializer=f$initial$method,probe_loss=ev$loss,loss=f$ev$loss,
    amplitude=f$ev$a,code=f$opt$convergence,function_calls=f$opt$counts[1],
    gradient_calls=f$opt$counts[2],valid=f$ev$a>0)
  write.csv(do.call(rbind,rows),file.path(out,"fit-index.csv"),row.names=FALSE)
}
set.seed(9286001)
x<-matrix(rnorm(128*2),128,2)
ly<-iapf_log_normal(x,c(.4,-.3),diag(c(.8,1.3)))
for(scaled in c(FALSE,TRUE))export_fit(paste0("gaussian_",scaled),x,ly,scaled)
ly<-iapf_logadd(iapf_log_normal(x,c(-1.5,.5),diag(c(.7,1.1))),
               iapf_log_normal(x,c(1.4,-.7),diag(c(.9,.6))))
for(scaled in c(FALSE,TRUE))export_fit(paste0("mixture_",scaled),x,ly,scaled)
ly<-.1*x[,1]^2-.2*x[,2]^2
for(scaled in c(FALSE,TRUE))export_fit(paste0("nonconcave_",scaled),x,ly,scaled)
data<-iapf_paper_data(5,8,.42,9286002);model<-iapf_linear_model(data$observations,data$A)
set.seed(9286003);cloud<-iapf_apf(model,replicate(8,iapf_constant_twist(),FALSE),256,.5,TRUE)$clouds[[5]]
ly<-model$log_observation(cloud,data$observations[5,],5)
for(scaled in c(FALSE,TRUE))export_fit(paste0("paper_cloud_",scaled),cloud,ly,scaled)

# Actual adaptive reference with common random arrays, saved at every invocation.
paper<-args[1]=="paper"
T<-if(paper)100 else 8;d<-if(paper)5 else 2
N0<-if(paper)1000 else 64;maxN<-if(paper)8000 else 512
data_seed<-if(paper)9286300 else 9286100;entropy_seed<-if(paper)9286400 else 9286200
data<-iapf_paper_data(d,T,.42,data_seed);model<-iapf_linear_model(data$observations,data$A)
call_id<-0L;calls<-list();backward_calls<-0L
controller_dir<-file.path(out,"controller");dir.create(controller_dir)
write.csv(data.frame(d=d,T=T,N0=N0,max_particles=maxN,max_iterations=12,
  data_seed=data_seed,entropy_seed_base=entropy_seed),
  file.path(controller_dir,"scope.csv"),row.names=FALSE)
for(key in c("A","C","R"))write_matrix(model[[key]],file.path(controller_dir,paste0(key,".csv")))
write_matrix(model$observations,file.path(controller_dir,"observations.csv"))
myfilter<-function(model,twists,N,kappa,store_clouds) {
  call_id<<-call_id+1L;set.seed(entropy_seed+call_id)
  folder<-file.path(controller_dir,paste0("call",call_id));dir.create(folder)
  noise<-lapply(seq_len(T),function(t)matrix(rnorm(N*d),N,d))
  mixture<-matrix(runif(T*N),T,N);uniforms<-matrix(runif(T*N),T,N);tt<-0L
  iapf_draw_proposal<<-function(proposal) {
    tt<<-tt+1L;z<-noise[[tt]]
    base<-proposal$original_means+z%*%iapf_chol(proposal$original_covariance)
    if(proposal$constant)return(base)
    adapted<-proposal$means+z%*%iapf_chol(proposal$covariance)
    choose<-mixture[tt,]<proposal$floor_probability
    adapted[choose,]<-base[choose,,drop=FALSE];adapted
  }
  assign("sample.int",function(n,size,replace,prob) {
    stopifnot(n==N,size==N,replace)
    cdf<-cumsum(prob/sum(prob))
    vapply(uniforms[tt+1L,],function(u)as.integer(min(N,sum(cdf<=u)+1L)),integer(1))
  },envir=.GlobalEnv)
  ref<-iapf_apf(model,twists,N,kappa,TRUE)
  values<-list(noise=do.call(rbind,noise),mixture=mixture,ancestor_uniforms=uniforms,
    clouds=do.call(rbind,ref$clouds),prefix=ref$prefix)
  for(key in names(values))write_matrix(values[[key]],file.path(folder,paste0(key,".csv")))
  calls[[length(calls)+1L]]<<-data.frame(call=call_id,N=N,log_likelihood=ref$log_likelihood)
  write.csv(do.call(rbind,calls),file.path(controller_dir,"calls.csv"),row.names=FALSE)
  ref
}
myfit<-function(model,clouds,N,maxit,floor_tail_power) {
  backward_calls<<-backward_calls+1L
  folder<-file.path(controller_dir,paste0("backward",backward_calls));dir.create(folder)
  twists<-vector("list",T);diagnostics<-list()
  for(t in T:1) {
    ly<-iapf_backward_log_target(model,clouds[[t]],t,if(t<T)twists[[t+1]] else NULL)
    f<-fit_one(clouds[[t]],ly,FALSE)
    saveRDS(list(points=clouds[[t]],targets=ly,initial=f$initial,opt=f$opt,
      ev=f$ev,evaluated=f$evaluated,accepted=f$accepted),file.path(folder,paste0("fit",t,".rds")))
    for(key in c("theta","gradient"))write_matrix(f$ev[[key]],file.path(folder,paste0("fit",t,"_",key,".csv")))
    write_matrix(ly,file.path(folder,paste0("fit",t,"_targets.csv")))
    write_matrix(f$initial$parameters,file.path(folder,paste0("fit",t,"_initial.csv")))
    write_matrix(do.call(rbind,f$accepted),file.path(folder,paste0("fit",t,"_accepted.csv")))
    stopifnot(f$ev$a>0)
    twists[[t]]<-sa_twist(f$ev$theta,N,1,model$transition_covariance)
    diagnostics[[t]]<-data.frame(time=t,code=f$opt$convergence,amplitude=f$ev$a,loss=f$ev$loss)
  }
  write_matrix(do.call(rbind,lapply(twists,function(z)z$mean)),file.path(folder,"centers.csv"))
  write_matrix(do.call(rbind,lapply(twists,function(z)z$covariance)),file.path(folder,"covariances.csv"))
  write_matrix(vapply(twists,function(z)z$log_floor,numeric(1)),file.path(folder,"floors.csv"))
  write.csv(do.call(rbind,diagnostics),file.path(folder,"fits.csv"),row.names=FALSE)
  list(twists=twists,diagnostics=diagnostics)
}
v<-iapf_iterate(model,N0=N0,max_iterations=12,max_particles=maxN,
               .filter=myfilter,.fit=myfit)
saveRDS(v,file.path(controller_dir,"result.rds"))
write.csv(data.frame(status=v$status,filter_calls=call_id,backward_calls=backward_calls,
  stop_iteration=if(is.null(v$stop_iteration))NA else v$stop_iteration,
  final_particles=if(is.null(v$final_particles))NA else v$final_particles,
  final_log_likelihood=if(is.null(v$final))NA else v$final$log_likelihood,
  kalman=iapf_kalman(model)$log_likelihood),file.path(controller_dir,"decision.csv"),row.names=FALSE)
stopifnot(v$status=="complete")
write.csv(data.frame(status="PASS",fit_cases=length(rows),controller=v$status,
  scope="INDEPENDENT_R_FINITE_RECONSTRUCTION",exact_matrix_roundtrips=roundtrip_checks),file.path(out,"decision.csv"),row.names=FALSE)
cat("Exported",length(rows),"fits and",call_id,"adaptive filter calls\n")

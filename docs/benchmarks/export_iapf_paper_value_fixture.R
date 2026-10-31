# Independent CPU reference fixtures for Algorithm 5, with shared random inputs.
args<-commandArgs(TRUE);out<-args[2];src<-args[3]
source(file.path(src,"reference_iapf_paper.R"))
options(digits=17)
campaign<-dirname(out)
pilot<-file.path(campaign,"attempt02_iapf_pilot")
index<-list()
write_matrix<-function(value,path)write.table(value,path,sep=",",row.names=FALSE,col.names=FALSE)
export_case<-function(name,model,twists,N,kappa,seed,exact=FALSE) {
  set.seed(seed);T<-model$horizon;d<-model$dimension;folder<-file.path(out,name);dir.create(folder)
  noise<-lapply(seq_len(T),function(t)matrix(rnorm(N*d),N,d))
  mixture<-matrix(runif(T*N),T,N);ancestor_uniforms<-matrix(runif(T*N),T,N)
  draw_time<-0L
  # Keep iapf_apf itself unchanged; inject the same variates supplied to TF.
  iapf_draw_proposal<<-function(proposal) {
    draw_time<<-draw_time+1L
    z<-noise[[draw_time]]
    base<-proposal$original_means+z%*%iapf_chol(proposal$original_covariance)
    if(proposal$constant)return(base)
    adapted<-proposal$means+z%*%iapf_chol(proposal$covariance)
    choose<-mixture[draw_time,]<proposal$floor_probability
    adapted[choose,]<-base[choose,,drop=FALSE]
    adapted
  }
  assign("sample.int",function(n,size,replace,prob) {
    stopifnot(n==N,size==N,replace)
    cdf<-cumsum(prob/sum(prob));u<-ancestor_uniforms[draw_time+1L,]
    vapply(u,function(v)as.integer(min(N,sum(cdf<=v)+1L)),integer(1))
  },envir=.GlobalEnv)
  ref<-iapf_apf(model,twists,N,kappa,TRUE)
  if(kappa==1)stopifnot(ref$resampling_count==T-1)
  if(kappa==0)stopifnot(ref$resampling_count==0)
  matrices<-list(A=model$A,C=model$C,initial_mean=model$initial_mean,
    P=model$initial_covariance,Q=model$transition_covariance,R=model$R,
    observations=model$observations,
    centers=do.call(rbind,lapply(twists,function(t)if(t$constant)rep(0,d) else t$mean)),
    covariances=do.call(rbind,lapply(twists,function(t)if(t$constant)diag(d) else t$covariance)),
    floors=vapply(twists,function(t)if(t$constant)0 else t$log_floor,numeric(1)),
    constants=as.integer(vapply(twists,function(t)t$constant,logical(1))),
    noise=do.call(rbind,noise),mixture=mixture,ancestor_uniforms=ancestor_uniforms,
    clouds=do.call(rbind,ref$clouds),weights=do.call(rbind,ref$log_weights),
    ancestors=do.call(rbind,ref$ancestors)-1L,ess=ref$ess,
    resampled=as.integer(ref$resampled),prefix=ref$prefix)
  for(key in names(matrices))write_matrix(matrices[[key]],file.path(folder,paste0(key,".csv")))
  oracle<-iapf_kalman(model)$log_likelihood
  if(exact)stopifnot(abs(ref$log_likelihood-oracle)<1e-8*(1+abs(oracle)))
  index[[length(index)+1L]]<<-data.frame(name=name,d=d,o=ncol(model$observations),N=N,T=T,
    kappa=kappa,seed=seed,log_likelihood=ref$log_likelihood,kalman=oracle,exact=exact,
    resampling_count=ref$resampling_count)
  write.csv(do.call(rbind,index),file.path(out,"index.csv"),row.names=FALSE)
}
data<-iapf_paper_data(2,6,.42,9284001)
data$observations[4,]<-data$observations[4,]+3
model<-iapf_linear_model(data$observations,data$A)
families<-list(constant=replicate(6,iapf_constant_twist(),FALSE),
  floor=lapply(iapf_observation_twists(model),function(t){t$log_floor<-log(.001);t}),
  exact=iapf_exact_twists(model))
i<-0
for(family in names(families))for(kappa in c(0,.5,1)){
  i<-i+1;export_case(paste0(family,"_",i),model,families[[family]],32,kappa,9284100+i,family=="exact")
}
saved<-readRDS(file.path(pilot,"dataset1_rep1_A_raw.rds"))$result
saveddata<-readRDS(file.path(pilot,"data1.rds"))$data
stopifnot(saved$status=="complete")
export_case("frozen_pilot",iapf_linear_model(saveddata$observations,saveddata$A),
  saved$twists,saved$final_particles,.5,9284201)
write.csv(data.frame(status="PASS",cases=length(index),
  conclusion="PAIRED_VALUE_FIXTURES_ONLY"),file.path(out,"decision.csv"),row.names=FALSE)
cat("Exported",length(index),"paired cases\n")

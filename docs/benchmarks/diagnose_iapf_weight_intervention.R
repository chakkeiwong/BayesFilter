# Independent CPU R intervention: unchanged safeguarded fitter, exponent 0/1.
# The zero-exponent arm is a log-regression extension, not paper Equation 15.
args<-commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==4,Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(args[2]))
stage<-args[1];out<-args[2];dir.create(out,recursive=TRUE)
datasets<-as.integer(args[3]);reps<-as.integer(args[4])
stopifnot(stage %in% c("intervention","confirmation"),datasets>=1,reps>=2)
source("docs/benchmarks/reference_iapf_paper.R")
source("docs/benchmarks/reference_iapf_author_choices.R")
source("docs/benchmarks/reference_iapf_constrained_diagnostic.R")
source("docs/benchmarks/reference_iapf_guide_diagnostics.R")
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
write_rows<-function(rows,file)if(length(rows))write.csv(do.call(rbind,rows),file,row.names=FALSE)
getvalue<-function(z,key,default=NA_real_)if(is.null(z[[key]]))default else z[[key]]
run_zero<-function(model) {
  fitter<-function(model,clouds,N,maxit,floor_tail_power) {
    twists<-diagnostics<-vector("list",model$horizon)
    for(t in rev(seq_len(model$horizon))) {
      target<-iapf_backward_log_target(model,clouds[[t]],t,
        if(t<model$horizon)twists[[t+1]] else NULL)
      fitted<-tryCatch(iapf_constrained_diagnostic_fit(clouds[[t]],target,N,exponent=0),
        iapf_fit_error=function(e){e$backward_time<-t;stop(e)})
      twists[[t]]<-fitted$twist;diagnostics[[t]]<-fitted$diagnostics
    }
    list(twists=twists,diagnostics=diagnostics)
  }
  iapf_iterate(model,N0=1000,k=5,tau=.5,kappa=.5,max_iterations=12,max_particles=4000,
    fit_maxit=1000,floor_tail_power=8,doubling_mode="after_k",stopping_window=6,
    cv_sd_mode="sample",.fit=fitter)
}
cell<-function(item) {
  d<-item$dimension;j<-item$data_index;r<-item$replication
  cellout<-file.path(out,paste0("d",d,"-j",j,"-r",r));dir.create(cellout)
  offset<-if(stage=="confirmation")1000000L else 0L
  data_seed<-93100000L+offset+10000L*j+d
  learning_seed<-93200000L+offset+10000L*j+101L*r+d
  probe_seed<-93300000L+offset+10000L*j+101L*r+d
  dat<-iapf_paper_data(d,100,seed=data_seed)
  model<-iapf_linear_model(dat$observations,dat$A);truth<-iapf_kalman(model)
  full<-iapf_exact_twists(model);mom<-iapf_diag_moments(model)
  saveRDS(list(data=dat,model=model,kalman=truth),file.path(cellout,"data.rds"))
  methods<-c("qr","ridge_bounded0","ridge_bounded1")
  shift<-(j+r+match(d,c(5,10,20,40,80)))%%3
  methods<-methods[((0:2+shift)%%3)+1]
  learning<-probes<-fits<-gaps<-replays<-list();finalguides<-list()
  for(method in methods) {
    set.seed(learning_seed);start<-proc.time()[[3]]
    answer<-tryCatch(switch(method,qr=iapf_choice_run(model,arm="qr",max_iterations=12,max_particles=4000),
      ridge_bounded0=run_zero(model),ridge_bounded1=iapf_constrained_diagnostic_run(model)),error=function(e)e)
    seconds<-proc.time()[[3]]-start;failed<-inherits(answer,"error")
    status<-if(failed)"candidate_failed" else answer$status
    valid<-identical(status,"complete")
    delta<-if(valid)answer$final$log_likelihood-truth$log_likelihood else NA_real_
    learning[[length(learning)+1L]]<-data.frame(stage=stage,dimension=d,data_index=j,replication=r,
      data_seed=data_seed,learning_seed=learning_seed,method=method,status=status,
      seconds=seconds,log_error=delta,ratio=exp(delta),stop_iteration=if(valid)answer$stop_iteration else NA,
      final_particles=if(valid)answer$final_particles else NA,
      message=if(failed)conditionMessage(answer)else "")
    if(valid) {
      finalguides[[method]]<-answer$twists
      # Same final-filter seed across methods; sampling paths may differ.
      set.seed(probe_seed);start<-proc.time()[[3]]
      probe<-iapf_apf(model,answer$twists,1000,store_clouds=FALSE)
      ps<-proc.time()[[3]]-start;de<-probe$log_likelihood-truth$log_likelihood
      probes[[length(probes)+1L]]<-data.frame(stage=stage,dimension=d,data_index=j,replication=r,
        method=method,seed=probe_seed,particles=1000,log_error=de,ratio=exp(de),filter_seconds=ps,
        learning_seconds=seconds,total_seconds=seconds+ps)
      for(it in seq_along(answer$fits))for(t in seq_along(answer$fits[[it]])) {
        z<-answer$fits[[it]][[t]]
        fits[[length(fits)+1L]]<-data.frame(dimension=d,data_index=j,replication=r,method=method,
          iteration=it-1,time=t,weight_ess=getvalue(z,"weight_ess"),lambda=getvalue(z,"lambda",0),
          kkt=getvalue(z,"kkt_residual"),active_constraints=getvalue(z,"active_constraints",0))
      }
      for(t in c(1,25,50,75,99,100)) {
        q<-iapf_diag_component(mom$prediction_mean[[t]],mom$prediction_covariance[[t]],answer$twists[[t]])
        gaps[[length(gaps)+1L]]<-data.frame(dimension=d,data_index=j,replication=r,method=method,time=t,
          kl=iapf_diag_gaussian_kl(mom$smoothing_mean[[t]],mom$smoothing_covariance[[t]],q$mean,q$covariance))
      }
      # Preserve the executed guides and fitting diagnostics without redundant final clouds.
      answer$final<-list(log_likelihood=answer$final$log_likelihood,
        floor_probability_max=answer$final$floor_probability_max,resampling_count=answer$final$resampling_count)
    }
    saveRDS(answer,file.path(cellout,paste0(method,"-learning.rds")))
    write_rows(learning,file.path(cellout,"learning.csv"));write_rows(probes,file.path(cellout,"probes.csv"))
  }
  for(method in c("constant","observation","moment_diagonal","precision_diagonal","exact_full")) {
    guides<-switch(method,constant=replicate(100,iapf_constant_twist(),simplify=FALSE),
      observation=iapf_observation_twists(model),
      moment_diagonal=lapply(full,function(g)iapf_gaussian_twist(g$mean,diag(diag(g$covariance)))),
      precision_diagonal=lapply(full,function(g)iapf_gaussian_twist(g$mean,diag(1/diag(solve(g$covariance))))),
      exact_full=full)
    set.seed(probe_seed);start<-proc.time()[[3]];probe<-iapf_apf(model,guides,1000,store_clouds=FALSE)
    seconds<-proc.time()[[3]]-start;de<-probe$log_likelihood-truth$log_likelihood
    if(method=="exact_full")stopifnot(is.finite(de),abs(de)<=1e-8)
    probes[[length(probes)+1L]]<-data.frame(stage=stage,dimension=d,data_index=j,replication=r,
      method=method,seed=probe_seed,particles=1000,log_error=de,ratio=exp(de),filter_seconds=seconds,
      learning_seconds=NA_real_,total_seconds=NA_real_)
  }
  if(stage=="intervention" && d %in% c(20,80) && j==1 && r==1) {
    # Actual bootstrap learning distribution; heldout points have the exact
    # smoothed law. Paired next-guide interventions distinguish recursion.
    set.seed(learning_seed);cold<-iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),1000)
    for(t in c(1,50,99,100))for(next_method in c("exact_full","qr","ridge_bounded1")) {
      nxt<-if(next_method=="exact_full")full else finalguides[[next_method]]
      if(is.null(nxt))next
      x<-cold$clouds[[t]]
      target<-iapf_backward_log_target(model,x,t,if(t<100)nxt[[t+1]] else NULL)
      set.seed(93500000L+100L*t+d)
      held<-iapf_draw_normal(matrix(rep(mom$smoothing_mean[[t]],each=2048),2048,d),mom$smoothing_covariance[[t]])
      heldtarget<-iapf_backward_log_target(model,held,t,if(t<100)nxt[[t+1]] else NULL)
      for(exponent in c(0,1)) {
        fit<-tryCatch(iapf_constrained_diagnostic_fit(x,target,1000,exponent=exponent),error=function(e)e)
        valid<-!inherits(fit,"error")
        replays[[length(replays)+1L]]<-data.frame(dimension=d,time=t,next_method=next_method,
          exponent=exponent,valid=valid,
          train_shape=if(valid)iapf_diag_shape(iapf_log_twist(x,fit$twist)-target)else NA_real_,
          heldout_shape=if(valid)iapf_diag_shape(iapf_log_twist(held,fit$twist)-heldtarget)else NA_real_,
          weight_ess=if(valid)fit$diagnostics$weight_ess else NA_real_,
          lambda=if(valid)fit$diagnostics$lambda else NA_real_,
          message=if(valid)""else conditionMessage(fit))
        saveRDS(list(points=x,target=target,heldout=held,heldout_target=heldtarget,fit=fit),
          file.path(cellout,paste0("replay-t",t,"-",next_method,"-w",exponent,".rds")))
      }
    }
  }
  write_rows(probes,file.path(cellout,"probes.csv"));write_rows(fits,file.path(cellout,"fits.csv"))
  write_rows(gaps,file.path(cellout,"guide-gaps.csv"));write_rows(replays,file.path(cellout,"replays.csv"))
  data.frame(dimension=d,data_index=j,replication=r,status="complete",path=cellout)
}
grid<-expand.grid(dimension=c(80,40,20,10,5),data_index=seq_len(datasets),replication=seq_len(reps))
workers<-as.integer(Sys.getenv("IAPF_DIAGNOSTIC_WORKERS","1"))
results<-parallel::mclapply(seq_len(nrow(grid)),function(i) {
  start<-proc.time()[[3]]
  ans<-tryCatch(cell(grid[i,]),error=function(e)data.frame(dimension=grid$dimension[i],
    data_index=grid$data_index[i],replication=grid$replication[i],status="harness_failed",path=conditionMessage(e)))
  ans$worker_seconds<-proc.time()[[3]]-start
  cat("cell",i,"status",ans$status,"seconds",ans$worker_seconds,"\n");flush.console()
  ans
},mc.cores=workers,mc.preschedule=FALSE,mc.set.seed=FALSE)
write_rows(results,file.path(out,"cells.csv"))
for(name in c("learning.csv","probes.csv","fits.csv","guide-gaps.csv","replays.csv")) {
  paths<-list.files(out,pattern=paste0("^",name,"$"),recursive=TRUE,full.names=TRUE)
  if(length(paths))write.csv(do.call(rbind,lapply(paths,read.csv)),file.path(out,name),row.names=FALSE)
}
stopifnot(all(vapply(results,function(z)z$status=="complete",TRUE)))
cat("complete",nrow(grid),"cells\n")

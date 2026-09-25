# CPU-only independent reference diagnostic. Never selects a runtime default.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)>=2,Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(args[2]))
stage <- args[1]; out <- args[2]; dir.create(out,recursive=TRUE)
source("docs/benchmarks/reference_iapf_paper.R")
source("docs/benchmarks/reference_iapf_author_choices.R")
source("docs/benchmarks/reference_iapf_constrained_diagnostic.R")
source("docs/benchmarks/reference_iapf_guide_diagnostics.R")
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
checks <- list()
check <- function(name,error,tolerance=1e-8) {
  ok <- is.finite(error) && error<=tolerance
  checks[[length(checks)+1L]] <<- data.frame(check=name,error=error,tolerance=tolerance,pass=ok)
  if(!ok) {
    write.csv(do.call(rbind,checks),file.path(out,"checks.csv"),row.names=FALSE)
    stop(paste(name,error,"exceeds",tolerance))
  }
}
save_rows <- function(rows,name)write.csv(do.call(rbind,rows),file.path(out,name),row.names=FALSE)
if(stage=="checks") {
  dat <- iapf_paper_data(5,8,seed=93000001)
  model <- iapf_linear_model(dat$observations,dat$A)
  full <- iapf_exact_twists(model); mom <- iapf_diag_moments(model)
  for(t in 1:8) {
    a<-mom$prediction_mean[[t]];P<-mom$prediction_covariance[[t]]
    b<-mom$smoothing_mean[[t]];S<-mom$smoothing_covariance[[t]]
    q<-iapf_diag_component(a,P,full[[t]])
    check(paste0("RTS_mean_",t),max(abs(q$mean-b)))
    check(paste0("RTS_covariance_",t),max(abs(q$covariance-S)))
    check(paste0("oracle_KL_",t),abs(iapf_diag_gaussian_kl(b,S,q$mean,q$covariance)))
    j<-diag(solve(full[[t]]$covariance))
    opt<-iapf_diag_projection(a,P,b,S,j)
    check(paste0("convex_KKT_",t),opt$kkt,1e-7)
    check(paste0("convex_feasible_comparison_",t),
      opt$kl-iapf_diag_gaussian_kl(b,S,b,solve(solve(P)+diag(j))),1e-9)
    fn<-function(v).5*(sum((solve(P)+diag(v))*t(S))-
      iapf_diag_logdet(solve(P)+diag(v)))
    gr<-.5*(diag(S)-diag(solve(solve(P)+diag(j))))
    fd<-vapply(1:5,function(k){u<-rep(0,5);u[k]<-1e-5;(fn(j+u)-fn(j-u))/2e-5},0.)
    check(paste0("projection_gradient_",t),max(abs(gr-fd)),1e-8)
    set.seed(93000002+t)
    x<-iapf_draw_normal(matrix(rep(b,each=1024),1024,5),S)
    actual<-iapf_diag_mixture(x,a,P,b,S,full[[t]])
    check(paste0("oracle_mixture_KL_",t),abs(actual$kl),1e-8)
  }
  set.seed(93000020)
  apf<-iapf_apf(model,full,64)
  check("actual_shared_APF_Kalman",abs(apf$log_likelihood-iapf_kalman(model)$log_likelihood))
  # At exponent zero, the safeguarded regression preserves a healthy QR fit.
  set.seed(93000021);x<-matrix(rnorm(1500),300,5)
  y<-iapf_log_normal(x,1:5/5,diag(seq(.6,1,length.out=5)))
  qr<-iapf_choice_fit(x,y,arm="qr")$twist
  safe<-iapf_constrained_diagnostic_fit(x,y,exponent=0)$twist
  check("unweighted_guard_mean_no_fire",max(abs(qr$mean-safe$mean)))
  check("unweighted_guard_covariance_no_fire",max(abs(qr$covariance-safe$covariance)))
  check("unweighted_guard_floor_no_fire",abs(qr$log_floor-safe$log_floor))
  save_rows(checks,"checks.csv")
  cat("checks",length(checks),"passed\n")
} else if(stage=="saved") {
  stopifnot(length(args)==3)
  old<-args[3]; index<-read.csv(file.path(old,"cells-completion.csv"))
  stopifnot(nrow(index)==80,all(index$status=="complete"))
  cache<-new.env();projcache<-new.env();rows<-localrows<-oraclerows<-list()
  chosen<-c(1,25,50,75,99,100)
  for(cell in seq_len(nrow(index))) {
    item<-index[cell,];d<-item$dimension;j<-item$data_index;r<-item$replication
    dir<-file.path(old,item$attempt,"results");key<-paste(d,j,sep="_")
    if(!exists(key,cache,inherits=FALSE)) {
      data<-readRDS(file.path(dir,"data.rds"));model<-data$model
      full<-iapf_exact_twists(model);mom<-iapf_diag_moments(model)
      projections<-vector("list",100);draws<-vector("list",100)
      for(t in 1:100) {
        a<-mom$prediction_mean[[t]];P<-mom$prediction_covariance[[t]]
        b<-mom$smoothing_mean[[t]];S<-mom$smoothing_covariance[[t]]
        q<-iapf_diag_component(a,P,full[[t]])
        check(paste0(key,"_RTS_",t),max(abs(c(q$mean-b,q$covariance-S))))
        pkey<-paste(d,t,sep="_")
        if(!exists(pkey,projcache,inherits=FALSE)) {
          assign(pkey,iapf_diag_projection(a,P,b,S,diag(solve(full[[t]]$covariance))),projcache)
        }
        opt<-get(pkey,projcache);opt$mean<-b;projections[[t]]<-opt
        if(t %in% chosen) {
          set.seed(93010000L+10000L*j+100L*t+d)
          draws[[t]]<-iapf_draw_normal(matrix(rep(b,each=1024),1024,d),S)
          controls<-list(constant=iapf_constant_twist(),observation=iapf_observation_twists(model)[[t]],
            moment_diagonal=iapf_gaussian_twist(full[[t]]$mean,diag(diag(full[[t]]$covariance))),
            precision_diagonal=iapf_gaussian_twist(full[[t]]$mean,diag(1/diag(solve(full[[t]]$covariance)))),
            exact_full=full[[t]])
          for(name in names(controls)) {
            cc<-iapf_diag_component(a,P,controls[[name]])
            oraclerows[[length(oraclerows)+1L]]<-data.frame(dimension=d,data_index=j,time=t,
              method=name,kl=iapf_diag_gaussian_kl(b,S,cc$mean,cc$covariance),
              diagonal_optimum_kl=opt$kl,projection_kkt=opt$kkt)
          }
        }
      }
      assign(key,list(model=model,full=full,mom=mom,projections=projections,draws=draws),cache)
    }
    z<-get(key,cache);model<-z$model;full<-z$full;mom<-z$mom
    for(method in c("qr","ridge_bounded1")) {
      fit<-readRDS(file.path(dir,paste0(method,"-learning.rds")))
      stopifnot(identical(fit$status,"complete"),length(fit$twists)==100)
      for(t in 1:100) {
        a<-mom$prediction_mean[[t]];P<-mom$prediction_covariance[[t]]
        b<-mom$smoothing_mean[[t]];S<-mom$smoothing_covariance[[t]]
        guide<-fit$twists[[t]];q<-iapf_diag_component(a,P,guide)
        kl<-iapf_diag_gaussian_kl(b,S,q$mean,q$covariance)
        opt<-z$projections[[t]]
        check(paste0(cell,"_",method,"_diagonal_lower_bound_",t),opt$kl-kl,1e-8)
        meanpart<-.5*sum((q$mean-b)*as.vector(solve(q$covariance,q$mean-b)))
        rows[[length(rows)+1L]]<-data.frame(dimension=d,data_index=j,replication=r,
          method=method,time=t,kl=kl,mean_kl=meanpart,covariance_kl=kl-meanpart,
          diagonal_optimum_kl=opt$kl,excess_kl=kl-opt$kl,
          floor_probability=exp(q$log_floor_probability))
        if(t %in% chosen) {
          x<-z$draws[[t]]
          mix<-iapf_diag_mixture(x,a,P,b,S,guide)
          lt<-iapf_log_twist(x,guide);le<-iapf_log_twist(x,full[[t]])
          target<-iapf_backward_log_target(model,x,t,if(t<100)fit$twists[[t+1]] else NULL)
          local<-lt-target;inherited<-target-le;total<-lt-le
          vlocal<-mean((local-mean(local))^2);vinherit<-mean((inherited-mean(inherited))^2)
          cross<-2*mean((local-mean(local))*(inherited-mean(inherited)))
          vtotal<-mean((total-mean(total))^2)
          check(paste0(cell,"_",method,"_decomposition_",t),abs(vtotal-vlocal-vinherit-cross),1e-8)
          localrows[[length(localrows)+1L]]<-data.frame(dimension=d,data_index=j,replication=r,
            method=method,time=t,mixture_kl=mix$kl,mixture_mcse=mix$mcse,
            floor_kl_change=mix$floor_kl_change,floor_change_mcse=mix$floor_change_mcse,
            floor_probability=mix$floor_probability,local_log_shape_rmse=sqrt(vlocal),
            inherited_log_shape_rmse=sqrt(vinherit),total_log_shape_rmse=sqrt(vtotal),
            twice_covariance=cross)
        }
      }
    }
    if(cell%%10==0)cat("saved cells",cell,"of",nrow(index),"\n")
  }
  save_rows(rows,"gaussian-gap.csv");save_rows(localrows,"recursive-gap.csv")
  save_rows(oraclerows,"oracle-gap.csv");save_rows(checks,"checks.csv")
  cat("complete",length(rows),"guide rows",length(localrows),"recursive rows\n")
} else stop("unknown stage")

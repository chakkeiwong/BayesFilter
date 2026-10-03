# CPU-only diagnostic reporting of an already frozen confirmation campaign.
args<-commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3,args[1]=="report",Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(args[2]))
input<-args[3];out<-args[2];dir.create(out,recursive=TRUE)
source("docs/benchmarks/reference_iapf_paper.R")
read<-function(name)read.csv(file.path(input,paste0(name,".csv")))
write<-function(x,name)write.csv(x,file.path(out,paste0(name,".csv")),row.names=FALSE)
cells<-read("cells");learn<-read("learning");probes<-read("probes");fits<-read("fits")
gaps<-read("guide-gaps")
key<-c("dimension","data_index","replication")
methods<-c("qr","ridge_bounded0","ridge_bounded1")
controls<-c("constant","observation","moment_diagonal","precision_diagonal")
checks<-data.frame(check=character(),passed=logical(),value=double())
check<-function(name,ok,value=NA_real_) {
  checks<<-rbind(checks,data.frame(check=name,passed=isTRUE(ok),value=value))
}
check("160 complete cells",nrow(cells)==160 && all(cells$status=="complete"),nrow(cells))
completed<-learn$status=="complete"
check("480 recorded learner outcomes",nrow(learn)==480 &&
  all(learn$status %in% c("complete","iteration_cap","particle_cap","candidate_failed")) &&
  all(is.finite(learn$log_error[completed])) && all(is.na(learn$log_error[!completed])),nrow(learn))
check("fresh probes match complete learners plus 800 controls",nrow(probes)==800+sum(completed) && all(is.finite(probes$log_error)) &&
  all(is.finite(probes$ratio)),nrow(probes))
check("unique learner keys",!anyDuplicated(learn[c(key,"method")]))
check("unique probe keys",!anyDuplicated(probes[c(key,"method")]))
check("all dimensions datasets seeds present",identical(sort(unique(learn$dimension)),c(5L,10L,20L,40L,80L)) &&
  all(table(learn$dimension,learn$data_index,learn$method)==8))
oracle<-max(abs(probes$log_error[probes$method=="exact_full"]))
check("exact full guide agrees with Kalman",is.finite(oracle) && oracle<=1e-8,oracle)
pair<-function(df,first,second) {
  a<-df[df$method==first,];b<-df[df$method==second,]
  merge(a,b,by=key,suffixes=c(".first",".second"),all=TRUE)
}
parity<-pair(learn,"qr","ridge_bounded0")
parity_error<-max(abs(parity$log_error.first-parity$log_error.second))
check("guarded zero and QR terminal parity",nrow(parity)==160 && is.finite(parity_error) &&
  parity_error<=1e-8,parity_error)
paired<-pair(learn,"ridge_bounded1","ridge_bounded0")
check("primary pairing uses identical seeds and data",nrow(paired)==160 &&
  all(paired$data_seed.first==paired$data_seed.second) &&
  all(paired$learning_seed.first==paired$learning_seed.second))
write(checks,"checks")
stopifnot(all(checks$passed))
write(learn[!completed,],"candidate-failures")
histories<-list()
for(i in which(!completed)) {
  row<-learn[i,]
  z<-readRDS(file.path(input,paste0("d",row$dimension,"-j",row$data_index,"-r",row$replication),
                       paste0(row$method,"-learning.rds")))
  if(!is.null(z$history))histories[[length(histories)+1L]]<-data.frame(
    dimension=row$dimension,data_index=row$data_index,replication=row$replication,method=row$method,
    iteration=seq_along(z$history)-1L,particles=z$counts,log_likelihood=z$history,
    window_cv=vapply(seq_along(z$history),function(k)
      if(k<6)NA_real_ else iapf_likelihood_cv(z$history[(k-5):k],"sample"),0.))
  # The runner emitted fit rows only for successful learners. Recover all
  # numerical diagnostics from the preserved capped attempts as well.
  for(it in seq_along(z$fits))for(t in seq_along(z$fits[[it]])) {
    f<-z$fits[[it]][[t]]
    fits<-rbind(fits,data.frame(dimension=row$dimension,data_index=row$data_index,
      replication=row$replication,method=row$method,iteration=it-1,time=t,
      weight_ess=f$weight_ess,lambda=f$lambda,kkt=f$kkt_residual,active_constraints=f$active_constraints))
  }
}
if(length(histories))write(do.call(rbind,histories),"capped-history")
set.seed(93800001)
intervals<-list()
for(d in c(5,10,20,40,80)) {
  z<-paired[paired$dimension==d,]
  difference<-abs(z$log_error.first)-abs(z$log_error.second)
  if(any(!is.finite(difference))) {
    intervals[[length(intervals)+1L]]<-data.frame(dimension=d,paired_cells=nrow(z),
      weighted_mae=NA_real_,guarded_zero_mae=mean(abs(z$log_error.second)),difference=NA_real_,
      marginal99_low=NA_real_,marginal99_high=NA_real_,family99_low=NA_real_,family99_high=NA_real_,
      conditional_direction="candidate_failure_no_accuracy_ranking")
    next
  }
  boot<-numeric(20000)
  for(j in sort(unique(z$data_index))) {
    x<-difference[z$data_index==j]
    boot<-boot+colMeans(matrix(sample(x,length(x)*20000,replace=TRUE),nrow=length(x)))/4
  }
  marginal<-quantile(boot,c(.005,.995),names=FALSE)
  family<-quantile(boot,c(.001,.999),names=FALSE)
  intervals[[length(intervals)+1L]]<-data.frame(dimension=d,paired_cells=nrow(z),
    weighted_mae=mean(abs(z$log_error.first)),guarded_zero_mae=mean(abs(z$log_error.second)),
    difference=mean(difference),marginal99_low=marginal[1],marginal99_high=marginal[2],
    family99_low=family[1],family99_high=family[2],
    conditional_direction=if(family[1]>0)"weighted_higher_error" else
      if(family[2]<0)"weighted_lower_error" else "unresolved")
}
write(do.call(rbind,intervals),"primary-intervals")
summary<-function(df,by) {
  df$absolute_log_error<-abs(df$log_error);df$squared_relative_error<-(df$ratio-1)^2
  result<-aggregate(df[c("absolute_log_error","squared_relative_error")],df[by],mean)
  counts<-aggregate(list(complete_n=as.integer(is.finite(df$log_error))),df[by],sum)
  result<-merge(result,counts,by=by)
  result$required_n<-if("data_index" %in% by)8L else 32L
  incomplete<-result$complete_n!=result$required_n
  result[incomplete,c("absolute_log_error","squared_relative_error")]<-NA_real_
  result$relative_rmse<-sqrt(result$squared_relative_error);result
}
write(summary(learn,c("dimension","method")),"learning-summary")
probe_summary<-summary(probes,c("dimension","data_index","method"))
write(probe_summary,"conditional-probes")
write(summary(probes,c("dimension","method")),"probe-summary")
heuristics<-list()
for(d in c(5,10,20,40,80))for(j in 1:4)for(method in methods)for(control in controls) {
  z<-probe_summary[probe_summary$dimension==d & probe_summary$data_index==j,]
  a<-z[z$method==method,];b<-z[z$method==control,]
  completion_veto<-a$complete_n!=8 || b$complete_n!=8
  heuristics[[length(heuristics)+1L]]<-data.frame(dimension=d,data_index=j,method=method,control=control,
    absolute_log_error_difference=a$absolute_log_error-b$absolute_log_error,
    relative_rmse_difference=a$relative_rmse-b$relative_rmse,
    completion_veto=completion_veto,
    promotion_veto=completion_veto || a$absolute_log_error>b$absolute_log_error || a$relative_rmse>b$relative_rmse,
    statistically_supported_ranking=FALSE)
}
heuristics<-do.call(rbind,heuristics);write(heuristics,"heuristic-screen")
fit_summary<-do.call(rbind,lapply(methods,function(method) {
  z<-fits[fits$method==method,]
  data.frame(method=method,fits=nrow(z),positive_ridges=sum(z$lambda>0,na.rm=TRUE),
    active_bounds=sum(z$active_constraints>0,na.rm=TRUE),
    min_weight_ess=if(all(is.na(z$weight_ess)))NA_real_ else min(z$weight_ess,na.rm=TRUE),
    max_kkt=if(all(is.na(z$kkt)))NA_real_ else max(z$kkt,na.rm=TRUE))
}));write(fit_summary,"fit-summary")
write(aggregate(gaps["kl"],gaps[c("dimension","method")],mean),"guide-summary")
verdict<-data.frame(method=methods,heuristic_promotion_veto=vapply(methods,function(method)
  any(heuristics$promotion_veto[heuristics$method==method]),TRUE),default_ready=FALSE,
  paper_replication=FALSE,author_identity=FALSE,score_validated=FALSE)
write(verdict,"decision")
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
cat("Report complete; all",nrow(checks),"integrity checks pass.\n")
print(do.call(rbind,intervals),row.names=FALSE)
print(fit_summary,row.names=FALSE)

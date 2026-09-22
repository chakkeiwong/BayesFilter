# Post-run statistical diagnostics for the independent R reference campaign.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) %in% c(2,3),Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(args[2]))
root <- args[1];out <- args[2];dir.create(out,recursive=TRUE)
cells <- read.csv(if(length(args)==3)args[3] else file.path(root,"cells.csv"),stringsAsFactors=FALSE)
read_kind <- function(name) {
  rows <- list()
  for(a in unique(cells$attempt[nzchar(cells$attempt)])) {
    p <- file.path(root,a,"results",name)
    if(file.exists(p))rows[[length(rows)+1L]] <- cbind(attempt=a,
      read.csv(p,stringsAsFactors=FALSE))
  }
  if(length(rows))do.call(rbind,rows) else data.frame()
}
learning <- read_kind("learning.csv");probes <- read_kind("probes.csv")
write.csv(learning,file.path(out,"learning.csv"),row.names=FALSE)
write.csv(probes,file.path(out,"probes.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"session-info.txt"))
B <- 20000L;set.seed(93000001)
boot_means <- function(x,groups) {
  if(!length(x) || any(!is.finite(x)))return(rep(NA_real_,B))
  indices <- lapply(split(seq_along(x),groups),function(g)
    matrix(g[sample.int(length(g),length(g)*B,replace=TRUE)],nrow=length(g)))
  indices <- do.call(rbind,indices)
  colMeans(matrix(x[indices],nrow=nrow(indices)))
}
interval <- function(x,probs)if(all(is.finite(x)))
  as.numeric(quantile(x,probs=probs,names=FALSE,type=8)) else rep(NA_real_,length(probs))
emit <- function(rows,name)if(length(rows))
  write.csv(do.call(rbind,rows),file.path(out,name),row.names=FALSE)
stopifnot(all(boot_means(c(1,2),c(1,2))==1.5),
  all(boot_means(rep(0,4),c(1,1,2,2))==0))
set.seed(93000001)
primary <- reliability <- grid <- matched <- heuristics <- controls <- list()
for(d in c(5,10,20,40,80)) {
  for(method in c("qr","ridge_bounded1")) {
    z <- learning[learning$dimension==d & learning$method==method,,drop=FALSE]
    n <- nrow(z);failures <- sum(z$status!="complete")
    reliability[[length(reliability)+1L]] <- data.frame(dimension=d,method=method,
      planned=16,recorded=n,complete=sum(z$status=="complete"),failures=failures,
      unrecorded=16-n,failure_probability_upper95=if(n)
        binom.test(failures,n,alternative="less")$conf.int[2] else NA_real_)
  }
  x <- learning[learning$dimension==d & learning$method=="qr",]
  y <- learning[learning$dimension==d & learning$method=="ridge_bounded1",]
  z <- merge(x,y,by=c("data_index","replication"),suffixes=c("_qr","_repair"))
  z <- z[z$status_qr=="complete" & z$status_repair=="complete" &
    is.finite(z$terminal_error_qr) & is.finite(z$terminal_error_repair),]
  diff <- abs(z$terminal_error_repair)-abs(z$terminal_error_qr)
  ci <- interval(boot_means(diff,z$data_index),c(.005,.995))
  ready <- nrow(z)==16 && sum(cells$dimension==d & cells$status=="complete")==16
  verdict <- if(!ready)"incomplete_descriptive_only" else if(ci[2]<0)
    "repair_lower_MAE_conditional_99pct_interval" else if(ci[1]>0)
    "QR_lower_MAE_conditional_99pct_interval" else "no_supported_ranking"
  primary[[length(primary)+1L]] <- data.frame(dimension=d,paired_complete=nrow(z),
    planned=16,qr_MAE=mean(abs(z$terminal_error_qr)),repair_MAE=mean(abs(z$terminal_error_repair)),
    difference=mean(diff),lower99=ci[1],upper99=ci[2],verdict=verdict)

  for(method in c("qr","ridge_bounded1","constant","observation",
      "moment_diagonal","precision_diagonal","exact_full")) {
    for(N in if(method %in% c("qr","ridge_bounded1"))c(250,1000,4000) else 1000) {
      a <- probes[probes$dimension==d & probes$method==method & probes$N==N,,drop=FALSE]
      a <- a[a$status=="complete",,drop=FALSE]
      sq <- a$relative_error^2
      rmse_samples <- sqrt(boot_means(sq,a$data_index))
      rmse_ci <- interval(rmse_samples,c(.025,.975))
      upper <- interval(rmse_samples,1-.05/30)
      ratio_ci <- interval(boot_means(a$likelihood_ratio,a$data_index),c(.025,.975))
      qualified <- nrow(a)==16 && sum(cells$dimension==d & cells$status=="complete")==16 &&
        all(is.finite(sq)) && is.finite(upper) && upper<=.5
      row <- data.frame(dimension=d,method=method,N=N,complete=nrow(a),planned=16,
        mean_absolute_log_error=mean(abs(a$terminal_error)),relative_RMSE=sqrt(mean(sq)),
        RMSE_lower95=rmse_ci[1],RMSE_upper95=rmse_ci[2],RMSE_upper_simultaneous=upper,
        mean_likelihood_ratio=mean(a$likelihood_ratio),ratio_lower95=ratio_ci[1],ratio_upper95=ratio_ci[2],
        near_zero_ratio_fraction=mean(a$likelihood_ratio<.01),
        mean_training_seconds=mean(a$guide_seconds),mean_filter_seconds=mean(a$filter_seconds),
        mean_total_seconds=mean(a$total_seconds),precision_target=.5,
        precision_qualified=qualified,
        calibration_interval_excludes_one=all(is.finite(ratio_ci)) &&
          (ratio_ci[1]>1+1e-7 || ratio_ci[2]<1-1e-7))
      grid[[length(grid)+1L]] <- row
    }
  }
  for(j in 1:2)for(method in c("qr","ridge_bounded1")) {
    base <- c("constant","observation","moment_diagonal","precision_diagonal")
    if(method=="ridge_bounded1")base <- c(base,"qr")
    for(comparator in base) {
      a <- probes[probes$dimension==d & probes$data_index==j & probes$N==1000 &
        probes$method==method & probes$status=="complete",]
      b <- probes[probes$dimension==d & probes$data_index==j & probes$N==1000 &
        probes$method==comparator & probes$status=="complete",]
      pair <- merge(a,b,by="replication",suffixes=c("_candidate","_control"))
      delta <- abs(pair$terminal_error_candidate)-abs(pair$terminal_error_control)
      heuristics[[length(heuristics)+1L]] <- data.frame(dimension=d,data_index=j,
        method=method,comparator=comparator,paired_complete=nrow(pair),
        candidate_MAE=mean(abs(pair$terminal_error_candidate)),
        comparator_MAE=mean(abs(pair$terminal_error_control)),mean_difference=mean(delta),
        observed_promotion_veto=length(delta)>0 && mean(delta)>0,
        statistically_supported_ranking="not_tested_by_this_descriptive_screen")
    }
  }
}
emit(primary,"primary-comparison.csv");emit(reliability,"reliability.csv")
emit(grid,"accuracy-cost-grid.csv");emit(heuristics,"conditional-heuristics.csv")
g <- do.call(rbind,grid)
for(d in c(5,10,20,40,80))for(method in c("qr","ridge_bounded1")) {
  a <- g[g$dimension==d & g$method==method & g$precision_qualified,,drop=FALSE]
  if(nrow(a)) {
    a <- a[which.min(a$mean_total_seconds),,drop=FALSE]
    matched[[length(matched)+1L]] <- data.frame(dimension=d,method=method,
      status="qualifying_measured_rung",N=a$N,relative_RMSE=a$relative_RMSE,
      upper_RMSE=a$RMSE_upper_simultaneous,mean_total_seconds=a$mean_total_seconds)
  } else matched[[length(matched)+1L]] <- data.frame(dimension=d,method=method,
    status="no_qualifying_rung",N=NA,relative_RMSE=NA,upper_RMSE=NA,mean_total_seconds=NA)
}
emit(matched,"precision-matched-cost.csv")
oracle <- probes[probes$method=="exact_full",]
write.csv(data.frame(check=c("all_planned_pairs_complete","all_exact_guides_present",
  "all_exact_guides_agree_Kalman","all_learning_values_finite_when_complete"),
  pass=c(sum(cells$status=="complete")==80,nrow(oracle)==80,
    nrow(oracle)>0 && all(abs(oracle$terminal_error)<=1e-7),
    all(is.finite(learning$terminal_error[learning$status=="complete"])))),
  file.path(out,"checks.csv"),row.names=FALSE)
cat("Report assembled; learning records",nrow(learning),"probe records",nrow(probes),"\n")

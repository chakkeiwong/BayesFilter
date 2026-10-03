# Independent CPU diagnostic of saved optimizer failures and F2 initialization.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==1,Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(args[1]))
out <- args[1];dir.create(out,recursive=TRUE)
source("docs/benchmarks/reference_iapf_paper.R")
source("docs/benchmarks/reference_iapf_author_choices.R")
old <- "docs/plans/artifacts/iapf-r-author-choice-hypotheses-20260922-01"
rows <- list();inputs <- character()
value <- function(x,key)if(is.null(x[[key]]))NA else x[[key]]
for(csv in sort(Sys.glob(file.path(old,"runs/fit_probe-*/results/replicates.csv")))) {
  records <- read.csv(csv,stringsAsFactors=FALSE)
  records <- records[grepl("^f[12]",records$method) & records$status=="candidate_failed",,drop=FALSE]
  for(j in seq_len(nrow(records))) {
    r <- records[j,]
    path <- file.path(dirname(csv),paste0(r$method,"-",r$replication,"-failure.rds"))
    e <- readRDS(path);inputs <- c(inputs,path)
    previous <- if(!is.null(e$backward_time))e$filter_twists[[e$backward_time]] else NULL
    valid_previous <- !is.null(previous) && !previous$constant &&
      all(is.finite(previous$mean)) && all(is.finite(previous$covariance)) &&
      !inherits(try(chol(previous$covariance),silent=TRUE),"try-error")
    rows[[length(rows)+1L]] <- data.frame(method=r$method,dimension=r$dimension,
      replication=r$replication,message=r$message,iteration=r$failure_iteration,
      time=r$failure_time,valid_previous=valid_previous,
      qr_rejection=r$message=="nonconcave log-quadratic regression",
      initial_shape_residual=value(e$diagnostics,"initial_relative_residual"),
      final_shape_residual=value(e$diagnostics,"relative_residual"),
      log_density_shift=value(e$diagnostics,"log_density_shift"),
      initial_loss=value(e$diagnostics,"initial_profiled_loss"),
      final_loss=value(e$diagnostics,"profiled_loss"),
      optimizer_convergence=value(e$diagnostics,"convergence"),
      source=path)
  }
}
tab <- do.call(rbind,rows)
write.csv(tab,file.path(out,"saved-optimizer-failures.csv"),row.names=FALSE)
example <- tab[tab$method=="f2_strict" & tab$qr_rejection & tab$valid_previous,,drop=FALSE][1,]
stopifnot(nrow(example)==1,!is.na(example$source))
e <- readRDS(example$source);previous <- e$filter_twists[[e$backward_time]]
optimizer_calls <- 0L;qr_calls <- 0L
optim <- function(...) {optimizer_calls <<- optimizer_calls+1L;stats::optim(...)}
original_initializer <- iapf_fit_initial
iapf_fit_initial <- function(...) {qr_calls <<- qr_calls+1L;original_initializer(...)}
replayed <- tryCatch(iapf_choice_fit(e$points,e$log_targets,arm="f2_strict",previous=previous),
  error=function(error)error)
checks <- data.frame(check=c("valid_previous_guide","strict_QR_rejection_reproduced",
  "QR_called_before_optimizer"),pass=c(example$valid_previous,
  inherits(replayed,"iapf_fit_error") && conditionMessage(replayed)==example$message,
  qr_calls==1 && optimizer_calls==0))
write.csv(checks,file.path(out,"checks.csv"),row.names=FALSE)
write.csv(data.frame(source=example$source,iteration=example$iteration,time=example$time,
  qr_calls=qr_calls,optimizer_calls=optimizer_calls),file.path(out,"call-chain.csv"),row.names=FALSE)
writeLines(inputs,file.path(out,"input-paths.txt"))
stopifnot(all(checks$pass))
cat("PASS",nrow(checks),"call-chain checks; F2 QR rejections:",
  sum(grepl("^f2",tab$method) & tab$qr_rejection),"; all with valid previous:",
  all(tab$valid_previous[grepl("^f2",tab$method) & tab$qr_rejection]),"\n")

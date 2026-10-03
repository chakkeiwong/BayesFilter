# Explanatory inspection of existing RDS evidence; no fitting or new filter runs.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]
rows <- list()
for(path in list.files(root,pattern="^iapf-[0-9]+\\.rds$",recursive=TRUE,full.names=TRUE)) {
  if(!grepl("/attempt[0-9]+-validation-",path)) next
  fit <- readRDS(path)
  attempt <- basename(dirname(dirname(path)))
  d <- as.integer(sub(".*-d([0-9]+)$","\\1",attempt))
  replication <- as.integer(sub("iapf-([0-9]+)\\.rds","\\1",basename(path)))
  stopifnot(fit$status=="complete",length(fit$history)==length(fit$counts))
  for(l in 5:(length(fit$history)-1)) {
    window <- fit$history[(l-4):(l+1)]
    normalized <- exp(window-max(window))
    cv <- sd(normalized)/mean(normalized)
    last_five <- tail(normalized,5)
    next_n <- if(l+2<=length(fit$counts))fit$counts[l+2] else NA_integer_
    rows[[length(rows)+1]] <- data.frame(attempt=attempt,dimension=d,
      replication=replication,iteration=l,particles=fit$counts[l+1],
      next_particles=next_n,window_cv=cv,stop_eligible=l>5,
      last_five_cv_explanatory_only=sd(last_five)/mean(last_five),
      stop_condition=l>5 && cv<.5,monotone=all(diff(window)>=0),
      same_n_window=fit$counts[l-4]==fit$counts[l+1],
      doubled=if(is.na(next_n))FALSE else next_n>fit$counts[l+1])
  }
}
write.csv(do.call(rbind,rows),output,row.names=FALSE)
cat("COMPLETE",length(rows),"saved controller windows; no new filtering\n")

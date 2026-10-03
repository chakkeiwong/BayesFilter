# Post-run independent R diagnostic on saved failures; no new filter runs.
a <- commandArgs(trailingOnly=TRUE); campaign <- a[1]; output <- a[2]
stopifnot(Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1",!dir.exists(output))
dir.create(output,recursive=TRUE)
files <- list.files(file.path(campaign,"runs"),pattern="-failure[.]rds$",
  recursive=TRUE,full.names=TRUE)
rows <- list()
for(path in files) {
  records_path <- file.path(dirname(path),"replicates.csv")
  if(!file.exists(records_path)) next
  records <- read.csv(records_path,stringsAsFactors=FALSE)
  names <- paste0(records$method,"-",records$replication,"-failure.rds")
  record <- records[match(basename(path),names),,drop=FALSE]
  if(nrow(record)!=1 || is.na(record$method) || !startsWith(record$method,"wlog")) next
  failure <- readRDS(path)
  if(is.null(failure$points) || is.null(failure$log_targets)) next
  points <- failure$points; target <- failure$log_targets; d <- ncol(points)
  center <- colMeans(points); scale <- sqrt(colMeans(sweep(points,2,center)^2))
  stopifnot(all(is.finite(scale)),all(scale>0),all(is.finite(target)))
  z <- sweep(sweep(points,2,center),2,scale,"/"); design <- cbind(1,z,z^2)
  exponent <- as.integer(sub("wlog","",record$method))
  weights <- exp(exponent*(target-max(target)))
  fit <- lm.wfit(design,target-max(target),w=weights)
  ess <- sum(weights)^2/sum(weights^2)
  condition <- kappa(design*sqrt(weights),exact=TRUE)
  stored <- failure$diagnostics$weight_ess
  if(!is.null(stored))stopifnot(abs(ess-stored)<=1e-10*max(1,ess))
  sha <- strsplit(system2("sha256sum",shQuote(path),stdout=TRUE)," ")[[1]][1]
  value <- function(x,default=NA)if(is.null(x))default else x
  rows[[length(rows)+1L]] <- data.frame(
    job_id=basename(dirname(dirname(path))),method=record$method,
    dimension=d,data_seed=record$data_seed,replication=record$replication,
    floor_rule=record$floor_rule,sd_mode=record$sd_mode,message=record$message,
    iteration=value(failure$iteration),backward_time=value(failure$backward_time),
    particles=nrow(points),design_columns=ncol(design),unweighted_rank=qr(design)$rank,
    weighted_rank=fit$rank,rank_tolerance=1e-7,weighted_condition=condition,
    weight_ess=ess,max_normalized_weight=max(weights)/sum(weights),
    top10_weight_share=sum(sort(weights,decreasing=TRUE)[seq_len(min(10,length(weights)))])/sum(weights),
    weights_above_relative_epsilon=sum(weights>.Machine$double.eps),
    stored_ess_checked=!is.null(stored),source_rds=path,source_sha256=sha)
}
stopifnot(length(rows)>0)
table <- do.call(rbind,rows)
write.csv(table,file.path(output,"weighted-failure-diagnostics.csv"),row.names=FALSE)
dput(list(cpu_only=TRUE,CUDA_VISIBLE_DEVICES="-1",question="saved weighted-fit numerical rank failures",
  baseline="unweighted design on identical stored particles",promotion_evidence=FALSE,
  random_seeds="N/A: deterministic saved-data diagnostic",R=R.version.string,
  rows=nrow(table)),file=file.path(output,"settings.R"))
cat("PASS saved-cloud diagnostics; stored ESS checked where available; rows",nrow(table),"\n")

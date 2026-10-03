# First Section-5.2 experiment, independent noncanonical R reference.
# Args: repository output dimension repeats first_replication_id data_seed mode
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]; dimension <- as.integer(args[3])
repeats <- as.integer(args[4]); first <- as.integer(args[5]); data_seed <- as.integer(args[6])
mode <- args[7]
fit_mode <- if(length(args)>=8)args[8] else "paper_eq15"
floor_power <- if(length(args)>=9)as.numeric(args[9]) else 2
doubling_mode <- if(length(args)>=10)args[10] else "first_full_window"
fit_maxit <- if(length(args)>=11)as.integer(args[11]) else 200L
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
stopifnot(mode %in% c("pilot","replication"),repeats>=1,!dir.exists(output))
dir.create(output,recursive=TRUE)
data <- iapf_paper_data(dimension,100,seed=data_seed)
model <- iapf_linear_model(data$observations,data$A)
kalman <- iapf_kalman(model)
write.csv(data$observations,file.path(output,"observations.csv"),row.names=FALSE)
write.csv(data.frame(time=1:100,prefix=kalman$prefix,innovation=kalman$innovation,
  situation=ifelse(kalman$innovation>qchisq(.9,dimension),"large_innovation","ordinary")),
  file.path(output,"kalman.csv"),row.names=FALSE)
append_rows <- function(rows,name) {
  if(!length(rows)) return(invisible(NULL))
  path <- file.path(output,name)
  exists <- file.exists(path)
  write.table(do.call(rbind,rows),path,append=exists,col.names=!exists,
    row.names=FALSE,sep=",",qmethod="double",na="NA")
}
settings <- list(mode=mode,dimension=dimension,horizon=100,N0=1000,BPF_N=10000,
  FA_N=5000,SIS_N=10000,k=5,tau=.5,kappa=.5,fit_maxit=fit_maxit,floor_tail_power=floor_power,
  fit_mode=fit_mode,doubling_mode=doubling_mode,
  equation_15_objective=fit_mode=="paper_eq15",
  data_seed=data_seed,first_replication=first,repeats=repeats,
  original_author_data=FALSE,cpu_only=TRUE,noncanonical_reference=TRUE)
dput(settings,file=file.path(output,"settings.R"))
for(replication in first:(first+repeats-1)) {
  for(method in c("iapf","bpf","fully_adapted","sis")) {
    rows <- list(); diagnostic_rows <- list(); prefix_rows <- list()
    method_id <- match(method,c("iapf","bpf","fully_adapted","sis"))
    seed <- 53000000+dimension*100000+replication*10+method_id
    set.seed(seed); started <- proc.time()[[3]]
    cat("BEGIN",dimension,replication,method,"seed",seed,"\n"); flush.console()
    answer <- tryCatch({
      if(method=="iapf") {
        iteration_call <- 0
        filter <- function(...) {
          iteration_call <<- iteration_call+1
          value <- iapf_apf(...)
          cat("APF",replication,iteration_call,"logZ",value$log_likelihood,"elapsed",proc.time()[[3]]-started,"\n")
          flush.console(); value
        }
        value <- iapf_iterate(model,.filter=filter,fit_mode=fit_mode,fit_maxit=fit_maxit,
          floor_tail_power=floor_power,doubling_mode=doubling_mode)
        saveRDS(value,file.path(output,paste0("iapf-",replication,".rds")))
        if(value$status!="complete") stop(paste("incomplete",value$status))
        for(iteration in seq_along(value$fits)) for(time in seq_along(value$fits[[iteration]])) {
          diagnostic_rows[[length(diagnostic_rows)+1]] <- data.frame(
            replication=replication,iteration=iteration-1,time=time,
            as.data.frame(value$fits[[iteration]][[time]],stringsAsFactors=FALSE))
        }
        list(final=value$final,iterations=length(value$history),N=value$final_particles,
             failures=sum(vapply(unlist(value$fits,recursive=FALSE),function(x)x$convergence!=0,TRUE)))
      } else if(method=="fully_adapted") {
        list(final=iapf_fully_adapted(model,5000),iterations=1,N=5000,failures=0)
      } else {
        list(final=iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),
          10000,if(method=="sis")0 else .5,FALSE),iterations=1,N=10000,failures=0)
      }
    },error=function(e) {
      saveRDS(e,file.path(output,paste0(method,"-",replication,"-failure.rds")))
      list(error=conditionMessage(e))
    })
    elapsed <- proc.time()[[3]]-started
    if(!is.null(answer$error)) {
      rows[[length(rows)+1]] <- data.frame(dimension=dimension,replication=replication,
        method=method,seed=seed,status="error",error=answer$error,log_likelihood=NA,
        log_ratio=NA,ratio=NA,wall_seconds=elapsed,iterations=NA,particles=NA,
        resampling_count=NA,fit_nonconvergence=NA,floor_probability_max=NA)
    } else {
      log_ratio <- answer$final$log_likelihood-kalman$log_likelihood
      rows[[length(rows)+1]] <- data.frame(dimension=dimension,replication=replication,
        method=method,seed=seed,status="complete",error="",log_likelihood=answer$final$log_likelihood,
        log_ratio=log_ratio,ratio=exp(log_ratio),wall_seconds=elapsed,
        iterations=answer$iterations,particles=answer$N,resampling_count=answer$final$resampling_count,
        fit_nonconvergence=answer$failures,
        floor_probability_max=if(is.null(answer$final$floor_probability_max))0 else answer$final$floor_probability_max)
      for(time in 1:100) prefix_rows[[length(prefix_rows)+1]] <- data.frame(
        dimension=dimension,replication=replication,method=method,time=time,
        log_prefix_error=answer$final$prefix[time]-kalman$prefix[time],
        situation=if(kalman$innovation[time]>qchisq(.9,dimension))"large_innovation" else "ordinary")
    }
    append_rows(rows,"replicates.csv")
    append_rows(diagnostic_rows,"fits.csv")
    append_rows(prefix_rows,"prefixes.csv")
    cat("END",dimension,replication,method,"elapsed",elapsed,
        "status",if(is.null(answer$error))"complete" else answer$error,"\n"); flush.console()
    if(!is.null(answer$error)) quit(status=2)
  }
}
cat("COMPLETE",dimension,repeats,"\n")

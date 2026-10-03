# Bounded independent-reference worker; invoked only through captured sources.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]; arm <- args[3]
d <- as.integer(args[4]); repeats <- as.integer(args[5])
first <- as.integer(args[6]); data_seed <- as.integer(args[7]); baselines <- args[8]=="yes"
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
source(file.path(root,"docs/benchmarks/reference_iapf_plausible_choices.R"))
source(file.path(root,"docs/benchmarks/diagnose_iapf_r_validation_tails.R"))
stopifnot(!dir.exists(output),Sys.getenv("CUDA_VISIBLE_DEVICES")=="-1")
dir.create(output,recursive=TRUE)
data <- iapf_paper_data(d,100,seed=data_seed)
model <- iapf_linear_model(data$observations,data$A); kalman <- iapf_kalman(model)
write.csv(data$observations,file.path(output,"observations.csv"),row.names=FALSE)
write.csv(data.frame(time=1:100,prefix=kalman$prefix,innovation=kalman$innovation),
  file.path(output,"kalman.csv"),row.names=FALSE)
settings <- c(list(arm=arm,dimension=d,horizon=100,repeats=repeats,first=first,
  data_seed=data_seed,cpu_only=TRUE,noncanonical_reference=TRUE,N0=1000,
  k=5,tau=.5,kappa=.5,max_iterations=20,max_particles=16000,baselines=baselines),
  iapf_reconstruction_settings(arm))
dput(settings,file=file.path(output,"settings.R"))
append_csv <- function(x,name) {
  path <- file.path(output,name); exists <- file.exists(path)
  write.table(x,path,append=exists,col.names=!exists,row.names=FALSE,sep=",",qmethod="double")
}
methods <- if(baselines)c("iapf","bpf","fully_adapted","sis") else "iapf"
for(replication in first:(first+repeats-1)) for(method in methods) {
  seed <- 53000000+d*100000+replication*10+match(method,c("iapf","bpf","fully_adapted","sis"))
  set.seed(seed); started <- proc.time()[[3]]
  cat("BEGIN",arm,d,replication,method,"seed",seed,"\n"); flush.console()
  answer <- tryCatch({
    if(method=="iapf") {
      value <- iapf_reconstruction(model,arm)
      if(value$status!="complete") {
        saveRDS(value,file.path(output,paste0("incomplete-",replication,".rds")))
        stop(paste("incomplete",value$status))
      }
      fit_rows <- list()
      for(iteration in seq_along(value$fits)) for(time in seq_along(value$fits[[iteration]]))
        fit_rows[[length(fit_rows)+1]] <- data.frame(replication=replication,
          iteration=iteration,time=time,value$fits[[iteration]][[time]])
      append_csv(do.call(rbind,fit_rows),"fits.csv")
      tails <- iapf_validation_tail_rows(model,value$twists,replication)
      append_csv(tails,"tails.csv")
      # Preserve guides/controller diagnostics without duplicating every cloud.
      value$final$clouds <- value$final$log_weights <- value$final$ancestors <- NULL
      saveRDS(value,file.path(output,paste0("iapf-",replication,".rds")))
      stopifnot(all(tails$passed))
      list(final=value$final,particles=value$final_particles,iterations=length(value$history))
    } else if(method=="fully_adapted") {
      list(final=iapf_fully_adapted(model,5000),particles=5000,iterations=1)
    } else {
      list(final=iapf_apf(model,replicate(100,iapf_constant_twist(),simplify=FALSE),
        10000,if(method=="sis")0 else .5,FALSE),particles=10000,iterations=1)
    }
  },error=function(e) {
    saveRDS(e,file.path(output,paste0(method,"-",replication,"-failure.rds")))
    if(inherits(e,"iapf_fit_error")) {
      fields <- c("message","backward_time","optimizer_code","relative_bound_margin",
        "evaluation","initial_evaluation")
      dput(e[intersect(names(e),fields)],file=file.path(output,"fit-failure-summary.R"))
    }
    list(error=conditionMessage(e))
  })
  elapsed <- proc.time()[[3]]-started
  if(!is.null(answer$error)) {
    append_csv(data.frame(dimension=d,replication=replication,method=method,seed=seed,
      status="error",error=answer$error,log_ratio=NA,ratio=NA,wall_seconds=elapsed,
      particles=NA,iterations=NA,resampling_count=NA,floor_probability_max=NA),"replicates.csv")
    cat("ERROR",answer$error,"elapsed",elapsed,"\n"); quit(status=2)
  }
  log_ratio <- answer$final$log_likelihood-kalman$log_likelihood
  append_csv(data.frame(dimension=d,replication=replication,method=method,seed=seed,
    status="complete",error="",log_ratio=log_ratio,ratio=exp(log_ratio),wall_seconds=elapsed,
    particles=answer$particles,iterations=answer$iterations,
    resampling_count=answer$final$resampling_count,
    floor_probability_max=if(is.null(answer$final$floor_probability_max))0 else answer$final$floor_probability_max),
    "replicates.csv")
  append_csv(data.frame(replication=replication,method=method,time=1:100,
    log_prefix_error=answer$final$prefix-kalman$prefix,
    situation=ifelse(kalman$innovation>qchisq(.9,d),"large_innovation","ordinary")),"prefixes.csv")
  cat("END",arm,d,replication,method,"elapsed",elapsed,"\n"); flush.console()
}
cat("COMPLETE",arm,d,repeats,"\n")

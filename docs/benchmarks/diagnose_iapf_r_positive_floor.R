# Independent CPU reference: a frozen positive-floor repair, not a new default.
args <- commandArgs(trailingOnly=TRUE)
root <- args[1]; output <- args[2]
source(file.path(root,"docs/benchmarks/reference_iapf_paper.R"))
stopifnot(args[7]=="floor_repair",as.numeric(args[9])==8,!dir.exists(output))
dir.create(output,recursive=TRUE)
run_phase <- function(command,log) {
  status <- system2("Rscript",shQuote(command),stdout=log,stderr=log)
  stopifnot(is.numeric(status),length(status)==1L,file.exists(log))
  status
}
campaign <- file.path(root,"docs/plans/artifacts/iapf-r-log-fit-validation-20260920-01")
saved <- file.path(campaign,"attempt06-d80-diagnostic/results")
change_floor <- function(twists,N,power) lapply(twists,function(twist) {
  peak <- -.5*(length(twist$mean)*log(2*pi)+
    as.numeric(determinant(twist$covariance,logarithm=TRUE)$modulus))
  twist$log_floor <- peak-.5*qchisq(N^(-power),df=length(twist$mean),lower.tail=FALSE)
  twist
})
margins <- list()
check_margins <- function(model,twists,label) {
  observation <- t(model$C) %*% solve(model$R,model$C)
  for (t in seq_along(twists)) {
    Q <- if(t==1)model$initial_covariance else model$transition_covariance
    H <- observation
    if(t<model$horizon) H <- H+t(model$A) %*%
      solve(model$transition_covariance+twists[[t+1]]$covariance,model$A)
    M <- solve(Q)+2*H-solve(twists[[t]]$covariance)
    eigenvalues <- eigen(M,symmetric=TRUE,only.values=TRUE)$values
    scale <- norm(solve(Q),"2")+2*norm(H,"2")+norm(solve(twists[[t]]$covariance),"2")
    relative <- min(eigenvalues)/scale
    margins[[length(margins)+1L]] <<- data.frame(case=label,time=t,
      minimum_eigenvalue=min(eigenvalues),scale=scale,relative_margin=relative,
      threshold=100*model$dimension*.Machine$double.eps)
    path <- file.path(output,"tail-margins.csv")
    exists <- file.exists(path)
    write.table(tail(margins,1)[[1]],path,append=exists,col.names=!exists,
      row.names=FALSE,sep=",",qmethod="double")
    stopifnot(is.finite(relative),relative>100*model$dimension*.Machine$double.eps)
  }
}
healthy <- iapf_linear_model(matrix(0,3,80),matrix(0,80,80))
healthy_twists <- replicate(3,iapf_gaussian_twist(rep(0,80),diag(80)),simplify=FALSE)
set.seed(76000001)
old_healthy <- iapf_apf(healthy,change_floor(healthy_twists,1000,2),1000)
set.seed(76000001)
new_healthy <- iapf_apf(healthy,change_floor(healthy_twists,1000,8),1000)
fields <- c("clouds","log_weights","ancestors","ess","resampled","prefix","log_likelihood")
checks <- data.frame(check=paste0("healthy_identical_",fields),
  passed=vapply(fields,function(name)identical(old_healthy[[name]],new_healthy[[name]]),TRUE))
write.csv(checks,file.path(output,"checks.csv"),row.names=FALSE)
stopifnot(all(checks$passed))
if(length(args)>=12 && args[12]=="mechanics_only") {
  for (status in c(0L,2L)) {
    log <- file.path(output,paste0("status-",status,".log"))
    command <- c("--vanilla","-e",paste0("cat('stdout sentinel'); message('stderr sentinel'); quit(status=",status,")"))
    observed <- run_phase(command,log)
    stopifnot(identical(observed,status),any(grepl("stdout sentinel",readLines(log))),
      any(grepl("stderr sentinel",readLines(log))))
  }
  quit(status=0)
}
data <- readRDS(file.path(saved,"model.rds"))
history <- readRDS(file.path(saved,"replayed-result.rds"))
frozen <- list()
for(call in c(15L,19L,20L)) {
  input <- readRDS(file.path(saved,paste0("filter-input-",call,".rds")))
  twists <- change_floor(input$twists,history$counts[call-1L],8)
  check_margins(data$model,twists,paste0("saved-call",call))
  assign(".Random.seed",input$rng,envir=.GlobalEnv)
  value <- iapf_apf(data$model,twists,input$N,input$kappa,FALSE)
  saveRDS(value,file.path(output,paste0("positive-floor-call",call,".rds")))
  frozen[[length(frozen)+1L]] <- data.frame(call=call,N=input$N,
    log_error=value$log_likelihood-data$kalman$log_likelihood,
    min_ess=min(value$ess),floor_probability_max=value$floor_probability_max)
  write.csv(do.call(rbind,frozen),file.path(output,"frozen-guides.csv"),row.names=FALSE)
  cat("FROZEN",call,"log_error",tail(frozen,1)[[1]]$log_error,"\n")
}
phases <- list(
  list(name="d20-a",d=20,seed=70000020,first=201,count=4,previous="attempt01-d20-a"),
  list(name="d20-b",d=20,seed=71000020,first=301,count=4,previous="attempt02-d20-b"),
  list(name="d80-pilot",d=80,seed=72000080,first=1,count=2,previous=NULL),
  list(name="d80-fresh",d=80,seed=74000080,first=101,count=4,previous=NULL))
records <- list(); conditional <- list(); regressions <- list()
for (phase in phases) {
  destination <- file.path(output,phase$name)
  command <- c("--vanilla",file.path(root,"docs/benchmarks/replicate_iapf_paper_linear.R"),
    root,destination,phase$d,phase$count,phase$first,phase$seed,"replication",
    "log_quadratic",8,"first_full_window",200)
  record <- data.frame(phase=phase$name,dimension=phase$d,data_seed=phase$seed,
    first=phase$first,repeats=phase$count,status="running",elapsed=NA_real_,
    command=paste(c("Rscript",shQuote(command)),collapse=" "))
  records[[length(records)+1L]] <- record
  write.csv(do.call(rbind,records),file.path(output,"phases.csv"),row.names=FALSE)
  started <- proc.time()[[3]]
  code <- run_phase(command,file.path(output,paste0(phase$name,".log")))
  record$elapsed <- proc.time()[[3]]-started
  record$status <- if(code==0)"complete" else "failed"
  records[[length(records)]] <- record
  write.csv(do.call(rbind,records),file.path(output,"phases.csv"),row.names=FALSE)
  if (code!=0) stop(paste("incomplete repair phase",phase$name))
  rows <- read.csv(file.path(destination,"replicates.csv"))
  fits <- read.csv(file.path(destination,"fits.csv"))
  stopifnot(nrow(rows)==4*phase$count,all(rows$status=="complete"),
    all(fits$convergence==0),!any(fits$boundary),all(fits$design_rank==2*phase$d+1),
    all(is.finite(rows$log_likelihood)))
  ratios <- rows$ratio[rows$method=="iapf"]
  stopifnot(all(is.finite(ratios)),all(ratios>=.1 & ratios<=10))
  prefix <- read.csv(file.path(destination,"prefixes.csv"))
  stopifnot(nrow(prefix)==400*phase$count,all(is.finite(prefix$log_prefix_error)))
  for (situation in unique(prefix$situation)) {
    values <- tapply(prefix$log_prefix_error[prefix$situation==situation]^2,
      prefix$method[prefix$situation==situation],mean)
    conditional[[length(conditional)+1L]] <- data.frame(phase=phase$name,
      situation=situation,method=names(values),mean_square_error=as.numeric(values),
      heuristic_veto=values[["iapf"]]>min(values[names(values)!="iapf"]))
  }
  write.csv(do.call(rbind,conditional),file.path(output,"conditional.csv"),row.names=FALSE)
  stopifnot(!any(do.call(rbind,conditional)$heuristic_veto))
  model_data <- iapf_paper_data(phase$d,100,seed=phase$seed)
  model <- iapf_linear_model(model_data$observations,model_data$A)
  for (id in phase$first:(phase$first+phase$count-1)) {
    fitted <- readRDS(file.path(destination,paste0("iapf-",id,".rds")))
    check_margins(model,fitted$twists,paste0(phase$name,"-",id))
  }
  if (!is.null(phase$previous)) {
    previous <- read.csv(file.path(campaign,phase$previous,"results/prefixes.csv"))
    previous <- previous[previous$replication %in% phase$first:(phase$first+phase$count-1),]
    for(situation in unique(prefix$situation)) {
      after <- mean(prefix$log_prefix_error[prefix$method=="iapf" & prefix$situation==situation]^2)
      before <- mean(previous$log_prefix_error[previous$method=="iapf" & previous$situation==situation]^2)
      regressions[[length(regressions)+1L]] <- data.frame(phase=phase$name,
        situation=situation,previous_mse=before,candidate_mse=after,ratio=after/before,
        passed=after<=2*before)
    }
    write.csv(do.call(rbind,regressions),file.path(output,"regressions.csv"),row.names=FALSE)
    stopifnot(all(do.call(rbind,regressions)$passed))
  }
  cat("PHASE",phase$name,"COMPLETE mean_ratio",mean(ratios),"\n"); flush.console()
}
cat("COMPLETE bounded positive-floor evaluation; no default promotion\n")
